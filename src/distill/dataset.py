"""Dataset quality pipeline: raw teacher outputs → clean train/val/test splits.

Fixes two v0.4 defects:

* Training text lacked Qwen's ``<|im_start|>``/``<|im_end|>`` special tokens, so
  the trained model saw a different format than every standard inference path
  (``apply_chat_template``, llama.cpp). Samples now use the exact Qwen2.5 template.
* No validation split and no quality gate. Samples are now screened for mojibake,
  short outputs, and duplicate instructions, then split 80/10/10 stratified by
  category with a fixed seed.

Usage::

    python -m distill.dataset            # build splits from teacher_outputs.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from . import config
from .generate_dataset import atomic_write_json
from .logging_utils import get_logger

logger = get_logger("dataset")

# Categories with fewer usable samples than this stay entirely in train:
# a 1-sample test slice of a tiny category measures noise, not generalization.
MIN_CATEGORY_FOR_SPLIT = 5


def qwen_chat_text(instruction: str, output: str, system_prompt: str | None = None) -> dict[str, str]:
    """Render the exact Qwen2.5 chat template for one sample.

    Returns ``prompt_text`` (everything up to and including the assistant header)
    and ``text`` (prompt + completion) so training can mask the prompt tokens.
    """
    system_prompt = system_prompt or config.SYSTEM_PROMPT
    prompt_text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return {"prompt_text": prompt_text, "text": prompt_text + output + "<|im_end|>"}


def _instruction_key(instruction: str) -> str:
    """Normalization key used to detect near-duplicate prompts."""
    normalized = unicodedata.normalize("NFKC", instruction).lower()
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def screen_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    """Apply the quality gate. Returns (accepted, rejection_reasons)."""
    rejected: Counter = Counter()
    seen_keys: set[str] = set()
    accepted: list[dict[str, Any]] = []
    for record in records:
        if not record.get("success"):
            rejected["not_successful"] += 1
            continue
        output = (record.get("output") or "").strip()
        if not output or len(output) < config.MIN_OUTPUT_CHARS:
            rejected["too_short"] += 1
            continue
        if "�" in output or "�" in record.get("instruction", ""):
            rejected["mojibake"] += 1
            continue
        key = _instruction_key(record["instruction"])
        if key in seen_keys:
            rejected["duplicate_instruction"] += 1
            continue
        seen_keys.add(key)
        accepted.append(record)
    return accepted, rejected


def stratified_split(
    samples: list[dict[str, Any]],
    *,
    validation_ratio: float | None = None,
    test_ratio: float | None = None,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic per-category split into (train, validation, test)."""
    validation_ratio = (
        config.VALIDATION_SPLIT_RATIO if validation_ratio is None else validation_ratio
    )
    test_ratio = config.TEST_SPLIT_RATIO if test_ratio is None else test_ratio
    rng = random.Random(config.SPLIT_SEED if seed is None else seed)

    by_category: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_category.setdefault(sample.get("category", "unknown"), []).append(sample)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for category in sorted(by_category):
        items = sorted(by_category[category], key=lambda s: s["id"])
        rng.shuffle(items)
        n = len(items)
        if n < MIN_CATEGORY_FOR_SPLIT:
            train.extend(items)
            continue
        n_test = max(1, math.floor(n * test_ratio))
        n_val = max(1, math.floor(n * validation_ratio))
        test.extend(items[:n_test])
        validation.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])
    return train, validation, test


def _format_split(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted = []
    for sample in samples:
        rendered = qwen_chat_text(sample["instruction"], sample["output"].strip())
        formatted.append(
            {
                "id": sample["id"],
                "category": sample["category"],
                "instruction": sample["instruction"],
                "output": sample["output"].strip(),
                "truncated": bool(sample.get("truncated")),
                **rendered,
            }
        )
    return formatted


def build_splits(raw_path: Path | None = None) -> dict[str, Any]:
    """Load raw outputs, screen, split, and return everything plus stats."""
    raw_path = raw_path or config.TEACHER_OUTPUT_FILE
    with open(raw_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("data", [])

    accepted, rejected = screen_records(records)
    train, validation, test = stratified_split(accepted)

    splits = {
        "train": _format_split(train),
        "validation": _format_split(validation),
        "test": _format_split(test),
    }
    per_split_categories = {
        name: dict(Counter(s["category"] for s in split))
        for name, split in splits.items()
    }
    stats = {
        "source_file": str(raw_path),
        "raw_records": len(records),
        "accepted": len(accepted),
        "rejected": dict(rejected),
        "truncated_kept": sum(1 for s in accepted if s.get("truncated")),
        "split_sizes": {name: len(split) for name, split in splits.items()},
        "categories": per_split_categories,
        "seed": config.SPLIT_SEED,
        "ratios": {
            "validation": config.VALIDATION_SPLIT_RATIO,
            "test": config.TEST_SPLIT_RATIO,
        },
    }
    return {"splits": splits, "stats": stats}


def run(raw_path: Path | None = None) -> dict[str, Any]:
    """Build splits and persist them to data/processed/."""
    config.ensure_directories()
    result = build_splits(raw_path)
    splits, stats = result["splits"], result["stats"]

    atomic_write_json(config.TRAIN_FILE, splits["train"])
    atomic_write_json(config.VALIDATION_FILE, splits["validation"])
    atomic_write_json(config.TEST_FILE, splits["test"])
    atomic_write_json(config.DATASET_STATS_FILE, stats)

    logger.info(
        "accepted=%d rejected=%s | train=%d val=%d test=%d (truncated kept: %d)",
        stats["accepted"],
        stats["rejected"] or "{}",
        *(stats["split_sizes"][k] for k in ("train", "validation", "test")),
        stats["truncated_kept"],
    )
    for category, count in sorted(stats["categories"]["train"].items()):
        val_n = stats["categories"]["validation"].get(category, 0)
        test_n = stats["categories"]["test"].get(category, 0)
        logger.info("  %-12s train=%-3d val=%-2d test=%-2d", category, count, val_n, test_n)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build train/val/test splits")
    parser.add_argument("--raw", type=Path, default=None, help="teacher outputs JSON")
    args = parser.parse_args(argv)
    run(args.raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
