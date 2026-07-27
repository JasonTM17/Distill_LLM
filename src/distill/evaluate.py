"""Rigorous evaluation of the merged student model on the held-out test split.

Measures, per category and overall:

* perplexity on the exact training-format text (never-seen samples), always
  reported together with the truncation cap and token coverage that produced it,
* ROUGE-L F1 and token-F1 of generated answers vs the teacher reference,
* optional LLM-as-judge scores via 9Router (skipped gracefully when API is down).

A perplexity without its truncation cap is not comparable to anything: the same
checkpoint scores 5.38 / 5.30 / 5.23 at caps 512 / 1024 / 2048 because the cap
decides how much of the held-out set is scored at all. Every figure emitted here
carries its cap, and two figures measured at different caps are never differenced.

Writes ``checkpoints/evaluation_results.json`` and a markdown report to
``plans/reports/evaluation-<label>.md``.

Usage::

    python -m distill.evaluate --label v0.5
    python -m distill.evaluate --label v0.5 --baseline-ppl 6.93 --baseline-cap 512
    python -m distill.evaluate --label v0.5 --ppl-caps 512,1024,2048
    python -m distill.evaluate --label v0.5 --judge --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .eval_metrics import mean, rouge_l_f1, token_f1
from .eval_report import render_report
from .generate_dataset import atomic_write_json
from .logging_utils import get_logger
from .model_loading import load_causal_lm, load_tokenizer

# render_report lives in .eval_report but stays importable from here: it is a
# public entry point and callers should not have to follow the split.
__all__ = [
    "aggregate",
    "build_parser",
    "clear_partial",
    "compute_perplexity",
    "compute_perplexity_sweep",
    "generate_answers",
    "judge_answers",
    "load_partial",
    "load_test_split",
    "main",
    "partial_results_path",
    "render_report",
    "save_partial",
    "seed_generation",
]

logger = get_logger("evaluate")

_JUDGE_RUBRIC = (
    "You are grading a student model's answer against a reference answer.\n"
    "Score the student 1-5: 1=wrong/nonsense, 2=mostly wrong, 3=partially correct,"
    " 4=correct but shallower than reference, 5=matches reference quality.\n"
    "Reply with ONLY the integer.\n\n"
    "Question:\n{question}\n\nReference answer:\n{reference}\n\nStudent answer:\n{answer}\n"
)


def load_test_split(max_samples: int | None = None) -> list[dict[str, Any]]:
    with open(config.TEST_FILE, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return rows[: max_samples or config.EVAL_MAX_SAMPLES]


# ── Perplexity ─────────────────────────────────────────────────────────────

def compute_perplexity(
    model, tokenizer, rows: list[dict[str, Any]], max_seq_length: int | None = None
) -> dict[str, Any]:
    """Token-weighted perplexity plus the measurement conditions that produced it.

    ``max_seq_length`` is the truncation cap applied to each sample. It is
    returned next to the perplexity, along with how many samples the cap cut and
    what fraction of the held-out tokens were actually scored, because a cap that
    discards part of the corpus measures a smaller corpus under the same name.
    """
    import torch

    cap = max_seq_length or config.EVAL_MAX_SEQ_LENGTH
    losses_by_category: dict[str, list[tuple[float, int]]] = defaultdict(list)
    truncated_samples = 0
    tokens_scored = 0
    tokens_total = 0
    model.eval()
    with torch.no_grad():
        for row in rows:
            # Untruncated pass first: cheap next to the forward pass, and the only
            # way to know how much of this sample the cap throws away.
            full_length = len(tokenizer(row["text"])["input_ids"])
            tokens = tokenizer(
                row["text"],
                truncation=True,
                max_length=cap,
                return_tensors="pt",
            ).to(model.device)
            output = model(tokens["input_ids"], labels=tokens["input_ids"])
            count = int(tokens["input_ids"].numel())
            tokens_total += full_length
            tokens_scored += count
            if full_length > count:
                truncated_samples += 1
            losses_by_category[row["category"]].append((float(output.loss), count))

    def _ppl(pairs: list[tuple[float, int]]) -> tuple[float, float]:
        total = sum(loss * n for loss, n in pairs)
        tokens = sum(n for _, n in pairs)
        avg = total / tokens if tokens else 0.0
        return avg, math.exp(avg)

    all_pairs = [pair for pairs in losses_by_category.values() for pair in pairs]
    overall_loss, overall_ppl = _ppl(all_pairs)
    per_category = {
        category: {"loss": round(loss, 4), "perplexity": round(ppl, 2), "samples": len(pairs)}
        for category, pairs in sorted(losses_by_category.items())
        for loss, ppl in [_ppl(pairs)]
    }
    return {
        "loss": round(overall_loss, 4),
        "perplexity": round(overall_ppl, 2),
        "per_category": per_category,
        "max_seq_length": cap,
        "truncated_samples": truncated_samples,
        "tokens_scored": tokens_scored,
        "tokens_total": tokens_total,
        "coverage": round(tokens_scored / tokens_total, 4) if tokens_total else 0.0,
    }


def compute_perplexity_sweep(
    model, tokenizer, rows: list[dict[str, Any]], caps: list[int]
) -> dict[str, dict[str, Any]]:
    """Measure perplexity at each cap inside a single model load, keyed by cap."""
    sweep: dict[str, dict[str, Any]] = {}
    for cap in sorted({int(cap) for cap in caps}):
        result = compute_perplexity(model, tokenizer, rows, cap)
        logger.info(
            "cap=%d perplexity=%.2f loss=%.4f truncated=%d/%d coverage=%.1f%%",
            cap, result["perplexity"], result["loss"], result["truncated_samples"],
            len(rows), result["coverage"] * 100,
        )
        sweep[str(cap)] = result
    return sweep


# ── Answer generation (checkpointed per sample) ────────────────────────────

def partial_results_path(label: str) -> Path:
    """Per-label partial-generation file, rewritten after every sample."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label) or "run"
    return config.CHECKPOINT_DIR / f"evaluation_partial_{safe}.json"


def load_partial(path: Path) -> dict[Any, dict[str, Any]]:
    """Load already-generated answers keyed by sample id; empty when absent/corrupt."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        logger.error("%s is corrupt; regenerating from scratch", path)
        return {}
    return {item["id"]: item for item in payload.get("data", []) if "id" in item}


def save_partial(path: Path, records: dict[Any, dict[str, Any]], seed: int | None) -> None:
    """Persist every answer generated so far, atomically."""
    atomic_write_json(
        path,
        {
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": seed,
            "completed": len(records),
            "data": list(records.values()),
        },
    )


def clear_partial(path: Path) -> None:
    """Drop the partial file once a superseding artifact has been written."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("could not remove %s: %s", path, exc)


def seed_generation(seed: int | None = None) -> int:
    """Seed torch so sampled generations are reproducible; returns the seed used."""
    import torch

    seed = config.EVAL_SEED if seed is None else seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def generate_answers(
    model,
    tokenizer,
    rows,
    max_new_tokens: int | None = None,
    *,
    partial_file: Path | None = None,
    resume: bool = True,
    seed: int | None = None,
):
    """Generate a student answer per row, checkpointing after every sample.

    A 51-sample run is roughly 90 minutes of GPU time, so flushing only after the
    loop means a kill at 36/51 leaves no artifact at all. Each answer is written
    to ``partial_file`` immediately and reloaded on restart. Sampling is seeded so
    ROUGE-L/token-F1 deltas are reproducible rather than one draw of unknown
    variance.
    """
    import torch

    max_new_tokens = max_new_tokens or config.EVAL_MAX_NEW_TOKENS
    seed = seed_generation(seed)
    done = load_partial(partial_file) if partial_file is not None and resume else {}
    if done:
        logger.info("resuming from %s: %d answer(s) already generated", partial_file, len(done))

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        cached = done.get(row["id"])
        if cached is not None:
            results.append(cached)
            logger.info(
                "[%d/%d] id=%s %s resumed rouge=%.2f",
                index, len(rows), row["id"], row["category"], cached.get("rouge_l", 0.0),
            )
            continue

        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": row["instruction"]},
        ]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=config.EVAL_TEMPERATURE,
                top_p=config.EVAL_TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        answer = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        record = {
            "id": row["id"],
            "category": row["category"],
            "instruction": row["instruction"],
            "reference": row["output"],
            "answer": answer,
            "rouge_l": round(rouge_l_f1(row["output"], answer), 4),
            "token_f1": round(token_f1(row["output"], answer), 4),
        }
        results.append(record)
        done[row["id"]] = record
        if partial_file is not None:
            save_partial(partial_file, done, seed)
        logger.info(
            "[%d/%d] id=%s %s rouge=%.2f",
            index, len(rows), row["id"], row["category"], record["rouge_l"],
        )
    return results


def judge_answers(results: list[dict[str, Any]]) -> None:
    """Attach 1-5 judge scores in place; skip silently if the API is down."""
    from .teacher_client import TeacherClient, TeacherError

    try:
        client = TeacherClient(model=config.JUDGE_MODEL)
        for row in results:
            prompt = _JUDGE_RUBRIC.format(
                question=row["instruction"],
                reference=row["reference"][:4000],
                answer=row["answer"][:4000],
            )
            reply = client.complete(
                prompt, max_tokens=config.JUDGE_MAX_TOKENS, temperature=0.0, max_retries=2
            )
            digits = [c for c in reply.text if c.isdigit()]
            row["judge_score"] = int(digits[0]) if digits else None
    except TeacherError as exc:
        logger.warning("judge unavailable, skipping: %s", exc)
    except Exception as exc:  # API client construction/network failure
        logger.warning("judge unavailable, skipping: %s", exc)


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_category[row["category"]].append(row)
    per_category = {
        category: {
            "samples": len(rows),
            "rouge_l": round(mean([r["rouge_l"] for r in rows]), 4),
            "token_f1": round(mean([r["token_f1"] for r in rows]), 4),
            "judge": round(
                mean([r["judge_score"] for r in rows if r.get("judge_score")]), 2
            )
            if any(r.get("judge_score") for r in rows)
            else None,
        }
        for category, rows in sorted(by_category.items())
    }
    return {
        "rouge_l": round(mean([r["rouge_l"] for r in results]), 4),
        "token_f1": round(mean([r["token_f1"] for r in results]), 4),
        "per_category": per_category,
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_caps(raw: str) -> list[int]:
    """Parse ``--ppl-caps 512,1024,2048`` into a sorted, de-duplicated list."""
    caps = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"expected comma-separated integers, got {chunk!r}"
            ) from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(f"caps must be positive, got {value}")
        caps.append(value)
    if not caps:
        raise argparse.ArgumentTypeError("expected at least one cap")
    return sorted(set(caps))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the merged student model")
    parser.add_argument("--label", default="v0.5", help="report label, e.g. v0.5")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--judge", action="store_true", help="add LLM-as-judge scores")
    parser.add_argument("--baseline-ppl", type=float, default=None)
    parser.add_argument(
        "--baseline-cap",
        type=int,
        default=None,
        help="truncation cap the baseline PPL was measured at; without it the baseline "
             "is printed as context and never differenced against this run",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="perplexity truncation cap (default EVAL_MAX_SEQ_LENGTH="
             f"{config.EVAL_MAX_SEQ_LENGTH}); independent of the training cap",
    )
    parser.add_argument(
        "--ppl-caps",
        type=_parse_caps,
        default=None,
        metavar="512,1024,2048",
        help="also measure perplexity at these caps within the same model load",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore any partial generation file and start the answer loop clean",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_seq_length is not None and args.max_seq_length <= 0:
        parser.error("--max-seq-length must be positive")
    if args.baseline_cap is not None and args.baseline_cap <= 0:
        parser.error("--baseline-cap must be positive")

    config.ensure_directories()
    cap = args.max_seq_length or config.EVAL_MAX_SEQ_LENGTH
    # The canonical cap always joins the sweep: it is a measurement at a cap like
    # any other, and omitting it would leave the headline figure unsourced.
    sweep_caps = sorted(set(args.ppl_caps or []) | {cap})

    model_path = args.model_path or config.MERGED_MODEL_DIR
    logger.info("loading model: %s", model_path)
    model = load_causal_lm(model_path)
    tokenizer = load_tokenizer(model_path)

    rows = load_test_split(args.max_samples)
    logger.info("evaluating %d held-out samples at cap %d", len(rows), cap)

    if len(sweep_caps) > 1:
        sweep = compute_perplexity_sweep(model, tokenizer, rows, sweep_caps)
        ppl = sweep[str(cap)]
    else:
        sweep = None
        ppl = compute_perplexity(model, tokenizer, rows, cap)
    logger.info(
        "perplexity=%.2f loss=%.4f cap=%d truncated=%d/%d coverage=%.1f%%",
        ppl["perplexity"], ppl["loss"], cap, ppl["truncated_samples"],
        len(rows), ppl["coverage"] * 100,
    )

    seed = config.EVAL_SEED
    partial_file = partial_results_path(args.label)
    results = generate_answers(
        model,
        tokenizer,
        rows,
        args.max_new_tokens,
        partial_file=partial_file,
        resume=not args.no_resume,
        seed=seed,
    )
    if args.judge:
        judge_answers(results)
    gen = aggregate(results)

    payload = {
        "label": args.label,
        "model_path": str(model_path),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_samples": len(rows),
        "baseline_ppl": args.baseline_ppl,
        "baseline_cap": args.baseline_cap,
        "max_seq_length": cap,
        "seed": seed,
        "perplexity": ppl,
        "generation": gen,
        "details": results,
    }
    if sweep:
        payload["perplexity_by_max_seq_length"] = sweep
    results_file = config.CHECKPOINT_DIR / "evaluation_results.json"
    atomic_write_json(results_file, payload)
    # Only now is the partial superseded. Judge scores land after the answer loop,
    # so dropping it any earlier would put them back at risk of a kill.
    clear_partial(partial_file)
    report_file = config.REPORTS_DIR / f"evaluation-{args.label}.md"
    report_file.write_text(render_report(payload), encoding="utf-8")
    logger.info("wrote %s and %s", results_file, report_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
