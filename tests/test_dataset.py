"""Unit tests for distill.dataset: quality gate, template, stratified splits."""

import json


from distill import config, dataset
from distill.dataset import (
    build_splits,
    qwen_chat_text,
    screen_records,
    stratified_split,
)


def _record(pid, category="math", instruction=None, output=None, success=True, **extra):
    return {
        "id": pid,
        "category": category,
        "instruction": instruction or f"Question number {pid}?",
        "output": output if output is not None else f"Answer {pid}. " + "detail " * 20,
        "success": success,
        **extra,
    }


# ── qwen_chat_text ─────────────────────────────────────────────────────────

def test_template_uses_qwen_special_tokens():
    rendered = qwen_chat_text("What is 2+2?", "4, because arithmetic.")
    assert rendered["text"].startswith("<|im_start|>system\n")
    assert "<|im_start|>user\nWhat is 2+2?<|im_end|>" in rendered["text"]
    assert rendered["text"].endswith("4, because arithmetic.<|im_end|>")
    assert rendered["prompt_text"].endswith("<|im_start|>assistant\n")
    assert rendered["text"].startswith(rendered["prompt_text"])


# ── screen_records quality gate ────────────────────────────────────────────

def test_screen_rejects_failures_short_and_mojibake():
    records = [
        _record(1),
        _record(2, success=False),
        _record(3, output="too short"),
        _record(4, output="Vi�t m�t " + "x" * 100),
        _record(5, output=""),
    ]
    accepted, rejected = screen_records(records)
    assert [r["id"] for r in accepted] == [1]
    assert rejected["not_successful"] == 1
    assert rejected["too_short"] == 2
    assert rejected["mojibake"] == 1


def test_screen_rejects_duplicate_instructions_case_insensitive():
    records = [
        _record(1, instruction="Explain   recursion in Python"),
        _record(2, instruction="explain recursion in python"),
    ]
    accepted, rejected = screen_records(records)
    assert len(accepted) == 1
    assert rejected["duplicate_instruction"] == 1


def test_screen_keeps_truncated_records():
    accepted, _ = screen_records([_record(1, truncated=True)])
    assert accepted and accepted[0].get("truncated") is True


# ── stratified_split ───────────────────────────────────────────────────────

def _many(category, count, start_id):
    return [_record(start_id + i, category=category) for i in range(count)]


def test_split_is_deterministic_and_disjoint():
    samples = _many("math", 50, 0) + _many("coding", 50, 100)
    first = stratified_split(samples, validation_ratio=0.1, test_ratio=0.1, seed=42)
    second = stratified_split(samples, validation_ratio=0.1, test_ratio=0.1, seed=42)
    assert [[s["id"] for s in split] for split in first] == [
        [s["id"] for s in split] for split in second
    ]
    ids = [s["id"] for split in first for s in split]
    assert len(ids) == len(set(ids)) == 100


def test_split_proportions_per_category():
    samples = _many("math", 50, 0)
    train, val, test = stratified_split(samples, validation_ratio=0.1, test_ratio=0.1, seed=42)
    assert len(test) == 5 and len(val) == 5 and len(train) == 40


def test_tiny_categories_stay_in_train():
    samples = _many("rare", 4, 0) + _many("math", 20, 100)
    train, val, test = stratified_split(samples, validation_ratio=0.1, test_ratio=0.1, seed=42)
    assert all(s["category"] != "rare" for s in val + test)
    assert sum(1 for s in train if s["category"] == "rare") == 4


# ── build_splits end-to-end ────────────────────────────────────────────────

def test_build_splits_from_raw_file(tmp_path):
    records = (
        [_record(i, category="math") for i in range(20)]
        + [_record(100 + i, category="coding") for i in range(20)]
        + [_record(900, success=False)]
    )
    raw = tmp_path / "teacher_outputs.json"
    raw.write_text(json.dumps({"data": records}), encoding="utf-8")

    result = build_splits(raw)
    stats = result["stats"]
    assert stats["raw_records"] == 41
    assert stats["accepted"] == 40
    assert stats["rejected"] == {"not_successful": 1}
    sizes = stats["split_sizes"]
    assert sizes["train"] + sizes["validation"] + sizes["test"] == 40
    assert sizes["test"] >= 2  # at least one per category
    sample = result["splits"]["train"][0]
    assert "<|im_start|>" in sample["text"]
    assert sample["output"] and sample["instruction"]


def test_run_writes_all_output_files(tmp_path, monkeypatch):
    records = [_record(i, category="math") for i in range(10)]
    raw = tmp_path / "teacher_outputs.json"
    raw.write_text(json.dumps({"data": records}), encoding="utf-8")
    processed = tmp_path / "processed"
    for name, filename in (
        ("TRAIN_FILE", "dataset_train.json"),
        ("VALIDATION_FILE", "dataset_validation.json"),
        ("TEST_FILE", "dataset_test.json"),
        ("DATASET_STATS_FILE", "dataset_stats.json"),
    ):
        monkeypatch.setattr(config, name, processed / filename)
    for name in ("DATA_DIR", "RAW_DIR", "PROCESSED_DIR", "CHECKPOINT_DIR", "REPORTS_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())

    stats = dataset.run(raw)
    for filename in (
        "dataset_train.json",
        "dataset_validation.json",
        "dataset_test.json",
        "dataset_stats.json",
    ):
        assert (processed / filename).exists()
    assert stats["accepted"] == 10
