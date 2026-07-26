"""Unit tests for distill.generate_dataset: resume logic, atomic writes, run loop."""

import json
from dataclasses import dataclass

import pytest

from distill import config, generate_dataset
from distill.generate_dataset import (
    atomic_write_json,
    load_existing,
    save_outputs,
    select_pending,
)
from distill.teacher_client import TeacherError, TeacherResponse

PROMPTS = [
    {"id": 1, "category": "math", "instruction": "What is 2+2?"},
    {"id": 2, "category": "math", "instruction": "FAIL this one"},
    {"id": 3, "category": "coding", "instruction": "Write hello world"},
    {"id": 4, "category": "coding", "instruction": "Explain recursion"},
]


def _success_record(pid, category="math"):
    return {"id": pid, "category": category, "instruction": "x", "output": "y" * 50,
            "tokens_used": 10, "success": True}


def _failure_record(pid, category="math"):
    return {"id": pid, "category": category, "instruction": "x", "output": "",
            "tokens_used": 0, "success": False, "error": "boom"}


# ── select_pending ─────────────────────────────────────────────────────────

def test_select_pending_includes_missing_failed_and_empty():
    existing = {
        1: _success_record(1),
        2: _failure_record(2),
        3: {**_success_record(3, "coding"), "output": "   "},
    }
    pending = select_pending(PROMPTS, existing)
    assert [p["id"] for p in pending] == [2, 3, 4]


def test_select_pending_can_skip_previous_failures():
    existing = {1: _success_record(1), 2: _failure_record(2)}
    pending = select_pending(PROMPTS, existing, retry_failed=False)
    assert [p["id"] for p in pending] == [3, 4]


def test_select_pending_category_filter():
    pending = select_pending(PROMPTS, {}, categories={"coding"})
    assert [p["id"] for p in pending] == [3, 4]


# ── atomic write / load round-trip ─────────────────────────────────────────

def test_atomic_write_and_load_existing_roundtrip(tmp_path):
    path = tmp_path / "out.json"
    atomic_write_json(path, {"data": [_success_record(7)]})
    loaded = load_existing(path)
    assert loaded[7]["success"] is True


def test_load_existing_corrupt_file_returns_empty(tmp_path):
    path = tmp_path / "out.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_existing(path) == {}


def test_load_existing_missing_file_returns_empty(tmp_path):
    assert load_existing(tmp_path / "nope.json") == {}


def test_save_outputs_metadata_counts_only_successes(tmp_path):
    path = tmp_path / "out.json"
    save_outputs({1: _success_record(1), 2: _failure_record(2)}, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["total_samples"] == 2
    assert payload["successful_samples"] == 1
    assert payload["total_tokens"] == 10
    assert [item["id"] for item in payload["data"]] == [1, 2]


# ── run() end-to-end with a stub teacher ───────────────────────────────────

@dataclass
class _StubTeacher:
    """Succeeds unless the instruction contains 'FAIL'."""

    calls: int = 0

    def complete(self, instruction, *, system_prompt=None):
        self.calls += 1
        if "FAIL" in instruction:
            raise TeacherError("exhausted 5 attempts; last error: boom")
        return TeacherResponse(
            text="z" * 100, prompt_tokens=5, completion_tokens=10, total_tokens=15,
            finish_reason="stop", model="stub", attempts=1,
        )


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    prompts_file = tmp_path / "prompts.json"
    prompts_file.write_text(json.dumps(PROMPTS), encoding="utf-8")
    output_file = tmp_path / "raw" / "teacher_outputs.json"
    for name in ("DATA_DIR", "RAW_DIR", "PROCESSED_DIR", "CHECKPOINT_DIR", "REPORTS_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    monkeypatch.setattr(config, "PROMPTS_FILE", prompts_file)
    monkeypatch.setattr(config, "TEACHER_OUTPUT_FILE", output_file)
    return output_file


def test_run_generates_and_records_failures(isolated_paths):
    summary = generate_dataset.run(client=_StubTeacher(), delay=0)
    assert summary["generated"] == 3
    assert summary["failed"] == 1
    payload = json.loads(isolated_paths.read_text(encoding="utf-8"))
    assert payload["successful_samples"] == 3
    failed = [r for r in payload["data"] if not r["success"]]
    assert len(failed) == 1 and failed[0]["id"] == 2


def test_run_resumes_and_retries_previous_failures(isolated_paths):
    generate_dataset.run(client=_StubTeacher(), delay=0)

    class _NowSucceeds(_StubTeacher):
        def complete(self, instruction, *, system_prompt=None):
            self.calls += 1
            return TeacherResponse(
                text="w" * 100, prompt_tokens=1, completion_tokens=2, total_tokens=3,
                finish_reason="stop", model="stub", attempts=1,
            )

    retry_client = _NowSucceeds()
    summary = generate_dataset.run(client=retry_client, delay=0)
    assert retry_client.calls == 1  # only the previously failed prompt is retried
    assert summary["successful_total"] == 4


def test_run_no_retry_flag_skips_failures(isolated_paths):
    generate_dataset.run(client=_StubTeacher(), delay=0)
    summary = generate_dataset.run(client=_StubTeacher(), retry_failed=False, delay=0)
    assert summary["generated"] == 0
    assert summary["failed"] == 0
