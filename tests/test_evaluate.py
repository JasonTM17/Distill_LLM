"""Unit tests for distill.evaluate: cap accounting, resume, report rendering, CLI.

No model weights are loaded anywhere here. ``compute_perplexity`` and
``generate_answers`` are exercised against stub model/tokenizer objects, which is
enough to pin the accounting (truncation, coverage, resume) that the real run
depends on.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from distill import config, evaluate
from distill.evaluate import (
    _parse_caps,
    aggregate,
    build_parser,
    clear_partial,
    compute_perplexity,
    compute_perplexity_sweep,
    generate_answers,
    load_partial,
    partial_results_path,
    render_report,
    save_partial,
)


# ── stubs ──────────────────────────────────────────────────────────────────

class _Batch(dict):
    """Mimics a transformers BatchEncoding closely enough for these paths."""

    def to(self, device):  # noqa: ARG002 - device is irrelevant for stubs
        return self


class _StubTokenizer:
    """Whitespace tokenizer: one id per word, so token counts are predictable."""

    pad_token_id = 0

    def __call__(self, text, truncation=False, max_length=None, return_tensors=None):
        ids = [len(word) for word in str(text).split()] or [1]
        if truncation and max_length:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return _Batch({"input_ids": torch.tensor([ids])})
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[-1]["content"]

    def decode(self, ids, skip_special_tokens=True):
        return "student answer text"


class _StubModel:
    """Constant-loss model; generation appends two fixed tokens to the prompt."""

    device = "cpu"

    def __init__(self):
        self.generate_calls = 0

    def eval(self):
        return self

    def __call__(self, input_ids, labels=None):
        return SimpleNamespace(loss=torch.tensor(1.0))

    def generate(self, input_ids=None, **kwargs):
        self.generate_calls += 1
        return torch.cat([input_ids, torch.tensor([[7, 7]])], dim=1)


PPL_ROWS = [
    {"id": 1, "category": "coding", "text": "a b c d e",
     "instruction": "Write a palindrome checker.", "output": "reference one"},
    {"id": 2, "category": "creative", "text": "a b",
     "instruction": "Describe flying.", "output": "reference two"},
]


# ── compute_perplexity records its own measurement conditions ──────────────

def test_compute_perplexity_reports_cap_truncation_and_coverage():
    result = compute_perplexity(_StubModel(), _StubTokenizer(), PPL_ROWS, max_seq_length=3)
    assert result["max_seq_length"] == 3
    assert result["truncated_samples"] == 1          # only the 5-token row is cut
    assert result["tokens_total"] == 7               # 5 + 2 untruncated
    assert result["tokens_scored"] == 5              # 3 + 2 after the cap
    assert result["coverage"] == pytest.approx(5 / 7, abs=1e-4)
    assert result["perplexity"] == pytest.approx(2.72, abs=0.01)


def test_compute_perplexity_full_coverage_when_cap_exceeds_every_sample():
    result = compute_perplexity(_StubModel(), _StubTokenizer(), PPL_ROWS, max_seq_length=64)
    assert result["truncated_samples"] == 0
    assert result["coverage"] == 1.0
    assert result["tokens_scored"] == result["tokens_total"] == 7


def test_compute_perplexity_defaults_to_eval_cap_not_training_cap(monkeypatch):
    monkeypatch.setattr(config, "EVAL_MAX_SEQ_LENGTH", 3)
    monkeypatch.setattr(config, "MAX_SEQ_LENGTH", 1)
    result = compute_perplexity(_StubModel(), _StubTokenizer(), PPL_ROWS)
    assert result["max_seq_length"] == 3


def test_compute_perplexity_sweep_keys_by_cap():
    sweep = compute_perplexity_sweep(_StubModel(), _StubTokenizer(), PPL_ROWS, [4, 2, 2])
    assert sorted(sweep) == ["2", "4"]
    assert sweep["2"]["max_seq_length"] == 2
    assert sweep["2"]["truncated_samples"] == 1
    assert sweep["4"]["truncated_samples"] == 1
    assert sweep["4"]["tokens_scored"] > sweep["2"]["tokens_scored"]


# ── per-sample persistence and resume ──────────────────────────────────────

def test_generate_answers_writes_a_partial_after_every_sample(tmp_path):
    partial = tmp_path / "partial.json"
    model = _StubModel()
    generate_answers(model, _StubTokenizer(), PPL_ROWS, 8, partial_file=partial, seed=1)
    saved = load_partial(partial)
    assert sorted(saved) == [1, 2]
    assert model.generate_calls == 2
    assert json.loads(partial.read_text(encoding="utf-8"))["seed"] == 1


def test_generate_answers_resumes_and_preserves_completed_values(tmp_path):
    partial = tmp_path / "partial.json"
    done = {
        1: {"id": 1, "category": "coding", "instruction": "old", "reference": "old",
            "answer": "previously generated", "rouge_l": 0.9999, "token_f1": 0.5}
    }
    save_partial(partial, done, seed=42)

    model = _StubModel()
    results = generate_answers(
        model, _StubTokenizer(), PPL_ROWS, 8, partial_file=partial, seed=42
    )
    assert model.generate_calls == 1                      # id 1 was skipped
    assert results[0]["answer"] == "previously generated"  # its value survived intact
    assert results[0]["rouge_l"] == 0.9999
    assert results[1]["id"] == 2
    assert sorted(load_partial(partial)) == [1, 2]


def test_generate_answers_no_resume_regenerates_everything(tmp_path):
    partial = tmp_path / "partial.json"
    save_partial(partial, {1: {"id": 1, "answer": "stale", "rouge_l": 0.9}}, seed=42)
    model = _StubModel()
    results = generate_answers(
        model, _StubTokenizer(), PPL_ROWS, 8, partial_file=partial, resume=False, seed=42
    )
    assert model.generate_calls == 2
    assert results[0]["answer"] == "student answer text"


def test_load_partial_tolerates_missing_and_corrupt_files(tmp_path):
    assert load_partial(tmp_path / "nope.json") == {}
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert load_partial(corrupt) == {}


def test_clear_partial_is_idempotent(tmp_path):
    partial = tmp_path / "partial.json"
    save_partial(partial, {1: {"id": 1}}, seed=42)
    clear_partial(partial)
    clear_partial(partial)
    assert not partial.exists()


def test_partial_path_is_label_keyed_and_filename_safe(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path)
    assert partial_results_path("v0.5").name == "evaluation_partial_v0.5.json"
    assert partial_results_path("a/b c").name == "evaluation_partial_a_b_c.json"


def test_generation_is_seeded_and_reproducible():
    first = evaluate.seed_generation(1234)
    a = torch.randn(3)
    assert first == 1234
    evaluate.seed_generation(1234)
    assert torch.equal(a, torch.randn(3))


def test_seed_generation_defaults_to_config_seed(monkeypatch):
    monkeypatch.setattr(config, "EVAL_SEED", 7)
    assert evaluate.seed_generation() == 7


# ── report fixtures ────────────────────────────────────────────────────────

def _ppl_block(
    cap=2048, perplexity=5.23, coverage=1.0, truncated=0, scored=30301,
    coding=2.53, creative=14.95,
):
    return {
        "loss": 1.6546,
        "perplexity": perplexity,
        "per_category": {
            "coding": {"loss": 0.93, "perplexity": coding, "samples": 2},
            "creative": {"loss": 2.7, "perplexity": creative, "samples": 1},
        },
        "max_seq_length": cap,
        "truncated_samples": truncated,
        "tokens_scored": scored,
        "tokens_total": 30301,
        "coverage": coverage,
    }


def _sweep():
    return {
        "512": _ppl_block(512, 5.38, 0.704, 26, 21335, coding=3.04, creative=15.04),
        "1024": _ppl_block(1024, 5.30, 0.919, 5, 27844, coding=2.64, creative=14.95),
        "2048": _ppl_block(),
    }


def _details(judge=False):
    rows = [
        {"id": 1, "category": "coding", "instruction": "Write a palindrome checker.",
         "reference": "r", "answer": "a", "rouge_l": 0.364, "token_f1": 0.51},
        {"id": 2, "category": "coding", "instruction": "Explain recursion simply.",
         "reference": "r", "answer": "a", "rouge_l": 0.2, "token_f1": 0.31},
        {"id": 3, "category": "creative", "instruction": "Describe flying without birds.",
         "reference": "r", "answer": "a", "rouge_l": 0.073, "token_f1": 0.25},
    ]
    if judge:
        for row, score in zip(rows, (5, 4, 3)):
            row["judge_score"] = score
    return rows


_UNSET = object()


def _payload(**overrides):
    details = overrides.pop("details", _UNSET)
    if details is _UNSET:
        details = _details()
    payload = {
        "label": "vtest",
        "model_path": "checkpoints/merged",
        "evaluated_at": "2026-07-26T12:00:00+00:00",
        "num_samples": 3,
        "baseline_ppl": None,
        "baseline_cap": None,
        "max_seq_length": 2048,
        "seed": 42,
        "perplexity": _ppl_block(),
        "generation": aggregate(details),
        "details": details,
    }
    payload.update(overrides)
    return payload


# ── the perplexity row always names its cap ────────────────────────────────

def test_report_perplexity_row_names_cap_and_coverage():
    report = render_report(_payload())
    assert "| Perplexity (held-out, cap 2048, 100% token coverage) | **5.23** |" in report
    assert "| Perplexity (held-out) |" not in report


def test_report_perplexity_row_shows_partial_coverage():
    report = render_report(_payload(perplexity=_ppl_block(512, 5.38, 0.704, 26, 21335)))
    assert "cap 512, 70.4% token coverage" in report


def test_report_marks_the_cap_as_unrecorded_when_the_payload_lacks_it():
    block = _ppl_block()
    del block["max_seq_length"], block["coverage"]
    report = render_report(_payload(perplexity=block))
    assert "cap not recorded" in report


# ── baseline comparison only happens at a matching cap ─────────────────────

def test_report_compares_baseline_when_caps_match():
    report = render_report(_payload(baseline_ppl=6.93, baseline_cap=2048))
    assert "| vs baseline PPL 6.93 @ cap 2048 | **better** (5.23 vs 6.93, -24.5%) |" in report
    assert "not compared" not in report


def test_report_reports_worse_when_caps_match_and_the_run_regressed():
    report = render_report(
        _payload(perplexity=_ppl_block(perplexity=7.5), baseline_ppl=6.93, baseline_cap=2048)
    )
    assert "**worse**" in report


def test_report_refuses_to_compare_when_the_baseline_cap_is_unknown():
    report = render_report(_payload(baseline_ppl=6.93, baseline_cap=None))
    assert "| Baseline PPL 6.93 (cap unknown) | not compared |" in report
    assert "**better**" not in report
    assert "**worse**" not in report
    assert "its truncation cap was never recorded" in report


def test_report_refuses_to_compare_when_the_caps_differ():
    report = render_report(_payload(baseline_ppl=6.93, baseline_cap=512))
    assert "| Baseline PPL 6.93 (cap 512) | not compared |" in report
    assert "**better**" not in report
    assert "**worse**" not in report
    assert "it was measured at cap 512 and this run used cap 2048" in report


def test_report_compares_at_the_matched_cap_when_the_sweep_supplies_it():
    report = render_report(
        _payload(
            baseline_ppl=6.93,
            baseline_cap=512,
            perplexity_by_max_seq_length=_sweep(),
        )
    )
    # The canonical 5.23 is never differenced against a cap-512 baseline; the
    # sweep's own cap-512 figure is.
    assert "| Baseline PPL 6.93 (cap 512) | not compared |" in report
    assert "vs baseline PPL 6.93 @ cap 512 (matched from sweep) | **better**" in report
    assert "5.38 vs 6.93" in report
    assert "5.23 vs 6.93" not in report


def test_report_omits_baseline_rows_entirely_when_no_baseline_given():
    report = render_report(_payload())
    assert "baseline" not in report.lower()


def test_report_refuses_to_compare_when_this_run_has_no_recorded_cap():
    block = _ppl_block()
    del block["max_seq_length"]
    report = render_report(_payload(perplexity=block, baseline_ppl=6.93, baseline_cap=512))
    assert "this run's cap was not recorded" in report
    assert "**better**" not in report


def test_report_labels_the_cap_even_without_coverage_data():
    block = _ppl_block()
    del block["coverage"]
    report = render_report(_payload(perplexity=block))
    assert "| Perplexity (held-out, cap 2048) | **5.23** |" in report


# ── measurement conditions and the sweep table ─────────────────────────────

def test_report_states_measurement_conditions():
    report = render_report(_payload())
    assert "## Measurement conditions" in report
    assert "| Perplexity truncation cap (`EVAL_MAX_SEQ_LENGTH`) | 2048 |" in report
    assert "| Samples truncated by the cap | 0 / 3 |" in report
    assert "| Tokens scored | 30,301 / 30,301 |" in report
    assert "| Token coverage | 100% |" in report
    assert "| Generation seed | 42 |" in report


def test_report_sweep_table_renders_when_the_sweep_is_present():
    report = render_report(_payload(perplexity_by_max_seq_length=_sweep()))
    assert "### Perplexity by truncation cap" in report
    assert "| 512 | 5.38 | 1.6546 | 26 / 3 | 21,335 / 30,301 | 70.4% |" in report
    assert "| **2048** | **5.23** |" in report
    assert "Cap 2048 (bold) is this run's canonical figure." in report


def test_report_sweep_table_absent_without_a_sweep():
    report = render_report(_payload())
    assert "### Perplexity by truncation cap" not in report


# ── per-category table ─────────────────────────────────────────────────────

def test_report_per_category_single_cap_columns():
    report = render_report(_payload())
    assert "| Category | Samples | PPL @2048 | ROUGE-L | Token-F1 |" in report
    assert "| coding | 2 | 2.53 | 0.2820 | 0.4100 |" in report


def test_report_per_category_carries_every_cap_when_the_sweep_ran():
    report = render_report(_payload(perplexity_by_max_seq_length=_sweep()))
    assert "| Category | Samples | PPL @2048 | PPL @1024 | PPL @512 | ROUGE-L | Token-F1 |" in report
    assert "| coding | 2 | 2.53 | 2.64 | 3.04 |" in report
    assert "| creative | 1 | 14.95 | 14.95 | 15.04 |" in report


# ── judge, seed, gaps, examples ────────────────────────────────────────────

def test_report_makes_the_skipped_judge_visible():
    report = render_report(_payload())
    assert "| LLM-as-judge (1-5) | **not run** — see Named gaps |" in report
    assert "**LLM-as-judge not run.**" in report


def test_report_shows_judge_scores_when_they_exist():
    report = render_report(_payload(details=_details(judge=True)))
    assert "| LLM-as-judge (1-5) | **4.00** over 3 rows |" in report
    assert "**LLM-as-judge not run.**" not in report
    assert "| Judge |" in report


def test_report_flags_a_missing_seed_as_a_named_gap():
    report = render_report(_payload(seed=None))
    assert "| Generation seed | not recorded |" in report
    assert "**Generation is not reproducible — no seed recorded.**" in report


def test_report_does_not_flag_the_seed_when_it_is_recorded():
    report = render_report(_payload())
    assert "no seed recorded" not in report


def test_report_flags_partial_token_coverage_as_a_named_gap():
    report = render_report(_payload(perplexity=_ppl_block(512, 5.38, 0.704, 26, 21335)))
    assert "discarded 29.6% of the held-out tokens" in report


def test_report_lists_best_and_worst_examples_from_details():
    report = render_report(_payload())
    assert "## Best and worst by ROUGE-L" in report
    assert "`[coding] rouge=0.364 id=1` — Write a palindrome checker." in report
    assert "`[creative] rouge=0.073 id=3` — Describe flying without birds." in report


def test_report_omits_examples_when_details_are_missing():
    report = render_report(_payload(details=[], generation=aggregate(_details())))
    assert "## Best and worst by ROUGE-L" not in report


def test_report_gaps_section_reports_a_clean_run():
    report = render_report(_payload(details=_details(judge=True)))
    assert "None recorded: judge scores present" in report


# ── CLI wiring ─────────────────────────────────────────────────────────────

def test_parse_caps_accepts_and_normalises():
    assert _parse_caps("512,1024,2048") == [512, 1024, 2048]
    assert _parse_caps(" 2048 , 512 , 512 ") == [512, 2048]


@pytest.mark.parametrize("raw", ["", "abc", "512,x", "0", "-1", ","])
def test_parse_caps_rejects_junk(raw):
    with pytest.raises(Exception):
        _parse_caps(raw)


def test_public_cli_flags_keep_their_meanings():
    args = build_parser().parse_args(
        ["--label", "v0.5", "--max-samples", "20", "--max-new-tokens", "256",
         "--judge", "--baseline-ppl", "6.93", "--model-path", "checkpoints/merged"]
    )
    assert args.label == "v0.5"
    assert args.max_samples == 20
    assert args.max_new_tokens == 256
    assert args.judge is True
    assert args.baseline_ppl == 6.93
    assert args.model_path == Path("checkpoints/merged")
    # new flags default to "behave as before the sweep existed"
    assert args.max_seq_length is None
    assert args.ppl_caps is None
    assert args.baseline_cap is None
    assert args.no_resume is False


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    for name in ("DATA_DIR", "RAW_DIR", "PROCESSED_DIR", "CHECKPOINT_DIR", "REPORTS_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    monkeypatch.setattr(evaluate, "load_causal_lm", lambda path: _StubModel())
    monkeypatch.setattr(evaluate, "load_tokenizer", lambda path: _StubTokenizer())
    monkeypatch.setattr(evaluate, "load_test_split", lambda max_samples=None: PPL_ROWS)
    return tmp_path


def _written_payload(tmp_path):
    path = tmp_path / "checkpoint_dir" / "evaluation_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_cli_payload_records_cap_and_seed(cli_env):
    assert evaluate.main(["--label", "tst", "--max-seq-length", "4"]) == 0
    payload = _written_payload(cli_env)
    assert payload["max_seq_length"] == 4
    assert payload["seed"] == config.EVAL_SEED
    assert payload["perplexity"]["max_seq_length"] == 4
    assert "perplexity_by_max_seq_length" not in payload
    assert (cli_env / "reports_dir" / "evaluation-tst.md").exists()


def test_cli_default_cap_is_the_eval_knob_not_the_training_knob(cli_env, monkeypatch):
    monkeypatch.setattr(config, "EVAL_MAX_SEQ_LENGTH", 3)
    monkeypatch.setattr(config, "MAX_SEQ_LENGTH", 1)
    assert evaluate.main(["--label", "tst"]) == 0
    assert _written_payload(cli_env)["max_seq_length"] == 3


def test_cli_sweep_emits_every_cap_including_the_canonical_one(cli_env):
    assert evaluate.main(["--label", "tst", "--max-seq-length", "4", "--ppl-caps", "2,3"]) == 0
    payload = _written_payload(cli_env)
    assert sorted(payload["perplexity_by_max_seq_length"], key=int) == ["2", "3", "4"]
    assert payload["perplexity"] == payload["perplexity_by_max_seq_length"]["4"]


def test_cli_records_the_baseline_cap(cli_env):
    evaluate.main(["--label", "tst", "--baseline-ppl", "6.93", "--baseline-cap", "512"])
    payload = _written_payload(cli_env)
    assert payload["baseline_ppl"] == 6.93
    assert payload["baseline_cap"] == 512


def test_cli_clears_the_partial_after_the_final_artifact_lands(cli_env):
    evaluate.main(["--label", "tst"])
    assert not partial_results_path("tst").exists()


def test_cli_rejects_a_non_positive_cap(cli_env):
    with pytest.raises(SystemExit):
        evaluate.main(["--max-seq-length", "0"])


def test_cli_rejects_a_non_positive_baseline_cap(cli_env):
    with pytest.raises(SystemExit):
        evaluate.main(["--baseline-cap", "0"])


def test_cli_judge_flag_runs_the_judge(cli_env, monkeypatch):
    monkeypatch.setattr(evaluate, "judge_answers", _mark_judged)
    evaluate.main(["--label", "tst", "--judge"])
    assert all(row["judge_score"] == 4 for row in _written_payload(cli_env)["details"])


def _mark_judged(results):
    for row in results:
        row["judge_score"] = 4


# ── judge skips gracefully instead of failing the run ──────────────────────

class _StubJudge:
    def __init__(self, *args, **kwargs):
        pass

    def complete(self, prompt, **kwargs):
        return SimpleNamespace(text="score: 4")


def _judge_rows():
    return [{"instruction": "q", "reference": "r", "answer": "a"}]


def test_judge_answers_parses_the_first_digit(monkeypatch):
    from distill import teacher_client

    monkeypatch.setattr(teacher_client, "TeacherClient", _StubJudge)
    rows = _judge_rows()
    evaluate.judge_answers(rows)
    assert rows[0]["judge_score"] == 4


def test_judge_answers_records_none_when_the_reply_has_no_digit(monkeypatch):
    from distill import teacher_client

    class _NoDigits(_StubJudge):
        def complete(self, prompt, **kwargs):
            return SimpleNamespace(text="unable to grade")

    monkeypatch.setattr(teacher_client, "TeacherClient", _NoDigits)
    rows = _judge_rows()
    evaluate.judge_answers(rows)
    assert rows[0]["judge_score"] is None


def test_judge_answers_skips_on_teacher_error(monkeypatch):
    from distill import teacher_client

    class _Failing(_StubJudge):
        def complete(self, prompt, **kwargs):
            raise teacher_client.TeacherError("judge down")

    monkeypatch.setattr(teacher_client, "TeacherClient", _Failing)
    rows = _judge_rows()
    evaluate.judge_answers(rows)          # must not raise
    assert "judge_score" not in rows[0]


def test_judge_answers_skips_when_the_client_cannot_be_built(monkeypatch):
    from distill import teacher_client

    def _boom(*args, **kwargs):
        raise RuntimeError("no API key")

    monkeypatch.setattr(teacher_client, "TeacherClient", _boom)
    rows = _judge_rows()
    evaluate.judge_answers(rows)          # must not raise
    assert "judge_score" not in rows[0]
