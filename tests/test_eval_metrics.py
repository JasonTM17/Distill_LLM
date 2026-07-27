"""Unit tests for distill.eval_metrics and a smoke test of the report renderer.

The renderer is covered in depth by tests/test_evaluate.py; what is pinned here
is that a perplexity never appears without its truncation cap and that a
baseline measured at a different cap is never differenced against this run.
"""

from distill.eval_metrics import lcs_length, mean, rouge_l_f1, token_f1
from distill.evaluate import aggregate, render_report


# ── lcs / rouge ────────────────────────────────────────────────────────────

def test_lcs_known_values():
    assert lcs_length(list("ABCBDAB"), list("BDCABA")) == 4
    assert lcs_length([], list("abc")) == 0
    assert lcs_length(list("abc"), list("abc")) == 3


def test_rouge_identical_is_one():
    assert rouge_l_f1("the quick brown fox", "the quick brown fox") == 1.0


def test_rouge_disjoint_is_zero():
    assert rouge_l_f1("alpha beta gamma", "delta epsilon zeta") == 0.0


def test_rouge_partial_overlap():
    # ref 4 tokens, hyp 2 tokens, LCS=2 -> p=1, r=0.5, f1=2/3
    score = rouge_l_f1("a b c d", "a b")
    assert abs(score - 2 / 3) < 1e-9


def test_rouge_empty_inputs():
    assert rouge_l_f1("", "anything") == 0.0
    assert rouge_l_f1("anything", "") == 0.0


def test_token_f1_order_insensitive():
    assert token_f1("a b c", "c b a") == 1.0


def test_mean_empty_is_zero():
    assert mean([]) == 0.0


# ── aggregate / report rendering (no model needed) ─────────────────────────

def _fake_payload():
    results = [
        {"id": 1, "category": "math", "rouge_l": 0.5, "token_f1": 0.6},
        {"id": 2, "category": "math", "rouge_l": 0.3, "token_f1": 0.4},
        {"id": 3, "category": "coding", "rouge_l": 0.8, "token_f1": 0.9},
    ]
    return {
        "label": "vtest",
        "model_path": "checkpoints/merged",
        "evaluated_at": "2026-07-26T12:00:00+00:00",
        "num_samples": 3,
        "baseline_ppl": 6.93,
        "baseline_cap": 512,
        "max_seq_length": 2048,
        "seed": 42,
        "perplexity": {
            "loss": 1.5,
            "perplexity": 4.48,
            "per_category": {
                "math": {"loss": 1.4, "perplexity": 4.1, "samples": 2},
                "coding": {"loss": 1.6, "perplexity": 4.9, "samples": 1},
            },
            "max_seq_length": 2048,
            "truncated_samples": 0,
            "tokens_scored": 1800,
            "tokens_total": 1800,
            "coverage": 1.0,
        },
        "generation": aggregate(results),
        "details": results,
    }


def test_aggregate_per_category_means():
    gen = aggregate(
        [
            {"category": "math", "rouge_l": 0.5, "token_f1": 0.6},
            {"category": "math", "rouge_l": 0.3, "token_f1": 0.4},
        ]
    )
    assert gen["rouge_l"] == 0.4
    assert gen["per_category"]["math"]["samples"] == 2
    assert gen["per_category"]["math"]["judge"] is None


def test_render_report_contains_key_sections():
    report = render_report(_fake_payload())
    assert "# Evaluation vtest" in report
    assert "| Perplexity (held-out, cap 2048, 100% token coverage) | **4.48** |" in report
    assert "| math | 2 | 4.10 |" in report
    assert "## Measurement conditions" in report
    assert "## Caveats" in report


def test_render_report_never_publishes_an_unlabelled_perplexity():
    report = render_report(_fake_payload())
    assert "| Perplexity (held-out) |" not in report


def test_render_report_refuses_a_cross_cap_baseline_verdict():
    # Baseline measured at cap 512, this run at cap 2048: differencing them would
    # compare perplexities over different fractions of the held-out set.
    report = render_report(_fake_payload())
    assert "| Baseline PPL 6.93 (cap 512) | not compared |" in report
    assert "**better**" not in report
    assert "**worse**" not in report
