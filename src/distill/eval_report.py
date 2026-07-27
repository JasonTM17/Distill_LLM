"""Markdown rendering of the evaluation payload.

Split out of :mod:`distill.evaluate` so the measurement code and the reporting
code can be read separately. Every function here is a pure function of the
payload dict: no model, no torch, no config, no I/O.

The invariant this module exists to enforce: a perplexity is never printed
without the truncation cap that produced it, and two perplexities measured at
different caps are never differenced.
"""

from __future__ import annotations

from typing import Any

from .eval_metrics import mean


def _pct(value: float | None) -> str:
    """Format a 0-1 fraction as a percentage without trailing zeros."""
    if value is None:
        return "—"
    return f"{value * 100:.1f}".rstrip("0").rstrip(".") + "%"


def _count(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else "—"


def _num(value: Any, places: int) -> str:
    """Fixed-width number, so a column of perplexities lines up decimal-wise.

    JSON round-trips ``5.30`` back as ``5.3``; printing that next to ``5.23`` in a
    comparison column is exactly the kind of ambiguity this report exists to remove.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.{places}f}"


def _one_line(text: str, limit: int = 110) -> str:
    """Collapse whitespace and clip, for quoting an instruction inside a bullet."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _judge_scores(payload: dict[str, Any]) -> list[int]:
    return [
        row["judge_score"]
        for row in payload.get("details") or []
        if isinstance(row.get("judge_score"), int)
    ]


def _render_overall(payload: dict[str, Any], lines: list[str]) -> str | None:
    """Append the Overall table. Returns a note when the baseline is not comparable."""
    ppl = payload["perplexity"]
    gen = payload["generation"]
    cap = ppl.get("max_seq_length")
    coverage = ppl.get("coverage")

    if cap is None:
        ppl_label = "Perplexity (held-out, cap not recorded)"
    elif coverage is None:
        ppl_label = f"Perplexity (held-out, cap {cap})"
    else:
        ppl_label = f"Perplexity (held-out, cap {cap}, {_pct(coverage)} token coverage)"

    judge = _judge_scores(payload)
    lines += [
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| {ppl_label} | **{_num(ppl['perplexity'], 2)}** |",
        f"| Avg loss | {_num(ppl['loss'], 4)} |",
        f"| ROUGE-L F1 vs teacher | **{_num(gen['rouge_l'], 4)}** |",
        f"| Token-F1 vs teacher | {_num(gen['token_f1'], 4)} |",
        f"| LLM-as-judge (1-5) | **{_num(mean(judge), 2)}** over {len(judge)} rows |"
        if judge
        else "| LLM-as-judge (1-5) | **not run** — see Named gaps |",
    ]
    return _render_baseline_rows(payload, lines, cap)


def _render_baseline_rows(
    payload: dict[str, Any], lines: list[str], cap: int | None
) -> str | None:
    """Emit a baseline verdict only when both sides share a truncation cap."""
    baseline = payload.get("baseline_ppl")
    if not baseline:
        return None
    baseline_cap = payload.get("baseline_cap")
    current = payload["perplexity"]["perplexity"]

    if cap is not None and baseline_cap is not None and int(baseline_cap) == int(cap):
        lines.append(_verdict_row(baseline, baseline_cap, current, ""))
        return None

    cap_text = f"cap {baseline_cap}" if baseline_cap is not None else "cap unknown"
    lines.append(f"| Baseline PPL {baseline} ({cap_text}) | not compared |")

    # A sweep may still hold this run's figure at the baseline's own cap; that
    # pairing is same-cap and therefore the only honest comparison available.
    sweep = payload.get("perplexity_by_max_seq_length") or {}
    matched = sweep.get(str(baseline_cap)) if baseline_cap is not None else None
    if matched:
        lines.append(
            _verdict_row(baseline, baseline_cap, matched["perplexity"], " (matched from sweep)")
        )

    if baseline_cap is None:
        reason = "its truncation cap was never recorded"
    elif cap is None:
        reason = f"it was measured at cap {baseline_cap} and this run's cap was not recorded"
    else:
        reason = f"it was measured at cap {baseline_cap} and this run used cap {cap}"
    note = (
        f"Baseline PPL {baseline} is shown as context only: {reason}. Perplexities "
        "measured at different truncation caps score different fractions of the "
        "held-out set, so differencing them is meaningless."
    )
    if matched:
        note += (
            f" The sweep's own cap-{baseline_cap} figure ({matched['perplexity']}) is "
            "same-cap, so that pairing is compared instead."
        )
    return note


def _verdict_row(baseline: float, baseline_cap: Any, current: float, suffix: str) -> str:
    direction = "better" if current < baseline else "worse" if current > baseline else "unchanged"
    delta = (current - baseline) / baseline * 100 if baseline else 0.0
    return (
        f"| vs baseline PPL {baseline} @ cap {baseline_cap}{suffix} "
        f"| **{direction}** ({_num(current, 2)} vs {_num(baseline, 2)}, {delta:+.1f}%) |"
    )


def _render_conditions(payload: dict[str, Any], lines: list[str]) -> None:
    ppl = payload["perplexity"]
    cap = ppl.get("max_seq_length")
    num_samples = payload.get("num_samples") or len(payload.get("details") or [])
    lines += [
        "",
        "## Measurement conditions",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Perplexity truncation cap (`EVAL_MAX_SEQ_LENGTH`) | "
        f"{cap if cap is not None else 'not recorded'} |",
    ]
    if ppl.get("truncated_samples") is not None:
        lines.append(
            f"| Samples truncated by the cap | {ppl['truncated_samples']} / {num_samples} |"
        )
    if ppl.get("tokens_total"):
        lines.append(
            f"| Tokens scored | {_count(ppl.get('tokens_scored'))} / "
            f"{_count(ppl.get('tokens_total'))} |"
        )
    if ppl.get("coverage") is not None:
        lines.append(f"| Token coverage | {_pct(ppl['coverage'])} |")
    seed = payload.get("seed")
    lines.append(f"| Generation seed | {seed if seed is not None else 'not recorded'} |")

    sweep = payload.get("perplexity_by_max_seq_length") or {}
    if not sweep:
        return
    lines += [
        "",
        "### Perplexity by truncation cap",
        "",
        "| Cap | PPL | Avg loss | Samples truncated | Tokens scored | Coverage |",
        "|---|---|---|---|---|---|",
    ]
    for key in sorted(sweep, key=int):
        entry = sweep[key]
        mark = "**" if cap is not None and int(key) == int(cap) else ""
        lines.append(
            f"| {mark}{key}{mark} | {mark}{_num(entry.get('perplexity', '—'), 2)}{mark} "
            f"| {_num(entry.get('loss', '—'), 4)} "
            f"| {entry.get('truncated_samples', '—')} / {num_samples} "
            f"| {_count(entry.get('tokens_scored'))} / {_count(entry.get('tokens_total'))} "
            f"| {_pct(entry.get('coverage'))} |"
        )
    if cap is not None and str(cap) in sweep:
        lines += ["", f"Cap {cap} (bold) is this run's canonical figure."]


def _render_per_category(payload: dict[str, Any], lines: list[str]) -> None:
    ppl = payload["perplexity"]
    gen = payload["generation"]
    cap = ppl.get("max_seq_length")
    sweep = payload.get("perplexity_by_max_seq_length") or {}
    sweep_caps = sorted((int(key) for key in sweep), reverse=True)
    has_judge = any(
        entry.get("judge") is not None for entry in gen.get("per_category", {}).values()
    )

    ppl_headers = (
        [f"PPL @{c}" for c in sweep_caps]
        if sweep_caps
        else [f"PPL @{cap}" if cap is not None else "PPL"]
    )
    headers = ["Category", "Samples", *ppl_headers, "ROUGE-L", "Token-F1"]
    if has_judge:
        headers.append("Judge")
    lines += [
        "",
        "## Per category",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    categories = sorted(set(ppl["per_category"]) | set(gen["per_category"]))
    for category in categories:
        p = ppl["per_category"].get(category, {})
        g = gen["per_category"].get(category, {})
        if sweep_caps:
            cells = [
                _num(sweep[str(c)]["per_category"].get(category, {}).get("perplexity", "—"), 2)
                for c in sweep_caps
            ]
        else:
            cells = [_num(p.get("perplexity", "—"), 2)]
        row = [
            category,
            str(g.get("samples", p.get("samples", 0))),
            *cells,
            _num(g.get("rouge_l", "—"), 4),
            _num(g.get("token_f1", "—"), 4),
        ]
        if has_judge:
            row.append(_num(g["judge"], 2) if g.get("judge") is not None else "—")
        lines.append("| " + " | ".join(row) + " |")


def _render_examples(payload: dict[str, Any], lines: list[str]) -> None:
    scored = [
        row
        for row in payload.get("details") or []
        if isinstance(row.get("rouge_l"), (int, float))
    ]
    if len(scored) < 2:
        return
    ranked = sorted(scored, key=lambda row: row["rouge_l"], reverse=True)
    take = max(1, min(4, len(ranked) // 2))

    def _bullet(row: dict[str, Any]) -> str:
        return (
            f"- `[{row.get('category', '?')}] rouge={row['rouge_l']:.3f} "
            f"id={row.get('id', '?')}` — {_one_line(row.get('instruction', ''))}"
        )

    lines += ["", "## Best and worst by ROUGE-L", "", "### Best", ""]
    lines += [_bullet(row) for row in ranked[:take]]
    lines += ["", "### Worst", ""]
    lines += [_bullet(row) for row in reversed(ranked[-take:])]


def _render_gaps(payload: dict[str, Any], lines: list[str], baseline_note: str | None) -> None:
    gaps: list[str] = []
    if not _judge_scores(payload):
        gaps.append(
            "**LLM-as-judge not run.** No `judge_score` is present on any row, so this "
            "report rests on perplexity and ROUGE-L/token-F1 alone. Re-run with `--judge` "
            "once the judge API is reachable to add the third metric."
        )
    if payload.get("seed") is None:
        gaps.append(
            "**Generation is not reproducible — no seed recorded.** Answers are sampled "
            "with `do_sample=True`, so every ROUGE-L and token-F1 figure here is a single "
            "draw of unknown variance and small deltas cannot be distinguished from noise. "
            "Perplexity is unaffected (deterministic forward pass)."
        )
    coverage = payload["perplexity"].get("coverage")
    if coverage is not None and coverage < 1.0:
        gaps.append(
            f"**The truncation cap discarded {_pct(1 - coverage)} of the held-out tokens.** "
            f"Only {_pct(coverage)} of the split was scored, so this perplexity describes a "
            "smaller corpus than the split it is named after. Raise `EVAL_MAX_SEQ_LENGTH` "
            "or pass `--max-seq-length` for full coverage."
        )
    if baseline_note:
        gaps.append(baseline_note)

    lines += ["", "## Named gaps", ""]
    if not gaps:
        lines.append(
            "None recorded: judge scores present, generation seeded, and the truncation "
            "cap scored the entire held-out split."
        )
        return
    lines += [f"{index}. {text}" for index, text in enumerate(gaps, start=1)]


def render_report(payload: dict[str, Any]) -> str:
    """Render the evaluation payload as a markdown report.

    Everything below is generated from the payload. Nothing is asserted that the
    payload cannot support: sections without data are omitted or explicitly
    marked as not measured, and perplexities are never differenced across caps.
    """
    ppl = payload["perplexity"]
    cap = ppl.get("max_seq_length")
    coverage = ppl.get("coverage")
    num_samples = payload.get("num_samples") or len(payload.get("details") or [])
    cap_line = (
        f"**Perplexity cap:** {cap} ({_pct(coverage)} token coverage)  "
        if cap is not None
        else "**Perplexity cap:** not recorded  "
    )
    lines = [
        f"# Evaluation {payload['label']} — distilled Qwen2.5-1.5B",
        "",
        f"**Date:** {payload['evaluated_at'][:10]}  ",
        f"**Model:** `{payload['model_path']}`  ",
        f"**Test split:** {num_samples} held-out samples (never trained on)  ",
        cap_line,
        "",
    ]
    baseline_note = _render_overall(payload, lines)
    _render_conditions(payload, lines)
    _render_per_category(payload, lines)
    _render_examples(payload, lines)
    _render_gaps(payload, lines, baseline_note)
    lines += [
        "",
        "## Caveats",
        "",
        "- Perplexity is only comparable at an identical truncation cap; figures measured",
        "  at different caps are reported side by side, never differenced.",
        "- Test split is regenerated per dataset version; cross-version comparisons are",
        "  indicative, not apples-to-apples.",
        "- ROUGE-L against a single teacher reference under-credits valid alternative answers.",
        "",
    ]
    return "\n".join(lines)

