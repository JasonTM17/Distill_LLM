---
phase: 4
title: "Rigorous evaluation suite"
status: pending
effort: "M"
---

# Phase 4: Rigorous evaluation suite

## Overview

One evaluation module producing a versioned report: held-out perplexity, ROUGE-L
vs teacher, per-category breakdown, and v0.4 vs v0.5 comparison. Replaces the
heuristic `len>20` check permanently.

## Implementation steps

1. `src/distill/evaluate.py` (port evaluate.py + evaluate_extended.py):
   - Perplexity on `dataset_test.json` (never-seen samples), per-category.
   - Generation on test prompts → ROUGE-L / token-F1 vs teacher reference.
   - Optional `--judge` mode: LLM-as-judge via 9Router (JUDGE_MODEL config,
     1-5 scale rubric) — run only if 9Router is up; skip gracefully otherwise.
   - Emit `plans/reports/evaluation-v0.5.md` + `checkpoints/evaluation_results.json`
     with v0.4 numbers (PPL 6.93, from evaluation-v04.md) side by side.
2. `tests/test_evaluate.py`: metric math on tiny fixtures (ROUGE-L known values),
   report rendering, graceful judge skip.
3. Run on v0.5 merged model (GPU-exclusive; after phase 3 completes).

## Files

- Create: `src/distill/evaluate.py`, `tests/test_evaluate.py`
- Output: `plans/reports/evaluation-v0.5.md`, `checkpoints/evaluation_results.json`

## Validation

- Report includes: overall + per-category PPL, ROUGE-L, sample count per category,
  honest comparison table v0.4 vs v0.5, and a verdict on which model ships
- If v0.5 is worse than v0.4 on held-out: stop, analyze, decide with evidence

## Risks / rollback

- New test split differs from v0.4's 38-sample split → comparison is indicative,
  not apples-to-apples; state this caveat in the report explicitly.
