---
phase: 1
title: "Package refactor + hardened generation"
status: done
effort: "M"
---

# Phase 1: Package refactor + hardened generation

## Overview

Root-level scripts (`gen_batch.py` etc.) lost 134/530 samples in v0.4 because transient
API errors were cached as permanent failures, and Vietnamese outputs were mojibake-corrupted.
Replace with a tested `src/distill` package featuring a resilient teacher client.

## Current state (2026-07-26)

- DONE: `src/distill/{config,teacher_client,generate_dataset,logging_utils}.py` written.
  Client classifies retryable vs fatal errors, exponential backoff + jitter, validates
  output (empty/short/U+FFFD rejected), atomic JSON writes, cross-run failure retry.
- RUNNING: full regeneration `python -m distill.generate_dataset --retry-failed`
  (PID 60140, started 17:07). At 17:47: 451/522 success, 1.35M tokens.
  9Router connection is intermittent — client rides through outages via backoff.

## Remaining steps

1. Wait for generation run to exit; inspect summary + per-category counts in
   `logs/generate.log`. Re-run `--retry-failed` for any remaining failures
   (possibly with higher `MAX_RETRIES` / longer delay) until ≥ 500/530 or failures
   are proven fatal (e.g. content policy) — document any permanently failed ids.
2. Add `pyproject.toml`: project metadata, `src` layout, pytest config,
   deps split (core vs train vs serve extras).
3. Add `tests/test_teacher_client.py` (error classification, backoff bounds, retry
   loop with fake client, output validation) and `tests/test_generate_dataset.py`
   (select_pending logic, atomic write, resume-from-existing, failure records).
4. Verify old root scripts still work or mark them deprecated in favor of
   `python -m distill.*` entry points (delete only in phase 8 docs sync).

## Files

- Modify: `pyproject.toml` (new), `tests/` (new)
- Keep: `src/distill/*` (already written), `data/raw/teacher_outputs.json` (growing)

## Validation

- `python -m pytest tests/ -q` green
- `logs/generate.log` final summary: successful_total ≥ 500/530, no mojibake
  (validated by client), philosophy + health categories non-zero

## Risks / rollback

- 9Router outage stalls generation → resumable by design, just re-run.
- Prompts that always fail (content policy) → cap effort at 3 full retry passes,
  document the ids, move on; dataset ≥ 500 is acceptable.
