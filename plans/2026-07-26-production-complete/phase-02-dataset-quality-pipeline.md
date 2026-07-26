---
phase: 2
title: "Dataset quality pipeline"
status: done
effort: "M"
---

# Phase 2: Dataset quality pipeline

## Overview

Turn raw teacher outputs into clean train/validation/test splits with a quality gate.
v0.4's `format_dataset.py` did a bare chat-template conversion with a 90/10 split and
no validation set, no dedup, no truncation handling.

## Implementation steps

1. `src/distill/dataset.py`:
   - Load raw outputs; keep only `success=True` records.
   - Quality gate: reject mojibake (U+FFFD or CJK-garbage heuristic), output < 40 chars,
     near-duplicate instructions (normalized-text hash), optionally flag `truncated=True`
     records (keep but count — teacher hit token ceiling mid-answer).
   - Stratified split by category: 80/10/10 train/validation/test, seed 42
     (config: TEST_SPLIT_RATIO, VALIDATION_SPLIT_RATIO, SPLIT_SEED).
   - Apply Qwen chat template (system + user + assistant) — port from format_dataset.py.
   - Write `dataset_train.json`, `dataset_validation.json`, `dataset_test.json`,
     `dataset_stats.json` (per-category counts, token stats, rejected counts + reasons).
2. `tests/test_dataset.py`: quality gate cases, stratification proportions, determinism
   with fixed seed, no id overlap between splits.
3. Run pipeline on the full regenerated data; review `dataset_stats.json`.

## Files

- Create: `src/distill/dataset.py`, `tests/test_dataset.py`
- Output: `data/processed/dataset_{train,validation,test}.json`, `dataset_stats.json`

## Validation

- pytest green; splits disjoint by id; every category appears in all three splits
  (where ≥ 3 samples exist); stats file reports ≥ 500 accepted or documents why fewer.

## Risks / rollback

- Old `dataset_train.json` (v0.4) is overwritten — v0.4 metrics already recorded in
  plans/reports/evaluation-v04.md, and raw data remains the source of truth.
