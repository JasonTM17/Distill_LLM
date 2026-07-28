---
title: "v0.7 — Capacity retrain (LORA_R 32) to recover v0.6 regressions"
description: "Retrain on the existing 570 dataset with doubled LoRA rank to test whether more capacity recovers science/philosophy and beats v0.6's 5.85; run the judge."
status: completed
priority: P1
effort: 1h
branch: master
tags: [distillation, training, evaluation, capacity-experiment]
created: 2026-07-27
---

## Status

Completed (2026-07-28). v0.7 beat v0.6 on the same held-out split but did NOT
beat v0.5; GGUF exported and smoke-tested; judge did not run. See Outcome.

## Outcome

| Metric | v0.5 (canonical) | v0.6 | **v0.7** | Reading |
|---|---:|---:|---:|---|
| Overall held-out PPL @cap 2048 | 5.23 | 5.85 | **5.81** | v0.7 < v0.6 (same 54-split, ✓ target); > v0.5 (different split, indicative) |
| Best validation loss | 1.4092 | 1.4599 | **1.4555** | v0.7 < v0.6 on the same val split |
| `creative` PPL | 14.95 | 14.21 | **14.11** | best of three |
| `vietnamese` PPL | 8.35 | 7.02 | **7.01** | best of three |
| `reasoning` PPL | 5.17 | 3.67 | **3.60** | best of three |
| `ml_ai` PPL | 5.49 | 4.17 | **4.15** | best of three |
| `math` PPL | 2.98 | 2.89 | **2.84** | best of three |
| `science` PPL | 4.81 | 6.06 | **6.02** | did NOT recover — still ~v0.6 |
| `philosophy` PPL | 5.26 | 6.22 | **6.22** | did NOT recover — unchanged from v0.6 |
| ROUGE-L vs teacher | 0.1534 | 0.1481 | 0.1435 | slightly down each version |
| LLM-as-judge | not run | not run | not run | API unreachable again |
| GGUF Q4_K_M + Q5_K_M | shipped | not exported | exported + smoke-tested | |

### What the experiment proved

Doubling LoRA rank (r=16 → r=32) squeezed out small gains over v0.6 on the same
split (5.85 → 5.81; val 1.4599 → 1.4555) and lifted the targeted weak categories
to their best-ever values. It did **not** recover `science` (6.06 → 6.02) or
`philosophy` (6.22 unchanged). Capacity was therefore **not** the root cause of
the v0.6 regression — the dataset imbalance from expanding only the weak
categories is. v0.8 must rebalance the catalogue (top up the strong categories
too) rather than just raise adapter capacity.

### Decision

v0.7 is exported and smoke-tested but **not promoted to canonical/served**: it
beats v0.6 but not v0.5, and the comparison to v0.5 is across different test
splits (indicative only). v0.5 Q4_K_M remains the served artifact. v0.7 Q4_K_M +
Q5_K_M are available in `checkpoints/gguf/` for anyone who wants to serve them
locally. See `docs/project-roadmap.md`.

## Context

v0.6 expanded only the weak categories (creative/vietnamese/reasoning) and
regressed overall: PPL 5.85 vs v0.5's 5.23, with science 4.81->6.06 and
philosophy 5.26->6.22. Hypothesis: the r=16 LoRA adapter reallocated its limited
capacity to the new hard samples and forgot the strong categories.

## Strategy — ONE changed variable

Retrain on the **same 570 dataset** (no new generation -> same stratified split,
seed 42 -> same 54-sample held-out set -> directly comparable to v0.6's 5.85).
Only change vs v0.6: **LORA_R 16 -> 32, LORA_ALPHA 32 -> 64** (keep alpha/r = 2)
for more adapter capacity. Everything else identical: LOAD_IN_4BIT=false,
GRADIENT_CHECKPOINTING=true, MAX_SEQ_LENGTH=512, NUM_EPOCHS=3, LR 2e-4,
early-stopping patience 3, target modules q/k/v/o.

## Phases

1. Back up v0.6 adapter -> `checkpoints/v0.6_adapter/` (re-mergeable later).
2. Train v0.7 (detached, `scripts/run-train-v07.py`) -> `logs/train-v07.log`.
3. Merge -> `checkpoints/merged/`.
4. Evaluate: `--label v0.7 --ppl-caps 512,1024,2048 --judge` (judge API is up).
5. If overall PPL @2048 < v0.6's 5.85 (and competitive with v0.5's 5.23):
   export GGUF Q4_K_M + Q5_K_M as v0.7 (`GGUF_MODEL_BASENAME=distill-gpt55-v0.7`).
6. Update docs (roadmap, changelog, README, PDR) + commit + push.

## Acceptance criteria

- v0.7 overall held-out PPL @cap 2048 < **5.85** (v0.6, same test split) — primary
- science PPL recovers toward 4.81 (v0.5); philosophy toward 5.26
- LLM-as-judge scores recorded (the metric v0.5/v0.6 both skipped)
- ruff clean; tests pass
- If the primary target is NOT met: report honestly, keep v0.5 canonical, do NOT
  ship a v0.7 GGUF

## Comparison discipline

- v0.7 vs v0.6: same 54-sample test split -> apples-to-apples (primary comparison)
- v0.7 vs v0.5 (5.23): different test splits (54 vs 51) -> indicative only, never
  differenced
- Every perplexity reports its truncation cap

## Risks

- r=32 OOM on 6GB (mitigated: LoRA params are ~35MB vs 3GB base; the OOM bound is
  the 152K-vocab logits at seq 512, unchanged from v0.6 which fit)
- Judge API flakiness (mitigated: judge is optional; PPL/ROUGE are the core metrics)
- Still does not beat 5.85 (mitigated: honest reporting; v0.5 stays canonical)
