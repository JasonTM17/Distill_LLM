---
title: "v0.6 — Expand weak categories and retrain with judge eval"
description: "Add 40 prompts for creative/vietnamese/reasoning, regenerate from cx/gpt-5.5-xhigh, retrain, evaluate with LLM-as-judge."
status: in-progress
priority: P1
effort: 4h
branch: master
tags: [distillation, training, evaluation]
created: 2026-07-27
---

## Status

In progress. Generation of 40 new teacher outputs running in background.

## Context

v0.5 evaluation showed `creative` (PPL 14.95 = 3x overall) and `vietnamese`
(8.35) as genuine model weaknesses, plus `reasoning` had the fewest prompts
(43). LLM-as-judge was never run. This plan expands those three categories and
retrains.

## Phases

1. **Add prompts** — 40 new prompts (IDs 531-570): 15 creative, 15 vietnamese,
   10 reasoning. ✅ DONE
2. **Generate teacher outputs** — `cx/gpt-5.5-xhigh` via 9Router for the 40 new
   prompts. Resumable. 🔄 RUNNING (PID 32140, detached)
3. **Re-split dataset** — `python -m distill.dataset` rebuilds 80/10/10 splits
   with the expanded 570-prompt set.
4. **Retrain v0.6** — bf16 LoRA (r=16, q/k/v/o), seq 512, grad checkpointing,
   3 epochs, early stopping. Env: `LOAD_IN_4BIT=false
   GRADIENT_CHECKPOINTING=true MAX_SEQ_LENGTH=512`.
5. **Merge** — adapter → merged bf16 model at `checkpoints/merged/`.
6. **Evaluate v0.6** — `--label v0.6 --ppl-caps 512,1024,2048 --judge
   --baseline-ppl 5.23 --baseline-cap 2048`. Judge runs this time.
7. **Export GGUF** — Q4_K_M + Q5_K_M via llama.cpp. `GGUF_MODEL_BASENAME`
   env set to `distill-gpt55-v0.6` so v0.5 artifacts are not overwritten.
8. **Update docs** — roadmap, README, changelog.

## Dependencies

- Phase 2 → Phase 3 (need all teacher outputs before re-splitting)
- Phase 3 → Phase 4 (need splits before training)
- Phase 4 → Phase 5 → Phase 6 → Phase 7

## Acceptance criteria

- 570/570 teacher outputs generated (530 existing + 40 new)
- Dataset re-split with all 10 categories represented in train/val/test
- v0.6 merged model trained and evaluated
- PPL @2048 ≤ 5.23 (v0.5 baseline) — improvement target
- LLM-as-judge scores recorded (the metric v0.5 skipped)
- Per-category PPL for creative < 14.95 (improvement target)
- GGUF Q4_K_M + Q5_K_M exported as v0.6
- Tests still pass; ruff clean

## Risks

- 6 GB VRAM OOM during training (mitigated: bf16 LoRA + grad checkpointing,
  seq 512 — same config as v0.5)
- Teacher API instability (mitigated: resumable generation, retry with backoff)
- Evaluation generation killed mid-run (mitigated: partial results resume)
