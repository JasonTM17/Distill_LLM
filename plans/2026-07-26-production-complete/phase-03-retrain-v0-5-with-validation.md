---
phase: 3
title: "Retrain v0.5 with validation"
status: pending
effort: "M"
---

# Phase 3: Retrain v0.5 with validation

## Overview

QLoRA retrain of Qwen2.5-1.5B-Instruct on the full cleaned dataset (~420+ train
samples vs 357 in v0.4), now with an eval loop on the validation split and early
stopping. Merge the adapter into a deployable fp16 model.

## Implementation steps

1. `src/distill/train.py` (port + upgrade `train_student.py`):
   - 4-bit NF4 load, LoRA r=16 α=32 on q/k/v/o, batch 1 × grad_accum 8,
     lr 2e-4 cosine, 3 epochs, fp16 (RTX 3060 has no bf16).
   - NEW: `eval_dataset` = validation split, `eval_steps=25`, early stopping
     patience 3, `load_best_model_at_end=True`, save_total_limit 2.
   - Keep the Windows pagefile pre-touch workaround from train_student.py.
   - Archive current adapter to `checkpoints/v0.4_adapter/` before overwriting.
2. `src/distill/merge.py`: merge adapter → `checkpoints/merged/` (fp16 safetensors),
   copy tokenizer files.
3. Pre-flight: C: ≥ 3GB free (pagefile), D: ≥ 6GB free (new merged model + old);
   GPU idle (no other CUDA process). Delete `checkpoints/v0.3_merged/` if D: is
   tight (reproducible from v0.3_adapter + base model).
4. Run training (~25-35 min expected), then merge, then smoke test generation.

## Files

- Create: `src/distill/train.py`, `src/distill/merge.py`
- Output: `checkpoints/adapter/` (v0.5), `checkpoints/merged/` (v0.5)
- Archive: `checkpoints/v0.4_adapter/`

## Validation

- Training completes without OOM; val loss curve decreasing then plateau
- `python -m distill.chat` (or test_model.py) produces coherent output from merged
- trainer_state.json shows eval_loss logged at each eval step

## Incident log (2026-07-26)

Two training launches crashed with 0xC0000005 at "Loading weights: 0%".
Bisection (tiny model OK, direct safetensors OK, bf16 full load OK):

- fp16 conversion during weight materialization AVs on torch nightly
  2.12.0.dev20260408 + transformers 5.14.1 + Python 3.14 on Windows.
- bnb on-the-fly 4-bit quantization AVs the same way regardless of dtype,
  even with the state dict preloaded in RAM.

Root cause narrowed: safetensors 0.8.0's direct-to-device path is broken
against this torch nightly — `safe_open(device='cuda:0').get_tensor()` raises
"Attempted to access the data pointer on an invalid python storage", and inside
transformers' loader the same incompatibility surfaces as a native AV. CPU
same-dtype loads are the only safe path.

Resolution: v0.5 trains **LoRA on the full bf16 model** (RTX 3060 is Ampere —
bf16 supported) with gradient checkpointing to fit 6GB VRAM, and every model
load in the package goes CPU-first then `.to("cuda:0")`. Run flags:
`LOAD_IN_4BIT=false GRADIENT_CHECKPOINTING=true`. QLoRA path kept in code
behind `LOAD_IN_4BIT` for environments where bnb loading works.

## Risks / rollback

- OOM on 6GB → reduce MAX_SEQ_LENGTH 1024→512 (v0.4 value) and retry
- Worse val loss than v0.4 → keep v0.4 adapter archive; compare in phase 4 before deciding
- "paging file too small" → known workaround already in code; check C: free space
