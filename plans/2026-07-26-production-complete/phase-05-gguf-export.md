---
phase: 5
title: "GGUF export"
status: pending
effort: "M"
---

# Phase 5: GGUF export

## Overview

Convert the v0.5 merged model to GGUF and quantize to Q4_K_M (deploy default) and
Q5_K_M (quality option). GGUF enables small CPU Docker images (phase 6) and
Ollama/llama.cpp use — critical because C: has only ~10GB free for Docker.

## Implementation steps

1. Obtain tooling (D: only, no C: installs):
   - `pip install gguf` + clone llama.cpp (shallow) OR download prebuilt Windows
     llama.cpp release binaries (llama-quantize, llama-cli) — prefer prebuilt.
   - `convert_hf_to_gguf.py` from llama.cpp on `checkpoints/merged/` → f16 GGUF (~3.1GB).
2. Quantize: f16 → Q4_K_M (~0.9GB) and Q5_K_M (~1.1GB) into `checkpoints/gguf/`.
3. Smoke test both with llama-cli: coherent Vietnamese + English + code answers.
4. Disk protocol (before step 1): need ~5GB headroom on D:.
   - Delete `checkpoints/v0.3_merged/` (~1.6GB, reproducible from v0.3_adapter).
   - Delete intermediate f16 GGUF after quantization succeeds.
   - Re-check C:/D: free after each step; abort if D: < 3GB.
5. `scripts/export_gguf.py` (or `src/distill/export_gguf.py`) wrapping the flow
   so it is reproducible.

## Files

- Create: `src/distill/export_gguf.py`
- Output: `checkpoints/gguf/distill-gpt55-v0.5-Q4_K_M.gguf`, `...Q5_K_M.gguf`

## Validation

- Both GGUFs load in llama-cli and answer a 3-prompt smoke set sensibly
- Q4_K_M perplexity spot-check within ~5% of merged fp16 (small sample OK)

## Risks / rollback

- Python 3.14 incompatibility with gguf/convert script → fall back to running
  converter under the skills venv or a pinned 3.12 env; document choice.
- Qwen2 arch support is mature in llama.cpp — low conversion risk.
