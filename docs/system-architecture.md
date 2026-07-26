# System Architecture

## Overview

Two planes: an **offline training pipeline** (Python package `src/distill`, GPU) that
produces a GGUF model artifact, and an **online serving stack** (two containers) that
serves it.

```
          OFFLINE (RTX 3060 6GB)                          ONLINE (docker compose)
┌─────────────────────────────────────────┐      ┌───────────────────────────────────┐
│ prompts.json (530)                      │      │  ┌─────────┐   REST/SSE  ┌─────┐  │
│   ↓ distill.generate_dataset            │      │  │   web   │────────────▶│ api │  │
│ teacher_outputs.json (9Router API)      │      │  │ (nginx) │  OpenAI-    │     │  │
│   ↓ distill.dataset  (screen+split)     │      │  │ React   │  compatible │Fast │  │
│ train/validation/test (80/10/10)        │      │  └─────────┘             │API  │  │
│   ↓ distill.train    (QLoRA 4-bit)      │      │       :3000              └──┬──┘  │
│ checkpoints/adapter (LoRA)              │      │                             │:8000│
│   ↓ distill.merge    (fp16 base)        │      │                    llama.cpp CPU  │
│ checkpoints/merged  (fp16)              │      │                             │     │
│   ↓ distill.evaluate (PPL, ROUGE-L)     │      │              /models volume ▼     │
│   ↓ distill.export_gguf                 │──────┼──▶ checkpoints/gguf/*.gguf (RO)   │
└─────────────────────────────────────────┘      └───────────────────────────────────┘
```

## Training pipeline (`src/distill`)

| Module | Responsibility |
|---|---|
| `config.py` | All tunables; env-overridable; loads `.env` (secrets never hardcoded) |
| `teacher_client.py` | OpenAI-compatible client: retryable/fatal error classification, exponential backoff + jitter, output validation (empty/short/U+FFFD rejected) |
| `generate_dataset.py` | Resumable generation over 530 prompts; atomic JSON writes; failures retried across runs |
| `dataset.py` | Quality gate (mojibake, dedup, min-length) → exact Qwen2.5 chat template (`<|im_start|>` tokens) → stratified 80/10/10 splits, seed 42 |
| `train.py` | QLoRA (NF4 4-bit, LoRA r16 q/k/v/o, batch 1×8, fp16 base) + validation loop, early stopping, best-checkpoint restore |
| `merge.py` | LoRA merged onto **fp16** base (not the 4-bit dequant — GGUF conversion rejects bitsandbytes-quantized checkpoints) |
| `evaluate.py` + `eval_metrics.py` | Held-out PPL, ROUGE-L/token-F1 vs teacher, optional LLM-as-judge (9Router), markdown report |
| `export_gguf.py` | merged → f16 GGUF → Q4_K_M / Q5_K_M via llama.cpp; smoke test; deletes intermediate |
| `safetensors_pretouch.py` | Windows pagefile workaround: pre-reads safetensors into OS cache before mmap |

Key hardware constraints: 6GB VRAM → 4-bit QLoRA mandatory, fp16 (no bf16 on
RTX 3060), batch 1 + grad-accum 8.

## Serving stack (`services/`)

### api — FastAPI + llama.cpp (`services/api`)
- `POST /v1/chat/completions` (OpenAI-compatible, SSE streaming + non-streaming)
- `/healthz`, `/readyz` (503 until GGUF loaded), `/metrics` (Prometheus)
- llama-cpp-python CPU runtime, single-flight lock (llama ctx not thread-safe),
  background model load, per-IP sliding-window rate limit
- Image: python:3.12-slim multi-stage, non-root 65532, HEALTHCHECK, ~sub-500MB;
  GGUF mounted read-only at `/models` — never baked into the image

### web — React chat UI (`services/web`)
- Vite + React + TS; API client **generated** from `docs/openapi.yaml`
  (`pnpm run generate-client`) — no hand-written contract types
- Streaming token rendering (buffered SSE parser), stop/abort, health polling,
  sanitized markdown (DOMPurify)
- Image: node builder → nginx-unprivileged alpine, non-root, HEALTHCHECK

### Contract
`docs/openapi.yaml` is canonical, exported from the FastAPI app; CI fails if the
committed generated client drifts from it.

## Directory layout

```
distill-gpt55/
├── src/distill/           ← training pipeline package (tested in tests/)
├── services/
│   ├── api/               ← inference service (own Dockerfile, tests, README)
│   └── web/               ← chat UI (own Dockerfile, tests, README)
├── data/                  ← prompts + raw teacher outputs + processed splits
├── checkpoints/           ← adapter / merged / gguf artifacts (gitignored)
├── docs/                  ← this file, openapi.yaml, PDR, roadmap, standards
├── plans/                 ← ClaudeKit plans + evaluation reports
├── docker-compose.yml     ← boots api + web with GGUF volume
└── D:/models/qwen15-1.5b  ← base model cache (outside repo)
```

## External dependencies

- **9Router** (`localhost:20128`) — teacher + judge API; needed only for data
  generation and judge evaluation, never at serving time.
- **llama.cpp** (`D:/tools/llama-cpp-b10107` bin + `D:/tools/llama.cpp-src`) —
  GGUF conversion/quantization tooling; paths via `LLAMACPP_BIN`/`LLAMACPP_SRC`.
