# System Architecture

## Overview

Two planes: an **offline training pipeline** (Python package `src/distill`, GPU) that
produces a GGUF model artifact, and an **online serving stack** (two containers) that
serves it.

```
          OFFLINE (RTX 3060 6GB)                          ONLINE (docker compose)
┌─────────────────────────────────────────┐      ┌───────────────────────────────────┐
│ prompts.json (570)                      │      │  ┌─────────┐   REST/SSE  ┌─────┐  │
│   ↓ distill.generate_dataset            │      │  │   web   │────────────▶│ api │  │
│ teacher_outputs.json (9Router API)      │      │  │ (nginx) │  OpenAI-    │     │  │
│   ↓ distill.dataset  (screen+split)     │      │  │ React   │  compatible │Fast │  │
│ train/validation/test (80/10/10)        │      │  └─────────┘             │API  │  │
│   ↓ distill.train    (LoRA bf16)        │      │       :3000              └──┬──┘  │
│ checkpoints/adapter (LoRA)              │      │                             │:8000│
│   ↓ distill.merge    (bf16 base)        │      │                    llama.cpp CPU  │
│ checkpoints/merged  (bf16)              │      │                             │     │
│   ↓ distill.evaluate (PPL, ROUGE-L)     │      │              /models volume ▼     │
│   ↓ distill.export_gguf                 │──────┼──▶ checkpoints/gguf/*.gguf (RO)   │
└─────────────────────────────────────────┘      └───────────────────────────────────┘
```

## Training pipeline (`src/distill`)

| Module | Responsibility |
|---|---|
| `config.py` | All tunables; env-overridable; loads `.env` (secrets never hardcoded) |
| `teacher_client.py` | OpenAI-compatible client: retryable/fatal error classification, exponential backoff + jitter, output validation (empty/short/U+FFFD rejected) |
| `generate_dataset.py` | Resumable generation over 570 prompts (v0.5: 530; v0.6/v0.7: +40 weak-category); atomic JSON writes; failures retried across runs |
| `dataset.py` | Quality gate (mojibake, dedup, min-length) → exact Qwen2.5 chat template (`<|im_start|>` tokens) → stratified 80/10/10 splits, seed 42 |
| `train.py` | LoRA on q/k/v/o over the **bf16** base, batch 1×8 + validation loop, early stopping, best-checkpoint restore. Rank env-overridable (`LORA_R`): v0.5/v0.6 ran r=16; v0.7 ran r=32 to test capacity. Branches on `LOAD_IN_4BIT`: the NF4 4-bit path exists but is unusable in this environment, so all shipped runs used the bf16 path (`fp16=False`, `bf16=True`) |
| `merge.py` | LoRA merged onto the **bf16** base (not a 4-bit dequant — GGUF conversion rejects bitsandbytes-quantized checkpoints) |
| `evaluate.py` + `eval_metrics.py` | Held-out PPL, ROUGE-L/token-F1 vs teacher, optional LLM-as-judge (9Router), markdown report |
| `export_gguf.py` | merged → f16 GGUF → Q4_K_M / Q5_K_M via llama.cpp; smoke test; deletes intermediate |
| `safetensors_pretouch.py` | Windows pagefile workaround: pre-reads safetensors into OS cache before mmap |

**Precision.** bf16 end-to-end. The RTX 3060 is Ampere, so bf16 compute is fully
supported, and every weight load goes through `model_loading.load_causal_lm`, which
hardcodes `dtype=torch.bfloat16` with no override. The shipped artifact agrees:
`checkpoints/merged/config.json` records `"dtype": "bfloat16"`.

**QLoRA is still the intended design, but it did not run.** 6GB VRAM makes 4-bit
QLoRA the natural fit, and `train.py` still branches on `LOAD_IN_4BIT` to build a
bitsandbytes NF4 config. That path is broken in this environment (see the known
limitations table in [project-roadmap.md](./project-roadmap.md)), so every run
(v0.5/v0.6/v0.7) trained LoRA on the full bf16 model with `LOAD_IN_4BIT=false`.
Restoring the 4-bit path is tracked on the roadmap; the branch is live code, not
a leftover.

Other constraints: batch 1 + grad-accum 8, and gradient checkpointing is required
for the bf16 path to fit in 6GB.

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
- Streaming token rendering (buffered SSE parser), stop/abort, `/readyz` polling,
  sanitized markdown (DOMPurify)
- Local-first conversation history in browser `localStorage`; no account, database,
  sync, server-side persistence, export, or recovery capability
- Image: node builder → nginx-unprivileged alpine, non-root, HEALTHCHECK

### Chat request and local-history data flow

The web service owns presentation state only. It does not expose an API and it never
posts persisted history to a storage service.

```text
Browser localStorage                         Browser memory                    API
distill-gpt55.chat-history.v1                selected conversation             FastAPI / llama.cpp
        │                                              │                               │
        ├─ app load → validate + bound ───────────────▶│                               │
        │                         system + saved valid messages + new prompt ────────▶ POST /v1/chat/completions
        │                                              │◀──────────── SSE token stream ┘
        │                                              ▼
        └◀── save after completion ─── completed user/assistant messages
```

`use-chat.ts` creates an assistant placeholder in memory before opening the stream, then
appends tokens to that one conversation. It disables selecting, creating and deleting
conversations while busy, so stream tokens cannot be written to a newly selected thread.
Persistence happens only after `busy` becomes false.

The storage module validates the stored JSON shape and fails closed: malformed JSON,
unexpected shape, unavailable storage and quota errors resolve to an empty/no-op store
instead of breaking the chat. Before writing, it keeps the 30 most recently updated
conversations and the last 100 non-empty, non-error messages in each. These limits bound
browser storage only; the client currently sends all valid selected messages to the API.

### Context boundary

The API defaults to a 4096-token llama.cpp context (`MAX_CONTEXT_TOKENS`). The web UI
does not tokenize, summarize, or truncate old history before constructing a request.
Therefore a persisted conversation can be within its 100-message storage limit yet still
exceed model context. This is an explicit local-demo trade-off, not a hidden guarantee of
unbounded conversation. Start a new thread when context becomes too long; token-aware
trimming/summarization would require a deliberate product and API decision.

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
