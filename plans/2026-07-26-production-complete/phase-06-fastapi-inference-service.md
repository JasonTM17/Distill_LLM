---
phase: 6
title: "FastAPI inference service"
status: pending
effort: "L"
---

# Phase 6: FastAPI inference service

## Overview

Standalone backend service (`services/api/`) serving the distilled model over an
OpenAI-compatible REST API. Serves the GGUF via llama-cpp-python so the Docker
image stays small (CPU, no torch/CUDA) — the same code path works locally.

## Implementation steps

1. `services/api/` layout: `app/main.py` (FastAPI factory), `app/routes_chat.py`
   (`POST /v1/chat/completions`, streaming SSE + non-streaming), `app/routes_ops.py`
   (`/healthz`, `/readyz`, `/metrics` Prometheus), `app/model_runtime.py`
   (llama-cpp-python wrapper, lazy load, single-flight lock), `app/schemas.py`
   (pydantic request/response validation), `app/rate_limit.py` (in-proc sliding
   window per IP — single-instance deployment; Redis not warranted here, note as
   a scale-out TODO in README).
2. Config via env: `MODEL_PATH` (GGUF), `MAX_CONTEXT_TOKENS`, generation defaults,
   `CORS_ALLOW_ORIGINS` (reuse names already in `src/distill/config.py`).
3. Commit OpenAPI contract: export `openapi.json` → `docs/openapi.yaml`; FE
   generates its client from it (phase 7).
4. `services/api/tests/`: schema validation, ops endpoints, chat happy path with a
   fake runtime, rate limit trip. No GPU/model needed in tests.
5. `services/api/README.md` per 6-section template (purpose, API, env, run, test, runbook).

## Files

- Create: `services/api/**` (app code, tests, README, requirements.txt)
- Create: `docs/openapi.yaml`

## Validation

- `uvicorn app.main:app` locally against `checkpoints/gguf/*Q4_K_M.gguf`:
  curl chat completion (stream + non-stream) returns model output
- pytest green without model present (fake runtime)
- /healthz always 200; /readyz 503 until model loaded, then 200

## Risks / rollback

- llama-cpp-python wheel availability on Python 3.14/Windows → if no wheel, pin
  service venv to Python 3.12 (service is isolated from training env anyway).
- 20-30s model load → readyz gate + lazy load keeps orchestration honest.
