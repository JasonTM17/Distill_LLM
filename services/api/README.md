# api — distilled-model inference service

## Purpose

Serves the distilled Qwen2.5-1.5B model (GGUF via llama.cpp) over an
OpenAI-compatible REST API. Called by the `web` chat UI and any OpenAI SDK
client; calls nothing else.

## API surface

- `POST /v1/chat/completions` — chat completion, streaming (SSE) and non-streaming
- `GET /healthz` — liveness
- `GET /readyz` — readiness (503 until the model finished loading)
- `GET /metrics` — Prometheus metrics

Canonical contract: [`docs/openapi.yaml`](../../docs/openapi.yaml) (generated from
this app — regenerate after route changes, the web client is built from it).

## Env vars

| name | required | default | description |
|---|---|---|---|
| `MODEL_PATH` | yes | `checkpoints/gguf/distill-gpt55-v0.5-Q4_K_M.gguf` | GGUF file to serve |
| `MODEL_ID` | no | `distill-gpt55-qwen2.5-1.5b` | model name reported to clients |
| `MAX_CONTEXT_TOKENS` | no | `4096` | llama.cpp context window |
| `N_THREADS` | no | `0` (auto) | CPU threads for inference |
| `DEFAULT_MAX_TOKENS` | no | `512` | completion cap when request omits it |
| `MAX_TOKENS_LIMIT` | no | `2048` | hard request cap |
| `RATE_LIMIT_REQUESTS` | no | `60` | requests per window per client IP |
| `RATE_LIMIT_WINDOW_SECONDS` | no | `60` | rate-limit window |
| `CORS_ALLOW_ORIGINS` | no | `http://localhost:3000` | comma-separated origins |

Rate limiting is in-process (single-instance service; generations serialize on
the model lock). Scale-out would require a shared store (Redis) — tracked as a
non-goal for the local deployment.

## Run locally

```bash
cd services/api
pip install -r requirements-dev.txt
MODEL_PATH=../../checkpoints/gguf/distill-gpt55-v0.5-Q4_K_M.gguf uvicorn app.main:app --port 8000
```

## Test

```bash
cd services/api
python -m pytest tests/ -q     # no model/GPU needed (fake runtime)
```

## Runbook

- **Model swap:** replace the GGUF, update `MODEL_PATH`, restart; watch `/readyz`.
- **Slow first response:** model loads lazily (~10-30s); `/readyz` returns 503
  until done — orchestration should gate on it.
- **429s:** raise `RATE_LIMIT_REQUESTS` or check for a runaway client.
- **OOM/crash on load:** verify the GGUF quantization fits available RAM; Q4_K_M
  of the 1.5B needs ~1.5GB.
