# Deployment Guide

## Prerequisites

- A quantized model at `checkpoints/gguf/distill-gpt55-v0.5-Q4_K_M.gguf`
  (produced by `python -m distill.export_gguf` after training + merge).
- Docker Desktop (compose v2). No GPU required at serving time — the API runs
  llama.cpp on CPU.

## Local (docker compose) — recommended

```bash
docker compose up --build
# web UI:  http://localhost:3000
# API:     http://localhost:8000  (OpenAI-compatible)
```

The GGUF is mounted read-only into the api container (`./checkpoints/gguf:/models`);
it is never baked into an image. The web container waits for the api healthcheck
(`/readyz`) before starting.

### Startup states

`/healthz` and `/readyz` answer different operational questions:

| Endpoint | Meaning | Expected during cold start |
|---|---|---|
| `GET /healthz` | FastAPI process is alive | `200 {"status":"ok"}` |
| `GET /readyz` | GGUF runtime can accept inference | `503` with `loading` until model load completes; then `200` with `ready` |

The web UI polls `/readyz` every five seconds and disables sending until it receives a
successful response. A temporary `model loading` badge is expected after a restart.

Smoke test:

```bash
curl http://localhost:8000/readyz
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}]}'
```

## Local (no Docker)

```bash
# API
cd services/api && pip install -r requirements-dev.txt
MODEL_PATH=../../checkpoints/gguf/distill-gpt55-v0.5-Q4_K_M.gguf uvicorn app.main:app --port 8000

# Web
cd services/web && pnpm install && pnpm dev   # http://localhost:3000
```

Any OpenAI SDK also works directly against the API
(`base_url="http://localhost:8000/v1"`, any api_key).

### Local frontend origin

The Vite app uses `VITE_API_BASE_URL`, defaulting to `http://localhost:8000`. This is a
build-time value, not a runtime API setting. If the API runs on another origin, set the
value before `pnpm dev` or the production build and add the web origin to the API's
comma-separated `CORS_ALLOW_ORIGINS` list.

## Ollama / llama.cpp direct

The GGUF is standard; it can be served without this repo's stack:

```bash
llama-server -m checkpoints/gguf/distill-gpt55-v0.5-Q4_K_M.gguf --port 8080
```

## Images / CI

- CI (`.github/workflows/ci.yml`): ruff + pytest (package and api) + vitest +
  contract-drift check + web build, on every push/PR.
- Publishing (`.github/workflows/docker-publish.yml`): on push to `master`,
  builds and pushes `nguyenson1710/distill-gpt55-api` and
  `nguyenson1710/distill-gpt55-web` with `latest` + commit-SHA tags.
  Requires repo secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`
  (Settings → Secrets and variables → Actions).

## Environment variables

See per-service READMEs: [`services/api/README.md`](../services/api/README.md),
[`services/web/README.md`](../services/web/README.md). Training-side settings:
`.env.example` at the repo root.

## Resource expectations

| Component | RAM | Notes |
|---|---|---|
| api (Q4_K_M 1.5B) | ~1.5-2GB | CPU inference, ~10-25 tok/s on a modern laptop |
| web | ~30MB | static nginx |

## Browser-local chat history

The web UI stores history on the user's browser, not in either container. Docker volume
backups, API logs, image rollback and API restarts do **not** preserve or restore chat
history. The client stores at most 30 conversations and 100 completed messages per
conversation in the current browser profile's localStorage. Clearing site data, using a
different profile or using a browser that blocks localStorage removes/avoids that history.

For a shared or sensitive deployment, do not assume this local-only behavior meets data
retention, export, audit or recovery requirements. Those features are absent and need a
separate authenticated persistence design.

## Rollback

Images are tagged by commit SHA — pin a previous SHA in `docker-compose.yml`
(or `docker run`) to roll back. Model rollback = point `MODEL_PATH` at an older
GGUF file.
