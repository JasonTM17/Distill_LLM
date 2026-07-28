# Deployment Guide

> **Language:** [Tiếng Việt](deployment-guide.md) · **English**

## Prerequisites

- Docker Desktop with Compose v2.
- Model file `checkpoints/gguf/distill-gpt55-v0.5-Q4_K_M.gguf`.

The model is not tracked in Git and is not included in container images. If the
artifact is unavailable, run the train → merge →
`python -m distill.export_gguf` pipeline. Serving uses llama.cpp on CPU; it does
not require a GPU or 9Router.

## Docker Compose

### Published images

`docker-compose.yml` points to Docker Hub:

```bash
docker compose pull
docker compose up --no-build
```

### Build from source

```bash
docker compose up --build
```

The web UI is available at http://localhost:3000 and the API at
http://localhost:8000. `./checkpoints/gguf` is mounted read-only at `/models`;
model weights are not baked into either image.

## Startup states

| Endpoint/state | Meaning | Result |
|---|---|---|
| `GET /healthz` | FastAPI process is alive | `200 {"status":"ok"}` |
| `/readyz`: `loading` | Runtime is loading the GGUF | `503` |
| `/readyz`: `ready` | Inference can accept requests | `200` |
| `/readyz`: `error` | Model loading failed | `503` with `detail` |

The API image healthcheck uses `/healthz`, so a standalone container can be
healthy while the model is still loading. Compose overrides it with `/readyz`;
the web service starts only after the API is ready.

```bash
docker compose logs api
curl http://localhost:8000/readyz
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}]}'
```

## Registries and tags

Every push to `master` publishes both images with `latest` and the full commit
SHA:

| Registry | API | Web |
|---|---|---|
| GHCR (always) | `ghcr.io/jasontm17/distill-gpt55-api` | `ghcr.io/jasontm17/distill-gpt55-web` |
| Docker Hub mirror | `nguyenson1710/distill-gpt55-api` | `nguyenson1710/distill-gpt55-web` |

GHCR uses `GITHUB_TOKEN`. The Docker Hub mirror runs only when
`DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` are configured; both secrets are
required for that mirror.

Direct GHCR pulls:

```bash
docker pull ghcr.io/jasontm17/distill-gpt55-api:latest
docker pull ghcr.io/jasontm17/distill-gpt55-web:latest
```

Compose uses Docker Hub names by default. To run GHCR images through Compose,
replace both `image:` values or use a dedicated Compose override.

## CI

- `.github/workflows/ci.yml`: runs on every pull request and every push to
  `master`; covers Ruff, core/API pytest, generated-client verification, Vitest,
  and the web build.
- `.github/workflows/docker-publish.yml`: builds and publishes API and web images
  on pushes to `master`.

## Run without Docker

### API — PowerShell

```powershell
cd services/api
pip install -r requirements-dev.txt
$env:MODEL_PATH='../../checkpoints/gguf/distill-gpt55-v0.5-Q4_K_M.gguf'
uvicorn app.main:app --port 8000
```

### API — Bash

```bash
cd services/api
pip install -r requirements-dev.txt
MODEL_PATH=../../checkpoints/gguf/distill-gpt55-v0.5-Q4_K_M.gguf \
  uvicorn app.main:app --port 8000
```

### Web

```bash
cd services/web
pnpm install
pnpm dev
```

`VITE_API_BASE_URL` is a build-time variable and defaults to
`http://localhost:8000`. If web and API use different origins, set it before
build/dev and add the web origin to `CORS_ALLOW_ORIGINS`.

OpenAI SDK clients that support a custom base URL can call `chat.completions`.
The API implements only a subset of the OpenAI contract: it has no `/v1/models`,
embeddings, or authentication, and the supplied API key is not validated.

## Chat data

History stays in browser `localStorage`: at most 30 conversations and 100
non-empty, non-error messages per conversation. Stopped partial output is saved.
Container restarts, volume backups, and image rollbacks do not restore history.

## Rollback

Replace `latest` with a known commit-SHA tag, then run:

```bash
docker compose pull
docker compose up --no-build
```

Do not use `--build` for an image rollback because a local build creates a new
image. Model rollback means selecting a previously verified GGUF through the
mount or `MODEL_PATH`.

## References

- [API runbook](../services/api/README.md)
- [Frontend guide](../services/web/README.md)
- [System architecture](system-architecture.en.md)
- [Security policy](../SECURITY.md)
