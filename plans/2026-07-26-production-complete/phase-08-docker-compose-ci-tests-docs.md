---
phase: 8
title: "Docker compose + CI + tests + docs"
status: pending
effort: "L"
---

# Phase 8: Docker compose + CI + tests + docs

## Overview

Containerize both services, wire compose, add CI (lint/test/build + Docker Hub
publish), finish test coverage, and sync all docs. C: has ~10GB free → images must
be small: API = python-slim + llama-cpp-python (CPU) ≈ 400MB; web = nginx-alpine ≈ 50MB.
GGUF model mounted as a volume, never baked into the image.

## Implementation steps

1. `services/api/Dockerfile`: multi-stage (builder wheels → slim runtime), non-root
   uid 65532, HEALTHCHECK /healthz, OCI labels. `services/web/Dockerfile`:
   node builder → nginx-alpine runtime, non-root, HEALTHCHECK.
2. `docker-compose.yml` (root): `api` (volume `./checkpoints/gguf:/models:ro`),
   `web` (depends_on api healthy), shared network; `.dockerignore` per service.
   Prune builder cache after local build to protect C: (`docker builder prune`).
3. CI `.github/workflows/ci.yml`: python lint (ruff) + pytest for src/ and
   services/api; pnpm build + vitest for web; advisory-first for any gate that
   would be red on day one. `docker-publish.yml`: build + push
   `nguyenson1710/distill-gpt55-api` and `-web` (latest + SHA tags) on master push
   — requires DOCKERHUB_USERNAME / DOCKERHUB_TOKEN repo secrets.
4. Repo hygiene: `.gitignore` covers checkpoints/, data/raw, logs/, .env*,
   private dirs (.claude/, .codex/ etc.); dependabot.yml;
   remove/deprecate superseded root scripts (gen_batch.py etc.) now that
   `src/distill` + services own the behavior.
5. Docs sync: README (v0.5 results, new quick start incl. compose), CHANGELOG,
   docs/system-architecture.md (service diagram), docs/deployment-guide.md,
   per-service READMEs verified, project-roadmap.md updated.

## Files

- Create: 2 Dockerfiles, docker-compose.yml, .dockerignore ×2,
  .github/workflows/{ci,docker-publish}.yml, .github/dependabot.yml
- Modify: README.md, CHANGELOG.md (new), docs/*, .gitignore

## Validation

- `docker compose up` → web at :3000 chats through api at :8000 (smoke test)
- CI workflow syntax valid; local `ruff check` + full pytest + vitest green
- `git status` shows no private files staged; C: free ≥ 5GB after builds

## Risks / rollback

- Docker Desktop disk pressure on C: → build images one at a time, prune between;
  if C: < 5GB abort builds and ship CI-built images only.
- No GitHub remote configured → commit workflows anyway; publishing waits for user.
