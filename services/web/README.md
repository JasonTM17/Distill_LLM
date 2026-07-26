# web — chat UI for the distilled model

## Purpose

Single-view chat frontend for the distilled Qwen2.5-1.5B model. Calls only the
`api` service (OpenAI-compatible endpoint) with token streaming; nothing calls it.

## API surface

None served — static SPA (nginx in Docker). Consumes the `api` service via a
client generated from [`docs/openapi.yaml`](../../docs/openapi.yaml):

```bash
pnpm run generate-client   # openapi.yaml -> src/api/schema.d.ts (committed)
```

Contract types are never hand-written; regenerate after any API change.

## Env vars

| name | required | default | description |
|---|---|---|---|
| `VITE_API_BASE_URL` | no | `http://localhost:8000` | api service origin (build-time) |

## Run locally

```bash
cd services/web
pnpm install
pnpm dev          # http://localhost:3000 (expects api on :8000)
```

## Test

```bash
pnpm test         # vitest: SSE parser + component tests
pnpm build        # tsc type check + production bundle
```

## Runbook

- **"API offline" badge:** the UI polls `/readyz` every 5s — check the api
  container/process and CORS origins.
- **Streaming stalls mid-answer:** check api logs; the Stop button aborts the
  fetch so the backend frees the model lock at the next token.
- **After changing the API contract:** regenerate the client and rebuild, or
  types will silently drift.
