# web — chat UI for the distilled model

Single-view React frontend for the distilled Qwen2.5-1.5B model. It calls only the
`api` service through an OpenAI-compatible endpoint and renders completion tokens as
they arrive over SSE. It is intentionally local-first: no account service, database,
analytics, or server-side conversation store exists.

## What the UI does

| Capability | Behaviour |
|---|---|
| Readiness | Polls `GET /readyz` every 5 seconds; sending is disabled until ready |
| Streaming | Sends `POST /v1/chat/completions` with `stream: true`; Stop aborts the fetch |
| Rendering | User text is plain text; assistant Markdown is sanitized with DOMPurify |
| Settings | Temperature (0–1.5) and max tokens (64–2048) are local draft controls |
| History | Create, select and delete local browser conversations |
| Responsive UI | Sidebar on desktop; horizontally scrollable conversation strip at ≤720px |

During generation, history navigation and deletion are disabled. The streaming request
captures its target conversation before it starts; locking navigation prevents tokens
from being appended to a conversation the user selected later.

## API surface

None served — static SPA (nginx in Docker). Consumes the `api` service via a
client generated from [`docs/openapi.yaml`](../../docs/openapi.yaml):

```bash
pnpm run generate-client   # openapi.yaml -> src/api/schema.d.ts (committed)
```

Contract types are never hand-written; regenerate after any API change.

## Conversation history and privacy

The history boundary is `src/chat-history.ts`. It stores a JSON array at browser key
`distill-gpt55.chat-history.v1` and treats browser storage as optional.

| Rule | Implementation |
|---|---|
| Persistence scope | Current browser profile and device only |
| Conversation limit | 30 most recently updated conversations |
| Message limit | 100 completed messages per conversation |
| Conversation title | First non-empty user message; whitespace collapsed, clipped to 48 characters |
| Excluded from persisted data | Empty assistant placeholders and messages marked with an error |
| Corrupt/unavailable/full storage | Ignored safely; in-memory chat remains usable |
| Not provided | Authentication, sync, export, import, backup, recovery, server-side history |

Messages from the selected conversation are included in the next API request, following
the fixed system prompt and preceding the new user message. The storage limits are not
context-window management: the API defaults to `MAX_CONTEXT_TOKENS=4096`, while this UI
does not count or trim history tokens before sending. Start a new chat when a thread is
too long or no longer relevant.

To remove locally stored conversations, use the UI delete control or clear this site's
browser storage. Do not describe this behavior as encrypted, synced, or recoverable;
the code provides none of those guarantees.

## Env vars

| name | required | default | description |
|---|---|---|---|
| `VITE_API_BASE_URL` | no | `http://localhost:8000` | api service origin (build-time) |

## Run locally

Requires Node.js 22+ and pnpm 11.0.9 (Corepack reads the pinned version from
`package.json`).

```bash
cd services/web
pnpm install
pnpm dev          # http://localhost:3000 (expects api on :8000)
```

`VITE_API_BASE_URL` is injected at build time. For a non-default local API origin:

```bash
cd services/web
set VITE_API_BASE_URL=http://localhost:8001
pnpm dev
```

When serving the compiled bundle, rebuild after changing this value. The API must also
allow the web origin through `CORS_ALLOW_ORIGINS`.

## Test

```bash
pnpm test         # vitest: SSE parser + component tests
pnpm build        # tsc type check + production bundle
```

The tests cover SSE buffering, sanitized assistant rendering, history persistence and
limits, history sidebar callbacks, and chat-hook streaming behavior. They run in jsdom;
they do not load a GGUF or call a live API.

## Runbook

| Symptom | Check / action |
|---|---|
| **API offline** | Verify API process/container and browser reachability to `${VITE_API_BASE_URL}/readyz`; also check CORS origins. |
| **Model loading** | Normal after API startup. `/readyz` is `503` until GGUF load completes; wait and inspect API logs if it never becomes ready. |
| **Streaming stalls** | Check API logs. Stop aborts the browser fetch; the backend releases its model lock when it observes the abort. |
| **History disappeared** | Confirm browser/profile, private-window behavior, site-data cleanup, and localStorage availability. No server copy exists. |
| **History does not persist** | Browser storage may be full/blocked. The app deliberately continues in memory rather than surfacing a fatal error. |
| **Contract changed** | Run `pnpm run generate-client`, inspect the generated diff, then run `pnpm test` and `pnpm build`. |

See [`../../docs/design-guidelines.md`](../../docs/design-guidelines.md) for visual,
accessibility and responsive rules, and [`../../docs/system-architecture.md`](../../docs/system-architecture.md)
for the full request/data flow.
