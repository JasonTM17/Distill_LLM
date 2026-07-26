---
phase: 7
title: "Web chat UI"
status: pending
effort: "M"
---

# Phase 7: Web chat UI

## Overview

Separate frontend service (`services/web/`): a lightweight Vite + React + TS chat
interface talking to the phase-6 API. No Next.js API routes — backend stays the
FastAPI service per architecture rules. Ships as a static nginx container.

## Implementation steps

1. Scaffold `services/web/` with Vite react-ts template; pnpm.
2. Generate typed API client from `docs/openapi.yaml` (`openapi-typescript` +
   small fetch wrapper for SSE streaming — no hand-written contract types).
3. UI: single chat view — message list (user/assistant bubbles, markdown + code
   highlight), streaming token rendering, stop button, model/params drawer
   (temperature, max tokens), health indicator polling /readyz, empty state,
   error state (API down), loading skeleton. Dark theme default.
4. Config: `VITE_API_BASE_URL` env (default http://localhost:8000).
5. Tests: vitest component tests for message rendering + client wrapper;
   `services/web/README.md` per template.

## Files

- Create: `services/web/**` (Vite app, generated client, tests, README)

## Validation

- `pnpm dev` against local API: full chat round-trip with streaming visible
- `pnpm build` clean; `pnpm test` green; no hand-written API types

## Risks / rollback

- SSE parsing edge cases (chunk splits mid-event) → cover with unit test fixtures.
- Node/pnpm availability on machine → verify first; fall back to npm.
