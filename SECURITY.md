# Security Policy

## Supported versions

| Scope | Supported | Notes |
|---|---|---|
| Repository/software v0.7.x | ✅ | Current code and documentation line |
| Canonical model v0.5 | ✅ | Default served GGUF Q4_K_M |
| Experimental model v0.7 | ⚠️ | Exported locally, not promoted to serving |
| Experimental model v0.6 | ⚠️ | Evaluation only; no GGUF published |
| Repository/software < v0.5 | ❌ | Superseded |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems. Instead
report privately so a fix can ship before the issue is disclosed:

- Preferred: GitHub's private vulnerability reporting
  (`Security` → `Report a vulnerability` on this repo).
- Or email: **jasonbmt06@gmail.com** with `[distill-gpt55 security]` in the
  subject.

Include: a description, steps to reproduce, affected version, and any impact
notes. You should get an acknowledgement within 72 hours.

## Scope

In scope: this repository's code, the FastAPI serving service, the web client,
the training pipeline, and the committed configuration.

Out of scope: the upstream 9Router teacher API, llama.cpp, and your local
runtime/host — report those to their upstream maintainers.

## Hardening notes already in place

- `.env` / secrets are gitignored; only `.env.example` is tracked.
- The API has rate limiting, `/healthz` + `/readyz`, and serializes generation
  because the llama.cpp context is not thread-safe.
- Inputs are validated at the API boundary via Pydantic schemas
  (`services/api/app/schemas.py`).
- The API never returns stack traces to clients; errors are mapped to typed
  responses in `routes_chat.py`.

## No secrets policy

Never commit API keys, tokens, model weights, or datasets that contain personal
data. If a secret has been committed by accident, force-push history removal and
rotate the secret immediately — then open a private security report.
