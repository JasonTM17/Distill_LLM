---
title: "Complete distill-gpt55 to production"
description: "Take the v0.4 distillation pipeline to a production-complete v0.5: full dataset, hardened generation, rigorous eval, GGUF export, containerized inference API + chat UI, CI."
status: in-progress
priority: P2
branch: "master"
tags: [ml, distillation, production]
blockedBy: []
blocks: []
created: "2026-07-26T10:02:21.166Z"
createdBy: "ck:plan"
source: skill
---

# Complete distill-gpt55 to production

## Overview

v0.4 shipped an honest held-out eval (PPL 6.93, 357 train / 38 test) but with known gaps:
127+ prompts lost to unretried transient API errors (philosophy + health = 0 samples),
mojibake in Vietnamese outputs, no validation split, no deployment artifacts.

This plan finishes the project: refactor scripts into a tested `src/distill` package,
regenerate the full 530-prompt dataset with a hardened client, retrain v0.5 with a
validation split, evaluate rigorously, export GGUF, and ship a containerized
FastAPI inference service + web chat UI with CI.

**Hardware constraints driving design:** RTX 3060 6GB VRAM (QLoRA 4-bit mandatory),
C: drive ~10GB free (small Docker images mandatory → GGUF/CPU serving in containers),
D: drive ~21GB free (clean reproducible artifacts before heavy phases).

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Package refactor + hardened generation](./phase-01-package-refactor-hardened-generation.md) | In progress |
| 2 | [Dataset quality pipeline](./phase-02-dataset-quality-pipeline.md) | Pending |
| 3 | [Retrain v0.5 with validation](./phase-03-retrain-v0-5-with-validation.md) | Pending |
| 4 | [Rigorous evaluation suite](./phase-04-rigorous-evaluation-suite.md) | Pending |
| 5 | [GGUF export](./phase-05-gguf-export.md) | Pending |
| 6 | [FastAPI inference service](./phase-06-fastapi-inference-service.md) | Pending |
| 7 | [Web chat UI](./phase-07-web-chat-ui.md) | Pending |
| 8 | [Docker compose + CI + tests + docs](./phase-08-docker-compose-ci-tests-docs.md) | Pending |

## Dependencies

- 1 → 2 → 3 → 4 → 5 (sequential: data → train → eval → export)
- 6 depends on 5 (serves GGUF in container) but its code can start once the OpenAPI
  contract is fixed; 7 depends on 6's contract; 8 depends on 6+7.
- Phase 3 (training) and phase 5 (GGUF perplexity check) are GPU-exclusive — never
  run concurrently with each other or with generation eval runs.
- External: 9Router API at localhost:20128 must be up for phases 1 and 4 (judge).

## Acceptance criteria

- [ ] ≥ 500/530 prompts with validated teacher outputs; philosophy + health present
- [ ] v0.5 trained on cleaned full dataset with train/val/test split; held-out PPL ≤ v0.4 (6.93)
- [ ] Eval report: PPL + ROUGE-L vs teacher + per-category, v0.4 vs v0.5 comparison
- [ ] GGUF Q4_K_M + Q5_K_M exported and smoke-tested
- [ ] FastAPI service with OpenAPI contract, /healthz /readyz /metrics, streaming chat
- [ ] Web chat UI in separate container, docker compose up boots the stack
- [ ] CI: lint + tests + Docker build; images pushed as nguyenson1710/distill-gpt55-{api,web}
- [ ] Docs synced (README, docs/, CHANGELOG); disk stayed healthy (C: ≥ 5GB, D: ≥ 5GB free)
