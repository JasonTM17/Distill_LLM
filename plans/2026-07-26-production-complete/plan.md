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
| 1 | [Package refactor + hardened generation](./phase-01-package-refactor-hardened-generation.md) | Done |
| 2 | [Dataset quality pipeline](./phase-02-dataset-quality-pipeline.md) | Done |
| 3 | [Retrain v0.5 with validation](./phase-03-retrain-v0-5-with-validation.md) | Done |
| 4 | [Rigorous evaluation suite](./phase-04-rigorous-evaluation-suite.md) | In progress |
| 5 | [GGUF export](./phase-05-gguf-export.md) | Done |
| 6 | [FastAPI inference service](./phase-06-fastapi-inference-service.md) | Done |
| 7 | [Web chat UI](./phase-07-web-chat-ui.md) | Done |
| 8 | [Docker compose + CI + tests + docs](./phase-08-docker-compose-ci-tests-docs.md) | Done |

Phase 4 is the only phase still open: `plans/reports/evaluation-v05.md` has not been
produced yet and `checkpoints/evaluation_results.json` still holds v0.4 numbers.
Phases 6-8 ran ahead of phase 4 because they depend on the GGUF artifacts, not on
the eval verdict.

## Dependencies

- 1 → 2 → 3 → 4 → 5 (sequential: data → train → eval → export)
- 6 depends on 5 (serves GGUF in container) but its code can start once the OpenAPI
  contract is fixed; 7 depends on 6's contract; 8 depends on 6+7.
- Phase 3 (training) and phase 5 (GGUF perplexity check) are GPU-exclusive — never
  run concurrently with each other or with generation eval runs.
- External: 9Router API at localhost:20128 must be up for phases 1 and 4 (judge).

## Acceptance criteria

- [x] ≥ 500/530 prompts with validated teacher outputs; philosophy + health present
      — 530 raw, 528 accepted (2 rejected as too short), all 10 categories populated
      (`data/processed/dataset_stats.json`)
- [x] v0.5 trained on cleaned full dataset with stratified train/val/test split
      (426/51/51), early stopping on the validation split
      (`checkpoints/adapter/checkpoint-162/`, merged weights in `checkpoints/merged/`)
- [ ] Eval report: PPL + ROUGE-L vs teacher + per-category, v0.4 vs v0.5 comparison,
      including whether held-out PPL beats v0.4's 6.93 — open, phase 4 in progress
- [x] GGUF Q4_K_M + Q5_K_M exported (`checkpoints/gguf/`, 0.92 GB / 1.05 GB per
      `logs/export-gguf.log`)
- [x] FastAPI service with OpenAPI contract (`docs/openapi.yaml`), `/healthz`
      `/readyz` `/metrics`, streaming chat
- [x] Web chat UI in its own container; `docker-compose.yml` wires api + web
- [x] CI: lint + tests + web contract-sync check; `docker-publish.yml` pushes
      `nguyenson1710/distill-gpt55-{api,web}` on every push to master
- [ ] Docs synced (README, docs/, CHANGELOG); disk stayed healthy (C: ≥ 5GB, D: ≥ 5GB free)
- [ ] Final smoke test: `docker compose up` boots the stack end-to-end and both
      GGUF quantizations answer the smoke prompt set
