---
title: Complete 530 dataset + retrain + evaluate
description: >-
  Hoàn thiện dataset 530 prompts, tách held-out, retrain Qwen2.5-1.5B, viết
  evaluation nghiêm túc, dọn docs, commit từng bước.
status: completed
priority: P1
branch: master
tags:
  - distill
  - ml
  - ck-workflow
blockedBy: []
blocks: []
created: '2026-07-25T07:46:07.852Z'
createdBy: 'ck:plan'
source: skill
---

# Complete 530 dataset + retrain + evaluate

## Overview

Dự án đã chạy được end-to-end với 396/530 prompts thành công, model 1.5B distill loss 1.43→1.14 / PPL 4.70. Tuy nhiên:
- 134 prompt fail (philosophy = 0, health = 0, nhiều Vietnamese bị UTF-8 corrupt)
- Không có held-out test set → perplexity đo trên training data
- Evaluation chỉ heuristic `len>20`
- Docs + config có vài inconsistency

Plan này đưa dự án lên production-grade: fix bug, generate đủ 530, tách held-out, retrain, benchmark nghiêm túc, dọn docs.

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [01-fix-bugs](./phase-01-01-fix-bugs.md) | Completed |
| 2 | [02-resume-generation](./phase-02-02-resume-generation.md) | In Progress |
| 3 | [03-split-held-out](./phase-03-03-split-held-out.md) | Completed |
| 4 | [04-retrain](./phase-04-04-retrain.md) | Completed |
| 5 | [05-evaluate](./phase-05-05-evaluate.md) | Completed |
| 6 | [06-docs-commit](./phase-06-06-docs-commit.md) | Completed |

## Dependencies

- Phase 2 chờ rate limit reset (429, ~29 phút từ lúc khảo sát 14:44)
- Phase 4 cần Phase 3 (held-out split) hoàn thành trước khi train
- Phase 5 cần Phase 4 (trained model) hoàn thành
- Phase 6 cuối cùng, sau khi mọi verify pass

## Risk

| Risk | Likelihood | Mitigation |
|---|---|---|
| Rate limit lặp lại khi resume gen | Medium | Tăng REQUEST_DELAY lên 3s, retry 5 lần |
| Train OOM với seq 512 + batch 1 | Low | Đã test ở v0.3, fallback: seq 384 |
| Held-out split lệch category | Low | Stratified split |
| Eval nghiêm túc fail (PPL tệ hơn v0.3) | Medium | Document thật lý do, không che giấu |

## Acceptance

- [ ] `data/raw/teacher_outputs.json` có 530 success (hoặc documented exceptions ≤ 5%)
- [ ] `data/processed/dataset_train.json` + `dataset_test.json` (stratified 90/10)
- [ ] Retrained adapter + merged model với loss progression rõ ràng
- [ ] Perplexity reported trên held-out, không phải training set
- [ ] At least 3 evaluation metrics (PPL + token-acc + qualitative)
- [ ] Docs (README, PDR, roadmap) khớp với code thật
- [ ] Conventional commits cho mỗi logical unit

## Cross-Plan Dependencies

Không có.
