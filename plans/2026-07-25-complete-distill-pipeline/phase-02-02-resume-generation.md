---
phase: 2
title: "02-resume-generation"
status: completed
effort: "2-4h (depends on rate limit)"
dependencies: ["phase-01-01-fix-bugs"]
---

## Outcome

**Final: 395/530 success (74.5%)** — below target 525/530 due to daily API quota on `cx/gpt-5.5-xhigh`.

- `philosophy` (0/50) and `health` (0/51) = 0 samples — sustained 429 errors
- 80 other prompts (Vietnamese corruption + others) failed intermittently
- Background `gen_robust.py` ran for ~2 hours with 429 retry loops, no new successes
- **Decision:** proceed with 395 samples rather than block indefinitely on quota reset

The model still trains on a diverse 8-category dataset (coding, math, science, ml_ai, vietnamese, creative, business, reasoning). v0.5 (full 530) deferred until API quota resets daily.

# Phase 2: 02-resume-generation

## Overview
Chạy tiếp `gen_batch.py` để generate 134 prompts còn lại, đặc biệt là `philosophy` (0/50) và `health` (0/51). Script đã có resume logic nên sẽ tự skip 396 prompts đã có.

## Context
- Hiện tại: 396/530 success, 134 fail.
- API vừa trả 429 (rate limit) lúc 14:44, reset sau ~29 phút.
- Categories bị fail nặng nhất: philosophy (50), health (51), reasoning (3), business (5), Vietnamese corrupted prompts (~25).

## Related Code Files
- `gen_batch.py` (đã có resume)
- `config.py` (REQUEST_DELAY, MAX_RETRIES)
- `data/prompts.json` (input)
- `data/raw/teacher_outputs.json` (output, atomic save per prompt)

## Implementation Steps

1. **Trước khi chạy**: kiểm tra 9Router listening trên port 20128 + test 1 API call thử.
2. **Tăng độ robust**: edit `config.py`:
   - `REQUEST_DELAY = 3.0` (từ 1.5 — giảm rate limit hit)
   - `MAX_RETRIES = 5` (từ 3)
3. **Chạy** `python gen_batch.py` ở background (`run_in_background=true`).
4. **Monitor** log mỗi 5 phút: count successes, check fail reason.
5. **Special handling cho philosophy + health**: nếu tiếp tục fail, ghi lại fail reason, quyết định:
   - Option A: dùng teacher khác `cx/gpt-5.6-terra` cho 2 category này
   - Option B: bỏ qua + document trong báo cáo
6. **Validate final**: assert `success >= 525/530` (cho phép 5 fail do rate limit không recover).

## Success Criteria
- [ ] `teacher_outputs.json` có ≥ 525 success
- [ ] Mỗi category có ≥ 1 success (trừ khi user chọn bỏ)
- [ ] Không có UTF-8 corruption trong output mới (Phase 1 fix có hiệu lực)
- [ ] Log không còn 429 errors (nếu có thì retry pass)

## Tests / Validation
- PowerShell: `(Get-Content data/raw/teacher_outputs.json -Raw | ConvertFrom-Json).data | Where-Object success | Group-Object category | Select Name, Count`
- Confirm ≥ 8/10 categories có data
- Confirm total_tokens ≥ 1.28M (baseline hiện tại)

## Risk
- Rate limit reset giữa chừng → pause + retry
- Teacher API trả content không phải string (object, error wrapper) → gen_batch cần guard
- File write bị interrupt → atomic save đã có, OK
