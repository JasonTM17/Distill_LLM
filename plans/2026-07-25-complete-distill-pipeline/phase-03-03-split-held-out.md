---
phase: 3
title: 03-split-held-out
status: completed
effort: 30m
dependencies:
  - phase-02-02-resume-generation
---

# Phase 3: 03-split-held-out

## Overview
Tách `data/processed/dataset_train.json` thành 2 file: `dataset_train.json` (90%, ~530 mẫu) và `dataset_test.json` (10%, ~60 mẫu), stratified theo category để đảm bảo mỗi category đều có mặt trong test set.

## Context
- Hiện tại toàn bộ 395 mẫu đều đi vào training. Perplexity đo trên 20 sample đầu của file này → không phản ánh generalization thật.
- `config.TEST_SPLIT_RATIO = 0.1` đã được define nhưng chưa dùng.

## Related Code Files
- `format_dataset.py` (cần refactor để split)
- `data/processed/dataset_train.json`
- `data/processed/dataset_test.json` (new)

## Implementation Steps

1. **Sửa `format_dataset.py`** để:
   - Load `teacher_outputs.json`
   - Filter `success=true` + `output` không rỗng
   - Group by category
   - Từ mỗi category, lấy `floor(N * 0.1)` mẫu cho test (stratified)
   - Save `dataset_train.json` (train split) + `dataset_test.json` (test split)
2. **Backward compat**: vì train_student.py đang load `PROCESSED_DATASET_FILE` (= `dataset_train.json`), sẽ tự động dùng train split mới → không cần sửa train script.
3. **Verify**:
   - Total = train + test = success count
   - Mỗi category có ≥ 1 mẫu test (nếu category có ≥ 10 success)
   - Categories nhỏ (philosophy/health) có test đại diện nếu có data

## Success Criteria
- [ ] `dataset_train.json` + `dataset_test.json` tồn tại, không trống
- [ ] Tổng samples = tổng success từ teacher_outputs.json
- [ ] Mỗi category có test:count ≥ 1 (nếu category đủ data)
- [ ] Script chạy idempotent (chạy lại không phá data)

## Tests
- PowerShell: kiểm tra counts từng file, kiểm tra category distribution
- Sanity: 1 sample đầu của test set in ra OK, format đúng chat template
