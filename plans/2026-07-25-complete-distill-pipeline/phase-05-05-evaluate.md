---
phase: 5
title: 05-evaluate
status: completed
effort: 30m
dependencies:
  - phase-04-04-retrain
---

# Phase 5: 05-evaluate

## Overview
Đánh giá model mới nghiêm túc: (1) perplexity trên held-out test set, (2) qualitative comparison với teacher, (3) ghi report `plans/reports/evaluation-v04.md`.

## Context
- v0.3 chỉ đo PPL trên training set (20 sample đầu) + heuristic `len>20`.
- Plan này evaluate trên `dataset_test.json` (held-out thật, ~60 sample stratified).

## Related Code Files
- `evaluate.py` (cần refactor để dùng `dataset_test.json`)
- `evaluate_extended.py` (cần refactor: thay `len>20` bằng so sánh với teacher)
- `plans/reports/evaluation-v04.md` (new)

## Implementation Steps

1. **Sửa `evaluate.py`**:
   - Load `config.PROCESSED_TEST_FILE` (mới thêm vào config) thay vì `PROCESSED_DATASET_FILE`
   - Tăng `MAX_EVAL_SAMPLES` lên 50-60 (full test set)
   - In per-category PPL nếu dataset có field `category`
2. **Thêm config constant**: `PROCESSED_TEST_FILE = os.path.join(PROCESSED_DIR, "dataset_test.json")`
3. **Sửa `evaluate_extended.py`** — thay heuristic `len>20` bằng:
   - Với mỗi prompt, lấy teacher output từ `teacher_outputs.json`
   - Tính ROUGE-L F1 (rouge-score package hoặc simple difflib.SequenceMatcher)
   - Pass nếu ROUGE-L ≥ 0.3 (heuristic threshold cho distillation quality)
4. **Chạy** cả 2 scripts:
   ```powershell
   python evaluate.py
   python evaluate_extended.py
   ```
5. **Viết report** `plans/reports/evaluation-v04.md`:
   - Bảng PPL held-out
   - Bảng ROUGE-L per category
   - So sánh v0.3 vs v0.4 (cùng prompts, khác dataset size)
   - Sample responses 3-5 prompts

## Success Criteria
- [ ] PPL held-out < 10 (matching v0.3 in-sample)
- [ ] ROUGE-L trung bình ≥ 0.3 trên test
- [ ] Mỗi category có ≥ 1 test sample evaluated
- [ ] Report Markdown đầy đủ 4 sections: Summary, PPL table, ROUGE table, Sample diff
- [ ] Report verify được bằng cách re-read commands chạy

## Risk
- Rouge-score package chưa cài → fallback pure Python (SequenceMatcher)
- Model output garbage ở test set → document thật
- ROUGE-L threshold 0.3 quá lỏng → adjust dựa trên kết quả

## Validation
- Re-run evaluate scripts sau khi viết report → confirm numbers khớp
- PowerShell check file size của report (>1KB)
