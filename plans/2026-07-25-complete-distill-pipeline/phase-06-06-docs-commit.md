---
phase: 6
title: 06-docs-commit
status: completed
effort: 1h
dependencies:
  - phase-05-05-evaluate
---

# Phase 6: 06-docs-commit

## Overview
Đồng bộ docs với code thật sau Phase 1-5: cập nhật README/PDR/roadmap, đánh dấu phase plan complete, commit từng logical unit theo conventional commits.

## Context
- Phase 1 sửa inconsistency → docs cũ có thể sai.
- Phase 4-5 ra kết quả mới (PPL held-out, ROUGE) → docs cần update.

## Related Code Files
- `README.md` (main results table)
- `docs/project-overview-pdr.md` (success metrics)
- `docs/project-roadmap.md` (v0.4 → completed, v0.5 next)
- `docs/system-architecture.md` (nếu có thay đổi)
- `docs/code-standards.md` (nếu có thay đổi)
- `plans/2026-07-25-complete-distill-pipeline/` (mark phases done)

## Implementation Steps

1. **Update `README.md`**:
   - Bảng kết quả: thêm row "v0.4" với PPL held-out thật
   - Project Structure: thêm `dataset_test.json`
   - Quick Start: giữ nguyên (đã đúng)
2. **Update `docs/project-overview-pdr.md`**:
   - Success Metrics: đổi "Dataset size 530" → thành actual achieved
   - Token accuracy, perplexity → update với held-out numbers
   - R7 (expand to 530) → check ✅
3. **Update `docs/project-roadmap.md`**:
   - v0.4 → check completed
   - v0.5 status update dựa trên kết quả
4. **Run ck plan checks**:
   ```powershell
   ck plan check phase-01-01-fix-bugs
   ck plan check phase-02-02-resume-generation
   ck plan check phase-03-03-split-held-out
   ck plan check phase-04-04-retrain
   ck plan check phase-05-05-evaluate
   ck plan check phase-06-06-docs-commit
   ```
5. **Commit** với conventional commits, MỖI phase 1 commit riêng (theo code-standards.md):
   - `fix: <description>` cho Phase 1
   - `feat: generate full 530 dataset` cho Phase 2
   - `feat: stratified train/test split` cho Phase 3
   - `feat: retrain on full 530 samples` cho Phase 4
   - `feat: rigorous held-out evaluation` cho Phase 5
   - `docs: sync v0.4 results` cho Phase 6
6. **Push** lên remote nếu user yêu cầu (mặc định KHÔNG push).

## Success Criteria
- [ ] README, PDR, roadmap updated với numbers thật
- [ ] Tất cả 6 phase được check completed trong plan
- [ ] ≥ 6 git commits (mỗi phase 1 commit)
- [ ] No secrets (.env) trong git diff
- [ ] Working tree clean

## Validation
- `git log --oneline -10` xem 6 commits mới
- `git diff HEAD~6 -- README.md` xem updates
- `ck plan status` confirm tất cả phases done
