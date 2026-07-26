---
phase: 1
title: 01-fix-bugs
status: completed
effort: 1h
dependencies: []
---

# Phase 1: 01-fix-bugs

## Overview
Fix bugs phát hiện trong khảo sát: UTF-8 corruption ở Vietnamese prompts, inconsistency trong config (`MERGED_DIR` vs `MERGED_MODEL_DIR`), docstring sai trong `train_student.py`, security (API key hardcoded trong `test_connection.py`).

## Context
- `gen_batch.py` đã chạy thành công 396/530 prompts. Nhiều prompt id 501-525 hiển thị `Vi���t m��Tt` — UTF-8 corrupt.
- Nghi vấn: API key hardcode trong `test_connection.py` (key `sk-59be692bbb02885c-kfjrks-07c55700` đã có trong `.env`).

## Related Code Files
- `gen_batch.py` (UTF-8 safe write/print)
- `config.py` (remove `MERGED_DIR` redundancy, ensure single source of truth)
- `train_student.py` (docstring đã sửa ở session trước — verify lại)
- `test_connection.py` (remove hardcoded API key, dùng `config.API_KEY`)

## Implementation Steps

1. **Verify `train_student.py` docstring** đã được sửa ở session trước (3B → 1.5B-Instruct).
2. **Fix `config.py`**: xóa `MERGED_DIR` (line 26) để chỉ còn `MERGED_MODEL_DIR`. Update `MERGED_DIR` references ở line 78 (for-loop tạo dirs) để dùng `MERGED_MODEL_DIR`.
3. **Fix `gen_batch.py`**:
   - Wrap payload trước khi gửi đi với `ensure_ascii=False` đã có. Thêm explicit encoding check.
   - In prompt preview với `errors='replace'` thay vì fail silently.
4. **Fix `test_connection.py`**: thay hardcoded API key bằng `import config; config.API_KEY`.

## Success Criteria
- [ ] `train_student.py` docstring khớp với model train (1.5B-Instruct)
- [ ] `config.py` chỉ còn 1 merged dir constant, không conflict
- [ ] `gen_batch.py` không có chỗ nào dùng print của non-UTF-8 string có thể gây exception
- [ ] `test_connection.py` đọc key từ config, không hardcode
- [ ] Không có thay đổi nào break file khác

## Tests
- `python -c "import config; print(config.MERGED_MODEL_DIR)"` → in ra path hợp lệ
- `python -c "from gen_batch import save; print('importable')"` → OK
- `python test_connection.py --help` (nếu có) hoặc grep để confirm không còn hardcoded key
