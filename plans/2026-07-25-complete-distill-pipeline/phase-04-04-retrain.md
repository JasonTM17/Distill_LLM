---
phase: 4
title: 04-retrain
status: completed
effort: 20-30m training
dependencies:
  - phase-03-03-split-held-out
---

# Phase 4: 04-retrain

## Overview
Retrain Qwen2.5-1.5B trên full dataset (~530 mẫu) với cùng hyperparameter đã verified ở v0.3. Save adapter + merged model mới, archive bản cũ.

## Context
- v0.3 trained trên 395 mẫu, loss 1.43→1.14, PPL 4.70 (in-sample).
- Plan này retrain trên toàn bộ 530 mẫu (sau Phase 2) → kỳ vọng PPL held-out tốt hơn và accuracy cao hơn.
- Adapter cũ: `checkpoints/adapter/` (~8.7MB). Merged cũ: `checkpoints/merged/model.safetensors` (1.6GB).

## Related Code Files
- `train_student.py` (driver)
- `config.py` (hyperparameters)
- `data/processed/dataset_train.json` (Phase 3 output)
- `checkpoints/adapter/` (overwrite)
- `checkpoints/merged/` (overwrite)
- `checkpoints/v0.3_adapter/` (new — archive old)
- `checkpoints/v0.3_merged/` (new — archive old)

## Implementation Steps

1. **Archive bản cũ** trước khi retrain:
   ```powershell
   Move-Item checkpoints/adapter checkpoints/v0.3_adapter
   Move-Item checkpoints/merged checkpoints/v0.3_merged
   ```
2. **Sanity check** dataset: `python -c "import json; d = json.load(open('data/processed/dataset_train.json')); print(len(d))"` → in ra ~530.
3. **Train**:
   ```powershell
   python train_student.py
   ```
   Chạy foreground, monitor log real-time (~20-30 phút cho 530 mẫu × 3 epochs).
4. **Verify output**:
   - `checkpoints/adapter/adapter_model.safetensors` tồn tại
   - `checkpoints/merged/model.safetensors` tồn tại (~1.6GB)
   - `checkpoints/adapter/checkpoint-*/trainer_state.json` có loss progression (3 logs giảm dần)

## Success Criteria
- [ ] Training chạy đủ 3 epochs không crash
- [ ] Final loss < 1.2 (cải thiện hoặc tương đương v0.3 là 1.14)
- [ ] Final token accuracy > 72%
- [ ] Adapter + merged files tồn tại
- [ ] Old v0.3 weights archived (không mất)

## Risk
- OOM mid-training → giảm `MAX_SEQ_LENGTH` xuống 384
- bitsandbytes crash trên RTX 3060 → fallback float16 thuần (slower)
- Power outage → chạy lại được (resume from scratch OK vì dataset đã save)

## Validation
- PowerShell: `Get-ChildItem checkpoints/adapter`, `Get-ChildItem checkpoints/merged`
- Đọc trainer_state.json: `epoch >= 3`, `loss[final] < 1.2`
- Log diff: so sánh loss progression với v0.3 (file cũ ở v0.3_adapter/checkpoint-39/)
