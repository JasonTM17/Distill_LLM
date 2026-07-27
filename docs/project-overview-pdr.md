# Project Overview — PDR (Product Development Requirements)

## Product

Chưng cất mô hình (knowledge distillation) từ GPT-5.5-xhigh sang Qwen2.5-1.5B, giúp chạy local trên GPU RTX 3060 6GB với chất lượng tiệm cận teacher.

## Core Requirements

| ID | Requirement | Status |
|----|------------|--------|
| R1 | Kết nối 9Router API và gọi cx/gpt-5.5-xhigh | ✅ |
| R2 | Sinh dataset đa dạng (code, toán, ML, tiếng Việt...) | ✅ 570 sinh / 568 qua quality gate, đủ 10/10 categories (v0.6) |
| R3 | Fine-tune student model trên 6GB VRAM (mục tiêu: QLoRA) | ✅ 1.5B — v0.5 chạy LoRA bf16 với `LOAD_IN_4BIT=false`; nhánh QLoRA 4-bit vẫn còn trong `train.py` nhưng bitsandbytes lỗi trong env này |
| R4 | Merge adapter và deploy model inference local | ✅ |
| R5 | Đánh giá chất lượng trên held-out set (không in-sample) | ✅ v0.5 PPL 5.23 @cap 2048 (100% token coverage) · 5.38 @cap 512 (khớp protocol v0.4, −22.4%) · ROUGE-L 0.1534 · [báo cáo v0.5](../plans/reports/evaluation-v0.5.md). v0.6: PPL 5.85 (lùi, thử nghiệm) · [báo cáo v0.6](../plans/reports/evaluation-v0.6.md) |
| R6 | Interactive chat để test model | ✅ |
| R7 | Expand dataset | ✅ 570 prompts (v0.5: 530; v0.6: +40 cho creative/vietnamese/reasoning) |
| R8 | Train student lớn hơn (3B) khi đủ VRAM | 🔜 |
| R9 | Support streaming generation | ✅ SSE streaming trong `services/api` |
| R10 | Export model sang GGUF/ONNX | ✅ GGUF Q4_K_M + Q5_K_M; ONNX chưa làm |

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Dataset size | ≥ 530 prompts | 570 sinh / 568 qua quality gate (v0.6); v0.5: 530/528; 10/10 categories |
| Training loss (3 epoch) | < 1.5 | 1.3785 |
| Validation loss (best) | — | 1.4092 (checkpoint-125) |
| Token accuracy | > 65% | không đo ở v0.5 (v0.4: 65.3%) — v0.5 đo ROUGE-L / token-F1, là metric khác |
| **Perplexity (held-out)** | < 10 | v0.5 **5.23** @cap 2048 (canonical) · v0.6 5.85 @cap 2048 (lùi — test split khó hơn + science/philosophy lùi; xem roadmap) · 5.38 @cap 512 |
| VRAM (train) | < 6GB | ~5GB |
| VRAM (inference) | < 3.5GB | ~3.5GB |
| Held-out test samples | ≥ 30 | v0.5: 51 (426/51/51) · v0.6: 54 (460/54/54) |
| Response quality | Clean code, accurate | ✅ |

## Tech Stack

- **Teacher API:** 9Router / OpenAI-compatible
- **Student:** Qwen2.5-1.5B-Instruct (Alibaba)
- **Training:** PyTorch + HuggingFace Transformers + PEFT — v0.5 train LoRA trên
  base bf16; QLoRA 4-bit là thiết kế dự kiến, code vẫn còn nhưng chưa chạy được
- **Quantization:** GGUF Q4_K_M / Q5_K_M (llama.cpp) cho serving. BitsAndBytes
  4-bit NF4 còn trong `train.py` sau cờ `LOAD_IN_4BIT` nhưng lỗi trong env này
- **Environment:** Windows 11, Python 3.14, RTX 3060 6GB

## Constraints

- **VRAM:** 6GB tối đa → v0.5 train LoRA bf16 + gradient checkpointing. QLoRA
  4-bit là hướng dự kiến để nới trần này, chưa chạy được trong env hiện tại
- **Ổ D:** 24GB trống → model + dataset vừa
- **API:** 9Router localhost, cần chạy nền
- **Python 3.14:** chưa có stable CUDA torch → phải dùng nightly

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Teacher API rate limit | Medium | Low | Delay 1.5s, retry 3 lần |
| OOM khi train | High | High | Giảm batch → 1, seq → 512, dùng 1.5B |
| bitsandbytes 4-bit lỗi trong env này | ⚠️ Đã xảy ra | Medium | v0.5 train LoRA trên base bf16 + gradient checkpointing. Đường thoát: quay lại QLoRA khi bnb chạy được (nhánh `LOAD_IN_4BIT` vẫn còn) |
| Ổ D full | Low | Medium | Monitoring, xóa cache |
