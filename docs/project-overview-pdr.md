# Project Overview — PDR (Product Development Requirements)

> **Ngôn ngữ:** Tiếng Việt (canonical) · [English](project-overview-pdr.en.md)

## Product

Fine-tune có giám sát Qwen2.5-1.5B từ output của GPT-5.5-xhigh để chạy local
trên RTX 3060 6GB. Chất lượng được đo bằng held-out perplexity và các metric
single-reference; chưa có LLM-as-judge nên chưa thể khẳng định ngang hoặc tiệm
cận teacher.

## Core Requirements

| ID | Requirement | Status |
|----|------------|--------|
| R1 | Kết nối 9Router API và gọi cx/gpt-5.5-xhigh | ✅ |
| R2 | Sinh dataset đa dạng (code, toán, ML, tiếng Việt...) | ✅ 570 sinh / 568 qua quality gate, đủ 10/10 categories (v0.6) |
| R3 | Fine-tune student model trên 6GB VRAM (mục tiêu: QLoRA) | ✅ 1.5B — v0.5/v0.6/v0.7 đều chạy LoRA bf16 với `LOAD_IN_4BIT=false`; v0.7 nâng rank r=16→32 (vẫn fit 6GB); nhánh QLoRA 4-bit vẫn còn trong `train.py` nhưng bitsandbytes lỗi trong env này |
| R4 | Merge adapter và deploy model inference local | ✅ |
| R5 | Đánh giá chất lượng trên held-out set (không in-sample) | ✅ v0.5 PPL 5.23 @cap 2048 (100% token coverage) · 5.38 @cap 512 (khớp protocol v0.4, −22.4%) · ROUGE-L 0.1534 · [báo cáo v0.5](../plans/reports/evaluation-v0.5.md). v0.6: PPL 5.85 (lùi) · [báo cáo v0.6](../plans/reports/evaluation-v0.6.md). v0.7: PPL 5.81 (thắng v0.6 trên cùng split, không phục hồi science/philosophy) · [báo cáo v0.7](../plans/reports/evaluation-v0.7.md) |
| R6 | Interactive chat để test model | ✅ |
| R7 | Expand dataset | ✅ 570 prompts (v0.5: 530; v0.6: +40 cho creative/vietnamese/reasoning) |
| R8 | Train student lớn hơn (3B) khi đủ VRAM | 🔜 |
| R9 | Support streaming generation | ✅ SSE streaming trong `services/api` |
| R10a | Export model sang GGUF | ✅ Q4_K_M + Q5_K_M |
| R10b | Export model sang ONNX | 🔜 Chưa triển khai |

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Dataset size | ≥ 530 prompts | 570 sinh / 568 qua quality gate (v0.6); v0.5: 530/528; 10/10 categories |
| Training loss v0.5 (3 epoch) | < 1.5 | 1.3785 |
| Validation loss v0.5 (best) | — | 1.4092 (checkpoint-125) |
| Token accuracy | > 65% | không đo ở v0.5 (v0.4: 65.3%) — v0.5 đo ROUGE-L / token-F1, là metric khác |
| **Perplexity (held-out)** | < 10 | v0.5 **5.23** @cap 2048, coverage 100% (canonical) · v0.6 5.85, coverage 95.4% · v0.7 **5.81**, coverage 95.4% (thắng v0.6 cùng split; kết quả ủng hộ giả thuyết mất cân bằng dữ liệu, không chứng minh quan hệ nhân quả) · v0.5 5.38 @cap 512 |
| VRAM (train) | < 6GB | ~5GB |
| Serving runtime | Chạy được không GPU | ✅ llama.cpp CPU; RAM/latency phụ thuộc host |
| Held-out test samples | ≥ 30 | v0.5: 51 (426/51/51) · v0.6: 54 (460/54/54) |
| Response quality | Held-out metrics + judge | PPL/ROUGE/token-F1 đã đo; LLM-as-judge và benchmark độc lập còn thiếu |

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
- **Disk:** cần theo dõi headroom trước train/export; không coi dung lượng trống
  của một máy tại một thời điểm là yêu cầu sản phẩm
- **API:** 9Router localhost, cần chạy nền
- **Python 3.14:** chưa có stable CUDA torch → phải dùng nightly

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Teacher API rate limit | Medium | Low | Delay 1.5s, retry 3 lần |
| OOM khi train | High | High | Giảm batch → 1, seq → 512, dùng 1.5B |
| bitsandbytes 4-bit lỗi trong env này | ⚠️ Đã xảy ra | Medium | v0.5 train LoRA trên base bf16 + gradient checkpointing. Đường thoát: quay lại QLoRA khi bnb chạy được (nhánh `LOAD_IN_4BIT` vẫn còn) |
| Hết dung lượng khi train/export | Medium | Medium | Kiểm tra free space thủ công, dọn cache và giữ artifact theo version |
