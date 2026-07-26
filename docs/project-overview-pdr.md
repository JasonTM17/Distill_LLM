# Project Overview — PDR (Product Development Requirements)

## Product

Chưng cất mô hình (knowledge distillation) từ GPT-5.5-xhigh sang Qwen2.5-1.5B, giúp chạy local trên GPU RTX 3060 6GB với chất lượng tiệm cận teacher.

## Core Requirements

| ID | Requirement | Status |
|----|------------|--------|
| R1 | Kết nối 9Router API và gọi cx/gpt-5.5-xhigh | ✅ |
| R2 | Sinh dataset đa dạng (code, toán, ML, tiếng Việt...) | ⚠️ 395/530 (rate-limited) |
| R3 | QLoRA fine-tune student model trên 6GB VRAM | ✅ 1.5B |
| R4 | Merge adapter và deploy model inference local | ✅ |
| R5 | Đánh giá chất lượng trên held-out set (không in-sample) | ✅ PPL=6.93, ROUGE=0.13 |
| R6 | Interactive chat để test model | ✅ |
| R7 | Expand dataset to 530 prompts | 🔜 (135 còn thiếu) |
| R8 | Train student lớn hơn (3B) khi đủ VRAM | 🔜 |
| R9 | Support streaming generation | 🔜 |
| R10 | Export model sang GGUF/ONNX | 🔜 |

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Dataset size | 530 prompts | 395 (8/10 categories) |
| Training loss (3 epoch) | < 1.5 | 1.44 |
| Token accuracy | > 65% | 65.3% |
| **Perplexity (held-out)** | < 10 | **6.93** (Excellent) |
| VRAM (train) | < 6GB | ~5GB |
| VRAM (inference) | < 3.5GB | ~3.5GB |
| Held-out test samples | ≥ 30 | 38 (stratified) |
| Response quality | Clean code, accurate | ✅ |

## Tech Stack

- **Teacher API:** 9Router / OpenAI-compatible
- **Student:** Qwen2.5-1.5B-Instruct (Alibaba)
- **Training:** PyTorch + HuggingFace Transformers + PEFT (QLoRA)
- **Quantization:** BitsAndBytes 4-bit NF4
- **Environment:** Windows 11, Python 3.14, RTX 3060 6GB

## Constraints

- **VRAM:** 6GB tối đa → bắt buộc QLoRA 4-bit
- **Ổ D:** 24GB trống → model + dataset vừa
- **API:** 9Router localhost, cần chạy nền
- **Python 3.14:** chưa có stable CUDA torch → phải dùng nightly

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Teacher API rate limit | Medium | Low | Delay 1.5s, retry 3 lần |
| OOM khi train | High | High | Giảm batch → 1, seq → 512, dùng 1.5B |
| bitsandbytes unstable | Medium | Medium | Float16, test kỹ |
| Ổ D full | Low | Medium | Monitoring, xóa cache |
