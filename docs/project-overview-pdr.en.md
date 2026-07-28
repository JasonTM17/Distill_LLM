# Project Overview — PDR (Product Development Requirements)

> **Language:** English translation. Canonical Vietnamese: [`project-overview-pdr.md`](project-overview-pdr.md)

## Product

Model distillation from GPT-5.5-xhigh to Qwen2.5-1.5B, enabling local execution on an RTX 3060 6GB GPU with quality approaching the teacher.

## Core Requirements

| ID | Requirement | Status |
|----|------------|--------|
| R1 | Connect to 9Router API and call cx/gpt-5.5-xhigh | ✅ |
| R2 | Generate a diverse dataset (code, math, ML, Vietnamese...) | ✅ 570 generated / 568 passed quality gate, covering all 10/10 categories (v0.6) |
| R3 | Fine-tune student model on 6GB VRAM (target: QLoRA) | ✅ 1.5B — v0.5/v0.6/v0.7 all run LoRA bf16 with `LOAD_IN_4BIT=false`; v0.7 raises rank r=16→32 (still fits 6GB); the QLoRA 4-bit branch is still in `train.py` but bitsandbytes errors in this env |
| R4 | Merge adapter and deploy local model inference | ✅ |
| R5 | Evaluate quality on held-out set (no in-sample) | ✅ v0.5 PPL 5.23 @cap 2048 (100% token coverage) · 5.38 @cap 512 (matches v0.4 protocol, −22.4%) · ROUGE-L 0.1534 · [v0.5 report](../plans/reports/evaluation-v0.5.md). v0.6: PPL 5.85 (regression) · [v0.6 report](../plans/reports/evaluation-v0.6.md). v0.7: PPL 5.81 (beats v0.6 on the same split, does not recover science/philosophy) · [v0.7 report](../plans/reports/evaluation-v0.7.md) |
| R6 | Interactive chat to test the model | ✅ |
| R7 | Expand dataset | ✅ 570 prompts (v0.5: 530; v0.6: +40 for creative/vietnamese/reasoning) |
| R8 | Train a larger student (3B) when VRAM is sufficient | 🔜 |
| R9 | Support streaming generation | ✅ SSE streaming in `services/api` |
| R10 | Export model to GGUF/ONNX | ✅ GGUF Q4_K_M + Q5_K_M; ONNX not done yet |

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Dataset size | ≥ 530 prompts | 570 generated / 568 passed quality gate (v0.6); v0.5: 530/528; 10/10 categories |
| Training loss (3 epoch) | < 1.5 | 1.3785 |
| Validation loss (best) | — | 1.4092 (checkpoint-125) |
| Token accuracy | > 65% | not measured at v0.5 (v0.4: 65.3%) — v0.5 measures ROUGE-L / token-F1, a different metric |
| **Perplexity (held-out)** | < 10 | v0.5 **5.23** @cap 2048 (canonical) · v0.6 5.85 @cap 2048 (regression) · v0.7 **5.81** @cap 2048 (beats v0.6 on same split, does not recover science/philosophy; root cause = dataset imbalance) · 5.38 @cap 512 |
| VRAM (train) | < 6GB | ~5GB |
| VRAM (inference) | < 3.5GB | ~3.5GB |
| Held-out test samples | ≥ 30 | v0.5: 51 (426/51/51) · v0.6: 54 (460/54/54) |
| Response quality | Clean code, accurate | ✅ |

## Tech Stack

- **Teacher API:** 9Router / OpenAI-compatible
- **Student:** Qwen2.5-1.5B-Instruct (Alibaba)
- **Training:** PyTorch + HuggingFace Transformers + PEFT — v0.5 trains LoRA on
  base bf16; QLoRA 4-bit is the intended design, code is still present but not yet runnable
- **Quantization:** GGUF Q4_K_M / Q5_K_M (llama.cpp) for serving. BitsAndBytes
  4-bit NF4 still in `train.py` behind the `LOAD_IN_4BIT` flag but errors in this env
- **Environment:** Windows 11, Python 3.14, RTX 3060 6GB

## Constraints

- **VRAM:** 6GB maximum → v0.5 trains LoRA bf16 + gradient checkpointing. QLoRA
  4-bit is the intended direction to raise this ceiling, not yet runnable in the current env
- **Drive D:** 24GB free → model + dataset fit
- **API:** 9Router localhost, needs to run in the background
- **Python 3.14:** no stable CUDA torch yet → must use nightly

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Teacher API rate limit | Medium | Low | Delay 1.5s, retry 3 times |
| OOM during training | High | High | Reduce batch → 1, seq → 512, use 1.5B |
| bitsandbytes 4-bit errors in this env | ⚠️ Has occurred | Medium | v0.5 trains LoRA on base bf16 + gradient checkpointing. Escape path: return to QLoRA when bnb works (the `LOAD_IN_4BIT` branch is still present) |
| Drive D full | Low | Medium | Monitoring, delete cache |