# Project Roadmap

## Current: v0.3 — Stable Distill Pipeline

Date: 2026-07-25
Status: ✅ Production-ready for inference, dataset growing

## Roadmap

### v0.4 — Full Dataset (In Progress)
- [ ] Generate full 530 prompts (230 remaining, ~3h API time)
- [ ] Retrain Qwen2.5-1.5B on 530 samples (3 epochs)
- [ ] Re-evaluate perplexity and token accuracy
- [ ] Compare performance: 200 vs 300 vs 530 samples

### v0.5 — Advanced Distillation
- [ ] Teacher ensemble: add `cx/gpt-5.6-terra` as comparative teacher
- [ ] Try **Unsloth** (2x faster training, lower VRAM)
- [ ] Temperature sweep (T=2,4,6,8) for optimal soft labels
- [ ] Alpha sweep (α=0.3,0.5,0.7,0.9) for teacher/ground-truth balance
- [ ] **True distillation** with logit matching (requires open-weight teacher)

### v0.6 — Model Scaling
- [ ] Upgrade to Qwen2.5-3B when VRAM allows
- [ ] Multi-GPU support (if available)
- [ ] Gradient checkpointing optimization
- [ ] Flash Attention 2 integration

### v0.7 — Deployment
- [ ] Export to **GGUF** (llama.cpp/ollama compatible)
- [ ] Export to **ONNX** (cross-platform inference)
- [ ] Docker container for inference
- [ ] Quantization comparison: Q4_K_M vs Q5_K_M vs Q8_0

### v1.0 — Production
- [ ] Benchmark suite (MMLU, HumanEval, GSM8K)
- [ ] CI/CD pipeline for auto-retraining
- [ ] Model versioning and registry
- [ ] REST API for inference
- [ ] Streaming generation support

## Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| RTX 3060 6GB VRAM | Cannot train >3B models | QLoRA 4-bit |
| Python 3.14 no stable CUDA wheels | Fragile install | Use nightly PyTorch |
| Windows no symlinks | Slower HF downloads | Direct download to D:/models/ |
| API-only teacher (no logits) | SFT not true distillation | Accept trade-off |
| 9Router localhost only | Cannot share training | Local dev only |

## Dependencies

| Package | Current | Target | Purpose |
|---------|---------|--------|---------|
| PyTorch | 2.12 nightly cu128 | 2.13 stable | CUDA training |
| transformers | 4.45+ | latest | Model loading |
| bitsandbytes | 0.49.2 | 0.50+ | 4-bit quantization |
| peft | latest | latest | LoRA adapters |
| trl | latest | latest | SFT Trainer |
| llama.cpp | — | future | GGUF inference |
| unsloth | — | future | Faster training |
