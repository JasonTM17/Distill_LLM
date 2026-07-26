# Project Roadmap

## Current: v0.4 — Held-out Evaluation

Date: 2026-07-25
Status: ✅ 357 train + 38 test stratified, perplexity 6.93 (Excellent), ROUGE-L 0.13

## Roadmap

### v0.4 — Held-out Evaluation (✅ Completed 2026-07-25)
- [x] Generate 395/530 prompts (rate-limited, philosophy + health = 0)
- [x] Stratified train/test split (90/10, seed=42)
- [x] Retrain Qwen2.5-1.5B on 357 samples (135 steps, 3 epochs)
- [x] Per-category PPL evaluation on held-out
- [x] ROUGE-L vs teacher comparison
- [x] Perplexity: 6.93 (Excellent), best math 3.61, worst creative 14.39

### v0.5 — Full 530 Dataset (In Progress)
- [ ] Wait for cx/gpt-5.5-xhigh quota reset (~daily cycle)
- [ ] Generate remaining 135 prompts (philosophy + health priority)
- [ ] Retrain on full 530 samples
- [ ] Re-evaluate to compare 395 vs 530 sample quality

### v0.6 — Advanced Distillation
- [ ] Teacher ensemble: add `cx/gpt-5.6-terra` as comparative teacher
- [ ] Try **Unsloth** (2x faster training, lower VRAM)
- [ ] Temperature sweep (T=2,4,6,8) for optimal soft labels
- [ ] Alpha sweep (α=0.3,0.5,0.7,0.9) for teacher/ground-truth balance
- [ ] **True distillation** with logit matching (requires open-weight teacher)

### v0.7 — Model Scaling
- [ ] Upgrade to Qwen2.5-3B when VRAM allows
- [ ] Multi-GPU support (if available)
- [ ] Gradient checkpointing optimization
- [ ] Flash Attention 2 integration

### v0.8 — Deployment
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
| **Daily API quota on cx/gpt-5.5-xhigh** | Generation rate-limited mid-run | Pre-touch safetensors workaround + wait for daily reset |
| **C: drive low space (<2GB)** | Pagefile cannot grow, model mmap fails | Pre-touch safetensors in `train_student.py`; ensure C: free |

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
