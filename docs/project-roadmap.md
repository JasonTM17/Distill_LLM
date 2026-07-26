# Project Roadmap

## Current: v0.5 — Production Complete (in progress, 2026-07-26)

Full 530-prompt dataset, hardened pipeline as a tested package, retrain with
validation, GGUF export, containerized API + chat UI, CI. Plan:
`plans/2026-07-26-production-complete/`.

## Roadmap

### v0.4 — Held-out Evaluation (✅ 2026-07-25)
- [x] 357 train + 38 stratified test, held-out PPL 6.93, ROUGE-L 0.13
- [x] Known gaps: 134 prompts lost to unretried transient errors
  (philosophy + health = 0 samples), Vietnamese mojibake, no validation split

### v0.5 — Production Complete (🔄 2026-07-26)
- [x] `src/distill` package: resilient teacher client + resumable generation
- [x] **530/530 prompts generated** — all 10 categories, zero mojibake
- [x] Dataset quality gate + stratified 80/10/10 splits (426/51/51)
- [x] Exact Qwen chat template with special tokens (v0.4 trained without them)
- [x] Serving stack: FastAPI + llama.cpp container (validated end-to-end),
      Vite/React chat UI with generated API client, docker-compose, CI
- [x] Retrain with validation + early stopping (bf16 LoRA — see phase-03
      incident log for the torch-nightly/safetensors/bnb crashes); merged weights
      in `checkpoints/merged/`
- [x] **GGUF Q4_K_M + Q5_K_M exported** (0.92 GB / 1.05 GB) — CPU serving artifacts
      for the api container
- [x] **Evaluation report v0.5 vs v0.4** — held-out PPL **5.23** at cap 2048 (100%
      token coverage), **5.38** at cap 512 vs v0.4's 6.93 on the matched protocol
      (−22.4%), ROUGE-L 0.1534, per-category breakdown at all three caps:
      [`plans/reports/evaluation-v0.5.md`](../plans/reports/evaluation-v0.5.md)
- [x] README/docs sync
- [ ] Final compose smoke test — blocked, Docker is not up

### v0.6 — Advanced Distillation
- [ ] Teacher ensemble: add `cx/gpt-5.6-terra` as comparative teacher
- [ ] Fix or replace broken bnb 4-bit loading (torch stable 2.13?) → retrain
      QLoRA at seq 1024 (dataset already stores full-length samples)
- [ ] Benchmark suite (GSM8K-mini, HumanEval-mini, MMLU subset)
- [ ] True distillation with logit matching (requires open-weight teacher)

### v0.7 — Model Scaling
- [ ] Qwen2.5-3B student (`D:/models/qwen25-3b` already downloaded, 5.76GB)
- [ ] Flash Attention 2 / Unsloth for faster training

### v1.0 — Production hardening
- [ ] Auto-retrain pipeline (generate → format → train → eval → publish)
- [ ] Model registry/versioning; multi-quantization publishing

## Known limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| RTX 3060 6GB VRAM | **training-scoped**: seq 512 cap for bf16 LoRA (152K-vocab logits OOM at 1024). Not a box-wide limit — forward-only evaluation runs fine at cap 2048 | gradient checkpointing; QLoRA when bnb fixed |
| torch nightly + safetensors 0.8 | direct-to-GPU load and fp16-at-load crash (AV) | CPU-first loads, bf16 end-to-end (see phase-03) |
| bnb on-the-fly 4-bit broken in this env | QLoRA unavailable | LoRA on full bf16 behind `LOAD_IN_4BIT=false` |
| API-only teacher (no logits) | SFT, not true KL distillation | accepted trade-off |
| 9Router localhost only | generation needs local API up | resumable generation rides outages |
| C: drive low space | Docker + pagefile pressure | GGUF/CPU serving images; prune builds; watch free space |

## Dependencies

| Package | Version | Note |
|---------|---------|------|
| PyTorch | 2.12 nightly cu128 | required by Python 3.14; source of load crashes |
| transformers | 5.14.1 | new core_model_loading path |
| trl / peft | 1.9.0 / 0.19.1 | prompt-completion loss masking |
| llama.cpp | b10107 | `D:/tools` binaries + source for GGUF |
| llama-cpp-python | 0.3.34 | serving runtime (CPU wheel works on 3.14) |
