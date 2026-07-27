# Project Roadmap

## Current: v0.6 closeout (2026-07-27)

v0.5 là model shipped/canonical. v0.6 mở rộng category yếu và retrain; cải thiện
được mục tiêu nhưng lùi overall → đóng lại như thử nghiệm, không ship. Tiếp theo:
v0.7 xử lý regression và chạy judge. Plans: `plans/2026-07-26-production-complete/`,
`plans/2026-07-27-v06-expand-and-retrain/`.

## Roadmap

### v0.4 — Held-out Evaluation (✅ 2026-07-25)
- [x] 357 train + 38 stratified test, held-out PPL 6.93, ROUGE-L 0.13
- [x] Known gaps: 134 prompts lost to unretried transient errors
  (philosophy + health = 0 samples), Vietnamese mojibake, no validation split

### v0.5 — Production Complete (✅ 2026-07-26) — canonical served model
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
- [ ] Final compose smoke test — blocked, Docker was not up at release time
      (manual smoke test documented in `docs/deployment-guide.md`)

### v0.6 — Expand weak categories + retrain (⚠️ 2026-07-27, experiment, not shipped)
Plan: `plans/2026-07-27-v06-expand-and-retrain/`. Report:
`plans/reports/evaluation-v0.6.md`.
- [x] Prompt catalogue expanded 530 → 570 (+15 creative, +15 vietnamese, +10 reasoning)
- [x] 570 teacher outputs generated, 568 accepted, stratified split 460/54/54
- [x] Retrain 3-epoch bf16 LoRA, best val loss 1.4599 @ step 174
- [x] Evaluate v0.6 — overall PPL **5.85** @cap 2048 (target ≤ 5.23: **NOT met**)
- [x] Target weak categories improved: creative 14.95→14.21, vietnamese
      8.35→7.02, reasoning 5.17→3.67, ml_ai 5.49→4.17
- [ ] `science` (4.81→6.06) and `philosophy` (5.26→6.22) regressed — expanding
      only weak categories diluted the strong ones
- [ ] LLM-as-judge still not run (judge API unreachable)
- [ ] v0.6 GGUF not exported — v0.5 GGUF remains the served artifact (better overall)

**Lesson:** a bare weak-category expansion lifts the targets but trades away
strong categories and raises the (harder, regenerated) test-set headline. v0.7
must expand without under-representing strong categories, and run the judge.

### v0.7 — Recovery + advanced distillation
- [ ] Beat v0.5's 5.23: re-expand catalogue without under-representing strong
      categories (rebalance, not just append); retrain; re-evaluate
- [ ] Run LLM-as-judge (the metric v0.5/v0.6 both skipped) once the judge API is up
- [ ] Teacher ensemble: add `cx/gpt-5.6-terra` as comparative teacher
- [ ] Fix or replace broken bnb 4-bit loading (torch stable?) → retrain QLoRA at
      seq 1024 (dataset stores full-length samples)
- [ ] Benchmark suite (GSM8K-mini, HumanEval-mini, MMLU subset)
- [ ] True distillation with logit matching (requires open-weight teacher)
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
