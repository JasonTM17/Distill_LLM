# Project Roadmap

> **Language:** English translation. Canonical Vietnamese: [`project-roadmap.md`](project-roadmap.md)

## Current: v0.7 closeout (2026-07-28)

v0.5 is the shipped/canonical model. v0.6 expanded weak categories and regressed
overall. v0.7 increased adapter capacity (LoRA r=32) on the same split and beat
v0.6 (5.81 < 5.85), but did not recover science/philosophy. This supports the
data-imbalance hypothesis and shows that rank alone is insufficient; it does not
establish causality. v0.5 remains canonical; v0.7 GGUF is available locally. Next:
v0.8 rebalance catalogue + run judge. Plans: `plans/2026-07-26-production-complete/`,
`plans/2026-07-27-v06-expand-and-retrain/`, `plans/2026-07-27-v07-capacity-retrain/`.

## Version status

| Scope | Version/status |
|---|---|
| Repository/changelog | v0.7.0 |
| Default canonical served model | v0.5 Q4_K_M |
| Local experimental artifacts | v0.6 (no GGUF), v0.7 (exported, not promoted) |
| API software reports | 0.5.0 |

## Roadmap

### v0.4 — Held-out Evaluation (✅ 2026-07-25)
- [x] 357 train + 38 stratified test, held-out PPL 6.93, ROUGE-L 0.13
- [x] Known gaps: 134 prompts lost to unretried transient errors
  (philosophy + health = 0 samples), Vietnamese mojibake, no validation split

### v0.5 — Canonical local-serving baseline (✅ 2026-07-26)
- [x] `src/distill` package: resilient teacher client + resumable generation
- [x] **530/530 prompts generated** — all 10 categories, zero mojibake
- [x] Dataset quality gate + stratified 80/10/10 splits (426/51/51)
- [x] Exact Qwen chat template with special tokens (v0.4 trained without them)
- [x] Serving stack: FastAPI + llama.cpp container (API/web and images tested),
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
strong categories and raises the (harder, regenerated) test-set headline.

### v0.7 — Capacity retrain (LoRA r=32) (⚠️ 2026-07-28, exported, not canonical)
Plan: `plans/2026-07-27-v07-capacity-retrain/`. Report:
`plans/reports/evaluation-v0.7.md`.
- [x] Retrain on the same 570 dataset / 54-sample split as v0.6 with LoRA rank
      doubled (r=16 → r=32, alpha 64) — one changed variable
- [x] Overall PPL @cap 2048 **5.81 < v0.6's 5.85** on the identical split (target met)
- [x] Best val loss 1.4555 (v0.6: 1.4599) on the same val split
- [x] Target weak categories at their best-ever: creative 14.11, vietnamese
      7.01, reasoning 3.60, ml_ai 4.15, math 2.84
- [x] GGUF Q4_K_M + Q5_K_M exported as `distill-gpt55-v0.7` + smoke-tested
      (llama-server health ok, reply correct)
- [ ] `science` (6.02) and `philosophy` (6.22) did not recover → increasing rank
      was insufficient; the result supports the data-imbalance hypothesis
- [ ] LLM-as-judge still not run (judge API unreachable again)
- [ ] Not promoted to canonical/served — still above v0.5's 5.23 (different
      split, indicative only); v0.5 Q4_K_M stays served

**Lesson:** doubling adapter rank produced only marginal gains on the fixed
dataset. Rebalancing the catalogue is a better next experiment than adding rank.

### v0.8 — Rebalance + recovery + advanced distillation
- [ ] Beat v0.5's 5.23: rebalance the catalogue (top up science/philosophy and
      the strong categories too, not just append weak ones); retrain; re-evaluate
- [ ] Run LLM-as-judge (the metric v0.5/v0.6/v0.7 all skipped) once the judge API is up
- [ ] Teacher ensemble: add `cx/gpt-5.6-terra` as comparative teacher
- [ ] Fix or replace broken bnb 4-bit loading (torch stable?) → retrain QLoRA at
      seq 1024 (dataset stores full-length samples)
- [ ] Benchmark suite (GSM8K-mini, HumanEval-mini, MMLU subset)
- [ ] True distillation with logit matching (requires open-weight teacher)
- [ ] Qwen2.5-3B student when VRAM and disk headroom permit
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
| Low disk headroom | Docker + pagefile + artifact pressure | prune builds; check free space before train/export |

## Dependencies

| Package | Version | Note |
|---------|---------|------|
| PyTorch | 2.12 nightly cu128 | required by Python 3.14; source of load crashes |
| transformers | 5.14.1 | new core_model_loading path |
| trl / peft | 1.9.0 / 0.19.1 | prompt-completion loss masking |
| llama.cpp | b10107 | binaries/source installed outside the repo for GGUF |
| llama-cpp-python | 0.3.34 | serving runtime (CPU wheel works on 3.14) |

## Governance

- [Changelog](../CHANGELOG.md)
- [Contributing](../CONTRIBUTING.md)
- [Security policy](../SECURITY.md)
