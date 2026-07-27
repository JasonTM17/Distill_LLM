# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

_Nothing yet._

## [0.6.0] - 2026-07-27

### Added
- Prompt catalogue expanded from 530 to 570 (IDs 531-570): +15 creative,
  +15 vietnamese, +10 reasoning — the three weakest categories in v0.5.
- v0.6 retrain on the expanded set: 570 teacher outputs generated, 568 accepted,
  stratified split 460 / 54 / 54, 3-epoch bf16 LoRA, best val loss 1.4599.
- Evaluation report v0.6 with per-category PPL at caps 512/1024/2048:
  [`plans/reports/evaluation-v0.6.md`](plans/reports/evaluation-v0.6.md).

### Changed
- None; serving artifacts unchanged (v0.5 GGUF still canonical — see Known issues).

### Known issues (this version)
- **Overall held-out PPL regressed: 5.85 @cap 2048 vs v0.5's 5.23.** Partly a
  harder, regenerated test split (51→54 samples, more high-PPL categories);
  partly real regressions in `science` (4.81→6.06) and `philosophy`
  (5.26→6.22) from expanding only the weak categories.
- Target weak categories did improve: `creative` 14.95→14.21, `vietnamese`
  8.35→7.02, `reasoning` 5.17→3.67, `ml_ai` 5.49→4.17.
- **LLM-as-judge still not run** (judge API unreachable) — the third metric
  remains missing across v0.5 and v0.6.
- **No v0.6 GGUF exported**; the api container still serves v0.5 Q4_K_M, which is
  the better overall model. v0.6 is a documented experiment, not a shipped model.

## [0.5.0] - 2026-07-26

### Added
- `src/distill` package: resilient teacher client (retryable/fatal error
  classification, backoff + jitter, output validation), resumable atomic
  generation, dataset quality pipeline (mojibake/dedup screening, stratified
  train/validation/test splits), training with validation + early stopping,
  bf16 merge, evaluation suite (held-out PPL, ROUGE-L vs teacher, optional
  LLM-as-judge), GGUF export wrapper.
- `services/api`: OpenAI-compatible FastAPI inference service over llama.cpp
  (streaming SSE, rate limiting, /healthz /readyz /metrics), tests, Dockerfile.
- `services/web`: Vite + React chat UI with client generated from the committed
  OpenAPI contract, streaming rendering, tests, Dockerfile (nginx).
- `docker-compose.yml`, GitHub Actions CI + Docker Hub publish, dependabot,
  pyproject, unit test suites, MIT license.
- Held-out evaluation v0.5: PPL 5.23 @cap 2048 (100% token coverage), 5.38
  @cap 512 (−22.4% vs v0.4's 6.93 on the matched protocol), ROUGE-L 0.1534:
  [`plans/reports/evaluation-v0.5.md`](plans/reports/evaluation-v0.5.md).
- GGUF Q4_K_M (0.92 GB) + Q5_K_M (1.05 GB) exported for CPU serving.

### Changed
- Training samples now use the exact Qwen2.5 chat template with
  `<|im_start|>`/`<|im_end|>` special tokens (v0.4 trained on a plain-text
  approximation, mismatching every standard inference path).
- Adapter merge now applies LoRA onto the bf16 base instead of the 4-bit
  dequantized base.

### Fixed
- Transient teacher API errors (quota/connection) no longer recorded as
  permanent failures — the v0.4 run lost 134/530 prompts to this.
- Vietnamese teacher outputs no longer mojibake-corrupted (UTF-8 handling +
  replacement-character validation at the client).

### Removed
- All 13 legacy root-level scripts (`gen_batch.py`, `train_student.py`,
  `format_dataset.py`, `evaluate*.py`, `chat.py`, root `config.py`, ...) —
  superseded by the tested `src/distill` package (`python -m distill.<module>`).

## [0.4.0] - 2026-07-25

### Added
- Honest held-out evaluation: 357 train / 38 stratified test samples,
  perplexity 6.93, ROUGE-L vs teacher; per-category breakdown.

## [0.3.0] - 2026-07-24

### Added
- First end-to-end pipeline: 300-sample generation, QLoRA training on RTX 3060
  6GB, merge, perplexity 4.70 (in-sample — superseded by v0.4's honest eval).
