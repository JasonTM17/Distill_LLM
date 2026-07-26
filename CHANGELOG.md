# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

### Changed
- Training samples now use the exact Qwen2.5 chat template with
  `<|im_start|>`/`<|im_end|>` special tokens (v0.4 trained on a plain-text
  approximation, mismatching every standard inference path).
- Adapter merge now applies LoRA onto the bf16 base instead of the 4-bit
  dequantized base.

### Fixed
- Transient teacher API errors (quota/connection) no longer recorded as
  permanent failures — the v0.4 run lost 127/530 prompts to this.
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
