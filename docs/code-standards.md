# Code Standards

## Layout

- Training pipeline lives in the **`src/distill` package** (`python -m distill.<module>`);
  root-level scripts are legacy and being retired.
- Services are standalone under `services/<name>/` — the api service must not
  import from `src/distill` (keeps torch out of its Docker image).
- Python: **snake_case** modules; ≤ 200 LOC per file — split when exceeding.
- Non-Python files: **kebab-case**; docs as Markdown in `docs/`.
- Imports at top, grouped: stdlib → third-party → local.

## Configuration

- Training tunables in `src/distill/config.py`, all env-overridable; secrets
  only via `.env` (gitignored) — never hardcoded, never printed.
- Service config in `services/api/app/config.py`, env-only (container-friendly).

## Error handling

- Teacher API calls: classify retryable vs fatal, exponential backoff + jitter,
  validate outputs (empty/short/U+FFFD rejected) — see `distill/teacher_client.py`.
- Dataset writes are atomic (temp file + rename), resumable across runs.
- File I/O: explicit `encoding="utf-8"`; console via `distill/logging_utils.py`.
- Training: let it crash for visibility; fix root cause (see phase-03 incident log).

## Testing & lint

- `python -m pytest tests/ -q` (package) and `services/api: python -m pytest tests/ -q`
  (fake runtime — no GPU/model needed); web: `pnpm test` + `pnpm build`.
- `ruff check src/ tests/ services/api/` must be clean (CI gate).
- API contract `docs/openapi.yaml` is generated from the FastAPI app; the web
  client is generated from it (`pnpm run generate-client`) — CI fails on drift.

## Git workflow

- **Conventional commits:** `feat:`, `fix:`, `docs:`, `test:`, `chore:`
- **No AI refs** in commit messages; sole author Nguyen Tien Son.
- Focused commits; no secrets, no private notes, no model artifacts.

## Model training (RTX 3060 6GB — hard-won constraints)

- **bf16 end-to-end** — the base checkpoint is bf16 and fp16 conversion at load
  crashes the current torch nightly on Windows (see phase-03 incident log).
- **Model loads go CPU-first, then `.to("cuda:0")`** — safetensors
  direct-to-GPU is broken against the current torch nightly.
- **LoRA on full bf16** (r16, q/k/v/o) with gradient checkpointing;
  the QLoRA 4-bit path is kept behind `LOAD_IN_4BIT` but bnb on-the-fly
  quantization crashes in this environment.
- **MAX_SEQ_LENGTH 512** — vocab-sized logits (152K) OOM 6GB at 1024.
- **adamw_8bit** optimizer; batch 1 × grad-accum 8; validation split with
  early stopping and best-checkpoint restore.

## Dataset

- 530 prompts / 10 categories; every sample validated at generation time.
- **Exact Qwen2.5 chat template** with `<|im_start|>`/`<|im_end|>` special
  tokens (a plain-text approximation trains a format no inference stack uses).
- Quality gate before splitting: mojibake, min-length, duplicate instructions.
- **Stratified 80/10/10 train/validation/test**, seed 42, splits disjoint by id.

## Evaluation

- Perplexity on the held-out test split (never in-sample), per category.
- ROUGE-L + token-F1 vs teacher reference; optional LLM-as-judge via 9Router.
- Reports versioned under `plans/reports/evaluation-<label>.md` with baseline
  comparison and explicit caveats.
