# Code Standards

## File Naming
- Python scripts: **snake_case** for descriptive names (`gen_batch.py`, `train_student.py`, `format_dataset.py`)
- Config: `config.py` at project root
- Docs: Markdown in `docs/` dir

## Code Structure
- **≤ 200 LOC** per file — split when exceeding (CK rule)
- **kebab-case** for non-Python files per CK convention
- **No dead code** — remove test/temp scripts after use
- **Imports at top**, grouped: stdlib → third-party → local

## Configuration
- All settings in `config.py` — no hardcoded values in scripts
- Directory creation handled automatically on `import config`
- Paths relative to `PROJECT_DIR`, computed with `os.path.join`

## Error Handling
- API calls: retry 3× with exponential backoff
- Training: let crash for visibility, fix root cause
- File I/O: explicit encoding `utf-8`
- UTF-8 output wrapper for Windows console: `TextIOWrapper`

## Git Workflow
- **Conventional commits:** `feat:`, `fix:`, `docs:`, `test:`, `chore:`
- **No AI refs** in commit messages (except Co-authored-by trailer)
- **Focused commits:** one logical unit per commit
- **No secrets:** API keys belong in env vars, not code (current: config.py, to be migrated)

## Model Training
- **4-bit NF4** quantization mandatory for 6GB VRAM
- **float16** compute dtype (RTX 3060 không hỗ trợ bfloat16 amp)
- **No gradient checkpointing** on RTX 3060 (causes OOM with bitsandbytes)
- **adamw_8bit** optimizer for memory efficiency
- **SFTTrainer** from TRL with `processing_class=` (new API)

## Dataset
- **300+ samples** minimum for meaningful distillation
- **10 categories:** coding, reasoning, math, creative, science, ml_ai, vietnamese, business, health, philosophy
- **Qwen chat template:** `system/user/assistant`
- **Save after every generation** to prevent data loss
- **Stratified 90/10 train/test split** for honest evaluation (PPL on held-out, not in-sample)

## Evaluation
- **Perplexity on held-out test set** as primary metric (NOT in-sample)
- **ROUGE-L vs teacher** for generation quality
- **Per-category PPL** to identify weak domains
- **Loss tracking per step** for convergence monitoring
- **Temperature=0.7 generation** for diverse output (deterministic for benchmarks)
- **Qualitative test** with diverse prompts (code, reasoning, general knowledge)
