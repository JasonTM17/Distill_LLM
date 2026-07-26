# distill-gpt55 — Distill GPT-5.5-xhigh → Qwen2.5-1.5B

**Knowledge distillation** from GPT-5.5-xhigh (via 9Router API) into
Qwen2.5-1.5B-Instruct, trained locally on an RTX 3060 6GB — then served as a
containerized OpenAI-compatible API + web chat UI.

## Results

### v0.5 (current) — full 530-prompt dataset

| Metric | v0.4 | **v0.5** |
|--------|------|----------|
| Held-out perplexity (cap 2048, 100% of test tokens) | not measured at this cap | **5.23** |
| Held-out perplexity (cap 512, v0.4 protocol match) | 6.93 | **5.38** (−22.4%) |
| Validation loss (best) | — (no val split) | **1.409** |
| Dataset (train/val/test) | 357 / 0 / 38 | **426 / 51 / 51** |
| Teacher outputs | 396/530 (philosophy+health = 0) | **530/530 generated, 528 kept** |
| Chat template | plain-text approximation | exact Qwen `<|im_start|>` template |

Perplexity is reported with its truncation cap because the two are inseparable:
the test split's median sample is 525 tokens, so a 512 cap scores only 70% of
held-out tokens, 1024 scores 92%, and 2048 scores 100%. **Only same-cap numbers
may be compared.** The −22.4% figure is 512-vs-512; v0.4's own truncation rate
is unmeasurable because its 38-sample split was regenerated for v0.5.

Reproduce the headline with `MAX_SEQ_LENGTH=2048 python -m distill.evaluate
--label v0.5` as a per-process env override only — `MAX_SEQ_LENGTH` is shared
with training, and persisting 2048 in `.env` reconfigures training to a length
that OOMs.

Full report: [`plans/reports/evaluation-v0.5.md`](plans/reports/evaluation-v0.5.md)

## Architecture

```
OFFLINE (RTX 3060 6GB)                        ONLINE (docker compose)
prompts.json (530)                            ┌─────────┐      ┌─────────┐
  → distill.generate_dataset (9Router)        │   web   │─────▶│   api   │
  → distill.dataset (screen + split)          │  React  │ SSE  │ FastAPI │
  → distill.train (bf16 LoRA + validation)    │  nginx  │      │llama.cpp│
  → distill.merge → distill.evaluate          └─────────┘      └────┬────┘
  → distill.export_gguf ──────────────────────── GGUF volume ───────┘
```

Details: [`docs/system-architecture.md`](docs/system-architecture.md)

## Quick start — serving

```bash
docker compose up --build
# web: http://localhost:3000   api: http://localhost:8000 (OpenAI-compatible)
```

Needs `checkpoints/gguf/distill-gpt55-v0.5-Q4_K_M.gguf` (see pipeline below or
[`docs/deployment-guide.md`](docs/deployment-guide.md)).

## Quick start — training pipeline

```bash
pip install -e .[train,dev]           # or: set PYTHONPATH=src
set PYTHONPATH=src

python -m distill.download_student    # base model -> D:/models/qwen15-1.5b
python -m distill.generate_dataset    # teacher outputs (resumable, retries failures)
python -m distill.dataset             # quality gate + 80/10/10 stratified splits
python -m distill.train               # LoRA + validation + early stopping
python -m distill.merge               # adapter -> merged bf16 model
python -m distill.evaluate --label v0.5   # cap decides the number — see Results
python -m distill.export_gguf         # Q4_K_M + Q5_K_M via llama.cpp
python -m distill.chat                # interactive smoke test
```

Environment knobs live in [`.env.example`](.env.example). On this machine
training runs with `LOAD_IN_4BIT=false GRADIENT_CHECKPOINTING=true MAX_SEQ_LENGTH=512`
(see `docs/code-standards.md` for the hard-won torch-nightly constraints).

## Hyperparameters (v0.5)

| Param | Value | Note |
|-------|-------|------|
| Method | LoRA r=16 α=32 on q/k/v/o | bf16 base (bnb 4-bit broken in this env) |
| Batch | 1 × grad-accum 8 | 6GB VRAM |
| LR / schedule | 2e-4 cosine, 3 epochs | early stopping on val loss |
| Max seq len | 512 | 152K-vocab logits OOM at 1024 |
| Eval | every 25 steps on 51-sample val split | best checkpoint restored |

## Tests

```bash
python -m pytest tests/ -q                    # pipeline package (49 tests)
cd services/api && python -m pytest tests/ -q # API (fake runtime, no model)
cd services/web && pnpm test && pnpm build    # UI + type check
ruff check src/ tests/ services/api/
```

## Repo layout

```
src/distill/       training pipeline package
services/api/      FastAPI + llama.cpp inference service (own README)
services/web/      React chat UI (own README)
data/              prompts + raw teacher outputs + processed splits
checkpoints/       adapter / merged / gguf artifacts (gitignored)
docs/              architecture, standards, roadmap, deployment, openapi.yaml
plans/             ClaudeKit plans + evaluation reports
```

## Constraints

- **6GB VRAM** → LoRA + gradient checkpointing (QLoRA when bnb works again)
- **Python 3.14** → torch nightly cu128; several loader bugs worked around
  (bf16-only, CPU-first loads — see `docs/code-standards.md`)
- **9Router** at `localhost:20128` needed only for generation/judge, never serving
