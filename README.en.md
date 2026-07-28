# distill-gpt55

> Knowledge distillation from GPT-5.5-xhigh (via 9Router) into Qwen2.5-1.5B-Instruct, trained locally on an RTX 3060 6GB, then served through an OpenAI-compatible API and a streaming web chat UI.

> **Languages:** [Tiếng Việt](README.md) · **English** ([README.en.md](README.en.md))

![Streaming chat UI of the distilled model](docs/assets/chat-streaming.gif)

<sub>Real screenshot: the 1.5B student replying via SSE from llama.cpp running on CPU, around 1.4 tokens/second; the video is sped up.</sub>

## Table of contents

- [What does this project do?](#what-does-this-project-do)
- [Results v0.5, v0.6 & v0.7](#results-v05-v06--v07)
- [Architecture](#architecture)
- [Quick start with Docker](#quick-start-with-docker)
- [Web chat and conversation history](#web-chat-and-conversation-history)
- [Training pipeline](#training-pipeline)
- [Testing](#testing)
- [Detailed documentation](#detailed-documentation)

## What does this project do?

`distill-gpt55` has two separate parts:

| Part | Purpose | When you need to run it |
|---|---|---|
| **Offline training** | Generate teacher outputs, filter/split the dataset, fine-tune LoRA, evaluate, export GGUF | When recreating or improving the model |
| **Online serving** | Serve the GGUF with FastAPI + llama.cpp and a React chat UI | When you want to use the model |

You do **not** need 9Router, a GPU, or a training environment to run the exported GGUF. Serving currently runs on CPU inside the API container.

## Results v0.5, v0.6 & v0.7

v0.5 is the **canonical served model** (best overall). v0.6 expanded only the weak categories and regressed; v0.7 doubled LoRA capacity and beat v0.6 but still did not beat v0.5 — so v0.5 stays served.

| Metric | v0.5 (canonical) | v0.6 (experiment) | v0.7 (experiment) |
|---|---:|---:|---:|
| Overall held-out PPL @cap 2048 | **5.23** | 5.85 | 5.81 |
| Best validation loss | 1.4092 | 1.4599 | 1.4555 |
| Dataset (train/val/test) | 426/51/51 | 460/54/54 | 460/54/54 |
| Teacher outputs | 528 kept | 568 kept | 568 kept (same) |
| Chat template | exact Qwen `<|im_start|>` | same | same |

## Architecture

```text
OFFLINE — RTX 3060 6GB                         ONLINE — docker compose
prompts.json (570)                             ┌────────────┐    REST / SSE   ┌──────────────┐
  → generate_dataset (9Router)                 │ web        │ ───────────────▶│ api          │
  → dataset (quality gate + split)             │ React/Vite │                  │ FastAPI      │
  → train (bf16 LoRA + validation)             │ nginx      │◀─────────────────│ llama.cpp CPU│
  → merge → evaluate → export_gguf             └────────────┘                  └──────┬───────┘
                                                                              GGUF /models (RO)
```

Component detail, contracts and data flow: [`docs/system-architecture.md`](docs/system-architecture.md).

## Quick start with Docker

### Prerequisites

1. Docker Desktop with Compose v2.
2. A GGUF file at `checkpoints/gguf/distill-gpt55-v0.5-Q4_K_M.gguf`.

### Start

```bash
docker compose up --build
```

| Service | URL | Notes |
|---|---|---|
| Web chat | http://localhost:3000 | Starts only after the API is ready |
| API | http://localhost:8000 | OpenAI-compatible |
## Web chat and conversation history

The chat UI is a Vite + React SPA. It streams the assistant reply token-by-token via SSE, renders assistant markdown through DOMPurify (sanitized), and keeps conversation history **locally in the browser** (`localStorage`) — no account, no database, no server-side persistence. The storage module validates the JSON shape and fails closed on malformed data. It keeps at most 30 conversations and 100 completed messages per conversation.

History is resent as context on the next request along with the system prompt. This is **not** long-term memory: the API defaults to a 4096-token context window and the web UI does not token-truncate history before a request. Keep conversations short if answers start to suffer from context overflow.

UX, accessibility and UI conventions: [`docs/design-guidelines.md`](docs/design-guidelines.md). Frontend guide: [`services/web/README.md`](services/web/README.md).

## Training pipeline

> Requires a matching Python environment, 9Router for generation/judge, and an RTX 3060 6GB for training per the v0.5 config.

```bash
pip install -e .[train,dev]
set PYTHONPATH=src

python -m distill.download_student
python -m distill.generate_dataset
python -m distill.dataset
python -m distill.train
python -m distill.merge
python -m distill.evaluate --label v0.5
python -m distill.export_gguf
python -m distill.chat
```

Copy `.env.example` to `.env` and fill in your API key before calling the teacher. On comparable hardware, v0.5 uses `LOAD_IN_4BIT=false`, `GRADIENT_CHECKPOINTING=true`, `MAX_SEQ_LENGTH=512`: bitsandbytes 4-bit is currently broken on Python 3.14 + torch nightly. Do not commit `.env`.

## Testing

```bash
python -m pytest tests/ -q                    # training pipeline
cd services/api && python -m pytest tests/ -q # API, fake runtime, no model/GPU needed
cd services/web && pnpm test && pnpm build    # UI tests + typecheck + production build
ruff check src/ tests/ services/api/
```

The canonical API contract is [`docs/openapi.yaml`](docs/openapi.yaml). After changing the API, regenerate the frontend types and check the diff:

```bash
cd services/web
pnpm run generate-client
pnpm build
```

## Detailed documentation

| Document | Contents |
|---|---|
| [`docs/project-overview-pdr.md`](docs/project-overview-pdr.md) | Requirements, metrics, constraints and product risks |
| [`docs/system-architecture.md`](docs/system-architecture.md) | Offline/online architecture and chat data flow |
| [`docs/deployment-guide.md`](docs/deployment-guide.md) | Docker, local run, smoke test, resources and rollback |
| [`docs/code-standards.md`](docs/code-standards.md) | Code conventions, tests, contracts and training constraints |
| [`docs/design-guidelines.md`](docs/design-guidelines.md) | Design tokens, responsive/accessibility, state and history UX |
| [`docs/project-roadmap.md`](docs/project-roadmap.md) | What is done, known limits and the roadmap |
| [`services/api/README.md`](services/api/README.md) | API endpoints, config and runbook |
| [`services/web/README.md`](services/web/README.md) | Web setup, history behaviour and frontend troubleshooting |

## Repo layout

```text
src/distill/       Training pipeline package
services/api/      FastAPI + llama.cpp inference service
services/web/      React chat UI
data/              Prompts, teacher outputs, processed splits
checkpoints/       Adapter, merged model, GGUF artifacts (gitignored)
docs/              Architecture, deployment, standards, roadmap, OpenAPI
plans/             ClaudeKit plans and evaluation reports
```

## Known constraints

- **6GB VRAM:** v0.5/v0.6/v0.7 use bf16 LoRA + gradient checkpointing; training sequence length is capped at 512.
- **Python 3.14:** requires torch nightly CUDA; load models CPU-first then move to GPU to avoid a known crash.
- **9Router:** only needed to generate the dataset / run the judge, not at serving time.
- **Local deployment:** the API serializes generation because the llama.cpp context is not thread-safe; it is not a multi-user scale-out service.

See the constraints, their reasons and remediation paths in [`docs/project-roadmap.md`](docs/project-roadmap.md).

---

<sup>This README is also available in [Tiếng Việt](README.md).</sup>

| Readiness | http://localhost:8000/readyz | `503` while the GGUF is loading |
| Liveness | http://localhost:8000/healthz | Process is alive |

Verify the API once the model has loaded:

```bash
curl http://localhost:8000/readyz
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}]}'
```

The web UI polls `/readyz` every 5s and disables sending until it succeeds. A temporary `model loading` badge is normal right after a restart.

| `creative` PPL | 14.95 | 14.21 | **14.11** |
| `vietnamese` PPL | 8.35 | 7.02 | **7.01** |
| `reasoning` PPL | 5.17 | 3.67 | **3.60** |
| `science` PPL | 4.81 | 6.06 | 6.02 |
| `philosophy` PPL | 5.26 | 6.22 | 6.22 |
| LLM-as-judge | not run | not run | not run |
| GGUF export | Q4_K_M + Q5_K_M | not exported | Q4_K_M + Q5_K_M (smoke-tested) |

Perplexity always carries its truncation cap. The test split's median is 525 tokens: cap 512 scores only ~70% of tokens, cap 2048 scores ~95-100%. Only compare figures measured at the **same cap**. v0.6/v0.7 share the same 54-sample test split (apples-to-apples); v0.5 used a different 51-sample split (cross-version comparisons are indicative, never differenced).

![Perplexity and ROUGE-L by category](docs/assets/evaluation-by-category.png)

Reproduce a headline: `python -m distill.evaluate --label v0.5`. Full reports:
[`plans/reports/evaluation-v0.5.md`](plans/reports/evaluation-v0.5.md),
[`evaluation-v0.6.md`](plans/reports/evaluation-v0.6.md),
[`evaluation-v0.7.md`](plans/reports/evaluation-v0.7.md).
