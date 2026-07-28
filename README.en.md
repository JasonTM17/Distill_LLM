# distill-gpt55

[![CI](https://github.com/JasonTM17/Distill_LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/JasonTM17/Distill_LLM/actions/workflows/ci.yml)
[![Containers](https://github.com/JasonTM17/Distill_LLM/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/JasonTM17/Distill_LLM/actions/workflows/docker-publish.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

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
- [Published containers](#published-containers)
- [Community and security](#community-and-security)

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
| `creative` PPL | 14.95 | 14.21 | **14.11** |
| `vietnamese` PPL | 8.35 | 7.02 | **7.01** |
| `reasoning` PPL | 5.17 | 3.67 | **3.60** |
| `science` PPL | 4.81 | 6.06 | 6.02 |
| `philosophy` PPL | 5.26 | 6.22 | 6.22 |
| LLM-as-judge | not run | not run | not run |
| GGUF export | Q4_K_M + Q5_K_M | not exported | Q4_K_M + Q5_K_M (smoke-tested) |

Perplexity always carries its truncation cap. Cap 2048 covers 100% of v0.5
tokens and 95.4% of v0.6/v0.7 tokens. v0.6 and v0.7 use the same 54-sample
split; comparisons with v0.5's different 51-sample split are indicative only.

![Perplexity and ROUGE-L by category](docs/assets/evaluation-by-category.png)

Full reports: [`v0.5`](plans/reports/evaluation-v0.5.md),
[`v0.6`](plans/reports/evaluation-v0.6.md), and
[`v0.7`](plans/reports/evaluation-v0.7.md).

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

Use the published Docker Hub images:

```bash
docker compose pull
docker compose up --no-build
```

Or build from source:

```bash
docker compose up --build
```

Images do not contain model weights. The GGUF file is still required at the
path above.

| Service | URL | Notes |
|---|---|---|
| Web chat | http://localhost:3000 | Starts only after the API is ready |
| API | http://localhost:8000 | OpenAI-compatible chat-completions subset |
| Readiness | http://localhost:8000/readyz | `503` while loading or after a load error |
| Liveness | http://localhost:8000/healthz | Process is alive |

Verify the API after the model has loaded:

```bash
curl http://localhost:8000/readyz
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}]}'
```

## Web chat and conversation history

The chat UI is a Vite + React SPA. It streams the assistant reply via SSE,
sanitizes assistant markdown with DOMPurify, and stores history **locally in the
browser** (`localStorage`) — no account, database, or server-side persistence.
It keeps at most 30 conversations and 100 non-empty, non-error messages per
conversation. A partial response retained when the user presses Stop is saved.

History is resent as context on the next request along with the system prompt. This is **not** long-term memory: the API defaults to a 4096-token context window and the web UI does not token-truncate history before a request. Keep conversations short if answers start to suffer from context overflow.

UX, accessibility and UI conventions: [`docs/design-guidelines.en.md`](docs/design-guidelines.en.md). Frontend guide: [`services/web/README.md`](services/web/README.md).

## Training pipeline

> Requires a matching Python environment, 9Router for generation/judge, and an RTX 3060 6GB for training per the v0.5 config.

```bash
pip install -e .[train,dev] trl
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
| [`docs/project-overview-pdr.en.md`](docs/project-overview-pdr.en.md) | Requirements, metrics, constraints and product risks |
| [`docs/system-architecture.en.md`](docs/system-architecture.en.md) | Offline/online architecture and chat data flow |
| [`docs/deployment-guide.en.md`](docs/deployment-guide.en.md) | Docker, local run, smoke test, resources and rollback |
| [`docs/code-standards.en.md`](docs/code-standards.en.md) | Code conventions, tests, contracts and training constraints |
| [`docs/design-guidelines.en.md`](docs/design-guidelines.en.md) | Design tokens, responsive/accessibility, state and history UX |
| [`docs/project-roadmap.en.md`](docs/project-roadmap.en.md) | What is done, known limits and the roadmap |
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

## Published containers

| Registry | API | Web |
|---|---|---|
| Docker Hub | [`nguyenson1710/distill-gpt55-api`](https://hub.docker.com/r/nguyenson1710/distill-gpt55-api) | [`nguyenson1710/distill-gpt55-web`](https://hub.docker.com/r/nguyenson1710/distill-gpt55-web) |
| GitHub Packages | [`ghcr.io/jasontm17/distill-gpt55-api`](https://github.com/users/JasonTM17/packages/container/package/distill-gpt55-api) | [`ghcr.io/jasontm17/distill-gpt55-web`](https://github.com/users/JasonTM17/packages/container/package/distill-gpt55-web) |

Each image has a `latest` tag and an immutable full commit-SHA tag.

## Known constraints

- **6GB VRAM:** v0.5/v0.6/v0.7 use bf16 LoRA + gradient checkpointing; training sequence length is capped at 512.
- **Windows GPU profile on Python 3.14:** requires a compatible CUDA PyTorch
  build; the verified setup uses a nightly build and loads models CPU-first
  before moving them to the GPU.
- **9Router:** only needed to generate the dataset / run the judge, not at serving time.
- **Local deployment:** the API serializes generation because the llama.cpp context is not thread-safe; it is not a multi-user scale-out service.

See the constraints, their reasons and remediation paths in
[`docs/project-roadmap.en.md`](docs/project-roadmap.en.md).

## Community and security

- Contributions: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Private vulnerability reports: [`SECURITY.md`](SECURITY.md)
- Release history: [`CHANGELOG.md`](CHANGELOG.md)
- License: [MIT](LICENSE)

---

<sup>This README is also available in [Tiếng Việt](README.md).</sup>
