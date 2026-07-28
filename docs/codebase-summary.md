# Codebase Summary

Orientation for a maintainer who has never opened this repo. It covers what each
module owns, the order the offline pipeline runs in, how the serving path differs
from it, where the FE/BE contract lives, and which boundaries not to cross.

For the component diagram and data flow see [system-architecture.md](./system-architecture.md).
For conventions and hard-won machine constraints see [code-standards.md](./code-standards.md).
For running the stack see [deployment-guide.md](./deployment-guide.md).

## The one thing to understand first

There are **two planes that never import each other**:

| Plane | Lives in | Runs on | Produces / consumes |
|---|---|---|---|
| Offline pipeline | `src/distill` | GPU workstation, Python 3.14 | Produces a GGUF file |
| Online serving | `services/api`, `services/web` | Containers, CPU | Consumes that GGUF file |

The only thing crossing the boundary is the quantized model file in
`checkpoints/gguf/`, mounted read-only into the api container. The serving code
has no torch, no transformers, no dataset code — deliberately, so the api image
stays small. Breaking that separation is the single most expensive mistake
available in this repo.

## Repo map

| Path | Files | ≈LOC | Owns |
|---|---|---|---|
| `src/distill/` | 15 | 1,539 | The whole offline pipeline, as an importable package |
| `services/api/app/` | 9 | 418 | FastAPI service wrapping llama.cpp |
| `services/web/src/api/` | 4 | 474 | Generated contract types + typed client + SSE parser |
| `services/web/src/` | 4 | 218 | App shell, entrypoint, stylesheet |
| `services/web/src/components/` | 3 | 146 | Chat composer, message bubble, their tests |
| `services/web/src/hooks/` | 1 | 57 | `use-chat` — conversation state + streaming |
| `tests/` | 4 | 396 | Pipeline unit tests (49 cases) |
| `services/api/tests/` | 3 | 135 | Route tests against a fake runtime (12 cases) |
| `docs/openapi.yaml` | 1 | 255 | The canonical API contract |
| `.github/workflows/` | 2 | 74 | CI and image publish |

Roughly 3,800 LOC of project code. Everything is small on purpose; if a module
starts sprawling, that is a signal, not a milestone.

## Offline pipeline — stages in order

Every stage is a module with a `main()` and is runnable as `python -m distill.<module>`.
Each reads files written by the previous stage, so stages are resumable and
independently re-runnable. Paths all come from `config.py`; nothing hardcodes a
location.

### 0. `download_student` — fetch the base model

Pulls `Qwen/Qwen2.5-1.5B-Instruct` into the local cache directory named by
`STUDENT_MODEL_ID` (default `D:/models/qwen15-1.5b`, outside the repo). Run once.

### 1. `generate_dataset` — teacher outputs

Reads `data/prompts.json`, calls the teacher through `teacher_client`, writes
`data/raw/teacher_outputs.json`.

The design exists because of a specific past failure: an earlier run recorded
transient API errors as permanent and silently lost a large fraction of the
prompt set. So this stage is built around never losing work —

- writes are **atomic** (temp file + replace), so a kill mid-write cannot truncate the JSON
- failures are recorded as records, not dropped, and are re-attempted on later runs
- `--retry-failed`, `--limit`, and `--categories` let you target exactly what is missing

`teacher_client.py` is the piece that makes this safe. It classifies exceptions
into retryable vs fatal, applies exponential backoff with jitter, and validates
that a response actually contains usable text before accepting it. Empty, too
short, and U+FFFD-bearing (mojibake) responses are rejected at the client, not
downstream.

### 2. `dataset` — quality gate and splits

Reads the raw teacher outputs, writes `data/processed/dataset_{train,validation,test}.json`
plus `dataset_stats.json`.

Three responsibilities, in this order:

1. **Screen** — drop failures, too-short outputs, mojibake, and duplicate instructions
   (case-insensitive, hashed in `_instruction_key`).
2. **Format** — render each sample with the exact Qwen2.5 chat template including
   `<|im_start|>` / `<|im_end|>` special tokens, via `qwen_chat_text`. This matters:
   training on text that lacks the special tokens teaches the model a format that no
   standard inference path (`apply_chat_template`, llama.cpp) will ever produce.
3. **Split** — stratified 80/10/10 by category with a fixed seed (`SPLIT_SEED`, default 42),
   so splits are reproducible. Categories with fewer than `MIN_CATEGORY_FOR_SPLIT` (5)
   samples stay wholly in train rather than producing a one-sample test stratum.

`dataset_stats.json` is the audit trail — raw count, accepted count, rejection
reasons, split sizes, and per-category counts. Read it before trusting any claim
about the dataset.

### 3. `train` — LoRA fine-tune with a validation loop

Reads the train and validation splits, writes `checkpoints/adapter/`.

- LoRA on attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), r16 / alpha32.
- trl `SFTTrainer` with prompt/completion format so loss is computed on completions
  only when `TRAIN_ON_COMPLETIONS_ONLY` is set.
- Evaluates on the validation split, early-stops on `EARLY_STOPPING_PATIENCE`, and
  restores the best checkpoint.
- The previous adapter is archived, not silently overwritten (`archive_previous_adapter`).

Two loading paths exist, selected by `LOAD_IN_4BIT`: a bitsandbytes NF4 4-bit path
and a plain bf16 path. Both request bf16 dtype. The v0.5 run used the bf16 path —
`logs/train.log` records `train=426 validation=51` and a bf16 CPU-then-CUDA load.

### 4. `merge` — fold the adapter into standalone weights

Reads the base model and `checkpoints/adapter/`, writes `checkpoints/merged/`.

Merges into a full-precision base rather than a quantized one, so merged quality is
bounded by the adapter rather than by quantization rounding. Output feeds evaluation,
GGUF export, and terminal chat.

### 5. `evaluate` — held-out measurement

Reads `checkpoints/merged/` and the test split, writes `checkpoints/evaluation_results.json`
and a markdown report at `plans/reports/evaluation-<label>.md`.

The report filename is derived from `--label` (default `v0.5`), so the v0.5 run
produces `evaluation-v0.5.md`. Measures perplexity on never-seen samples, ROUGE-L
F1 and token-F1 against the teacher reference, per category and overall, plus an
optional LLM-as-judge pass (`--judge`) that skips gracefully when the judge API is
unreachable. `--baseline-ppl` accepts a prior number for side-by-side comparison.

`eval_metrics.py` deliberately holds the pure metric math — LCS, ROUGE-L F1,
token-F1 — with no torch or transformers imports, so it is unit-testable anywhere
and fast to test.

### 6. `export_gguf` — the artifact the serving stack consumes

Reads `checkpoints/merged/`, writes `checkpoints/gguf/distill-gpt55-v0.5-{Q4_K_M,Q5_K_M}.gguf`.

Converts to an intermediate f16 GGUF, quantizes each requested type, then deletes
the intermediate to protect free disk space. Depends on llama.cpp tooling that lives
**outside the repo**, located via `LLAMACPP_BIN` and `LLAMACPP_SRC`. A `smoke_test`
helper runs one short CPU generation per artifact to prove it loads and answers;
it runs by default and is skippable with `--skip-smoke`.

### Interactive check: `chat`

`python -m distill.chat` loads the merged model for terminal chat, one-shot with
`--prompt`. This is the pre-container sanity check and does not touch the serving path.

## Shared offline infrastructure

Three modules exist purely to encode environment constraints. Route through them
rather than reimplementing:

| Module | Why it exists |
|---|---|
| `model_loading.py` | Every causal-LM load goes through here: bf16 dtype, CPU-first then move to CUDA, pre-touch first. The direct-to-device and fp16-at-load paths crash on this torch/safetensors combination. |
| `safetensors_pretouch.py` | Sequentially reads shards into the OS cache before mmap, so loads do not fault against a constrained Windows pagefile. Call `install()` before loading weights. |
| `logging_utils.py` | Forces UTF-8 on stdout/stderr. The default Windows console codepage mangles UTF-8, which is how mojibake got into a dataset once already. |

`config.py` is the single tunable surface: every path, teacher setting, split ratio,
LoRA hyperparameter, and eval knob, each overridable by environment variable and
loaded from a local `.env` when present. Secrets are read from the environment and
never hardcoded; `summary()` returns a redacted snapshot for logging that reports
`api_key_set` as a boolean rather than the key.

## Online serving path

### Request lifecycle

A streamed chat request traverses:

```
browser → use-chat hook → streamChatCompletion (client.ts)
        → POST /v1/chat/completions  (stream: true)
        → rate limiter → schema validation → LlamaCppRuntime.stream
        → SSE chunks ← SseBuffer ← onToken → React state → MessageBubble
```

### `services/api` — FastAPI over llama.cpp

| Module | Owns |
|---|---|
| `main.py` | App factory. Middleware assigns/echoes `x-request-id` and records metrics. Accepts injected `runtime` and `rate_limiter` — this is what makes the route tests fast. |
| `model_runtime.py` | `LlamaCppRuntime`: background model load, `ready` flag, and a single lock serializing every call. llama.cpp contexts are not safe for concurrent use. |
| `routes_chat.py` | `POST /v1/chat/completions`, streaming and non-streaming. |
| `routes_ops.py` | `/healthz` (always ok), `/readyz` (503 until the GGUF is loaded, surfacing load errors), `/metrics`. |
| `schemas.py` | Pydantic request/response models — the OpenAI-compatible subset. |
| `rate_limit.py` | In-process sliding-window limiter, keyed per client host. |
| `metrics.py` | A private `CollectorRegistry` plus request, latency, and generated-token metrics. |
| `config.py` | Env-only configuration. |

Behaviour worth knowing before changing anything:

- **Readiness is a real state, not decoration.** The model loads in a background
  thread; `/readyz` returns 503 while loading and reports the load error if it failed.
  The web UI polls it and disables the composer until ready.
- **Generation is serialized.** Non-streaming calls hop to a threadpool so the event
  loop is not blocked; streaming iterates a sync generator that Starlette runs in a
  worker thread. Both take the same lock.
- **The stream always terminates.** `_sse_stream` emits `data: [DONE]` from a `finally`
  block, so aborts and mid-stream errors still close the stream cleanly. Errors are
  delivered as an SSE payload rather than by tearing the connection down.
- **The rate limiter is in-process on purpose.** One instance serving one local model,
  where generations serialize on the model lock anyway. Horizontal scale would need a
  shared store, and a model-server change first.

### `services/web` — single-view chat UI

`App.tsx` composes one view: header with a readiness badge, message list, composer.
`use-chat.ts` owns conversation state, assembles history, and drives the stream with
an `AbortController` for stop. `components/` holds the composer and the message bubble.
See [design-guidelines.md](./design-guidelines.md) for UI conventions and states.

## The FE/BE contract

`docs/openapi.yaml` is canonical. It describes `/healthz`, `/readyz`, `/metrics`,
and `/v1/chat/completions`.

**The web client's types are generated from it, not hand-written.**

```
pnpm run generate-client   # openapi-typescript ../../docs/openapi.yaml -o src/api/schema.d.ts
```

`src/api/schema.d.ts` carries a do-not-edit header. `client.ts` imports its
request/response types from it rather than declaring shapes locally. CI regenerates
the client and runs `git diff --exit-code src/api/schema.d.ts`, so a contract change
that is not reflected in the committed client fails the build.

Changing the API means: update the contract, regenerate, commit both.

`sse.ts` is the one piece of transport code that is hand-written and must stay that
way — `SseBuffer` buffers until it sees a blank-line terminator so an event split
across two network chunks parses correctly, and `decodeChatEvent` handles the
`[DONE]` sentinel and raises on error payloads.

## Tests

| Suite | Runner | Cases | Covers |
|---|---|---|---|
| `tests/` | pytest | 49 | Pipeline logic |
| `services/api/tests/` | pytest | 12 | Routes against a fake runtime |
| `services/web/src/**/*.test.*` | vitest | 12 | SSE parsing, message rendering |

What the pipeline tests actually pin down:

- `test_teacher_client.py` — error classification (transient vs permanent vs unknown),
  backoff bounds and cap, output validation, and the retry loop end to end including
  exhaustion and the truncated-response flag.
- `test_generate_dataset.py` — which prompts are selected as pending, atomic write and
  reload round-trip, corrupt and missing files returning empty rather than raising,
  and resume-with-retry behaviour.
- `test_dataset.py` — that the template carries Qwen special tokens, every screening
  rule, split determinism and disjointness, per-category proportions, and the tiny-category
  rule.
- `test_eval_metrics.py` — metric math against known values, including empty and
  disjoint inputs, plus report aggregation and rendering.

The api tests inject a fake runtime through `create_app`, so they never load a model:
happy path, defaults, validation rejections, 503 while loading, SSE chunk shape, and
the 429 path. Web tests cover chunk-boundary SSE parsing and that assistant markdown
is sanitized.

Everything runs without a GPU, without a model, and without network. Keep it that way.

## Boundaries not to cross

1. **`services/api` must never import `src/distill`.** Its `config.py` says so explicitly.
   The image would grow by gigabytes and the CPU serving story would collapse.
2. **Do not hand-edit `services/web/src/api/schema.d.ts`.** It is generated; CI will
   catch you. Edit `docs/openapi.yaml` and regenerate.
3. **Do not load model weights outside `model_loading.py`.** The dtype and device
   sequence are load-bearing workarounds, not preferences.
4. **Do not bake the GGUF into the api image.** It is mounted read-only at `/models`
   by compose. Baking it makes the image enormous and pins it to one model version.
5. **Do not print dataset content before `configure_console()`.** That is how mojibake
   enters a corpus.
6. **Do not add torch/transformers to `eval_metrics.py`.** Its portability is why the
   metric tests are fast.
7. **Treat `checkpoints/` and `data/raw/` as build outputs.** They are gitignored and
   reproducible from the pipeline. Never stage them.

## Containers and CI

Both images are multi-stage and run non-root. The api builds llama-cpp-python wheels
in a builder stage and installs them into a slim runtime with no compilers, exposing
8000 with a `/healthz` HEALTHCHECK. The web builds a Vite bundle and serves it from
nginx-unprivileged on 3000. Both carry OCI source/licence/revision labels.

`docker-compose.yml` boots both, mounts `./checkpoints/gguf` read-only at `/models`,
and gates web startup on the api being healthy.

CI runs two jobs: Python (ruff over `src/`, `tests/`, `services/api/`, then both pytest
suites) and web (install, contract-sync check, vitest, build). A separate workflow
publishes both images on pushes to master.

## Where the artifacts live

| Artifact | Path | In git? |
|---|---|---|
| Prompts | `data/prompts.json` | yes |
| Raw teacher outputs | `data/raw/teacher_outputs.json` | no |
| Splits + stats | `data/processed/` | yes |
| LoRA adapter | `checkpoints/adapter/` | no |
| Merged weights | `checkpoints/merged/` | no |
| GGUF quantizations | `checkpoints/gguf/` | no |
| Eval report | `plans/reports/evaluation-<label>.md` | yes |
| Run logs | `logs/` | no |

Base model weights live outside the repo entirely, at `STUDENT_MODEL_ID`.
