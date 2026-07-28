# Codebase Summary

> **Language:** English translation. Canonical Vietnamese: [`codebase-summary.md`](codebase-summary.md)

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
it runs by default and is skippable with `--skip-smoke`.
