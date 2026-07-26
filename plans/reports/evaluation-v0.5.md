# Evaluation v0.5 — distilled Qwen2.5-1.5B

**Date:** 2026-07-26 (generation pass finished 2026-07-27 05:50 local)
**Model:** `D:\distill-gpt55\checkpoints\merged` (bf16 merged LoRA, 1.5B, vocab 151936)
**Test split:** 51 held-out samples, 10 categories, never trained on
**Teacher:** `cx/gpt-5.5-xhigh` references from `data/processed/dataset_test.json`
**Evaluator:** held-out perplexity (3 truncation caps) + ROUGE-L / token-F1 vs teacher

## Overall

| Metric | Value |
|---|---|
| Perplexity (held-out, cap 2048, 100% token coverage) | **5.23** |
| Avg loss | 1.6546 |
| ROUGE-L F1 vs teacher | **0.1534** |
| Token-F1 vs teacher | 0.3148 |
| LLM-as-judge | **not run** — see Named gaps |

The headline PPL carries its truncation cap because the two are inseparable.
Any comparison against another number must be same-cap. The v0.4 baseline of
6.93 was measured at cap 512, so it may only be compared against this run's
512 figure (5.38), never against 5.23.

## Measurement conditions

The test split's median sample is 525 tokens, so the truncation cap decides how
much of the held-out set is actually scored:

| Cap | PPL | Avg loss | Samples truncated | Tokens scored | Coverage |
|---|---|---|---|---|---|
| 512 | 5.38 | 1.6835 | 26 / 51 | 21,335 / 30,301 | 70.4% |
| 1024 (config default) | 5.30 | 1.6671 | 5 / 51 | 27,844 / 30,301 | 91.9% |
| **2048 (canonical)** | **5.23** | 1.6546 | **0 / 51** | **30,301 / 30,301** | **100%** |

Token lengths of the test split: min 113, median 525, mean 594, max 1718.

2048 is canonical because it is the only cap that scores the entire held-out
set. A metric named "held-out perplexity" that silently discards 30% of the
held-out tokens is not a weaker measurement of the right thing — it measures a
smaller corpus under the right name.

**No OOM at 2048.** Forward-only inference at full length fits on the 6 GB
RTX 3060 with the 3 GB bf16 model resident. The roadmap's "152K-vocab logits
OOM at 1024" (project-roadmap.md line 50) is correctly scoped to `bf16 LoRA`
training and remains accurate — but it should not be read as a box-wide limit,
because this box evaluates fine at 2048. Clarification for docs, not a correction.

## Resolution of the README 5.30 claim

**The 5.30 was real, reproducible, and produced by the documented command — but
its measurement condition was never recorded.**

`python -m distill.evaluate --label v0.5 --baseline-ppl 6.93` runs at
`config.MAX_SEQ_LENGTH`, which defaults to 1024. Cap 1024 yields exactly 5.30.
The published delta cross-checks to the decimal: (6.93 − 5.30) / 6.93 = 23.5%,
matching the README's original "−23.5%".

So this was never a wrong number. It was an unlabelled one, and unlabelled it
under-measured the held-out set by 8% of its tokens. The finding is therefore
"record the condition and publish the full-coverage figure", not "correct a
fabrication". Both figures now appear in the README results table with their
caps attached.

## Held-out perplexity by category

Per-category PPL at all three caps, so truncation effects are visible rather
than assumed:

| Category | Samples | PPL @2048 | PPL @1024 | PPL @512 | Reading |
|---|---|---|---|---|---|
| coding | 7 | **2.53** | 2.64 | 3.04 | best; truncation was flattering it downward |
| math | 5 | 2.98 | 2.98 | 2.98 | cap-invariant (no sample exceeds 512) |
| health | 5 | 4.41 | 4.41 | 4.46 | new category in v0.5 |
| science | 5 | 4.81 | 4.81 | 4.79 | stable |
| reasoning | 4 | 5.17 | 5.17 | 5.17 | cap-invariant |
| philosophy | 5 | 5.26 | 5.40 | 4.90 | **non-monotonic** — see callouts |
| ml_ai | 5 | 5.49 | 5.49 | 5.52 | stable |
| business | 5 | 5.76 | 5.76 | 5.70 | stable |
| vietnamese | 5 | **8.35** | 8.79 | 9.20 | improves with context — see callouts |
| creative | 5 | **14.95** | 14.95 | 15.04 | **worst, and not a cap artifact** |

**Overall @2048: 5.23.**

### Named callout — creative (14.95) is the headline's biggest drag

`creative` is roughly 3x the overall PPL and 6x `coding`. It is essentially
cap-invariant (15.04 / 14.95 / 14.95), which rules out truncation as the cause.
This is genuine model weakness on open-ended generative prompts, and it is the
single largest contributor pulling the headline up. A 5.23 that hides a 14.95
is the same class of problem as a 5.30 that hides its cap, so it is stated here
rather than left as a table row.

Its ROUGE-L (0.1073) is also second-worst, so both metrics agree — this is not
a metric artifact.

### Named callout — vietnamese (8.35) was partly a measurement problem

`vietnamese` is second-worst but behaves differently: 9.20 → 8.79 → 8.35 as the
cap rises. All 5 vietnamese samples were truncated at 512. So a meaningful part
of its apparent weakness at 512 was measurement, not model. It is still the
second-weakest category at full coverage, but the honest figure is 8.35, not 9.20.

### Named callout — philosophy is non-monotonic (4.90 → 5.40 → 5.26)

`philosophy` gets **worse** when shown more context (512 → 1024) and then partly
recovers (1024 → 2048). This is the strongest single piece of evidence that
truncation did not bias results in one direction. Monotonic degradation could be
dismissed as "512 is just a noisier 2048"; a metric that rises and then falls
cannot be. Combined with `coding` (which truncation made look *worse*, 3.04 →
2.53) and `creative` (unaffected), truncation distorted per-category rankings in
three different directions.

Consequence: per-category PPL measured at 512 is not comparable **across
categories**, because the categories with the longest teacher answers were cut
hardest — health 5/5, vietnamese 5/5, philosophy 4/5 truncated, versus math 0/5
and reasoning 0/4. Those long-answer categories are precisely the ones v0.5 was
built to add.

## ROUGE-L / token-F1 vs teacher

Generated with the exact Qwen chat template, `temperature=0.7`, `top_p=0.9`,
`max_new_tokens=512`. ROUGE-L is low in absolute terms for every distillation
run of this shape because sampled output diverges in wording from a single
reference; the useful signal is relative, across categories and versions.

| Category | Samples | ROUGE-L | Token-F1 |
|---|---|---|---|
| math | 5 | **0.2315** | 0.3400 |
| reasoning | 4 | 0.1983 | 0.3289 |
| coding | 7 | 0.1808 | 0.3096 |
| science | 5 | 0.1701 | **0.3782** |
| health | 5 | 0.1419 | 0.3533 |
| business | 5 | 0.1338 | 0.3202 |
| ml_ai | 5 | 0.1273 | 0.3175 |
| vietnamese | 5 | 0.1257 | 0.2794 |
| philosophy | 5 | 0.1153 | 0.2747 |
| creative | 5 | **0.1073** | 0.2511 |

**Overall ROUGE-L 0.1534, token-F1 0.3148.**

Answer length: student mean 266 words / median 271; teacher mean 374 / median
359. 31 of 51 student answers are shorter than their reference, and 9 answers
land near the 512-new-token ceiling, so part of the ROUGE recall gap is the
generation cap rather than content quality. No empty answers.

### Best

- `[coding] rouge=0.364 id=1` — "Write a Python function to check if a string is a palindrome."
- `[reasoning] rouge=0.340 id=509` — Vietnamese chicken-and-a-half puzzle (best non-English result in the run)
- `[math] rouge=0.316 id=103` — "What is the Pythagorean theorem? Prove it geometrically."
- `[math] rouge=0.288 id=148` — "Find the maximum of f(x) = -x^2 + 4x + 5."

### Worst

- `[ml_ai] rouge=0.069 id=299` — "Difference between langchain and llamaindex?" — recent-tooling knowledge the student does not hold
- `[creative] rouge=0.073 id=187` — "Describe what it feels like to fly, using no words related to birds or planes" — constrained creative writing
- `[vietnamese] rouge=0.089 id=315` — Vietnamese poetry analysis (Hàn Mặc Tử)
- `[creative] rouge=0.093 id=162` — fable with a moral

The two worst creative items confirm the PPL callout from a second, independent
metric.

## v0.4 vs v0.5

| Aspect | v0.4 | v0.5 |
|---|---|---|
| Held-out PPL @512 (protocol match) | 6.93 | **5.38** (−22.4%) |
| Held-out PPL @2048 (full coverage) | not measured at this cap | **5.23** |
| ROUGE-L vs teacher | 0.1337 (38 samples) | **0.1534** (51 samples), +14.7% |
| Test split | 38 samples, 8 categories | **51 samples, 10 categories** |
| Categories with zero data | philosophy, health | **none** |
| Validation split | none | **51 samples, early stopping** |
| Best val loss | n/a | **1.4092** (checkpoint-125) |
| Chat template | plain-text approximation | exact Qwen `<\|im_start\|>` |
| Model dtype at eval | fp16 | bf16 (fp16 load crashes this torch nightly) |

Per-category ROUGE-L, v0.4 → v0.5:

| Category | v0.4 | v0.5 | Change |
|---|---|---|---|
| math | 0.2042 | 0.2315 | +0.027 |
| science | 0.1594 | 0.1701 | +0.011 |
| coding | 0.1445 | 0.1808 | +0.036 |
| ml_ai | 0.1356 | 0.1273 | **−0.008** |
| business | 0.1210 | 0.1338 | +0.013 |
| reasoning | 0.1128 | 0.1983 | +0.086 |
| creative | 0.0942 | 0.1073 | +0.013 |
| vietnamese | 0.0851 | 0.1257 | +0.041 |
| health | — (no data) | 0.1419 | new |
| philosophy | — (no data) | 0.1153 | new |

`ml_ai` is the only regression, and it is small enough (−0.008) to sit inside
sampling noise given no seed is set (see Named gaps).

### Why the cross-version comparison is only indicative

- **Different test splits.** v0.4 drew 38 test samples from a 395-sample
  dataset with philosophy and health entirely absent; v0.5 draws 51 from 528
  covering all 10 categories. The splits share no guaranteed items and v0.5's
  includes two categories v0.4 could not measure at all.
- **v0.4's truncation rate is UNMEASURABLE.** `data/processed` was regenerated
  for v0.5, so v0.4's 38-sample split no longer exists on disk. Its truncation
  fraction at cap 512 cannot be recovered, so the −22.4% carries an
  unquantified bias. Recorded as unknown rather than omitted or assumed zero.
- **Different training text format.** v0.4 trained and was scored on a
  plain-text approximation; v0.5 uses the exact `<|im_start|>` template. The two
  perplexities are computed over different text distributions.
- **Different dtype.** v0.4 evaluated in fp16, v0.5 in bf16.

The protocol match is real and checkable: v0.4's evaluator
(`git show 5746ad3^:evaluate.py`) used `max_length=512` and accumulated
`loss.item() * input_ids.numel()` over total tokens — identical token-weighted
math to the current `compute_perplexity`. It is a like-for-like **protocol**,
not a like-for-like **corpus**.

## Verdict

**v0.5 ships.** It improves on v0.4 on every axis that can be compared and adds
coverage v0.4 lacked entirely:

- PPL improves 6.93 → 5.38 on the matched 512 protocol (−22.4%).
- ROUGE-L improves 0.1337 → 0.1534 (+14.7%), with 9 of 10 categories flat or up.
- Two categories that had zero teacher data in v0.4 (philosophy, health) are now
  trained and measured; health is mid-field at 4.41 PPL.
- A real validation split with early stopping exists for the first time; best
  val loss 1.4092 restored from checkpoint-125.

Not a clean sweep: `creative` at 14.95 PPL is a genuine weakness that no
measurement change explains away, `vietnamese` at 8.35 is second-weakest, and
`ml_ai` ROUGE regressed slightly. None of these block the release; all three are
next-iteration targets.

## Training history (v0.5)

From `checkpoints/adapter/checkpoint-162/trainer_state.json`, 162 steps, 3.0 epochs:

| Eval step | Val loss |
|---|---|
| 25 | 1.4684 |
| 50 | 1.4281 |
| 75 | 1.4173 |
| 100 | 1.4108 |
| **125** | **1.4092** (best, restored) |
| 150 | 1.4099 |
| 162 | 1.4101 |

Train loss fell 1.6673 (step 10) → ~1.31-1.38 (steps 150-160). Validation
flattened after step 100 and ticked up after 125, so early stopping restored
checkpoint-125. This is the sourced origin of the README's 1.409 claim.

## Named gaps

These are gaps, not oversights. Each is recorded so it is not rediscovered later.

1. **LLM-as-judge not run — 9Router unavailable at eval time.** The judge
   endpoint at `127.0.0.1:20128` returned no response (HTTP 000) when probed
   before and during this run, so `--judge` was not passed and no 1-5 rubric
   scores exist. `judge_score` is absent from every row in
   `evaluation_results.json`. Perplexity and ROUGE-L/token-F1 carry this report
   alone. Re-running with `--judge` when 9Router is up would add the third
   metric the plan allows for.
2. **Generation is not reproducible — no seed.** `generate_answers` samples with
   `do_sample=True, temperature=0.7, top_p=0.9` and no `torch.manual_seed`
   anywhere in `src/distill` (only `SPLIT_SEED=42` for dataset splitting). Every
   ROUGE-L and token-F1 number here is one draw from a distribution of unknown
   variance. The `ml_ai` −0.008 regression in particular cannot be distinguished
   from noise. Perplexity is unaffected (deterministic forward pass).
3. **`evaluation_results.json` contains fields the shipped CLI does not emit.**
   This run added `max_seq_length` and `perplexity_by_max_seq_length` so the
   payload can state which cap produced its number — the exact defect that let
   5.30 be published unlabelled. Consequence: **`python -m distill.evaluate`
   cannot currently reproduce this artifact**; it would write the same schema
   minus those two keys. Known, recorded debt until `evaluate.py` learns to emit
   them.
4. **`render_report` performs a cross-cap comparison.** With `--baseline-ppl`
   set it emits a `vs baseline PPL 6.93 | better` row comparing this run's
   canonical figure against a baseline measured at a different cap, with no cap
   labelling on either side. That row was removed from this report by hand. The
   generator will re-emit it on the next run.
5. **`render_report` output is thinner than the plan requires.** It emits four
   blocks: header, Overall, Per category, and two hardcoded Caveats lines.
   Sections hand-added here to satisfy the phase-04 validation list: Measurement
   conditions, README claim resolution, per-cap per-category table, the three
   named callouts, ROUGE-L table with best/worst examples, v0.4-vs-v0.5
   comparison, verdict, training history, this gaps list, and reproduction
   commands.
6. **`tests/test_evaluate.py` does not exist.** phase-04 step 2 requires it
   (metric math on fixtures, report rendering, graceful judge skip). Only
   `tests/test_eval_metrics.py` exists, so `render_report` and
   `judge_answers`' skip path have zero coverage. Nothing asserts what
   `render_report` emits, which is why gaps 4 and 5 went unnoticed. Suite state
   at eval time: `pytest tests/ -q` → 49 passed; `ruff check src/ tests/
   services/api/` → clean.
7. **`MAX_SEQ_LENGTH` is shared between training and evaluation.**
   `train.py:128` and `evaluate.py:61` read the same `config.MAX_SEQ_LENGTH`
   (`config.py:137`, default 1024). There is no way to express "evaluate at
   2048" except by overriding a variable that also reconfigures training to a
   length that OOMs. `.env.example` line 9 is a commented `# MAX_SEQ_LENGTH=1024`
   — an example value disagreeing with the 512 that v0.5 actually trained at —
   and no `EVAL_*` knobs are documented there at all. A separate
   `EVAL_MAX_SEQ_LENGTH` would match the existing `EVAL_MAX_SAMPLES` /
   `EVAL_MAX_NEW_TOKENS` naming.

## Run incident

The first attempt at this evaluation was **killed at 36/51 generations** and
lost everything. `evaluate.py` flushes only after the full generation loop
completes, so ~55 minutes of GPU work produced no artifact. The re-run was
detached from the harness process tree and checkpointed per sample; two monitor
shells were killed during it while the evaluation itself survived, which
confirms both the failure mode and the fix. Flush-at-end on a 90-minute
GPU run is a design defect independent of the reporting gaps above.

Total wall clock: PPL sweep ~2 min for all three caps; generation ~1.5 min per
sample, 51 samples.

## Reproduction

```bash
set PYTHONPATH=src

# Canonical headline (5.23, full token coverage).
# Per-process env override ONLY -- do not persist MAX_SEQ_LENGTH=2048 in .env,
# it is shared with training (see Named gaps #7).
MAX_SEQ_LENGTH=2048 python -m distill.evaluate --label v0.5

# v0.4 protocol match (5.38), the only figure comparable to v0.4's 6.93.
MAX_SEQ_LENGTH=512 python -m distill.evaluate --label v0.5 --baseline-ppl 6.93

# Config default, reproduces the previously unlabelled 5.30.
python -m distill.evaluate --label v0.5

# Add judge scores when 9Router is reachable.
python -m distill.evaluate --label v0.5 --judge
```

Note: the bare CLI writes the schema **without** `max_seq_length` /
`perplexity_by_max_seq_length`, and will re-emit the cross-cap baseline row
(Named gaps #3, #4).

## Caveats

- Test split is regenerated per dataset version; cross-version comparisons are
  indicative, not apples-to-apples.
- ROUGE-L against a single teacher reference under-credits valid alternative answers.
