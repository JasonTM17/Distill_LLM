# Evaluation v0.6 — distilled Qwen2.5-1.5B

**Date:** 2026-07-27  
**Model:** `D:\distill-gpt55\checkpoints\merged`  
**Test split:** 54 held-out samples (never trained on)  
**Perplexity cap:** 2048 (95.4% token coverage)  

## Overall

| Metric | Value |
|---|---|
| Perplexity (held-out, cap 2048, 95.4% token coverage) | **5.85** |
| Avg loss | 1.7659 |
| ROUGE-L F1 vs teacher | **0.1481** |
| Token-F1 vs teacher | 0.3063 |
| LLM-as-judge (1-5) | **not run** — see Named gaps |
| vs baseline PPL 5.23 @ cap 2048 | **worse** (5.85 vs 5.23, +11.9%) |

## Measurement conditions

| Setting | Value |
|---|---|
| Perplexity truncation cap (`EVAL_MAX_SEQ_LENGTH`) | 2048 |
| Samples truncated by the cap | 4 / 54 |
| Tokens scored | 46,421 / 48,675 |
| Token coverage | 95.4% |
| Generation seed | 42 |

### Perplexity by truncation cap

| Cap | PPL | Avg loss | Samples truncated | Tokens scored | Coverage |
|---|---|---|---|---|---|
| 512 | 5.39 | 1.6852 | 33 / 54 | 23,074 / 48,675 | 47.4% |
| 1024 | 5.58 | 1.7183 | 17 / 54 | 34,773 / 48,675 | 71.4% |
| **2048** | **5.85** | 1.7659 | 4 / 54 | 46,421 / 48,675 | 95.4% |

Cap 2048 (bold) is this run's canonical figure.

## Per category

| Category | Samples | PPL @2048 | PPL @1024 | PPL @512 | ROUGE-L | Token-F1 |
|---|---|---|---|---|---|---|
| business | 5 | 5.71 | 5.71 | 5.67 | 0.1486 | 0.3313 |
| coding | 7 | 2.54 | 2.65 | 3.04 | 0.2072 | 0.2847 |
| creative | 6 | 14.21 | 15.17 | 16.32 | 0.0879 | 0.2189 |
| health | 5 | 4.48 | 4.48 | 4.54 | 0.1584 | 0.3306 |
| math | 5 | 2.89 | 2.89 | 2.92 | 0.2009 | 0.3617 |
| ml_ai | 5 | 4.17 | 4.07 | 4.26 | 0.1487 | 0.3531 |
| philosophy | 5 | 6.22 | 6.38 | 6.16 | 0.0827 | 0.2333 |
| reasoning | 5 | 3.67 | 3.67 | 3.67 | 0.1887 | 0.3225 |
| science | 5 | 6.06 | 5.86 | 5.42 | 0.1593 | 0.3580 |
| vietnamese | 6 | 7.02 | 7.24 | 7.52 | 0.0971 | 0.2967 |

## Best and worst by ROUGE-L

### Best

- `[coding] rouge=0.475 id=1` — Write a Python function to check if a string is a palindrome.
- `[reasoning] rouge=0.302 id=509` — Nếu 1 con gà và 1 rưỡi đẻ 1 quả trứng và 1 rưỡi trong 1 ngày và 1 rưỡi, hỏi 3 con gà đẻ bao nhiêu trứng trong…
- `[science] rouge=0.289 id=247` — Explain how the seasons change on Earth.
- `[math] rouge=0.272 id=124` — Explain Taylor series expansion with an example: approximate e^x at x=0.

### Worst

- `[creative] rouge=0.056 id=165` — Write a story where the main character slowly realizes they are in a simulation.
- `[philosophy] rouge=0.057 id=469` — What is moral relativism? Arguments for and against.
- `[coding] rouge=0.062 id=17` — Write a Python script to parse a large CSV file and calculate summary statistics.
- `[reasoning] rouge=0.067 id=92` — If there are 6 people in a room, what is the probability that at least 2 share the same birthday?

## Named gaps

1. **LLM-as-judge not run.** No `judge_score` is present on any row, so this report rests on perplexity and ROUGE-L/token-F1 alone. Re-run with `--judge` once the judge API is reachable to add the third metric.
2. **The truncation cap discarded 4.6% of the held-out tokens.** Only 95.4% of the split was scored, so this perplexity describes a smaller corpus than the split it is named after. Raise `EVAL_MAX_SEQ_LENGTH` or pass `--max-seq-length` for full coverage.

## Caveats

- Perplexity is only comparable at an identical truncation cap; figures measured
  at different caps are reported side by side, never differenced.
- Test split is regenerated per dataset version; cross-version comparisons are
  indicative, not apples-to-apples.
- ROUGE-L against a single teacher reference under-credits valid alternative answers.
