# Evaluation v0.7 — distilled Qwen2.5-1.5B

**Date:** 2026-07-28  
**Model:** `D:\distill-gpt55\checkpoints\merged`  
**Test split:** 54 held-out samples (never trained on)  
**Perplexity cap:** 2048 (95.4% token coverage)  

## Overall

| Metric | Value |
|---|---|
| Perplexity (held-out, cap 2048, 95.4% token coverage) | **5.81** |
| Avg loss | 1.7599 |
| ROUGE-L F1 vs teacher | **0.1435** |
| Token-F1 vs teacher | 0.3050 |
| LLM-as-judge (1-5) | **not run** — see Named gaps |
| vs baseline PPL 5.23 @ cap 2048 | **worse** (5.81 vs 5.23, +11.1%) |

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
| 512 | 5.32 | 1.6714 | 33 / 54 | 23,074 / 48,675 | 47.4% |
| 1024 | 5.52 | 1.7089 | 17 / 54 | 34,773 / 48,675 | 71.4% |
| **2048** | **5.81** | 1.7599 | 4 / 54 | 46,421 / 48,675 | 95.4% |

Cap 2048 (bold) is this run's canonical figure.

## Per category

| Category | Samples | PPL @2048 | PPL @1024 | PPL @512 | ROUGE-L | Token-F1 |
|---|---|---|---|---|---|---|
| business | 5 | 5.64 | 5.64 | 5.56 | 0.1447 | 0.3293 |
| coding | 7 | 2.52 | 2.63 | 3.01 | 0.1790 | 0.2798 |
| creative | 6 | 14.11 | 14.98 | 16.01 | 0.0905 | 0.2258 |
| health | 5 | 4.43 | 4.43 | 4.49 | 0.1319 | 0.3093 |
| math | 5 | 2.84 | 2.84 | 2.87 | 0.2336 | 0.4134 |
| ml_ai | 5 | 4.15 | 4.03 | 4.22 | 0.1558 | 0.3503 |
| philosophy | 5 | 6.22 | 6.35 | 6.05 | 0.0959 | 0.2789 |
| reasoning | 5 | 3.60 | 3.60 | 3.60 | 0.1460 | 0.2473 |
| science | 5 | 6.02 | 5.80 | 5.36 | 0.1598 | 0.3552 |
| vietnamese | 6 | 7.01 | 7.20 | 7.46 | 0.1025 | 0.2895 |

## Best and worst by ROUGE-L

### Best

- `[coding] rouge=0.469 id=1` — Write a Python function to check if a string is a palindrome.
- `[math] rouge=0.328 id=124` — Explain Taylor series expansion with an example: approximate e^x at x=0.
- `[math] rouge=0.286 id=134` — Calculate the expected value of rolling a fair 6-sided die.
- `[science] rouge=0.251 id=247` — Explain how the seasons change on Earth.

### Worst

- `[reasoning] rouge=0.029 id=509` — Nếu 1 con gà và 1 rưỡi đẻ 1 quả trứng và 1 rưỡi trong 1 ngày và 1 rưỡi, hỏi 3 con gà đẻ bao nhiêu trứng trong…
- `[creative] rouge=0.052 id=165` — Write a story where the main character slowly realizes they are in a simulation.
- `[reasoning] rouge=0.072 id=92` — If there are 6 people in a room, what is the probability that at least 2 share the same birthday?
- `[vietnamese] rouge=0.078 id=550` — Viết bài giới thiệu về các địa điểm du lịch sinh thái ít người biết đến ở miền Tây Nam Bộ.

## Named gaps

1. **LLM-as-judge not run.** No `judge_score` is present on any row, so this report rests on perplexity and ROUGE-L/token-F1 alone. Re-run with `--judge` once the judge API is reachable to add the third metric.
2. **The truncation cap discarded 4.6% of the held-out tokens.** Only 95.4% of the split was scored, so this perplexity describes a smaller corpus than the split it is named after. Raise `EVAL_MAX_SEQ_LENGTH` or pass `--max-seq-length` for full coverage.

## Caveats

- Perplexity is only comparable at an identical truncation cap; figures measured
  at different caps are reported side by side, never differenced.
- Test split is regenerated per dataset version; cross-version comparisons are
  indicative, not apples-to-apples.
- ROUGE-L against a single teacher reference under-credits valid alternative answers.
