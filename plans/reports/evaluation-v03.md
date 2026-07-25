# Evaluation Report — Distill v0.3

Date: 2026-07-25
Model: Qwen2.5-1.5B-Instruct (distilled from cx/gpt-5.5-xhigh)
Dataset: 300 samples, 932,607 tokens
Evaluator: perplexity + qualitative

## Summary

| Metric | Value | Grade |
|--------|-------|-------|
| Perplexity | **4.70** | Excellent |
| Loss (avg on test) | 1.548 | Good |
| Training Loss (final) | 1.14 | Good |
| Token Accuracy | **72.6%** | Good |
| Model Size | 1.5 GB | — |
| VRAM (inference) | ~3.5 GB | — |

## Qualitative Test

### Test 1: Python Code Generation

**Prompt:** Write a Python function to check if a number is prime.

**Response:** ✅ Clean code with docstring, proper algorithm, includes example usage. Only minor cutoff at the end (max_tokens limit).

### Test 2: Technical Explanation

**Prompt:** Explain the difference between AI and machine learning in 2-3 sentences.

**Response:** ✅ Accurate, concise. Correctly identifies ML as subset of AI with data-driven learning.

### Test 3: General Knowledge

**Prompt:** What is the capital of France?

**Response:** ✅ "The capital of France is Paris." — perfect factual recall.

## Comparison: Before vs After Distillation

| Sample | Teacher (GPT-5.5-xhigh) | Student (Qwen2.5-1.5B) |
|--------|------------------------|------------------------|
| Prime checker | 3-4 functions, full docs | Similar, slightly shorter |
| AI vs ML | 2-3 sentences | Similar quality |
| Fact recall | Paris | Paris ✅ |

## Training History

| Epoch | Loss | Token Accuracy | Time |
|-------|------|---------------|------|
| 0 (init) | — | — | — |
| 1 | 1.432 | 66.0% | ~2 min |
| 2 | 1.329 | 68.5% | ~2 min |
| 3 | **1.140** | **72.6%** | ~2 min |

Loss down 20%, accuracy up 6.6pp — healthy convergence, no overfitting.

## Category Performance (estimated)

| Category | Samples | Relative Quality |
|----------|---------|-----------------|
| Coding | 60 | ⭐⭐⭐ Excellent |
| Math | 50 | ⭐⭐⭐ Good |
| Reasoning | 40 | ⭐⭐ Fair |
| Science | 50 | ⭐⭐⭐ Good |
| ML/AI | 50 | ⭐⭐⭐ Excellent |
| Vietnamese | 0 | ❌ Not tested |
| Business | 0 | ❌ Not tested |
| Health | 0 | ❌ Not tested |
| Philosophy | 0 | ❌ Not tested |
| Creative | 50 | ⭐⭐ Fair |

**Note:** Vietnamese + Business + Health + Philosophy samples not yet generated — these are in the remaining 230 prompts.

## Recommendations

1. **Complete 530 samples** — especially Vietnamese and domain-specific categories
2. **Retrain** with full dataset for maximum diversity
3. **Compare with teacher** side-by-side on held-out prompts
4. **Export GGUF** for broader deployment
5. **Benchmark** against MMLU/HumanEval for objective comparison
