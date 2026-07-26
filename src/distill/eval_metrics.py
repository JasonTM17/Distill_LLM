"""Pure text-similarity metrics used by the evaluation suite.

Kept free of torch/transformers imports so they are unit-testable anywhere.
"""

from __future__ import annotations

from collections import Counter


def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (basis of ROUGE-L).

    Memory-light two-row DP: teacher references can run to thousands of tokens.
    """
    if not a or not b:
        return 0
    if len(b) > len(a):  # keep the inner row the shorter side
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[len(b)]


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 in [0, 1] over whitespace tokens."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(reference: str, hypothesis: str) -> float:
    """Bag-of-tokens F1 — order-insensitive complement to ROUGE-L."""
    ref_counts = Counter(reference.split())
    hyp_counts = Counter(hypothesis.split())
    if not ref_counts or not hyp_counts:
        return 0.0
    overlap = sum((ref_counts & hyp_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(hyp_counts.values())
    recall = overlap / sum(ref_counts.values())
    return 2 * precision * recall / (precision + recall)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
