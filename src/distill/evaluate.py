"""Rigorous evaluation of the merged student model on the held-out test split.

Measures, per category and overall:

* perplexity on the exact training-format text (never-seen samples),
* ROUGE-L F1 and token-F1 of generated answers vs the teacher reference,
* optional LLM-as-judge scores via 9Router (skipped gracefully when API is down).

Writes ``checkpoints/evaluation_results.json`` and a markdown report to
``plans/reports/evaluation-<label>.md``.

Usage::

    python -m distill.evaluate --label v0.5 --baseline-ppl 6.93
    python -m distill.evaluate --label v0.5 --judge --max-samples 20
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .eval_metrics import mean, rouge_l_f1, token_f1
from .generate_dataset import atomic_write_json
from .logging_utils import get_logger
from .model_loading import load_causal_lm, load_tokenizer

logger = get_logger("evaluate")

_JUDGE_RUBRIC = (
    "You are grading a student model's answer against a reference answer.\n"
    "Score the student 1-5: 1=wrong/nonsense, 2=mostly wrong, 3=partially correct,"
    " 4=correct but shallower than reference, 5=matches reference quality.\n"
    "Reply with ONLY the integer.\n\n"
    "Question:\n{question}\n\nReference answer:\n{reference}\n\nStudent answer:\n{answer}\n"
)


def load_test_split(max_samples: int | None = None) -> list[dict[str, Any]]:
    with open(config.TEST_FILE, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return rows[: max_samples or config.EVAL_MAX_SAMPLES]


def compute_perplexity(model, tokenizer, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Token-weighted perplexity over each row's full training-format text."""
    import torch

    losses_by_category: dict[str, list[tuple[float, int]]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for row in rows:
            tokens = tokenizer(
                row["text"],
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH,
                return_tensors="pt",
            ).to(model.device)
            output = model(tokens["input_ids"], labels=tokens["input_ids"])
            count = int(tokens["input_ids"].numel())
            losses_by_category[row["category"]].append((float(output.loss), count))

    def _ppl(pairs: list[tuple[float, int]]) -> tuple[float, float]:
        total = sum(loss * n for loss, n in pairs)
        tokens = sum(n for _, n in pairs)
        avg = total / tokens if tokens else 0.0
        import math

        return avg, math.exp(avg)

    all_pairs = [pair for pairs in losses_by_category.values() for pair in pairs]
    overall_loss, overall_ppl = _ppl(all_pairs)
    per_category = {
        category: {"loss": round(loss, 4), "perplexity": round(ppl, 2), "samples": len(pairs)}
        for category, pairs in sorted(losses_by_category.items())
        for loss, ppl in [_ppl(pairs)]
    }
    return {
        "loss": round(overall_loss, 4),
        "perplexity": round(overall_ppl, 2),
        "per_category": per_category,
    }


def generate_answers(model, tokenizer, rows, max_new_tokens: int | None = None):
    """Generate a student answer per row using the standard chat template."""
    import torch

    results = []
    max_new_tokens = max_new_tokens or config.EVAL_MAX_NEW_TOKENS
    for index, row in enumerate(rows, start=1):
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": row["instruction"]},
        ]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=config.EVAL_TEMPERATURE,
                top_p=config.EVAL_TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        answer = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        results.append(
            {
                "id": row["id"],
                "category": row["category"],
                "instruction": row["instruction"],
                "reference": row["output"],
                "answer": answer,
                "rouge_l": round(rouge_l_f1(row["output"], answer), 4),
                "token_f1": round(token_f1(row["output"], answer), 4),
            }
        )
        logger.info(
            "[%d/%d] id=%s %s rouge=%.2f",
            index, len(rows), row["id"], row["category"], results[-1]["rouge_l"],
        )
    return results


def judge_answers(results: list[dict[str, Any]]) -> None:
    """Attach 1-5 judge scores in place; skip silently if the API is down."""
    from .teacher_client import TeacherClient, TeacherError

    try:
        client = TeacherClient(model=config.JUDGE_MODEL)
        for row in results:
            prompt = _JUDGE_RUBRIC.format(
                question=row["instruction"],
                reference=row["reference"][:4000],
                answer=row["answer"][:4000],
            )
            reply = client.complete(
                prompt, max_tokens=config.JUDGE_MAX_TOKENS, temperature=0.0, max_retries=2
            )
            digits = [c for c in reply.text if c.isdigit()]
            row["judge_score"] = int(digits[0]) if digits else None
    except TeacherError as exc:
        logger.warning("judge unavailable, skipping: %s", exc)
    except Exception as exc:  # API client construction/network failure
        logger.warning("judge unavailable, skipping: %s", exc)


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_category[row["category"]].append(row)
    per_category = {
        category: {
            "samples": len(rows),
            "rouge_l": round(mean([r["rouge_l"] for r in rows]), 4),
            "token_f1": round(mean([r["token_f1"] for r in rows]), 4),
            "judge": round(
                mean([r["judge_score"] for r in rows if r.get("judge_score")]), 2
            )
            if any(r.get("judge_score") for r in rows)
            else None,
        }
        for category, rows in sorted(by_category.items())
    }
    return {
        "rouge_l": round(mean([r["rouge_l"] for r in results]), 4),
        "token_f1": round(mean([r["token_f1"] for r in results]), 4),
        "per_category": per_category,
    }


def render_report(payload: dict[str, Any]) -> str:
    """Render the evaluation payload as a markdown report."""
    ppl = payload["perplexity"]
    gen = payload["generation"]
    lines = [
        f"# Evaluation {payload['label']} — distilled Qwen2.5-1.5B",
        "",
        f"**Date:** {payload['evaluated_at'][:10]}  ",
        f"**Model:** `{payload['model_path']}`  ",
        f"**Test split:** {payload['num_samples']} held-out samples (never trained on)",
        "",
        "## Overall",
        "",
        "| Metric | Value |" ,
        "|---|---|",
        f"| Perplexity (held-out) | **{ppl['perplexity']}** |",
        f"| Avg loss | {ppl['loss']} |",
        f"| ROUGE-L F1 vs teacher | **{gen['rouge_l']}** |",
        f"| Token-F1 vs teacher | {gen['token_f1']} |",
    ]
    if payload.get("baseline_ppl"):
        direction = "better" if ppl["perplexity"] < payload["baseline_ppl"] else "worse"
        lines.append(
            f"| vs baseline PPL {payload['baseline_ppl']} | {direction} |"
        )
    lines += [
        "",
        "## Per category",
        "",
        "| Category | Samples | PPL | ROUGE-L | Token-F1 |",
        "|---|---|---|---|---|",
    ]
    categories = sorted(
        set(ppl["per_category"]) | set(gen["per_category"])
    )
    for category in categories:
        p = ppl["per_category"].get(category, {})
        g = gen["per_category"].get(category, {})
        lines.append(
            f"| {category} | {g.get('samples', p.get('samples', 0))} "
            f"| {p.get('perplexity', '—')} | {g.get('rouge_l', '—')} "
            f"| {g.get('token_f1', '—')} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- Test split is regenerated per dataset version; cross-version comparisons are",
        "  indicative, not apples-to-apples.",
        "- ROUGE-L against a single teacher reference under-credits valid alternative answers.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the merged student model")
    parser.add_argument("--label", default="v0.5", help="report label, e.g. v0.5")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--judge", action="store_true", help="add LLM-as-judge scores")
    parser.add_argument("--baseline-ppl", type=float, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    args = parser.parse_args(argv)

    model_path = args.model_path or config.MERGED_MODEL_DIR
    logger.info("loading model: %s", model_path)
    model = load_causal_lm(model_path)
    tokenizer = load_tokenizer(model_path)

    rows = load_test_split(args.max_samples)
    logger.info("evaluating %d held-out samples", len(rows))

    ppl = compute_perplexity(model, tokenizer, rows)
    logger.info("perplexity=%.2f loss=%.4f", ppl["perplexity"], ppl["loss"])

    results = generate_answers(model, tokenizer, rows, args.max_new_tokens)
    if args.judge:
        judge_answers(results)
    gen = aggregate(results)

    payload = {
        "label": args.label,
        "model_path": str(model_path),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_samples": len(rows),
        "baseline_ppl": args.baseline_ppl,
        "perplexity": ppl,
        "generation": gen,
        "details": results,
    }
    results_file = config.CHECKPOINT_DIR / "evaluation_results.json"
    atomic_write_json(results_file, payload)
    report_file = config.REPORTS_DIR / f"evaluation-{args.label}.md"
    report_file.write_text(render_report(payload), encoding="utf-8")
    logger.info("wrote %s and %s", results_file, report_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
