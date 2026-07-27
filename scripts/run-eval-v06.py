"""Background evaluator for v0.6 — held-out evaluation WITH LLM-as-judge.

Runs the full evaluation suite that v0.5 ran, plus the --judge flag that v0.5
skipped. Compares against the v0.5 headline (PPL 5.23 at cap 2048).

Must run AFTER distill.merge has produced checkpoints/merged/.
"""
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from distill.evaluate import main as eval_main

if __name__ == "__main__":
    # PPL sweep at all three caps for comparability + judge (was skipped in v0.5)
    # + baseline comparison against v0.5's canonical full-coverage figure.
    raise SystemExit(
        eval_main([
            "--label", "v0.6",
            "--ppl-caps", "512,1024,2048",
            "--judge",
            "--baseline-ppl", "5.23",
            "--baseline-cap", "2048",
        ])
    )
