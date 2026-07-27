"""Background generator for v0.6 — teacher outputs for new prompts (IDs 531-570).

Generates teacher outputs via cx/gpt-5.5-xhigh for the 40 new prompts added to
creative, vietnamese, and reasoning categories. Resumable: re-run picks up where
it left off. Writes to data/raw/teacher_outputs.json atomically per record.
"""
import sys
from pathlib import Path

# Ensure src/ is importable when run from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from distill.generate_dataset import main

if __name__ == "__main__":
    raise SystemExit(
        main(["--categories", "creative", "vietnamese", "reasoning", "--delay", "0"])
    )
