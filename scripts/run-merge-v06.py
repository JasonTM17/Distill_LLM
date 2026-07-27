"""Merge v0.6 adapter into the base model — produces checkpoints/merged/.

Run AFTER distill.train has produced checkpoints/adapter/.
Quick (~2 min on CPU) but may exceed the 30s tool timeout, so run detached.
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from distill.merge import main as merge_main

if __name__ == "__main__":
    raise SystemExit(merge_main())
