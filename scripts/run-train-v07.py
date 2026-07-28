"""Background trainer for v0.7 — LoRA capacity experiment on the existing 570 dataset.

v0.6 regressed overall (PPL 5.85 vs v0.5 5.23) after expanding only the weak
categories: science (4.81->6.06) and philosophy (5.26->6.22) lost capacity.
v0.7 tests whether more LoRA capacity recovers them, with ONE changed variable
vs v0.6: LORA_R 16 -> 32 (and LORA_ALPHA 32 -> 64 to keep alpha/r = 2).

Same dataset, same stratified split (seed 42), same bf16 / gradient-checkpointing
/ seq-512 / 3-epoch config as v0.6 -> the held-out set is identical, so v0.7's
PPL is directly comparable to v0.6's 5.85 (not to v0.5's 5.23, whose 51-sample
test split differs).

Env knobs that v0.5/v0.6 trained with on the 6 GB RTX 3060 (unchanged except rank):
  LOAD_IN_4BIT=false        — bnb 4-bit is broken on Python 3.14 + torch nightly
  GRADIENT_CHECKPOINTING=true — required for bf16 LoRA to fit 6 GB VRAM
  MAX_SEQ_LENGTH=512        — 152K-vocab logits OOM at 1024
Then runs python -m distill.train in-process so the env vars are inherited.
"""
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# Training env — identical to v0.6 except the LoRA rank/alpha (capacity).
os.environ["LOAD_IN_4BIT"] = "false"
os.environ["GRADIENT_CHECKPOINTING"] = "true"
os.environ["MAX_SEQ_LENGTH"] = "512"
os.environ["LORA_R"] = "32"
os.environ["LORA_ALPHA"] = "64"

# Run train.py in-process (same env, same detached context) so the env vars
# are inherited by the torch/training subprocess it spawns.
sys.path.insert(0, str(PROJECT / "src"))
from distill.train import main as train_main

if __name__ == "__main__":
    raise SystemExit(train_main())
