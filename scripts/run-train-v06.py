"""Background trainer for v0.6 — retrain Qwen2.5-1.5B on the expanded dataset.

Sets the env knobs that v0.5 trained with on the 6 GB RTX 3060:
  LOAD_IN_4BIT=false        — bnb 4-bit is broken on Python 3.14 + torch nightly
  GRADIENT_CHECKPOINTING=true — required for bf16 LoRA to fit 6 GB VRAM
  MAX_SEQ_LENGTH=512        — 152K-vocab logits OOM at 1024
Then runs python -m distill.train as a fully detached process.
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# Training env — must match v0.5's known-good config for this hardware.
os.environ["LOAD_IN_4BIT"] = "false"
os.environ["GRADIENT_CHECKPOINTING"] = "true"
os.environ["MAX_SEQ_LENGTH"] = "512"

LOG = PROJECT / "logs" / "train-v06.log"
ERR = PROJECT / "logs" / "train-v06.err"
LOG.parent.mkdir(parents=True, exist_ok=True)

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

# Run train.py in-process (same env, same detached context) so the env vars
# are inherited by the torch/training subprocess it spawns.
sys.path.insert(0, str(PROJECT / "src"))
from distill.train import main as train_main

if __name__ == "__main__":
    raise SystemExit(train_main())
