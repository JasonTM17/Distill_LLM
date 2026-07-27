"""Export v0.6 merged model to GGUF (Q4_K_M + Q5_K_M).

Sets GGUF_MODEL_BASENAME=distill-gpt55-v0.6 so v0.5 GGUFs are not overwritten.
Run AFTER distill.merge has produced checkpoints/merged/.
"""
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
os.environ["GGUF_MODEL_BASENAME"] = "distill-gpt55-v0.6"
sys.path.insert(0, str(PROJECT / "src"))

from distill.export_gguf import main as export_main

if __name__ == "__main__":
    raise SystemExit(export_main())
