"""Download the student base model into the local model cache.

Usage::

    python -m distill.download_student
"""

from __future__ import annotations

from pathlib import Path

from . import config
from .logging_utils import get_logger

logger = get_logger("download")

HF_REPO = "Qwen/Qwen2.5-1.5B-Instruct"


def main() -> int:
    from huggingface_hub import snapshot_download

    target = Path(config.STUDENT_MODEL_ID)
    target.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s -> %s", HF_REPO, target)
    snapshot_download(HF_REPO, local_dir=str(target), max_workers=2)

    weights = list(target.glob("*.safetensors"))
    total_gb = sum(f.stat().st_size for f in weights) / 1e9
    logger.info("done — %d weight files, %.2f GB", len(weights), total_gb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
