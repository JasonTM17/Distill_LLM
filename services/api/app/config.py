"""Service configuration, entirely env-driven so the container needs no files.

The service is deliberately standalone: it must not import from the training
package (``src/distill``) so its Docker image stays free of torch/transformers.
"""

from __future__ import annotations

import os


def _int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/gguf/distill-gpt55-v0.5-Q4_K_M.gguf")
MODEL_ID = os.environ.get("MODEL_ID", "distill-gpt55-qwen2.5-1.5b")
N_CTX = _int("MAX_CONTEXT_TOKENS", 4096)
N_THREADS = _int("N_THREADS", 0)  # 0 = llama.cpp decides

DEFAULT_MAX_TOKENS = _int("DEFAULT_MAX_TOKENS", 512)
MAX_TOKENS_LIMIT = _int("MAX_TOKENS_LIMIT", 2048)
DEFAULT_TEMPERATURE = _float("DEFAULT_TEMPERATURE", 0.7)
DEFAULT_TOP_P = _float("DEFAULT_TOP_P", 0.9)

RATE_LIMIT_REQUESTS = _int("RATE_LIMIT_REQUESTS", 60)
RATE_LIMIT_WINDOW_SECONDS = _int("RATE_LIMIT_WINDOW_SECONDS", 60)

CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
