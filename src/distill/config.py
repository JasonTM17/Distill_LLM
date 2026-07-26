"""Central configuration for the distill-gpt55 project.

All tunables live here. Secrets are read from the environment (loaded from a
local `.env` file when present) and are never hardcoded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Project layout ─────────────────────────────────────────────────────────
# src/distill/config.py -> src/distill -> src -> <project root>
PROJECT_DIR = Path(__file__).resolve().parents[2]


def _load_dotenv(env_path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env without overriding real env vars.

    Deliberately minimal (no python-dotenv dependency). Supports `#` comments,
    blank lines, `export KEY=value` prefixes, and quoted values.
    """
    path = env_path or (PROJECT_DIR / ".env")
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv()


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
REPORTS_DIR = PROJECT_DIR / "plans" / "reports"

PROMPTS_FILE = DATA_DIR / "prompts.json"
TEACHER_OUTPUT_FILE = RAW_DIR / "teacher_outputs.json"

TRAIN_FILE = PROCESSED_DIR / "dataset_train.json"
VALIDATION_FILE = PROCESSED_DIR / "dataset_validation.json"
TEST_FILE = PROCESSED_DIR / "dataset_test.json"
DATASET_STATS_FILE = PROCESSED_DIR / "dataset_stats.json"

ADAPTER_DIR = CHECKPOINT_DIR / "adapter"
MERGED_MODEL_DIR = CHECKPOINT_DIR / "merged"
GGUF_DIR = CHECKPOINT_DIR / "gguf"

# Backwards-compatible aliases used by older scripts/tests.
PROCESSED_DATASET_FILE = TRAIN_FILE
PROCESSED_TEST_FILE = TEST_FILE


# ── Teacher API (9Router / OpenAI-compatible) ──────────────────────────────
API_BASE_URL = _env_str("API_BASE_URL", "http://127.0.0.1:20128/v1")
API_KEY = _env_str("API_KEY", "")

TEACHER_MODEL = _env_str("TEACHER_MODEL", "cx/gpt-5.5-xhigh")
TEACHER_MAX_TOKENS = _env_int("TEACHER_MAX_TOKENS", 2048)
TEACHER_TEMPERATURE = _env_float("TEACHER_TEMPERATURE", 0.7)
TEACHER_TIMEOUT = _env_float("TEACHER_TIMEOUT", 180.0)

# Judge model used by the LLM-as-judge evaluator. Kept separate from the teacher
# so scoring is not biased by using the exact same sampling configuration.
JUDGE_MODEL = _env_str("JUDGE_MODEL", "cx/gpt-5.5-high")
JUDGE_MAX_TOKENS = _env_int("JUDGE_MAX_TOKENS", 512)

# ── Generation loop ────────────────────────────────────────────────────────
REQUEST_DELAY = _env_float("REQUEST_DELAY", 3.0)
MAX_RETRIES = _env_int("MAX_RETRIES", 5)
RETRY_BACKOFF_BASE = _env_float("RETRY_BACKOFF_BASE", 4.0)
RETRY_BACKOFF_MAX = _env_float("RETRY_BACKOFF_MAX", 120.0)
MIN_OUTPUT_CHARS = _env_int("MIN_OUTPUT_CHARS", 40)
GENERATION_CONCURRENCY = _env_int("GENERATION_CONCURRENCY", 1)


# ── Dataset split ──────────────────────────────────────────────────────────
TEST_SPLIT_RATIO = _env_float("TEST_SPLIT_RATIO", 0.10)
VALIDATION_SPLIT_RATIO = _env_float("VALIDATION_SPLIT_RATIO", 0.10)
SPLIT_SEED = _env_int("SPLIT_SEED", 42)

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable assistant. Answer thoroughly and clearly."
)


# ── Student model / QLoRA training ─────────────────────────────────────────
STUDENT_MODEL_ID = _env_str("STUDENT_MODEL_ID", "D:/models/qwen15-1.5b")

LORA_R = _env_int("LORA_R", 16)
LORA_ALPHA = _env_int("LORA_ALPHA", 32)
LORA_DROPOUT = _env_float("LORA_DROPOUT", 0.05)
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

BATCH_SIZE = _env_int("BATCH_SIZE", 1)
GRADIENT_ACCUMULATION_STEPS = _env_int("GRADIENT_ACCUMULATION_STEPS", 8)
LEARNING_RATE = _env_float("LEARNING_RATE", 2e-4)
NUM_EPOCHS = _env_float("NUM_EPOCHS", 3)
MAX_SEQ_LENGTH = _env_int("MAX_SEQ_LENGTH", 1024)
WARMUP_RATIO = _env_float("WARMUP_RATIO", 0.03)
WEIGHT_DECAY = _env_float("WEIGHT_DECAY", 0.01)
LR_SCHEDULER_TYPE = _env_str("LR_SCHEDULER_TYPE", "cosine")
MAX_GRAD_NORM = _env_float("MAX_GRAD_NORM", 0.3)

LOGGING_STEPS = _env_int("LOGGING_STEPS", 10)
EVAL_STEPS = _env_int("EVAL_STEPS", 25)
SAVE_STEPS = _env_int("SAVE_STEPS", 25)
SAVE_TOTAL_LIMIT = _env_int("SAVE_TOTAL_LIMIT", 2)
EARLY_STOPPING_PATIENCE = _env_int("EARLY_STOPPING_PATIENCE", 3)

GRADIENT_CHECKPOINTING = _env_bool("GRADIENT_CHECKPOINTING", False)
TRAIN_ON_COMPLETIONS_ONLY = _env_bool("TRAIN_ON_COMPLETIONS_ONLY", True)

# 4-bit quantization (mandatory on 6 GB VRAM)
# bfloat16 throughout: the checkpoint is bf16 and converting to fp16 during
# weight materialization access-violates on this torch nightly (Windows).
# RTX 3060 is Ampere, so bf16 compute is fully supported.
LOAD_IN_4BIT = _env_bool("LOAD_IN_4BIT", True)
BNB_4BIT_QUANT_TYPE = _env_str("BNB_4BIT_QUANT_TYPE", "nf4")
BNB_4BIT_COMPUTE_DTYPE = _env_str("BNB_4BIT_COMPUTE_DTYPE", "bfloat16")
BNB_4BIT_DOUBLE_QUANT = _env_bool("BNB_4BIT_DOUBLE_QUANT", True)


# ── Evaluation ─────────────────────────────────────────────────────────────
EVAL_MAX_SAMPLES = _env_int("EVAL_MAX_SAMPLES", 200)
EVAL_MAX_NEW_TOKENS = _env_int("EVAL_MAX_NEW_TOKENS", 512)
EVAL_TEMPERATURE = _env_float("EVAL_TEMPERATURE", 0.7)
EVAL_TOP_P = _env_float("EVAL_TOP_P", 0.9)


# ── Inference service ──────────────────────────────────────────────────────
API_HOST = _env_str("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 8000)
MODEL_PATH = _env_str("MODEL_PATH", str(MERGED_MODEL_DIR))
MODEL_DTYPE = _env_str("MODEL_DTYPE", "float16")
MODEL_LOAD_IN_4BIT = _env_bool("MODEL_LOAD_IN_4BIT", False)
MAX_CONTEXT_TOKENS = _env_int("MAX_CONTEXT_TOKENS", 4096)
RATE_LIMIT_REQUESTS = _env_int("RATE_LIMIT_REQUESTS", 60)
RATE_LIMIT_WINDOW_SECONDS = _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
CORS_ALLOW_ORIGINS = [
    o.strip()
    for o in _env_str("CORS_ALLOW_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]


@dataclass(frozen=True)
class GenerationDefaults:
    """Default sampling parameters for the inference service."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    stop: list[str] = field(default_factory=list)


GENERATION_DEFAULTS = GenerationDefaults()


def ensure_directories() -> None:
    """Create the directories the pipeline writes into."""
    for directory in (
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        CHECKPOINT_DIR,
        REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def summary() -> dict[str, object]:
    """Return a redacted snapshot of the active configuration for logging."""
    return {
        "project_dir": str(PROJECT_DIR),
        "api_base_url": API_BASE_URL,
        "api_key_set": bool(API_KEY),
        "teacher_model": TEACHER_MODEL,
        "judge_model": JUDGE_MODEL,
        "student_model_id": STUDENT_MODEL_ID,
        "max_seq_length": MAX_SEQ_LENGTH,
        "num_epochs": NUM_EPOCHS,
        "load_in_4bit": LOAD_IN_4BIT,
    }
