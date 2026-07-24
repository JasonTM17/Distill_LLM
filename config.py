"""Central configuration for GPT-5.5 Distill project."""

import os

# ── Paths ──────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
MERGED_DIR = os.path.join(PROJECT_DIR, "merged_model")

# ── 9Router API ─────────────────────────────────────────
API_BASE_URL = "http://127.0.0.1:20128/v1"
API_KEY = "sk-59be692bbb02885c-kfjrks-07c55700"

# ── Teacher (distill source) ────────────────────────────
TEACHER_MODEL = "cx/gpt-5.5-xhigh"
TEACHER_MAX_TOKENS = 2048
TEACHER_TEMPERATURE = 0.7

# ── Student (distill target) ────────────────────────────
STUDENT_MODEL_ID = "D:/models/qwen15-1.5b"
STUDENT_LOCAL_DIR = os.path.join(CHECKPOINT_DIR, "student_base")

# ── Dataset generation ──────────────────────────────────
PROMPTS_FILE = os.path.join(DATA_DIR, "prompts.json")
TEACHER_OUTPUT_FILE = os.path.join(RAW_DIR, "teacher_outputs.json")
PROCESSED_DATASET_FILE = os.path.join(PROCESSED_DIR, "dataset_train.json")
TEST_SPLIT_RATIO = 0.1
REQUEST_DELAY = 1.5          # seconds between API calls
CHECKPOINT_EVERY = 10        # save partial after N prompts
MAX_RETRIES = 3

# ── LoRA / QLoRA Training ───────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512
SAVE_STEPS = 100
LOGGING_STEPS = 10
FP16 = True
WARMUP_RATIO = 0.03
GRADIENT_CHECKPOINTING = False

# 4-bit quantization
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_DOUBLE_QUANT = True

# ── Adapter save paths ──────────────────────────────────
ADAPTER_DIR = os.path.join(CHECKPOINT_DIR, "adapter")
MERGED_MODEL_DIR = os.path.join(CHECKPOINT_DIR, "merged")

# Ensure all directories exist on import
for _d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR, MERGED_DIR]:
    os.makedirs(_d, exist_ok=True)
