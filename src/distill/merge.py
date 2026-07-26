"""Merge the trained LoRA adapter into a standalone bf16 model.

Unlike v0.4 (which merged into the 4-bit-quantized base, baking NF4 rounding
error into the merged weights), this loads the base unquantized on CPU and
merges there — the merged model quality is bounded by the adapter, not the
quantization. Output feeds evaluation, the GGUF export, and the API service.

Precision is bf16, not fp16: ``load_causal_lm`` pins ``torch.bfloat16`` and
exposes no override, so the saved ``config.json`` records ``bfloat16``.

Usage::

    python -m distill.merge
"""

from __future__ import annotations

from . import config
from .logging_utils import get_logger
from .model_loading import load_causal_lm, load_tokenizer

logger = get_logger("merge")


def main() -> int:
    from peft import PeftModel

    if not (config.ADAPTER_DIR / "adapter_config.json").exists():
        logger.error("no trained adapter at %s — run distill.train first", config.ADAPTER_DIR)
        return 1

    logger.info("loading base model: %s", config.STUDENT_MODEL_ID)
    base = load_causal_lm(config.STUDENT_MODEL_ID)
    logger.info("applying adapter: %s", config.ADAPTER_DIR)
    model = PeftModel.from_pretrained(base, str(config.ADAPTER_DIR))
    merged = model.merge_and_unload()

    config.MERGED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("saving merged model -> %s", config.MERGED_MODEL_DIR)
    merged.save_pretrained(str(config.MERGED_MODEL_DIR), safe_serialization=True)

    tokenizer = load_tokenizer(config.STUDENT_MODEL_ID)
    tokenizer.save_pretrained(str(config.MERGED_MODEL_DIR))
    logger.info("done — merged bf16 model ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
