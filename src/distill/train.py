"""QLoRA fine-tune of the student model with a real validation loop.

v0.5 upgrades over ``train_student.py``:

* trains on the phase-2 splits (Qwen chat template with special tokens),
* completion-only loss via trl's prompt/completion dataset format,
* eval on the validation split with early stopping + best-checkpoint restore,
* previous adapter is archived instead of silently overwritten.

Usage::

    python -m distill.train
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import config
from .logging_utils import get_logger
from .model_loading import load_causal_lm
from .safetensors_pretouch import install as install_pretouch

logger = get_logger("train")


def load_split(path: Path) -> list[dict[str, str]]:
    """Load one processed split as trl prompt/completion (or plain text) rows."""
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    formatted = []
    for row in rows:
        if config.TRAIN_ON_COMPLETIONS_ONLY and row.get("prompt_text"):
            completion = row["text"][len(row["prompt_text"]) :]
            formatted.append({"prompt": row["prompt_text"], "completion": completion})
        else:
            formatted.append({"text": row["text"]})
    return formatted


def archive_previous_adapter() -> None:
    """Move the current adapter out of the way once, as v0.4."""
    adapter = config.ADAPTER_DIR
    archive = config.CHECKPOINT_DIR / "v0.4_adapter"
    if adapter.exists() and any(adapter.iterdir()) and not archive.exists():
        logger.info("archiving previous adapter -> %s", archive)
        shutil.move(str(adapter), str(archive))


def main() -> int:
    install_pretouch()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
    )
    from trl import SFTConfig, SFTTrainer

    logger.info("student=%s", config.STUDENT_MODEL_ID)
    logger.info(
        "gpu=%s vram=%.1fGB",
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
    )

    train_rows = load_split(config.TRAIN_FILE)
    val_rows = load_split(config.VALIDATION_FILE)
    logger.info("train=%d validation=%d", len(train_rows), len(val_rows))
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(val_rows)

    # dtype must stay bf16 (checkpoint native): fp16 conversion during weight
    # materialization crashes this torch nightly on Windows, and the bnb
    # on-the-fly 4-bit path crashes the same way regardless of dtype. With
    # LOAD_IN_4BIT=false we train LoRA on the full bf16 model instead (fits
    # 6GB VRAM only with gradient checkpointing enabled).
    if config.LOAD_IN_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE),
            bnb_4bit_use_double_quant=config.BNB_4BIT_DOUBLE_QUANT,
        )
        logger.info("loading base model (4-bit NF4)...")
        model = AutoModelForCausalLM.from_pretrained(
            config.STUDENT_MODEL_ID,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info("loading base model (bf16, no quantization)...")
        model = load_causal_lm(config.STUDENT_MODEL_ID)
        if config.GRADIENT_CHECKPOINTING:
            model.enable_input_require_grads()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_MODEL_ID)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("trainable params: %.1fM", trainable / 1e6)

    archive_previous_adapter()

    sft_config = SFTConfig(
        output_dir=str(config.ADAPTER_DIR),
        max_length=config.MAX_SEQ_LENGTH,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        fp16=False,
        bf16=not config.LOAD_IN_4BIT,
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)],
    )

    logger.info("starting training...")
    trainer.train()

    logger.info("saving adapter -> %s", config.ADAPTER_DIR)
    trainer.save_model(str(config.ADAPTER_DIR))
    tokenizer.save_pretrained(str(config.ADAPTER_DIR))

    metrics = trainer.evaluate()
    logger.info("final validation: %s", {k: round(v, 4) for k, v in metrics.items()
                                         if isinstance(v, float)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
