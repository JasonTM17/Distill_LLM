"""QLoRA fine-tune Qwen2.5-3B on teacher-generated dataset."""
import sys
import io
import json
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

import config


def load_and_format_dataset(path):
    """Load teacher outputs and format as ShareGPT-style for Qwen chat template."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    formatted = []
    for item in raw["data"]:
        if not item.get("success") or not item.get("output"):
            continue
        text = (
            f"<|im_start|>system\nYou are a helpful, knowledgeable assistant. "
            "Answer thoroughly and clearly.<|im_end|>\n"
            f"<|im_start|>user\n{item['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{item['output']}<|im_end|>"
        )
        formatted.append({"text": text})

    print(f"Formatted {len(formatted)} valid samples")
    return Dataset.from_list(formatted)


def main():
    print("=" * 60)
    print(f"Student: {config.STUDENT_MODEL_ID}")
    print(f"Dataset: {config.TEACHER_OUTPUT_FILE}")
    print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    print("=" * 60)

    # ── Quantization config ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.LOAD_IN_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_use_double_quant=config.BNB_4BIT_DOUBLE_QUANT,
    )

    # ── Load base model ──
    print("Loading base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        config.STUDENT_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        config.STUDENT_MODEL_ID,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── LoRA config ──
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
    print(f"Trainable params: {trainable:,} ({trainable/1e6:.1f}M)")

    # ── Load dataset ──
    print("Loading dataset...")
    train_dataset = load_and_format_dataset(config.TEACHER_OUTPUT_FILE)

    # ── Training args ──
    training_args = TrainingArguments(
        output_dir=config.ADAPTER_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        fp16=False,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        save_strategy="steps",
        warmup_ratio=config.WARMUP_RATIO,
        gradient_checkpointing=False,
        optim="adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    # ── SFT Trainer ──
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )

    print("\nStarting training...")
    trainer.train()

    # ── Save adapter ──
    print(f"\nSaving adapter to {config.ADAPTER_DIR}")
    model.save_pretrained(config.ADAPTER_DIR)
    tokenizer.save_pretrained(config.ADAPTER_DIR)

    # ── Merge adapter into base model ──
    print(f"\nMerging adapter -> {config.MERGED_MODEL_DIR}")
    merged = model.merge_and_unload()
    merged.save_pretrained(config.MERGED_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(config.MERGED_MODEL_DIR)
    print("Done! Merged model ready for inference.")


if __name__ == "__main__":
    main()
