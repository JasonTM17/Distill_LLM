"""QLoRA fine-tune Qwen2.5-1.5B-Instruct on teacher-generated dataset."""
import sys
import io
import json
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Windows paging file workaround: pre-touch the safetensors file to force OS to
# load it into the file system cache. This avoids the pagefile exhaustion error
# when transformers' safe_open tries to mmap a 3GB file on systems where C: has
# insufficient free space for pagefile growth.
import safetensors
from safetensors import safe_open as _orig_safe_open
import os as _os_fs

_touched_files = set()


def _touch_file(filepath):
    """Force OS to read the file into filesystem cache by reading it sequentially."""
    if filepath in _touched_files or not _os_fs.path.isfile(filepath):
        return
    print(f"  [fs-cache] pre-loading {os.path.basename(filepath)} into OS cache...", flush=True)
    size = _os_fs.path.getsize(filepath)
    # Read in 64MB chunks to avoid MemoryError
    chunk = 64 * 1024 * 1024
    with open(filepath, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
    _touched_files.add(filepath)


def _patched_safe_open(filename, framework, device=None, **kwargs):
    if isinstance(filename, str):
        _touch_file(filename)
    return _orig_safe_open(filename, framework, device=device, **kwargs)


safetensors.safe_open = _patched_safe_open

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
    """Load processed dataset (list of {text, category, id}) and return as Dataset."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept either list (new format from format_dataset.py) or dict with 'data' key (legacy)
    if isinstance(data, dict) and "data" in data:
        items = data["data"]
    else:
        items = data

    formatted = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "text" in item and item.get("text"):
            formatted.append({"text": item["text"]})
            continue
        # Legacy fallback: build text from instruction/output
        if item.get("output"):
            text = (
                f"system\nYou are a helpful, knowledgeable assistant. "
                "Answer thoroughly and clearly.\n"
                f"user\n{item['instruction']}\n"
                f"assistant\n{item['output']}"
            )
            formatted.append({"text": text})

    print(f"Formatted {len(formatted)} valid samples")
    return Dataset.from_list(formatted)


def main():
    print("=" * 60)
    print(f"Student: {config.STUDENT_MODEL_ID}")
    print(f"Dataset: {config.PROCESSED_DATASET_FILE}")
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
    print("Loading base model (4-bit)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.STUDENT_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
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
    train_dataset = load_and_format_dataset(config.PROCESSED_DATASET_FILE)

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