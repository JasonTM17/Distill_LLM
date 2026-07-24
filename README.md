# Distill GPT-5.5-xhigh → Qwen2.5-1.5B

**Knowledge Distillation** từ GPT-5.5-xhigh (9Router API) sang Qwen2.5-1.5B-Instruct, chạy local trên RTX 3060 6GB VRAM.

## Kết quả

| Metric | Before | After |
|--------|--------|-------|
| Training Loss | 1.43 | **1.14** ↓20% |
| Token Accuracy | 66.0% | **72.6%** |
| Perplexity | — | **4.70** (Excellent) |

## Kiến trúc

```
GPT-5.5-xhigh ──9Router API──▶ teacher_outputs.json ──▶ QLoRA Train ──▶ Merged Model
         (teacher)              (300 samples, 933K tokens)     (Qwen2.5-1.5B)
```

- **Teacher:** `cx/gpt-5.5-xhigh` (9Router, 2048 max tokens)
- **Student:** `Qwen2.5-1.5B-Instruct` (Alibaba, 4-bit NF4)
- **GPU:** RTX 3060 Laptop 6GB VRAM
- **Framework:** PyTorch + Transformers + PEFT (LoRA) + BitsAndBytes

## Quick Start

```bash
# Test connection
python test_connection.py

# Generate data from teacher (resumable, ~14s/prompt)
python gen_batch.py

# Format for training
python format_dataset.py

# Train (3 epochs, ~8 min for 200 samples)
python train_student.py

# Evaluate
python evaluate.py
python test_model.py

# Chat
python chat.py
```

## Hyperparameters

| Param | Value | Note |
|-------|-------|------|
| LoRA r | 16 | |
| LoRA alpha | 32 | |
| Learning rate | 2e-4 | |
| Batch | 1 + grad_accum 8 | vừa 6GB VRAM |
| Max seq len | 512 | |
| Epochs | 3 | |
| Quant | NF4, float16 | bfloat16 không hỗ trợ RTX 3060 |
| Optimizer | AdamW 8-bit | |

## Project Structure

```
distill-gpt55/
├── config.py              # Central config
├── gen_batch.py           # API data generation (resumable)
├── format_dataset.py      # Convert to chat template
├── train_student.py       # QLoRA training
├── evaluate.py            # Perplexity evaluation
├── test_model.py          # Quick inference test
├── chat.py                # Interactive chat
├── test_connection.py     # Verify 9Router API
├── data/
│   ├── prompts.json       # 530 prompts (10 categories)
│   ├── raw/               # Teacher outputs
│   └── processed/         # Training dataset
├── checkpoints/
│   ├── adapter/           # LoRA weights
│   └── merged/            # Final model (1.5GB)
└── docs/                  # Project documentation
```

## Pipeline Stages

1. **Generate:** API calls GPT-5.5-xhigh, saves after each prompt
2. **Format:** Convert to Qwen `<|im_start|>` chat template
3. **Train:** QLoRA 4-bit + LoRA rank 16, 3 epochs
4. **Merge:** Combine adapter with base model
5. **Evaluate:** Perplexity + qualitative tests

## Constraints

- **6GB VRAM** → mandatory QLoRA 4-bit
- **Python 3.14** → requires nightly PyTorch CUDA 12.8 build
- **Windows symlink limitation** → model cache in `D:/models/`
- **API:** 9Router localhost:20128 must be running
