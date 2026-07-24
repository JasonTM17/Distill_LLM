# System Architecture

## Pipeline Overview

```
  ┌──────────────┐     ┌───────────────┐     ┌───────────────┐     ┌─────────────┐
  │  prompts.json │────▶│ 9Router API   │────▶│   QLoRA Train │────▶│   Merged    │
  │    (530)      │     │ cx/gpt-5.5    │     │ Qwen2.5-1.5B │     │   Model     │
  └──────────────┘     └───────┬───────┘     └───────┬───────┘     └──────┬──────┘
                               │                     │                    │
                               ▼                     ▼                    ▼
                       raw/teacher_outputs   checkpoints/adapter    chat.py / CLI
                            (JSON)            (LoRA weights)       interactive test
```

## Component Detail

### 1. Data Generation (`gen_batch.py`)
- **Input:** `data/prompts.json` — 530 instructions phân bố đều qua 10 categories
- **API Call:** OpenAI-compatible POST đến `127.0.0.1:20128/v1`
- **Model:** `cx/gpt-5.5-xhigh`, max_tokens=2048, temperature=0.7
- **Output:** `data/raw/teacher_outputs.json` — list [{instruction, output, tokens, success}]
- **Resilience:** Save sau mỗi prompt, retry 3 lần, delay 1.5s

### 2. Dataset Formatting (`format_dataset.py`)
- **Input:** `data/raw/teacher_outputs.json`
- **Transform:** Convert thành Qwen `<|im_start|>system/user/assistant<|im_end|>` chat template
- **Output:** `data/processed/dataset_train.json`

### 3. QLoRA Training (`train_student.py`)
- **Base Model:** Qwen2.5-1.5B-Instruct (local `D:/models/qwen15-1.5b`)
- **Quantization:** BitsAndBytes 4-bit NF4, double quantization, float16 compute
- **LoRA:** rank=16, alpha=32, target=q/k/v/o projections, dropout=0.05
- **Training:** SFTTrainer (TRL), 3 epochs, batch=1, grad_accum=8, lr=2e-4
- **Output:** `checkpoints/adapter/` (LoRA weights) → merge thành `checkpoints/merged/`

### 4. Evaluation (`evaluate.py`, `test_model.py`)
- **Perplexity:** Compute trên 20 held-out samples → PPL=4.70
- **Quality test:** 3 diverse prompts (code, knowledge, reasoning)

### 5. Inference (`chat.py`, `test_model.py`)
- **Load:** 4-bit merged model → ~3.5GB VRAM
- **Generate:** temperature=0.7, top_p=0.9, max_new_tokens=200-512

## Data Flow

```
prompts.json (530)
  ↓ gen_batch.py ── 9Router API ── teacher_outputs.json
  ↓ format_dataset.py
dataset_train.json
  ↓ train_student.py ── Qwen2.5-1.5B + QLoRA
checkpoints/merged/
  ↓ chat.py / test_model.py
User interaction
```

## Directory Layout

```
distill-gpt55/
├── config.py              ← Trung tâm cấu hình
├── gen_batch.py           ← Sinh dataset (resumable)
├── format_dataset.py      ← Convert sang chat template
├── train_student.py       ← QLoRA training
├── evaluate.py            ← Đánh giá perplexity
├── test_model.py           ← Test nhanh inference
├── chat.py                ← Interactive chat
├── test_connection.py     ← Verify 9Router
├── download_student.py    ← Tải model từ HF
├── data/
│   ├── prompts.json       ← 530 prompt mẫu
│   ├── raw/               ← Teacher outputs
│   └── processed/         ← Dataset formatted
├── checkpoints/
│   ├── adapter/           ← LoRA weights
│   └── merged/            ← Merged model ready
├── docs/                  ← Project documentation
└── D:/models/
    ├── qwen15-1.5b/       ← Base model (3GB)
    └── qwen25-3b/         ← Base model (5.7GB, chưa dùng)
```
