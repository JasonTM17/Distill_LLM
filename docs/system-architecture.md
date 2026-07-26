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
- **Transform:** Convert to Qwen `system/user/assistant` chat template + stratified 90/10 split
- **Output:** `data/processed/dataset_train.json` (357 samples) + `dataset_test.json` (38 samples)

### 3. QLoRA Training (`train_student.py`)
- **Base Model:** Qwen2.5-1.5B-Instruct (local `D:/models/qwen15-1.5b`)
- **Quantization:** BitsAndBytes 4-bit NF4, double quantization, float16 compute
- **LoRA:** rank=16, alpha=32, target=q/k/v/o projections, dropout=0.05
- **Training:** SFTTrainer (TRL), 3 epochs, batch=1, grad_accum=8, lr=2e-4
- **Workaround:** Pre-touch safetensors to OS cache (Windows pagefile error 1455)
- **Output:** `checkpoints/adapter/` (LoRA weights) → merge to `checkpoints/merged/`

### 4. Evaluation (`evaluate.py`, `evaluate_extended.py`)
- **Perplexity:** Compute on held-out test set (38 samples) → PPL=6.93 (Excellent)
- **ROUGE-L vs teacher:** LCS-based similarity on test set → avg=0.1337
- **Per-category PPL:** math 3.61, coding 4.66, science 5.67, creative 14.39 (weakest)

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
