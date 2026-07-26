# Distill GPT-5.5-xhigh → Qwen2.5-1.5B

**Knowledge Distillation** từ GPT-5.5-xhigh (9Router API) sang Qwen2.5-1.5B-Instruct, chạy local trên RTX 3060 6GB VRAM.

## Kết quả

### v0.4 (current — held-out evaluation)

| Metric | v0.3 (in-sample, 200) | **v0.4 (held-out, 395)** | Direction |
|--------|----------------------|--------------------------|-----------|
| Perplexity (held-out) | 4.70 (in-sample, misleading) | **6.93** (real test) | Honest eval |
| Training Loss (final) | 1.14 | 1.44 | +0.30 |
| Token Accuracy | 72.6% | 65.3% | -7.3pp |
| Train samples | 200 | **357** | +78% |
| Test samples | 0 (none) | **38 stratified** | Real eval |

**Best categories (held-out PPL):** math 3.61, coding 4.66, science 5.67
**Weakest:** creative 14.39 (needs more data)

See [evaluation-v04.md](plans/reports/evaluation-v04.md) for full report.

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

# Format for training (creates train/test split)
python format_dataset.py

# Train (3 epochs, ~22 min for 357 samples on RTX 3060 6GB)
python train_student.py

# Evaluate
python evaluate.py            # Perplexity on held-out test set
python evaluate_extended.py   # ROUGE-L vs teacher on test set
python test_model.py          # Quick inference smoke test

# Chat
python chat.py
```

**Note:** If `train_student.py` fails with "paging file is too small" on Windows,
the script auto-pre-touches safetensors to load into OS cache. Ensure C: drive
has at least 2 GB free (pagefile growth target).

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
│   └── processed/         # Train + test splits (stratified 90/10)
├── checkpoints/
│   ├── adapter/           # LoRA weights (current run)
│   ├── v0.3_adapter/      # Archived v0.3 weights
│   └── merged/            # Final model (1.6GB)
└── docs/                  # Project documentation
```

## Pipeline Stages

1. **Generate:** API calls GPT-5.5-xhigh, saves after each prompt (resumable)
2. **Format:** Convert to Qwen chat template + stratified train/test split
3. **Train:** QLoRA 4-bit + LoRA rank 16, 3 epochs (357 samples)
4. **Merge:** Combine adapter with base model
5. **Evaluate:** Perplexity on held-out (38 samples) + ROUGE-L vs teacher

## Constraints

- **6GB VRAM** → mandatory QLoRA 4-bit
- **Python 3.14** → requires nightly PyTorch CUDA 12.8 build
- **Windows symlink limitation** → model cache in `D:/models/`
- **API:** 9Router localhost:20128 must be running
