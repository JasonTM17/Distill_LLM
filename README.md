# Distill GPT-5.5-xhigh → Qwen2.5-3B

Dùng **knowledge distillation** (chưng cất mô hình) chuyển kiến thức từ GPT-5.5-xhigh qua Qwen2.5-3B-Instruct, chạy local trên RTX 3060 6GB.

## Kiến trúc

```
Phase 1: GEN DATA ──→ Phase 2: TRAIN ──→ Phase 3: EVAL
  (API 9Router)       (QLoRA local)      (Chat test)
```

- **Teacher:** `cx/gpt-5.5-xhigh` qua 9Router API
- **Student:** `Qwen/Qwen2.5-3B-Instruct` (QLoRA 4-bit)
- **GPU:** NVIDIA RTX 3060 Laptop 6GB VRAM
- **Framework:** PyTorch + transformers + PEFT + bitsandbytes

## Cài đặt

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers datasets accelerate peft trl bitsandbytes openai tqdm
```

## Cách dùng

### 1. Cấu hình
Sửa API key và paths trong `config.py`.

### 2. Test kết nối
```bash
python test_connection.py
```

### 3. Thu thập dataset (kéo dài)
```bash
python generate_data.py
```
Gọi API 530 lần, mỗi lần ~14s, tổng ~2 tiếng.

### 4. QLoRA fine-tune
```bash
python train_student.py
```
3 epochs, batch 2 + grad accum 4, chạy ~10-15 phút.

### 5. Chat test
```bash
python chat.py
```

## Hyperparameters

| Param | Giá trị |
|-------|---------|
| LoRA r | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 |
| Batch size | 2 |
| Max seq length | 1024 |
| Epochs | 3 |
| Quantization | NF4, double quant |

## Project structure

```
distill-gpt55/
├── config.py              # Central configuration
├── generate_data.py       # Collect teacher outputs from API
├── train_student.py       # QLoRA fine-tune
├── chat.py                # Interactive inference test
├── test_connection.py     # Verify 9Router API
├── data/
│   ├── prompts.json       # 530 diverse training prompts
│   ├── raw/               # Teacher outputs
│   └── processed/         # Formatted for training
├── checkpoints/           # Model checkpoints
└── README.md
```

## Credits

- Teacher: GPT-5.5-xhigh via 9Router
- Student: Qwen2.5-3B-Instruct by Alibaba
- Quantization: BitsAndBytes + PEFT
