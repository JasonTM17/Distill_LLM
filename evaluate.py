"""Evaluate distilled model with perplexity on held-out test set."""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import config

MODEL_PATH = config.MERGED_MODEL_DIR
DATA_PATH = config.PROCESSED_TEST_FILE
BATCH_SIZE = 1
MAX_EVAL_SAMPLES = 60


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = data[:MAX_EVAL_SAMPLES]
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.data[idx]["text"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"input_ids": tokens["input_ids"].squeeze(0)}


def compute_perplexity(model, dataloader):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()


def main():
    print(f"Evaluating model: {MODEL_PATH}", flush=True)
    print(f"Test set:       {DATA_PATH}", flush=True)
    print(f"Max samples:    {MAX_EVAL_SAMPLES}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = TextDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    loss, ppl = compute_perplexity(model, dataloader)
    print(f"\n=== Overall Results ({len(dataset)} held-out samples) ===", flush=True)
    print(f"  Avg loss:      {loss:.4f}", flush=True)
    print(f"  Perplexity:    {ppl:.2f}", flush=True)

    if ppl < 10:
        grade = "Excellent"
    elif ppl < 20:
        grade = "Good"
    elif ppl < 50:
        grade = "Fair"
    else:
        grade = "Poor"
    print(f"  Grade:         {grade}", flush=True)

    # Per-category PPL
    print(f"\n=== Per-Category Breakdown ===", flush=True)
    cat_datasets = defaultdict(list)
    for item in data[:MAX_EVAL_SAMPLES]:
        cat = item.get("category", "unknown")
        cat_datasets[cat].append(item)
    for cat, items in sorted(cat_datasets.items()):
        ds = TextDataset(items, tokenizer)
        dl = DataLoader(ds, batch_size=BATCH_SIZE)
        if len(dl) > 0:
            cat_loss, cat_ppl = compute_perplexity(model, dl)
            print(f"  {cat:12s} ({len(items):2d}): loss={cat_loss:.4f}, ppl={cat_ppl:.2f}", flush=True)


if __name__ == "__main__":
    main()