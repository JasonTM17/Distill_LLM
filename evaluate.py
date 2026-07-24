"""Evaluate distilled model with perplexity on a test subset."""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import config

MODEL_PATH = config.MERGED_MODEL_DIR
DATA_PATH = config.PROCESSED_DATASET_FILE
BATCH_SIZE = 1
MAX_EVAL_SAMPLES = 20


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
    print(f"Test samples: {MAX_EVAL_SAMPLES}", flush=True)

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
    print(f"\nResults on {MAX_EVAL_SAMPLES} samples:", flush=True)
    print(f"  Avg loss:      {loss:.4f}", flush=True)
    print(f"  Perplexity:    {ppl:.2f}", flush=True)

    if ppl < 10:
        grade = "Excellent — low perplexity, well-distilled"
    elif ppl < 20:
        grade = "Good — reasonable for a 1.5B model"
    elif ppl < 50:
        grade = "Fair — may benefit from more data or epochs"
    else:
        grade = "Poor — check dataset quality or hyperparameters"
    print(f"  Grade:         {grade}", flush=True)


if __name__ == "__main__":
    main()
