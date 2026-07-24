"""Format teacher outputs into Qwen chat template for SFT training."""
import json, os
import config

with open(config.TEACHER_OUTPUT_FILE, 'r', encoding='utf-8') as f:
    raw = json.load(f)

formatted = []
skipped = 0

for item in raw["data"]:
    if not item.get("success") or not item.get("output"):
        skipped += 1
        continue

    text = (
        f"<|im_start|>system\n"
        f"You are a helpful, knowledgeable assistant. Answer thoroughly and clearly.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{item['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{item['output']}<|im_end|>"
    )
    formatted.append({"text": text, "category": item.get("category", "")})

os.makedirs(os.path.dirname(config.PROCESSED_DATASET_FILE), exist_ok=True)
with open(config.PROCESSED_DATASET_FILE, 'w', encoding='utf-8') as f:
    json.dump(formatted, f, indent=2, ensure_ascii=False)

print(f"Formatted {len(formatted)} samples (skipped {skipped} failures)")
print(f"Saved to: {config.PROCESSED_DATASET_FILE}")

# Show sample
if formatted:
    print(f"\nSample (first 200 chars):\n{formatted[0]['text'][:200]}...")
