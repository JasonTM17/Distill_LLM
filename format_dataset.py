"""Format teacher outputs into Qwen chat template + stratified train/test split.

Outputs:
- data/processed/dataset_train.json (90%)
- data/processed/dataset_test.json  (10%, stratified by category)
"""
import json
import math
import os
import random
import config


def load_valid_samples():
    with open(config.TEACHER_OUTPUT_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for item in raw["data"]:
        if not item.get("success") or not item.get("output", "").strip():
            continue
        samples.append({
            "id": item["id"],
            "category": item.get("category", "unknown"),
            "instruction": item["instruction"],
            "output": item["output"],
        })
    return samples


def format_chat(item):
    """Convert to Qwen chat template format."""
    text = (
        "system\n"
        "You are a helpful, knowledgeable assistant. Answer thoroughly and clearly.\n"
        "user\n"
        f"{item['instruction']}\n"
        "assistant\n"
        f"{item['output']}"
    )
    return {"text": text, "category": item["category"], "id": item["id"]}


def stratified_split(samples, test_ratio=0.1, seed=42):
    """Split samples by category. Returns (train, test)."""
    rng = random.Random(seed)
    by_cat = {}
    for s in samples:
        by_cat.setdefault(s["category"], []).append(s)

    train, test = [], []
    for cat, items in by_cat.items():
        rng.shuffle(items)
        n_test = max(1, math.floor(len(items) * test_ratio)) if len(items) >= 10 else 0
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    return train, test


def main():
    samples = load_valid_samples()
    print(f"Loaded {len(samples)} valid samples from {config.TEACHER_OUTPUT_FILE}")

    train, test = stratified_split(samples, test_ratio=config.TEST_SPLIT_RATIO)
    print(f"Split: train={len(train)}, test={len(test)}")

    train_fmt = [format_chat(s) for s in train]
    test_fmt = [format_chat(s) for s in test]

    os.makedirs(os.path.dirname(config.PROCESSED_DATASET_FILE), exist_ok=True)
    with open(config.PROCESSED_DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(train_fmt, f, indent=2, ensure_ascii=False)
    print(f"Saved train: {config.PROCESSED_DATASET_FILE}")

    with open(config.PROCESSED_TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(test_fmt, f, indent=2, ensure_ascii=False)
    print(f"Saved test:  {config.PROCESSED_TEST_FILE}")

    # Per-category counts
    from collections import Counter
    train_cats = Counter(s["category"] for s in train_fmt)
    test_cats = Counter(s["category"] for s in test_fmt)
    print("\nPer-category distribution:")
    for cat in sorted(set(train_cats) | set(test_cats)):
        print(f"  {cat:12s}: train={train_cats.get(cat, 0):3d}, test={test_cats.get(cat, 0):2d}")

    # Sample preview
    if train_fmt:
        print(f"\nSample train (first 200 chars):\n{train_fmt[0]['text'][:200]}...")


if __name__ == "__main__":
    main()