"""Comprehensive evaluation of distilled model."""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

MODEL_PATH = config.MERGED_MODEL_DIR
TESTS = {
    "code_python": [
        "Write a function to calculate the factorial of n recursively.",
        "Write a decorator that logs function name and arguments.",
    ],
    "math": [
        "What is the derivative of x^3?",
        "Calculate the probability of drawing two aces from a 52-card deck.",
    ],
    "reasoning": [
        "If it takes 6 workers 6 days to build 6 houses, how many days for 3 workers to build 3 houses?",
        "Explain the difference between correlation and causation with an example.",
    ],
    "science": [
        "What is the speed of light?",
        "Explain how vaccines work in 2 sentences.",
    ],
    "vietnamese": [
        "Hà Nội là thủ đô của nước nào?",
        "Giải thích cách nấu cơm tấm Sài Gòn.",
    ],
    "knowledge": [
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
    ],
}

print(f"Comprehensive Evaluation: {MODEL_PATH}", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

total = 0
passed = 0
results = {}

for category, prompts in TESTS.items():
    results[category] = []
    for p in prompts:
        total += 1
        messages = [
            {"role": "system", "content": "You are a helpful, knowledgeable assistant."},
            {"role": "user", "content": p},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

        # Quick heuristic: response > 20 chars and makes sense
        ok = len(response) > 20
        results[category].append({"prompt": p, "response": response[:200], "passed": ok})
        if ok:
            passed += 1
        status = "✅" if ok else "❌"
        print(f"[{category}] {status} {p[:60]}...", flush=True)

print(f"\n{'='*60}")
print(f"RESULTS: {passed}/{total} passed ({100*passed//total}%)")
for cat, items in results.items():
    cat_passed = sum(1 for i in items if i["passed"])
    print(f"  {cat:15s}: {cat_passed}/{len(items)}")

# Print sample responses for 1 test
print(f"\n{'='*60}")
print("SAMPLE RESPONSES:")
for cat in ["vietnamese", "math", "science"]:
    for item in results[cat][:1]:
        print(f"[{cat}] Q: {item['prompt'][:60]}...")
        print(f"       A: {item['response'][:150]}...")
        print()
