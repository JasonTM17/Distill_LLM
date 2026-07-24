"""Test distilled model inference."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/merged",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("checkpoints/merged", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Loaded!", flush=True)

test_prompts = [
    "Write a Python function to check if a number is prime.",
    "Explain the difference between AI and machine learning in 2-3 sentences.",
    "What is the capital of France?",
]

for p in test_prompts:
    print(f"\n[Q] {p}", flush=True)
    messages = [
        {"role": "system", "content": "You are a helpful, knowledgeable assistant."},
        {"role": "user", "content": p},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    print(f"[A] {response}", flush=True)
    print("-" * 50, flush=True)
