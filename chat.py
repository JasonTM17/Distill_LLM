"""Interactive chat with the distilled model for evaluation."""
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def load_model(model_path=config.MERGED_MODEL_DIR):
    """Load merged student model."""
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loaded! VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=512):
    """Generate response from model."""
    messages = [
        {"role": "system", "content": "You are a helpful, knowledgeable assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response


def chat():
    """Interactive chat loop."""
    model, tokenizer = load_model()

    print("\n" + "=" * 60)
    print("Distilled Student Model Chat")
    print("Commands: /exit, /clear, /batch")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("Goodbye!")
            break

        if user_input == "/clear":
            os.system("cls" if sys.platform == "win32" else "clear")
            continue

        if user_input == "/batch":
            run_test_set(model, tokenizer)
            continue

        print("Student: ", end="", flush=True)
        response = generate(model, tokenizer, user_input)
        print(response)
        print("-" * 40)


def run_test_set(model, tokenizer):
    """Run a small test set for evaluation."""
    test_prompts = [
        "Write a Python function to calculate Fibonacci numbers.",
        "Explain what black holes are in simple terms.",
        "What is the difference between AI and machine learning?",
        "Giải thích cách nấu phở bò truyền thống.",
        "If I flip a coin 3 times, what is the probability of getting exactly 2 heads?",
    ]
    for prompt in test_prompts:
        print(f"\n[Q] {prompt}")
        print(f"[A] ", end="", flush=True)
        resp = generate(model, tokenizer, prompt, max_tokens=256)
        print(resp[:300])
        print("-" * 40)


if __name__ == "__main__":
    chat()
