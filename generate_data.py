"""Generate training dataset by calling 9Router teacher model (GPT-5.5-xhigh)."""
import sys
import io
import json
import time
import os
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from openai import OpenAI
from tqdm import tqdm

import config


client = OpenAI(
    base_url=config.API_BASE_URL,
    api_key=config.API_KEY,
)


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def call_teacher(instruction, max_retries=config.MAX_RETRIES):
    """Call 9Router teacher and return (output, tokens_used) or (None, 0) on failure."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=config.TEACHER_MODEL,
                messages=[{"role": "user", "content": instruction}],
                max_tokens=config.TEACHER_MAX_TOKENS,
                temperature=config.TEACHER_TEMPERATURE,
            )
            content = resp.choices[0].message.content
            tokens = resp.usage.total_tokens if resp.usage else 0
            return content, tokens
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [Retry {attempt+1}/{max_retries}] Error: {e} — waiting {wait}s")
            time.sleep(wait)
    return None, 0


def save_checkpoint(results, path, total_tokens):
    """Save partial results so we don't lose progress."""
    checkpoint = {
        "teacher_model": config.TEACHER_MODEL,
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(results),
        "total_tokens": total_tokens,
        "data": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


def main():
    prompts = load_prompts(config.PROMPTS_FILE)
    total_prompts = len(prompts)
    print(f"Loaded {total_prompts} prompts from {config.PROMPTS_FILE}")
    print(f"Teacher: {config.TEACHER_MODEL}")
    print()

    results = []
    total_tokens = 0
    start_time = time.time()

    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        instruction = prompt["instruction"]
        output, tokens = call_teacher(instruction)

        results.append({
            "id": prompt["id"],
            "category": prompt["category"],
            "instruction": instruction,
            "output": output if output else "",
            "tokens_used": tokens,
            "success": output is not None,
        })

        if output:
            total_tokens += tokens

        # Save checkpoint periodically
        if (i + 1) % config.CHECKPOINT_EVERY == 0:
            save_checkpoint(results, config.TEACHER_OUTPUT_FILE, total_tokens)

        time.sleep(config.REQUEST_DELAY)

    # Final save
    save_checkpoint(results, config.TEACHER_OUTPUT_FILE, total_tokens)

    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r["success"])
    print()
    print("=" * 60)
    print(f"Done! {success_count}/{total_prompts} successful")
    print(f"Total tokens used: {total_tokens}")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/total_prompts:.1f}s per prompt)")
    print(f"Saved to: {config.TEACHER_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
