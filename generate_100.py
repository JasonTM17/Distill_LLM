"""Generate teacher outputs - batch of 100 prompts."""
import sys, io, json, time, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from openai import OpenAI
import config

client = OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)
BATCH_SAVE_EVERY = 10
TOTAL = 100

with open(config.PROMPTS_FILE, 'r', encoding='utf-8') as f:
    prompts = json.load(f)[:TOTAL]

# Load existing
existing_ids = set()
results = []
total_tokens = 0
if os.path.exists(config.TEACHER_OUTPUT_FILE):
    with open(config.TEACHER_OUTPUT_FILE, 'r', encoding='utf-8') as f:
        prev = json.load(f)
        for item in prev["data"]:
            if item.get("success"):
                existing_ids.add(item["id"])
                results.append(item)
                total_tokens += item.get("tokens_used", 0)

print(f"Resuming: {len(existing_ids)}/{TOTAL} already done, {total_tokens} tokens")

for p in prompts:
    if p["id"] in existing_ids:
        continue
    idx = len(results) + 1
    print(f"[{idx}/{TOTAL}] {p['instruction'][:70]}...", end=" ", flush=True)
    try:
        resp = client.chat.completions.create(
            model=config.TEACHER_MODEL,
            messages=[{"role": "user", "content": p["instruction"]}],
            max_tokens=config.TEACHER_MAX_TOKENS,
            temperature=config.TEACHER_TEMPERATURE,
        )
        output = resp.choices[0].message.content
        tokens = resp.usage.total_tokens
        results.append({"id": p["id"], "category": p["category"], "instruction": p["instruction"], "output": output, "tokens_used": tokens, "success": True})
        total_tokens += tokens
        existing_ids.add(p["id"])
        print(f"OK ({tokens}t)")
    except Exception as e:
        results.append({"id": p["id"], "category": p["category"], "instruction": p["instruction"], "output": "", "tokens_used": 0, "success": False, "error": str(e)})
        print(f"FAIL: {e}")

    if idx % BATCH_SAVE_EVERY == 0:
        with open(config.TEACHER_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump({"teacher_model": config.TEACHER_MODEL, "total_samples": len(results), "total_tokens": total_tokens, "data": results}, f, indent=2, ensure_ascii=False)
        print(f"  [Saved: {len(existing_ids)}/{TOTAL}]")

    time.sleep(config.REQUEST_DELAY)

with open(config.TEACHER_OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump({"teacher_model": config.TEACHER_MODEL, "total_samples": len(results), "total_tokens": total_tokens, "data": results}, f, indent=2, ensure_ascii=False)

success = sum(1 for r in results if r.get("success"))
print(f"\nCOMPLETE: {success}/{TOTAL}, {total_tokens} tokens")
