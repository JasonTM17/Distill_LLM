"""Generate dataset from 9Router teacher. Resumable — saves checkpoints every N prompts."""
import sys, io, json, time, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from openai import OpenAI
import config

client = OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)
SAVE_EVERY = 10

with open(config.PROMPTS_FILE, 'r', encoding='utf-8') as f:
    prompts = json.load(f)

existing = {}
results = []
total_tokens = 0

if os.path.exists(config.TEACHER_OUTPUT_FILE):
    with open(config.TEACHER_OUTPUT_FILE, 'r', encoding='utf-8') as f:
        prev = json.load(f)
        for item in prev["data"]:
            if item.get("success"):
                existing[item["id"]] = True
                results.append(item)
                total_tokens += item.get("tokens_used", 0)

print(f"Resuming: {len(results)}/{len(prompts)} done, {total_tokens} tokens")
start_time = time.time()

for p in prompts:
    if p["id"] in existing:
        continue

    idx = len(results) + 1
    print(f"[{idx}/{len(prompts)}] {p['instruction'][:65]}...", end=" ", flush=True)

    try:
        resp = client.chat.completions.create(
            model=config.TEACHER_MODEL,
            messages=[{"role": "user", "content": p["instruction"]}],
            max_tokens=config.TEACHER_MAX_TOKENS,
            temperature=config.TEACHER_TEMPERATURE,
        )
        out = resp.choices[0].message.content
        t = resp.usage.total_tokens
        results.append({"id": p["id"], "category": p["category"], "instruction": p["instruction"], "output": out, "tokens_used": t, "success": True})
        total_tokens += t
        existing[p["id"]] = True
        print(f"OK ({t}t)")
    except Exception as e:
        results.append({"id": p["id"], "category": p["category"], "instruction": p["instruction"], "output": "", "tokens_used": 0, "success": False, "error": str(e)})
        existing[p["id"]] = False
        print(f"FAIL")

    if idx % SAVE_EVERY == 0:
        with open(config.TEACHER_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump({"teacher_model": config.TEACHER_MODEL, "total_samples": len(results), "total_tokens": total_tokens, "data": results}, f, indent=2, ensure_ascii=False)
        elapsed = time.time() - start_time
        print(f"  [Saved {idx}/{len(prompts)} | {elapsed:.0f}s]")

    time.sleep(config.REQUEST_DELAY)

with open(config.TEACHER_OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump({"teacher_model": config.TEACHER_MODEL, "total_samples": len(results), "total_tokens": total_tokens, "data": results}, f, indent=2, ensure_ascii=False)

ok = sum(1 for r in results if r.get("success"))
elapsed = time.time() - start_time
print(f"\nDone: {ok}/{len(prompts)} | {total_tokens} tokens | {elapsed:.0f}s")
