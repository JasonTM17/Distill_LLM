"""Generate teacher outputs — saves after every sample. Resumable."""
import sys, io, json, time, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from openai import OpenAI
import config

client = OpenAI(base_url=config.API_BASE_URL, api_key=config.API_KEY)
TARGET = 100

with open(config.PROMPTS_FILE, 'r', encoding='utf-8') as f:
    prompts = json.load(f)

existing = {}
if os.path.exists(config.TEACHER_OUTPUT_FILE):
    with open(config.TEACHER_OUTPUT_FILE, 'r', encoding='utf-8') as f:
        try:
            prev = json.load(f)
            for item in prev["data"]:
                existing[item["id"]] = item
        except:
            existing = {}

results = list(existing.values())
total_tokens = sum(r.get("tokens_used", 0) for r in results if r.get("success"))
ok = sum(1 for r in results if r.get("success"))

def save():
    with open(config.TEACHER_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"teacher_model": config.TEACHER_MODEL, "total_samples": len(results), "total_tokens": total_tokens, "data": results}, f, indent=2, ensure_ascii=False)

print(f"Starting: {ok}/{len(prompts)} existing, target {TARGET}")

for p in prompts:
    if ok >= TARGET:
        break
    if p["id"] in existing:
        continue

    short = p["instruction"][:60].replace('\n', ' ')
    print(f"[{ok+1}/{TARGET}] {short}...", end=" ", flush=True)

    try:
        resp = client.chat.completions.create(
            model=config.TEACHER_MODEL,
            messages=[{"role": "user", "content": p["instruction"]}],
            max_tokens=config.TEACHER_MAX_TOKENS,
            temperature=config.TEACHER_TEMPERATURE,
        )
        out = resp.choices[0].message.content
        t = resp.usage.total_tokens
        entry = {"id": p["id"], "category": p["category"], "instruction": p["instruction"], "output": out, "tokens_used": t, "success": True}
        results.append(entry)
        existing[p["id"]] = entry
        total_tokens += t
        ok += 1
        print(f"OK {t}t")
    except Exception as e:
        entry = {"id": p["id"], "category": p["category"], "instruction": p["instruction"], "output": "", "tokens_used": 0, "success": False}
        results.append(entry)
        existing[p["id"]] = entry
        print(f"FAIL: {str(e)[:80]}")

    save()
    sys.stdout.flush()
    time.sleep(config.REQUEST_DELAY)

save()
print(f"\nDone: {ok}/{TARGET}, {total_tokens} tokens")
