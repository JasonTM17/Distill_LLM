"""Test connection to 9Router - english only for clean output."""
from openai import OpenAI
import sys
import io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_URL = "http://127.0.0.1:20128/v1"
API_KEY = "sk-59be692bbb02885c-kfjrks-07c55700"
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

MODELS_TO_TEST = [
    "cx/gpt-5.5",
    "cx/gpt-5.5-high",
    "cx/gpt-5.5-xhigh",
    "cx/gpt-5.4-mini",
    "cx/gpt-5.6-sol",
    "cx/gpt-5.6-terra",
]

PROMPT = "Explain photosynthesis in exactly 2 sentences."

for model_id in MODELS_TO_TEST:
    print("=" * 60)
    print(f"MODEL: {model_id}")
    print("=" * 60)
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=200,
            temperature=0.7
        )
        content = resp.choices[0].message.content
        usage = resp.usage
        print(f"OK | tokens={usage.total_tokens} (in={usage.prompt_tokens}, out={usage.completion_tokens})")
        print(f"Output: {content[:300]}")
    except Exception as e:
        print(f"FAIL: {e}")
    print()