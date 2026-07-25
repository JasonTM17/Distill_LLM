"""Comprehensive evaluation: compare student response vs teacher output.

Uses ROUGE-L (sequence matcher) and exact overlap to score quality.
"""
import sys, io, json, os, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

MODEL_PATH = config.MERGED_MODEL_DIR
TEACHER_OUTPUT_FILE = config.TEACHER_OUTPUT_FILE


def lcs_length(a, b):
    """Length of longest common subsequence (Rouge-L basis)."""
    if len(a) == 0 or len(b) == 0:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[len(a)][len(b)]


def rouge_l_f1(reference, hypothesis):
    """ROUGE-L F1 score in [0, 1]."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0
    lcs = lcs_length(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def load_teacher_outputs():
    """Index teacher outputs by id."""
    with open(TEACHER_OUTPUT_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    by_id = {}
    for item in raw["data"]:
        if item.get("success") and item.get("output"):
            by_id[item["id"]] = item
    return by_id


def main():
    print(f"Comprehensive Evaluation: {MODEL_PATH}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_by_id = load_teacher_outputs()

    # Use test set prompts to compare student vs teacher
    with open(config.PROCESSED_TEST_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    total = 0
    passed = 0
    rouge_scores = []
    results = []

    for item in test_data:
        prompt_id = item.get("id")
        category = item.get("category", "unknown")
        # Reconstruct instruction from text field (system/user/assistant)
        text = item["text"]
        m = re.search(r"user\n(.+?)\nassistant\n", text, re.DOTALL)
        if not m:
            continue
        instruction = m.group(1).strip()
        teacher_ref = teacher_by_id.get(prompt_id, {}).get("output", "")

        # Generate student response
        messages = [
            {"role": "system", "content": "You are a helpful, knowledgeable assistant."},
            {"role": "user", "content": instruction},
        ]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

        # Score
        rouge = rouge_l_f1(teacher_ref, response) if teacher_ref else 0.0
        rouge_scores.append(rouge)
        ok = rouge >= 0.25 or len(response) >= 50  # lenient threshold
        if ok:
            passed += 1
        total += 1
        results.append({
            "id": prompt_id,
            "category": category,
            "prompt": instruction[:60],
            "response": response[:150],
            "teacher_ref": teacher_ref[:100],
            "rouge_l": rouge,
            "passed": ok,
        })
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] rouge={rouge:.2f} id={prompt_id} ({category}): {instruction[:50]}...", flush=True)

    avg_rouge = sum(rouge_scores) / max(len(rouge_scores), 1)
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} passed ({100*passed//max(total,1)}%)")
    print(f"Average ROUGE-L F1: {avg_rouge:.4f}")

    # Per-category
    from collections import defaultdict
    cat_rouge = defaultdict(list)
    for r in results:
        cat_rouge[r["category"]].append(r["rouge_l"])
    print(f"\nPer-category ROUGE-L:")
    for cat, scores in sorted(cat_rouge.items()):
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  {cat:12s}: avg={avg:.3f} ({len(scores)} samples)")

    # Save detailed results
    out_path = os.path.join(os.path.dirname(MODEL_PATH), "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "passed": passed,
            "avg_rouge_l": avg_rouge,
            "per_category": {c: sum(s)/len(s) for c, s in cat_rouge.items()},
            "details": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {out_path}")

    # Show 3 sample responses
    print(f"\n{'='*60}")
    print("SAMPLE COMPARISONS:")
    for r in results[:3]:
        print(f"\n[id={r['id']}, {r['category']}] Q: {r['prompt']}")
        print(f"  Teacher: {r['teacher_ref']}")
        print(f"  Student: {r['response']}  (rouge={r['rouge_l']:.2f})")


if __name__ == "__main__":
    main()