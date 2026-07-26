"""Interactive terminal chat with the merged model (bf16, GPU when available).

Usage::

    python -m distill.chat
    python -m distill.chat --prompt "One-shot question"   # non-interactive
"""

from __future__ import annotations

import argparse

from . import config
from .logging_utils import configure_console, get_logger
from .model_loading import load_causal_lm, load_tokenizer

logger = get_logger("chat")


def load_model():
    logger.info("loading merged model: %s", config.MERGED_MODEL_DIR)
    model = load_causal_lm(config.MERGED_MODEL_DIR)
    tokenizer = load_tokenizer(config.MERGED_MODEL_DIR)
    return model, tokenizer


def generate_reply(model, tokenizer, messages: list[dict[str, str]]) -> str:
    import torch

    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
            temperature=config.EVAL_TEMPERATURE,
            top_p=config.EVAL_TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with the distilled model")
    parser.add_argument("--prompt", default=None, help="one-shot prompt, then exit")
    args = parser.parse_args(argv)

    configure_console()
    model, tokenizer = load_model()
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        print(generate_reply(model, tokenizer, messages))
        return 0

    print("Chat ready — type 'exit' to quit, 'reset' to clear history.")
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text or user_text.lower() == "exit":
            break
        if user_text.lower() == "reset":
            messages = messages[:1]
            print("(history cleared)")
            continue
        messages.append({"role": "user", "content": user_text})
        reply = generate_reply(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"\nAssistant: {reply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
