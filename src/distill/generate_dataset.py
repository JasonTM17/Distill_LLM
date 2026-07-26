"""Generate teacher outputs for every prompt, resumably and safely.

Key differences from the original ``gen_batch.py``:

* failures are retried across runs instead of being cached as permanent,
* the output file is written atomically (no truncated JSON if killed mid-write),
* progress and per-category statistics are reported,
* validation rejects empty/short/mojibake completions.

Usage::

    python -m distill.generate_dataset                # fill every missing prompt
    python -m distill.generate_dataset --retry-failed # also re-attempt failures
    python -m distill.generate_dataset --limit 20     # cap this run
    python -m distill.generate_dataset --categories philosophy health
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import config
from .logging_utils import get_logger
from .teacher_client import TeacherClient, TeacherError

logger = get_logger("generate")


def load_prompts(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the prompt catalogue and validate its shape."""
    path = path or config.PROMPTS_FILE
    with open(path, "r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    seen: set[int] = set()
    for prompt in prompts:
        missing = {"id", "category", "instruction"} - set(prompt)
        if missing:
            raise ValueError(f"prompt {prompt.get('id')!r} missing keys: {sorted(missing)}")
        if prompt["id"] in seen:
            raise ValueError(f"duplicate prompt id: {prompt['id']}")
        seen.add(prompt["id"])
    return prompts


def load_existing(path: Path | None = None) -> dict[int, dict[str, Any]]:
    """Load previously generated outputs keyed by prompt id."""
    path = path or config.TEACHER_OUTPUT_FILE
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        logger.error("%s is corrupt; starting from scratch", path)
        return {}
    return {item["id"]: item for item in payload.get("data", []) if "id" in item}


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON to ``path`` via a temp file + rename so it is never partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def save_outputs(records: dict[int, dict[str, Any]], path: Path | None = None) -> None:
    """Persist all records sorted by id with aggregate metadata."""
    path = path or config.TEACHER_OUTPUT_FILE
    ordered = [records[key] for key in sorted(records)]
    successes = [r for r in ordered if r.get("success")]
    atomic_write_json(
        path,
        {
            "teacher_model": config.TEACHER_MODEL,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total_samples": len(ordered),
            "successful_samples": len(successes),
            "total_tokens": sum(r.get("tokens_used", 0) for r in successes),
            "data": ordered,
        },
    )


def select_pending(
    prompts: Iterable[dict[str, Any]],
    existing: dict[int, dict[str, Any]],
    *,
    retry_failed: bool = True,
    categories: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return prompts still needing a successful teacher completion."""
    pending = []
    for prompt in prompts:
        if categories and prompt["category"] not in categories:
            continue
        record = existing.get(prompt["id"])
        if record is None:
            pending.append(prompt)
        elif not record.get("success"):
            if retry_failed:
                pending.append(prompt)
        elif not (record.get("output") or "").strip():
            pending.append(prompt)
    return pending


def _record(prompt: dict[str, Any], response: Any) -> dict[str, Any]:
    return {
        "id": prompt["id"],
        "category": prompt["category"],
        "instruction": prompt["instruction"],
        "output": response.text,
        "tokens_used": response.total_tokens,
        "completion_tokens": response.completion_tokens,
        "finish_reason": response.finish_reason,
        "truncated": response.truncated,
        "teacher_model": response.model,
        "attempts": response.attempts,
        "success": True,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _failure_record(prompt: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "id": prompt["id"],
        "category": prompt["category"],
        "instruction": prompt["instruction"],
        "output": "",
        "tokens_used": 0,
        "success": False,
        "error": error[:300],
        "attempted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run(
    *,
    limit: int | None = None,
    retry_failed: bool = True,
    categories: set[str] | None = None,
    client: TeacherClient | None = None,
    delay: float | None = None,
) -> dict[str, Any]:
    """Execute the generation loop and return a summary dict."""
    config.ensure_directories()
    prompts = load_prompts()
    records = load_existing()
    pending = select_pending(
        prompts, records, retry_failed=retry_failed, categories=categories
    )
    if limit is not None:
        pending = pending[:limit]

    already_ok = sum(1 for r in records.values() if r.get("success"))
    logger.info(
        "prompts=%d already_successful=%d pending=%d",
        len(prompts),
        already_ok,
        len(pending),
    )
    if not pending:
        logger.info("nothing to generate")
        return {"generated": 0, "failed": 0, "successful_total": already_ok}

    client = client or TeacherClient()
    request_delay = config.REQUEST_DELAY if delay is None else delay

    generated = 0
    failed = 0
    started = time.time()

    for index, prompt in enumerate(pending, start=1):
        preview = prompt["instruction"][:60].replace("\n", " ")
        try:
            response = client.complete(
                prompt["instruction"], system_prompt=config.SYSTEM_PROMPT
            )
            records[prompt["id"]] = _record(prompt, response)
            generated += 1
            logger.info(
                "[%d/%d] id=%s %s | OK %d tok%s",
                index,
                len(pending),
                prompt["id"],
                preview,
                response.total_tokens,
                " (truncated)" if response.truncated else "",
            )
        except TeacherError as exc:
            records[prompt["id"]] = _failure_record(prompt, str(exc))
            failed += 1
            logger.error("[%d/%d] id=%s FAILED: %s", index, len(pending), prompt["id"], exc)
        except KeyboardInterrupt:
            logger.warning("interrupted by user; saving progress")
            save_outputs(records)
            raise

        save_outputs(records)
        if index < len(pending) and request_delay > 0:
            time.sleep(request_delay)

    save_outputs(records)
    successful_total = sum(1 for r in records.values() if r.get("success"))
    elapsed = time.time() - started
    logger.info(
        "done in %.1fs | generated=%d failed=%d successful_total=%d/%d",
        elapsed,
        generated,
        failed,
        successful_total,
        len(prompts),
    )

    by_cat = Counter(
        r["category"] for r in records.values() if r.get("success")
    )
    for category in sorted(by_cat):
        logger.info("  %-12s %d", category, by_cat[category])

    return {
        "generated": generated,
        "failed": failed,
        "successful_total": successful_total,
        "elapsed_seconds": round(elapsed, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate teacher outputs")
    parser.add_argument("--limit", type=int, default=None, help="max prompts this run")
    parser.add_argument(
        "--no-retry-failed",
        action="store_true",
        help="skip prompts that previously failed",
    )
    parser.add_argument(
        "--categories", nargs="*", default=None, help="restrict to these categories"
    )
    parser.add_argument(
        "--delay", type=float, default=None, help="seconds between requests"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(
        limit=args.limit,
        retry_failed=not args.no_retry_failed,
        categories=set(args.categories) if args.categories else None,
        delay=args.delay,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
