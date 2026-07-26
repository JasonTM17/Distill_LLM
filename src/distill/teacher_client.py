"""Resilient client for the OpenAI-compatible teacher API.

The v0.4 generation run lost 127/530 samples because a transient quota error was
recorded as a permanent failure and never retried. This module fixes that class
of bug by:

* classifying errors into retryable vs fatal,
* applying exponential backoff with jitter on retryable errors,
* validating that the response actually contains usable text,
* surfacing a structured result instead of a bare string.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

from . import config
from .logging_utils import get_logger

logger = get_logger(__name__)

# Substrings that indicate the request may succeed if tried again later.
_RETRYABLE_MARKERS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "quota",
    "429",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "temporarily",
    "overloaded",
    "connection",
    "econnreset",
    "server error",
    "internal error",
    "try again",
    "capacity",
    "unavailable",
)

# Substrings that indicate retrying is pointless (bad request/auth/model).
_FATAL_MARKERS = (
    "invalid api key",
    "incorrect api key",
    "unauthorized",
    "401",
    "403",
    "model not found",
    "does not exist",
    "content policy",
    "invalid_request_error",
)


class TeacherError(RuntimeError):
    """Raised when a teacher request fails permanently."""


@dataclass
class TeacherResponse:
    """Structured result of a single teacher completion."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    model: str
    attempts: int

    @property
    def truncated(self) -> bool:
        """True when the teacher hit the token ceiling mid-answer."""
        return self.finish_reason == "length"


def classify_error(exc: BaseException) -> str:
    """Return ``"retryable"`` or ``"fatal"`` for an exception from the API."""
    message = f"{type(exc).__name__}: {exc}".lower()
    for marker in _FATAL_MARKERS:
        if marker in message:
            return "fatal"
    for marker in _RETRYABLE_MARKERS:
        if marker in message:
            return "retryable"
    # Unknown errors are treated as retryable: a wasted retry is far cheaper
    # than silently dropping a training sample, which is what v0.4 did.
    return "retryable"


def backoff_delay(attempt: int, base: float | None = None, cap: float | None = None) -> float:
    """Exponential backoff with full jitter for ``attempt`` (1-indexed)."""
    base = config.RETRY_BACKOFF_BASE if base is None else base
    cap = config.RETRY_BACKOFF_MAX if cap is None else cap
    raw = min(cap, base * (2 ** max(0, attempt - 1)))
    return random.uniform(raw * 0.5, raw)


def validate_output(text: str | None, min_chars: int | None = None) -> str:
    """Return cleaned text or raise :class:`TeacherError` when unusable.

    An empty/whitespace/too-short body means the sample would poison training,
    so it is treated as a failure and retried rather than being written out.
    """
    min_chars = config.MIN_OUTPUT_CHARS if min_chars is None else min_chars
    cleaned = (text or "").strip()
    if not cleaned:
        raise TeacherError("empty completion")
    if len(cleaned) < min_chars:
        raise TeacherError(f"completion too short ({len(cleaned)} < {min_chars} chars)")
    if "\ufffd" in cleaned:
        raise TeacherError("completion contains replacement characters (encoding damage)")
    return cleaned


class TeacherClient:
    """Thin retrying wrapper over an OpenAI-compatible chat completions API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or config.TEACHER_MODEL
        self.timeout = timeout if timeout is not None else config.TEACHER_TIMEOUT
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI  # imported lazily so tests can inject a fake

            self._client = OpenAI(
                base_url=base_url or config.API_BASE_URL,
                api_key=api_key or config.API_KEY or "not-needed",
                timeout=self.timeout,
            )

    def complete(
        self,
        instruction: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
        max_retries: int | None = None,
        sleep: Any = time.sleep,
    ) -> TeacherResponse:
        """Request one completion, retrying transient failures.

        Raises :class:`TeacherError` when all attempts are exhausted or when the
        failure is classified as fatal.
        """
        max_retries = config.MAX_RETRIES if max_retries is None else max_retries
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": instruction})

        last_error: BaseException | None = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = self._client.chat.completions.create(
                    model=model or self.model,
                    messages=messages,
                    max_tokens=max_tokens or config.TEACHER_MAX_TOKENS,
                    temperature=(
                        config.TEACHER_TEMPERATURE if temperature is None else temperature
                    ),
                )
                choice = raw.choices[0]
                text = validate_output(choice.message.content)
                usage = getattr(raw, "usage", None)
                return TeacherResponse(
                    text=text,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    total_tokens=getattr(usage, "total_tokens", 0) or 0,
                    finish_reason=getattr(choice, "finish_reason", "") or "",
                    model=getattr(raw, "model", model or self.model),
                    attempts=attempt,
                )
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                last_error = exc
                kind = classify_error(exc)
                if kind == "fatal":
                    raise TeacherError(f"fatal teacher error: {exc}") from exc
                if attempt >= max_retries:
                    break
                delay = backoff_delay(attempt)
                logger.warning(
                    "attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt,
                    max_retries,
                    str(exc)[:120],
                    delay,
                )
                sleep(delay)

        raise TeacherError(
            f"exhausted {max_retries} attempts; last error: {last_error}"
        ) from last_error
