"""Unit tests for distill.teacher_client: error classes, backoff, retry loop."""

from types import SimpleNamespace

import pytest

from distill.teacher_client import (
    TeacherClient,
    TeacherError,
    backoff_delay,
    classify_error,
    validate_output,
)


# ── classify_error ─────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message",
    ["Rate limit exceeded", "HTTP 503 Service Unavailable", "Connection error.", "Request timed out"],
)
def test_transient_errors_are_retryable(message):
    assert classify_error(Exception(message)) == "retryable"


@pytest.mark.parametrize(
    "message",
    ["401 Unauthorized", "Invalid API key provided", "model not found", "content policy violation"],
)
def test_permanent_errors_are_fatal(message):
    assert classify_error(Exception(message)) == "fatal"


def test_unknown_errors_default_to_retryable():
    assert classify_error(Exception("something nobody anticipated")) == "retryable"


# ── backoff_delay ──────────────────────────────────────────────────────────

def test_backoff_first_attempt_within_half_to_full_base():
    for _ in range(50):
        assert 2.0 <= backoff_delay(1, base=4.0, cap=120.0) <= 4.0


def test_backoff_is_capped():
    for _ in range(50):
        assert 60.0 <= backoff_delay(10, base=4.0, cap=120.0) <= 120.0


# ── validate_output ────────────────────────────────────────────────────────

def test_validate_rejects_empty_and_whitespace():
    with pytest.raises(TeacherError):
        validate_output("")
    with pytest.raises(TeacherError):
        validate_output("   \n ")
    with pytest.raises(TeacherError):
        validate_output(None)


def test_validate_rejects_too_short():
    with pytest.raises(TeacherError, match="too short"):
        validate_output("hi", min_chars=40)


def test_validate_rejects_replacement_characters():
    with pytest.raises(TeacherError, match="replacement"):
        validate_output("Vi��t m�t câu chuyện dài " + "x" * 40)


def test_validate_returns_stripped_text():
    text = "  " + "a" * 50 + "  "
    assert validate_output(text, min_chars=40) == "a" * 50


# ── TeacherClient.complete with a fake OpenAI client ───────────────────────

class _FakeCompletions:
    """Pops one scripted result (response or exception) per create() call."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        result = self._results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _fake_client(results):
    return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(results)))


def _response(text=None, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text if text is not None else "x" * 80),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        model="fake-model",
    )


def test_complete_success_first_attempt():
    fake = _fake_client([_response()])
    client = TeacherClient(client=fake)
    result = client.complete("hello", sleep=lambda _s: None)
    assert result.attempts == 1
    assert result.total_tokens == 30
    assert not result.truncated


def test_complete_retries_transient_then_succeeds():
    fake = _fake_client([Exception("Request timed out"), _response()])
    client = TeacherClient(client=fake)
    result = client.complete("hello", sleep=lambda _s: None)
    assert result.attempts == 2
    assert fake.chat.completions.calls == 2


def test_complete_fatal_error_raises_without_retry():
    fake = _fake_client([Exception("401 Unauthorized")])
    client = TeacherClient(client=fake)
    with pytest.raises(TeacherError, match="fatal"):
        client.complete("hello", sleep=lambda _s: None)
    assert fake.chat.completions.calls == 1


def test_complete_exhausts_retries():
    fake = _fake_client([Exception("timeout")] * 3)
    client = TeacherClient(client=fake)
    with pytest.raises(TeacherError, match="exhausted"):
        client.complete("hello", max_retries=3, sleep=lambda _s: None)
    assert fake.chat.completions.calls == 3


def test_complete_empty_body_is_retried():
    fake = _fake_client([_response(text=""), _response()])
    client = TeacherClient(client=fake)
    result = client.complete("hello", sleep=lambda _s: None)
    assert result.attempts == 2


def test_truncated_flag_from_length_finish_reason():
    fake = _fake_client([_response(finish_reason="length")])
    client = TeacherClient(client=fake)
    assert client.complete("hello", sleep=lambda _s: None).truncated
