"""Shared fixtures: app wired with a fake model runtime (no llama.cpp needed)."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import create_app  # noqa: E402
from app.rate_limit import SlidingWindowRateLimiter  # noqa: E402


class FakeRuntime:
    """Deterministic runtime standing in for llama.cpp in tests."""

    model_id = "fake-model"

    def __init__(self, ready: bool = True, tokens: list[str] | None = None):
        self.ready = ready
        self.load_error = None
        self.tokens = tokens or ["Hello", " world", "!"]
        self.calls: list[dict] = []

    def generate(self, messages, params):
        from app.model_runtime import GenerationResult, RuntimeNotReady

        if not self.ready:
            raise RuntimeNotReady("model still loading")
        self.calls.append({"messages": messages, "params": params})
        return GenerationResult(
            text="".join(self.tokens),
            prompt_tokens=7,
            completion_tokens=len(self.tokens),
            finish_reason="stop",
        )

    def stream(self, messages, params):
        from app.model_runtime import RuntimeNotReady

        if not self.ready:
            raise RuntimeNotReady("model still loading")
        self.calls.append({"messages": messages, "params": params})
        for token in self.tokens:
            yield token, None
        yield None, "stop"


@pytest.fixture
def fake_runtime():
    return FakeRuntime()


@pytest.fixture
def client(fake_runtime):
    app = create_app(runtime=fake_runtime)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def strict_client(fake_runtime):
    """Client with a 2-requests-per-minute limiter for rate-limit tests."""
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    app = create_app(runtime=fake_runtime, rate_limiter=limiter)
    with TestClient(app) as test_client:
        yield test_client
