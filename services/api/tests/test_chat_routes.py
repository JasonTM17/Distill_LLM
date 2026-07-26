"""Tests for the OpenAI-compatible chat endpoint."""

import json

from fastapi.testclient import TestClient

from app.main import create_app

from conftest import FakeRuntime


def _payload(**overrides):
    payload = {"messages": [{"role": "user", "content": "Say hello"}]}
    payload.update(overrides)
    return payload


def test_chat_completion_happy_path(client, fake_runtime):
    response = client.post("/v1/chat/completions", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello world!"
    assert body["usage"]["total_tokens"] == 10
    assert fake_runtime.calls[0]["messages"][0]["role"] == "user"


def test_chat_completion_applies_defaults(client, fake_runtime):
    client.post("/v1/chat/completions", json=_payload())
    params = fake_runtime.calls[0]["params"]
    assert params.max_tokens == 512
    assert params.temperature == 0.7


def test_rejects_empty_messages(client):
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 422


def test_rejects_assistant_last_message(client):
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "assistant", "content": "I speak first"}]},
    )
    assert response.status_code == 422


def test_rejects_oversized_max_tokens(client):
    response = client.post("/v1/chat/completions", json=_payload(max_tokens=999_999))
    assert response.status_code == 422


def test_503_when_model_not_ready():
    app = create_app(runtime=FakeRuntime(ready=False))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=_payload())
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "model_not_ready"


def test_streaming_emits_sse_chunks_and_done(client):
    with client.stream(
        "POST", "/v1/chat/completions", json=_payload(stream=True)
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]

    assert lines[-1] == "data: [DONE]"
    events = [json.loads(line[len("data: ") :]) for line in lines[:-1]]
    contents = [
        event["choices"][0]["delta"].get("content", "")
        for event in events
        if event["choices"][0]["delta"]
    ]
    assert "".join(contents) == "Hello world!"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


def test_rate_limit_returns_429(strict_client):
    for _ in range(2):
        assert strict_client.post("/v1/chat/completions", json=_payload()).status_code == 200
    response = strict_client.post("/v1/chat/completions", json=_payload())
    assert response.status_code == 429
    assert "Retry-After" in response.headers
