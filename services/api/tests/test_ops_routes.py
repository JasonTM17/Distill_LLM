"""Tests for /healthz, /readyz, /metrics."""

from fastapi.testclient import TestClient

from app.main import create_app

from conftest import FakeRuntime


def test_healthz_always_ok(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_ready(client):
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["model"] == "fake-model"


def test_readyz_503_while_loading():
    app = create_app(runtime=FakeRuntime(ready=False))
    with TestClient(app) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "loading"


def test_metrics_exposes_prometheus_text(client):
    client.get("/healthz")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
