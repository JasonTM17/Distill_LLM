"""Prometheus metrics shared across routes and middleware."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

REGISTRY = CollectorRegistry()

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests",
    labelnames=("route", "method", "status"),
    registry=REGISTRY,
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    labelnames=("route", "method"),
    registry=REGISTRY,
)
GENERATED_TOKENS = Counter(
    "generated_tokens_total",
    "Completion tokens produced",
    registry=REGISTRY,
)
