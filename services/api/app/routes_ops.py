"""Operational endpoints: liveness, readiness, Prometheus metrics."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .metrics import REGISTRY

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request, response: Response) -> dict[str, str]:
    runtime = request.app.state.runtime
    if runtime.ready:
        return {"status": "ready", "model": runtime.model_id}
    response.status_code = 503
    detail = getattr(runtime, "load_error", None)
    return {"status": "loading" if not detail else "error", "detail": detail or "model loading"}


@router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
