"""FastAPI application factory for the distilled-model inference service."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .metrics import HTTP_LATENCY, HTTP_REQUESTS
from .rate_limit import SlidingWindowRateLimiter
from .routes_chat import router as chat_router
from .routes_ops import router as ops_router


def create_app(runtime=None, rate_limiter=None) -> FastAPI:
    """Build the app; pass fakes for ``runtime``/``rate_limiter`` in tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if hasattr(app.state.runtime, "start_background_load") and not app.state.runtime.ready:
            app.state.runtime.start_background_load()
        yield

    app = FastAPI(
        title="distill-gpt55 inference API",
        version="0.5.0",
        description="OpenAI-compatible chat API for the distilled Qwen2.5-1.5B model.",
        lifespan=lifespan,
    )

    if runtime is None:
        from .model_runtime import LlamaCppRuntime

        runtime = LlamaCppRuntime()
    app.state.runtime = runtime
    app.state.rate_limiter = rate_limiter or SlidingWindowRateLimiter()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ALLOW_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        HTTP_REQUESTS.labels(route_path, request.method, str(response.status_code)).inc()
        HTTP_LATENCY.labels(route_path, request.method).observe(time.perf_counter() - started)
        response.headers["x-request-id"] = request_id
        return response

    app.include_router(ops_router)
    app.include_router(chat_router)
    return app


app = create_app()
