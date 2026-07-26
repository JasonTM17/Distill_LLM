"""OpenAI-compatible chat completions endpoint (streaming and non-streaming)."""

from __future__ import annotations

import json
from typing import Iterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import config
from .metrics import GENERATED_TOKENS
from .model_runtime import GenerationParams, RuntimeNotReady
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
    completion_id,
    response_from_text,
    stream_chunk,
)

router = APIRouter()


def _error(status: int, message: str, error_type: str, headers: dict | None = None):
    return JSONResponse(
        {"error": {"message": message, "type": error_type}}, status_code=status, headers=headers
    )


def _params(body: ChatCompletionRequest) -> GenerationParams:
    return GenerationParams(
        max_tokens=body.max_tokens or config.DEFAULT_MAX_TOKENS,
        temperature=(
            config.DEFAULT_TEMPERATURE if body.temperature is None else body.temperature
        ),
        top_p=config.DEFAULT_TOP_P if body.top_p is None else body.top_p,
        stop=body.stop,
    )


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        429: {"description": "rate limit exceeded"},
        503: {"description": "model not ready"},
    },
)
async def chat_completions(body: ChatCompletionRequest, request: Request):
    runtime = request.app.state.runtime
    limiter = request.app.state.rate_limiter

    client_key = request.client.host if request.client else "unknown"
    allowed, retry_after = limiter.allow(client_key)
    if not allowed:
        return _error(
            429,
            "rate limit exceeded",
            "rate_limit_error",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    messages = [message.model_dump() for message in body.messages]
    params = _params(body)

    if body.stream:
        return StreamingResponse(
            _sse_stream(runtime, messages, params),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await run_in_threadpool(runtime.generate, messages, params)
    except RuntimeNotReady as exc:
        return _error(503, str(exc), "model_not_ready")
    GENERATED_TOKENS.inc(result.completion_tokens)
    usage = Usage(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.prompt_tokens + result.completion_tokens,
    )
    return response_from_text(
        result.text, model=runtime.model_id, usage=usage, finish_reason=result.finish_reason
    )


def _sse_stream(runtime, messages, params) -> Iterator[str]:
    """Sync generator — Starlette iterates it in a worker thread."""
    completion = completion_id()
    completion_tokens = 0
    try:
        for content, finish in runtime.stream(messages, params):
            if content:
                completion_tokens += 1
                chunk = stream_chunk(completion, content=content, model=runtime.model_id)
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            if finish:
                final = stream_chunk(
                    completion, content=None, model=runtime.model_id, finish_reason=finish
                )
                yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    except RuntimeNotReady as exc:
        yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'model_not_ready'}})}\n\n"
    finally:
        GENERATED_TOKENS.inc(completion_tokens)
        yield "data: [DONE]\n\n"
