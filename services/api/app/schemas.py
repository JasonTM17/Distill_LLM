"""Pydantic schemas: an OpenAI-compatible chat completions subset."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from . import config


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32_000)


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    max_tokens: int | None = Field(default=None, ge=1, le=config.MAX_TOKENS_LIMIT)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | None = Field(default=None, max_length=4)

    @field_validator("messages")
    @classmethod
    def last_message_not_assistant(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if messages[-1].role == "assistant":
            raise ValueError("last message must be from user or system")
        return messages


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


def completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def response_from_text(
    text: str, *, model: str, usage: Usage, finish_reason: str = "stop"
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=completion_id(),
        created=int(time.time()),
        model=model,
        choices=[Choice(message=ChoiceMessage(content=text), finish_reason=finish_reason)],
        usage=usage,
    )


def stream_chunk(
    completion: str, *, content: str | None, model: str, finish_reason: str | None = None
) -> dict:
    """One SSE chunk in OpenAI chat.completion.chunk shape."""
    delta: dict = {}
    if content:
        delta["content"] = content
    return {
        "id": completion,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
