"""Model runtime: loads the GGUF via llama.cpp and serves generations.

llama.cpp contexts are not safe for concurrent use, so all calls serialize
through one lock — acceptable for a single-user/local deployment. The model
loads in a background thread; ``ready`` gates /readyz.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Iterator

from . import config


@dataclass
class GenerationParams:
    max_tokens: int
    temperature: float
    top_p: float
    stop: list[str] | None = None


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


class RuntimeNotReady(RuntimeError):
    """Raised when a generation is requested before the model finished loading."""


class LlamaCppRuntime:
    """Wraps ``llama_cpp.Llama`` with lazy background loading and a call lock."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or config.MODEL_PATH
        self.model_id = config.MODEL_ID
        self._llama: Any = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    @property
    def ready(self) -> bool:
        return self._llama is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        """Blocking load; call from a background thread at startup."""
        try:
            from llama_cpp import Llama

            kwargs: dict[str, Any] = {
                "model_path": self.model_path,
                "n_ctx": config.N_CTX,
                "verbose": False,
            }
            if config.N_THREADS > 0:
                kwargs["n_threads"] = config.N_THREADS
            self._llama = Llama(**kwargs)
        except Exception as exc:  # surfaced via /readyz for diagnosability
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise

    def start_background_load(self) -> threading.Thread:
        thread = threading.Thread(target=self._load_quietly, name="model-load", daemon=True)
        thread.start()
        return thread

    def _load_quietly(self) -> None:
        try:
            self.load()
        except Exception:
            pass  # error already recorded in _load_error

    # ── inference ──────────────────────────────────────────────────────────
    def _require_model(self) -> Any:
        if self._llama is None:
            raise RuntimeNotReady(self._load_error or "model still loading")
        return self._llama

    def generate(self, messages: list[dict[str, str]], params: GenerationParams) -> GenerationResult:
        llama = self._require_model()
        with self._lock:
            raw = llama.create_chat_completion(
                messages=messages,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                stop=params.stop or [],
            )
        choice = raw["choices"][0]
        usage = raw.get("usage", {})
        return GenerationResult(
            text=(choice["message"].get("content") or "").strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason") or "stop",
        )

    def stream(
        self, messages: list[dict[str, str]], params: GenerationParams
    ) -> Iterator[tuple[str | None, str | None]]:
        """Yield ``(content_delta, finish_reason)`` tuples; finish arrives last."""
        llama = self._require_model()
        with self._lock:
            for chunk in llama.create_chat_completion(
                messages=messages,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                stop=params.stop or [],
                stream=True,
            ):
                choice = chunk["choices"][0]
                content = choice.get("delta", {}).get("content")
                finish = choice.get("finish_reason")
                if content or finish:
                    yield content, finish
