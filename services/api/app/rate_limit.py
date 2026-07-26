"""Per-client sliding-window rate limiter.

In-process on purpose: this service runs as a single instance serving one local
model (generations serialize on the model lock anyway, so horizontal scale
would need a model-server change first). Scale-out would swap this for Redis.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from . import config


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int | None = None, window_seconds: float | None = None):
        self.max_requests = max_requests or config.RATE_LIMIT_REQUESTS
        self.window_seconds = window_seconds or config.RATE_LIMIT_WINDOW_SECONDS
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic() if now is None else now
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.max_requests:
                return False, max(0.0, events[0] + self.window_seconds - now)
            events.append(now)
            return True, 0.0
