"""Console logging helpers that behave correctly on Windows terminals.

The default Windows console codepage mangles UTF-8, so every entrypoint routes
its output through :func:`configure_console` before printing dataset content.
"""

from __future__ import annotations

import io
import logging
import sys

_CONFIGURED = False


def configure_console() -> None:
    """Force UTF-8 on stdout/stderr. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        # ``reconfigure`` exists on TextIOWrapper (Python 3.7+) and is cheaper
        # than rebuilding the wrapper; fall back for exotic stream objects.
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
                continue
            except (ValueError, OSError):
                pass
        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            setattr(
                sys,
                stream_name,
                io.TextIOWrapper(buffer, encoding="utf-8", errors="replace"),
            )
    _CONFIGURED = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a single stream handler and no duplicate output."""
    configure_console()
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger
