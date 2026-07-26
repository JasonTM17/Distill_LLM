"""Windows pagefile workaround for mmap-loading large safetensors files.

transformers' ``safe_open`` mmaps model shards; on Windows machines where C: has
little free space for pagefile growth this raises "The paging file is too small".
Sequentially reading the file first pulls it into the OS file cache so the mmap
never faults against the pagefile. No-op on files already touched this process.

Call :func:`install` before importing/loading any model weights.
"""

from __future__ import annotations

import os

from .logging_utils import get_logger

logger = get_logger(__name__)

_CHUNK_BYTES = 64 * 1024 * 1024
_touched: set[str] = set()
_installed = False


def _touch_file(filepath: str) -> None:
    if filepath in _touched or not os.path.isfile(filepath):
        return
    logger.info("[fs-cache] pre-loading %s into OS cache", os.path.basename(filepath))
    with open(filepath, "rb") as handle:
        while handle.read(_CHUNK_BYTES):
            pass
    _touched.add(filepath)


def install() -> None:
    """Patch ``safetensors.safe_open`` to pre-touch files before mmap."""
    global _installed
    if _installed:
        return
    import safetensors

    original = safetensors.safe_open

    def patched(filename, framework, device=None, **kwargs):
        if isinstance(filename, str):
            _touch_file(filename)
        return original(filename, framework, device=device, **kwargs)

    safetensors.safe_open = patched
    _installed = True
