"""Export the merged model to GGUF: f16 convert → Q4_K_M / Q5_K_M quantize.

Relies on llama.cpp tooling living outside the repo (binaries + source clone),
paths overridable via env because they are machine-specific:

* ``LLAMACPP_BIN``  — dir with ``llama-quantize.exe`` / ``llama-cli.exe``
* ``LLAMACPP_SRC``  — llama.cpp source checkout (for ``convert_hf_to_gguf.py``)

The intermediate f16 GGUF (~3.1GB) is deleted after successful quantization to
protect free space on D:.

Usage::

    python -m distill.export_gguf                 # both quantizations
    python -m distill.export_gguf --quant Q4_K_M  # single quantization
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import config
from .logging_utils import get_logger

logger = get_logger("export-gguf")

LLAMACPP_BIN = Path(os.environ.get("LLAMACPP_BIN", "D:/tools/llama-cpp-b10107"))
LLAMACPP_SRC = Path(os.environ.get("LLAMACPP_SRC", "D:/tools/llama.cpp-src"))

MODEL_BASENAME = "distill-gpt55-v0.5"
DEFAULT_QUANTS = ("Q4_K_M", "Q5_K_M")


def _run(cmd: list[str], **kwargs) -> None:
    logger.info("$ %s", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True, **kwargs)


def convert_to_f16(merged_dir: Path, out_file: Path) -> None:
    """HF safetensors → f16 GGUF via llama.cpp's converter."""
    converter = LLAMACPP_SRC / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"converter not found: {converter} (set LLAMACPP_SRC)")
    _run(
        [
            sys.executable,
            str(converter),
            str(merged_dir),
            "--outfile",
            str(out_file),
            "--outtype",
            "f16",
        ],
        cwd=str(LLAMACPP_SRC),
    )


def quantize(f16_file: Path, out_file: Path, quant: str) -> None:
    quantizer = LLAMACPP_BIN / "llama-quantize.exe"
    if not quantizer.exists():
        raise FileNotFoundError(f"llama-quantize not found: {quantizer} (set LLAMACPP_BIN)")
    _run([str(quantizer), str(f16_file), str(out_file), quant])


def smoke_test(gguf_file: Path, prompt: str = "What is 2+2? Answer briefly.") -> str:
    """One short CPU generation to prove the GGUF loads and answers."""
    cli = LLAMACPP_BIN / "llama-cli.exe"
    result = subprocess.run(
        [str(cli), "-m", str(gguf_file), "-p", prompt, "-n", "48", "--temp", "0", "-no-cnv"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"smoke test failed: {result.stderr[-500:]}")
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export merged model to GGUF")
    parser.add_argument("--quant", nargs="*", default=list(DEFAULT_QUANTS))
    parser.add_argument("--keep-f16", action="store_true", help="keep intermediate f16 GGUF")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args(argv)

    merged = config.MERGED_MODEL_DIR
    if not (merged / "config.json").exists():
        logger.error("no merged model at %s — run distill.merge first", merged)
        return 1

    config.GGUF_DIR.mkdir(parents=True, exist_ok=True)
    f16_file = config.GGUF_DIR / f"{MODEL_BASENAME}-f16.gguf"

    if not f16_file.exists():
        convert_to_f16(merged, f16_file)
    logger.info("f16 GGUF: %.2f GB", f16_file.stat().st_size / 1024**3)

    outputs = []
    for quant in args.quant:
        out_file = config.GGUF_DIR / f"{MODEL_BASENAME}-{quant}.gguf"
        quantize(f16_file, out_file, quant)
        logger.info("%s: %.2f GB", out_file.name, out_file.stat().st_size / 1024**3)
        outputs.append(out_file)

    if not args.keep_f16:
        f16_file.unlink()
        logger.info("deleted intermediate %s", f16_file.name)

    if not args.skip_smoke:
        for out_file in outputs:
            answer = smoke_test(out_file)
            logger.info("smoke %s -> %s", out_file.name, answer[-200:])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
