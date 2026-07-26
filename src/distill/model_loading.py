"""Shared model-loading helper encoding this machine's hard-won constraints.

Every causal-LM load in the package must go through here:

* dtype stays bf16 (checkpoint native) — fp16 conversion during weight
  materialization access-violates on the current torch nightly (Windows),
* weights load on CPU first, then move to CUDA — safetensors 0.8's
  direct-to-device path raises "invalid python storage" against this torch,
* safetensors files are pre-touched into the OS cache first so mmap never
  faults against a constrained pagefile.
"""

from __future__ import annotations

from pathlib import Path

from .safetensors_pretouch import install as install_pretouch


def load_causal_lm(model_path: str | Path, *, prefer_cuda: bool = True):
    """Load a causal LM in bf16, CPU-first, optionally moved to CUDA."""
    install_pretouch()
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    if prefer_cuda and torch.cuda.is_available():
        model = model.to("cuda:0")
    return model


def load_tokenizer(model_path: str | Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
