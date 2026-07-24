"""Download Qwen2.5-3B to local directory without symlinks."""
import os, sys
from huggingface_hub import snapshot_download

target = "D:/models/qwen25-3b"
os.makedirs(target, exist_ok=True)

print("Downloading Qwen2.5-3B-Instruct...", flush=True)
snapshot_download(
    "Qwen/Qwen2.5-3B-Instruct",
    local_dir=target,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=2,
)
print("Downloaded successfully!", flush=True)

# Verify
files = os.listdir(target)
weights = [f for f in files if f.endswith('.safetensors')]
print(f"Files: {len(files)} total, {len(weights)} weight files", flush=True)
for w in sorted(weights):
    print(f"  {w}", flush=True)
