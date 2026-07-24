"""Download Qwen2.5-1.5B-Instruct to local directory."""
import os
from huggingface_hub import snapshot_download

target = "D:/models/qwen15-1.5b"
os.makedirs(target, exist_ok=True)

print("Downloading Qwen2.5-1.5B-Instruct...", flush=True)
snapshot_download(
    "Qwen/Qwen2.5-1.5B-Instruct",
    local_dir=target,
    max_workers=2,
)
print("Done!", flush=True)

files = os.listdir(target)
weights = [f for f in files if f.endswith('.safetensors')]
total = sum(os.path.getsize(os.path.join(target, f)) for f in weights) / 1e9
print(f"Weights: {len(weights)} files, {total:.2f} GB", flush=True)
