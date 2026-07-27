"""Back up v0.5 artifacts before v0.6 training overwrites them.

Training overwrites checkpoints/adapter/ and merge overwrites checkpoints/merged/.
This copies both to versioned dirs so v0.5 can still be evaluated/served.

Idempotent: skips if the backup target already exists.
"""
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT / "checkpoints"

PAIRS = [
    (CHECKPOINT / "adapter", CHECKPOINT / "v0.5_adapter"),
    (CHECKPOINT / "merged", CHECKPOINT / "v0.5_merged"),
]


def main() -> int:
    for src, dst in PAIRS:
        if dst.exists():
            print(f"SKIP {dst.name} (already exists)")
            continue
        if not src.exists() or not any(src.iterdir()):
            print(f"SKIP {src.name} (source missing/empty)")
            continue
        print(f"COPY {src} -> {dst}")
        shutil.copytree(str(src), str(dst))
        print(f"  done ({sum(f.stat().st_size for f in dst.rglob('*') if f.is_file())/1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
