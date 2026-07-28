"""v0.7 post-training pipeline (detached): wait for training, then merge + evaluate (+ judge).

Launched after run-train-v07.py. Waits for training to finish (the
"final validation" log marker, or the training PID exiting), then runs merge and
the full evaluation (PPL @ 512/1024/2048 + ROUGE-L/token-F1 + LLM-as-judge via
cx/gpt-5.5-high, which the 9Router exposes).

Export is intentionally NOT done here: whether to ship a v0.7 GGUF depends on
whether v0.7 beat v0.6's 5.85, which is decided after reading the report.

Usage: python run-v07-post-train.py <training_pid>
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
TRAIN_LOG = PROJECT / "logs" / "run-train-v07.log"
TRAIN_ERR = PROJECT / "logs" / "run-train-v07.err"
DONE_MARKER = "final validation"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _pid_alive(pid: int) -> bool:
    """True if `pid` is still a running process (Windows OpenProcess)."""
    if not pid:
        return True  # unknown pid -> be permissive
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        SYNCHRONIZE = 0x00100000
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    except Exception:
        return True


def wait_for_training(train_pid: int, timeout_s: int = 3600) -> bool:
    """Return True once training reports its final-validation line."""
    start = time.time()
    saw_start = False
    while time.time() - start < timeout_s:
        log = _read(TRAIN_LOG)
        err = _read(TRAIN_ERR)
        if "starting training" in log:
            saw_start = True
        if DONE_MARKER in log:
            return True
        if saw_start and ("Traceback" in log or "CUDA out of memory" in err) and not _pid_alive(train_pid):
            print("training crashed (traceback/OOM and process gone)", flush=True)
            return False
        if saw_start and not _pid_alive(train_pid) and DONE_MARKER not in log:
            print("training process exited without the final-validation marker", flush=True)
            return False
        time.sleep(15)
    print("timed out waiting for training", flush=True)
    return False


def run(step: str, module_args: list[str]) -> int:
    print(f"\n=== {step} ===", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT / "src")
    return subprocess.call([sys.executable, "-m", *module_args], cwd=str(PROJECT), env=env)


REPORT = PROJECT / "plans" / "reports" / "evaluation-v0.7.md"
VERDICT = PROJECT / "logs" / "v07-verdict.txt"
V06_PPL = 5.85
V05_PPL = 5.23


def parse_cap2048_ppl() -> float | None:
    """Pull the bolded overall PPL at cap 2048 out of the eval report."""
    if not REPORT.exists():
        return None
    text = REPORT.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"cap 2048[^\n]*?\*\*([0-9.]+)\*\*", text)
    return float(m.group(1)) if m else None


def main() -> int:
    train_pid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"waiting for v0.7 training (pid={train_pid}) to finish...", flush=True)
    if not wait_for_training(train_pid):
        print("ABORT: training did not complete cleanly", flush=True)
        return 1
    print("training done -> merge", flush=True)
    if run("merge", ["distill.merge"]):
        print("ABORT: merge failed", flush=True)
        return 2
    print("merge done -> evaluate (PPL + ROUGE + judge; baseline v0.5 5.23@2048)", flush=True)
    rc = run(
        "evaluate",
        ["distill.evaluate", "--label", "v0.7", "--ppl-caps", "512,1024,2048", "--judge",
         "--baseline-ppl", "5.23", "--baseline-cap", "2048"],
    )
    if rc:
        print("WARNING: evaluate returned non-zero (judge may be down); PPL/ROUGE may still be in the report", flush=True)

    ppl = parse_cap2048_ppl()
    exported = False
    lines: list[str] = []
    if ppl is None:
        lines.append("PPL_2048=PARSE_FAILED (read plans/reports/evaluation-v0.7.md manually)")
    else:
        beat_v06 = ppl < V06_PPL
        vs_v05 = "below" if ppl < V05_PPL else ("equal" if ppl == V05_PPL else "above")
        lines.append(f"PPL_2048={ppl}")
        lines.append(f"beat_v06({V06_PPL})={'yes' if beat_v06 else 'no'}")
        lines.append(f"vs_v05({V05_PPL})={vs_v05} (indicative; v0.5 used a different test split)")
        if beat_v06:
            print(f"v0.7 PPL {ppl} < v0.6 {V06_PPL} -> exporting GGUF v0.7", flush=True)
            env = os.environ.copy()
            env["GGUF_MODEL_BASENAME"] = "distill-gpt55-v0.7"
            env["PYTHONPATH"] = str(PROJECT / "src")
            erc = subprocess.call([sys.executable, "-m", "distill.export_gguf"], cwd=str(PROJECT), env=env)
            exported = erc == 0
            lines.append(f"exported_gguf={'yes' if exported else 'no (export step failed)'}")
        else:
            lines.append("exported_gguf=no (did not beat v0.6; v0.5 stays canonical)")
    VERDICT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("VERDICT:\n" + "\n".join(lines), flush=True)
    print("post-train pipeline complete -> plans/reports/evaluation-v0.7.md + logs/v07-verdict.txt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
