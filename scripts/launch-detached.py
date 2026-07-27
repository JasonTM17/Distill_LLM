"""Launcher: starts any script as a fully detached background process.

Usage: python launch-detached.py <script.py> [args...]

Uses CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS so the child survives even if
the parent shell (and the tool call that spawned it) is killed or times out.
Prints the child PID so the orchestrator can poll it.

Redirects stdout/stderr to logs/<script-basename>.log / .err.
"""
import os
import subprocess
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: launch-detached.py <script.py> [args...]", file=sys.stderr)
    sys.exit(2)

PROJECT = Path(__file__).resolve().parent.parent
script = Path(sys.argv[1]).resolve()
script_args = sys.argv[2:]

stem = script.stem
LOG = PROJECT / "logs" / f"{stem}.log"
ERR = PROJECT / "logs" / f"{stem}.err"
LOG.parent.mkdir(parents=True, exist_ok=True)

# Windows detached-process creation flags.
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

proc = subprocess.Popen(
    [sys.executable, str(script), *script_args],
    cwd=str(PROJECT),
    stdout=open(LOG, "w", encoding="utf-8"),
    stderr=open(ERR, "w", encoding="utf-8"),
    stdin=subprocess.DEVNULL,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
    close_fds=True,
)
print(f"LAUNCHED_PID={proc.pid}")
print(f"LOG={LOG}")
print(f"ERR={ERR}")
