"""List running python.exe processes with their command lines (for debugging)."""
import subprocess
import sys

# Use PowerShell Get-CimInstance to avoid wmic quoting hell.
cmd = [
    "powershell", "-NoProfile", "-Command",
    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
    "Select-Object ProcessId,CommandLine | Format-List",
]
r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout)
if r.stderr:
    print("STDERR:", r.stderr[:500], file=sys.stderr)
