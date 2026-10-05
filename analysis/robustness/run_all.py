"""
Run every sweep, retrying any that dies on the interpreter fault described in the README.

The sweeps are deterministic: given the same artifacts they produce byte-identical JSON.
On this machine the interpreter intermittently dies with a native fault (Windows
0xC0000005 or 0xC0000409) partway through the longer sweeps, independent of shell,
output redirection, and whether numpy is imported. A run either completes correctly or
dies without writing; there is no partial-output failure mode, because each sweep writes
its JSON in a single call at the end.

This driver therefore just retries until each sweep exits cleanly, and reports how many
attempts each needed so the instability stays visible rather than hidden.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEPS = [
    "sweep_continuation_budget.py",
    "sweep_segmentation.py",
    "sweep_n_convention.py",
    "sweep_alpha_beta.py",
]
MAX_ATTEMPTS = 12


def main():
    summary, failed = [], []
    for script in SWEEPS:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"[{script}] attempt {attempt} ...", flush=True)
            proc = subprocess.run(
                [sys.executable, "-u", str(HERE / script)],
                cwd=HERE, capture_output=True, text=True,
            )
            if proc.returncode == 0:
                tail = [ln for ln in proc.stdout.strip().split("\n") if ln.strip()][-1:]
                print(f"[{script}] OK after {attempt} attempt(s). {tail[0] if tail else ''}",
                      flush=True)
                summary.append((script, attempt, "ok"))
                break
            print(f"[{script}] exit {proc.returncode}, retrying", flush=True)
        else:
            print(f"[{script}] FAILED after {MAX_ATTEMPTS} attempts", flush=True)
            summary.append((script, MAX_ATTEMPTS, "failed"))
            failed.append(script)

    print("\n" + "=" * 62)
    for script, attempts, status in summary:
        print(f"  {script:34s} {status:7s} ({attempts} attempt(s))")
    print("=" * 62)
    if failed:
        print(f"\n{len(failed)} sweep(s) did not complete: {', '.join(failed)}")
        return 1
    print("\nAll sweeps completed. Results written to the repository root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
