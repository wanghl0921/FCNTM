"""Run a command, restarting it when it dies, so long jobs survive crashes.

    python supervise.py [--tries N] [--sleep S] -- python -u train.py ...

The wrapped command is expected to be resumable (see
``run_public_missing_field.py --dump-state`` which writes a full training
state every N epochs and continues from it).
"""

from __future__ import annotations

import subprocess
import sys
import time


def main() -> int:
    argv = sys.argv[1:]
    tries = 12
    sleep = 5
    while argv and argv[0].startswith("--"):
        if argv[0] == "--tries":
            tries = int(argv[1])
            argv = argv[2:]
        elif argv[0] == "--sleep":
            sleep = int(argv[1])
            argv = argv[2:]
        elif argv[0] == "--":
            argv = argv[1:]
            break
        else:
            break
    if not argv:
        print(__doc__)
        return 2
    for attempt in range(1, tries + 1):
        print(f"[supervise] attempt {attempt}/{tries}: {' '.join(argv)}",
              flush=True)
        rc = subprocess.run(argv).returncode
        print(f"[supervise] attempt {attempt} exited with {rc}", flush=True)
        if rc == 0:
            return 0
        time.sleep(sleep)
    print("[supervise] giving up", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
