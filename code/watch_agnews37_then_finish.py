# -*- coding: utf-8 -*-
"""Wait for the resumed AG News seed-37 training, then run the rest of the chain.

``run_public_missing_field.py`` prints a final ``saved <path> | <path>`` line,
so the watcher polls the training log for it instead of following a PID (which
does not survive a session restart).  ``finish_agnews_occlusion.py`` then writes
the missing whole-field row and the AG News seed-37 occlusion curves.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "outputs" / "rerun" / "missing_agnews37.log"
DEADLINE_HOURS = 4.0


def training_finished() -> bool:
    if not LOG.exists():
        return False
    text = LOG.read_text(encoding="utf-8", errors="replace")
    return "saved " in text


def main() -> int:
    deadline = time.time() + DEADLINE_HOURS * 3600
    while not training_finished():
        if time.time() > deadline:
            print("timed out waiting for the seed-37 training", flush=True)
            return 1
        time.sleep(60)
    print("training finished; starting the occlusion chain", flush=True)
    return subprocess.run(
        [sys.executable, "-u", "finish_agnews_occlusion.py"],
        cwd=str(ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
