# -*- coding: utf-8 -*-
"""Wait for the anchor re-pin to finish, then rebuild the manifest and re-check.

Operational helper for the 2026-10-12 round: after ``baseline_fieldview.py``
(the sidecar now describes the merged file) the code digest moved, so the four
new-seed sidecars are re-pinned by ``verify_newseed_archives.py``.  When that
job is done the manifest (which lists those CSVs and the scripts) has to be
rebuilt, and the summary + paper check re-run.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\Administrator\Downloads\FCNTM完整代码")
WATCH = ROOT / "outputs" / "rerun" / "verify_newseed3.log"
LOG = ROOT / "outputs" / "rerun" / "repin_after.log"


def run(*args) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(args)}\n")
        fh.flush()
        subprocess.run(args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)


def main() -> int:
    deadline = time.time() + 3 * 3600
    while time.time() < deadline:
        try:
            text = WATCH.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if "pinned fair_readout_agnews_newseeds.csv" in text:
            break
        time.sleep(30)
    else:
        return 2
    time.sleep(10)
    run(sys.executable, "summarize_seed_extension.py")
    run(sys.executable, "results_manifest.py")
    run(sys.executable, "results_manifest.py", "--verify")
    run(sys.executable, "check_paper_numbers.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
