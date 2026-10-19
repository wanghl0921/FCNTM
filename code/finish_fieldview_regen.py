# -*- coding: utf-8 -*-
"""Wait for repin_fieldview_archive.py, then rebuild the manifest and re-check.

``repin_fieldview_archive.py`` replaces outputs/baseline_fieldview.csv only
when every number reproduces the archived one.  Whether or not it replaces the
file, the manifest and the paper-number check have to be refreshed afterwards
-- the CSV is a listed artifact, so its hash is part of the published chain.

    python finish_fieldview_regen.py     # waits, then rebuilds + verifies
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "outputs" / "rerun" / "fieldview_regen_report.json"
LOG = ROOT / "outputs" / "rerun" / "fieldview_after.log"


def run(*args) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(args)}\n")
        fh.flush()
        subprocess.run(args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)


def main() -> int:
    deadline = time.time() + 10 * 3600
    while time.time() < deadline:
        try:
            text = REPORT.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if '"finished_utc"' in text:
            break
        time.sleep(30)
    else:
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write("\nthe regeneration report never appeared\n")
        return 2
    time.sleep(10)
    run(sys.executable, "results_manifest.py")
    run(sys.executable, "results_manifest.py", "--verify")
    run(sys.executable, "check_paper_numbers.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
