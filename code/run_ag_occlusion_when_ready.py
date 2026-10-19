# -*- coding: utf-8 -*-
"""Wait for the AG News FC-NTM state, then run the AG occlusion experiment.

The FC-NTM adapter needs a trained model (encoder + decoder), which
``run_public_missing_field.py --dump-state`` writes only after training.  This
runner polls for that file and then launches the same
``occlusion_robustness.py`` protocol used for GitHub, so the AG News curves can
be produced unattended.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "outputs" / "agnews_missing_state.pt"


def main() -> int:
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        if STATE.exists() and STATE.stat().st_size > 0:
            break
        time.sleep(60)
    else:
        print("timed out waiting for", STATE, flush=True)
        return 1
    print("state ready:", STATE, flush=True)
    cmd = [sys.executable, "-u", "occlusion_robustness.py",
           "--corpus", "agnews", "--seed", "11",
           "--models", "FC-NTM,LDA,NMF,ProdLDA,ETM",
           "--rates", "0,0.25,0.5,0.75,1.0",
           "--fcntm-state", str(STATE),
           "--out", "occlusion_robustness_agnews_seed11.csv"]
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
