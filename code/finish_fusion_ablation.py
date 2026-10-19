# -*- coding: utf-8 -*-
"""Wait for the GitHub field-dropout run, then rebuild the fusion ablation.

Table "是否训练融合表示" of the paper still quotes the pre-revision ablation
(fixed 300 epochs, no validation split, a different base model).  The current
``run_ablation_public.py`` starts from exactly the paper's full model
(``FULL_CONFIG``: field dropout 0.5, gate fusion, balanced fused-reconstruction
weight 0.5, 20% held-out validation, 700-epoch budget), so only the
``no-fusion`` arm has to be trained: the trained-fusion arm is the main FC-NTM
run already stored in ``outputs/fcntm_github_final.csv``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
WAIT_LOG = ROOT / "outputs" / "rerun" / "github_fielddrop_off.log"
DEADLINE_HOURS = 6.0


def main() -> int:
    deadline = time.time() + DEADLINE_HOURS * 3600
    while time.time() < deadline:
        text = (WAIT_LOG.read_text(encoding="utf-8", errors="replace")
                if WAIT_LOG.exists() else "")
        if "saved " in text:
            break
        time.sleep(120)
    else:
        print("[chain] timed out waiting for the field-dropout run", flush=True)
        return 1
    print("[chain] field-dropout run finished; starting the fusion ablation",
          flush=True)
    cmd = [PY, "-u", "run_ablation_public.py",
           "--dataset", "github", "--seeds", "11,23,37", "--k", "30",
           "--lr", "5e-4", "--configs", "no-fusion",
           "--out", "ablation_public_github_nofusion.csv"]
    print("$", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    print(f"[chain] fusion ablation exit={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
