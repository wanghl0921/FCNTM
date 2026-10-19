# -*- coding: utf-8 -*-
"""Second ablation round: the missing 2x2 cell and the fused-weight sweep.

    1. gate-w0p25       the paper's main model (learnable gate) with the fused
                        reconstruction weight at 0.25
    2. gate-w1p0        same, weight 1.0

The "gate without fused reconstruction" cell is *not* run: FCNTM only uses the
gate inside ``if self.fused_recon:``, so without that term the parameter gets no
gradient and the model degenerates to mean fusion (documented in the paper).

Together with the already stored ``no-fusion`` (mean, no fused recon),
``fused-mean`` (mean, weight 0.5) and the paper's main model (gate, weight 0.5)
this closes the 2x2 and gives a 3-point weight sweep.  Sequential on purpose:
this box trains one model at a time.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OUT = ROOT / "outputs"

CONFIGS = {
    "gate-w0p25": "ablation_public_github_gate_w0p25.csv",
    "gate-w1p0": "ablation_public_github_gate_w1p0.csv",
}


def run(cmd, tag):
    print(f"\n=== [{time.strftime('%H:%M:%S')}] {tag} ===\n"
          f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    rc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT)).returncode
    print(f"=== [{time.strftime('%H:%M:%S')}] {tag} exit={rc} ===",
          flush=True)
    return rc


def main() -> int:
    for name, out in CONFIGS.items():
        rc = run([PY, "-u", "run_ablation_public.py",
                  "--dataset", "github", "--seeds", "11,23,37", "--k", "30",
                  "--lr", "5e-4", "--configs", name, "--out", out],
                 f"ablation {name}")
        if rc != 0:
            return rc
    print("\n[chain] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
