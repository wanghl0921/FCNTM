# -*- coding: utf-8 -*-
"""Finish the AG News field-occlusion chain after the seed-37 crash.

The unattended chain in ``run_multiseed_occlusion.py`` died at
``agnews / fielddrop_on / seed=37`` because that training run was killed by an
out-of-memory error at epoch 200, so the partial-occlusion step afterwards had
no trained FC-NTM state to load and aborted with exit code 2.

This runner picks the work up from there.  Every step is idempotent: the
whole-field table skips (dataset, setting, seed) keys already present, and
``occlusion_robustness.py`` skips models whose curve is already complete.

  1. whole-field occlusion rows for GitHub seed 11 (both the field-dropout and
     the no-dropout setting) -- the stored models are reused, so this only
     re-scores them and restores the numbers the paper quotes for the
     whole-field occlusion table (tab:occlusion)
  2. whole-field occlusion rows for AG News (seed 11 reuses the stored model,
     seed 37 is trained by the caller before this runner is started)
  3. partial-occlusion curves for AG News seed 37
  4. refresh outputs/occlusion_comparison.csv over every seed on disk
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
STATE = ROOT / "outputs" / "agnews_missing_state.pt"


def run(cmd, tag):
    print(f"\n=== [{time.strftime('%H:%M:%S')}] {tag} ===\n"
          f"$ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    print(f"=== [{time.strftime('%H:%M:%S')}] {tag} exit={rc} ===",
          flush=True)
    return rc


def main() -> int:
    rc = run([PY, "-u", "run_public_missing_field.py",
              "--dataset", "github", "--seeds", "11",
              "--settings", "fielddrop_on,fielddrop_off",
              "--reuse-state", "outputs/github_missing_state.pt"],
             "github whole-field rows seed=11")
    if rc != 0:
        return rc
    rc = run([PY, "-u", "run_public_missing_field.py",
              "--dataset", "agnews", "--seeds", "11,37",
              "--settings", "fielddrop_on",
              "--dump-state", "outputs/agnews_missing_state.pt",
              "--reuse-state", "outputs/agnews_missing_state.pt",
              "--resume-every", "50"],
             "agnews whole-field rows seeds=11,37")
    if rc != 0:
        return rc
    rc = run([PY, "-u", "occlusion_robustness.py",
              "--corpus", "agnews", "--seed", "37",
              "--models", "FC-NTM,LDA,NMF,ProdLDA,ETM",
              "--rates", "0,0.25,0.5,0.75,1.0",
              "--fcntm-state", str(STATE),
              "--out", "occlusion_robustness_agnews_seed37.csv"],
             "occlusion curves agnews seed=37")
    if rc != 0:
        return rc
    return run([PY, "-u", "compare_occlusion.py"],
               "refresh occlusion comparison")


if __name__ == "__main__":
    raise SystemExit(main())
