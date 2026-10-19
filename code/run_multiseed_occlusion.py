# -*- coding: utf-8 -*-
"""Unattended chain: multi-seed field-occlusion experiments, both corpora.

For every requested seed this needs
  * an FC-NTM model state (encoder + decoder) per corpus, and
  * the baselines, which ``occlusion_robustness.py`` trains itself.

The FC-NTM states are produced by ``run_public_missing_field.py`` with
``--dump-state`` (it also writes the whole-field occlusion row used in
``tab:occlusion``), merging into the existing state file through
``--reuse-state``.

Everything is sequential on purpose: this box has ~2 GB of usable memory.
Existing work is reused (the occlusion script skips models whose curve is
already complete for that seed/corpus), so the chain can be restarted.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

SEEDS = ["23", "37"]
CORPORA = ["github", "agnews"]
STATES = {
    "github": ROOT / "outputs" / "github_missing_state.pt",
    "agnews": ROOT / "outputs" / "agnews_missing_state.pt",
}


def run(cmd, tag):
    print(f"\n=== [{time.strftime('%H:%M:%S')}] {tag} ===\n"
          f"$ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=str(ROOT))
    print(f"=== [{time.strftime('%H:%M:%S')}] {tag} exit={res.returncode} ===",
          flush=True)
    return res.returncode


def main() -> int:
    seeds = sys.argv[1].split(",") if len(sys.argv) > 1 else SEEDS
    # 1) FC-NTM states for the extra seeds (also fills the whole-field table)
    for corpus in CORPORA:
        state = STATES[corpus]
        cmd = [PY, "-u", "run_public_missing_field.py",
               "--dataset", corpus, "--seeds", ",".join(seeds),
               "--settings", "fielddrop_on",
               "--dump-state", str(state),
               "--resume-every", "50"]
        if state.exists():
            cmd += ["--reuse-state", str(state)]
        rc = run(cmd, f"train FC-NTM state {corpus} seeds={seeds}")
        if rc != 0:
            # never continue on a failed step: the 2026-09-13 run kept going
            # after an OOM here, then aborted 50 minutes later inside the
            # occlusion step for the missing seed, and the outer supervise.py
            # still reported success.
            return rc
    # 2) occlusion curves for every seed/corpus
    for seed in seeds:
        for corpus in CORPORA:
            state = STATES[corpus]
            cmd = [PY, "-u", "occlusion_robustness.py",
                   "--corpus", corpus, "--seed", seed,
                   "--models", "FC-NTM,LDA,NMF,ProdLDA,ETM",
                   "--rates", "0,0.25,0.5,0.75,1.0",
                   "--fcntm-state", str(state),
                   "--out", f"occlusion_robustness_{corpus}_seed{seed}.csv"]
            rc = run(cmd, f"occlusion curves {corpus} seed={seed}")
            if rc != 0:
                return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
