# -*- coding: utf-8 -*-
"""Extend the field-occlusion robustness curves to 10 seeds for the three
baselines that were still at 3 seeds: ETM, NMF, ProdLDA.

One invocation of occlusion_robustness.py per (corpus, seed) with
--models NMF,ProdLDA,ETM, writing
rerun/occlusion_robustness_{corpus}_etmnmfprod_seed{seed}.csv
(all three models accumulate in the same frame; a later invocation must not
overwrite a file that already carries all three models).  Existing complete
work is skipped, so the chain can be restarted at any point.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv-occl" / "Scripts" / "python.exe"
MODELS = ["NMF", "ProdLDA", "ETM"]
SEEDS = ["53", "71", "89", "101", "113", "127", "139"]
CORPORA = ["github", "agnews", "civil"]


def complete(out: Path) -> bool:
    if not out.exists():
        return False
    try:
        with open(out, encoding="utf-8-sig") as fh:
            have = {r.get("model") for r in csv.DictReader(fh)}
        return all(m in have for m in MODELS)
    except Exception:
        return False


def main() -> int:
    failed = []
    combos = [(c, s) for c in CORPORA for s in SEEDS]
    total = len(combos)
    for i, (corpus, seed) in enumerate(combos, 1):
        out = ROOT / "outputs" / "rerun" / \
            f"occlusion_robustness_{corpus}_etmnmfprod_seed{seed}.csv"
        tag = f"[{i}/{total}] {corpus} seed{seed}"
        if complete(out):
            print(f"skip {tag}: complete ({out.name})", flush=True)
            continue
        cmd = [PY, "-u", "occlusion_robustness.py",
               "--corpus", corpus, "--seed", seed,
               "--models", ",".join(MODELS),
               "--out", str(out)]
        print(f"\n=== [{time.strftime('%H:%M:%S')}] {tag} ===\n"
              f"$ {' '.join(str(c) for c in cmd)}", flush=True)
        res = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
        print(f"=== [{time.strftime('%H:%M:%S')}] {tag} exit={res.returncode} ===",
              flush=True)
        if res.returncode != 0:
            failed.append(tag)
    print(f"\nALL DONE. failed={len(failed)}: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
