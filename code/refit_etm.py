# -*- coding: utf-8 -*-
"""Re-run ETM after fixing the shared word-embedding buffer.

``EmbeddingDecoder`` used to wrap the caller's float32 array without copying;
with ``trainable_embeddings=True`` (ETM) Adam then mutated that shared array,
so seeds 23/37 started from seed 11's updated embeddings.  ``models.py`` now
clones the input.  This chain refits ETM for the three seeds of both corpora
(main table) and rebuilds the ETM rows of the fair-read-out table.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OUT = ROOT / "outputs"


def run(cmd, tag):
    print(f"\n=== [{time.strftime('%H:%M:%S')}] {tag} ===\n"
          f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    rc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT)).returncode
    print(f"=== [{time.strftime('%H:%M:%S')}] {tag} exit={rc} ===", flush=True)
    return rc


def main() -> int:
    for corpus in ("agnews", "github"):
        rc = run([PY, "-u", "benchmark_public.py", "--dataset", corpus,
                  "--models", "ETM", "--seeds", "11,23,37", "--k", "30",
                  "--no-resume",
                  "--out", f"rerun_etm_{corpus}_fixed.csv",
                  "--summary", f"rerun_etm_{corpus}_fixed_summary.csv"],
                 f"ETM refit {corpus}")
        if rc != 0:
            return rc
        rc = run([PY, "-u", "fair_readout.py", "--corpus", corpus,
                  "--seeds", "11,23,37", "--k", "30", "--models", "ETM",
                  "--out", f"fair_readout_{corpus}_etm_fixed.csv"],
                 f"fair readout ETM {corpus}")
        if rc != 0:
            return rc
        # merge the refitted ETM rows into the stored fair-readout table
        new = pd.read_csv(OUT / f"fair_readout_{corpus}_etm_fixed.csv")
        path = OUT / f"fair_readout_{corpus}_3seed.csv"
        old = pd.read_csv(path)
        merged = pd.concat([old[old["model"] != "ETM"], new],
                           ignore_index=True)
        merged.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  merged ETM rows into {path.name}", flush=True)
    print("\n[chain] ETM refit DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
