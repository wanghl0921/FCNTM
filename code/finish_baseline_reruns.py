# -*- coding: utf-8 -*-
"""Re-run the three baseline rows the paper still marks as pending.

CombinedTM, ProdLDA and CTM-lite were produced by the pre-revision training
loop (double decoder call per stream, single-sample validation loss, 3000-epoch
fallback).  ``benchmark_public.py`` now trains every neural baseline with the
same 20% held-out validation, 700-epoch budget and the shared dictionary, so
re-running these three is what removes the red daggers from Tables 1/2.

ProdLDA is written under its historical file name ``NeuralLDA``: the existing
per-seed rows use that name and ``merge_benchmark_rows.py`` matches on
(model, seed, K_target), so using ``ProdLDA`` here would append a duplicate row
instead of replacing the stale one.

Everything is sequential on purpose (this box has ~5 GB usable memory, and the
earlier agnews run already hit MemoryError when models were allowed to stack).
After each corpus the new rows are merged into the main run table, so the
chain is restartable and each step is idempotent.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OUT = ROOT / "outputs"
MODELS = "CombinedTM,NeuralLDA,CTMLite"
SEEDS = "11,23,37"
TARGETS = {
    "agnews": OUT / "public_benchmark_runs_agnews.csv",
    "github": OUT / "public_benchmark_runs_githubtrain6000_test2000.csv",
}
SUMMARIES = {
    "agnews": OUT / "public_benchmark_summary_agnews.csv",
    "github": OUT / "public_benchmark_summary_github.csv",
}


def run(cmd, tag):
    print(f"\n=== [{time.strftime('%H:%M:%S')}] {tag} ===\n"
          f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    rc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT)).returncode
    print(f"=== [{time.strftime('%H:%M:%S')}] {tag} exit={rc} ===",
          flush=True)
    return rc


def main() -> int:
    for corpus, target in TARGETS.items():
        rc = run([PY, "-u", "benchmark_public.py",
                  "--dataset", corpus,
                  "--models", MODELS,
                  "--seeds", SEEDS,
                  "--k", "30",
                  "--out", f"rerun_main_baselines_{corpus}.csv",
                  "--summary", f"rerun_main_baselines_{corpus}_summary.csv"],
                 f"benchmark rerun {corpus}")
        if rc != 0:
            return rc
        rc = run([PY, "-u", "merge_benchmark_rows.py",
                  target,
                  OUT / f"rerun_main_baselines_{corpus}.csv",
                  "--summary", SUMMARIES[corpus]],
                 f"merge {corpus} rows")
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
