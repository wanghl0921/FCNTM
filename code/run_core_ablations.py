# -*- coding: utf-8 -*-
"""The final-protocol core-mechanism ablations (fifth review, item 3).

The paper calls cross-field reconstruction, the JS consistency term, field-level
word masking and field dropout the components of FC-NTM, but the final protocol
only ablated the fusion objective (mean vs gated, and the fusion weight).  This
chain runs the four component ablations with the *same* budget, read-out and
seeds as the main results:

    no-cross        cross_reconstruction=False
    no-consistency  consistency_weight=0
    no-mask         cause_dropout=item_dropout=0
    no-fielddrop    field_dropout=0

GitHub runs first (it is the corpus where the mechanism matters), then AG News;
rows are flushed as they are produced, so a partial file is still usable.  The
summary compares each ablation with the full model of the same corpus using
seed-matched differences.

    python run_core_ablations.py                     # both corpora, 4 configs
    python run_core_ablations.py --corpus github --configs no-cross
    python run_core_ablations.py --threads 2          # if memory is tight

Each fit costs roughly 20--30 minutes (GitHub, 700 epochs) to about an hour on
this CPU-only machine, so the chain is resumable: rows are flushed as they are
scored and a rerun skips the (config, seed) pairs that are already complete in
``outputs/ablation_public_<corpus>_core.csv``.  Run it with nothing else
training at the same time (进度记录.md §2: two concurrent trainings OOM here).

Outputs
    outputs/ablation_public_github_core.csv
    outputs/ablation_public_agnews_core.csv
    outputs/rerun/core_ablation_summary.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PY = sys.executable
DEFAULT_CONFIGS = "no-cross,no-consistency,no-mask,no-fielddrop"
SEEDS = "11,23,37"
READOUTS = ("beta", "cTFIDF")


def child_env(threads: int) -> dict:
    """Pin the BLAS/OpenMP thread counts for the child fits.

    The machine has 8 GB of RAM and no GPU; leaving these unset is what killed
    the first attempt at this chain with ``DefaultCPUAllocator: not enough
    memory`` (see 进度记录.md §2).  The values only bound the thread pools, so
    they cannot change the mathematics of a fit.
    """
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[var] = str(int(threads))
    return env


def run(cmd, tag, env=None):
    print(f"\n=== [{time.strftime('%H:%M:%S')}] {tag} ===\n"
          f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    rc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env).returncode
    print(f"=== [{time.strftime('%H:%M:%S')}] {tag} exit={rc} ===",
          flush=True)
    return rc


def summarise(corpus: str) -> pd.DataFrame:
    path = OUT / f"ablation_public_{corpus}_core.csv"
    final = OUT / f"fcntm_{corpus}_final.csv"
    if not path.exists() or not final.exists():
        return pd.DataFrame()
    abl = pd.read_csv(path)
    full = pd.read_csv(final)
    rows = []
    for config, group in abl.groupby("config"):
        for readout in READOUTS:
            g = group[group["readout"] == readout].sort_values("seed")
            if g.empty:
                continue
            model = "FC-NTM" if readout == "beta" else "FC-NTM-cTFIDF"
            f = full[full["model"] == model].sort_values("seed")
            merged = g.merge(f, on="seed", suffixes=("_abl", "_full"))
            entry = {
                "corpus": corpus, "config": config, "readout": readout,
                "n_seeds": len(g),
                "CV_mean": g["CV"].mean(), "CV_sd": g["CV"].std(ddof=1),
                "NPMI_mean": g["NPMI"].mean(),
                "NPMI_sd": g["NPMI"].std(ddof=1),
            }
            if len(merged):
                for metric in ("CV", "NPMI"):
                    diff = (merged[f"{metric}_abl"]
                            - merged[f"{metric}_full"]).to_numpy()
                    entry[f"delta_{metric}_mean"] = float(diff.mean())
                    entry[f"delta_{metric}_positive"] = int((diff > 0).sum())
            rows.append(entry)
    return pd.DataFrame(rows)


def main() -> int:
    # The chain writes its log to a file via shell redirection, where Python
    # falls back to the ANSI code page (GBK here) and dies on typographic
    # characters such as U+2212.  Reconfigure instead of relying on the locale.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=["github", "agnews", "civil", "both"],
                    default="both")
    ap.add_argument("--configs", default=DEFAULT_CONFIGS)
    ap.add_argument("--seeds", default=SEEDS)
    ap.add_argument("--max-epochs", type=int, default=700)
    ap.add_argument("--threads", type=int, default=4,
                    help="BLAS/OpenMP threads handed to each child fit")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()

    corpora = ("github", "agnews") if args.corpus == "both" else (args.corpus,)
    env = child_env(args.threads)
    for corpus in corpora:
        rc = run([PY, "-u", "run_ablation_public.py", "--dataset", corpus,
                  "--seeds", args.seeds, "--k", "30", "--lr", "5e-4",
                  "--configs", args.configs, "--max-epochs",
                  str(args.max_epochs), "--resume",
                  "--out", f"ablation_public_{corpus}_core.csv"],
                 f"core ablations {corpus}", env)
        if rc != 0:
            print(f"\n[chain] {corpus} stopped with exit={rc}; the rows already "
                  f"written to outputs/ablation_public_{corpus}_core.csv are "
                  f"kept — rerun this script (it resumes) after freeing "
                  f"memory.", flush=True)
            return rc
    if args.no_summary:
        print("\n[chain] core ablations DONE (summary skipped)", flush=True)
        return 0
    summary = pd.concat([summarise(c) for c in corpora], ignore_index=True)
    if not summary.empty:
        (OUT / "rerun").mkdir(parents=True, exist_ok=True)
        summary.to_csv(OUT / "rerun" / "core_ablation_summary.csv",
                       index=False, encoding="utf-8-sig")
        show = summary.copy()
        for column in show.columns:
            if show[column].dtype.kind == "f":
                show[column] = show[column].round(4)
        print("\ncore-mechanism ablations (difference = ablation - full model)")
        print(show.to_string(index=False))
    print("\n[chain] core ablations DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
