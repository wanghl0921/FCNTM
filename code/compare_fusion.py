# -*- coding: utf-8 -*-
"""Compare two per-seed result CSVs written by run_fcntm_train.py.

Usage:
    python compare_fusion.py A.csv B.csv [name_a] [name_b]

Prints, for every ``model`` present in both files:
  * mean +/- std of the main metrics in each file;
  * the difference of the means;
  * the paired per-seed difference on the seeds the two files share.

Both files must carry the columns written by ``run_fcntm_train.py``:
corpus, model, seed, K_used, best_epoch, seconds, topics, CV, NPMI, UCI,
UMass, EXCL, EXCL_KADJ, IRBO.
"""

from __future__ import annotations

import sys

import pandas as pd

METRICS = ["CV", "NPMI", "UCI", "UMass", "EXCL_KADJ", "IRBO"]


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    a = pd.read_csv(sys.argv[1])
    b = pd.read_csv(sys.argv[2])
    name_a = sys.argv[3] if len(sys.argv) > 3 else "A"
    name_b = sys.argv[4] if len(sys.argv) > 4 else "B"
    for df in (a, b):
        for metric in METRICS:
            if metric not in df.columns:
                raise SystemExit(f"missing column {metric!r}")
    models = sorted(set(a["model"]) & set(b["model"]))
    if not models:
        raise SystemExit("no model name present in both files")
    for model in models:
        ta = a[a["model"] == model]
        tb = b[b["model"] == model]
        print(f"\n=== {model} ===")
        rows = []
        for metric in METRICS:
            rows.append({
                "metric": metric,
                f"{name_a}_mean": ta[metric].mean(),
                f"{name_a}_std": ta[metric].std(ddof=1),
                f"{name_b}_mean": tb[metric].mean(),
                f"{name_b}_std": tb[metric].std(ddof=1),
                "delta": tb[metric].mean() - ta[metric].mean(),
            })
        print(pd.DataFrame(rows).set_index("metric").round(4).to_string())
        merged = ta.merge(tb, on="seed", suffixes=(f"_{name_a}", f"_{name_b}"))
        if len(merged):
            paired = pd.DataFrame({
                "seed": merged["seed"].astype(int),
                **{m: merged[f"{m}_{name_b}"] - merged[f"{m}_{name_a}"]
                   for m in METRICS},
            })
            print(f"\npaired per-seed delta ({name_b} - {name_a}):")
            print(paired.round(4).to_string(index=False))
            print(f"n_paired = {len(paired)}, seeds = {list(paired['seed'])}")
            for m in ("CV", "NPMI"):
                wins = int((paired[m] > 0).sum())
                print(f"  {m}: {wins}/{len(paired)} seeds positive")


if __name__ == "__main__":
    main()
