# -*- coding: utf-8 -*-
"""Cross-corpus comparison of the partial field-occlusion curves.

Reads outputs/occlusion_robustness_<corpus>_seed<k>.csv (produced by
occlusion_robustness.py) and reports, per model and corpus:

  NLL@25%, NLL@100%      per-token NLL of the occluded background tokens
  relative degradation   NLL(100%) / NLL(25%)  (calibration-free)
  agreement@100%         same top-1 topic as in the *unoccluded* inference
                         (rate 0); the script's ``run_model`` takes rate 0 as
                         the base, not rate 0.25
  adv_vs_best_baseline   relative degradation of the best non-FC-NTM model
                         minus that of FC-NTM (positive = FC-NTM more stable)

Usage: python compare_occlusion.py [--seed 11]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"


def available_seeds(corpus: str) -> list:
    seeds = []
    for path in OUT.glob(f"occlusion_robustness_{corpus}_seed*.csv"):
        tag = path.stem.rsplit("seed", 1)[-1]
        if tag.isdigit():
            seeds.append(int(tag))
    return sorted(seeds)


def summarise(corpus: str, seed: int) -> pd.DataFrame:
    path = OUT / f"occlusion_robustness_{corpus}_seed{seed}.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    rows = []
    for model, g in frame.groupby("model"):
        g = g.set_index("rate").sort_index()
        if 0.25 not in g.index or 1.0 not in g.index:
            continue
        nll25 = float(g.loc[0.25, "hidden_NLL"])
        nll100 = float(g.loc[1.0, "hidden_NLL"])
        rows.append({
            "corpus": corpus, "model": model,
            "NLL@25": nll25, "NLL@100": nll100,
            "rel_degradation": nll100 / nll25,
            "agreement@100": float(g.loc[1.0, "agreement"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0,
                    help="0 = every seed found on disk")
    ap.add_argument("--corpora", default="github,agnews")
    args = ap.parse_args()
    frames = []
    for corpus in args.corpora.split(","):
        seeds = ([args.seed] if args.seed
                 else available_seeds(corpus))
        for seed in seeds:
            frame = summarise(corpus, seed)
            if not frame.empty:
                frame["seed"] = seed
                frames.append(frame)
    table = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if table.empty:
        print("no occlusion results yet")
        return
    raw = table.copy()
    # keep full precision in the file the paper aggregates from: rounding the
    # stored ratios first turned every later mean into a mean of rounded values
    # (the fourth review spotted the 4-decimal truncation).  Only the printed
    # view is rounded.
    print(table.round(4).to_string(index=False))
    # one row per (corpus, model) only holds for a single seed; with several
    # seeds on disk the mean is reported here and the pairing is done below
    pivot = table.pivot_table(index="model", columns="corpus",
                              values="rel_degradation", aggfunc="mean")
    print("\nmean relative degradation (NLL(100%)/NLL(25%)):")
    print(pivot.round(4).to_string())
    # seed-matched advantage: compare with the best baseline *of the same
    # seed* so that both numbers share the same masking realisation
    print("\nseed-matched advantage over the best baseline of the same seed:")
    for corpus in sorted(raw["corpus"].unique()):
        sub = raw[raw["corpus"] == corpus]
        advantages, winners = [], []
        for _seed, g in sub.groupby("seed"):
            rel = g.set_index("model")["rel_degradation"]
            if "FC-NTM" not in rel.index or len(rel) < 2:
                continue
            baselines = rel.drop("FC-NTM")
            advantages.append(float(baselines.min() - rel["FC-NTM"]))
            winners.append(str(baselines.idxmin()))
        if advantages:
            arr = np.asarray(advantages, dtype=float)
            sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            print(f"{corpus}: FC-NTM advantage {arr.mean():+.4f} "
                  f"+/- {sd:.4f} over {arr.size} seed(s); "
                  f"positive in {int((arr > 0).sum())}/{arr.size} "
                  f"(best baseline per seed: {', '.join(winners)})")
    table.to_csv(OUT / "occlusion_comparison.csv",
                 index=False, encoding="utf-8-sig")
    print("\nsaved outputs/occlusion_comparison.csv")
    if args.seed == 0:
        agg = (table.groupby(["corpus", "model"])
               .agg(rel_mean=("rel_degradation", "mean"),
                    rel_std=("rel_degradation", "std"),
                    n=("rel_degradation", "size"),
                    agree_mean=("agreement@100", "mean"),
                    nll100_mean=("NLL@100", "mean"))
               .round(4))
        print("\nacross seeds:")
        print(agg.to_string())
        for corpus, g in agg.reset_index().groupby("corpus"):
            fc = g[g.model == "FC-NTM"]
            base = g[g.model != "FC-NTM"]
            if not fc.empty and not base.empty:
                print(f"{corpus}: FC-NTM {fc.rel_mean.iloc[0]:.4f} vs best "
                      f"baseline {base.rel_mean.min():.4f} -> advantage "
                      f"{base.rel_mean.min() - fc.rel_mean.iloc[0]:+.4f}")


if __name__ == "__main__":
    main()
