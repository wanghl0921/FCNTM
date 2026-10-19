# -*- coding: utf-8 -*-
"""Aggregate outputs/variant_public.csv into a per-variant comparison table.

Prints, for every corpus and variant:
  * mean/std over seeds of the two read-outs' C_V / NPMI / UMass / EXCL_KADJ
  * cross-field imputation recall@10 / precision@10
  * paired deltas against the ``base`` variant (same seeds)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
METRICS = ["CV", "NPMI", "UMass", "EXCL_KADJ", "IRBO",
           "recall", "precision", "cause_CV", "cause_NPMI", "item_CV"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(OUT / "variant_public.csv"))
    ap.add_argument("--base", default="base")
    args = ap.parse_args()

    df = pd.read_csv(args.path)
    df = df[df["variant"].notna()]
    for corpus, block in df.groupby("corpus", sort=False):
        print("=" * 100)
        print(f"corpus = {corpus}   (seeds: "
              f"{sorted(block['seed'].unique().tolist())}, "
              f"epochs={sorted(block['epochs'].unique().tolist())})")
        print("=" * 100)
        for readout in ("beta", "cTFIDF"):
            sub = block[block["readout"] == readout]
            agg = sub.groupby("variant")[METRICS].agg(["mean", "std"])
            base = sub[sub["variant"] == args.base].set_index("seed")
            print(f"\n--- read-out: {readout} ---")
            header = (f"{'variant':8s} {'CV':>16s} {'NPMI':>16s} {'UMass':>16s} "
                      f"{'EXCL_KADJ':>10s} {'IRBO':>7s} {'recall':>7s} "
                      f"{'prec':>7s} {'causeCV':>7s} {'itemCV':>7s}")
            print(header)
            for name, row in agg.iterrows():
                def fmt(m):
                    mean, std = row[(m, "mean")], row[(m, "std")]
                    std = 0.0 if pd.isna(std) else std
                    return f"{mean:7.3f}±{std:.3f}"
                print(f"{name:8s} {fmt('CV'):>16s} {fmt('NPMI'):>16s} "
                      f"{fmt('UMass'):>16s} {row[('EXCL_KADJ','mean')]:10.3f} "
                      f"{row[('IRBO','mean')]:7.3f} {row[('recall','mean')]:7.3f} "
                      f"{row[('precision','mean')]:7.3f} "
                      f"{row[('cause_CV','mean')]:7.3f} "
                      f"{row[('item_CV','mean')]:7.3f}")
            if not base.empty:
                print(f"\n  paired delta vs '{args.base}' (per seed):")
                for name, sub_v in sub.groupby("variant"):
                    if name == args.base:
                        continue
                    cur = sub_v.set_index("seed")
                    common = sorted(set(cur.index) & set(base.index))
                    if not common:
                        continue
                    parts = []
                    for m in ("CV", "NPMI", "recall", "EXCL_KADJ"):
                        d = cur.loc[common, m].to_numpy(float) - \
                            base.loc[common, m].to_numpy(float)
                        wins = int(np.sum(d > 0))
                        parts.append(f"{m}{'+' if d.mean() >= 0 else ''}"
                                     f"{d.mean():.3f}({wins}/{len(common)})")
                    print(f"    {name:8s} " + "  ".join(parts))


if __name__ == "__main__":
    main()
