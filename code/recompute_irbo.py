# -*- coding: utf-8 -*-
"""Recompute the IRBO column of every stored run under the corrected RBO.

The paper reports ``1 - RBO_trunc``, and ``RBO_trunc`` of two identical topic
lists is ``1 - p^L = 0.651`` rather than 1, so identical topics were credited
with 0.349 "diversity".  Nothing about the models changes here: the topic word
lists are already stored in the per-seed CSVs, so both definitions can be
recomputed exactly without retraining.

Outputs
    outputs/irbo_recomputed.csv          per (file, model, seed): old + new IRBO
    outputs/irbo_summary.csv             per (file, model): mean +- std of both
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import metrics as M

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

SOURCES = [
    OUT / "public_benchmark_runs_agnews.csv",
    OUT / "public_benchmark_runs_githubtrain6000_test2000.csv",
    OUT / "codev_agnews_gate.csv",
    OUT / "codev_github_gate_bal.csv",
    OUT / "fcntm_github_twomodel_per_seed.csv",
    OUT / "ablation_public_github.csv",
    OUT / "ablation_public_agnews.csv",
]


def main() -> None:
    rows = []
    for path in SOURCES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "topics" not in df.columns:
            continue
        for _, r in df.iterrows():
            try:
                topics = json.loads(r["topics"])
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(topics, list) or not topics:
                continue
            rows.append({
                "source": path.name,
                "model": r.get("model", ""),
                "seed": r.get("seed", ""),
                "readout": r.get("readout", ""),
                "config": r.get("config", ""),
                "n_topics": len(topics),
                "IRBO_paper_trunc": round(
                    M.irbo_diversity(topics, tail="trunc"), 6),
                "IRBO_corrected": round(
                    M.irbo_diversity(topics, tail="extrapolate"), 6),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "irbo_recomputed.csv", index=False,
                 encoding="utf-8-sig")
    keys = ["source", "model", "readout", "config"]
    summary = (frame.groupby(keys, dropna=False)
               .agg(n=("IRBO_corrected", "size"),
                    IRBO_paper_trunc_mean=("IRBO_paper_trunc", "mean"),
                    IRBO_corrected_mean=("IRBO_corrected", "mean"),
                    IRBO_corrected_std=("IRBO_corrected", "std"))
               .reset_index()
               .round(6))
    summary.to_csv(OUT / "irbo_summary.csv", index=False,
                   encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print("\nsaved outputs/irbo_recomputed.csv | outputs/irbo_summary.csv")


if __name__ == "__main__":
    main()
