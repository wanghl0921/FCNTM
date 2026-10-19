# -*- coding: utf-8 -*-
"""Assemble the civil-engineering main table (Table for the paper).

Sources
  outputs/fair_readout_civil_3seed.csv        fair read-out (LDA, NMF, BTM,
                                              ProdLDA, ETM, CTMLite)
  outputs/rerun_main_baselines_civil_summary.csv  public baselines
                                              (CombinedTM, BERTopic, Top2Vec)
  outputs/fcntm_civil_final.csv               FC-NTM rows (beta + cTFIDF)

Prints LaTeX-ready rows in the same column layout as the other main tables
(TopiCs, C_V, NPMI, UMass, EXCL_KADJ, IRBO), 3-seed means.

Usage:
    python make_civil_table.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def fmt(mean: float, sd: float, n: int) -> str:
    return f"{mean:.3f}$\\pm${sd:.3f}" if n > 1 else f"{mean:.3f}"


def main() -> None:
    fair = pd.read_csv(OUT / "fair_readout_civil_3seed.csv")
    fair = fair[fair["readout"] == "beta"]
    bench_path = OUT / "rerun_main_baselines_civil_summary.csv"
    bench = pd.read_csv(bench_path) if bench_path.exists() else pd.DataFrame()
    fcntm = pd.read_csv(OUT / "fcntm_civil_final.csv")

    rows = []

    def add(model, g, label=None):
        n = len(g)
        rows.append({
            "model": label or model,
            "topics": int(round(g["topic_count"].mean())) if "topic_count" in g
            else int(round(g["K_used"].mean())) if "K_used" in g else 30,
            "CV": fmt(g["CV"].mean(), g["CV"].std(ddof=1) if n > 1 else 0.0, n),
            "NPMI": fmt(g["NPMI"].mean(),
                        g["NPMI"].std(ddof=1) if n > 1 else 0.0, n),
            "UMass": f"{g['UMass'].mean():.2f}",
            "EXCL_KADJ": f"{g['EXCL_KADJ'].mean():.3f}",
            "IRBO": f"{g['IRBO'].mean():.3f}",
            "n": n,
        })

    # FC-NTM rows from the final run
    for name in ("FC-NTM-cTFIDF", "FC-NTM"):
        g = fcntm[fcntm["model"] == name]
        if not g.empty:
            add(name, g)

    # classical + neural fair read-out baselines
    for name in ("BTM", "NMF", "LDA", "ProdLDA", "ETM", "CTMLite"):
        g = fair[fair["model"] == name]
        if not g.empty:
            add(name, g, "CTM-lite" if name == "CTMLite" else name)

    # embedding-clustering and CombinedTM baselines from the public runner
    if not bench.empty:
        for name, label in (("CombinedTM", "CombinedTM"),
                            ("BERTopic", "BERTopic"),
                            ("Top2Vec", "Top2Vec")):
            g = bench[bench["model"] == name]
            if g.empty:
                continue
            rows.append({
                "model": label,
                "topics": int(round(g["topic_count_mean"].iloc[0])),
                "CV": fmt(float(g["CV_mean"].iloc[0]),
                          float(g["CV_std"].iloc[0]),
                          int(g["n_seeds"].iloc[0])),
                "NPMI": fmt(float(g["NPMI_mean"].iloc[0]),
                            float(g["NPMI_std"].iloc[0]),
                            int(g["n_seeds"].iloc[0])),
                "UMass": f"{float(g['UMass_mean'].iloc[0]):.2f}",
                "EXCL_KADJ": f"{float(g['EXCL_KADJ_mean'].iloc[0]):.3f}",
                "IRBO": f"{float(g['IRBO_mean'].iloc[0]):.3f}",
                "n": int(g["n_seeds"].iloc[0]),
            })

    frame = pd.DataFrame(rows)
    order = ["BTM", "CombinedTM", "FC-NTM-cTFIDF", "FC-NTM", "ETM",
             "ProdLDA", "Top2Vec", "NMF", "BERTopic", "CTM-lite", "LDA"]
    frame["__o"] = frame["model"].apply(
        lambda m: order.index(m) if m in order else len(order))
    frame = frame.sort_values("__o").drop(columns="__o")
    print(frame.to_string(index=False))
    out = OUT / "civil_main_table.csv"
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    print("saved", out)


if __name__ == "__main__":
    main()
