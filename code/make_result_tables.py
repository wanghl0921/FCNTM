# -*- coding: utf-8 -*-
"""Assemble the numbers of Tables 1/2 (and the ablation/occlusion tables).

Sources
  outputs/public_benchmark_runs_<ds>.csv   baselines, 3 seeds (old code)
  outputs/rerun_btm_<ds>.csv               BTM with the corrected Gibbs sampler
  outputs/rerun_etm_<ds>.csv               ETM with the standard decoder (1 seed)
  outputs/codev_*.csv                      FC-NTM rows
  outputs/irbo_summary.csv                 every model, corrected IRBO

    python make_result_tables.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

DATASETS = {
    "agnews": ("AG News", OUT / "public_benchmark_runs_agnews.csv",
               OUT / "codev_agnews_gate.csv"),
    "github": ("GitHub issues",
               OUT / "public_benchmark_runs_githubtrain6000_test2000.csv",
               OUT / "codev_github_gate_bal.csv"),
}

# the re-run main results (current implementation).  When present they replace
# the historical codev_* files for the FC-NTM rows.
FINAL_FCNTM = {
    "agnews": OUT / "fcntm_agnews_final.csv",
    "github": OUT / "fcntm_github_final.csv",
}

# the result files use a few implementation names that differ from the paper's
ALIASES = {"CTM-lite": "CTMLite", "ProdLDA": "NeuralLDA"}


def corrected_irbo(source: str, model: str) -> float:
    # ``irbo_summary.csv`` is keyed by the *file* model name, so the paper's
    # name has to be translated first (ProdLDA is stored as NeuralLDA);
    # otherwise the lookup missed and the IRBO column came out as NaN.
    frame = pd.read_csv(OUT / "irbo_summary.csv")
    row = frame[(frame["source"] == source)
                & (frame["model"] == ALIASES.get(model, model))]
    if row.empty:
        return float("nan")
    return float(row["IRBO_corrected_mean"].iloc[0])


def load_etm(tag: str) -> pd.DataFrame:
    """Both ETM batches (seed 11, then 23/37) under one protocol.

    ETM was re-run twice: ``rerun_etm_<tag>.csv`` holds seed 11 and
    ``rerun_etm_<tag>_seeds2337.csv`` holds seeds 23 and 37.  Both come from
    ``benchmark_public.py --models ETM --k 30`` with the default ctfidf
    read-out, so they are concatenated to give the 3-seed mean.  (The paper
    used to quote the single seed-11 run.)
    """
    # ``_fixed`` holds the same three seeds refit after the shared word-embedding
    # buffer was fixed (``EmbeddingDecoder`` used to wrap the caller's float32
    # array without copying, so the second seed in each process trained on the
    # first seed's updated embeddings).  Prefer it; fall back to the original
    # two batches.
    parts = [OUT / f"rerun_etm_{tag}_fixed.csv",
             OUT / f"rerun_etm_{tag}.csv",
             OUT / f"rerun_etm_{tag}_seeds2337.csv"]
    existing = [p for p in parts if p.exists()]
    if existing and existing[0].name.endswith("_fixed.csv"):
        existing = existing[:1]
    frames = [pd.read_csv(p) for p in existing]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def table_rows(tag: str) -> pd.DataFrame:
    """Per-model means/stds for one corpus, with the corrected baselines merged."""
    name, baseline_path, fcntm_path = DATASETS[tag]
    final_path = FINAL_FCNTM.get(tag)
    fcntm_is_final = bool(
        final_path and final_path.exists()
        and len(pd.read_csv(final_path)) > 0)
    if fcntm_is_final:
        fcntm_path = final_path
    base = pd.read_csv(baseline_path)
    btm = pd.read_csv(OUT / f"rerun_btm_{tag}.csv")
    etm = load_etm(tag)
    fcntm = pd.read_csv(fcntm_path)
    base_src = baseline_path.name
    rows = []
    for model in ["BTM", "CombinedTM", "FC-NTM-cTFIDF", "FC-NTM", "ETM",
                  "ProdLDA", "Top2Vec", "NMF", "BERTopic",
                  "CTMLite", "LDA"]:
        if model == "BTM":
            g, source = btm, None
        elif model == "ETM":
            g, source = etm, None
        elif model.startswith("FC-NTM"):
            g = fcntm[fcntm["model"] == model]
            # the re-run CSV already stores IRBO with the corrected definition
            source = None if fcntm_is_final else fcntm_path.name
        else:
            g = base[base["model"] == ALIASES.get(model, model)]
            source = base_src
        if g.empty:
            continue
        rows.append({
            "model": model,
            "file_model": ALIASES.get(model, model),
            "n": len(g),
            "CV": g["CV"].mean(), "CVsd": g["CV"].std(ddof=1) if len(g) > 1 else 0.0,
            "NPMI": g["NPMI"].mean(),
            "NPMIsd": g["NPMI"].std(ddof=1) if len(g) > 1 else 0.0,
            "UMass": g["UMass"].mean(),
            "EXCL_KADJ": g["EXCL_KADJ"].mean(),
            "IRBO": (g["IRBO"].mean() if source is None
                     else corrected_irbo(source, model)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    for tag, (name, baseline_path, fcntm_path) in DATASETS.items():
        base = pd.read_csv(baseline_path)
        btm = pd.read_csv(OUT / f"rerun_btm_{tag}.csv")
        # both ETM batches, all three seeds (see load_etm)
        etm = load_etm(tag)
        fcntm = pd.read_csv(fcntm_path)
        final_path = FINAL_FCNTM.get(tag)
        if final_path and final_path.exists() and len(pd.read_csv(final_path)):
            fcntm = pd.read_csv(final_path)
        base_src = baseline_path.name
        print("=" * 78)
        print(name)
        print("=" * 78)
        rows = []
        # same model list and file-name aliases as ``table_rows``: ProdLDA is
        # stored as NeuralLDA in the per-seed CSVs, so iterating both names
        # printed a duplicate (empty) row.
        models = ["BTM", "CombinedTM", "FC-NTM-cTFIDF", "FC-NTM", "ETM",
                  "ProdLDA", "Top2Vec", "NMF", "BERTopic", "CTMLite", "LDA"]
        for model in models:
            if model == "BTM":
                g, source, mark = btm, None, " (fixed)"
            elif model == "ETM":
                g, source, mark = etm, None, " (fixed, 3 seeds)"
            elif model.startswith("FC-NTM"):
                g = fcntm[fcntm["model"] == model]
                source, mark = None, " (final)"
            else:
                g = base[base["model"] == ALIASES.get(model, model)]
                source, mark = base_src, ""
            if g.empty:
                continue
            n = len(g)
            cv, cv_sd = g["CV"].mean(), g["CV"].std(ddof=1)
            npmi, npmi_sd = g["NPMI"].mean(), g["NPMI"].std(ddof=1)
            rows.append({
                "model": model, "n": n,
                "CV": f"{cv:.3f}$\\pm${cv_sd:.3f}" if n > 1 else f"{cv:.3f}",
                "NPMI": (f"{npmi:.3f}$\\pm${npmi_sd:.3f}" if n > 1
                         else f"{npmi:+.3f}"),
                "UMass": f"{g['UMass'].mean():.2f}",
                "EXCL": f"{g['EXCL_KADJ'].mean():.3f}",
                "IRBO": (f"{g['IRBO'].mean():.3f}" if source is None
                         else f"{corrected_irbo(source, model):.3f}"),
                "themes": (f"{g['topic_count'].mean():.1f}"
                           if "topic_count" in g.columns
                           else f"{g['K_used'].mean():.1f}"),
                "note": mark,
            })
        print(pd.DataFrame(rows).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
