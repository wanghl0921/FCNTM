# -*- coding: utf-8 -*-
"""Re-run the BTM baseline with the corrected collapsed-Gibbs conditional.

The conditional probability used here is the one implied by integrating the
topic-word Dirichlet out of the biterm likelihood (Yan et al. 2013, eq. 4):

    p(z_b = k | -b)  \\propto  (n_k^-b + alpha)
        (n_kw_i^-b + eta)(n_kw_j^-b + eta + [w_i = w_j])
        / [(2 n_k^-b + V eta)(2 n_k^-b + V eta + 1)]

An earlier revision omitted the ``+1`` factor and the [w_i = w_j] indicator.
Both the sampling and the metrics are identical to
``benchmark_public.py``, so the rows written here can be merged into the
benchmark runs CSV with ``merge_benchmark_rows.py``.

Usage:
    python rerun_btm.py --dataset github --seeds 11,23,37
    python rerun_btm.py --dataset agnews --seeds 11,23,37
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baselines as BL
import metrics as M
from config import OUT_DIR, TOPN


def load_dataset(name: str):
    if name == "agnews":
        import public_data as PD
        return PD.load_agnews(1500, 1900, need_train_bow=False)
    import public_data_github as PGH
    return PGH.load_github(1500, 500, need_train_bow=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["agnews", "github"], default="github")
    ap.add_argument("--seeds", default="11,23,37")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--max-pairs", type=int, default=150)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = load_dataset(args.dataset)
    train, test = data["train"], data["test"]
    vocab = train["vocab"]
    seeds = [int(s) for s in args.seeds.split(",")]

    fieldnames = ["model", "seed", "K_target", "seconds", "topic_count",
                  "n_outlier_docs", "epochs_run", "best_epoch", "topics",
                  "n_biterms", "window", "max_pairs", "iterations",
                  "CV_n", "CV", "NPMI_n", "NPMI", "UCI_n", "UCI",
                  "UMass_n", "UMass", "EXCL", "EXCL_KADJ", "IRBO",
                  "status"]
    out_path = Path(args.out) if args.out else (
        OUT_DIR / f"rerun_btm_{args.dataset}.csv")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for seed in seeds:
            t0 = time.time()
            rec = BL.run_btm(
                train["doc_tokens"], train["dictionary"], vocab, args.k, seed,
                iterations=args.iterations, window=args.window,
                max_pairs=args.max_pairs)
            topics = rec["topics"]
            phi = np.asarray(rec["beta"], dtype=float)
            ev = M.coherence_scores(
                topics, test["doc_tokens"], test["corpus"],
                test["dictionary"], topn=TOPN)
            ev["EXCL"] = M.exclusivity(phi, topn=TOPN)
            ev["EXCL_KADJ"] = M.exclusivity_kadj(phi, topn=TOPN)
            ev["IRBO"] = M.irbo_diversity(topics)
            row = {
                "model": "BTM", "seed": seed, "K_target": args.k,
                "seconds": round(time.time() - t0, 2),
                "topic_count": rec["topic_count"],
                "n_outlier_docs": 0, "epochs_run": "", "best_epoch": "",
                "topics": json.dumps(topics, ensure_ascii=False),
                "n_biterms": rec["n_biterms"],
                "window": rec["window"], "max_pairs": rec["max_pairs"],
                "iterations": rec["iterations_run"],
                "status": "ok",
                **{k: (round(float(v), 6) if isinstance(v, (int, float, np.floating))
                       else v) for k, v in ev.items()},
            }
            writer.writerow(row)
            fh.flush()
            print(f"  BTM seed {seed} CV={ev['CV']:.4f} NPMI={ev['NPMI']:.4f} "
                  f"IRBO={ev['IRBO']:.4f} ({row['seconds']}s)", flush=True)
    print("saved", out_path)


if __name__ == "__main__":
    main()
