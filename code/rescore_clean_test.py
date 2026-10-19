# -*- coding: utf-8 -*-
"""Re-score every stored topic list on the development-free test subset.

The design decisions behind the current model (gate fusion, balanced fusion
normalisation, variant screening) were taken on 400 documents drawn from the
official test split.  This script keeps those documents out and recomputes the
coherence metrics of every stored run on the disjoint remainder, so the final
numbers can be reported on documents that were not used for development.

    python rescore_clean_test.py --corpus github,agnews

Outputs
    outputs/clean_test_scores.csv     per (source, model, seed, read-out)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import dev_split
import metrics as M
from config import TOPN

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

SOURCES = {
    "agnews": [
        OUT / "public_benchmark_runs_agnews.csv",
        # BTM's stored topic lists: native beta + the class-TF-IDF read-out
        # built from its document-topic posterior (Yan et al. section 3.3)
        OUT / "rerun_btm_agnews_readouts.csv",
        OUT / "rerun_etm_agnews_fixed.csv",
        OUT / "codev_agnews_gate.csv",
        OUT / "fcntm_agnews_final.csv",
    ],
    "github": [
        OUT / "public_benchmark_runs_githubtrain6000_test2000.csv",
        OUT / "rerun_btm_github_readouts.csv",
        OUT / "rerun_etm_github_fixed.csv",
        OUT / "codev_github_gate_bal.csv",
        OUT / "fcntm_github_final.csv",
    ],
}


def load_clean(corpus: str):
    if corpus == "agnews":
        import public_data as PD
        data = PD.load_agnews(1500, 1900, need_train_bow=False)
    else:
        import public_data_github as PGH
        data = PGH.load_github(1500, 500, need_train_bow=False)
    test = data["test"]
    mask = dev_split.clean_test_mask(corpus, test["df"]).to_numpy()
    idx = np.where(mask)[0]
    doc_tokens = [test["doc_tokens"][i] for i in idx]
    bows = [test["dictionary"].doc2bow(t) for t in doc_tokens]
    print(f"[{corpus}] clean test docs: {len(doc_tokens)} "
          f"(excluded {int((~mask).sum())} development documents)", flush=True)
    return doc_tokens, bows, test["dictionary"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="github,agnews")
    ap.add_argument("--out", default="outputs/clean_test_scores.csv")
    args = ap.parse_args()

    rows = []
    for corpus in args.corpus.split(","):
        doc_tokens, bows, dictionary = load_clean(corpus)
        for path in SOURCES[corpus]:
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if "topics" not in frame.columns:
                continue
            for _, r in frame.iterrows():
                try:
                    topics = json.loads(r["topics"])
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(topics, list) or not topics:
                    continue
                ev = M.coherence_scores(topics, doc_tokens, bows, dictionary,
                                        topn=TOPN)
                rows.append({
                    "corpus": corpus,
                    "source": path.name,
                    "model": r.get("model", ""),
                    "seed": r.get("seed", ""),
                    "readout": r.get("readout", ""),
                    "n_topics": len(topics),
                    "CV": round(float(ev["CV"]), 6),
                    "NPMI": round(float(ev["NPMI"]), 6),
                    "UMass": round(float(ev["UMass"]), 6),
                    "CV_full_test": r.get("CV"),
                    "NPMI_full_test": r.get("NPMI"),
                })
                print(f"  {corpus} {path.name} {r.get('model')} "
                      f"seed={r.get('seed')} CV={ev['CV']:.4f} "
                      f"(full test {r.get('CV')})", flush=True)
    frame = pd.DataFrame(rows)
    out_path = ROOT / args.out
    frame.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("saved", out_path)


if __name__ == "__main__":
    main()
