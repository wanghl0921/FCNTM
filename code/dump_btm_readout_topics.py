# -*- coding: utf-8 -*-
"""Persist BTM's two read-outs on the full test split as topic lists.

``rescore_clean_test.py`` re-scores *stored topic lists* on the
development-free test subset, so a model can only enter that table if its
topic-word lists were written out.  BTM's class-TF-IDF read-out (native
``beta`` -> ``p(z|d)`` from Yan et al. 2013 section 3.3 -> ``theta^T counts``
-> ``class_tfidf``, exactly as ``fair_readout.evaluate_both`` does it) had never
been saved, which is why the clean-test table could only compare
FC-NTM-cTFIDF against BTM's *native* read-out.

This script fits BTM once per corpus and seed and stores both read-outs, so the
clean-test re-scoring can compare under the *matched* read-out.

    python dump_btm_readout_topics.py

Outputs
    outputs/rerun/rerun_btm_<corpus>_readouts.csv
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import baselines as BL
import metrics as M
import occlusion_robustness as O
from config import TOPN
from run_fcntm_train import class_tfidf, top_words

K = 30
SEEDS = (11, 23, 37)


def score(topics, test):
    return M.coherence_scores(topics, test["doc_tokens"], test["corpus"],
                              test["dictionary"], topn=TOPN)


def main() -> int:
    for corpus in ("github", "agnews"):
        train, test = O.load_corpus(corpus)
        vocab = test["vocab"]
        doc_counts = (train["reason_bow"] + train["item_bow"]).astype(float)
        rows = []
        for seed in SEEDS:
            t0 = time.time()
            rec = BL.run_btm(train["doc_tokens"], train["dictionary"],
                             train["vocab"], K, seed)
            print(f"  [{corpus} seed {seed}] BTM fitted in "
                  f"{time.time() - t0:.0f}s, {rec['n_biterms']} biterms",
                  flush=True)
            for readout, topics in (
                ("beta", M.topics_from_matrix(np.asarray(rec["beta"], float),
                                              vocab, TOPN)),
                ("ctfidf", top_words(
                    class_tfidf(np.asarray(rec["theta"], float).T @ doc_counts),
                    vocab)),
            ):
                ev = score(topics, test)
                rows.append({
                    "model": "BTM",
                    "seed": seed,
                    "readout": readout,
                    "n_topics": len(topics),
                    "topics": json.dumps(topics, ensure_ascii=False),
                    "CV": ev["CV"],
                    "NPMI": ev["NPMI"],
                    "UMass": ev["UMass"],
                })
                print(f"    readout={readout:6s} CV={ev['CV']:.4f} "
                      f"NPMI={ev['NPMI']:.4f}", flush=True)
        out = O.OUT_DIR / f"rerun_btm_{corpus}_readouts.csv"
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  saved {out}", flush=True)
    print("\n[chain] BTM read-out topic lists DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
