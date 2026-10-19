# -*- coding: utf-8 -*-
"""Recompute every NMF row that depended on the broken normalisation.

Two places applied ``normalize(W) @ normalize(H)`` (or treated the raw W as a
document mixture), which is not the NMF reconstruction:

  * ``occlusion_robustness.NMFAdapter`` -> the NMF curves of the six
    ``occlusion_robustness_<corpus>_seed<k>.csv`` files
  * ``baselines.run_nmf`` -> the NMF rows of the two ``fair_readout_*_3seed.csv``

Both are fixed in the code; this script re-scores the affected rows (NMF refits
are seconds) and merges them back, leaving every other model untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

import baselines as BL
import metrics as M
import occlusion_robustness as O
from config import TOPN
from run_fcntm_train import class_tfidf, top_words

RATES = [0.0, 0.25, 0.5, 0.75, 1.0]
OUT = O.OUT_DIR


def masks_for(test, seed: int):
    rng = np.random.RandomState(seed + 7)
    seeds = rng.randint(0, 10 ** 6, size=len(test["reason_tokens"]))
    visible, hidden = {}, {}
    for rate in RATES:
        vis, hid = [], []
        for i, toks in enumerate(test["reason_tokens"]):
            v, h = O.mask_tokens(toks, rate, np.random.RandomState(seeds[i]))
            vis.append(v)
            hid.append(h)
        visible[rate], hidden[rate] = vis, hid
    return visible, hidden


def main() -> int:
    for corpus in ("github", "agnews"):
        train, test = O.load_corpus(corpus)
        vocab = test["vocab"]
        word2id = test["dictionary"].token2id
        fair_rows = []
        for seed in (11, 23, 37):
            # --- occlusion rows -------------------------------------------
            visible, hidden = masks_for(test, seed)
            adapter = O.NMFAdapter(train, 30, seed)
            adapter.set_word2id(word2id)
            rows = O.run_model(adapter, visible, test["item_tokens"], hidden,
                               RATES, word2id, "NMF")
            path = OUT / f"occlusion_robustness_{corpus}_seed{seed}.csv"
            prev = pd.read_csv(path)
            keep = prev[prev["model"] != "NMF"]
            pd.concat([keep, pd.DataFrame(rows)], ignore_index=True).to_csv(
                path, index=False, encoding="utf-8-sig")
            print(f"  occlusion {path.name}: NMF recomputed", flush=True)

            # --- fair-readout rows ----------------------------------------
            # ``O.load_corpus`` skips the (docs x V) doc_bow matrix to save
            # memory; the field matrices sum to exactly the same counts.
            doc_counts = (train["reason_bow"] + train["item_bow"]).astype(float)
            rec = BL.run_nmf(sp.csr_matrix(doc_counts.astype(np.float32)),
                             vocab, 30, seed)
            test_scores = lambda mat, topics: M.coherence_scores(  # noqa: E731
                topics, test["doc_tokens"], test["corpus"],
                test["dictionary"], topn=TOPN)
            out = []
            tb = test_scores(rec["beta"], rec["topics"])
            out.append({"source": "fair_readout", "model": "NMF", "seed": seed,
                        "readout": "beta", "CV": tb["CV"], "NPMI": tb["NPMI"],
                        "UMass": tb["UMass"]})
            soft = class_tfidf(rec["theta"].T @ doc_counts)
            topics_ctfidf = top_words(soft, vocab)
            tc = test_scores(soft, topics_ctfidf)
            out.append({"source": "fair_readout", "model": "NMF", "seed": seed,
                        "readout": "ctfidf", "CV": tc["CV"], "NPMI": tc["NPMI"],
                        "UMass": tc["UMass"]})
            fair_rows.extend(out)
        # merge once per corpus: replacing inside the seed loop would keep only
        # the last seed's rows
        fp = OUT / f"fair_readout_{corpus}_3seed.csv"
        fprev = pd.read_csv(fp)
        fkeep = fprev[fprev["model"] != "NMF"]
        pd.concat([fkeep, pd.DataFrame(fair_rows)], ignore_index=True).to_csv(
            fp, index=False, encoding="utf-8-sig")
        print(f"  fair_readout {fp.name}: NMF recomputed for "
              f"{len(fair_rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
