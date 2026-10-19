# -*- coding: utf-8 -*-
"""Fit BTM for extra seeds and write the same read-outs as the 3-seed runs.

Caches and merged CSVs carry a complete fitting/evaluation identity
(seed_identity.py); a fit or a file from another configuration is refitted /
archived instead of being adopted.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import baselines as BL
import occlusion_robustness as O
from add_btm_doc_topic import masks_for
from fair_readout import evaluate_both
from seed_identity import (fair_readout_identity, fit_identity,
                           load_cached_npz, merge_pinned, occlusion_identity,
                           save_cached_npz)

RATES = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_or_fit(corpus, train, K, seed):
    fit = O.OUT_DIR / "rerun" / f"btm_fit_{corpus}_seed{seed}.npz"
    ident = fit_identity("extend_seeds_btm.fit", corpus, train, K, seed)
    cached = load_cached_npz(
        fit, ident, arrays=("beta", "theta"),
        shapes={"beta": (K, len(train["vocab"])),
                "theta": (train["n_docs"], K)})
    if cached is not None:
        print(f"  [{corpus} seed {seed}] reusing pinned BTM fit", flush=True)
        z = np.load(fit, allow_pickle=True)
        return {k: (z[k].item() if z[k].ndim == 0 else z[k])
                for k in z.files
                if k not in ("identity_id", "identity_json")}
    t0 = time.time()
    rec = BL.run_btm(train["doc_tokens"], train["dictionary"], train["vocab"],
                     K, seed)
    save_cached_npz(fit, ident, **rec)
    print(f"  [{corpus} seed {seed}] BTM fitted in {time.time() - t0:.0f}s "
          f"({rec.get('n_biterms', '?')} biterms)", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="github,agnews")
    ap.add_argument("--seeds", default="53,71,89,101,113,127,139")
    ap.add_argument("--k", type=int, default=30)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    for corpus in args.corpora.split(","):
        train, test = O.load_corpus(corpus)
        vocab = test["vocab"]
        word2id = test["dictionary"].token2id
        doc_counts = (train["reason_bow"] + train["item_bow"]).astype(float)
        fair_rows, occ_rows = [], []
        for seed in seeds:
            rec = load_or_fit(corpus, train, args.k, seed)
            fair_rows.extend(evaluate_both(
                rec["beta"], rec["theta"], vocab, doc_counts, train, test,
                "BTM", seed, "fair_readout"))
            visible, hidden = masks_for(test, seed)
            adapter = O.BTMAdapter(train, args.k, seed, rec=rec)
            adapter.dictionary = train["dictionary"]
            adapter.set_word2id(word2id)
            rows = O.run_model(adapter, visible, test["item_tokens"], hidden,
                               RATES, word2id, "BTM")
            for row in rows:
                row["corpus"] = corpus
                row["seed"] = seed
            occ_rows.extend(rows)
        merge_pinned(
            O.OUT_DIR / "rerun" / f"fair_readout_{corpus}_newseeds.csv",
            fair_rows, fair_readout_identity(corpus, train, test, seeds,
                                             args.k),
            ["model", "seed", "readout"])
        merge_pinned(
            O.OUT_DIR / "rerun" / f"occlusion_robustness_{corpus}_newseeds.csv",
            occ_rows, occlusion_identity(corpus, train, test, seeds, args.k),
            ["model", "seed", "rate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())