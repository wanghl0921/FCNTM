# -*- coding: utf-8 -*-
"""Give BTM its document-topic posterior and re-run the BTM-dependent tables.

BTM's own paper (Yan et al. 2013, §3.3) defines the document mixture
``p(z|d) = sum_b p(b|d) p(z|b)``; the previous implementation never exported it,
which is why the paper had to exclude BTM from the class-TF-IDF read-out and
from the occlusion protocol.  ``baselines.run_btm`` now returns that posterior
(and the fitted Gibbs counts), so this chain

  1. fits BTM once per corpus and seed,
  2. writes its class-TF-IDF + native read-out rows into
     ``fair_readout_<corpus>_3seed.csv``,
  3. adds its partial-occlusion curve (re-inferring the mixture on the masked
     text with the fitted counts) to ``occlusion_robustness_<corpus>_seed<k>.csv``.

The BTM fit is the expensive part (~15 min per corpus/seed); everything else is
re-scoring, so each fit is used for both tables.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import baselines as BL
import occlusion_robustness as O
from config import TOPN

RATES = [0.0, 0.25, 0.5, 0.75, 1.0]


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
        doc_counts = (train["reason_bow"] + train["item_bow"]).astype(float)
        fair_rows = []
        for seed in (11, 23, 37):
            t0 = time.time()
            rec = BL.run_btm(train["doc_tokens"], train["dictionary"],
                             train["vocab"], 30, seed)
            print(f"  [{corpus} seed {seed}] BTM fitted in "
                  f"{time.time() - t0:.0f}s, {rec['n_biterms']} biterms",
                  flush=True)

            # --- fair-read-out rows (native beta + class TF-IDF) -----------
            from fair_readout import evaluate_both
            fair_rows.extend(evaluate_both(
                rec["beta"], rec["theta"], vocab, doc_counts, train, test,
                "BTM", seed, "fair_readout"))

            # --- partial-occlusion curve -----------------------------------
            visible, hidden = masks_for(test, seed)
            adapter = O.BTMAdapter(train, 30, seed, rec=rec)
            adapter.dictionary = train["dictionary"]
            adapter.set_word2id(word2id)
            rows = O.run_model(adapter, visible, test["item_tokens"], hidden,
                               RATES, word2id, "BTM")
            path = O.OUT_DIR / f"occlusion_robustness_{corpus}_seed{seed}.csv"
            prev = pd.read_csv(path)
            keep = prev[prev["model"] != "BTM"]
            pd.concat([keep, pd.DataFrame(rows)], ignore_index=True).to_csv(
                path, index=False, encoding="utf-8-sig")
            print(f"    -> occlusion rows written to {path.name}", flush=True)

        fp = O.OUT_DIR / f"fair_readout_{corpus}_3seed.csv"
        fprev = pd.read_csv(fp)
        keep = fprev[fprev["model"] != "BTM"]
        pd.concat([keep, pd.DataFrame(fair_rows)], ignore_index=True).to_csv(
            fp, index=False, encoding="utf-8-sig")
        print(f"  fair_readout {fp.name}: BTM rows written", flush=True)
    print("\n[chain] BTM document-topic DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
