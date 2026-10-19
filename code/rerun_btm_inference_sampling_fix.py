# -*- coding: utf-8 -*-
"""Re-score the BTM occlusion curves after the inference-sampling fix.

The fourth review showed that ``_biterms_from_tokens`` consumed one shared
random stream in document order, so a document that exceeds ``max_pairs``
received different biterms - and a different posterior - depending on which
other documents shared the batch (1772/2000 GitHub and 3273/7600 AG News
records exceed the cap).  Inference now derives a per-document generator from
the document's own text and the experiment seed; training keeps the original
shared-stream sampling so the published Gibbs fits stay reproducible.

Consequences:

  * the fitted counts are unchanged -> the BTM read-outs (beta and class
    TF-IDF topic lists) must come out bit-identical, which this script checks;
  * the occlusion curves change -> they are recomputed and merged;
  * the fits are now stored, so a future change can be re-scored without
    refitting.

    python rerun_btm_inference_sampling_fix.py

Outputs
    outputs/rerun/btm_fit_<corpus>_seed<k>.npz      fitted counts (reusable)
    outputs/occlusion_robustness_<corpus>_seed<k>.csv  BTM curve (recomputed)
    outputs/rerun/btm_inference_sampling.csv        invariance + read-out check
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


def save_fit(path, rec, seed):
    np.savez_compressed(
        path, beta=np.asarray(rec["beta"], dtype=float),
        n_bz=np.asarray(rec["n_bz"], dtype=float),
        n_wz=np.asarray(rec["n_wz"], dtype=float),
        alpha=float(rec["alpha"]), beta_prior=float(rec["beta_prior"]),
        window=int(rec["window"]), max_pairs=int(rec["max_pairs"]),
        seed=int(seed), n_biterms=int(rec["n_biterms"]))


def main() -> int:
    rows = []
    for corpus in ("github", "agnews"):
        train, test = O.load_corpus(corpus)
        vocab = test["vocab"]
        word2id = test["dictionary"].token2id
        doc_counts = (train["reason_bow"] + train["item_bow"]).astype(float)
        for seed in (11, 23, 37):
            t0 = time.time()
            rec = BL.run_btm(train["doc_tokens"], train["dictionary"],
                             train["vocab"], 30, seed)
            print(f"  [{corpus} seed {seed}] BTM fitted in "
                  f"{time.time() - t0:.0f}s", flush=True)
            save_fit(O.OUT_DIR / "rerun" / f"btm_fit_{corpus}_seed{seed}.npz",
                     rec, seed)

            # the read-outs must be unchanged: same counts, same training
            # sampling, so the topic lists have to match what is on disk
            stored = pd.read_csv(O.OUT_DIR / f"rerun_btm_{corpus}_readouts.csv")
            stored = stored[stored["seed"] == seed]
            topics_beta = M.topics_from_matrix(np.asarray(rec["beta"], float),
                                               vocab, TOPN)
            topics_ct = top_words(class_tfidf(
                np.asarray(rec["theta"], float).T @ doc_counts), vocab)
            same = True
            for readout, topics in (("beta", topics_beta),
                                    ("ctfidf", topics_ct)):
                old = stored[stored["readout"] == readout]
                if not old.empty:
                    same = same and (json.loads(old["topics"].iloc[0])
                                     == topics)
            print(f"    read-outs identical to the stored ones: {same}",
                  flush=True)

            # occlusion curve with per-document inference sampling
            visible, hidden = masks_for(test, seed)
            adapter = O.BTMAdapter(train, 30, seed, rec=rec)
            adapter.dictionary = train["dictionary"]
            adapter.set_word2id(word2id)
            rows_out = O.run_model(adapter, visible, test["item_tokens"],
                                   hidden, RATES, word2id, "BTM")
            path = O.OUT_DIR / f"occlusion_robustness_{corpus}_seed{seed}.csv"
            prev = pd.read_csv(path)
            keep = prev[prev["model"] != "BTM"]
            pd.concat([keep, pd.DataFrame(rows_out)],
                      ignore_index=True).to_csv(path, index=False,
                                                encoding="utf-8-sig")
            print(f"    -> BTM curve refreshed in {path.name}", flush=True)
            rows.append({
                "corpus": corpus, "seed": seed,
                "readouts_unchanged": bool(same),
                "nll_100": float([r for r in rows_out if r["rate"] == 1.0][0]
                                 ["hidden_NLL"]),
            })

    diag = pd.DataFrame(rows)
    diag.to_csv(O.OUT_DIR / "rerun" / "btm_inference_sampling.csv",
                index=False, encoding="utf-8-sig")
    print(diag.to_string(index=False))
    print("\n[chain] BTM inference-sampling re-score DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
