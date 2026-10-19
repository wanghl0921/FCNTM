# -*- coding: utf-8 -*-
"""Re-score everything affected by the two probability-boundary fixes.

The 2026-09-17 review found that two adapters could return a document-topic row
that is not a distribution:

  * BTM: a document whose visible text yields no biterm kept an all-zero row
    (``theta`` summed to 0, so every word probability was clipped to 1e-12 and
    scored as NLL = 27.63).  Affected 34/2000 GitHub and 576/7599 AG News
    records at 100% masking.
  * NMF: a visible bag with no evidence projects to ``W = 0``, same failure.

Both now fall back to the corpus-level topic prior and are validated with
``baselines.check_simplex``.  BTM's document posterior also switched to the
point-estimate form of Yan et al. (2013) section 3.3 that the paper cites.

This chain refits BTM once per corpus/seed (~25 min each) and reuses that fit
for all three affected products; the NMF curves only need a refit, which takes
seconds.

    python rerun_btm_nmf_boundary_fix.py

Outputs
    outputs/fair_readout_<corpus>_3seed.csv          BTM rows (both read-outs)
    outputs/rerun_btm_<corpus>_readouts.csv          topic lists for clean-test
    outputs/occlusion_robustness_<corpus>_seed<k>.csv  BTM + NMF curves
    outputs/rerun/btm_empty_fallback_counts.csv      fallback diagnostics
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


def empty_visible_counts(visible, field_B):
    """How many documents have no visible token at all, per masking rate."""
    out = {}
    for rate in RATES:
        n = sum(1 for a, b in zip(visible[rate], field_B) if not a and not b)
        out[rate] = int(n)
    return out


def main() -> int:
    diagnostics = []
    for corpus in ("github", "agnews"):
        train, test = O.load_corpus(corpus)
        vocab = test["vocab"]
        word2id = test["dictionary"].token2id
        doc_counts = (train["reason_bow"] + train["item_bow"]).astype(float)
        fair_rows, readout_rows = [], []
        for seed in (11, 23, 37):
            t0 = time.time()
            rec = BL.run_btm(train["doc_tokens"], train["dictionary"],
                             train["vocab"], 30, seed)
            print(f"  [{corpus} seed {seed}] BTM fitted in "
                  f"{time.time() - t0:.0f}s, {rec['n_biterms']} biterms",
                  flush=True)

            # --- fair read-out (beta + class TF-IDF from the fixed posterior)
            from fair_readout import evaluate_both
            fair_rows.extend(evaluate_both(
                rec["beta"], rec["theta"], vocab, doc_counts, train, test,
                "BTM", seed, "fair_readout"))

            # --- topic lists, so ``rescore_clean_test.py`` stays reproducible
            topics_beta = M.topics_from_matrix(np.asarray(rec["beta"], float),
                                               vocab, TOPN)
            topics_ctfidf = top_words(
                class_tfidf(np.asarray(rec["theta"], float).T @ doc_counts),
                vocab)
            for readout, topics in (("beta", topics_beta),
                                    ("ctfidf", topics_ctfidf)):
                ev = M.coherence_scores(topics, test["doc_tokens"],
                                        test["corpus"], test["dictionary"],
                                        topn=TOPN)
                readout_rows.append({
                    "model": "BTM", "seed": seed, "readout": readout,
                    "n_topics": len(topics),
                    "topics": json.dumps(topics, ensure_ascii=False),
                    "CV": ev["CV"], "NPMI": ev["NPMI"], "UMass": ev["UMass"],
                })

            # --- partial-occlusion curve, now with the documented fallback
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
            print(f"    -> BTM occlusion rows written to {path.name}",
                  flush=True)

            # --- NMF curve (zero-input fallback); refit is seconds
            nmf = O.NMFAdapter(train, 30, seed)
            nmf.set_word2id(word2id)
            nrows = O.run_model(nmf, visible, test["item_tokens"], hidden,
                               RATES, word2id, "NMF")
            prev = pd.read_csv(path)
            keep = prev[prev["model"] != "NMF"]
            pd.concat([keep, pd.DataFrame(nrows)], ignore_index=True).to_csv(
                path, index=False, encoding="utf-8-sig")
            print(f"    -> NMF occlusion rows refreshed in {path.name}",
                  flush=True)

            counts = empty_visible_counts(visible, test["item_tokens"])
            diagnostics.append({"corpus": corpus, "seed": seed, **counts})

        fp = O.OUT_DIR / f"fair_readout_{corpus}_3seed.csv"
        fprev = pd.read_csv(fp)
        keep = fprev[fprev["model"] != "BTM"]
        pd.concat([keep, pd.DataFrame(fair_rows)], ignore_index=True).to_csv(
            fp, index=False, encoding="utf-8-sig")
        rp = O.OUT_DIR / f"rerun_btm_{corpus}_readouts.csv"
        pd.DataFrame(readout_rows).to_csv(rp, index=False,
                                          encoding="utf-8-sig")
        print(f"  {fp.name} / {rp.name} updated", flush=True)

    diag = pd.DataFrame(diagnostics)
    diag.to_csv(O.OUT_DIR / "rerun" / "btm_empty_fallback_counts.csv",
                index=False, encoding="utf-8-sig")
    print(diag.to_string(index=False))
    print("\n[chain] BTM/NMF boundary re-scoring DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
