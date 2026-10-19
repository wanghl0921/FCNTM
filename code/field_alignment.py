# -*- coding: utf-8 -*-
"""How much do the two fields of a corpus tell the *same* story?

Model-free, held-out measures of cross-field alignment, computed in the same
TF-IDF / TruncatedSVD space used by the redundancy measure:

  cos(A,B)   mean cosine similarity between a document's background vector and
             its own summary vector (high = the two fields point the same way)
  R@1 / MRR  cross-field retrieval: rank the true partner among all documents
             when the summary is used as the query and the background as the
             database (high = the summary identifies the document, i.e. the
             fields carry the same information)
  nn_overlap mean Jaccard overlap of the two fields' nearest-neighbour sets
             (high = both fields organise documents the same way)

Usage: python field_alignment.py [--corpus github,agnews]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from field_redundancy import svd_views

OUT = Path(__file__).resolve().parent / "outputs"


def _cosine(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum(a * b, axis=1)))


def measure(data, seed: int = 0) -> dict:
    train, test = data["train"], data["test"]
    _, te_sm = svd_views(train["item_tokens"], test["item_tokens"], seed=seed)
    _, te_bg = svd_views(train["reason_tokens"], test["reason_tokens"],
                         seed=seed)
    out = {"n_test": int(te_sm.shape[0])}
    out["cos_sm_bg"] = _cosine(te_sm, te_bg)

    # cross-field retrieval: summary -> background (chunked, so that the
    # (n_query x n_database) similarity matrix never has to be materialised)
    sm = te_sm / np.maximum(np.linalg.norm(te_sm, axis=1, keepdims=True), 1e-12)
    bg = te_bg / np.maximum(np.linalg.norm(te_bg, axis=1, keepdims=True), 1e-12)
    rng = np.random.RandomState(seed)
    n_q = min(1500, sm.shape[0])
    q_idx = np.sort(rng.choice(sm.shape[0], size=n_q, replace=False))
    ranks = np.zeros(n_q, dtype=int)
    nn_sm, nn_bg = [], []
    k = 10
    chunk = 250
    for start in range(0, n_q, chunk):
        sl = slice(start, min(start + chunk, n_q))
        rows = q_idx[sl]
        sims = sm[rows] @ bg.T                      # (<=250, n_docs)
        ranks[sl] = np.array(
            [int(np.where(np.argsort(-sims[i]) == rows[i])[0][0]) + 1
             for i in range(sims.shape[0])])
        nn_sm.append(np.argsort(-sims, axis=1)[:, :k])
        nn_bg.append(np.argsort(-sims.T, axis=1)[:, :k])
        del sims
    nn_sm = np.vstack(nn_sm)
    nn_bg = np.vstack(nn_bg)
    out["R@1"] = float(np.mean(ranks == 1))
    out["R@5"] = float(np.mean(ranks <= 5))
    out["MRR"] = float(np.mean(1.0 / ranks))
    # nearest-neighbour set overlap
    ov = [len(set(nn_sm[i]) & set(nn_bg[i])) / k for i in range(nn_sm.shape[0])]
    out["nn_overlap"] = float(np.mean(ov))
    out["n_query"] = int(n_q)
    return out


def load(name: str):
    if name == "agnews":
        import public_data as PD
        return PD.load_agnews(1500, 1900, need_train_bow=False)
    import public_data_github as PGH
    return PGH.load_github(1500, 500, need_train_bow=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="github,agnews")
    args = ap.parse_args()
    table = {}
    for name in args.corpus.split(","):
        table[name] = measure(load(name))
        print(f"{name:8s} " + "  ".join(f"{k}={v:+.4f}"
                                        for k, v in table[name].items()),
              flush=True)
    (OUT / "field_alignment.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved outputs/field_alignment.json")


if __name__ == "__main__":
    main()
