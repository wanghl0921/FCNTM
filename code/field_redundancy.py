# -*- coding: utf-8 -*-
"""How redundant are the two fields of a corpus?

Two held-out measures are reported.  Both are fitted on the training split
only: the TF-IDF vocabulary/IDF, the SVD basis and the regression / CCA
projections are all estimated on train and then applied unchanged to the test
split (the previous revision re-estimated IDF separately on each split, so "the
same document" had two different representations, and it fitted the projection
slope on the test documents it then scored).

  ridge_R2_sm2bg, ridge_R2_bg2sm
      held-out R^2 of a ridge map summary -> background (and the reverse)
      between 64-dimensional TruncatedSVD views of each field's TF-IDF.
      This is a *predictive* redundancy measure: it answers "can one field be
      reconstructed from the other".

  cca_sq_mean
      mean squared canonical correlation between the two views, with the CCA
      basis fitted on train and the correlations evaluated on the held-out
      split (a squared correlation, so it lives in [0, 1] by construction).
      This is the classical multi-view redundancy score.

Constructive controls (``--controls``) check the measure is well behaved: when
the summary field is replaced by an exact copy of the background field the
score must be ~1, and when the summary field is shuffled across documents it
must be ~0.

Usage:
    python field_redundancy.py                       # both corpora, controls
    python field_redundancy.py --corpus github --n-train 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

OUT = Path(__file__).resolve().parent / "outputs"
N_COMP = 64
N_CCA = 32


def fit_idf(tokens, vocab_index, n_vocab):
    """Document frequency -> IDF weights, fitted on one split."""
    rows, cols = [], []
    for i, ts in enumerate(tokens):
        for w in ts:
            j = vocab_index.get(w)
            if j is not None:
                rows.append(i)
                cols.append(j)
    mat = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(tokens), n_vocab))
    df = np.asarray((mat > 0).sum(axis=0)).ravel()
    return np.log((1 + mat.shape[0]) / (1 + df)) + 1.0


def tfidf(tokens, vocab_index, n_vocab, idf=None):
    """Legacy TF-IDF helper kept for the historical scripts.

    Passing ``idf`` (from :func:`fit_idf` on the training split) is strongly
    preferred: without it the IDF is re-estimated from ``tokens`` themselves,
    which is what made the same document receive a different representation in
    the training and the test split.
    """
    rows, cols, vals = [], [], []
    for i, ts in enumerate(tokens):
        for w in ts:
            j = vocab_index.get(w)
            if j is not None:
                rows.append(i)
                cols.append(j)
                vals.append(1.0)
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(len(tokens), n_vocab))
    if idf is None:
        df = np.asarray((mat > 0).sum(axis=0)).ravel()
        idf = np.log((1 + mat.shape[0]) / (1 + df)) + 1.0
    return mat.multiply(idf).tocsr()


def tfidf_fit_transform(train_tokens, test_tokens, min_df: int = 2):
    """TF-IDF fitted on train, applied unchanged to test.

    The analyzer is the identity, so the "documents" stay pre-tokenised lists
    and the vocabulary/IDF come from the training split alone.
    """
    vec = TfidfVectorizer(analyzer=lambda tokens: tokens, lowercase=False,
                          min_df=min_df, sublinear_tf=False)
    train_mat = vec.fit_transform(train_tokens)
    test_mat = vec.transform(test_tokens)
    return train_mat, test_mat


def svd_views(train_tokens, test_tokens, seed: int = 0, n_comp: int = N_COMP):
    tr, te = tfidf_fit_transform(train_tokens, test_tokens)
    n_comp = int(min(n_comp, min(tr.shape) - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    return svd.fit_transform(tr), svd.transform(te)


def measure(data, seed: int = 0, n_train: int = 0, controls: bool = False):
    train, test = data["train"], data["test"]
    if n_train:
        train = dict(train)
        train["item_tokens"] = train["item_tokens"][:n_train]
        train["reason_tokens"] = train["reason_tokens"][:n_train]

    out = {}
    reps = {}
    for field, key in (("sm", "item_tokens"), ("bg", "reason_tokens")):
        reps[field] = svd_views(train[key], test[key], seed=seed)

    for src, dst, name in (("sm", "bg", "ridge_R2_sm2bg"),
                           ("bg", "sm", "ridge_R2_bg2sm")):
        model = Ridge(alpha=1.0).fit(reps[src][0], reps[dst][0])
        pred = model.predict(reps[src][1])
        out[name] = float(r2_score(reps[dst][1], pred,
                                   multioutput="variance_weighted"))
    out["ridge_R2_mean"] = float(np.mean(
        [out["ridge_R2_sm2bg"], out["ridge_R2_bg2sm"]]))

    n = int(min(N_CCA, min(reps["sm"][0].shape[1], reps["bg"][0].shape[1])))
    cca = CCA(n_components=n, max_iter=500)
    cca.fit(reps["sm"][0], reps["bg"][0])
    # held-out canonical correlations: no test data is used to fit anything
    x_test, y_test = cca.transform(reps["sm"][1], reps["bg"][1])
    sq = []
    for k in range(n):
        if np.std(x_test[:, k]) < 1e-12 or np.std(y_test[:, k]) < 1e-12:
            continue
        sq.append(float(np.corrcoef(x_test[:, k], y_test[:, k])[0, 1] ** 2))
    out["cca_sq_mean"] = float(np.mean(sq)) if sq else float("nan")
    out["cca_sq_n"] = len(sq)

    if controls:
        rng = np.random.RandomState(seed)
        # control 1: the summary field is an exact copy of the background field
        copy_data = {
            "train": dict(train, item_tokens=train["reason_tokens"]),
            "test": dict(test, item_tokens=test["reason_tokens"]),
        }
        copy_res = measure(copy_data, seed=seed)
        out["control_copy_cca"] = copy_res["cca_sq_mean"]
        out["control_copy_ridge"] = copy_res["ridge_R2_mean"]
        # control 2: the summary field is shuffled across documents
        perm_tr = rng.permutation(len(train["item_tokens"]))
        perm_te = rng.permutation(len(test["item_tokens"]))
        shuf_data = {
            "train": dict(train, item_tokens=[train["item_tokens"][i]
                                              for i in perm_tr]),
            "test": dict(test, item_tokens=[test["item_tokens"][i]
                                            for i in perm_te]),
        }
        shuf_res = measure(shuf_data, seed=seed)
        out["control_shuffled_cca"] = shuf_res["cca_sq_mean"]
        out["control_shuffled_ridge"] = shuf_res["ridge_R2_mean"]
    return out


def load_dataset(name: str):
    if name == "agnews":
        import public_data as PD
        return PD.load_agnews(1500, 1900, need_train_bow=False)
    import public_data_github as PGH
    return PGH.load_github(1500, 500, need_train_bow=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="github,agnews")
    ap.add_argument("--n-train", type=int, default=0,
                    help="subsample the training split (0 = all 6000)")
    ap.add_argument("--no-controls", action="store_true")
    args = ap.parse_args()
    table = {}
    for name in args.corpus.split(","):
        data = load_dataset(name)
        table[name] = measure(data, n_train=args.n_train,
                              controls=not args.no_controls)
        print(f"{name:8s} " + "  ".join(
            f"{k}={v:+.4f}" for k, v in table[name].items()), flush=True)
    (OUT / "field_redundancy.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved outputs/field_redundancy.json")


if __name__ == "__main__":
    main()
