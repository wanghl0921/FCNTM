# -*- coding: utf-8 -*-
"""Is FC-NTM's advantage a model effect or a read-out effect?

FC-NTM-cTFIDF builds its topic words by aggregating the *training* posterior
``theta`` (soft class assignment) and applying the BERTopic-style class TF-IDF
normalisation.  The beta-based baselines (LDA, BTM, NMF, ProdLDA, ETM,
CTM-lite, CombinedTM) were scored on their own topic-word matrices instead, so
the comparison mixed "which model is better" with "which read-out is better".

This script removes that confound: every model that exposes a document-topic
distribution is scored twice, on its native beta matrix and on the *same*
class-TF-IDF read-out used by FC-NTM.  BTM is biterm-based and has no
document-topic matrix in this implementation, so it is reported only under its
native read-out and flagged.

    python fair_readout.py --corpus github --seeds 11 --models ETM,ProdLDA,LDA,NMF

Outputs: outputs/fair_readout_<corpus>.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import baselines as BL
import experiment as EX
import metrics as M
from config import OUT_DIR, TOPN
from run_fcntm_train import class_tfidf, top_words

TRAINABLE = {"ProdLDA", "ETM", "CTMLite"}


def train_baseline(name, corpus, K, seed, word_emb, doc_ctx):
    """Train one baseline and return (beta, theta, seconds-ish info)."""
    prep = EX._prepare(corpus)
    if name == "ProdLDA":
        rec = EX._run_prodlda(corpus, K, seed, prep["doc_bow"])
    elif name == "ETM":
        rec = EX._run_entm(corpus, K, seed, prep["doc_bow"], word_emb)
    elif name == "CTMLite":
        rec = EX._run_ctmlite(corpus, K, seed, doc_ctx)
    elif name == "LDA":
        rec = BL.run_lda(corpus["corpus"], corpus["dictionary"],
                         corpus["vocab"], K, seed)
    elif name == "NMF":
        rec = BL.run_nmf(sp.csr_matrix(corpus["doc_bow"].astype(np.float32)),
                         corpus["vocab"], K, seed)
    elif name == "BTM":
        rec = BL.run_btm(corpus["doc_tokens"], corpus["dictionary"],
                         corpus["vocab"], K, seed)
    else:
        raise ValueError(name)
    return rec


def evaluate_both(beta, theta, vocab, doc_counts, train, test, name, seed,
                  source):
    out = []
    topics_beta = M.topics_from_matrix(np.asarray(beta, float), vocab, TOPN)
    scores = M.coherence_scores(topics_beta, test["doc_tokens"],
                                test["corpus"], test["dictionary"], topn=TOPN)
    scores["EXCL_KADJ"] = M.exclusivity_kadj(np.asarray(beta, float),
                                             topn=TOPN)
    scores["IRBO"] = M.irbo_diversity(topics_beta)
    out.append({"source": source, "model": name, "seed": seed,
                "readout": "beta", "CV": scores["CV"], "NPMI": scores["NPMI"],
                "UMass": scores["UMass"],
                "EXCL_KADJ": scores["EXCL_KADJ"], "IRBO": scores["IRBO"]})
    if theta is not None:
        theta = np.asarray(theta, dtype=float)
        if theta.shape[0] == doc_counts.shape[0]:
            soft = class_tfidf(theta.T @ doc_counts)
            topics_ctfidf = top_words(soft, vocab)
            scores = M.coherence_scores(topics_ctfidf, test["doc_tokens"],
                                        test["corpus"], test["dictionary"],
                                        topn=TOPN)
            scores["EXCL_KADJ"] = M.exclusivity_kadj(soft, topn=TOPN)
            scores["IRBO"] = M.irbo_diversity(topics_ctfidf)
            out.append({"source": source, "model": name, "seed": seed,
                        "readout": "ctfidf", "CV": scores["CV"],
                        "NPMI": scores["NPMI"], "UMass": scores["UMass"],
                        "EXCL_KADJ": scores["EXCL_KADJ"],
                        "IRBO": scores["IRBO"]})
    return out


def run_corpus(corpus: str, seeds, models, K: int, out_name: str) -> None:
    import gc

    import pandas as pd

    if corpus == "agnews":
        import public_data as PD
        data = PD.load_agnews(1500, 1900)
    elif corpus == "civil":
        import public_data_civil as PC
        data = PC.load_civil(6000, 2000)
    else:
        import public_data_github as PGH
        data = PGH.load_github(1500, 500)
    train, test = data["train"], data["test"]
    vocab = test["vocab"]
    doc_counts = (train["reason_bow"] + train["item_bow"]).astype(float)

    rows = []
    import embed_public as EP
    word_emb = None
    doc_ctx = None
    if "ETM" in models:
        emb = EP.embed_words(vocab)
        word_emb = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
    if "CTMLite" in models:
        kind = "civil_doc" if corpus == "civil" else "agnews_doc"
        doc_ctx = EP.embed_texts(train["doc_texts"], kind=kind)
    for seed in seeds:
        for name in models:
            print(f"[{corpus}] training {name} seed={seed}", flush=True)
            rec = train_baseline(name, train, K, seed, word_emb, doc_ctx)
            rows.extend(evaluate_both(
                rec["beta"], rec.get("theta"), vocab, doc_counts,
                train, test, name, seed, "fair_readout"))
            del rec
            gc.collect()
    out_path = OUT_DIR / out_name
    frame = pd.DataFrame(rows)
    frame.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(frame.round(4).to_string(index=False))
    print("saved", out_path)
    del data, train, test
    gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="github")
    ap.add_argument("--seeds", default="11")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--models", default="ETM,ProdLDA,CTMLite,LDA,NMF")
    ap.add_argument("--out", default="",
                    help="output name for a single corpus (ignored when "
                         "--corpus lists several)")
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    seeds = [int(s) for s in args.seeds.split(",")]
    for corpus in args.corpus.split(","):
        run_corpus(corpus, seeds, models, args.k,
                   args.out or f"fair_readout_{corpus}.csv")


if __name__ == "__main__":
    main()
