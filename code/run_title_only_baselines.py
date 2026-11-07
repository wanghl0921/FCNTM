# -*- coding: utf-8 -*-
"""Title-only-trained classical baselines vs. FC-NTM with masked abstract.

Adds the class TF-IDF read-out on the same title-only reference text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"C:\Users\20102\Desktop\FCNTM\FCNTM完整代码")
sys.path.insert(0, str(ROOT))
import public_data_civil as PC
import baselines as B
from run_fcntm_train import evaluate, top_words, class_tfidf
from bn_readout_sensitivity import load_state
import models as NN

SEEDS = [11, 23, 37]
K = 30
OUT = ROOT / "outputs" / "rerun" / "civil_title_only_baselines.csv"


def as_title_only(corpus: dict, dictionary) -> dict:
    d = dict(corpus)
    n = len(corpus["item_tokens"])
    d["reason_tokens"] = [[] for _ in range(n)]
    if "reason_bow" in corpus:
        d["reason_bow"] = np.zeros_like(corpus["reason_bow"])
    d["doc_tokens"] = [list(t) for t in corpus["item_tokens"]]
    d["doc_texts"] = [" ".join(t) for t in d["doc_tokens"]]
    d["corpus"] = [dictionary.doc2bow(t) for t in d["doc_tokens"]]
    if "doc_bow" in corpus and corpus.get("doc_bow") is not None:
        d["doc_bow"] = corpus["item_bow"].copy()
    return d


def dense_counts(corpus: dict, n_vocab: int) -> np.ndarray:
    X = np.zeros((len(corpus["corpus"]), n_vocab), dtype=np.float32)
    for i, bow in enumerate(corpus["corpus"]):
        for j, v in bow:
            X[i, j] = v
    return X


def eval_topics(topics, matrix, title_test):
    return evaluate(topics, matrix, title_test)


def add(rows, model, seed, readout, sc):
    rows.append({"model": model, "seed": seed, "readout": readout,
                 "CV": sc["CV"], "NPMI": sc["NPMI"], "UMass": sc["UMass"],
                 "EXCL_KADJ": sc["EXCL_KADJ"], "IRBO": sc["IRBO"]})


def main() -> None:
    data = PC.load_civil(6000, 2000, need_train_bow=True, need_doc_bow=False)
    train, test = data["train"], data["test"]
    dictionary = train["dictionary"]
    vocab = test["vocab"]
    title_train = as_title_only(train, dictionary)
    title_test = as_title_only(test, dictionary)
    X = dense_counts(title_train, len(vocab))
    X_test = dense_counts(title_test, len(vocab))
    rows = []
    for seed in SEEDS:
        print(f"seed {seed}: LDA", flush=True)
        lda = B.run_lda(title_train["corpus"], dictionary, vocab, K, seed)
        for readout in ["beta", "ctfidf"]:
            if readout == "beta":
                topics, mat = lda["topics"], lda["beta"]
            else:
                ctf = class_tfidf(lda["theta"].T @ X)
                topics, mat = top_words(ctf, vocab, 10), ctf
            sc = eval_topics(topics, mat, title_test)
            add(rows, "LDA-title-only", seed, readout, sc)
            print(f"  LDA {readout} CV {sc['CV']:.4f} NPMI {sc['NPMI']:.4f}", flush=True)

        print(f"seed {seed}: NMF", flush=True)
        nmf = B.run_nmf(X, vocab, K, seed)
        for readout in ["beta", "ctfidf"]:
            if readout == "beta":
                topics, mat = nmf["topics"], nmf["beta"]
            else:
                ctf = class_tfidf(nmf["theta"].T @ X)
                topics, mat = top_words(ctf, vocab, 10), ctf
            sc = eval_topics(topics, mat, title_test)
            add(rows, "NMF-title-only", seed, readout, sc)
            print(f"  NMF {readout} CV {sc['CV']:.4f} NPMI {sc['NPMI']:.4f}", flush=True)

        print(f"seed {seed}: FC-NTM masked abstract (saved checkpoint)", flush=True)
        model = load_state(ROOT / "outputs" / "civil_missing_state.pt",
                           "civil", seed, K, len(vocab))
        model.eval()
        cause_train = torch.from_numpy(np.zeros((X.shape[0], len(vocab)), dtype=np.float32))
        item_train = torch.from_numpy(X)
        has_train = torch.zeros(item_train.shape[0], dtype=torch.bool)
        with torch.no_grad():
            theta_train, _ = model.encode_doc_theta(
                NN._normalize_rows(cause_train), NN._normalize_rows(item_train), has_train)
        theta_train = theta_train.numpy()
        beta = np.asarray(model.get_beta(), dtype=float)
        for readout in ["beta", "ctfidf"]:
            if readout == "beta":
                topics, mat = top_words(beta, vocab, 10), beta
            else:
                ctf = class_tfidf(theta_train.T @ X)
                topics, mat = top_words(ctf, vocab, 10), ctf
            sc = eval_topics(topics, mat, title_test)
            add(rows, "FC-NTM-masked-abstract", seed, readout, sc)
            print(f"  FC-NTM {readout} CV {sc['CV']:.4f} NPMI {sc['NPMI']:.4f}", flush=True)
        del model

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("\nmeans:")
    print(frame.groupby(["model", "readout"])[["CV", "NPMI", "UMass", "EXCL_KADJ", "IRBO"]].mean().round(4))
    print("saved", OUT)


if __name__ == "__main__":
    main()
