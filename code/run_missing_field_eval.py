# -*- coding: utf-8 -*-
"""Missing-field-grouped evaluation.

The private change-order corpus has 285/577 (49.4%) documents whose cause field
is absent.  This script asks whether topic quality holds up on that
subpopulation, and whether the missing-field *prior fallback* (FC-NTM default)
is what buys the robustness compared with force-encoding the empty field.

Protocol
--------
Models are trained on the 80% train split only.  Their global topics are then
scored with a reference corpus built from three document groups:

  all           every document in the eval scope
  with_cause    only documents that carry a cause field (paired)
  without_cause only documents whose cause field is missing

Two eval scopes are reported: the 20% held-out split, and all 577 documents.
A model whose topics are genuinely usable when the cause field is missing
should keep its ``without_cause`` coherence close to its ``with_cause`` value.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
import torch

import metrics as M
import models as NN
from baselines import run_btm, run_nmf
from config import TOPN
from embeddings import embed_words

torch.set_num_threads(2)
torch.set_num_interop_threads(1)


def score(mat, vocab, dictionary, tokens, bows):
    topics = M.topics_from_matrix(np.asarray(mat, dtype=float), vocab, TOPN)
    return M.coherence_scores(topics, tokens, bows, dictionary, topn=TOPN)


def train_fcntm_emb(train, vocab, K, seed, emb, force_encode_missing):
    cause_bow = torch.from_numpy(train["reason_bow"]).float()
    item_bow = torch.from_numpy(train["item_bow"]).float()
    cause_norm = NN._normalize_rows(cause_bow)
    item_norm = NN._normalize_rows(item_bow)
    has_cause = torch.tensor(
        [len(t) > 0 for t in train["reason_tokens"]], dtype=torch.bool)
    NN.set_torch_seed(seed)
    model = NN.FCNTM(
        len(vocab), K, word_embeddings=emb, decoder_type="embedding",
        consistency_weight=0.5, cross_reconstruction=True,
        cause_recon_weight=2.0, cause_dropout=0.30, item_dropout=0.40,
        exclusivity_weight=0.0,
        force_encode_missing=force_encode_missing)

    def loss_fn(idx, kl_weight):
        return model.train_step(
            cause_norm[idx], item_norm[idx], cause_bow[idx], item_bow[idx],
            has_cause[idx], kl_weight)

    NN.train_topic_vae_valid(
        model, loss_fn, loss_fn, cause_bow.shape[0], seed,
        val_fraction=0.2, kl_max=0.05)
    model.eval()
    return model.get_beta()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="../_priv_cache.pkl")
    parser.add_argument("--seeds", default="11,23,37,53,71")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--out", default="outputs/missing_field_eval.csv")
    args = parser.parse_args()

    with open(args.cache, "rb") as fh:
        data = pickle.load(fh)
    vocab = data["vocab"]
    dictionary = data["dictionary"]
    emb = embed_words(vocab)
    has_cause_all = np.asarray(
        [len(t) > 0 for t in data["reason_tokens"]], dtype=bool)
    doc_tokens = data["doc_tokens"]
    doc_bows = [dictionary.doc2bow(t) for t in doc_tokens]
    print(f"docs={data['n_docs']} with_cause={int(has_cause_all.sum())} "
          f"without_cause={int((~has_cause_all).sum())}", flush=True)

    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(data["n_docs"])
        n_test = int(round(0.2 * data["n_docs"]))
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        train = {
            "reason_bow": data["reason_bow"][train_idx],
            "item_bow": data["item_bow"][train_idx],
            "reason_tokens": [data["reason_tokens"][i] for i in train_idx],
            "item_tokens": [data["item_tokens"][i] for i in train_idx],
            "doc_tokens": [doc_tokens[i] for i in train_idx],
        }
        print(f"seed {seed}: train={len(train_idx)} test={len(test_idx)}",
              flush=True)

        mats = {}
        for name, force in (("FC-NTM-Emb", False),
                            ("FC-NTM-Emb-ForceEnc", True)):
            try:
                mats[name] = train_fcntm_emb(
                    train, vocab, args.k, seed, emb, force)
            except Exception as exc:  # noqa: BLE001
                print(f"  {name} failed: {exc}", flush=True)
        btm = run_btm(train["doc_tokens"], dictionary, vocab, args.k, seed)
        mats["BTM"] = np.asarray(btm["beta"], dtype=float)
        nmf = run_nmf(
            train["reason_bow"] + train["item_bow"], vocab, args.k, seed)
        mats["NMF"] = np.asarray(nmf["beta"], dtype=float)

        scopes = {
            "heldout20": test_idx,
            "all577": np.arange(data["n_docs"]),
        }
        groups = {
            "all": lambda idx: idx,
            "with_cause": lambda idx: idx[has_cause_all[idx]],
            "without_cause": lambda idx: idx[~has_cause_all[idx]],
        }
        for scope, scope_idx in scopes.items():
            for group, sel_fn in groups.items():
                idx = sel_fn(scope_idx)
                toks = [doc_tokens[i] for i in idx]
                bows = [doc_bows[i] for i in idx]
                for name, mat in mats.items():
                    ev = score(mat, vocab, dictionary, toks, bows)
                    rows.append({
                        "seed": seed, "model": name, "scope": scope,
                        "group": group, "n_docs": int(len(idx)),
                        "CV": ev["CV"], "NPMI": ev["NPMI"],
                        "UMass": ev["UMass"],
                    })
                print(f"  {scope:9s} {group:13s} n={len(idx):3d} "
                      + " ".join(f"{n}={score(m, vocab, dictionary, toks, bows)['CV']:.3f}"
                                 for n, m in mats.items()), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print()
    piv = df.groupby(["scope", "group", "model"])[
        ["CV", "NPMI", "UMass"]].mean().round(4)
    print(piv.to_string())
    print("saved", args.out)


if __name__ == "__main__":
    main()
