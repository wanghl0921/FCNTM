# -*- coding: utf-8 -*-
"""Screen FC-NTM design variants on a public small corpus.

Every variant changes exactly one design point of the reference configuration
(``cause_recon_weight=2.0``, ``field_dropout=0.5``) and is trained with the
same budget on the same split, so any difference is attributable to that point.

Reported per (corpus, variant, seed):

  beta       C_V / NPMI / UMass / EXCL_KADJ / IRBO of the decoder read-out
  cTFIDF     the same for the soft class-TF-IDF read-out of the fused theta
  recall     cross-field imputation: predict background words from the title
             alone (cause field masked), recall@10 / precision@10
  fieldview  cause-side and item-side coherence of the same topic matrix

Variants
--------
base    reference configuration
fused   train the fused document mixture as well (``fused_recon``)
gate    fused + learnable scalar gate (now actually receives gradients)
poe     fused + product-of-experts fusion instead of a 0.5 average
autow   reconstruction weights balanced by corpus field lengths
token   per-token reconstruction loss (no length domination)
excl    topic exclusivity regulariser turned on
prior   learned missing-field prior for cause-less records (fused)
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch

import config
import metrics as M
import models as NN
from config import TOPN
from public_small import load_small
from run_fcntm_train import class_tfidf, evaluate, top_words
from run_small import field_ids, ref_vocab, subtopics

# run_fcntm_train / run_small pin the thread counts at import time; keep the
# intra-op pool at the value the 8 GB CPU-only box was tuned for.
torch.set_num_threads(4)

OUT = Path(__file__).resolve().parent / "outputs"

BASE = dict(consistency_weight=0.5, cross_reconstruction=True,
            cause_dropout=0.30, item_dropout=0.40, exclusivity_weight=0.0,
            field_dropout=0.5, cause_recon_weight=2.0)

VARIANTS = {
    "base":  dict(),
    "fused": dict(fused_recon=True),
    "gate":  dict(fused_recon=True, fusion_mode="gate"),
    "poe":   dict(fused_recon=True, fusion_mode="poe"),
    "autow": dict(auto_weights=True),
    "token": dict(recon_norm="token", recon_scale="auto"),
    "excl":  dict(exclusivity_weight=0.05),
    "prior": dict(fused_recon=True, missing_prior="learned"),
    "excl2": dict(exclusivity_weight=2.0),
    "fused_autow": dict(fused_recon=True, auto_weights=True),
    "fused_excl": dict(fused_recon=True, exclusivity_weight=2.0),
    "gate_autow": dict(fused_recon=True, fusion_mode="gate",
                       auto_weights=True),
    "prior_autow": dict(fused_recon=True, missing_prior="learned",
                        auto_weights=True),
}

FIELDS = ["corpus", "variant", "seed", "K", "epochs", "best_epoch", "seconds",
          "readout", "CV", "NPMI", "UMass", "EXCL_KADJ", "IRBO",
          "recall", "precision",
          "cause_CV", "cause_NPMI", "item_CV", "item_NPMI",
          "gate", "missing_bias_norm"]


def auto_field_weights(cause_lens, item_lens):
    """Equalise the two fields' reconstruction contributions.

    Keeps the total reconstruction magnitude of the reference configuration
    (cause weight 2.0, item weight 1.0) so the KL weight keeps its meaning.
    """
    lc = max(float(np.mean(cause_lens)), 1.0)
    li = max(float(np.mean(item_lens)), 1.0)
    bar = 0.5 * (lc + li)
    w_c, w_i = bar / lc, bar / li
    scale = (2.0 * lc + 1.0 * li) / (w_c * lc + w_i * li)
    return (w_c * scale, w_i * scale)


def predict_words(theta, beta, topn):
    idx = np.argsort(-np.asarray(beta, dtype=float), axis=1)[:, :topn]
    scores = np.zeros((theta.shape[0], beta.shape[1]))
    for k in range(beta.shape[0]):
        scores[:, idx[k]] += theta[:, k:k + 1]
    return np.argsort(-scores, axis=1)[:, :topn]


def recall_precision(pred, true_bow, topn):
    recs, precs = [], []
    for i in range(pred.shape[0]):
        true_words = set(np.nonzero(true_bow[i])[0].tolist())
        if not true_words:
            continue
        hit = len(set(pred[i, :topn].tolist()) & true_words)
        recs.append(hit / len(true_words))
        precs.append(hit / topn)
    return (float(np.mean(recs)) if recs else float("nan"),
            float(np.mean(precs)) if precs else float("nan"))


def run_fit(train, K, seed, epochs, lr, overrides):
    cause_bow = torch.from_numpy(train["reason_bow"]).float()
    item_bow = torch.from_numpy(train["item_bow"]).float()
    cause_norm = NN._normalize_rows(cause_bow)
    item_norm = NN._normalize_rows(item_bow)
    has_cause = torch.tensor([len(t) > 0 for t in train["reason_tokens"]],
                             dtype=torch.bool)
    kwargs = dict(BASE)
    kwargs.update({k: v for k, v in overrides.items()
                   if k not in ("auto_weights", "recon_scale")})
    if overrides.get("auto_weights"):
        kwargs["field_weights"] = auto_field_weights(
            [len(t) for t in train["reason_tokens"]],
            [len(t) for t in train["item_tokens"]])
    if overrides.get("recon_scale") == "auto":
        kwargs["recon_scale"] = float(np.mean(
            [len(a) + len(b) for a, b in zip(train["reason_tokens"],
                                             train["item_tokens"])]))
    NN.set_torch_seed(seed)
    model = NN.FCNTM(train["n_vocab"], K, **kwargs)

    def loss_fn(idx, kl_weight):
        return model.train_step(cause_norm[idx], item_norm[idx],
                                cause_bow[idx], item_bow[idx], has_cause[idx],
                                kl_weight)

    info = NN.train_topic_vae_valid(
        model, loss_fn, loss_fn, train["n_docs"], seed, val_fraction=0.2,
        kl_max=0.05, num_epochs=epochs, lr=lr, recover_nan=True)
    model.eval()
    with torch.no_grad():
        fused, streams = model.encode_doc_theta(cause_norm, item_norm, has_cause)
    extras = {}
    if getattr(model, "fusion_mode", "") == "gate":
        extras["gate"] = float(torch.sigmoid(model.fusion_logit).item())
    if getattr(model, "missing_prior", "") == "learned":
        extras["missing_bias_norm"] = float(model.missing_bias.norm().item())
    return model, info, fused.numpy(), streams, extras


def score(train, test, model, fused, info, extras, variant, corpus, seed, K,
          epochs, seconds):
    vocab = train["vocab"]
    doc_counts = (train["reason_bow"] + train["item_bow"]).astype(float)
    beta = np.asarray(model.get_beta(), dtype=float)
    soft = class_tfidf(fused.T @ doc_counts)

    cause_ids = field_ids(train["reason_tokens"], vocab)
    item_ids = field_ids(train["item_tokens"], vocab)
    test_cause = [t for t in test["reason_tokens"] if t]
    test_item = [t for t in test["item_tokens"] if t]
    cause_ref = ref_vocab(test_cause)
    item_ref = ref_vocab(test_item)
    test_cause_bows = [train["dictionary"].doc2bow(t) for t in test_cause]
    test_item_bows = [train["dictionary"].doc2bow(t) for t in test_item]

    # ---- cross-field imputation: title alone must predict background words
    word2id = train["dictionary"].token2id
    n_vocab = train["n_vocab"]
    test_cause_np = np.zeros((test["n_docs"], n_vocab), dtype=np.float32)
    test_item_np = np.zeros_like(test_cause_np)
    for d, toks in enumerate(test["reason_tokens"]):
        for w in toks:
            if w in word2id:
                test_cause_np[d, word2id[w]] += 1.0
    for d, toks in enumerate(test["item_tokens"]):
        for w in toks:
            if w in word2id:
                test_item_np[d, word2id[w]] += 1.0
    test_has_cause = np.array([len(t) > 0 for t in test["reason_tokens"]])
    test_item_t = torch.from_numpy(test_item_np).float()
    item_norm = NN._normalize_rows(test_item_t)
    empty_cause = torch.zeros_like(test_item_t)
    with torch.no_grad():
        item_only, _ = model.encode_doc_theta(
            empty_cause, item_norm, torch.zeros(test["n_docs"], dtype=torch.bool))
    pred = predict_words(item_only.numpy(), beta, TOPN)
    recall, precision = recall_precision(pred[test_has_cause],
                                         test_cause_np[test_has_cause], TOPN)

    rows = []
    for readout, mat in (("beta", beta), ("cTFIDF", soft)):
        ev = evaluate(top_words(mat, vocab), mat, test)
        isc = M.coherence_scores(
            subtopics(mat, item_ids, vocab), test_item, test_item_bows,
            train["dictionary"], topn=TOPN, reference_vocab=item_ref)
        csc = M.coherence_scores(
            subtopics(mat, cause_ids, vocab), test_cause, test_cause_bows,
            train["dictionary"], topn=TOPN, reference_vocab=cause_ref)
        rows.append({
            "corpus": corpus, "variant": variant, "seed": seed, "K": K,
            "epochs": epochs, "best_epoch": info["best_epoch"],
            "seconds": round(seconds, 1), "readout": readout,
            "CV": ev["CV"], "NPMI": ev["NPMI"], "UMass": ev["UMass"],
            "EXCL_KADJ": ev["EXCL_KADJ"], "IRBO": ev["IRBO"],
            "recall": recall, "precision": precision,
            "cause_CV": csc["CV"], "cause_NPMI": csc["NPMI"],
            "item_CV": isc["CV"], "item_NPMI": isc["NPMI"],
            "gate": extras.get("gate", ""),
            "missing_bias_norm": extras.get("missing_bias_norm", ""),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="github")
    ap.add_argument("--seeds", default="11")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else OUT / "variant_public.csv"
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    fh = open(out_path, "a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    data = load_small(args.corpus)
    train, test = data["train"], data["test"]
    print(f"[{args.corpus}] train={train['n_docs']} test={test['n_docs']} "
          f"vocab={train['n_vocab']} K={args.k} seeds={args.seeds} "
          f"epochs={args.epochs}", flush=True)
    print(f"  mean field lengths: cause={np.mean([len(t) for t in train['reason_tokens']]):.1f} "
          f"item={np.mean([len(t) for t in train['item_tokens']]):.1f}", flush=True)
    print(f"  field weights if auto: {auto_field_weights([len(t) for t in train['reason_tokens']], [len(t) for t in train['item_tokens']])}",
          flush=True)

    for seed in [int(s) for s in args.seeds.split(",") if s]:
        for name in [v for v in args.variants.split(",") if v]:
            overrides = VARIANTS[name]
            t0 = time.time()
            model, info, fused, _, extras = run_fit(
                train, args.k, seed, args.epochs, args.lr, overrides)
            secs = time.time() - t0
            rows = score(train, test, model, fused, info, extras, name,
                         args.corpus, seed, args.k, args.epochs, secs)
            for row in rows:
                writer.writerow(row)
            fh.flush()
            b = rows[0]
            c = rows[1]
            print(f"  {name:6s} seed={seed} best_ep={info['best_epoch']:>4} "
                  f"beta CV={b['CV']:.3f} NPMI={b['NPMI']:.3f} EX={b['EXCL_KADJ']:.3f} | "
                  f"cTF CV={c['CV']:.3f} NPMI={c['NPMI']:.3f} | "
                  f"recall={b['recall']:.3f} prec={b['precision']:.3f} "
                  f"causeCV={b['cause_CV']:.3f} ({secs:.0f}s)", flush=True)
    fh.close()
    print("saved", out_path)


if __name__ == "__main__":
    main()
