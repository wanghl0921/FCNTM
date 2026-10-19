# -*- coding: utf-8 -*-
"""Small-scale experiment runner for the public-only paper.

Iteration harness: 1000 train / 400 test documents, K=20, a couple of seeds and
a few hundred epochs, so a full sweep finishes in minutes instead of a day.
The protocol mirrors the full-scale scripts exactly; only the size differs.

Tasks
-----
main        topic quality of FC-NTM (both read-outs) and the bag-of-words baselines
fieldview   coherence restricted to each field's vocabulary
missing     artificial field loss at a sweep of masking rates
ablation    component ablations
imputation  cross-field word prediction (background words from the title alone)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

import baselines as BL
import config
import metrics as M
import models as NN
from config import TOPN
from public_small import load_small, mask_background
from run_fcntm_train import class_tfidf, evaluate, top_words

# run_fcntm_train already pinned the interop thread count at import time;
# calling it twice raises, so only the intra-op count is adjusted here.
torch.set_num_threads(6)

OUT = Path(__file__).resolve().parent / "outputs" / "small"
OUT.mkdir(parents=True, exist_ok=True)

BASE = dict(consistency_weight=0.5, cross_reconstruction=True, cause_dropout=0.30,
            item_dropout=0.40, exclusivity_weight=0.0, field_dropout=0.5)

# Filled from --fusion; applied to every fit so the whole sweep uses the same
# design.  Task-specific ``overrides`` still win, so the ablations keep working.
EXTRA = {}


def train_fcntm(train, K, seed, epochs, lr, overrides=None):
    overrides = overrides or {}
    cause_bow = torch.from_numpy(train["reason_bow"]).float()
    item_bow = torch.from_numpy(train["item_bow"]).float()
    cause_norm = NN._normalize_rows(cause_bow)
    item_norm = NN._normalize_rows(item_bow)
    has_cause = torch.tensor(
        [len(t) > 0 for t in train["reason_tokens"]], dtype=torch.bool)
    kwargs = dict(BASE)
    kwargs.update(EXTRA)
    kwargs.update(overrides)
    NN.set_torch_seed(seed)
    model = NN.FCNTM(train["n_vocab"], K, **kwargs)

    def loss_fn(idx, kl_weight):
        return model.train_step(
            cause_norm[idx], item_norm[idx], cause_bow[idx], item_bow[idx],
            has_cause[idx], kl_weight)

    info = NN.train_topic_vae_valid(
        model, loss_fn, loss_fn, train["n_docs"], seed, val_fraction=0.2,
        kl_max=0.05, num_epochs=epochs, lr=lr, recover_nan=True)
    model.eval()
    with torch.no_grad():
        theta, _ = model.encode_doc_theta(cause_norm, item_norm, has_cause)
        item_theta, _ = model.encode_doc_theta(
            torch.zeros_like(cause_norm), item_norm,
            torch.zeros_like(has_cause))
    return model, np.asarray(model.get_beta(), dtype=float), theta.numpy(), \
        item_theta.numpy(), info


def run_baseline(name, train, K, seed):
    if name == "LDA":
        return BL.run_lda(train["corpus"], train["dictionary"], train["vocab"], K, seed)
    if name == "BTM":
        return BL.run_btm(train["doc_tokens"], train["dictionary"], train["vocab"], K, seed)
    if name == "NMF":
        return BL.run_nmf(sparse.csr_matrix(train["doc_bow"].astype(np.float32)),
                          train["vocab"], K, seed)
    raise ValueError(name)


def field_ids(tokens, vocab):
    present = set(w for ts in tokens for w in ts)
    return [i for i, w in enumerate(vocab) if w in present]


def subtopics(mat, ids, vocab, topn=TOPN):
    return M.topics_from_matrix(mat[:, ids], [vocab[i] for i in ids], topn=topn)


def ref_vocab(token_lists):
    """Words the reference field actually uses, for field-view coherence."""
    return {w for toks in token_lists for w in toks}


def fit_all(data, seeds, K, epochs, lr):
    """Train every model once per seed and keep the topic-word matrices.

    Reusing one fit for both the quality table and the field-view table halves
    the cost of the sweep.
    """
    train, test = data["train"], data["test"]
    vocab = train["vocab"]
    cause_ids = field_ids(train["reason_tokens"], vocab)
    item_ids = field_ids(train["item_tokens"], vocab)
    test_item = [t for t in test["item_tokens"] if t]
    test_item_bows = [train["dictionary"].doc2bow(t) for t in test_item]
    test_cause = [t for t in test["reason_tokens"] if t]
    test_cause_bows = [train["dictionary"].doc2bow(t) for t in test_cause]
    item_ref = ref_vocab(test_item)
    cause_ref = ref_vocab(test_cause)
    doc_counts = (train["reason_bow"] + train["item_bow"]).astype(float)

    main_rows, field_rows = [], []
    for seed in seeds:
        t0 = time.time()
        model, beta, theta, _, info = train_fcntm(train, K, seed, epochs, lr)
        soft = class_tfidf(theta.T @ doc_counts)
        fits = [("FC-NTM", "beta", beta), ("FC-NTM-cTFIDF", "cTFIDF", soft)]
        for name in ("LDA", "BTM", "NMF"):
            rec = run_baseline(name, train, K, seed)
            fits.append((name, "beta", np.asarray(rec["beta"], dtype=float)))
        print(f"  seed {seed} fitted in {time.time() - t0:.0f}s "
              f"(fcntm best_epoch={info['best_epoch']})", flush=True)
        for name, readout, mat in fits:
            ev = evaluate(top_words(mat, vocab), mat, test)
            main_rows.append({"model": name, "readout": readout, "seed": seed,
                              "best_epoch": info["best_epoch"] if name.startswith("FC-NTM") else "",
                              **ev})
            it = subtopics(mat, item_ids, vocab)
            ct = subtopics(mat, cause_ids, vocab)
            isc = M.coherence_scores(it, test_item, test_item_bows,
                                     train["dictionary"], topn=TOPN,
                                     reference_vocab=item_ref)
            csc = M.coherence_scores(ct, test_cause, test_cause_bows,
                                     train["dictionary"], topn=TOPN,
                                     reference_vocab=cause_ref)
            field_rows.append({"model": name, "readout": readout, "seed": seed,
                               "item_CV": isc["CV"], "item_NPMI": isc["NPMI"],
                               "item_n": isc["CV_n"], "cause_CV": csc["CV"],
                               "cause_NPMI": csc["NPMI"], "cause_n": csc["CV_n"],
                               "n_topics": isc["n_topics"]})
    return main_rows, field_rows


def task_missing(data, seeds, K, epochs, lr, rates, dropouts):
    """Artificial field loss: mask a share of training backgrounds and see what
    survives.  Bag-of-words baselines are given the same masked corpus."""
    train, test = data["train"], data["test"]
    vocab = train["vocab"]
    cause_ids = field_ids(train["reason_tokens"], vocab)
    item_ids = field_ids(train["item_tokens"], vocab)
    test_item = [t for t in test["item_tokens"] if t]
    test_item_bows = [train["dictionary"].doc2bow(t) for t in test_item]
    test_cause = [t for t in test["reason_tokens"] if t]
    test_cause_bows = [train["dictionary"].doc2bow(t) for t in test_cause]
    item_ref = ref_vocab(test_item)
    cause_ref = ref_vocab(test_cause)
    rows = []
    for rate in rates:
        for seed in seeds:
            masked = mask_background(train, rate, seed)
            doc_counts = (masked["reason_bow"] + masked["item_bow"]).astype(float)
            for fdrop in dropouts:
                model, beta, theta, _, info = train_fcntm(
                    masked, K, seed, epochs, lr, {"field_dropout": fdrop})
                soft = class_tfidf(theta.T @ doc_counts)
                for readout, mat in (("beta", beta), ("cTFIDF", soft)):
                    ev = evaluate(top_words(mat, vocab), mat, test)
                    item_ev = M.coherence_scores(
                        subtopics(mat, item_ids, vocab), test_item,
                        test_item_bows, train["dictionary"], topn=TOPN,
                        reference_vocab=item_ref)
                    cause_ev = M.coherence_scores(
                        subtopics(mat, cause_ids, vocab), test_cause,
                        test_cause_bows, train["dictionary"], topn=TOPN,
                        reference_vocab=cause_ref)
                    rows.append({
                        "mask_rate": rate, "field_dropout": fdrop, "seed": seed,
                        "model": "FC-NTM" if readout == "beta" else "FC-NTM-cTFIDF",
                        "masked_docs": masked.get("masked_docs", 0),
                        "CV": ev["CV"], "NPMI": ev["NPMI"], "UMass": ev["UMass"],
                        "item_CV": item_ev["CV"], "item_NPMI": item_ev["NPMI"],
                        "item_n": item_ev["CV_n"], "cause_CV": cause_ev["CV"],
                        "cause_n": cause_ev["CV_n"],
                        "best_epoch": info["best_epoch"],
                    })
                    print(f"  mask={rate} fdrop={fdrop} seed={seed} {readout} "
                          f"CV={ev['CV']:.3f} item_CV={item_ev['CV']:.3f}",
                          flush=True)
            for name in ("LDA", "BTM"):
                rec = run_baseline(name, masked, K, seed)
                mat = np.asarray(rec["beta"], dtype=float)
                ev = evaluate(rec["topics"], mat, test)
                item_ev = M.coherence_scores(
                    subtopics(mat, item_ids, vocab), test_item, test_item_bows,
                    train["dictionary"], topn=TOPN)
                rows.append({
                    "mask_rate": rate, "field_dropout": "", "seed": seed,
                    "model": name, "masked_docs": masked.get("masked_docs", 0),
                    "CV": ev["CV"], "NPMI": ev["NPMI"], "UMass": ev["UMass"],
                    "item_CV": item_ev["CV"], "item_NPMI": item_ev["NPMI"],
                    "item_n": item_ev["CV_n"], "cause_CV": float("nan"),
                    "cause_n": float("nan"), "best_epoch": "",
                })
                print(f"  mask={rate} seed={seed} {name} CV={ev['CV']:.3f} "
                      f"item_CV={item_ev['CV']:.3f}", flush=True)
    return rows


ABLATIONS = {
    "full": {},
    "no-consistency": {"consistency_weight": 0.0},
    "no-cross": {"cross_reconstruction": False},
    "no-mask": {"cause_dropout": 0.0, "item_dropout": 0.0},
    "no-fielddrop": {"field_dropout": 0.0},
}


def task_ablation(data, seeds, K, epochs, lr, configs):
    train, test = data["train"], data["test"]
    vocab = train["vocab"]
    doc_counts = (train["reason_bow"] + train["item_bow"]).astype(float)
    rows = []
    for config_name in configs:
        for seed in seeds:
            model, beta, theta, _, info = train_fcntm(
                train, K, seed, epochs, lr, ABLATIONS[config_name])
            soft = class_tfidf(theta.T @ doc_counts)
            for readout, mat in (("beta", beta), ("cTFIDF", soft)):
                ev = evaluate(top_words(mat, vocab), mat, test)
                rows.append({"config": config_name, "seed": seed, "readout": readout,
                             "CV": ev["CV"], "NPMI": ev["NPMI"], "UMass": ev["UMass"],
                             "IRBO": ev["IRBO"], "best_epoch": info["best_epoch"]})
                print(f"  {config_name} seed={seed} {readout} CV={ev['CV']:.3f} "
                      f"NPMI={ev['NPMI']:.3f}", flush=True)
    return rows


def _predict_words(theta, beta, topn):
    idx = np.argsort(-np.asarray(beta, float), axis=1)[:, :topn]
    scores = np.zeros((theta.shape[0], beta.shape[1]))
    for k in range(beta.shape[0]):
        scores[:, idx[k]] += theta[:, k:k + 1]
    return np.argsort(-scores, axis=1)[:, :topn]


def _recall_precision(pred, true_bow, topn):
    recs, precs = [], []
    for i in range(pred.shape[0]):
        true_words = set(np.nonzero(true_bow[i])[0].tolist())
        if not true_words:
            continue
        hit = len(set(pred[i, :topn].tolist()) & true_words)
        recs.append(hit / len(true_words))
        precs.append(hit / topn)
    return float(np.mean(recs)), float(np.mean(precs))


def task_imputation(data, seeds, K, epochs, lr):
    """Predict the background field's words from the title alone."""
    train, test = data["train"], data["test"]
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
    test_item = torch.from_numpy(test_item_np).float()
    test_item_norm = NN._normalize_rows(test_item)
    empty_cause = torch.zeros_like(test_item)

    rows = []
    for label, override in (("recon2", {}), ("noxrec", {"cross_reconstruction": False})):
        for seed in seeds:
            train_bow_c = torch.from_numpy(train["reason_bow"]).float()
            item_bow_t = torch.from_numpy(train["item_bow"]).float()
            cause_norm = NN._normalize_rows(train_bow_c)
            item_norm = NN._normalize_rows(item_bow_t)
            has_cause = torch.tensor(
                [len(t) > 0 for t in train["reason_tokens"]], dtype=torch.bool)
            kwargs = dict(BASE)
            kwargs.update(override)
            NN.set_torch_seed(seed)
            model = NN.FCNTM(train["n_vocab"], K, **kwargs)

            def loss_fn(idx, kl_weight, _m=model, _c=cause_norm, _i=item_norm,
                        _cb=train_bow_c, _ib=item_bow_t, _hc=has_cause):
                return _m.train_step(_c[idx], _i[idx], _cb[idx], _ib[idx],
                                     _hc[idx], kl_weight)

            NN.train_topic_vae_valid(
                model, loss_fn, loss_fn, train["n_docs"], seed, val_fraction=0.2,
                kl_max=0.05, num_epochs=epochs, lr=lr, recover_nan=True)
            model.eval()
            with torch.no_grad():
                item_theta, _ = model.encode_doc_theta(
                    empty_cause, test_item_norm,
                    torch.zeros(test["n_docs"], dtype=torch.bool))
            beta = np.asarray(model.get_beta(), dtype=float)
            pred = _predict_words(item_theta.numpy(), beta, TOPN)
            rec, prec = _recall_precision(pred[test_has_cause],
                                          test_cause_np[test_has_cause], TOPN)
            rows.append({"config": label, "seed": seed, "readout": "item_only",
                         "recall": rec, "precision": prec})
            print(f"  imputation {label} seed={seed} recall={rec:.4f} "
                  f"precision={prec:.4f}", flush=True)
    return rows


TAG = ""



def per_doc_predictability(data, n_comp=64, seed=0):
    """Per-document version of the cross-view redundancy score.

    A corpus-level ridge map summary -> background is fitted on the train
    split; every held-out document then gets its own R^2 against that map.
    Documents whose background is already determined by the summary score
    near 1 (redundant fields); documents the map cannot explain score low
    (the two fields carry different information).
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import Ridge
    from field_redundancy import fit_idf, tfidf

    train, test = data["train"], data["test"]
    vocab_index = train["dictionary"].token2id
    n_vocab = train["n_vocab"]
    # one IDF, fitted on the training split, reused for the held-out split
    idf_sm = fit_idf(train["item_tokens"], vocab_index, n_vocab)
    idf_bg = fit_idf(train["reason_tokens"], vocab_index, n_vocab)
    xtr_sm = tfidf(train["item_tokens"], vocab_index, n_vocab, idf_sm)
    xtr_bg = tfidf(train["reason_tokens"], vocab_index, n_vocab, idf_bg)
    xte_sm = tfidf(test["item_tokens"], vocab_index, n_vocab, idf_sm)
    xte_bg = tfidf(test["reason_tokens"], vocab_index, n_vocab, idf_bg)
    svd_sm = TruncatedSVD(n_components=n_comp, random_state=seed).fit(xtr_sm)
    svd_bg = TruncatedSVD(n_components=n_comp, random_state=seed).fit(xtr_bg)
    ztr_sm, ztr_bg = svd_sm.transform(xtr_sm), svd_bg.transform(xtr_bg)
    zte_sm, zte_bg = svd_sm.transform(xte_sm), svd_bg.transform(xte_bg)
    ridge = Ridge(alpha=1.0).fit(ztr_sm, ztr_bg)
    pred = ridge.predict(zte_sm)
    mean = ztr_bg.mean(axis=0)
    err = ((zte_bg - pred) ** 2).sum(axis=1)
    base = ((zte_bg - mean) ** 2).sum(axis=1)
    return 1.0 - err / np.maximum(base, 1e-9)


def task_strata(data, seeds, K, epochs, lr, split_name="test"):
    """Cross-field imputation measured separately on redundant / complementary docs."""
    train, test = data["train"], data["test"]
    word2id = train["dictionary"].token2id
    n_vocab = train["n_vocab"]
    score = per_doc_predictability(data)
    median = float(np.median(score))
    low = score <= median          # summary does NOT determine the background
    high = ~low                    # redundant pair

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
    has_cause = np.array([len(t) > 0 for t in test["reason_tokens"]])
    strata = {
        "low_predictability(互补)": low & has_cause,
        "high_predictability(冗余)": high & has_cause,
    }
    test_item = torch.from_numpy(test_item_np).float()
    test_item_norm = NN._normalize_rows(test_item)
    empty_cause = torch.zeros_like(test_item)

    rows = []
    for label, override in (("recon2", {}), ("noxrec", {"cross_reconstruction": False})):
        for seed in seeds:
            cause_bow = torch.from_numpy(train["reason_bow"]).float()
            item_bow = torch.from_numpy(train["item_bow"]).float()
            cause_norm = NN._normalize_rows(cause_bow)
            item_norm = NN._normalize_rows(item_bow)
            has_c = torch.tensor(
                [len(t) > 0 for t in train["reason_tokens"]], dtype=torch.bool)
            kwargs = dict(BASE)
            kwargs.update(override)
            NN.set_torch_seed(seed)
            model = NN.FCNTM(train["n_vocab"], K, **kwargs)

            def loss_fn(idx, kl_weight, _m=model, _c=cause_norm, _i=item_norm,
                        _cb=cause_bow, _ib=item_bow, _hc=has_c):
                return _m.train_step(_c[idx], _i[idx], _cb[idx], _ib[idx],
                                     _hc[idx], kl_weight)

            NN.train_topic_vae_valid(
                model, loss_fn, loss_fn, train["n_docs"], seed, val_fraction=0.2,
                kl_max=0.05, num_epochs=epochs, lr=lr, recover_nan=True)
            model.eval()
            with torch.no_grad():
                item_theta, _ = model.encode_doc_theta(
                    empty_cause, test_item_norm,
                    torch.zeros(test["n_docs"], dtype=torch.bool))
            beta = np.asarray(model.get_beta(), dtype=float)
            pred = _predict_words(item_theta.numpy(), beta, TOPN)
            for name, mask in strata.items():
                if mask.sum() == 0:
                    continue
                rec, prec = _recall_precision(pred[mask], test_cause_np[mask], TOPN)
                rows.append({"config": label, "seed": seed, "stratum": name,
                             "n_docs": int(mask.sum()), "recall": rec,
                             "precision": prec})
                print(f"  strata {label} seed={seed} {name} n={int(mask.sum())} "
                      f"recall={rec:.4f} precision={prec:.4f}", flush=True)
    return rows


def write_csv(path, rows):
    if TAG:
        path = path.with_name(f"{path.stem}_{TAG}{path.suffix}")
    if not rows:
        print("no rows for", path)
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("saved", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["github", "agnews"], default="github")
    ap.add_argument("--tasks", default="main,missing")
    ap.add_argument("--seeds", default="11,23")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--rates", default="0,0.25,0.5,0.75")
    ap.add_argument("--dropouts", default="0.5,0.0")
    ap.add_argument("--configs", default="full,no-consistency,no-cross,no-mask,no-fielddrop")
    ap.add_argument("--tag", default="",
                    help="suffix for the output CSVs, so a re-run at a different "
                         "epoch budget does not overwrite the first pass")
    ap.add_argument("--fusion", choices=["none", "fused", "gate", "poe"],
                    default="none",
                    help="train the document-level fused mixture instead of "
                         "leaving it an untrained 0.5 average of the streams")
    args = ap.parse_args()

    global TAG, EXTRA
    TAG = args.tag
    if args.fusion in ("fused", "gate", "poe"):
        EXTRA["fused_recon"] = True
    if args.fusion in ("gate", "poe"):
        EXTRA["fusion_mode"] = args.fusion

    data = load_small(args.corpus)
    train = data["train"]
    print(f"[{args.corpus}] train={train['n_docs']} test={data['test']['n_docs']} "
          f"vocab={train['n_vocab']} K={args.k} seeds={args.seeds}", flush=True)

    tasks = [t for t in args.tasks.split(",") if t]
    seeds = [int(s) for s in args.seeds.split(",")]

    if "main" in tasks or "fieldview" in tasks:
        main_rows, field_rows = fit_all(data, seeds, args.k, args.epochs, args.lr)
        if "main" in tasks:
            write_csv(OUT / f"main_{args.corpus}.csv", main_rows)
        if "fieldview" in tasks:
            write_csv(OUT / f"fieldview_{args.corpus}.csv", field_rows)
    if "missing" in tasks:
        rates = [float(r) for r in args.rates.split(",")]
        dropouts = [float(d) for d in args.dropouts.split(",")]
        rows = task_missing(data, seeds, args.k, args.epochs, args.lr, rates, dropouts)
        write_csv(OUT / f"missing_{args.corpus}.csv", rows)
    if "ablation" in tasks:
        configs = [c for c in args.configs.split(",") if c]
        rows = task_ablation(data, seeds, args.k, args.epochs, args.lr, configs)
        write_csv(OUT / f"ablation_{args.corpus}.csv", rows)
    if "strata" in tasks:
        rows = task_strata(data, seeds, args.k, args.epochs, args.lr)
        write_csv(OUT / f"strata_{args.corpus}.csv", rows)
    if "imputation" in tasks:
        rows = task_imputation(data, seeds, args.k, args.epochs, args.lr)
        write_csv(OUT / f"imputation_{args.corpus}.csv", rows)


if __name__ == "__main__":
    main()
