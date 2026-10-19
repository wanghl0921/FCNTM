# -*- coding: utf-8 -*-
"""Run FC-NTM / baselines / ablations, evaluate and persist results."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

import baselines as BL
import metrics as M
import models as NN
from config import (
    CAUSE_RECON_WEIGHT,
    FIELD_DROPOUT,
    FIG_DIR,
    K_SWEEP,
    MAIN_K,
    NUM_EPOCHS,
    OUT_DIR,
    SEEDS,
    TOPN,
)
from embeddings import embed_texts, embed_words

torch.set_num_threads(8)
try:                      # already set by another module in the same process
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _prepare(corpus: Dict[str, object]):
    cause_bow = torch.from_numpy(corpus["reason_bow"]).float()
    item_bow = torch.from_numpy(corpus["item_bow"]).float()
    doc_bow = torch.from_numpy(corpus["doc_bow"]).float()
    n_docs = len(corpus["doc_tokens"])
    has_cause = torch.tensor(
        [len(t) > 0 for t in corpus["reason_tokens"]], dtype=torch.bool
    )
    paired_idx = torch.nonzero(has_cause, as_tuple=True)[0]
    return {
        "cause_bow": cause_bow,
        "item_bow": item_bow,
        "doc_bow": doc_bow,
        "n_docs": n_docs,
        "has_cause": has_cause,
        "paired_idx": paired_idx,
    }


def word_embeddings(vocab: List[str], semantic: bool = True) -> np.ndarray:
    if semantic:
        emb = embed_words(vocab)
    else:
        rng = np.random.RandomState(0)
        emb = rng.normal(0.0, 0.1, size=(len(vocab), 768)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=-1, keepdims=True)
    emb = emb / (norms + 1e-8)
    return emb.astype(np.float32)


def _run_prodlda(corpus, K, seed, bow_t):
    NN.set_torch_seed(seed)
    model = NN.ProdLDAModel(corpus["n_vocab"], K)
    n_docs = bow_t.shape[0]
    bow_norm = NN._normalize_rows(bow_t)
    history: List[float] = []

    def loss_fn(idx, kl_weight):
        loss, comp = model.train_step(bow_norm[idx], bow_t[idx], kl_weight)
        return loss, comp

    info = NN.train_topic_vae_valid(
        model, loss_fn, loss_fn, n_docs, seed, val_fraction=0.2
    )
    model.eval()
    with torch.no_grad():
        theta, _, _ = model.encode_theta(bow_norm, sample=False)
    beta = model.get_beta()
    topics = M.topics_from_matrix(beta, corpus["vocab"], TOPN)
    return {
        "model": model,
        "beta": beta,
        "topics": topics,
        # ``theta`` is what the unified c-TF-IDF read-out needs; exposing it
        # lets every neural baseline be scored under the same read-out as
        # FC-NTM instead of only under its native beta matrix
        "theta": theta.numpy(),
        "theta_doc": theta.numpy(),
        "loss_history": info["loss_history"],
        "epochs_run": info["epochs_run"],
        "best_epoch": info["best_epoch"],
        "train_loss": info["train_loss"],
    }


def _run_entm(corpus, K, seed, bow_t, emb, semantic=True):
    NN.set_torch_seed(seed)
    model = NN.ENTMModel(corpus["n_vocab"], K, emb, trainable_embeddings=True)
    n_docs = bow_t.shape[0]
    bow_norm = NN._normalize_rows(bow_t)

    def loss_fn(idx, kl_weight):
        return model.train_step(bow_norm[idx], bow_t[idx], kl_weight)

    info = NN.train_topic_vae_valid(
        model, loss_fn, loss_fn, n_docs, seed, val_fraction=0.2
    )
    model.eval()
    with torch.no_grad():
        theta, _, _ = model.encode_theta(bow_norm, sample=False)
    beta = model.get_beta()
    topics = M.topics_from_matrix(beta, corpus["vocab"], TOPN)
    return {
        "model": model,
        "beta": beta,
        "topics": topics,
        "theta": theta.numpy(),
        "theta_doc": theta.numpy(),
        "loss_history": info["loss_history"],
        "epochs_run": info["epochs_run"],
        "best_epoch": info["best_epoch"],
        "train_loss": info["train_loss"],
    }


def _run_ctmlite(corpus, K, seed, doc_ctx, logger=None):
    NN.set_torch_seed(seed)
    model = NN.CTMLiteModel(corpus["n_vocab"], K, doc_ctx.shape[1])
    n_docs = len(corpus["doc_tokens"])
    bow = torch.from_numpy(corpus["doc_bow"]).float()
    bow_norm = NN._normalize_rows(bow)
    ctx = torch.from_numpy(doc_ctx).float()

    def loss_fn(idx, kl_weight):
        return model.train_step(bow_norm[idx], ctx[idx], bow[idx], kl_weight)

    info = NN.train_topic_vae_valid(
        model, loss_fn, loss_fn, n_docs, seed, val_fraction=0.2, logger=logger
    )
    model.eval()
    with torch.no_grad():
        theta, _, _ = model.encode_theta(bow_norm, ctx, sample=False)
    beta = model.get_beta()
    topics = M.topics_from_matrix(beta, corpus["vocab"], TOPN)
    return {
        "model": model,
        "beta": beta,
        "topics": topics,
        "theta": theta.numpy(),
        "theta_doc": theta.numpy(),
        "loss_history": info["loss_history"],
        "epochs_run": info["epochs_run"],
        "best_epoch": info["best_epoch"],
        "train_loss": info["train_loss"],
    }


def _run_fcntm(
    corpus,
    K,
    seed,
    emb=None,
    consistency_weight=0.5,
    cross_recon=True,
    train_paired_only=False,
    decoder_type="linear",
    cause_recon_weight=None,
    field_dropout=None,
    fusion_mode="gate",
    fused_recon=True,
    fused_recon_norm="balanced",
    fused_recon_weight=0.5,
    cause_dropout=0.30,
    item_dropout=0.40,
    kl_max=0.05,
    lr=5e-4,
    num_epochs=None,
):
    """Train FC-NTM with the configuration reported in the paper.

    The fusion/gate arguments default to the final model (gate fusion whose
    mixture is itself trained by a fused reconstruction term).  They used to be
    left at the constructor defaults, so the FC-NTM rows produced by
    ``benchmark_public.py`` were a *different*, weaker model than the one in
    the main table.

    The training call is also the one used for the main results: word-level
    field masking (0.30 / 0.40), KL ceiling 0.05 with 20-epoch warm-up,
    learning rate 5e-4, 20% held-out validation and model selection on the
    validation loss.  An earlier revision fell back to ``train_topic_vae``
    (training-loss selection, config lr 2e-3, KL 0.02, no word masking), so
    "reproducing FC-NTM with benchmark_public.py" produced a different method.
    """
    NN.set_torch_seed(seed)
    model = NN.FCNTM(
        corpus["n_vocab"],
        K,
        emb,
        consistency_weight=consistency_weight,
        cross_reconstruction=cross_recon,
        trainable_embeddings=True,
        decoder_type=decoder_type,
        fusion_mode=fusion_mode,
        fused_recon=fused_recon,
        fused_recon_norm=fused_recon_norm,
        fused_recon_weight=fused_recon_weight,
        cause_dropout=cause_dropout,
        item_dropout=item_dropout,
        cause_recon_weight=(
            float(cause_recon_weight) if cause_recon_weight is not None else CAUSE_RECON_WEIGHT
        ),
        field_dropout=(
            float(field_dropout) if field_dropout is not None else FIELD_DROPOUT
        ),
    )
    cause_bow = torch.from_numpy(corpus["reason_bow"]).float()
    item_bow = torch.from_numpy(corpus["item_bow"]).float()
    cause_norm = NN._normalize_rows(cause_bow)
    item_norm = NN._normalize_rows(item_bow)
    has_cause = torch.tensor([len(t) > 0 for t in corpus["reason_tokens"]], dtype=torch.bool)
    n_docs = len(corpus["doc_tokens"])
    if train_paired_only:
        paired_idx = torch.nonzero(has_cause, as_tuple=True)[0]
        train_cause, train_item, train_cause_norm, train_item_norm = (
            cause_bow[paired_idx],
            item_bow[paired_idx],
            cause_norm[paired_idx],
            item_norm[paired_idx],
        )
        train_has_cause = torch.ones(len(paired_idx), dtype=torch.bool)
        train_n = len(paired_idx)
    else:
        train_cause, train_item, train_cause_norm, train_item_norm = (
            cause_bow,
            item_bow,
            cause_norm,
            item_norm,
        )
        train_has_cause = has_cause
        train_n = n_docs

    def loss_fn(idx, kl_weight):
        loss, comp = model.train_step(
            train_cause_norm[idx],
            train_item_norm[idx],
            train_cause[idx],
            train_item[idx],
            train_has_cause[idx],
            kl_weight,
        )
        return loss, comp

    info = NN.train_topic_vae_valid(
        model, loss_fn, loss_fn, train_n, seed, val_fraction=0.2,
        kl_max=kl_max, lr=lr,
        num_epochs=int(num_epochs) if num_epochs else NUM_EPOCHS,
    )
    model.eval()
    with torch.no_grad():
        fused, streams = model.encode_doc_theta(cause_norm, item_norm, has_cause)
    beta = model.get_beta()
    topics = M.topics_from_matrix(beta, corpus["vocab"], TOPN)
    return {
        "beta": beta,
        "topics": topics,
        "theta_doc": fused.numpy(),
        "theta_cause": streams["cause"].numpy(),
        "theta_item": streams["item"].numpy(),
        "has_cause": streams["both"],
        "loss_history": info["loss_history"],
        "epochs_run": info["epochs_run"],
        "best_epoch": info["best_epoch"],
        "train_loss": info["train_loss"],
    }


def evaluate_run(record: Dict[str, object], corpus: Dict[str, object]) -> Dict[str, float]:
    topics = record["topics"]
    scores = M.coherence_scores(
        topics,
        corpus["doc_tokens"],
        corpus["corpus"],
        corpus["dictionary"],
        topn=TOPN,
    )
    beta = record["beta"]
    scores["EXCL"] = M.exclusivity(beta, topn=TOPN)
    scores["EXCL_KADJ"] = M.exclusivity_kadj(beta, topn=TOPN)
    scores["IRBO"] = M.irbo_diversity(topics)
    return scores


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def dispatch_run(
    model_name: str,
    corpus: Dict[str, object],
    prep: Dict[str, object],
    K: int,
    seed: int,
    emb: Optional[np.ndarray] = None,
    ctx: Optional[np.ndarray] = None,
    consistency_weight: float = 0.5,
    cross_recon: bool = True,
    train_paired_only: bool = False,
) -> Dict[str, object]:
    if model_name == "LDA":
        return BL.run_lda(
            corpus["corpus"], corpus["dictionary"], corpus["vocab"], K, seed
        )
    if model_name == "BTM":
        return BL.run_btm(
            corpus["doc_tokens"], corpus["dictionary"], corpus["vocab"], K, seed
        )
    if model_name == "NMF":
        return BL.run_nmf(corpus["doc_bow"], corpus["vocab"], K, seed)
    if model_name == "GSDMM":
        return BL.run_gsdmm(
            corpus["doc_tokens"], corpus["dictionary"], corpus["vocab"], K, seed
        )
    if model_name == "ProdLDA":
        return _run_prodlda(corpus, K, seed, prep["doc_bow"])
    if model_name == "ProdLDAItem":
        return _run_prodlda(corpus, K, seed, torch.from_numpy(corpus["item_bow"]).float())
    if model_name in ("ENTM", "ETM"):
        return _run_entm(corpus, K, seed, prep["doc_bow"], emb)
    if model_name == "CTMLite":
        return _run_ctmlite(corpus, K, seed, ctx)
    if model_name.startswith("FC-NTM"):
        return _run_fcntm(
            corpus,
            K,
            seed,
            emb=emb,
            consistency_weight=consistency_weight,
            cross_recon=cross_recon,
            train_paired_only=train_paired_only,
        )
    raise ValueError(model_name)


MODEL_SET = ["LDA", "BTM", "NMF", "ETM", "CTMLite", "FC-NTM"]

ABLATIONS = [
    ("FC-NTM", {}),
    ("FC-NTM−Cons", {"consistency_weight": 0.0}),
    ("FC-NTM−XRec", {"cross_recon": False}),
    ("FC-NTM−Dual", {"__model": "ProdLDA"}),
    ("FC-NTM−ItemOnly", {"__model": "ProdLDAItem"}),
    ("FC-NTM−Paired", {"train_paired_only": True}),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="quick", choices=["quick", "sweep", "main", "ablation", "all"])
    parser.add_argument("--models", default=",".join(MODEL_SET))
    parser.add_argument("--seeds", default="11")
    parser.add_argument("--k", type=int, default=MAIN_K)
    parser.add_argument("--out", default="")
    parser.add_argument("--details", default="")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # the private corpus module pulls in jieba; import it only here so that the
    # public experiments (and the regression tests) do not require it
    import corpus as C

    corpus = C.load_corpus()
    prep = _prepare(corpus)
    models = [m for m in args.models.split(",") if m]
    need_semantic = any(m in ("ENTM", "ETM") for m in models)
    emb_sem = word_embeddings(corpus["vocab"], semantic=True) if need_semantic else None
    emb_rand = word_embeddings(corpus["vocab"], semantic=False) if need_semantic else None
    doc_ctx = None
    if "CTMLite" in models or args.mode in ("all",):
        doc_ctx = embed_texts(
            [
                f"变更原因：{r}；变更事项：{i}".replace("\n", " ")
                for r, i in zip(corpus["reason_texts"], corpus["item_texts"])
            ],
            kind="doc",
        )
    seeds = [int(s) for s in args.seeds.split(",")]
    main_k = args.k
    suffix = args.mode

    rows: List[Dict[str, object]] = []

    def run_and_eval(model, K, seed, emb=None, **kw):
        t0 = time.time()
        rec = dispatch_run(model, corpus, prep, K, seed, emb=emb, ctx=doc_ctx, **kw)
        rec["model"] = model
        rec["K"] = K
        rec["seed"] = seed
        rec["seconds"] = time.time() - t0
        rec["metrics"] = evaluate_run(rec, corpus)
        return rec

    if args.mode in ("quick",):
        for model in models:
            emb = emb_sem if model in ("ETM", "ENTM") else None
            rec = run_and_eval(model, main_k, int(seeds[0]), emb=emb)
            print(model, "topics:", len(rec["topics"]), "metrics:", rec["metrics"])
            for i, t in enumerate(rec["topics"]):
                print(" ", i, " ".join(t))
            rows.append(rec)
    elif args.mode in ("sweep", "all"):
        for model in models:
            for K in K_SWEEP:
                for seed in seeds:
                    emb = emb_sem if model in ("ETM", "ENTM") else None
                    rec = run_and_eval(model, K, seed, emb=emb)
                    rows.append(rec)
    elif args.mode in ("main", "all"):
        for model in models:
            for seed in seeds:
                emb = emb_sem if model in ("ETM", "ENTM") else None
                rec = run_and_eval(model, main_k, seed, emb=emb)
                rows.append(rec)

    if args.mode in ("ablation", "all"):
        for name, kw in ABLATIONS:
            for seed in seeds:
                kw2 = dict(kw)
                model_key = kw2.pop("__model", "FC-NTM")
                rec = run_and_eval(model_key, main_k, seed, emb=emb_sem, **kw2)
                rec["model"] = name
                rows.append(rec)

    frame = pd.DataFrame(
        [
            {
                "model": r["model"],
                "K": r["K"],
                "seed": r["seed"],
                "seconds": r["seconds"],
                "epochs_run": r.get("epochs_run", ""),
                "best_epoch": r.get("best_epoch", ""),
                "train_loss": r.get("train_loss", ""),
                "topics": json.dumps(r["topics"], ensure_ascii=False),
                **{k: v for k, v in r["metrics"].items()},
            }
            for r in rows
        ]
    )
    results_path = OUT_DIR / (args.out or f"run_results_{suffix}.csv")
    details_path = OUT_DIR / (args.details or f"run_details_{suffix}.json")
    frame.to_csv(results_path, index=False, encoding="utf-8-sig")
    json.dump(
        [
            {
                "model": r["model"],
                "K": r["K"],
                "seed": r["seed"],
                "beta": r["beta"].tolist(),
                "theta_doc": r.get("theta_doc", np.zeros((1, r["K"]))).tolist(),
                "theta_cause": (
                    r["theta_cause"].tolist() if r.get("theta_cause") is not None else None
                ),
                "theta_item": (
                    r["theta_item"].tolist() if r.get("theta_item") is not None else None
                ),
                "has_cause": (
                    r["has_cause"].tolist() if r.get("has_cause") is not None else None
                ),
                "loss_history": r.get("loss_history", []),
            }
            for r in rows
        ],
        open(details_path, "w", encoding="utf-8"),
        ensure_ascii=False,
    )
    print("saved", len(rows), "runs ->", results_path)


if __name__ == "__main__":
    main()
