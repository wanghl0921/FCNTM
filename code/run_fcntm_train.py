# -*- coding: utf-8 -*-
"""Train FC-NTM (masked protocol) and evaluate both topic-word read-outs.

Protocol used for every result in the paper:
  - field-level word masking on encoder inputs: cause 0.30 / item 0.40,
    reconstruction targets always use the full word counts;
  - KL ceiling 0.05 with 20-epoch linear warm-up;
  - 20% validation split, early stopping on validation loss;
  - K = 30, seeds 11/23/37/53/71 (private) or 11/23/37 (GitHub issues).

For every trained model two topic-word read-outs are computed:
  * FC-NTM        : top words of the shared decoder beta matrix;
  * FC-NTM-cTFIDF : soft class TF-IDF over the training posterior theta
                    (class counts S_k = sum_d theta_dk * x_d, then the
                    BERTopic-style log-normalization with average class
                    length A and per-word class frequency df_v).

Metrics are computed with the same corpus and TOP-10 setting as the paper.
One row per (corpus, seed, read-out) is written to the results CSV; the file
is overwritten by default and can be appended to with ``--append``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import time

import numpy as np
import torch

import config
import metrics as M
import result_identity as RI

torch.set_num_threads(8)
try:                      # already set by another module in the same process
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def class_tfidf(counts: np.ndarray) -> np.ndarray:
    """BERTopic ClassTfidfTransformer applied to class-word count rows."""
    """BERTopic ClassTfidfTransformer applied to class-word count rows."""
    counts = np.asarray(counts, dtype=float)
    row_sum = counts.sum(axis=1)
    tf = np.zeros_like(counts)
    keep = row_sum > 1e-12
    tf[keep] = counts[keep] / row_sum[keep, None]
    # BERTopic: the denominator is the word frequency summed over all classes
    # (``counts.sum(axis=0)``), not the number of classes containing the word.
    df = counts.sum(axis=0)
    avg = float(row_sum[keep].mean()) if keep.any() else 1.0
    idf = np.log(avg / np.maximum(df, 1e-12) + 1.0)
    return tf * idf


def top_words(scores: np.ndarray, vocab, topn: int = config.TOPN):
    return [[vocab[int(i)] for i in np.argsort(-row)[:topn]] for row in scores]


def config_hash(kwargs: dict) -> str:
    """Stable fingerprint of a training configuration (for reuse checks)."""
    payload = json.dumps(kwargs, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def evaluate(topics, score_matrix, corpus):
    scores = M.coherence_scores(topics, corpus["doc_tokens"],
                                corpus["corpus"], corpus["dictionary"],
                                topn=config.TOPN)
    scores["EXCL"] = M.exclusivity(score_matrix, topn=config.TOPN)
    scores["EXCL_KADJ"] = M.exclusivity_kadj(score_matrix, topn=config.TOPN)
    scores["IRBO"] = M.irbo_diversity(topics)
    return scores


def run_corpus(corpus_train, corpus_test, seeds, K, out_path, append=False,
               field_dropout=0.0, recover_nan=False, max_epochs=None,
               lr=None, details_path=None, fusion="none",
               fusion_norm="joint", fusion_weight=0.5,
               resume_every=50, allow_resume=True, state_path=None):
    import models as NN

    cause_bow = torch.from_numpy(corpus_train["reason_bow"]).float()
    item_bow = torch.from_numpy(corpus_train["item_bow"]).float()
    has_cause = torch.tensor(
        [len(t) > 0 for t in corpus_train["reason_tokens"]], dtype=torch.bool)
    # the encoder inputs are normalised per mini-batch (see loss_fn below)
    # instead of once for the whole corpus: two (docs x V) buffers less, which
    # is what lets this run inside the memory budget of a CPU-only box
    vocab = corpus_test["vocab"]
    fieldnames = ["corpus", "model", "seed", "K_used", "best_epoch",
                  "seconds", "topics",
                  "CV", "CV_n", "NPMI", "NPMI_n", "UCI", "UCI_n",
                  "UMass", "UMass_n", "EXCL", "EXCL_KADJ", "IRBO",
                  "state_hash", "code_hash"]
    detail_records = []
    # the identity of *this* configuration: a stored CSV may only be reused
    # (seeds skipped, rows appended) when it was produced by the same script
    # version, corpus text, seed set and hyper-parameters
    ident = RI.identity({
        "script": "run_fcntm_train.run_corpus",
        "n_vocab": int(corpus_train["n_vocab"]),
        "n_docs": int(corpus_train["reason_bow"].shape[0]),
        "k": int(K),
        "seeds": [int(s) for s in seeds],
        "field_dropout": float(field_dropout),
        "fusion": fusion,
        "fusion_norm": fusion_norm,
        "fusion_weight": float(fusion_weight),
        "recover_nan": bool(recover_nan),
        "max_epochs": int(max_epochs or 0),
        "lr": float(lr or 0.0),
        "code": RI.code_version(),
        # field-aware digest: hashing only the concatenated document text let a
        # field swap keep the same identity (the review's counter-example), so
        # the two fields and the test split are hashed separately
        "data": RI.text_digest([corpus_train.get("doc_tokens"),
                                corpus_test.get("doc_tokens")]),
        "fields": RI.text_digest([corpus_train.get("reason_tokens"),
                                  corpus_train.get("item_tokens"),
                                  corpus_test.get("reason_tokens"),
                                  corpus_test.get("item_tokens")]),
    })
    # a resumable run restarts the process from scratch, so the CSV must be
    # appended to (never truncated) and already-scored seeds must be skipped;
    # otherwise every restart wipes the previous seeds' rows
    scored = set()
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    key = (row["model"], int(row["seed"]))
                except (KeyError, TypeError, ValueError):
                    continue
                # a row only counts as "already scored" when its metrics are
                # actually usable: a NaN row written by an interrupted run must
                # not make the seed look finished
                try:
                    if not (np.isfinite(float(row["CV"]))
                            and np.isfinite(float(row["NPMI"]))):
                        continue
                except (KeyError, TypeError, ValueError):
                    continue
                scored.add(key)
    resume_ok = bool(allow_resume) and RI.matches(out_path, ident)
    if scored and allow_resume and not resume_ok:
        print(f"    stored rows in {out_path} are not pinned to this "
              f"configuration ({RI.why_not(out_path, ident)}); re-scoring "
              f"every seed", flush=True)
    if scored and allow_resume and not resume_ok:
        # never destroy the old numbers: move them aside first
        stash = RI.stash_unpinned(out_path)
        if stash:
            print(f"    old rows moved to {stash}", flush=True)
        scored = set()
    keep_existing = resume_ok and (append or bool(scored))
    write_header = not keep_existing or not os.path.exists(out_path) \
        or os.path.getsize(out_path) == 0
    with open(out_path, "a" if keep_existing else "w",
              encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for seed in seeds:
            if resume_ok and ("FC-NTM", int(seed)) in scored and \
                    ("FC-NTM-cTFIDF", int(seed)) in scored:
                print(f"    skip seed {seed}: already scored "
                      f"(identity {ident['id']})", flush=True)
                continue
            NN.set_torch_seed(seed)
            # ``fusion`` trains the document-level mixture (the one the cTFIDF
            # read-out and every document-level score use) instead of leaving
            # it an untrained average of the two stream posteriors.
            fusion_kwargs = {}
            if fusion in ("fused", "gate", "poe"):
                fusion_kwargs["fused_recon"] = True
                fusion_kwargs["fused_recon_norm"] = fusion_norm
                fusion_kwargs["fused_recon_weight"] = fusion_weight
            if fusion in ("gate", "poe"):
                fusion_kwargs["fusion_mode"] = fusion
            model_kwargs = dict(
                consistency_weight=0.5,
                cross_reconstruction=True,
                cause_dropout=0.30,
                item_dropout=0.40,
                exclusivity_weight=0.0,
                field_dropout=field_dropout,
                **fusion_kwargs,
            )
            model = NN.FCNTM(corpus_train["n_vocab"], K, **model_kwargs)

            def loss_fn(idx, kl_weight):
                cause_batch = cause_bow[idx]
                item_batch = item_bow[idx]
                return model.train_step(
                    NN._normalize_rows(cause_batch),
                    NN._normalize_rows(item_batch),
                    cause_batch, item_batch,
                    has_cause[idx], kl_weight)

            t0 = time.time()
            extra = {"recover_nan": recover_nan}
            if max_epochs:
                extra["num_epochs"] = int(max_epochs)
            if lr:
                extra["lr"] = float(lr)
            # resumable training: long runs on a CPU box can be interrupted, so
            # the full state (weights + optimizer + epoch + RNG) is written
            # every ``resume_every`` epochs next to the details dump
            resume_file = None
            if details_path:
                resume_file = f"{details_path}.resume.seed{int(seed)}"
                # tie the resume state to the same experiment as the result
                # CSV: resuming by path alone let a state trained under another
                # objective, split or learning rate be reused silently
                # (2026-10-07 review, item 1)
                resume_identity = RI.identity({
                    "script": "run_fcntm_train.run_corpus.resume",
                    "seed": int(seed),
                    "n_vocab": int(corpus_train["n_vocab"]),
                    "n_docs": int(corpus_train["n_docs"]),
                    "k": int(K),
                    "field_dropout": float(field_dropout),
                    "fusion": fusion,
                    "fusion_norm": fusion_norm,
                    "fusion_weight": float(fusion_weight),
                    "recover_nan": bool(recover_nan),
                    "lr": float(lr or 0.0),
                    "val_fraction": 0.2,
                    "kl_max": 0.05,
                    "code": RI.code_version(),
                    "data": RI.text_digest([corpus_train.get("doc_tokens")]),
                    "fields": RI.text_digest(
                        [corpus_train.get("reason_tokens"),
                         corpus_train.get("item_tokens")]),
                })
                extra["resume_path"] = resume_file
                extra["resume_identity"] = resume_identity
                extra["resume_every"] = int(resume_every)
                extra["allow_resume"] = bool(allow_resume)
                extra["logger"] = (
                    lambda ep, loss, klw: print(
                        f"    [train] epoch {ep} loss {loss:.3f} klw {klw:.4f}",
                        flush=True) if ep % 25 == 0 or ep <= 5 else None)
            info = NN.train_topic_vae_valid(
                model, loss_fn, loss_fn, corpus_train["n_docs"], seed,
                val_fraction=0.2, kl_max=0.05, **extra)
            seconds = time.time() - t0
            model.eval()
            # batched inference: normalisation happens per batch, so no extra
            # (docs x V) buffers are needed at scoring time either
            chunks = []
            with torch.no_grad():
                for start in range(0, cause_bow.shape[0], 1024):
                    sl = slice(start, start + 1024)
                    fused, _ = model.encode_doc_theta(
                        NN._normalize_rows(cause_bow[sl]),
                        NN._normalize_rows(item_bow[sl]),
                        has_cause[sl])
                    chunks.append(fused)
            theta = torch.cat(chunks, dim=0).numpy()
            beta = np.asarray(model.get_beta(), dtype=float)
            if state_path:
                # occlusion-compatible state: the same key layout that
                # run_public_missing_field.py writes, so the new seeds can be
                # scored with the partial-occlusion protocol without retraining
                payload = {}
                if os.path.exists(state_path):
                    try:
                        payload = torch.load(state_path, map_location="cpu",
                                             weights_only=False)
                    except Exception:  # noqa: BLE001
                        payload = {}
                payload[f"fielddrop_on:{int(seed)}"] = {
                    "model_state": {k: v.detach().clone()
                                    for k, v in model.state_dict().items()},
                    "kwargs": model_kwargs,
                    "best_epoch": int(info["best_epoch"]),
                }
                os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
                torch.save(payload, state_path)
            if details_path:
                detail_records.append({
                    "model": "FC-NTM",
                    "seed": int(seed),
                    "best_epoch": int(info["best_epoch"]),
                    "K": int(beta.shape[0]),
                    "beta": beta.tolist(),
                    "theta": theta.tolist(),
                    "config_hash": config_hash({
                        "consistency_weight": 0.5,
                        "cross_reconstruction": True,
                        "cause_dropout": 0.30,
                        "item_dropout": 0.40,
                        "exclusivity_weight": 0.0,
                        "field_dropout": float(field_dropout),
                        "fusion_mode": ("gate" if fusion in ("gate", "poe")
                                        else (fusion if fusion == "fused" else "mean")),
                        "fused_recon": fusion in ("fused", "gate", "poe"),
                        "fused_recon_norm": fusion_norm,
                        "fused_recon_weight": float(fusion_weight),
                    }),
                    "epochs_budget": int(max_epochs) if max_epochs else None,
                    "resumed_from": (resume_file if resume_file
                                     and os.path.exists(resume_file) else None),
                    # full fitting identity: reuse must match every field
                    # (2026-10-11 review, item 2)
                    "fit_identity": RI.identity({
                        "script": "run_fcntm_train.details",
                        "seed": int(seed),
                        "k": int(K),
                        "n_vocab": int(corpus_train["n_vocab"]),
                        "n_docs": int(corpus_train["n_docs"]),
                        "vocab_digest": hashlib.sha1(
                            "\x1f".join(corpus_train["vocab"]).encode(
                                "utf-8")).hexdigest()[:16],
                        "data": RI.text_digest(
                            [corpus_train.get("doc_tokens")]),
                        "fields": RI.text_digest(
                            [corpus_train.get("reason_tokens"),
                             corpus_train.get("item_tokens")]),
                        "code": RI.code_version(),
                        "field_dropout": float(field_dropout),
                        "fusion": fusion,
                        "fusion_norm": fusion_norm,
                        "fusion_weight": float(fusion_weight),
                        "lr": float(lr or 0.0),
                        "epochs_budget": (int(max_epochs) if max_epochs
                                          else None),
                    }),
                })

            def write_row(model_name, topics, score_mat):
                ev = evaluate(topics, score_mat, corpus_test)
                row = {
                    "corpus": corpus_train.get("name", "corpus"),
                    "model": model_name,
                    "seed": seed,
                    "K_used": len(topics),
                    "best_epoch": info["best_epoch"],
                    "seconds": round(float(seconds), 2),
                    "topics": json.dumps(topics, ensure_ascii=False),
                    # digest of the trained state this row was scored from
                    "state_hash": RI.array_digest(beta, theta),
                    "code_hash": RI.code_version(),
                    **{k: round(float(v), 6) for k, v in ev.items()},
                }
                writer.writerow(row)
                f.flush()          # keep finished seeds on disk across restarts
                print("  ", model_name, "seed", seed,
                      {k: round(ev[k], 4) for k in ("CV", "NPMI", "IRBO")},
                      flush=True)

            # base read-out: decoder beta
            write_row("FC-NTM", top_words(beta, vocab), beta)

            # upgraded read-out: soft class TF-IDF (all K topics kept)
            # aggregated as two products so that no (docs x V) copy of the
            # combined counts is materialised
            soft_counts = (theta.T @ cause_bow.numpy()
                           + theta.T @ item_bow.numpy())
            soft_scores = class_tfidf(soft_counts)
            write_row("FC-NTM-cTFIDF", top_words(soft_scores, vocab),
                      soft_scores)
    if details_path and detail_records:
        # merge with any records written by an earlier attempt of the same run
        merged = {}
        if os.path.exists(details_path) and os.path.getsize(details_path) > 0:
            try:
                with open(details_path, encoding="utf-8") as fh:
                    for rec in json.load(fh):
                        merged[int(rec["seed"])] = rec
            except (ValueError, KeyError, TypeError):
                merged = {}
        for rec in detail_records:
            merged[int(rec["seed"])] = rec
        with open(details_path, "w", encoding="utf-8") as fh:
            json.dump([merged[k] for k in sorted(merged)], fh)
        print("saved", details_path)
    RI.write(out_path, ident)
    print("saved", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["private", "github", "agnews", "civil"],
                        default="github")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--out", default="")
    parser.add_argument("--append", action="store_true",
                        help="append rows instead of overwriting the CSV")
    parser.add_argument("--field-dropout", type=float,
                        default=config.FIELD_DROPOUT)
    parser.add_argument("--fusion", choices=["none", "fused", "gate", "poe"],
                        default="none",
                        help="train the document-level fused mixture "
                             "(fused/gate/poe) instead of leaving it an "
                             "untrained average; gate/poe need it enabled")
    parser.add_argument("--fusion-norm", choices=["joint", "balanced"],
                        default="joint",
                        help="joint: the fused decoder counts as one more "
                             "active stream, so every reconstruction/KL term "
                             "is divided by (n_streams + 1) and the consistency "
                             "and orthogonality penalties become relatively "
                             "1.5x heavier (historical behaviour). "
                             "balanced: per-stream terms keep their scale and "
                             "the fused reconstruction is added separately "
                             "with --fusion-weight")
    parser.add_argument("--fusion-weight", type=float, default=0.5,
                        help="weight of the fused reconstruction term under "
                             "--fusion-norm balanced")
    parser.add_argument("--recover-nan", action="store_true",
                        help="roll back + halve LR on non-finite loss")
    parser.add_argument("--max-epochs", type=int, default=0,
                        help="cap the epoch budget (0 = config default)")
    parser.add_argument("--lr", type=float, default=0.0,
                        help="override the Adam learning rate")
    parser.add_argument("--dump-details", default="",
                        help="also write beta/theta per seed to this JSON")
    parser.add_argument("--dump-state", default="",
                        help="also write each trained main model's state "
                             "(occlusion-compatible keys) to this file")
    parser.add_argument("--resume-every", type=int, default=50,
                        help="write a full resumable training state every N "
                             "epochs next to --dump-details")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore existing resumable states and train afresh")
    args = parser.parse_args()

    if args.corpus == "private":
        import corpus as C
        data = C.load_corpus()
        seeds = [int(x) for x in args.seeds.split(",")] if args.seeds \
            else config.SEEDS
        out_path = os.path.join(config.ROOT, args.out) if args.out \
            else os.path.join(config.ROOT, "outputs",
                              "fcntm_private_twomodel_per_seed.csv")
        run_corpus(data, data, seeds, args.k,
                   out_path, append=args.append,
                   field_dropout=args.field_dropout,
                   recover_nan=args.recover_nan,
                   max_epochs=args.max_epochs, lr=args.lr,
                   details_path=(os.path.join(config.ROOT, args.dump_details)
                                 if args.dump_details else None),
                   fusion=args.fusion, fusion_norm=args.fusion_norm,
                   fusion_weight=args.fusion_weight,
                   resume_every=args.resume_every,
                   allow_resume=not args.no_resume,
                   state_path=(os.path.join(config.ROOT, args.dump_state)
                               if args.dump_state else None))
    elif args.corpus == "github":
        import public_data_github as PGH
        data = PGH.load_github(1500, 500, need_doc_bow=False)
        train, test = data["train"], data["test"]
        del data
        gc.collect()
        train["name"] = "GitHubIssues-4repo"
        seeds = [int(x) for x in args.seeds.split(",")] if args.seeds \
            else [11, 23, 37]
        out_path = os.path.join(config.ROOT, args.out) if args.out \
            else os.path.join(config.ROOT, "outputs",
                              "fcntm_github_twomodel_per_seed.csv")
        run_corpus(train, test, seeds, args.k,
                   out_path, append=args.append,
                   field_dropout=args.field_dropout,
                   recover_nan=args.recover_nan,
                   max_epochs=args.max_epochs, lr=args.lr,
                   details_path=(os.path.join(config.ROOT, args.dump_details)
                                 if args.dump_details else None),
                   fusion=args.fusion, fusion_norm=args.fusion_norm,
                   fusion_weight=args.fusion_weight,
                   resume_every=args.resume_every,
                   allow_resume=not args.no_resume,
                   state_path=(os.path.join(config.ROOT, args.dump_state)
                               if args.dump_state else None))
    elif args.corpus == "civil":
        import public_data_civil as PC
        data = PC.load_civil(6000, 2000, need_doc_bow=False)
        train, test = data["train"], data["test"]
        del data
        gc.collect()
        train["name"] = "CivilEng-OpenAlex"
        seeds = [int(x) for x in args.seeds.split(",")] if args.seeds \
            else [11, 23, 37]
        out_path = os.path.join(config.ROOT, args.out) if args.out \
            else os.path.join(config.ROOT, "outputs",
                              "fcntm_civil_twomodel_per_seed.csv")
        run_corpus(train, test, seeds, args.k,
                   out_path, append=args.append,
                   field_dropout=args.field_dropout,
                   recover_nan=args.recover_nan,
                   max_epochs=args.max_epochs, lr=args.lr,
                   details_path=(os.path.join(config.ROOT, args.dump_details)
                                 if args.dump_details else None),
                   fusion=args.fusion, fusion_norm=args.fusion_norm,
                   fusion_weight=args.fusion_weight,
                   resume_every=args.resume_every,
                   allow_resume=not args.no_resume,
                   state_path=(os.path.join(config.ROOT, args.dump_state)
                               if args.dump_state else None))
    else:
        import public_data as PD
        data = PD.load_agnews(1500, 1900, need_doc_bow=False)
        train, test = data["train"], data["test"]
        del data
        gc.collect()
        train["name"] = "AGNews"
        seeds = [int(x) for x in args.seeds.split(",")] if args.seeds \
            else [11, 23, 37]
        out_path = os.path.join(config.ROOT, args.out) if args.out \
            else os.path.join(config.ROOT, "outputs",
                              "fcntm_agnews_twomodel_per_seed.csv")
        run_corpus(train, test, seeds, args.k,
                   out_path, append=args.append,
                   field_dropout=args.field_dropout,
                   recover_nan=args.recover_nan,
                   max_epochs=args.max_epochs, lr=args.lr,
                   details_path=(os.path.join(config.ROOT, args.dump_details)
                                 if args.dump_details else None),
                   fusion=args.fusion, fusion_norm=args.fusion_norm,
                   fusion_weight=args.fusion_weight,
                   resume_every=args.resume_every,
                   allow_resume=not args.no_resume,
                   state_path=(os.path.join(config.ROOT, args.dump_state)
                               if args.dump_state else None))


if __name__ == "__main__":
    main()
