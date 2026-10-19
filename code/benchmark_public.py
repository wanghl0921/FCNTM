# -*- coding: utf-8 -*-
"""AG News public benchmark: train on public train split, evaluate on official
test split.  Runs classical, neural and contextual baselines plus FC-NTM.

AG News provides two natural fields (title / description) so FC-NTM keeps its
field-collaborative structure: field1 = title, field2 = description.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy import sparse

import baselines as BL
import metrics as M
import new_baselines as NB
import public_data as PD
import public_data_civil as PC
import public_data_github as PGH
import result_identity as RI
from config import FIG_DIR, MAIN_K, OUT_DIR, TOPN
from embed_public import embed_texts, embed_words
from experiment import (
    _prepare,
    _run_ctmlite,
    _run_entm,
    _run_fcntm,
    _run_prodlda,
    evaluate_run,
)

torch.set_num_threads(8)
warnings.filterwarnings("ignore")

PUBLIC_MODELS = [
    "LDA",
    "BTM",
    "NMF",
    # The paper calls this baseline ProdLDA (Srivastava & Sutton, 2017);
    # "NeuralLDA" is kept as the historical name used in the stored result
    # files so that new runs stay mergeable with them.
    "NeuralLDA",
    "ETM",
    "CTMLite",
    "CombinedTM",
    "BERTopic",
    "Top2Vec",
    "FC-NTM",
]


# ``--readout`` only changes the two non-parametric baselines, which have no
# topic-word matrix of their own: they are re-read from their document-topic
# assignment.  Every other model is always scored on its own beta matrix.
NATIVE_READOUT_MODELS = ("BERTopic", "Top2Vec")


def actual_readout(model: str, requested: str) -> str:
    """The read-out that really produced this row's topic list."""
    return requested if model in NATIVE_READOUT_MODELS else "beta"


def row_identity(model: str, requested: str, rec: Dict[str, object]) -> dict:
    """Columns that pin a result row to the state it was scored from."""
    return {
        "actual_readout": actual_readout(model, requested),
        # digest of the scored state: identical numbers imply an identical hash
        "state_hash": RI.array_digest(rec.get("beta"), rec.get("theta")),
        "code_hash": RI.code_version(),
    }


def _row_is_complete(row: Dict[str, object]) -> bool:
    """True only for rows that hold a usable metric and carry no error.

    ``not row.get("error")`` is *not* a valid completeness test: an empty CSV
    cell is read back as ``float('nan')`` (or ``''``), both of which are
    truthy, so every valid row was treated as missing and the benchmark never
    skipped anything on resume.
    """
    err = row.get("error")
    if err is not None and not (isinstance(err, float) and np.isnan(err)):
        if str(err).strip() not in ("", "nan", "None"):
            return False
    if str(row.get("status", "")).strip().lower() in ("failed", "error"):
        return False
    cv = row.get("CV")
    try:
        return bool(np.isfinite(float(cv)))
    except (TypeError, ValueError):
        return False


def _evaluate_on(
    record: Dict[str, object],
    corpus: Dict[str, object],
    topn: int = TOPN,
) -> Dict[str, float]:
    topics = record["topics"]
    scores = M.coherence_scores(
        topics,
        corpus["doc_tokens"],
        corpus["corpus"],
        corpus["dictionary"],
        topn=topn,
    )
    beta = record["beta"]
    scores["EXCL"] = M.exclusivity(beta, topn=topn)
    scores["EXCL_KADJ"] = M.exclusivity_kadj(beta, topn=topn)
    scores["IRBO"] = M.irbo_diversity(topics)
    return scores


def _topics_table(
    record: Dict[str, object],
    vocab: List[str],
    topn: int = TOPN,
) -> List[List[str]]:
    beta = record["beta"]
    return M.topics_from_matrix(beta, vocab, topn)


def run_model(
    model_name: str,
    data: Dict[str, Dict[str, object]],
    K: int,
    seed: int,
    word_emb: np.ndarray,
    doc_ctx: np.ndarray,
    topn: int = TOPN,
    readout: str = "ctfidf",
) -> Dict[str, object]:
    train = data["train"]
    prep = _prepare(train)
    t0 = time.time()
    if model_name == "LDA":
        rec = BL.run_lda(train["corpus"], train["dictionary"], train["vocab"], K, seed)
    elif model_name == "BTM":
        rec = BL.run_btm(train["doc_tokens"], train["dictionary"], train["vocab"], K, seed)
    elif model_name == "NMF":
        mat = sparse.csr_matrix(train["doc_bow"].astype(np.float32))
        rec = BL.run_nmf(mat, train["vocab"], K, seed)
    elif model_name in ("NeuralLDA", "ProdLDA"):
        rec = _run_prodlda(train, K, seed, prep["doc_bow"])
    elif model_name == "ETM":
        rec = _run_entm(train, K, seed, prep["doc_bow"], word_emb)
    elif model_name == "CTMLite":
        def ctm_logger(epoch, loss, kl_w):
            if epoch % 25 == 0 or epoch <= 5:
                print(f"  [CTMLite seed {seed}] epoch {epoch} loss {loss:.3f} klw {kl_w:.4f}")
        rec = _run_ctmlite(train, K, seed, doc_ctx, logger=ctm_logger)
    elif model_name == "FC-NTM":
        rec = _run_fcntm(train, K, seed, emb=word_emb)
    elif model_name == "CombinedTM":
        bow_csr = sparse.csr_matrix(train["doc_bow"].astype(np.float32))
        rec = NB.run_combinedtm(bow_csr, doc_ctx, train["vocab"], K, seed)
    elif model_name == "BERTopic":
        rec = NB.run_bertopic(
            train["doc_texts"],
            train["doc_texts"],
            doc_ctx,
            train["vocab"],
            K,
            seed,
            bow=train["doc_bow"],
            topn=topn,
            readout=readout,
        )
    elif model_name == "Top2Vec":
        docs_bow = [" ".join(t) for t in train["doc_tokens"]]
        rec = NB.run_top2vec(
            docs_bow, train["vocab"], seed, workers=1,
            bow=train["doc_bow"], topn=topn, readout=readout,
        )
    else:
        raise ValueError(model_name)
    # ``--topn`` must apply to *every* model, not only to the two non-parametric
    # ones: the neural/classical baselines build their topic lists from their
    # own beta matrix with the global TOPN, which silently broke the
    # "same number of topic words" protocol whenever --topn was changed.
    beta = rec.get("beta")
    if beta is not None and topn != TOPN:
        rec["topics"] = M.topics_from_matrix(
            np.asarray(beta, dtype=float), train["vocab"], topn)
    rec["seconds"] = time.time() - t0
    rec["model"] = model_name
    rec["K_target"] = K
    rec["seed"] = seed
    rec["topic_count"] = int(len(rec["topics"]))
    return rec


def summarize_rows(rows: List[Dict[str, object]]) -> pd.DataFrame:
    """Per-model mean/std over *valid, de-duplicated* runs.

    Failed rows (error string or missing metric) are dropped before the
    aggregation instead of being averaged in as NaN, and each
    (model, K_target, seed) run is counted once so retried runs cannot be
    double counted.
    """
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    keep = np.array([_row_is_complete(r) for r in frame.to_dict("records")])
    valid = frame[keep]
    # group/deduplicate on the full run key: the same model+seed at a different
    # K (or with a different read-out / top-n) is a different experiment and
    # must not be averaged into one row
    key_cols = [c for c in ("model", "K_target", "topn", "readout") if c in valid.columns]
    dedup_cols = key_cols + ["seed"]
    if dedup_cols:
        valid = valid.drop_duplicates(subset=dedup_cols, keep="last")
    failed_counts = (frame[~keep]["model"].value_counts().to_dict()
                     if (~keep).any() else {})
    metric_cols = ["CV", "NPMI", "UCI", "UMass", "EXCL", "EXCL_KADJ", "IRBO"]
    summary = []
    for keys, g in valid.groupby(key_cols or ["model"], sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(key_cols or ["model"], keys))
        row = {**key_map, "n_seeds": len(g),
               "K_target": int(g["K_target"].iloc[0])}
        row["n_failed"] = int(failed_counts.get(key_map.get("model"), 0))
        row["topic_count_mean"] = float(np.mean(g["topic_count"]))
        for col in metric_cols:
            vals = g[col].astype(float)
            row[f"{col}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
            row[f"{col}_std"] = (
                float(np.nanstd(vals, ddof=1))
                if len(vals) > 1 and np.isfinite(vals).sum() > 1
                else 0.0
            )
        summary.append(row)
    return pd.DataFrame(summary)


def print_topics(model_name: str, rec: Dict[str, object], n: int = 3) -> None:
    print(f"--- {model_name} (seed {rec['seed']}, topics {rec['topic_count']}, "
          f"time {rec['seconds']:.1f}s) ---")
    for i, topic in enumerate(rec["topics"][:n]):
        print("   ", i, " ".join(topic))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["agnews", "github", "civil"],
                        default="github")
    parser.add_argument("--models", default=",".join(PUBLIC_MODELS))
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--train-per-class", type=int, default=1500)
    parser.add_argument("--test-per-class", type=int, default=0,
                        help="0 = dataset default (1900 per AG News class, "
                             "500 per GitHub repository)")
    parser.add_argument("--out", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--topn", type=int, default=TOPN,
                        help="number of topic words kept for every model")
    parser.add_argument("--readout", choices=["ctfidf", "native"],
                        default="ctfidf",
                        help="read-out for BERTopic/Top2Vec: ctfidf aggregates "
                             "their document-topic assignment over the shared "
                             "vocabulary so they report the same number of "
                             "topic words as every other model; native keeps "
                             "their own representative words")
    parser.add_argument("--details-out", default="",
                        help="also dump per-run beta (and theta) to this JSON")
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    models = [m for m in args.models.split(",") if m]
    seeds = [int(s) for s in args.seeds.split(",")]

    test_per_class = args.test_per_class or (
        1900 if args.dataset == "agnews" else 500)
    expected_test_override = 0
    if args.dataset == "civil":
        # the civil corpus uses a flat 6000/2000 split (no per-class layout)
        data = PC.load_civil(6000, 2000)
        ds_tag = "civil"
        expected_test_override = 2000
    elif args.dataset == "agnews":
        data = PD.load_agnews(
            args.train_per_class,
            test_per_class,
            force_splits=args.refresh_data,
        )
        ds_tag = "agnews"
    else:
        data = PGH.load_github(
            args.train_per_class,
            test_per_class,
            force_splits=args.refresh_data,
        )
        ds_tag = "github"
    train = data["train"]
    expected_test = (expected_test_override if expected_test_override
                     else 4 * test_per_class)
    if data["test"]["n_docs"] != expected_test:
        raise SystemExit(
            f"cached test split has {data['test']['n_docs']} documents but "
            f"--test-per-class={test_per_class} implies {expected_test}; "
            f"re-run with --refresh-data to rebuild it")
    print("train docs", train["n_docs"], "vocab", train["n_vocab"])
    print("test docs", data["test"]["n_docs"])
    print("meta", data.get("meta"))

    need_word_emb = any(m == "ETM" for m in models)
    word_emb = None
    if need_word_emb:
        word_emb = embed_words(train["vocab"])
        norms = np.linalg.norm(word_emb, axis=-1, keepdims=True)
        word_emb = word_emb / (norms + 1e-8)
        word_emb = word_emb.astype(np.float32)
    need_doc_emb = any(m in ("CTMLite", "CombinedTM", "BERTopic") for m in models)
    doc_kind = "civil_doc" if ds_tag == "civil" else "agnews_doc"
    doc_ctx = embed_texts(train["doc_texts"], kind=doc_kind) if need_doc_emb else None
    if doc_ctx is not None:
        print("doc embeddings", doc_ctx.shape)

    out_csv = OUT_DIR / (
        args.out or (
            f"public_benchmark_runs_{ds_tag}"
            f"train{train['n_docs']}_test{data['test']['n_docs']}.csv"
        )
    )
    rows: List[Dict[str, object]] = []
    if out_csv.exists() and not args.no_resume:
        try:
            rows = pd.read_csv(out_csv).to_dict("records")
            print("resuming with", len(rows), "existing rows")
        except Exception:  # noqa: BLE001
            rows = []
    if rows:
        key = [c for c in ("model", "seed", "K_target") if c in rows[0]]
        frame = pd.DataFrame(rows)
        if key:
            before = len(frame)
            frame = frame.drop_duplicates(subset=key, keep="last")
            if len(frame) != before:
                print(f"dropped {before - len(frame)} duplicated run rows")
        rows = frame.to_dict("records")

    details: List[Dict[str, object]] = []
    details_path = Path(args.details_out) if args.details_out else None
    if details_path is not None and details_path.exists():
        try:
            details = json.loads(details_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            details = []

    def save_details() -> None:
        if details_path is not None:
            details_path.write_text(
                json.dumps(details), encoding="utf-8")

    def _run_key(r):
        return (r["model"], int(r["seed"]), int(r.get("K_target") or args.k),
                r.get("readout") or args.readout, int(r.get("topn") or args.topn))

    done = {_run_key(r) for r in rows if _row_is_complete(r)}
    print(f"resume: {len(done)} of {len(models) * len(seeds)} runs already done")
    for model in models:
        for seed in seeds:
            if (model, seed, args.k, args.readout, args.topn) in done:
                print("skip", model, seed)
                continue
            try:
                rec = run_model(model, data, args.k, seed, word_emb, doc_ctx,
                                topn=args.topn, readout=args.readout)
                test_scores = _evaluate_on(rec, data["test"], topn=args.topn)
                rec["metrics"] = test_scores
                row = {
                    "model": rec["model"],
                    "seed": seed,
                    "K_target": args.k,
                    "seconds": round(rec["seconds"], 2),
                    "topic_count": rec["topic_count"],
                    "n_outlier_docs": rec.get("n_outlier_docs", 0),
                    "epochs_run": rec.get("epochs_run", ""),
                    "best_epoch": rec.get("best_epoch", ""),
                    "topics": json.dumps(rec["topics"], ensure_ascii=False),
                    "status": "ok",
                    "topn": args.topn,
                    # NOTE: this column records the *requested* read-out, not
                    # the one each model was actually scored with.  --readout
                    # only re-reads BERTopic/Top2Vec; every other model is
                    # always scored on its own beta matrix (see the fair
                    # read-out table in the paper).  Kept as-is so that the
                    # resume key of the existing per-seed files stays valid.
                    "readout": args.readout,
                    # what was really scored, plus a digest of the scored state
                    **row_identity(model, args.readout, rec),
                    **{k: v for k, v in test_scores.items()},
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
                if details_path is not None:
                    details = [d for d in details
                               if not (d["model"] == model and int(d["seed"]) == seed)]
                    details.append({
                        "model": model,
                        "seed": int(seed),
                        "K": int(rec["topic_count"]),
                        "beta": np.asarray(rec["beta"], dtype=float).tolist(),
                        "theta": (np.asarray(rec["theta"], dtype=float).tolist()
                                  if rec.get("theta") is not None else None),
                    })
                    save_details()
                print_topics(model, rec)
                print("   ", {k: round(v, 4) for k, v in test_scores.items()})
            except Exception as exc:  # noqa: BLE001
                print("ERROR", model, seed, repr(exc))
                traceback.print_exc()
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "K_target": args.k,
                        "seconds": 0.0,
                        "topic_count": 0,
                        "n_outlier_docs": 0,
                        "error": str(exc),
                        "status": "failed",
                        "topn": args.topn,
                        "readout": args.readout,
                        "actual_readout": actual_readout(model, args.readout),
                    }
                )
                pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = summarize_rows(rows)
    summary_path = OUT_DIR / (args.summary or "public_benchmark_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print("saved ->", out_csv, "|", summary_path)


if __name__ == "__main__":
    main()
