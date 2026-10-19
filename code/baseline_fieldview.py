# -*- coding: utf-8 -*-
"""Field-restricted read-out for the classical baselines (2026-10-07 review,
item 10; identities hardened after the 2026-10-11 review).

The paper's field-view table restricts a model's topic-word matrix to one
field's training vocabulary and scores it against that field's own test text.
This script computes exactly the same read-out for LDA, BTM, NMF and ProdLDA.

Every cached fit carries a full fitting identity; a cache that does not match
is refit instead of adopted (shared BTM/NMF fit files are never overwritten:
a separate field-view cache is written next to them).

Each row also records the *evaluation* identity of its corpus protocol and the
identity of the fit it was scored from (2026-10-12 review, item C4).  Fits
that predate the identity scheme are only used with ``--adopt-legacy-fits``,
in which case the row keeps a content digest (``legacy:<digest>``) instead of
a run identity.

    python baseline_fieldview.py --models LDA,NMF,BTM --corpora github,agnews
    python baseline_fieldview.py --models ProdLDA --corpora github,agnews,civil
"""

from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd
import torch

import metrics as M
import occlusion_robustness as O
import result_identity as RI
import run_public_missing_field as RMF
from config import TOPN
from seed_identity import (fieldview_identity, fit_identity,
                           load_cached_npz, save_cached_npz)

OUT = O.OUT_DIR
FIT = OUT / "rerun"


def legacy_beta(path, K: int, n_vocab: int, adopt: bool):
    """Read an unpinned legacy fit and record its content digest.

    Fits written before the identity scheme existed carry no ``identity_id``;
    they are never adopted silently.  With ``adopt`` the matrix is read and
    every row keeps ``legacy:<digest>`` for it, so the exact numbers still
    have a content-addressed provenance (2026-10-12 review, item C4).
    """
    if not (adopt and path.exists()):
        return None
    raw = np.load(path, allow_pickle=True)
    if "beta" not in raw.files:
        return None
    beta = raw["beta"]
    beta = np.asarray(beta.item() if beta.ndim == 0 else beta, dtype=float)
    if beta.shape != (K, n_vocab):
        return None
    print(f"  adopting unpinned legacy fit {path.name}", flush=True)
    return beta, "legacy:" + RI.array_digest(beta)


def baseline_beta(model: str, corpus: str, seed: int, K: int, train, doc_bow,
                  adopt_legacy: bool = False):
    """``(beta, fit_id)`` of one baseline (identity-checked cache)."""
    ident = fit_identity(f"baseline_fieldview.{model}", corpus, train, K, seed)
    shape = {"beta": (K, len(train["vocab"]))}
    if model == "LDA":
        return np.asarray(O.LDAAdapter(train, K, seed).beta,
                          dtype=float), ident["id"]
    if model in ("NMF", "BTM"):
        shared = FIT / f"{model.lower()}_fit_{corpus}_seed{seed}.npz"
        cached = load_cached_npz(shared, ident, ("beta",), shape)
        if cached is not None:
            return np.asarray(cached["beta"], dtype=float), ident["id"]
        cache = FIT / f"fieldview_fit_{model}_{corpus}_seed{seed}.npz"
        cached = load_cached_npz(cache, ident, ("beta",), shape)
        if cached is not None:
            return np.asarray(cached["beta"], dtype=float), ident["id"]
        legacy = legacy_beta(shared, K, len(train["vocab"]), adopt_legacy)
        if legacy is not None:
            return legacy
        adapter = O.NMFAdapter(train, K, seed) if model == "NMF" \
            else O.BTMAdapter(train, K, seed)
        beta = np.asarray(adapter.beta, dtype=float)
        save_cached_npz(cache, ident, beta=beta)
        return beta, ident["id"]
    if model == "ProdLDA":
        cache = FIT / f"fieldview_fit_ProdLDA_{corpus}_seed{seed}.npz"
        cached = load_cached_npz(cache, ident, ("beta",), shape)
        if cached is not None:
            return np.asarray(cached["beta"], dtype=float), ident["id"]
        legacy = legacy_beta(cache, K, len(train["vocab"]), adopt_legacy)
        if legacy is not None:
            return legacy
        adapter = O.ProdLDAAdapter(train, K, seed, doc_bow)
        beta = np.asarray(adapter.model.get_beta(), dtype=float)
        save_cached_npz(cache, ident, beta=beta)
        return beta, ident["id"]
    raise ValueError(model)


def file_identity(frame, k: int, seeds) -> dict:
    """Identity of the *file*: every corpus protocol and fit it contains.

    A regeneration may merge several invocations into one CSV (LDA/NMF/BTM in
    one run, ProdLDA in another), so the sidecar has to describe the merged
    table rather than the invocation that happened to write it last.
    """
    corpora = sorted(str(c) for c in frame["dataset"].unique())
    eval_ids = {}
    for corpus in corpora:
        ids = sorted(str(i) for i in
                     frame.loc[frame["dataset"] == corpus, "eval_id"].unique())
        eval_ids[corpus] = ids[0] if len(ids) == 1 else ids
    return RI.identity({
        "script": "baseline_fieldview.main", "k": int(k),
        "seeds": [int(s) for s in seeds], "corpora": corpora,
        "models": sorted(str(m) for m in frame["model"].unique()),
        "eval_ids": eval_ids,
        "fit_ids": sorted({str(f) for f in frame["fit_id"].unique()}),
        "code": RI.code_version()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="LDA,NMF,BTM")
    ap.add_argument("--corpora", default="github,agnews,civil")
    ap.add_argument("--seeds", default="11,23,37")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--adopt-legacy-fits", action="store_true",
                    help="use archived fits that carry no identity sidecar, "
                         "recording their content digest in every row")
    args = ap.parse_args()

    rows = []
    eval_ids = {}
    seeds = [int(s) for s in args.seeds.split(",")]
    models = [m for m in args.models.split(",") if m]
    for corpus in [c for c in args.corpora.split(",") if c]:
        train, test = O.load_corpus(corpus)
        vocab = train["vocab"]
        cause_ids = RMF.field_vocab_ids(train["reason_tokens"], vocab)
        item_ids = RMF.field_vocab_ids(train["item_tokens"], vocab)
        counts = (np.asarray(train["reason_bow"])
                  + np.asarray(train["item_bow"])).astype(np.float32)
        doc_bow = torch.from_numpy(counts)
        eval_ident = fieldview_identity(corpus, train, test, seeds, args.k)
        eval_ids[corpus] = eval_ident["id"]
        for model in models:
            for seed in seeds:
                beta, fit_id = baseline_beta(
                    model, corpus, seed, args.k, train, doc_bow,
                    args.adopt_legacy_fits)
                beta = np.asarray(beta, dtype=float)
                for view, ids, field_tokens in (
                        ("background", cause_ids,
                         [t for t in test["reason_tokens"] if t]),
                        ("summary", item_ids,
                         [t for t in test["item_tokens"] if t])):
                    field_bows = [test["dictionary"].doc2bow(t)
                                  for t in field_tokens]
                    topics = RMF.topics_in_field(beta, ids, vocab)
                    ev = M.coherence_scores(topics, field_tokens, field_bows,
                                            test["dictionary"], topn=TOPN)
                    rows.append({
                        "dataset": corpus, "model": model, "seed": seed,
                        "K": int(beta.shape[0]), "view": view,
                        "eval_id": eval_ident["id"], "fit_id": fit_id,
                        "n_topics": int(ev["n_topics"]),
                        "CV": round(float(ev["CV"]), 6),
                        "CV_n": int(ev["CV_n"]),
                        "NPMI": round(float(ev["NPMI"]), 6),
                        "NPMI_n": int(ev["NPMI_n"]),
                    })
                    print(f"[{corpus} {model} seed {seed}] {view}: "
                          f"CV {ev['CV']:.4f} NPMI {ev['NPMI']:.4f} "
                          f"(n={ev['CV_n']})", flush=True)
                del beta
                gc.collect()
        del train, test, doc_bow
        gc.collect()
    out = OUT / "baseline_fieldview.csv"
    frame = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        for column in ("eval_id", "fit_id"):
            if column not in old.columns:
                old[column] = ""
        mapped = old["dataset"].map(eval_ids)
        superseded = mapped.notna() & \
            (old["eval_id"].fillna("") != mapped.fillna(""))
        if superseded.any():
            backup = out.with_name(
                "backup_baseline_fieldview_pre_identity" + out.suffix)
            if not backup.exists():
                backup.write_bytes(out.read_bytes())
                print("previous rows backed up to", backup.name)
            old = old[~superseded]
        frame = pd.concat([old, frame], ignore_index=True)
        frame = frame.drop_duplicates(
            subset=["dataset", "model", "seed", "view", "K", "eval_id"],
            keep="last")
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    RI.write(out, file_identity(frame, args.k, seeds))
    print("\nsaved", out)
    print(frame.pivot_table(index=["dataset", "model"], columns="view",
                            values="CV").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())