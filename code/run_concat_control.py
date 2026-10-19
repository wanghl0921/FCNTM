# -*- coding: utf-8 -*-
"""Concatenated-field single-encoder control for the FC-NTM paper.

Trains ``models.FCNTMConcat`` -- one encoder over ``[x_bg; x_sm]``, one
variational stream, the same shared linear decoder, field weights, KL
warm-up and topic-orthogonality penalty as FC-NTM -- on the same corpora,
seeds, vocabulary and read-outs, and writes the same two read-outs through
``fair_readout.evaluate_both``.  A per-seed state file is saved so the model
can be reused for the title-only archival task.

Usage:
    python run_concat_control.py --corpora github,agnews,civil \
        --seeds 11,23,37 --k 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
import models as NN
import result_identity as RI
from fair_readout import evaluate_both
from config import OUT_DIR
from seed_identity import vocab_digest


def load_corpus(name: str):
    if name == "agnews":
        import public_data as PD
        return PD.load_agnews(1500, 1900)
    if name == "civil":
        import public_data_civil as PC
        return PC.load_civil(6000, 2000)
    import public_data_github as PGH
    return PGH.load_github(1500, 500)


def concat_identity(corpus: str, train: dict, K: int, num_epochs: int,
                    lr: float) -> dict:
    """Full fitting identity of the concatenated-field control."""
    return {
        "corpus": corpus,
        "vocab_digest": vocab_digest(train),
        "n_vocab": int(train["n_vocab"]),
        "k": int(K),
        "code": RI.code_version(),
        "lr": float(lr),
        "num_epochs": int(num_epochs),
        "field_dropout": 0.5,
        "data": RI.text_digest([train.get("doc_tokens")]),
        "fields": RI.text_digest([train.get("reason_tokens"),
                                  train.get("item_tokens")]),
    }


def concat_eval_identity(corpus: str, test: dict, K: int) -> dict:
    """Evaluation identity: what a stored score row was computed on.

    2026-10-15 v3 review C1: matching only the *fitting* identity reused
    stale scores after the test set had changed.  Fitting and evaluation
    are now tracked separately.
    """
    return {
        "corpus": corpus,
        "test_data": RI.text_digest([test.get("doc_tokens")]),
        "test_fields": RI.text_digest([test.get("reason_tokens"),
                                       test.get("item_tokens")]),
        "readouts": ["beta", "ctfidf"],
        "k": int(K),
    }


def _fit_matches(rec, expected: dict) -> bool:
    """Full fitting identity; missing fields are never filled with defaults."""
    stored = rec.get("identity") or {}
    required = ("corpus", "vocab_digest", "n_vocab", "k", "code",
                "lr", "num_epochs", "field_dropout", "data", "fields")
    if any(key not in stored for key in required):
        return False
    if any(stored[key] != expected[key] for key in required):
        return False
    return not any(key in expected and expected[key] != value
                   for key, value in stored.items())


def _eval_matches(rec, expected: dict) -> bool:
    stored = rec.get("eval_identity") or {}
    return all(stored.get(key) == value for key, value in expected.items())


def _rows_valid(rec, seed: int):
    """Both read-outs for exactly this seed, with all reported metrics     
    finite."""
    rows = rec.get("rows")
    if not rows:
        return None
    try:
        frame = pd.DataFrame(rows)
        # 2026-10-15 v4 review R2: rows without the full metric columns are
        # incomplete caches and must not be reused
        required_columns = {"seed", "readout", "CV", "NPMI", "UMass",
                            "EXCL_KADJ", "IRBO"}
        if not required_columns.issubset(frame.columns):
            return None
        if set(frame["seed"].astype(int)) != {int(seed)}:
            return None
        if set(frame["readout"].astype(str)) != {"beta", "ctfidf"}:
            return None
        if frame.duplicated(subset=["seed", "readout"]).any():
            return None
        for column in ("CV", "NPMI", "UMass", "EXCL_KADJ", "IRBO"):
            if not np.isfinite(
                    frame[column].to_numpy(dtype=float)).all():
                return None
    except Exception:  # noqa: BLE001
        return None
    return rows


def reusable_rows(rec, expected_fit: dict, expected_eval: dict, seed: int):
    """Complete reuse: fitting identity, evaluation identity, valid rows."""
    if not rec or not _fit_matches(rec, expected_fit):
        return None
    if not _eval_matches(rec, expected_eval):
        return None
    return _rows_valid(rec, seed)


def run_corpus(name: str, seeds, K: int, out_dir: Path,
               num_epochs: int, lr: float) -> None:
    data = load_corpus(name)
    train, test = data["train"], data["test"]
    vocab = test["vocab"]
    cause_bow = torch.from_numpy(train["reason_bow"]).float()
    item_bow = torch.from_numpy(train["item_bow"]).float()
    has_cause = torch.tensor(
        [len(t) > 0 for t in train["reason_tokens"]], dtype=torch.bool)
    doc_counts = (train["reason_bow"] + train["item_bow"]).astype(float)

    rows = []
    state_path = out_dir / f"concat_state_{name}.pt"
    expected = concat_identity(name, train, K, num_epochs, lr)
    expected_eval = concat_eval_identity(name, test, K)
    for seed in seeds:
        payload = {}
        if state_path.exists():
            try:
                payload = torch.load(state_path, map_location="cpu",
                                     weights_only=False)
            except Exception:  # noqa: BLE001
                payload = {}
        rec = payload.get(f"seed:{int(seed)}")
        if rec and _fit_matches(rec, expected):
            if _eval_matches(rec, expected_eval):
                reused = _rows_valid(rec, seed)
                if reused is not None:
                    print(f"  [concat {name} seed {seed}] reusing completed "
                          "run", flush=True)
                    rows.extend(reused)
                    continue
            # 2026-10-15 v3 review C1: the trained weights are still valid;
            # only the evaluation inputs changed, so re-score the stored
            # model instead of retraining or reusing stale scores
            print(f"  [concat {name} seed {seed}] test set or read-out "
                  "protocol changed; re-evaluating the stored model",
                  flush=True)
            model = NN.FCNTMConcat(train["n_vocab"], K, **rec["kwargs"])
            model.load_state_dict(rec["model_state"])
            model.eval()
            chunks = []
            with torch.no_grad():
                for start in range(0, cause_bow.shape[0], 1024):
                    sl = slice(start, start + 1024)
                    theta_b, _ = model.encode_doc_theta(
                        NN._normalize_rows(cause_bow[sl]),
                        NN._normalize_rows(item_bow[sl]),
                        has_cause[sl])
                    chunks.append(theta_b)
            theta = torch.cat(chunks, dim=0).numpy()
            beta = np.asarray(model.get_beta(), dtype=float)
            seed_rows = evaluate_both(
                beta, theta, vocab, doc_counts, train, test,
                "Concat", seed, "fair_readout")
            rows.extend(seed_rows)
            payload[f"seed:{int(seed)}"] = {
                "rows": seed_rows,
                "model_state": rec["model_state"],
                "kwargs": rec["kwargs"],
                "best_epoch": rec.get("best_epoch", -1),
                "identity": rec["identity"],
                "eval_identity": dict(expected_eval),
            }
            torch.save(payload, state_path)
            continue
        if rec is not None:
            print(f"  [concat {name} seed {seed}] stored run does not match "
                  "the current fitting identity; retraining", flush=True)
        NN.set_torch_seed(seed)
        model = NN.FCNTMConcat(
            train["n_vocab"], K,
            cause_dropout=0.30, item_dropout=0.40, field_dropout=0.5,
        )

        def loss_fn(idx, kl_weight):
            cause_batch = cause_bow[idx]
            item_batch = item_bow[idx]
            return model.train_step(
                NN._normalize_rows(cause_batch),
                NN._normalize_rows(item_batch),
                cause_batch, item_batch,
                has_cause[idx], kl_weight)

        logger = (lambda ep, loss, klw: print(
            f"    [train] epoch {ep} loss {loss:.3f} klw {klw:.4f}",
            flush=True) if ep % 25 == 0 or ep <= 5 else None)
        t0 = time.time()
        info = NN.train_topic_vae_valid(
            model, loss_fn, loss_fn, train["n_docs"], seed,
            val_fraction=0.2, kl_max=0.05, num_epochs=num_epochs,
            lr=lr, logger=logger)
        seconds = time.time() - t0
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, cause_bow.shape[0], 1024):
                sl = slice(start, start + 1024)
                theta, _ = model.encode_doc_theta(
                    NN._normalize_rows(cause_bow[sl]),
                    NN._normalize_rows(item_bow[sl]),
                    has_cause[sl])
                chunks.append(theta)
        theta = torch.cat(chunks, dim=0).numpy()
        beta = np.asarray(model.get_beta(), dtype=float)
        seed_rows = evaluate_both(
            beta, theta, vocab, doc_counts, train, test,
            "Concat", seed, "fair_readout")
        rows.extend(seed_rows)
        print(f"  [concat {name} seed {seed}] best_epoch "
              f"{info['best_epoch']} ({seconds:.0f}s)", flush=True)

        payload[f"seed:{int(seed)}"] = {
            "rows": seed_rows,
            "model_state": {k: v.detach().clone()
                            for k, v in model.state_dict().items()},
            "kwargs": {
                "cause_dropout": 0.30, "item_dropout": 0.40,
                "field_dropout": 0.5,
            },
            "best_epoch": int(info["best_epoch"]),
            # 2026-10-15 review: the full fitting identity (corpus, data,
            # vocabulary, K, budget, LR, dropout, code); reuse compares it
            # field by field and refuses mismatching states
            "identity": dict(expected),
            "eval_identity": dict(expected_eval),
        }
        torch.save(payload, state_path)

    frame = pd.DataFrame(rows)
    out_path = out_dir / f"concat_control_{name}.csv"
    frame.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(frame.round(4).to_string(index=False))
    print("saved", out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="github,agnews,civil")
    ap.add_argument("--seeds", default="11,23,37")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=config.PAPER_EPOCHS)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (OUT_DIR / "rerun")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    for name in args.corpora.split(","):
        print(f"=== concat control: {name} seeds {seeds} ===", flush=True)
        run_corpus(name, seeds, args.k, out_dir, args.epochs, args.lr)


if __name__ == "__main__":
    main()
