# -*- coding: utf-8 -*-
"""Component ablation for FC-NTM on a public corpus.

Configs toggled one at a time:
  full            cross-field reconstruction + JS consistency + field masking
  no-consistency  drop the symmetric JS consistency term
  no-cross        each stream reconstructs only its own field
  no-mask         drop field-level word masking (cause/item dropout)

Field dropout is ablated separately in run_public_missing_field.py, where its
effect on the missing-field sub-population is the measured quantity.

The ``full`` row is normally reused from a details dump so the reference run is
not paid for twice.

Usage
-----
python run_ablation_public.py --dataset github --seeds 11,23,37 \
    --reuse-details outputs/github_fcntm_details.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import config as cfg
import metrics as M
import models as NN
import result_identity as RI
from run_fcntm_train import class_tfidf, evaluate, top_words

# ``run_fcntm_train`` is imported above and already pins the thread counts at
# import time; torch raises if set_num_interop_threads is called a second time,
# so only the intra-op thread count is set here.
torch.set_num_threads(8)

OUT_DIR = Path(__file__).resolve().parent / "outputs"

# The reference configuration is the model actually reported in the paper.
# The previous revision of this script started from a *different* base model
# (field_dropout 0, no fused reconstruction, mean fusion), so its "full" row was
# not the paper's full model and every ablation changed more than one factor.
FULL_CONFIG = {
    "consistency_weight": 0.5,
    "cross_reconstruction": True,
    "cause_dropout": 0.30,
    "item_dropout": 0.40,
    "exclusivity_weight": 0.0,
    "field_dropout": cfg.FIELD_DROPOUT,
    "fusion_mode": "gate",
    "fused_recon": True,
    "fused_recon_norm": "balanced",
    "fused_recon_weight": 0.5,
}

CONFIGS = {
    "full": {},
    "no-consistency": {"consistency_weight": 0.0},
    "no-cross": {"cross_reconstruction": False},
    "no-mask": {"cause_dropout": 0.0, "item_dropout": 0.0},
    "no-fielddrop": {"field_dropout": 0.0},
    # additional control: mean fusion + fused reconstruction, i.e. the same
    # extra objective as "full" but without the learnable gate
    "fused-mean": {"fusion_mode": "mean"},
    "no-fusion": {"fused_recon": False, "fusion_mode": "mean"},
    # NOTE: this cell of the 2x2 is *degenerate*, not a real control.  The gate
    # parameter ``fusion_logit`` (initialised to 0, i.e. sigmoid = 0.5) is only
    # ever used inside ``if self.fused_recon:`` in FCNTM.train_step, so without
    # the fused-reconstruction term it receives no gradient and the model is
    # exactly mean fusion.  Kept here only to document why the paper does not
    # report a "gate without fused reconstruction" row.
    "gate-no-fusion": {"fused_recon": False, "fusion_mode": "gate"},
    # fused-reconstruction weight sweep: mean fusion (isolates the weight) and
    # gate fusion (the paper's main model)
    "fused-w0p25": {"fusion_mode": "mean", "fused_recon_weight": 0.25},
    "fused-w1p0": {"fusion_mode": "mean", "fused_recon_weight": 1.0},
    "gate-w0p25": {"fusion_mode": "gate", "fused_recon_weight": 0.25},
    "gate-w1p0": {"fusion_mode": "gate", "fused_recon_weight": 1.0},
    # keeps the fused branch *enabled* but multiplies its loss by zero: the
    # control that separates "adds a fused-reconstruction objective" from
    # "runs one extra decoder forward pass" (BatchNorm/dropout state, RNG)
    "fused-w0": {"fusion_mode": "mean", "fused_recon_weight": 0.0},
}


def load_dataset(name, abstract_words: int = 0):
    if name == "agnews":
        import public_data as PD
        return PD.load_agnews(1500, 1900)
    if name == "civil":
        import public_data_civil as PC
        return PC.load_civil(6000, 2000, max_abstract_words=abstract_words)
    import public_data_github as PGH
    return PGH.load_github(1500, 500)


def config_hash(kwargs: dict) -> str:
    payload = json.dumps(kwargs, sort_keys=True).encode("utf-8")
    import hashlib
    return hashlib.sha1(payload).hexdigest()[:12]


def reusable_full_record(rec, want_fit, expected_full, k, n_vocab,
                         n_docs):
    """(beta, theta, why) for a details record that may serve as reuse.

    Every field of the fitting identity must match: a correct config hash
    plus matching matrix shapes is not enough (2026-10-11 review, item 2).
    """
    import numpy as np
    if rec.get("model") != "FC-NTM":
        return None, None, ["not an FC-NTM record"]
    beta = np.asarray(rec["beta"], dtype=float)
    theta = (np.asarray(rec["theta"], dtype=float)
             if rec.get("theta") is not None else None)
    why = []
    if rec.get("config_hash") != expected_full:
        why.append("configuration hash does not match the full configuration")
    fit = rec.get("fit_identity")
    if not fit:
        why.append("details carry no fit identity")
    else:
        payload = fit.get("payload", {})
        if payload.get("seed") != rec.get("seed"):
            why.append("record seed does not match the fit identity seed")
        for key, value in want_fit.items():
            if payload.get(key) != value:
                why.append(f"fit identity differs in {key!r}")
    if beta.ndim != 2 or beta.shape != (int(k), int(n_vocab)):
        why.append("beta shape is not (K, n_vocab)")
    if theta is None:
        why.append("details carry no theta for the cTFIDF read-out")
    elif theta.ndim != 2 or theta.shape != (int(n_docs), int(k)):
        why.append("theta shape is not (n_docs, K)")
    return beta, theta, why


def complete_pairs(rows, expected_hash, dataset, k, max_epochs, configs):
    """Keep only rows that this exact run could have produced.

    ``--resume`` used to skip on (config, seed) alone, so a file produced
    with another dataset, K, learning rate or epoch budget was reused
    (2026-10-07 review, item 2).  A row now has to name the same dataset,
    K and epoch budget and carry the config hash of the requested config;
    both read-outs must have finite metrics for the pair to be complete.
    """
    kept, have = [], {}
    for row in rows:
        try:
            current = (row["config"] in configs
                       and row["config"] in expected_hash
                       and row.get("dataset") == dataset
                       and int(row.get("K_used", -1)) == k
                       and int(row.get("epochs_budget", -1)) == max_epochs
                       and row.get("config_hash")
                       == expected_hash[row["config"]]
                       and math.isfinite(float(row["CV"]))
                       and math.isfinite(float(row["NPMI"])))
        except (KeyError, TypeError, ValueError):
            current = False
        if not current:
            continue
        kept.append(row)
        have.setdefault((row["config"], int(row["seed"])), set()).add(
            row.get("readout"))
    done = {key for key, readouts in have.items()
            if {"beta", "cTFIDF"} <= readouts}
    return kept, done


def train_model(train, K, seed, overrides, lr, max_epochs=cfg.PAPER_EPOCHS):
    cause_bow = torch.from_numpy(train["reason_bow"]).float()
    item_bow = torch.from_numpy(train["item_bow"]).float()
    cause_norm = NN._normalize_rows(cause_bow)
    item_norm = NN._normalize_rows(item_bow)
    has_cause = torch.tensor(
        [len(t) > 0 for t in train["reason_tokens"]], dtype=torch.bool)
    kwargs = dict(FULL_CONFIG)
    kwargs.update(overrides)
    NN.set_torch_seed(seed)
    model = NN.FCNTM(train["n_vocab"], K, **kwargs)

    def loss_fn(idx, kl_weight):
        return model.train_step(
            cause_norm[idx], item_norm[idx], cause_bow[idx], item_bow[idx],
            has_cause[idx], kl_weight)

    extra = {"recover_nan": True, "num_epochs": int(max_epochs)}
    if lr:
        extra["lr"] = float(lr)
    NN.train_topic_vae_valid(
        model, loss_fn, loss_fn, train["n_docs"], seed,
        val_fraction=0.2, kl_max=0.05, **extra)
    model.eval()
    with torch.no_grad():
        theta, _ = model.encode_doc_theta(cause_norm, item_norm, has_cause)
    return np.asarray(model.get_beta(), dtype=float), theta.numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["agnews", "github", "civil"],
                    default="github")
    ap.add_argument("--seeds", default="11,23,37")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--configs",
                    default="no-consistency,no-cross,no-mask,no-fielddrop")
    ap.add_argument("--reuse-details", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--dump-beta", default="")
    ap.add_argument("--max-epochs", type=int, default=cfg.PAPER_EPOCHS)
    ap.add_argument("--abstract-words", type=int, default=0,
                    help="civil corpus only: truncate each abstract to the "
                         "first N words (builds the field-ratio ladder)")
    ap.add_argument("--resume", action="store_true",
                    help="keep the rows already in --out and skip the "
                         "(config, seed) pairs that are already complete")
    args = ap.parse_args()

    data = load_dataset(args.dataset, args.abstract_words)
    train, test = data["train"], data["test"]
    del data
    gc.collect()
    vocab = test["vocab"]
    doc_counts = (train["reason_bow"] + train["item_bow"]).astype(float)

    seeds = [int(s) for s in args.seeds.split(",")]
    configs = [c for c in args.configs.split(",") if c]
    expected_hash = {name: config_hash({**FULL_CONFIG, **CONFIGS[name]})
                     for name in configs}
    ident = RI.identity({
        "script": "run_ablation_public.main",
        "dataset": args.dataset,
        "abstract_words": int(args.abstract_words),
        "k": int(args.k),
        "lr": float(args.lr),
        "max_epochs": int(args.max_epochs),
        "seeds": [int(s) for s in seeds],
        "configs": sorted(configs),
        "full_config": FULL_CONFIG,
        "code": RI.code_version(),
        "data": RI.text_digest([train.get("doc_tokens"),
                                 test.get("doc_tokens")]),
        "fields": RI.text_digest([train.get("reason_tokens"),
                                  train.get("item_tokens"),
                                  test.get("reason_tokens"),
                                  test.get("item_tokens")]),
    })

    reused = {}
    if args.reuse_details:
        p = Path(args.reuse_details)
        expected_full = expected_hash.get("full", config_hash(dict(FULL_CONFIG)))
        if p.exists() and p.stat().st_size:
            refused = []
            want_fit = {
                "k": int(args.k),
                "n_vocab": int(train["n_vocab"]),
                "n_docs": int(train["n_docs"]),
                "vocab_digest": __import__("hashlib").sha1(
                    "\x1f".join(train["vocab"]).encode("utf-8")
                ).hexdigest()[:16],
                "data": RI.text_digest([train.get("doc_tokens")]),
                "fields": RI.text_digest([train.get("reason_tokens"),
                                          train.get("item_tokens")]),
                "code": RI.code_version(),
                "field_dropout": float(FULL_CONFIG["field_dropout"]),
                "fusion": "gate",
                "fusion_norm": FULL_CONFIG["fused_recon_norm"],
                "fusion_weight": float(FULL_CONFIG["fused_recon_weight"]),
                "lr": float(args.lr),
                "epochs_budget": int(args.max_epochs),
            }
            for rec in json.loads(p.read_text(encoding="utf-8")):
                if rec.get("model") != "FC-NTM":
                    continue
                seed = int(rec["seed"])
                beta, theta, why = reusable_full_record(
                    rec, want_fit, expected_full, args.k,
                    train["n_vocab"], train["n_docs"])
                if why:
                    refused.append((seed, "; ".join(why)))
                    continue
                reused[seed] = (beta, theta, rec.get("config_hash"))
            if reused:
                print("reused full-config beta for seeds", sorted(reused),
                      flush=True)
            for seed, why in refused:
                print(f"  refusing reused details for seed {seed}: {why}",
                      flush=True)
            if refused:
                print("  unverified full-config details are not used as "
                      "the reference row; those seeds are trained",
                      flush=True)

    # every other entry point resolves a relative ``--out`` against the outputs
    # directory (benchmark_public, run_public_missing_field, run_fcntm_train);
    # this one used to resolve it against the *current* directory, so the same
    # command line wrote to two different places depending on who called it.
    out_path = Path(args.out or f"ablation_public_{args.dataset}.csv")
    if not out_path.is_absolute():
        out_path = OUT_DIR / out_path
    fieldnames = ["dataset", "config", "seed", "K_used", "readout",
                  "config_hash", "reuse_verified", "identity_id",
                  "epochs_budget", "seconds",
                  "CV", "CV_n", "NPMI", "NPMI_n", "UCI", "UCI_n",
                  "UMass", "UMass_n", "EXCL", "EXCL_KADJ", "IRBO"]
    # ``--resume``: a 700-epoch fit costs about an hour, so a crash in the
    # middle of the chain must not throw away everything already scored.
    kept_rows = []
    done = set()
    if args.resume and out_path.exists() and out_path.stat().st_size:
        if not RI.matches(out_path, ident):
            print(f"  stored rows in {out_path.name} are not pinned to "
                  f"this configuration ({RI.why_not(out_path, ident)}); "
                  f"they cannot be skipped or reused", flush=True)
            stash = RI.stash_unpinned(out_path)
            if stash:
                print(f"  old rows moved to {stash}", flush=True)
        else:
            with open(out_path, "r", encoding="utf-8-sig",
                      newline="") as fh:
                kept_raw = [r for r in csv.DictReader(fh)
                            if r.get("config") and r.get("seed")]
            kept_rows, done = complete_pairs(
                kept_raw, expected_hash, args.dataset, args.k,
                args.max_epochs, configs)
            print(f"resume: {len(kept_rows)} of {len(kept_raw)} row(s) "
                  f"verified for this configuration, {len(done)} complete "
                  f"(config, seed) pair(s)", flush=True)
    beta_dump = []
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in kept_rows:
            writer.writerow(row)
        fh.flush()
        # pin the file (even a partial one) as soon as this run owns it, so
        # an interrupted run resumes against the same identity
        RI.write(out_path, ident)
        # NB: the loop variable must not shadow the imported ``config`` module
        # (``for config in configs`` made ``config.ROOT`` below a string
        # attribute error on any relative --dump-beta path).
        for config_name in configs:
            for seed in seeds:
                if (config_name, seed) in done:
                    print(f"   {config_name} seed {seed} already complete; "
                          f"skipped", flush=True)
                    continue
                t0 = time.time()
                if config_name == "full":
                    if seed not in reused:
                        print(f"  no reused beta for seed {seed}; training", flush=True)
                        beta, theta = train_model(
                            train, args.k, seed, CONFIGS[config_name], args.lr,
                            args.max_epochs)
                        stored_hash, reuse_verified = None, "trained"
                    else:
                        beta, theta, stored_hash = reused[seed]
                        reuse_verified = bool(stored_hash)
                else:
                    beta, theta = train_model(
                        train, args.k, seed, CONFIGS[config_name], args.lr,
                        args.max_epochs)
                    stored_hash, reuse_verified = None, "trained"
                run_cfg = dict(FULL_CONFIG)
                run_cfg.update(CONFIGS[config_name])
                # a reused model keeps the hash of the configuration it was
                # actually trained with; never relabel it with the current one
                row_hash = stored_hash or config_hash(run_cfg)
                seconds = time.time() - t0
                mats = [("beta", beta)]
                if theta is not None:
                    mats.append(("cTFIDF", class_tfidf(theta.T @ doc_counts)))
                for readout, mat in mats:
                    ev = evaluate(top_words(mat, vocab), mat, test)
                    writer.writerow({
                        "dataset": args.dataset, "config": config_name,
                        "seed": seed,
                        "K_used": int(beta.shape[0]), "readout": readout,
                        "config_hash": row_hash,
                        "reuse_verified": reuse_verified,
                        "identity_id": ident["id"],
                        "epochs_budget": int(args.max_epochs),
                        "seconds": round(seconds, 2),
                        **{k: round(float(v), 6) for k, v in ev.items()},
                    })
                    fh.flush()
                    print("  ", config_name, "seed", seed, readout,
                          {k: round(float(ev[k]), 4) for k in ("CV", "NPMI")},
                          flush=True)
                beta_dump.append({
                    "model": f"FC-NTM[{config_name}]", "config": config_name,
                    "seed": int(seed), "K": int(beta.shape[0]),
                    "beta": beta.tolist(),
                    "config_hash": row_hash,
                    "reuse_verified": reuse_verified,
                })
    if args.dump_beta and beta_dump:
        dp = Path(args.dump_beta)
        if not dp.is_absolute():
            dp = cfg.ROOT / dp
        dp.write_text(json.dumps(beta_dump), encoding="utf-8")
        print("saved", dp)
    print("saved", out_path)


if __name__ == "__main__":
    main()
