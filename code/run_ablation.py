# -*- coding: utf-8 -*-
"""Component ablations for FC-NTM (paper extension).

Each configuration toggles one mechanism of the field-collaborative model:

- full                : cross-field reconstruction + JS consistency + field masking
- no-cross            : each stream reconstructs only its own field
- no-consistency      : drop the symmetric JS consistency term
- no-mask             : drop field-level word masking (cause/item dropout)
- force-encode-missing: encode the empty cause field instead of falling back
                         to the prior when the reason field is absent

Example
-------
python run_ablation.py --seeds 11,23,37,53,71 --k 30 --decoder embedding
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

import config
import metrics as M
import models as NN
from run_fcntm_train import class_tfidf, evaluate, top_words

torch.set_num_threads(8)

CONFIGS = {
    "full": {},
    "no-cross": {"cross_reconstruction": False},
    "no-consistency": {"consistency_weight": 0.0},
    "no-mask": {"cause_dropout": 0.0, "item_dropout": 0.0},
    "force-encode-missing": {"force_encode_missing": True},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["private", "github"], default="private")
    parser.add_argument("--seeds", default="11,23,37,53,71")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--decoder", choices=["linear", "embedding"], default="embedding")
    parser.add_argument("--configs", default="full,no-cross,no-consistency,no-mask")
    parser.add_argument("--out", default="")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    if args.corpus == "private":
        import corpus as C

        train = test = C.load_corpus()
        train["name"] = "private-change-order"
    else:
        import public_data_github as PGH

        data = PGH.load_github(1500, 500)
        train, test = data["train"], data["test"]
        train["name"] = "GitHubIssues-4repo"

    seeds = [int(x) for x in args.seeds.split(",") if x]
    names = [x for x in args.configs.split(",") if x]
    vocab = test["vocab"]
    cause_bow = torch.from_numpy(train["reason_bow"]).float()
    item_bow = torch.from_numpy(train["item_bow"]).float()
    cause_norm = NN._normalize_rows(cause_bow)
    item_norm = NN._normalize_rows(item_bow)
    has_cause = torch.tensor(
        [len(t) > 0 for t in train["reason_tokens"]], dtype=torch.bool)
    doc_counts = (cause_bow + item_bow).numpy()

    word_embeddings = None
    if args.decoder == "embedding":
        if str(train.get("name", "")).startswith("GitHub"):
            from embed_public import embed_words
        else:
            from embeddings import embed_words

        word_embeddings = embed_words(vocab)
        print(f"embeddings: {word_embeddings.shape}", flush=True)

    out_path = args.out or os.path.join(
        config.ROOT, "outputs",
        f"ablation_{args.corpus}_{args.decoder}_k{args.k}.csv")
    fieldnames = ["corpus", "config", "seed", "K_used", "decoder", "readout",
                  "best_epoch", "seconds", "CV", "NPMI", "UCI", "UMass",
                  "EXCL_KADJ", "IRBO"]
    write_header = not args.append or not os.path.exists(out_path) \
        or os.path.getsize(out_path) == 0
    details = []
    with open(out_path, "a" if args.append else "w", encoding="utf-8-sig",
              newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for seed in seeds:
            for name in names:
                kwargs = dict(
                    consistency_weight=0.5,
                    cross_reconstruction=True,
                    cause_dropout=0.30,
                    item_dropout=0.40,
                )
                kwargs.update(CONFIGS[name])
                NN.set_torch_seed(seed)
                model = NN.FCNTM(
                    train["n_vocab"], args.k,
                    word_embeddings=word_embeddings,
                    decoder_type=args.decoder,
                    **kwargs,
                )

                def loss_fn(idx, kl_weight):
                    return model.train_step(
                        cause_norm[idx], item_norm[idx],
                        cause_bow[idx], item_bow[idx], has_cause[idx],
                        kl_weight)

                t0 = time.time()
                info = NN.train_topic_vae_valid(
                    model, loss_fn, loss_fn, train["n_docs"], seed,
                    val_fraction=0.2, kl_max=0.05)
                secs = time.time() - t0
                model.eval()
                with torch.no_grad():
                    theta, _ = model.encode_doc_theta(
                        cause_norm, item_norm, has_cause)
                theta = theta.numpy()
                beta = np.asarray(model.get_beta(), dtype=float)
                readouts = {"beta": (top_words(beta, vocab), beta)}
                soft = class_tfidf(theta.T @ doc_counts)
                readouts["cTFIDF"] = (top_words(soft, vocab), soft)
                for readout, (topics, score_mat) in readouts.items():
                    ev = evaluate(topics, score_mat, test)
                    writer.writerow({
                        "corpus": train.get("name", "corpus"),
                        "config": name, "seed": seed, "K_used": args.k,
                        "decoder": args.decoder, "readout": readout,
                        "best_epoch": info["best_epoch"],
                        "seconds": round(secs, 2),
                        **{k: round(float(v), 6) for k, v in ev.items()},
                    })
                    f.flush()
                    print(f"  {name} seed={seed} {readout:6s} "
                          f"CV={ev['CV']:.3f} NPMI={ev['NPMI']:+.3f} "
                          f"UMass={ev['UMass']:.2f} "
                          f"EXCLk={ev['EXCL_KADJ']:.3f}", flush=True)
                details.append({
                    "model": name, "seed": seed, "K": args.k,
                    "biterm_weight": 0.0, "warmstart": "none",
                    "beta": beta.tolist(), "theta": theta.tolist(),
                    "has_cause": has_cause.numpy().tolist(),
                })
    print("saved", out_path)
    detail_path = os.path.join(
        config.ROOT, "outputs",
        f"ablation_{args.corpus}_{args.decoder}_k{args.k}_details.json")
    with open(detail_path, "w", encoding="utf-8") as fh:
        json.dump(details, fh)
    print("saved", detail_path)


if __name__ == "__main__":
    main()
