# -*- coding: utf-8 -*-
"""Sensitivity of the beta read-out (2026-10-07 review, item 11).

FC-NTM's document distribution is softmax(BN(theta @ B)) while the paper's beta
read-out ranks the rows of the raw parameter B.  For every corpus and seed this
script compares

  raw       top-10 words of each row of B
  bn        softmax(BN(e_k @ B)) -- the unit-topic generative distribution
  shift     B + c with the BN running mean also shifted by c: the generative
            distribution is unchanged, but the raw ordering may not be

and writes outputs/bn_readout_sensitivity.csv.

    python bn_readout_sensitivity.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import metrics as M
import models as NN
import occlusion_robustness as O
from run_fcntm_train import evaluate, top_words


def load_state(state_path, corpus, seed, K, n_vocab):
    ck = torch.load(state_path, map_location="cpu", weights_only=False)
    key = f"fielddrop_on:{seed}"
    if key not in ck:
        raise KeyError(f"{state_path} has no {key}")
    rec = ck[key]
    kwargs = rec.get("kwargs") or O.final_model_kwargs(0.5, "gate",
                                                       "balanced", 0.5)
    model = NN.FCNTM(n_vocab, K, **kwargs)
    model.load_state_dict(rec["model_state"])
    model.eval()
    return model


def main() -> int:
    rows = []
    for corpus in ("github", "agnews", "civil"):
        state = O.OUT_DIR / f"{corpus}_missing_state.pt"
        train, test = O.load_corpus(corpus)
        vocab = test["vocab"]
        n_vocab = len(vocab)
        for seed in (11, 23, 37):
            model = load_state(state, corpus, seed, 30, n_vocab)
            beta = np.asarray(model.get_beta(), dtype=float)
            with torch.no_grad():
                theta = torch.eye(beta.shape[0])
                logits = model.decode(theta)
                bn_prob = torch.softmax(logits, dim=-1).numpy()
            topics_raw = top_words(beta, vocab, 10)
            topics_bn = top_words(bn_prob, vocab, 10)
            raw_scores = evaluate(topics_raw, beta, test)
            bn_scores = evaluate(topics_bn, bn_prob, test)
            jac = [len(set(a) & set(b)) / len(set(a) | set(b))
                   for a, b in zip(topics_raw, topics_bn)]
            top1 = float(np.mean([a[0] != b[0]
                                  for a, b in zip(topics_raw, topics_bn)]))
            # per-word shift invariance: adding the same vector c_w to every
            # topic row and to the BN running mean leaves the generative
            # distribution unchanged, but can reorder the raw parameter rows
            rng = np.random.RandomState(seed)
            c = rng.uniform(-1.0, 1.0, size=n_vocab).astype(np.float32)
            with torch.no_grad():
                shifted = torch.from_numpy(beta + c[None, :]).float()
                bn = model.decoder.beta_bn
                logits2 = (theta @ shifted
                           - (bn.running_mean + torch.from_numpy(c))) / torch.sqrt(
                               bn.running_var + bn.eps)
                bn_prob2 = torch.softmax(logits2, dim=-1).numpy()
            shift_topics = top_words(beta + c[None, :], vocab, 10)
            shift_changed = float(np.mean([
                a[:10] != b[:10] for a, b in zip(topics_raw, shift_topics)]))
            rows.append({
                "corpus": corpus, "seed": seed,
                "raw_CV": raw_scores["CV"], "raw_NPMI": raw_scores["NPMI"],
                "bn_CV": bn_scores["CV"], "bn_NPMI": bn_scores["NPMI"],
                "cv_diff": bn_scores["CV"] - raw_scores["CV"],
                "npmi_diff": bn_scores["NPMI"] - raw_scores["NPMI"],
                "mean_top10_jaccard": float(np.mean(jac)),
                "top1_change_rate": top1,
                "shift_max_prob_diff": float(np.max(np.abs(bn_prob - bn_prob2))),
                "shift_raw_topic_change_rate": shift_changed,
            })
            print(f"[{corpus} seed {seed}] raw CV {raw_scores['CV']:.4f} / "
                  f"BN CV {bn_scores['CV']:.4f} | jaccard "
                  f"{np.mean(jac):.3f} | shift prob diff "
                  f"{np.max(np.abs(bn_prob - bn_prob2)):.2e} | raw topics "
                  f"changed by shift {shift_changed:.2f}", flush=True)
            del model
    frame = pd.DataFrame(rows)
    out = O.OUT_DIR / "bn_readout_sensitivity.csv"
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    print("\nsaved", out)
    summary = frame.groupby("corpus").agg(
        raw_CV=("raw_CV", "mean"), bn_CV=("bn_CV", "mean"),
        raw_NPMI=("raw_NPMI", "mean"), bn_NPMI=("bn_NPMI", "mean"),
        jaccard=("mean_top10_jaccard", "mean"),
        top1_change=("top1_change_rate", "mean"),
        shift_prob=("shift_max_prob_diff", "max"),
        shift_topic_change=("shift_raw_topic_change_rate", "mean"))
    print(summary.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())