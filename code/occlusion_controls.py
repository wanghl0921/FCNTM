# -*- coding: utf-8 -*-
"""Controls requested by the 2026-10-07 review (item 6).

Three cheap controls, all scored with the *same* masking protocol as
``occlusion_robustness.py`` (same seeds, same hidden targets, same
per-document token-average NLL):

  unigram      constant training-set unigram, reads no test document at all;
               if its relative degradation is ~1 it proves the ratio alone
               cannot distinguish "uses the other field" from "ignores input"
  shuffled     FC-NTM sees another document's title while predicting this
               document's hidden background tokens (a derangement)
  fielddrop-off  the partial-occlusion curve of the model trained *without*
               field dropout, for comparison with the main (fielddrop_on)
               curve

    python occlusion_controls.py --control unigram
    python occlusion_controls.py --control shuffled
    python occlusion_controls.py --control fielddrop-off
"""

from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd

import occlusion_robustness as O
from add_civil_btm_occlusion import masks_for

RATES = [0.0, 0.25, 0.5, 0.75, 1.0]


class ConstantUnigramAdapter(O.Adapter):
    """One fixed word distribution for every document; ignores the input."""

    name = "Constant-unigram"

    def __init__(self, train, smoothing: float = 0.01):
        word2id = train["dictionary"].token2id
        counts = np.zeros(len(train["vocab"]), dtype=float)
        for ts in train["doc_tokens"]:
            for w in ts:
                j = word2id.get(w)
                if j is not None:
                    counts[j] += 1.0
        p = counts + float(smoothing)
        self.logp = np.log(p / p.sum())

    def set_word2id(self, word2id):
        self.word2id = word2id

    def theta(self, visible_A, field_B):
        return np.ones((len(visible_A), 1), dtype=float)

    def log_prob(self, theta):
        return np.tile(self.logp[None, :], (theta.shape[0], 1))


class FCNTMKeyAdapter(O.FCNTMAdapter):
    """FC-NTM loaded from a named key (e.g. ``fielddrop_off:23``)."""

    def __init__(self, state_path, vocab_size, K, seed, key_prefix):
        import torch

        import models as NN
        ck = torch.load(state_path, map_location="cpu", weights_only=False)
        key = f"{key_prefix}:{seed}"
        if key not in ck:
            raise KeyError(f"{state_path} has no {key} "
                           f"(available: {sorted(ck)})")
        rec = ck[key]
        kwargs = rec.get("kwargs") or O.final_model_kwargs(0.5, "gate",
                                                           "balanced", 0.5)
        self.model = NN.FCNTM(vocab_size, K, **kwargs)
        self.model.load_state_dict(rec["model_state"])
        self.model.eval()
        self.key = key


def derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed + 99)
    identity = np.arange(n)
    perm = rng.permutation(n)
    for _ in range(10000):
        if not np.any(perm == identity):
            return perm
        perm = rng.permutation(n)
    return rng.permutation(n)


def state_path(corpus: str, control: str):
    if control == "fielddrop-off":
        if corpus == "civil":
            return O.OUT_DIR / "civil_missing_state.pt"
        return O.OUT_DIR / f"{corpus}_fielddrop_off.pt"
    return O.OUT_DIR / f"{corpus}_missing_state.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True,
                    choices=["unigram", "shuffled", "fielddrop-off"])
    ap.add_argument("--corpora", default="github,agnews,civil")
    ap.add_argument("--seeds", default="11,23,37")
    ap.add_argument("--k", type=int, default=30)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    for corpus in args.corpora.split(","):
        train, test = O.load_corpus(corpus)
        word2id = test["dictionary"].token2id
        frames = []
        for seed in seeds:
            visible, hidden = masks_for(test, RATES, seed)
            field_B = list(test["item_tokens"])
            if args.control == "unigram":
                adapter = ConstantUnigramAdapter(train)
                label = "Constant-unigram"
            elif args.control == "shuffled":
                adapter = O.FCNTMAdapter(state_path(corpus, args.control),
                                         len(test["vocab"]), args.k, seed=seed)
                perm = derangement(len(field_B), seed)
                field_B = [test["item_tokens"][i] for i in perm]
                label = "FC-NTM-shuffled-item"
            else:
                adapter = FCNTMKeyAdapter(state_path(corpus, args.control),
                                          len(test["vocab"]), args.k, seed,
                                          "fielddrop_off")
                label = "FC-NTM-fielddrop_off"
            print(f"[{corpus} seed {seed}] {label}", flush=True)
            rows = O.run_model(adapter, visible, field_B, hidden, RATES,
                               word2id, label)
            for row in rows:
                row["corpus"] = corpus
                row["seed"] = seed
            frames.append(pd.DataFrame(rows))
            del adapter
            gc.collect()
        frame = pd.concat(frames, ignore_index=True)
        out = O.OUT_DIR / f"occlusion_control_{args.control.replace('-', '_')}_{corpus}.csv"
        frame.to_csv(out, index=False, encoding="utf-8-sig")
        print("saved", out, flush=True)
        ratios = []
        for seed, g in frame[frame["rate"] > 0].groupby("seed"):
            g = g.set_index("rate").sort_index()
            ratios.append(float(g.loc[1.0, "hidden_NLL"]
                                / g.loc[0.25, "hidden_NLL"]))
        ratios = np.asarray(ratios)
        print(f"  {corpus}: relative degradation {ratios.mean():.6f} "
              f"+/- {ratios.std(ddof=1):.6f} over {ratios.size} seeds",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())