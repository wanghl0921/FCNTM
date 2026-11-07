# -*- coding: utf-8 -*-
"""Fast FC-NTM vs BTM abstract-occlusion curve (10%-100%) using saved fits.

Uses FC-NTM checkpoints and saved BTM fits for seeds 53/71/89, so no model is
refitted.  This is a pilot for the 10%-step masking curve.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\20102\Desktop\FCNTM\FCNTM完整代码")
sys.path.insert(0, str(ROOT))
import occlusion_robustness as O
import baselines as BL

RATES = [0.0] + [round(0.1 * i, 1) for i in range(1, 11)]
SEEDS = [53, 71, 89]
OUT = ROOT / "outputs" / "rerun" / "fcntm_btm_civil_10pct.csv"


def make_btm(seed: int, dictionary, vocab):
    z = np.load(ROOT / "outputs" / "rerun" / f"btm_fit_civil_seed{seed}.npz")
    b = object.__new__(O.BTMAdapter)
    b.beta = z["beta"]
    b.n_bz = z["n_bz"]
    b.n_wz = z["n_wz"]
    b.alpha = float(z["alpha"])
    b.beta_prior = float(z["beta_prior"])
    b.window = int(z["window"])
    b.max_pairs = int(z["max_pairs"])
    b.seed = seed
    b.K = int(b.n_bz.shape[0])
    b.V = len(vocab)
    b.dictionary = dictionary
    return b


def main():
    train, test = O.load_corpus("civil")
    vocab = test["vocab"]
    word2id = test["dictionary"].token2id
    rows = []
    for seed in SEEDS:
        print("seed", seed, flush=True)
        rng = np.random.RandomState(seed + 7)
        seeds = rng.randint(0, 10 ** 6, size=len(test["reason_tokens"]))
        visible, hidden = {}, {}
        for rate in RATES:
            vis, hid = [], []
            for i, toks in enumerate(test["reason_tokens"]):
                v, h = O.mask_tokens(toks, rate, np.random.RandomState(seeds[i]))
                vis.append(v)
                hid.append(h)
            visible[rate] = vis
            hidden[rate] = hid
        fc = O.FCNTMAdapter(ROOT / "outputs" / "rerun" / "fcntm_main_state_civil.pt",
                            len(vocab), 30, seed=seed)
        for name, adapter in [("FC-NTM", fc), ("BTM", make_btm(seed, test["dictionary"], vocab))]:
            got = O.run_model(adapter, visible, test["item_tokens"], hidden,
                              RATES, word2id, name)
            for r in got:
                r["seed"] = seed
                rows.append(r)
            del adapter
        del fc
    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(frame.groupby(["model", "rate"])[["agreement", "mean_js"]].mean().round(4))
    print("saved", OUT)


if __name__ == "__main__":
    main()
