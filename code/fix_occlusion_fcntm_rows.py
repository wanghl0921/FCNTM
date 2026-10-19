# -*- coding: utf-8 -*-
"""Recompute the FC-NTM rows of the partial-occlusion curves with the right model.

The stored ``occlusion_robustness_<corpus>_seed23.csv`` curves were produced
with the *seed-11* checkpoint (bit-identical at rate 1.0, where the mask plays
no role, and a seed-11 replay with the seed-23 masks reproduces every point).
The baseline rows in those files do not depend on the FC-NTM state, so only the
``FC-NTM`` rows are recomputed here and merged back in place; nothing is
retrained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import occlusion_robustness as O

RATES = [0.0, 0.25, 0.5, 0.75, 1.0]
OUT = O.OUT_DIR


def masks_for(test, seed: int):
    rng = np.random.RandomState(seed + 7)
    seeds = rng.randint(0, 10 ** 6, size=len(test["reason_tokens"]))
    visible, hidden = {}, {}
    for rate in RATES:
        vis, hid = [], []
        for i, toks in enumerate(test["reason_tokens"]):
            v, h = O.mask_tokens(toks, rate, np.random.RandomState(seeds[i]))
            vis.append(v)
            hid.append(h)
        visible[rate], hidden[rate] = vis, hid
    return visible, hidden


def main() -> int:
    for corpus in ("github", "agnews"):
        train, test = O.load_corpus(corpus)
        vocab = test["vocab"]
        word2id = test["dictionary"].token2id
        for seed in (11, 23, 37):
            visible, hidden = masks_for(test, seed)
            state = str(OUT / f"{corpus}_missing_state.pt")
            adapter = O.FCNTMAdapter(state, len(vocab), 30, seed=seed)
            adapter.set_word2id(word2id)
            rows = O.run_model(adapter, visible, test["item_tokens"], hidden,
                               RATES, word2id, "FC-NTM")
            path = OUT / f"occlusion_robustness_{corpus}_seed{seed}.csv"
            prev = pd.read_csv(path)
            keep = prev[prev["model"] != "FC-NTM"]
            merged = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
            merged.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  wrote {path.name}: FC-NTM recomputed, "
                  f"{len(keep)} baseline rows kept", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
