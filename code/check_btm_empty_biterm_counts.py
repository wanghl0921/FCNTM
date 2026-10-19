# -*- coding: utf-8 -*-
"""How many scored records have no biterm in the visible text?

These are the records the BTM adapter used to score as an all-zero topic row
(NLL = log 1e12 = 27.63).  The count needs no model: a biterm only depends on
the visible tokens and the dictionary.  Reported per corpus, seed and masking
rate so the paper's disclosure can be checked.

    python check_btm_empty_biterm_counts.py

Outputs
    outputs/rerun/btm_empty_biterm_counts.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import baselines as BL
import occlusion_robustness as O

RATES = [0.25, 0.5, 0.75, 1.0]


def main() -> int:
    rows = []
    for corpus in ("github", "agnews"):
        train, test = O.load_corpus(corpus)
        dictionary = test["dictionary"]
        for seed in (11, 23, 37):
            rng = np.random.RandomState(seed + 7)
            seeds = rng.randint(0, 10 ** 6, size=len(test["reason_tokens"]))
            for rate in RATES:
                no_biterm = scored = 0
                for i, toks in enumerate(test["reason_tokens"]):
                    visible, hidden = O.mask_tokens(
                        toks, rate, np.random.RandomState(seeds[i]))
                    if not hidden:
                        continue            # nothing to score for this record
                    scored += 1
                    doc = visible + test["item_tokens"][i]
                    pairs, _ = BL._biterms_from_tokens(
                        [doc], dictionary, 12, max_pairs=150, seed=seed)
                    if not pairs:
                        no_biterm += 1
                rows.append({"corpus": corpus, "seed": seed, "rate": rate,
                             "scored_records": scored,
                             "no_biterm_records": no_biterm,
                             "share": no_biterm / scored if scored else 0.0})
    frame = pd.DataFrame(rows)
    out = O.OUT_DIR / "rerun" / "btm_empty_biterm_counts.csv"
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
