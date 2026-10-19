# -*- coding: utf-8 -*-
"""Fit LDA for the extra seeds and add its occlusion curve.

AG News' 3-seed best baseline is LDA, so the 10-seed "advantage over the best
same-seed baseline" needs LDA curves for the new seeds.  Rows are merged into
the shared pinned occlusion file.
"""

from __future__ import annotations

import argparse

import occlusion_robustness as O
from add_btm_doc_topic import masks_for
from seed_identity import merge_pinned, occlusion_identity

RATES = [0.0, 0.25, 0.5, 0.75, 1.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="github,agnews")
    ap.add_argument("--seeds", default="53,71,89,101,113,127,139")
    ap.add_argument("--k", type=int, default=30)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    for corpus in args.corpora.split(","):
        train, test = O.load_corpus(corpus)
        word2id = test["dictionary"].token2id
        rows = []
        for seed in seeds:
            adapter = O.LDAAdapter(train, args.k, seed)
            adapter.dictionary = test["dictionary"]
            visible, hidden = masks_for(test, seed)
            scored = O.run_model(adapter, visible, test["item_tokens"],
                                 hidden, RATES, word2id, "LDA")
            for row in scored:
                row["corpus"] = corpus
                row["seed"] = seed
            rows.extend(scored)
            del adapter
        merge_pinned(
            O.OUT_DIR / "rerun" / f"occlusion_robustness_{corpus}_newseeds.csv",
            rows, occlusion_identity(corpus, train, test, seeds, args.k),
            ["model", "seed", "rate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())