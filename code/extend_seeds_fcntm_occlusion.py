# -*- coding: utf-8 -*-
"""Score the extra FC-NTM main seeds with the partial-occlusion protocol.

The checkpoint is verified against the corpus identity before scoring and the
merged CSV is pinned to the shared evaluation identity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import occlusion_robustness as O
from add_btm_doc_topic import masks_for
from seed_identity import (merge_pinned, occlusion_identity, state_path,
                           verify_checkpoint)

RATES = [0.0, 0.25, 0.5, 0.75, 1.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="github,agnews")
    ap.add_argument("--seeds", default="53,71,89,101,113,127,139")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--state-dir", default="")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    for corpus in args.corpora.split(","):
        train, test = O.load_corpus(corpus)
        word2id = test["dictionary"].token2id
        state = Path(args.state_dir) / f"fcntm_main_state_{corpus}.pt" \
            if args.state_dir else state_path(corpus)
        if not state.exists():
            raise SystemExit(f"no FC-NTM state for {corpus}: {state}")
        verify_checkpoint(state, train, seeds, args.k)
        rows = []
        for seed in seeds:
            adapter = O.FCNTMAdapter(state, len(test["vocab"]), args.k,
                                     seed=seed)
            visible, hidden = masks_for(test, seed)
            scored = O.run_model(adapter, visible, test["item_tokens"],
                                 hidden, RATES, word2id, "FC-NTM")
            for row in scored:
                row["corpus"] = corpus
                row["seed"] = seed
            rows.extend(scored)
        merge_pinned(
            O.OUT_DIR / "rerun" / f"occlusion_robustness_{corpus}_newseeds.csv",
            rows, occlusion_identity(corpus, train, test, seeds, args.k,
                                     checkpoint=state),
            ["model", "seed", "rate"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())