# -*- coding: utf-8 -*-
"""What does the model actually do when the *summary* field is empty?

The paper used to say that the topic proportions are then "inferred from the
background counts alone".  The implementation always runs the summary encoder
with an all-True presence mask, so a zero-input summary still contributes a
posterior (determined by the trained parameters) that enters the gate fusion.
This script measures that difference on the seed-11 checkpoints so the paper
can state the behaviour instead of assuming a symmetric fallback.

    python check_empty_summary_behavior.py

Outputs
    outputs/rerun/empty_summary_behavior.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import models as NN
import occlusion_robustness as O
from config import TOPN


def main() -> int:
    rows = []
    for corpus in ("github", "agnews"):
        train, test = O.load_corpus(corpus)
        word2id = test["dictionary"].token2id
        n_vocab = len(test["vocab"])
        state = (O.OUT_DIR / f"{corpus}_missing_state.pt")
        adapter = O.FCNTMAdapter(state, n_vocab, 30, seed=11)
        adapter.set_word2id(word2id)

        idx = [i for i, t in enumerate(test["item_tokens"]) if not t]
        cause = [test["reason_tokens"][i] for i in idx]
        item = [[] for _ in idx]
        A = torch.from_numpy(O._bow(cause, n_vocab, word2id))
        B = torch.from_numpy(O._bow(item, n_vocab, word2id))
        has_cause = torch.tensor([len(t) > 0 for t in cause], dtype=torch.bool)
        with torch.no_grad():
            fused, parts = adapter.model.encode_doc_theta(
                NN._normalize_rows(A), NN._normalize_rows(B), has_cause)
        fused = fused.numpy()
        background_only = parts["cause"].numpy()
        diff = np.abs(fused - background_only).max(axis=1)
        changed = int(np.sum(fused.argmax(axis=1) != background_only.argmax(axis=1)))
        rows.append({
            "corpus": corpus, "seed": 11, "n_empty_summary_test": len(idx),
            "max_abs_diff_vs_background_only": float(diff.max()) if len(idx) else 0.0,
            "mean_abs_diff_vs_background_only": float(diff.mean()) if len(idx) else 0.0,
            "argmax_changed": changed,
        })
        print(f"{corpus}: {len(idx)} empty-summary test records, "
              f"max|fused - background-only| = {rows[-1]['max_abs_diff_vs_background_only']:.6f}, "
              f"argmax changed {changed}", flush=True)
    out = O.OUT_DIR / "rerun" / "empty_summary_behavior.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
