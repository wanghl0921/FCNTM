# -*- coding: utf-8 -*-
"""Same-architecture title-only FC-NTM baseline for the civil corpus.

The model is trained on titles only: every training record has an empty
background field (reason_bow=0, reason_tokens=[]), so the summary stream is the
only active stream at training time.  At test time the same empty-background
condition is used.  Everything else follows the paper's main configuration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\20102\Desktop\FCNTM\FCNTM完整代码")
sys.path.insert(0, str(ROOT))
import public_data_civil as PC
from run_fcntm_train import run_corpus


def as_title_only(corpus: dict, dictionary) -> dict:
    d = dict(corpus)
    n = len(corpus["item_tokens"])
    d["reason_tokens"] = [[] for _ in range(n)]
    if "reason_bow" in corpus:
        d["reason_bow"] = np.zeros_like(corpus["reason_bow"])
    d["doc_tokens"] = [list(t) for t in corpus["item_tokens"]]
    d["doc_texts"] = [" ".join(t) for t in d["doc_tokens"]]
    d["corpus"] = [dictionary.doc2bow(t) for t in d["doc_tokens"]]
    if "doc_bow" in corpus and corpus.get("doc_bow") is not None:
        d["doc_bow"] = corpus["item_bow"].copy()
    return d


def main() -> None:
    data = PC.load_civil(6000, 2000, need_train_bow=True, need_doc_bow=False)
    dictionary = data["train"]["dictionary"]
    train = as_title_only(data["train"], dictionary)
    test = as_title_only(data["test"], dictionary)
    out = ROOT / "outputs" / "rerun" / "fcntm_civil_titleonly_3seed.csv"
    run_corpus(train, test, [11, 23, 37], 30, str(out),
               field_dropout=0.5, recover_nan=True, max_epochs=700, lr=5e-4,
               fusion="gate", fusion_norm="balanced", fusion_weight=0.5,
               resume_every=50, allow_resume=True,
               state_path=str(ROOT / "outputs" / "rerun" /
                              "fcntm_civil_titleonly_state.pt"))
    print("saved", out)


if __name__ == "__main__":
    main()
