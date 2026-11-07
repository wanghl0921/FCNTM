# -*- coding: utf-8 -*-
"""Field-restricted read-out of the FC-NTM generative beta read-out."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd

ROOT = Path(r"C:\Users\20102\Desktop\FCNTM\FCNTM完整代码")
sys.path.insert(0, str(ROOT))
import occlusion_robustness as O
import metrics as M
from run_public_missing_field import field_vocab_ids, topics_in_field
import config as cfg

K = 30
SEEDS = [11, 23, 37]
OUT = ROOT / "outputs" / "rerun" / "fieldview_beta_bn.json"


def main():
    res = {}
    for corpus in ["github", "agnews", "civil"]:
        train, test = O.load_corpus(corpus)
        vocab = test["vocab"]
        cause_ids = field_vocab_ids(train["reason_tokens"], vocab)
        item_ids = field_vocab_ids(train["item_tokens"], vocab)
        rows = []
        for seed in SEEDS:
            path = ROOT / "outputs" / f"{corpus}_missing_state.pt"
            ad = O.FCNTMAdapter(path, len(vocab), K, seed=seed)
            with torch.no_grad():
                beta = torch.softmax(ad.model.decode(torch.eye(K)), dim=-1).numpy()
            for view, ids, toks in (
                ("background", cause_ids,
                 [t for t in test["reason_tokens"] if t]),
                ("summary", item_ids,
                 [t for t in test["item_tokens"] if t])):
                bows = [test["dictionary"].doc2bow(t) for t in toks]
                topics = topics_in_field(beta, ids, vocab)
                ev = M.coherence_scores(topics, toks, bows,
                                        test["dictionary"], topn=cfg.TOPN)
                rows.append({"dataset": corpus, "seed": seed, "view": view,
                             "CV": float(ev["CV"]), "NPMI": float(ev["NPMI"]),
                             "CV_n": int(ev["CV_n"])})
            del ad
        res[corpus] = rows
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    frame = pd.DataFrame([r for rows in res.values() for r in rows])
    print(frame.groupby(["dataset", "view"])[["CV", "NPMI"]].agg(["mean", "std"]).round(4).to_string())
    print("saved", OUT)


if __name__ == "__main__":
    main()
