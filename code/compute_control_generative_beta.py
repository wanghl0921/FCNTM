# -*- coding: utf-8 -*-
"""Generative beta metrics for the concatenated single-encoder control."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"C:\Users\20102\Desktop\FCNTM\FCNTM完整代码")
sys.path.insert(0, str(ROOT))
import models as NN
import occlusion_robustness as O
from run_fcntm_train import evaluate, top_words

K = 30
SEEDS = [11, 23, 37]
OUT = ROOT / "outputs" / "rerun" / "control_generative_beta.json"


def main():
    result = {}
    for corpus in ["github", "agnews", "civil"]:
        train, test = O.load_corpus(corpus)
        vocab = test["vocab"]
        ck = torch.load(ROOT / "outputs" / "rerun" / f"concat_state_{corpus}.pt",
                        map_location="cpu", weights_only=False)
        rows = []
        for seed in SEEDS:
            rec = ck[f"seed:{seed}"]
            model = NN.FCNTMConcat(len(vocab), K, **rec["kwargs"])
            model.load_state_dict(rec["model_state"])
            model.eval()
            with torch.no_grad():
                logits = model.decode(torch.eye(K))
                prob = torch.softmax(logits, dim=-1).numpy()
            topics = top_words(prob, vocab, 10)
            sc = evaluate(topics, prob, test)
            rows.append({"seed": seed, "CV": sc["CV"], "NPMI": sc["NPMI"],
                         "EXCL_KADJ": sc["EXCL_KADJ"], "IRBO": sc["IRBO"]})
            print(corpus, seed, sc["CV"], sc["NPMI"], flush=True)
        result[corpus] = {
            "mean_CV": float(np.mean([r["CV"] for r in rows])),
            "sd_CV": float(np.std([r["CV"] for r in rows], ddof=1)),
            "mean_NPMI": float(np.mean([r["NPMI"] for r in rows])),
            "sd_NPMI": float(np.std([r["NPMI"] for r in rows], ddof=1)),
            "rows": rows,
        }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
