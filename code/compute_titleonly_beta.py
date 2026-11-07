# -*- coding: utf-8 -*-
"""Generative beta metrics for the same-architecture title-only FC-NTM."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"C:\Users\20102\Desktop\FCNTM\FCNTM完整代码")
sys.path.insert(0, str(ROOT))
import public_data_civil as PC
import models as NN
from run_fcntm_train import evaluate, top_words
from bn_readout_sensitivity import load_state

K = 30
SEEDS = [11, 23, 37]
OUT = ROOT / "outputs" / "rerun" / "fcntm_civil_titleonly_beta.json"


def as_title_only(corpus, dictionary):
    d = dict(corpus)
    n = len(corpus["item_tokens"])
    d["reason_tokens"] = [[] for _ in range(n)]
    d["reason_bow"] = np.zeros_like(corpus["reason_bow"])
    d["doc_tokens"] = [list(t) for t in corpus["item_tokens"]]
    d["corpus"] = [dictionary.doc2bow(t) for t in d["doc_tokens"]]
    return d


def main():
    data = PC.load_civil(6000, 2000, need_train_bow=True, need_doc_bow=False)
    test = as_title_only(data["test"], data["train"]["dictionary"])
    rows = []
    for seed in SEEDS:
        model = load_state(ROOT / "outputs" / "rerun" / "fcntm_civil_titleonly_state.pt",
                           "civil", seed, K, len(test["vocab"]))
        with torch.no_grad():
            prob = torch.softmax(model.decode(torch.eye(K)), dim=-1).numpy()
        sc = evaluate(top_words(prob, test["vocab"], 10), prob, test)
        rows.append({"seed": seed, **{k: float(sc[k]) for k in
                                      ("CV", "NPMI", "UMass", "EXCL_KADJ", "IRBO")}})
        print(seed, sc["CV"], sc["NPMI"], flush=True)
        del model
    out = {k: float(np.mean([r[k] for r in rows])) for k in
           ("CV", "NPMI", "UMass", "EXCL_KADJ", "IRBO")}
    out["rows"] = rows
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
