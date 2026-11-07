# -*- coding: utf-8 -*-
"""Compute generative beta read-out for FC-NTM (10 seeds) and compare to BTM.

Uses saved checkpoints only; no training.  Writes per-seed and summary JSON for
the main-table update.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(r"C:\Users\20102\Desktop\FCNTM\FCNTM完整代码")
sys.path.insert(0, str(ROOT))
import models as NN
import occlusion_robustness as O
from run_fcntm_train import evaluate, top_words

K = 30
SEEDS = [11, 23, 37, 53, 71, 89, 101, 113, 127, 139]
OUT = ROOT / "outputs" / "rerun" / "generative_beta_readout.json"


def load_model(corpus: str, seed: int, n_vocab: int):
    key = f"fielddrop_on:{seed}"
    for path in [ROOT / "outputs" / f"{corpus}_missing_state.pt",
                 ROOT / "outputs" / "rerun" / f"fcntm_main_state_{corpus}.pt"]:
        if not path.exists():
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if key in ck:
            rec = ck[key]
            kwargs = rec.get("kwargs") or O.final_model_kwargs(0.5, "gate", "balanced", 0.5)
            model = NN.FCNTM(n_vocab, K, **kwargs)
            model.load_state_dict(rec["model_state"])
            model.eval()
            return model
    raise KeyError(f"no state for {corpus} seed {seed}")


def btm_beta_scores(corpus: str):
    rows = []
    for path in [ROOT / "outputs" / f"fair_readout_{corpus}_3seed.csv",
                 ROOT / "outputs" / "rerun" / f"fair_readout_{corpus}_newseeds.csv"]:
        if path.exists():
            rows.append(pd.read_csv(path, encoding="utf-8-sig"))
    df = pd.concat(rows, ignore_index=True)
    df = df[(df["model"] == "BTM") & (df["readout"] == "beta")]
    return {int(r.seed): float(r.CV) for r in df.itertuples()}


def stats_for(diffs):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    tcrit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else float("nan")
    pos = int((d > 0).sum())
    sign_p = float(stats.binomtest(pos, n, 0.5, alternative="two-sided").pvalue)
    try:
        wil = float(stats.wilcoxon(d, alternative="two-sided").pvalue)
    except Exception:
        wil = float("nan")
    return {"n": n, "mean": mean, "ci_low": mean - tcrit * se,
            "ci_high": mean + tcrit * se, "positive": pos,
            "sign_p": sign_p, "wilcoxon_p": wil}


def main() -> None:
    out = {}
    per_seed = []
    for corpus in ["github", "agnews", "civil"]:
        print("corpus", corpus, flush=True)
        train, test = O.load_corpus(corpus)
        vocab = test["vocab"]
        n_vocab = len(vocab)
        btm = btm_beta_scores(corpus)
        vals = {}
        for seed in SEEDS:
            model = load_model(corpus, seed, n_vocab)
            with torch.no_grad():
                theta = torch.eye(K)
                logits = model.decode(theta)
                bn_prob = torch.softmax(logits, dim=-1).numpy()
            topics = top_words(bn_prob, vocab, 10)
            sc = evaluate(topics, bn_prob, test)
            vals[seed] = sc
            per_seed.append({"corpus": corpus, "seed": seed,
                             "CV": sc["CV"], "NPMI": sc["NPMI"],
                             "UMass": sc["UMass"],
                             "EXCL_KADJ": sc["EXCL_KADJ"],
                             "IRBO": sc["IRBO"],
                             "BTM_beta_CV": btm.get(seed)})
            print(f"  seed {seed}: FC BN CV {sc['CV']:.4f} / BTM {btm.get(seed)}",
                  flush=True)
            del model
        arr = [vals[s]["CV"] - btm[s] for s in SEEDS if s in btm]
        out[corpus] = {
            "mean_CV": float(np.mean([vals[s]["CV"] for s in SEEDS])),
            "mean_NPMI": float(np.mean([vals[s]["NPMI"] for s in SEEDS])),
            "mean_EXCL_KADJ": float(np.mean([vals[s]["EXCL_KADJ"] for s in SEEDS])),
            "mean_IRBO": float(np.mean([vals[s]["IRBO"] for s in SEEDS])),
            "mean_UMass": float(np.mean([vals[s]["UMass"] for s in SEEDS])),
            "three_seed_mean_CV": float(np.mean([vals[s]["CV"] for s in SEEDS[:3]])),
            "three_seed_mean_NPMI": float(np.mean([vals[s]["NPMI"] for s in SEEDS[:3]])),
            "three_seed_mean_EXCL_KADJ": float(np.mean([vals[s]["EXCL_KADJ"] for s in SEEDS[:3]])),
            "three_seed_mean_IRBO": float(np.mean([vals[s]["IRBO"] for s in SEEDS[:3]])),
            "paired_vs_BTM": stats_for(arr),
            "BTM_mean_CV_3seed": float(np.mean([btm[s] for s in SEEDS[:3]])),
            "BTM_mean_CV_10seed": float(np.mean([btm[s] for s in SEEDS])),
        }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(per_seed).to_csv(OUT.with_suffix(".csv"), index=False,
                                  encoding="utf-8-sig")
    print(json.dumps(out, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
