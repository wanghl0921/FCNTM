# -*- coding: utf-8 -*-
'''汇总"字段不对称度阶梯"：每个截断变体内，各消融配置相对 full 的配对差。'''

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
VARIANTS = [("full", 0), ("abs60", 60), ("abs30", 30)]
READOUTS = (("beta", "FC-NTM"), ("cTFIDF", "FC-NTM-cTFIDF"))


def load_variant(tag):
    '''One frame per variant: dataset/config/seed/readout/CV/NPMI.

    The untruncated ("full") reference lives in the main-results dump
    ``fcntm_civil_final.csv``; the core-ablation table only holds the
    no-cross / no-consistency / no-mask / no-fielddrop rows.
    '''
    if tag == "full":
        path = OUT / "fcntm_civil_final.csv"
        if not path.exists():
            return None
        raw = pd.read_csv(path)
        rows = []
        for seed in sorted(int(s) for s in raw["seed"].unique()):
            for readout, model in READOUTS:
                rec = raw[(raw["seed"] == seed) & (raw["model"] == model)]
                if rec.empty:
                    continue
                r = rec.iloc[0]
                rows.append({"dataset": "civil", "config": "full",
                             "seed": seed, "readout": readout,
                             "CV": r["CV"], "NPMI": r["NPMI"]})
        return pd.DataFrame(rows)
    path = OUT / f"ablation_civil_{tag}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return frame[frame["dataset"] == "civil"] if "dataset" in frame else frame


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    rows = []
    for tag, words in VARIANTS:
        base_frame = load_variant(tag)
        if base_frame is None or base_frame.empty:
            print(f"  skip {tag}: no full-config rows found")
            continue
        if tag == "full":
            abl_path = OUT / "ablation_public_civil_core.csv"
            if not abl_path.exists():
                print("  skip full: ablation_public_civil_core.csv not found")
                continue
            abl_frame = pd.read_csv(abl_path)
            if "dataset" in abl_frame:
                abl_frame = abl_frame[abl_frame["dataset"] == "civil"]
        else:
            abl_frame = base_frame
        # field ratio of this variant, measured exactly as the paper defines
        # it: mean reason tokens / mean item tokens on the training split
        # after the shared vocabulary filter, so full / abs60 / abs30 use one
        # convention (2026-10-07 review, item 8)
        import public_data_civil as PC
        data_variant = PC.load_civil(6000, 2000, max_abstract_words=words,
                                     need_train_bow=False, need_doc_bow=False)
        t = np.mean([len(x) for x in data_variant["train"]["item_tokens"]])
        a = np.mean([len(x) for x in data_variant["train"]["reason_tokens"]])
        ratio = a / max(t, 1e-9)
        del data_variant
        for readout in ("beta", "cTFIDF"):
            base = base_frame[(base_frame["config"] == "full")
                              & (base_frame["readout"] == readout)]
            base = base.sort_values("seed")
            if base.empty:
                continue
            for config in ("no-cross", "no-mask"):
                other = abl_frame[(abl_frame["config"] == config)
                                  & (abl_frame["readout"] == readout)].sort_values("seed")
                if other.empty:
                    continue
                m = other.merge(base, on="seed", suffixes=("_a", "_f"))
                dcv = (m["CV_a"] - m["CV_f"]).to_numpy()
                rows.append({
                    "variant": tag, "abstract_words": words,
                    "field_ratio": round(ratio, 2), "readout": readout,
                    "config": config, "n_seeds": len(m),
                    "full_CV": round(float(base["CV"].mean()), 4),
                    "CV_mean_abl": round(float(other["CV"].mean()), 4),
                    "delta_CV": round(float(dcv.mean()), 4),
                    "delta_CV_positive": int((dcv > 0).sum()),
                })
    if not rows:
        print("no ladder rows yet")
        return 1
    table = pd.DataFrame(rows)
    (OUT / "rerun").mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "rerun" / "ladder_summary.csv", index=False,
                 encoding="utf-8-sig")
    print("\n字段不对称度阶梯（差 = 去掉该组件 − full，同种子配对）")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())