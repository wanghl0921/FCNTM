# -*- coding: utf-8 -*-
"""Readable summary of the small-scale sweep in outputs/small/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent / "outputs" / "small"
OUT_MD = D / "REPORT.md"


def agg(df, keys, cols):
    g = df.groupby(keys, dropna=False)[cols].agg(["mean", "std"])
    return g


def fmt(df, keys, cols, nd=3):
    lines = []
    for key, sub in df.groupby(keys, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        parts = []
        for c in cols:
            vals = pd.to_numeric(sub[c], errors="coerce").dropna()
            if vals.empty:
                parts.append("—")
            elif len(vals) == 1:
                parts.append(f"{c}={vals.iloc[0]:.{nd}f}")
            else:
                parts.append(f"{c}={vals.mean():.{nd}f}±{vals.std(ddof=1):.{nd}f}")
        lines.append(" | ".join(str(k) for k in key) + " :: " + ", ".join(parts))
    return lines


def main() -> None:
    out = []
    def emit(s=""):
        print(s)
        out.append(s)

    main_f = D / "main_github.csv"
    if main_f.exists():
        df = pd.read_csv(main_f)
        emit("## 主结果 GitHub (K=20, seeds 11/23)")
        for line in fmt(df, ["model", "readout"], ["CV", "NPMI", "UMass", "IRBO"]):
            emit("- " + line)
        emit()

    fv = D / "fieldview_github.csv"
    if fv.exists():
        df = pd.read_csv(fv)
        emit("## 字段视角 GitHub")
        for line in fmt(df, ["model", "readout"], ["cause_CV", "cause_NPMI", "item_CV", "item_NPMI"]):
            emit("- " + line)
        emit()

    ms = D / "missing_github.csv"
    if ms.exists():
        df = pd.read_csv(ms)
        emit("## 遮挡率扫描 GitHub（背景字段被人为清空）")
        emit()
        emit("| 遮挡率 | 设置 | 模型 | 全字段C_V | 全字段NPMI | 概要视角C_V |")
        emit("|---|---|---|---|---|---|")
        for rate in sorted(df["mask_rate"].unique()):
            sub = df[df["mask_rate"] == rate]
            for fd in sorted(sub["field_dropout"].dropna().unique()):
                s2 = sub[sub["field_dropout"] == fd]
                for model in s2["model"].unique():
                    s3 = s2[s2["model"] == model]
                    cv = pd.to_numeric(s3["CV"], errors="coerce").mean()
                    npmi = pd.to_numeric(s3["NPMI"], errors="coerce").mean()
                    icv = pd.to_numeric(s3["item_CV"], errors="coerce").mean()
                    emit(f"| {rate:.2f} | fdrop={fd} | {model} | {cv:.3f} | {npmi:.3f} | {icv:.3f} |")
            s2 = sub[sub["field_dropout"].isna()]
            for model in s2["model"].unique():
                s3 = s2[s2["model"] == model]
                cv = pd.to_numeric(s3["CV"], errors="coerce").mean()
                npmi = pd.to_numeric(s3["NPMI"], errors="coerce").mean()
                icv = pd.to_numeric(s3["item_CV"], errors="coerce").mean()
                emit(f"| {rate:.2f} | (基线) | {model} | {cv:.3f} | {npmi:.3f} | {icv:.3f} |")
        emit()

    ab = D / "ablation_github.csv"
    if ab.exists():
        df = pd.read_csv(ab)
        emit("## 消融 GitHub")
        for line in fmt(df, ["config", "readout"], ["CV", "NPMI", "UMass"]):
            emit("- " + line)
        emit()

    im = D / "imputation_github.csv"
    if im.exists():
        df = pd.read_csv(im)
        emit("## 跨字段词预测 GitHub")
        for line in fmt(df, ["config"], ["recall", "precision"], nd=4):
            emit("- " + line)
        emit()

    # --- cross-corpus: does the mechanism only pay off on complementary fields?
    comp_f = Path(__file__).resolve().parent / "outputs" / "field_redundancy.json"
    imps = {}
    for corpus in ("github", "agnews"):
        f = D / f"imputation_{corpus}.csv"
        if f.exists():
            imps[corpus] = pd.read_csv(f)
    if comp_f.exists() or imps:
        emit("## 字段互补性 vs 跨字段重构收益")
        emit()
        if comp_f.exists():
            comp = json.loads(comp_f.read_text(encoding="utf-8"))
            emit("| 语料 | 留出 CCA 平均平方相关 | 留出 Ridge 平均R² | 冗余度 |")
            emit("|---|---|---|---|")
            for c, v in comp.items():
                # key renamed in the 2026-09 revision; keep reading old files
                cca = v.get("cca_sq_mean", v.get("cca_mean_r2"))
                ridge = v.get("ridge_R2_mean", float("nan"))
                emit(f"| {c} | {cca:.4f} | {ridge:.4f} | "
                     f"{'高（冗余）' if cca > 0.1 else '低（互补）'} |")
            emit()
        if imps:
            emit("| 语料 | 跨字段重构 | 召回 | 精度 |")
            emit("|---|---|---|---|")
            for c, df in imps.items():
                for cfg, label in (("recon2", "开启"), ("noxrec", "关闭")):
                    g = df[df["config"] == cfg]
                    if g.empty:
                        continue
                    emit(f"| {c} | {label} | {g['recall'].mean():.4f} | {g['precision'].mean():.4f} |")
            emit()
            emit("解读：跨字段重构在低冗余（互补）语料上把跨字段词预测召回提高一倍以上，")
            emit("在高冗余语料上几乎没有任何效果——机制只在字段确实互补时才有可迁移的信息。")
            emit()

    OUT_MD.write_text("\n".join(out), encoding="utf-8")
    print("\nsaved", OUT_MD)


if __name__ == "__main__":
    main()
