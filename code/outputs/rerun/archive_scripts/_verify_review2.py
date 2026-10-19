# -*- coding: utf-8 -*-
"""Second batch of review checks: duplicates, paired tests, small-sample split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent


def duplicated_token_sequences() -> None:
    print("=" * 70)
    print("duplicate token sequences across train/test (pre-vocab-filter)")
    for name, loader in (("agnews", "public_data"), ("github", "public_data_github")):
        if name == "agnews":
            import public_data as PD
            data = PD.load_agnews(1500, 1900, need_train_bow=False)
        else:
            import public_data_github as PGH
            data = PGH.load_github(1500, 500, need_train_bow=False)
        tr = [" ".join(t) for t in data["train"]["doc_tokens"]]
        te = [" ".join(t) for t in data["test"]["doc_tokens"]]
        inter = set(tr) & set(te)
        print(f"  {name}: train={len(tr)} test={len(te)} "
              f"identical doc-token sequences shared: {len(inter)}")
        # label balance of AG News train
        if name == "agnews":
            labels = np.asarray(data["train"]["labels"])
            print("   train label counts:", dict(zip(*np.unique(labels, return_counts=True))))


def paired_tests() -> None:
    print("=" * 70)
    print("FC-NTM-cTFIDF vs BTM, paired across seeds (CV)")
    for name, base, fc in (
        ("AGNews", "outputs/public_benchmark_runs_agnews.csv",
         "outputs/codev_agnews_gate.csv"),
        ("GitHub", "outputs/public_benchmark_runs_githubtrain6000_test2000.csv",
         "outputs/codev_github_gate_bal.csv"),
    ):
        b = pd.read_csv(ROOT / base)
        f = pd.read_csv(ROOT / fc)
        b = b[b["model"] == "BTM"].sort_values("seed")
        f = f[f["model"] == "FC-NTM-cTFIDF"].sort_values("seed")
        d = f["CV"].to_numpy() - b["CV"].to_numpy()
        t, p = stats.ttest_rel(f["CV"].to_numpy(), b["CV"].to_numpy())
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(),
                              scale=stats.sem(d))
        print(f"  {name}: diffs={np.round(d, 6)} mean={d.mean():+.6f} "
              f"CI=[{ci[0]:+.6f},{ci[1]:+.6f}] p={p:.4f}")


def small_sample_origin() -> None:
    print("=" * 70)
    print("public_small sample provenance")
    import public_small as PS
    for name in ("github", "agnews"):
        d = PS.load_small(name)
        print(f"  {name}: train={len(d['train']['doc_tokens'])} "
              f"test={len(d['test']['doc_tokens'])}")
    print("  (the development metrics reuse a 400-document slice of the "
          "official test.csv)")
    fc = json.loads((ROOT / "outputs" / "field_complementarity.json").read_text(
        encoding="utf-8"))
    for name, splits in fc.items():
        tr = splits.get("train", {})
        print(f"  field_complementarity[{name}][train]: n_docs={tr.get('n_docs')} "
              f"bg={tr.get('mean_bg_len')} sm={tr.get('mean_sm_len')}")


def author_note() -> None:
    print("=" * 70)
    print("bibliography entries flagged by the review (offline: not verifiable "
          "here)")
    tex = (ROOT.parent / "FCNTM完整论文" / "main.tex").read_text(encoding="utf-8")
    for key in ("esmaeili2019structured", "bianchi2021ctm", "wu2024survey",
                "srivastava2014dropout"):
        for line in tex.splitlines():
            if line.startswith(f"\\bibitem{{{key}}}"):
                print("  ", line)


if __name__ == "__main__":
    duplicated_token_sequences()
    paired_tests()
    small_sample_origin()
    author_note()
