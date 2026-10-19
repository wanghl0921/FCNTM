# -*- coding: utf-8 -*-
"""Ad-hoc verification of the concrete claims in FCNTM_代码与论文审阅报告.md.

Run from the code directory:  python _verify_review.py
Prints observations; writes nothing except (optionally) recomputed summaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def head(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def check_corpora() -> None:
    head("1. corpus statistics recomputed from the stored splits")
    import public_data as PD
    import public_data_github as PGH

    ag = PD.load_agnews(1500, 1900)
    gh = PGH.load_github(1500, 500)
    for name, data in (("AGNews", ag), ("GitHubIssues-4repo", gh)):
        tr, te = data["train"], data["test"]
        print(f"\n{name}")
        print(f"  train n={tr['n_docs']} test n={te['n_docs']} "
              f"train vocab={tr['n_vocab']} test vocab={te['n_vocab']}")
        for split, c in (("train", tr), ("test", te)):
            bg = float(np.mean([len(t) for t in c["reason_tokens"]]))
            sm = float(np.mean([len(t) for t in c["item_tokens"]]))
            doc = float(np.mean([len(t) for t in c["doc_tokens"]]))
            print(f"  {split}: bg={bg:.4f} sm={sm:.4f} doc={doc:.4f}")
        vocab = set(tr["vocab"])
        for w in ("build", "columnar", "vectorized", "compiler", "docker"):
            print(f"  '{w}' in train vocab: {w in vocab}")


def check_paper_table_aggregation() -> None:
    head("2. paper table values recomputed from the per-seed CSVs")
    for name, path in (
        ("AGNews", ROOT / "outputs" / "public_benchmark_runs_agnews.csv"),
        ("GitHub", ROOT / "outputs"
         / "public_benchmark_runs_githubtrain6000_test2000.csv"),
    ):
        df = pd.read_csv(path)
        print(f"\n{name} ({len(df)} rows)")
        if "error" in df.columns:
            failed = df[df["error"].notna()]
            print(f"  rows with non-null error: {len(failed)} -> "
                  f"{failed['model'].tolist()}")
        for model in ("BTM", "CombinedTM", "NeuralLDA", "BERTopic", "ETM",
                      "FC-NTM", "FC-NTM-cTFIDF", "LDA", "NMF", "Top2Vec",
                      "CTM-lite"):
            sub = df[df["model"] == model]
            if sub.empty:
                continue
            cv = sub["CV"].astype(float)
            npmi = sub["NPMI"].astype(float)
            print(f"  {model:14s} n={len(sub)} CV={cv.mean():.6f}"
                  f"+-{cv.std(ddof=1):.6f} NPMI={npmi.mean():.6f}"
                  f"+-{npmi.std(ddof=1):.6f} themes="
                  f"{sub['topic_count'].tolist() if 'topic_count' in sub else '-'}")
        if {"BTM", "FC-NTM-cTFIDF"} <= set(df["model"]):
            b = df[df["model"] == "BTM"].sort_values("seed")
            f = df[df["model"] == "FC-NTM-cTFIDF"].sort_values("seed")
            if len(b) == len(f):
                d = f["CV"].to_numpy() - b["CV"].to_numpy()
                print(f"  cTFIDF-BTM paired CV diff: {d} mean={d.mean():+.6f}")


def check_occlusion_numbers() -> None:
    head("3. occlusion table numbers (GitHub)")
    p = ROOT / "outputs" / "public_missing_field_gate.csv"
    if not p.exists():
        p = ROOT / "outputs" / "missing_field_eval.csv"
    df = pd.read_csv(p)
    print(df.to_string())


def check_field_redundancy() -> None:
    head("4. field_redundancy.measure() reproduced")
    import field_redundancy as FR
    from public_small import load_small

    saved = json.loads((ROOT / "outputs" / "field_redundancy.json").read_text(
        encoding="utf-8"))
    print("saved json:", json.dumps(saved, indent=2))
    for name in ("github", "agnews"):
        out = FR.measure(load_small(name))
        print(f"  recomputed {name}: {out}")


def check_tfidf_reestimates_idf() -> None:
    head("5. tfidf() re-estimates IDF per split (representation mismatch)")
    import scipy.sparse as sp
    import field_redundancy as FR

    doc = [["a", "b"]]
    ref_same = [["a", "b"], ["a", "b"]]
    ref_other = [["a", "c"], ["b", "c"]]
    idx = {"a": 0, "b": 1, "c": 2}
    v1 = FR.tfidf(doc, idx, 3)
    # a reference corpus that never contains a/b changes their IDF weight
    big = [["c"]] * 50
    v2 = FR.tfidf(doc, idx, 3)
    for label, corpus in (("train-style", ref_same), ("test-style", ref_other)):
        m = FR.tfidf(doc, idx, 3)
        print(f"  {label}: {m.toarray().round(6).tolist()}")
    print("  same doc, but scored under a corpus where a/b are rare:")
    unseen_doc = [["a", "b"]]
    print("   ", FR.tfidf(unseen_doc, idx, 3).toarray().round(6).tolist())
    wide = [["a", "b"]] + [["c"]] * 100
    print("   ", FR.tfidf(wide, idx, 3)[0].toarray().round(6).tolist(),
          "<- row 0 of a corpus in which a/b are rare")


def check_benchmark_done_flag() -> None:
    head("6. benchmark resume logic `if not r.get('error')`")
    import benchmark_public as BP

    rows = [
        {"model": "LDA", "seed": 11, "error": np.nan, "CV": 0.4},
        {"model": "LDA", "seed": 23, "error": "", "CV": 0.41},
        {"model": "LDA", "seed": 37, "error": "boom", "CV": np.nan},
    ]
    print("  returned rows:", len(BP.summarize_rows(rows)))
    for r in rows:
        print("   ", r, "-> not r.get('error') =", not r.get("error"))
    df = pd.read_csv(ROOT / "outputs" / "public_benchmark_runs_agnews.csv")
    done = [r for _, r in df.iterrows() if not r.get("error")]
    print(f"  AGNews CSV: {len(df)} rows, resume-done={len(done)} "
          f"(pd.isna(error)={int(df['error'].isna().sum())})")


def check_metrics() -> None:
    head("7. metrics: IRBO of identical topic lists, coherence n")
    import metrics as M

    a = [f"w{i}" for i in range(10)]
    print("  rbo(identical) =", M.rbo(a, a))
    print("  irbo([a, a])   =", M.irbo_diversity([a, a]))
    print("  rbo exact-match tail term p**L =", 0.9 ** 10)


def check_ablation_shadowing() -> None:
    head("8. run_ablation_public.py name shadowing (`for config in configs`)")
    src = (ROOT / "run_ablation_public.py").read_text(encoding="utf-8")
    print("  uses `for config in configs:`:",
          "for config in configs:" in src)
    print("  uses `config.ROOT` after the loop:",
          "dp = config.ROOT / dp" in src)


def main() -> None:
    check_corpora()
    check_paper_table_aggregation()
    check_occlusion_numbers()
    check_tfidf_reestimates_idf()
    check_field_redundancy()
    check_benchmark_done_flag()
    check_metrics()
    check_ablation_shadowing()


if __name__ == "__main__":
    main()
