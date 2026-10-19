# -*- coding: utf-8 -*-
"""Numbers the review asks the paper to disclose (all read-only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import public_data as PD
import public_data_github as PG

for tag, loader in (("agnews", lambda: PD.load_agnews(1500, 1900,
                                                      need_doc_bow=False)),
                    ("github", lambda: PG.load_github(1500, 500,
                                                      need_doc_bow=False))):
    d = loader()
    tr, te = d["train"], d["test"]
    tr_keys = set(tuple(t) for t in tr["doc_tokens"] if t)
    dup = sum(1 for t in te["doc_tokens"] if t and tuple(t) in tr_keys)
    print(f"=== {tag}")
    print(f"  token-sequence duplicates across splits: {dup} "
          f"({100.0 * dup / len(te['doc_tokens']):.2f}% of test)")
    print(f"  empty summary (train/test): "
          f"{sum(1 for t in tr['item_tokens'] if not t)}/"
          f"{sum(1 for t in te['item_tokens'] if not t)}")
    print(f"  doc tokens max (train/test): "
          f"{max(len(t) for t in tr['doc_tokens'])}/"
          f"{max(len(t) for t in te['doc_tokens'])}")
    print(f"  background tokens max (train/test): "
          f"{max(len(t) for t in tr['reason_tokens'])}/"
          f"{max(len(t) for t in te['reason_tokens'])}")

print("\n=== topic support in the main run tables")
for tag, path in (("agnews", "outputs/public_benchmark_runs_agnews.csv"),
                  ("github",
                   "outputs/public_benchmark_runs_githubtrain6000_test2000.csv")):
    f = pd.read_csv(path)
    if "CV_n" in f.columns and "topic_count" in f.columns:
        g = (f.groupby("model")
              .agg(topics=("topic_count", "mean"), cv_n=("CV_n", "mean"))
              .round(2))
        print(f"-- {tag}")
        print(g.to_string())
