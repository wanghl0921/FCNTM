# -*- coding: utf-8 -*-
"""Per-repository decomposition of the "unusable field" counts.

``public_missing_stats.py`` folds three different situations into one
``too_short`` bucket.  The paper needs them separated:

    empty_body      raw body string is empty
    body_short      body non-empty but fewer than 8 tokens after cleaning
    title_short     body usable, title shorter than 2 tokens
"""
from __future__ import annotations

import pandas as pd

from public_data_github import (REPO_FILES, SRC_DIR, clean_issue_text,
                                en_tokenize_issue)

MIN_BODY_TOKENS = 8

rows = []
for repo, fname in REPO_FILES.items():
    df = pd.read_parquet(SRC_DIR / fname,
                         columns=["is_pull_request", "title", "body"])
    if "is_pull_request" in df.columns:
        df = df[~df["is_pull_request"]]
    n = len(df)
    empty = body_short = title_short = usable = 0
    for t, b in zip(df["title"].fillna(""), df["body"].fillna("").astype(str)):
        bs = str(b).strip()
        if bs == "":
            empty += 1
            continue
        nb = len(en_tokenize_issue(clean_issue_text(bs)))
        nt = len(en_tokenize_issue(t))
        if nb < MIN_BODY_TOKENS:
            body_short += 1
        elif nt < 2:
            title_short += 1
        else:
            usable += 1
    rows.append({"repo": repo, "issues": n, "empty": empty,
                 "body_short": body_short, "title_short": title_short,
                 "usable": usable,
                 "body_unusable": empty + body_short,
                 "body_unusable_rate": (empty + body_short) / n,
                 "empty_rate": empty / n})

frame = pd.DataFrame(rows)
print(frame.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
tot = frame[["issues", "empty", "body_short", "title_short", "usable",
             "body_unusable"]].sum()
print("\nTOTAL", dict(tot))
print("body-unusable rate %.4f | empty rate %.4f | title-short rate %.4f"
      % (tot["body_unusable"] / tot["issues"], tot["empty"] / tot["issues"],
         tot["title_short"] / tot["issues"]))
