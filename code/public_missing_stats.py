# -*- coding: utf-8 -*-
"""Compute natural missing-field statistics for the public GitHub corpus.

Field 1 (reason) = issue body; field 2 (item) = issue title.
A record counts as "missing field" when the body is empty or too short to be
usable.  Reported per repository and overall, plus a token-length profile of
the usable records.  No subsampling: this scans every issue in the four
parquet files, so the numbers describe the corpus, not our split.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from public_data_github import (
    REPO_FILES, SRC_DIR, clean_issue_text, en_tokenize_issue,
)

MIN_BODY_TOKENS = 8

rows = []
totals = {"issues": 0, "empty_body": 0, "too_short_body": 0, "no_title": 0}
body_lens = []
title_lens = []

for repo, fname in REPO_FILES.items():
    df = pd.read_parquet(SRC_DIR / fname, columns=["is_pull_request", "title", "body"])
    n_all = len(df)
    if "is_pull_request" in df.columns:
        df = df[~df["is_pull_request"]]
    n_iss = len(df)
    title = df["title"].fillna("")
    body = df["body"].fillna("").astype(str)
    empty = 0
    short = 0
    notitle = 0
    for t, b in zip(title, body):
        if str(t).strip() == "":
            notitle += 1
        bs = str(b).strip()
        if bs == "":
            empty += 1
            continue
        nb = len(en_tokenize_issue(clean_issue_text(bs)))
        nt = len(en_tokenize_issue(t))
        if nb < MIN_BODY_TOKENS or nt < 2:
            short += 1
            continue
        body_lens.append(nb)
        title_lens.append(nt)
    rows.append({
        "repo": repo, "all_rows": n_all, "issues_no_pr": n_iss,
        "empty_body": empty, "too_short": short, "no_title": notitle,
        "usable": n_iss - empty - short,
        "missing_rate": round((empty + short) / max(1, n_iss), 4),
    })
    totals["issues"] += n_iss
    totals["empty_body"] += empty
    totals["too_short_body"] += short
    totals["no_title"] += notitle

out = pd.DataFrame(rows)
print(out.to_string(index=False))
print()
miss = totals["empty_body"] + totals["too_short_body"]
print(f"ALL: issues={totals['issues']} empty_body={totals['empty_body']} "
      f"too_short={totals['too_short_body']} no_title={totals['no_title']}")
print(f"natural missing-field rate = {miss}/{totals['issues']} = {miss/totals['issues']:.4f}")
import numpy as np
print(f"body tokens: mean={np.mean(body_lens):.1f} median={np.median(body_lens):.0f} "
      f"p10={np.percentile(body_lens,10):.0f} p90={np.percentile(body_lens,90):.0f}")
print(f"title tokens: mean={np.mean(title_lens):.1f} median={np.median(title_lens):.0f}")
Path("outputs/public_missing_stats.json").write_text(json.dumps({
    "min_body_tokens": MIN_BODY_TOKENS, "per_repo": rows, "totals": totals,
    "natural_missing_rate": miss / totals["issues"],
    "body_token_mean": float(np.mean(body_lens)),
    "body_token_median": float(np.median(body_lens)),
    "title_token_mean": float(np.mean(title_lens)),
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved outputs/public_missing_stats.json")
