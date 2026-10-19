# -*- coding: utf-8 -*-
"""Identify the development records that overlap the final test split.

The small-scale loaders (``public_small.py``) subsample the *official test
split* for design iteration, and several design decisions (gate vs mean fusion,
fusion normalisation, reading of variants) were made on those 400 documents.
Reporting the final numbers on the full test split therefore reuses documents
that were already seen during development.

This module finds exactly which final-test rows those are, so evaluation can be
repeated on the disjoint remainder:

    train/dev  : the 1000-document development training subsample
    final test : official test split
    clean test : official test split minus the 400 development documents

``clean_test_mask`` returns a boolean mask over the final test split.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config

SMALL_DIR = config.DATA_DIR / "small"


def _dev_frame(corpus: str) -> pd.DataFrame:
    path = SMALL_DIR / f"{corpus}_test.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: run public_small.load_small('{corpus}') first")
    return pd.read_csv(path)


def _key(corpus: str) -> list:
    return ["row_id"] if corpus == "agnews" else ["repo", "title", "body"]


def clean_test_mask(corpus: str, test_df: pd.DataFrame) -> "pd.Series":
    """True for final-test rows that were NOT used during development."""
    dev = _dev_frame(corpus)
    keys = _key(corpus)
    for col in keys:
        if col not in test_df.columns:
            raise KeyError(f"{corpus}: final test split has no column {col!r}")
    dev_keys = set(map(tuple, dev[keys].astype(str).to_numpy()))
    test_keys = list(map(tuple, test_df[keys].astype(str).to_numpy()))
    return pd.Series([k not in dev_keys for k in test_keys], index=test_df.index)


def summary() -> None:
    for corpus in ("github", "agnews"):
        path = config.DATA_DIR / ("github_issues" if corpus == "github"
                                  else "agnews") / "test.csv"
        test_df = pd.read_csv(path)
        mask = clean_test_mask(corpus, test_df)
        print(f"{corpus:8s} test={len(test_df)} "
              f"dev_overlap={int((~mask).sum())} clean={int(mask.sum())}")


if __name__ == "__main__":
    summary()
