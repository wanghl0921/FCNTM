# -*- coding: utf-8 -*-
"""Fold re-run rows back into a benchmark runs CSV and refresh the summary.

Usage:
    python merge_benchmark_rows.py runs.csv new1.csv [new2.csv ...]
        [--summary public_benchmark_summary_x.csv]

Rows are matched on (model, seed, K_target); a row from the new files replaces
the old one *entirely* (so a stale ``error`` string is cleared when a re-run
succeeds), and keys that are not present yet are appended.  Rows whose new
result is missing or failed never overwrite an existing good row.  The summary
is recomputed with the same function the benchmark itself uses, so the two
stay consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import benchmark_public as BP


def main() -> None:
    argv = sys.argv[1:]
    summary_override = None
    if "--summary" in argv:
        idx = argv.index("--summary")
        summary_override = argv[idx + 1]
        del argv[idx:idx + 2]
    if len(argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    main_path = Path(argv[0])
    extra_paths = [Path(p) for p in argv[1:]]
    frame = pd.read_csv(main_path)
    before = len(frame)
    for path in extra_paths:
        extra = pd.read_csv(path)
        for _, row in extra.iterrows():
            record = row.to_dict()
            if not BP._row_is_complete(record):
                print(f"skip incomplete/failed row: "
                      f"{row.get('model')} seed {row.get('seed')}")
                continue
            mask = (frame["model"] == row["model"]) & (
                frame["seed"] == int(row["seed"]))
            if ("K_target" in frame.columns and "K_target" in extra.columns
                    and pd.notna(row.get("K_target"))):
                mask &= (frame["K_target"] == row["K_target"])
            if mask.any():
                # replace the whole row: a column that the new run does not
                # carry (e.g. "error") must not keep its old value
                for column in frame.columns:
                    frame.loc[mask, column] = pd.NA
                for column in extra.columns:
                    if column not in frame.columns:
                        frame[column] = pd.NA
                    frame.loc[mask, column] = row[column]
            else:
                frame = pd.concat([frame, row.to_frame().T], ignore_index=True)
    key = [c for c in ("model", "seed", "K_target") if c in frame.columns]
    frame = frame.drop_duplicates(subset=key, keep="last")
    frame.to_csv(main_path, index=False, encoding="utf-8-sig")
    summary = BP.summarize_rows(frame.to_dict("records"))
    if summary_override:
        summary_path = Path(summary_override)
        if not summary_path.is_absolute():
            summary_path = Path.cwd() / summary_path
    else:
        tag = main_path.stem.replace("public_benchmark_runs_", "")
        summary_path = main_path.parent / f"public_benchmark_summary_{tag}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"{before} -> {len(frame)} rows; summary -> {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
