# -*- coding: utf-8 -*-
"""Regenerate outputs/baseline_fieldview.csv with per-row identities.

2026-10-12 review, item C4: the archived field-view CSV predates the per-row
``eval_id``/``fit_id`` columns and the identity-aware merge.  This script
re-runs the field-view evaluation into a temporary directory -- reusing the
archived fit matrices where they exist (``--adopt-legacy-fits``, recorded as
``legacy:<digest>``) and refitting the rows that were never cached (civil BTM
and NMF) -- then compares every number with the archived CSV and only replaces
the archive when the comparison is exact at the stored precision.

    python repin_fieldview_archive.py            # run and (on exact match) swap
    python repin_fieldview_archive.py --dry-run  # compare only, never swap

The fit caches written while refitting (``outputs/rerun/fieldview_fit_*.npz``)
stay in place either way; they carry the current fit identity.  This wrapper is
deliberately *not* part of ``result_identity.CODE_FILES``: the identity of the
CSV is defined by ``baseline_fieldview.py`` (tracked) and the corpus protocol
in ``seed_identity.py`` (tracked).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import baseline_fieldview as BF  # noqa: E402

ARCHIVE = BF.OUT / "baseline_fieldview.csv"
REPORT = BF.OUT / "rerun" / "fieldview_regen_report.json"
BACKUP = BF.OUT / "backup_baseline_fieldview_pre_identity.csv"
KEYS = ["dataset", "model", "seed", "view", "K"]
VALUES = ["n_topics", "CV", "CV_n", "NPMI", "NPMI_n"]

RUNS = (
    ("github+agnews LDA/NMF/BTM",
     ["--models", "LDA,NMF,BTM", "--corpora", "github,agnews",
      "--seeds", "11,23,37", "--k", "30", "--adopt-legacy-fits"]),
    ("ProdLDA all corpora",
     ["--models", "ProdLDA", "--corpora", "github,agnews,civil",
      "--seeds", "11,23,37", "--k", "30", "--adopt-legacy-fits"]),
    ("civil LDA/NMF/BTM (refit)",
     ["--models", "LDA,NMF,BTM", "--corpora", "civil",
      "--seeds", "11,23,37", "--k", "30", "--adopt-legacy-fits"]),
)


def compare(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    g = old.merge(new, on=KEYS, how="outer", suffixes=("_old", "_new"),
                  indicator=True)
    unmatched = int((g["_merge"] != "both").sum())
    worst, per = 0.0, {}
    for column in VALUES:
        a = pd.to_numeric(g[column + "_old"], errors="coerce")
        b = pd.to_numeric(g[column + "_new"], errors="coerce")
        if bool(a.isna().ne(b.isna()).any()):
            return {"unmatched_keys": unmatched, "nan_mismatch": column,
                    "max_delta": None}
        ok = a.notna() & b.notna()
        delta = float((a[ok] - b[ok]).abs().max()) if bool(ok.any()) else 0.0
        per[column] = delta
        worst = max(worst, delta)
    return {"unmatched_keys": unmatched, "max_delta": worst, "per_column": per,
            "rows_old": int(len(old)), "rows_new": int(len(new))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--atol", type=float, default=1e-9)
    args = ap.parse_args()

    old = pd.read_csv(ARCHIVE)
    tmp = Path(tempfile.mkdtemp(prefix="fieldview_regen_"))
    report = {"started_utc": datetime.now(timezone.utc).isoformat(
        timespec="seconds"), "tmp": str(tmp), "runs": []}
    print(f"regenerating into {tmp}", flush=True)
    saved_out = BF.OUT
    try:
        for tag, argv in RUNS:
            BF.OUT = tmp
            sys.argv = ["baseline_fieldview.py"] + argv
            t0 = time.time()
            rc = BF.main()
            report["runs"].append(
                {"tag": tag, "rc": rc, "seconds": round(time.time() - t0, 1)})
            if rc not in (0, None):
                raise SystemExit(f"{tag}: main() returned {rc}")
    finally:
        BF.OUT = saved_out

    new = pd.read_csv(tmp / "baseline_fieldview.csv")
    result = compare(old, new)
    report["comparison"] = result
    report["new_columns"] = [c for c in new.columns if c not in old.columns]
    exact = (result.get("unmatched_keys") == 0
             and result.get("nan_mismatch") is None
             and (result.get("max_delta") or 0.0) <= args.atol)
    report["replaced"] = False
    if exact and not args.dry_run:
        if not BACKUP.exists():
            shutil.copy2(ARCHIVE, BACKUP)
            print("archived previous CSV to", BACKUP.name, flush=True)
        shutil.copy2(tmp / "baseline_fieldview.csv", ARCHIVE)
        shutil.copy2(tmp / "baseline_fieldview.csv.identity.json",
                     str(ARCHIVE) + ".identity.json")
        report["replaced"] = True
        report["backup"] = BACKUP.name
        print("archive replaced with the identity-carrying CSV", flush=True)
    elif not exact:
        report["kept"] = "numbers differ; archive left untouched"
        print("numbers differ; archive left untouched", flush=True)
    else:
        report["kept"] = "dry run"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    REPORT.write_text(json.dumps(report, indent=1, ensure_ascii=False),
                      encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)
    return 0 if exact else 3


if __name__ == "__main__":
    raise SystemExit(main())
