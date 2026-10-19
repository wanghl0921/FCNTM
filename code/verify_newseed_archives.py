# -*- coding: utf-8 -*-
"""Re-score the archived new-seed rows and pin them to the evaluation identity.

2026-10-12 review, item C4: ``fair_readout_*_newseeds.csv`` and
``occlusion_robustness_*_newseeds.csv`` were written before the evaluation
identity existed, so nothing stopped a later run from merging rows that come
from a different test split or a different checkpoint.  This script

  1. recomputes the rows of one anchor seed (default 53) for every model in
     the archived files - FC-NTM from the delivered checkpoint, BTM from the
     archived Gibbs counts, LDA by an independent fit - under the *current*
     evaluation protocol,
  2. compares them with the archived rows and aborts on any mismatch above
     ``--atol`` (default 1e-6), printing the maximum deviation,
  3. only then rewrites the file: the recomputed anchor rows replace the
     archived anchor rows, the other seeds stay untouched, the previous file
     is backed up next to it, and the identity sidecar is written.

    python verify_newseed_archives.py --corpora github,agnews
    python verify_newseed_archives.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import occlusion_robustness as O
import result_identity as RI
from add_btm_doc_topic import masks_for
from fair_readout import evaluate_both
from seed_identity import (fair_readout_identity, occlusion_identity,
                           state_path, verify_checkpoint)

FIT = O.OUT_DIR / "rerun"
SEEDS = [53, 71, 89, 101, 113, 127, 139]
RATES = [0.0, 0.25, 0.5, 0.75, 1.0]
OCC_KEYS = ["model", "seed", "rate"]
FAIR_KEYS = ["model", "seed", "readout"]


def archived_btm(corpus: str, seed: int) -> dict:
    """The archive holds the fitted Gibbs counts, not the arrays we refit."""
    path = FIT / f"btm_fit_{corpus}_seed{seed}.npz"
    z = np.load(path, allow_pickle=True)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k])
            for k in z.files if k not in ("identity_id", "identity_json")}


def rescore_occlusion(corpus: str, train, test, seed: int, K: int):
    word2id = test["dictionary"].token2id
    visible, hidden = masks_for(test, seed)
    rows = []
    adapter = O.FCNTMAdapter(state_path(corpus), len(test["vocab"]), K,
                             seed=seed)
    rows += O.run_model(adapter, visible, test["item_tokens"], hidden,
                        RATES, word2id, "FC-NTM")
    del adapter
    btm = O.BTMAdapter(train, K, seed, rec=archived_btm(corpus, seed))
    btm.dictionary = train["dictionary"]
    rows += O.run_model(btm, visible, test["item_tokens"], hidden, RATES,
                        word2id, "BTM")
    del btm
    lda = O.LDAAdapter(train, K, seed)
    lda.dictionary = test["dictionary"]
    rows += O.run_model(lda, visible, test["item_tokens"], hidden, RATES,
                        word2id, "LDA")
    del lda
    for row in rows:
        row["corpus"] = corpus
        row["seed"] = seed
    return pd.DataFrame(rows)


def rescore_fair(corpus: str, train, test, seed: int):
    doc_counts = (train["reason_bow"] + train["item_bow"]).astype(float)
    rec = archived_btm(corpus, seed)
    rows = evaluate_both(rec["beta"], rec["theta"], test["vocab"], doc_counts,
                         train, test, "BTM", seed, "fair_readout")
    return pd.DataFrame(rows)


def compare(archived: pd.DataFrame, fresh: pd.DataFrame, keys, values,
            label: str, atol: float) -> float:
    both = archived.merge(fresh, on=keys, suffixes=("_old", "_new"))
    if len(both) != len(fresh) or len(both) != len(archived):
        raise SystemExit(f"{label}: {len(archived)} archived rows vs "
                         f"{len(fresh)} recomputed rows, {len(both)} matched")
    worst = 0.0
    for column in values:
        a = pd.to_numeric(both[column + "_old"], errors="coerce")
        b = pd.to_numeric(both[column + "_new"], errors="coerce")
        if bool(a.isna().ne(b.isna()).any()):
            raise SystemExit(f"{label}: {column} NaN pattern differs")
        both_ok = a.notna() & b.notna()
        if bool(both_ok.any()):
            delta = float((a[both_ok] - b[both_ok]).abs().max())
            worst = max(worst, delta)
    print(f"  {label}: {len(both)} rows, max |delta| = {worst:.3e}",
          flush=True)
    if worst > atol:
        raise SystemExit(f"{label}: archived rows do not reproduce "
                         f"(max |delta| = {worst:.3e} > {atol:g}); "
                         f"nothing was rewritten")
    return worst


def pin(path: Path, ident: dict, fresh: pd.DataFrame, keys) -> None:
    old = pd.read_csv(path)
    backup = path.with_name(f"backup_{path.stem}_pre_pin{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  previous rows backed up to {backup.name}", flush=True)
    frame = pd.concat([old[~old["seed"].isin(set(fresh["seed"]))], fresh],
                      ignore_index=True)
    frame = frame.drop_duplicates(subset=keys, keep="last")
    missing = [c for c in keys if frame[c].isna().any()]
    if missing:
        raise SystemExit(f"{path.name}: empty key column {missing}")
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    RI.write(path, ident)
    print(f"  pinned {path.name} ({len(frame)} rows)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="github,agnews")
    ap.add_argument("--anchor", type=int, default=53)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.anchor not in SEEDS:
        raise SystemExit(f"anchor {args.anchor} is not one of {SEEDS}")
    for corpus in [c for c in args.corpora.split(",") if c]:
        train, test = O.load_corpus(corpus)
        verify_checkpoint(state_path(corpus), train, [args.anchor], args.k)
        print(f"[{corpus}] re-scoring anchor seed {args.anchor}", flush=True)
        occ_path = FIT / f"occlusion_robustness_{corpus}_newseeds.csv"
        old = pd.read_csv(occ_path)
        fresh = rescore_occlusion(corpus, train, test, args.anchor, args.k)
        compare(old[old["seed"] == args.anchor], fresh, OCC_KEYS,
                ["hidden_NLL", "n_docs", "agreement", "mean_js"],
                f"{corpus} occlusion seed {args.anchor}", args.atol)
        fair_path = FIT / f"fair_readout_{corpus}_newseeds.csv"
        old_fair = pd.read_csv(fair_path)
        fresh_fair = rescore_fair(corpus, train, test, args.anchor)
        compare(old_fair[old_fair["seed"] == args.anchor], fresh_fair,
                FAIR_KEYS, ["CV", "NPMI", "UMass"],
                f"{corpus} fair read-out seed {args.anchor}", args.atol)
        if args.dry_run:
            print(f"[{corpus}] dry run: sidecars not written", flush=True)
            continue
        fresh["corpus"] = corpus
        fresh["seed"] = args.anchor
        pin(occ_path,
            occlusion_identity(corpus, train, test, SEEDS, args.k,
                               checkpoint=state_path(corpus)),
            fresh, OCC_KEYS)
        pin(fair_path,
            fair_readout_identity(corpus, train, test, SEEDS, args.k),
            fresh_fair, FAIR_KEYS)
        del train, test
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
