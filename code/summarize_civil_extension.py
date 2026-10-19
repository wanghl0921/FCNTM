# -*- coding: utf-8 -*-
"""Ten-seed paired summary for the civil-engineering corpus.

Mirrors ``summarize_seed_extension.py`` for the third corpus: FC-NTM and
BTM are extended to the ten seeds, both read-outs are compared, and the
partial-occlusion curves are summarised seed by seed.  Writes
``outputs/rerun/civil_seed_extension_summary.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import summarize_seed_extension as SSE

OUT = Path(__file__).resolve().parent / "outputs"
RERUN = OUT / "rerun"
SEEDS = SSE.SEEDS
RATES = SSE.RATES


def _conflict_rows(paths, keys, label, problems) -> None:
    """Reject same-key rows with differing metrics before de-duplication.

    2026-10-15 review: the duplicate check ran after ``fc_rows``/``btm_rows``
    had already dropped duplicates, so conflicting rows could pass silently.
    """
    frames = []
    for p in paths:
        p = Path(p)
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return
    frame = pd.concat(frames, ignore_index=True)
    dup = frame[frame.duplicated(subset=keys, keep=False)]
    if dup.empty:
        return
    for key, group in dup.groupby(keys):
        if (group["CV"].nunique(dropna=False) > 1
                or group["NPMI"].nunique(dropna=False) > 1):
            problems.append(f"{label}: conflicting duplicate rows for {key}")


def gather():
    fc = SSE.fc_rows("civil")
    btm = SSE.btm_rows("civil")
    curves = SSE.load_curves("civil")
    problems = []
    # 2026-10-14 review C5: reuse the full sidecar identity checks and add
    # the finite-value / duplicate-key guards this entry point was missing
    problems.extend(SSE.verify_sidecars("civil"))
    # 2026-10-15 review: conflict check on the raw sources, before the
    # summariser's de-duplication can hide a conflicting same-key row
    _conflict_rows([OUT / "fcntm_civil_final.csv",
                    RERUN / "fcntm_civil_seed53plus.csv"],
                   ["model", "seed"], "FC-NTM sources", problems)
    _conflict_rows([OUT / "fair_readout_civil_3seed.csv",
                    RERUN / "fair_readout_civil_newseeds.csv"],
                   ["model", "seed", "readout"], "BTM sources", problems)
    for model in ("FC-NTM", "FC-NTM-cTFIDF"):
        g = fc[fc["model"] == model]
        missing = sorted(set(SEEDS) - set(int(s) for s in g["seed"]))
        if missing:
            problems.append(f"FC-NTM {model}: missing seeds {missing}")
    for readout in ("beta", "ctfidf"):
        g = btm[btm["readout"] == readout]
        missing = sorted(set(SEEDS) - set(int(s) for s in g["seed"]))
        if missing:
            problems.append(f"BTM/{readout}: missing seeds {missing}")
        if not np.isfinite(g["CV"].to_numpy()).all():
            problems.append(f"BTM/{readout}: non-finite CV")
    if not np.isfinite(fc["CV"].to_numpy()).all():
        problems.append("FC-NTM rows: non-finite CV")
    if fc.duplicated(subset=["model", "seed"]).any():
        problems.append("FC-NTM rows: duplicate (model, seed) keys")
    if btm.duplicated(subset=["model", "seed", "readout"]).any():
        problems.append("BTM rows: duplicate (model, seed, readout) keys")
    for seed in SEEDS:
        frame = curves.get(seed)
        if frame is None:
            problems.append(f"seed {seed}: no occlusion curves")
            continue
        for model in ("FC-NTM", "BTM"):
            g = frame[frame["model"] == model]
            if g.empty:
                problems.append(f"seed {seed}: missing {model} curve")
                continue
            rates = {round(float(r), 2) for r in g["rate"]}
            if not RATES <= rates:
                problems.append(f"seed {seed} {model}: rates {sorted(rates)}")
    return fc, btm, curves, problems


def main() -> int:
    fc, btm, curves, problems = gather()
    if problems:
        print("input validation failed:", flush=True)
        for problem in problems:
            print("  " + problem, flush=True)
        return 2

    rows = []
    for fc_model, readout, name in (("FC-NTM-cTFIDF", "ctfidf", "ctfidf"),
                                    ("FC-NTM", "beta", "beta")):
        a = fc[fc["model"] == fc_model].set_index("seed")["CV"]
        b = btm[btm["readout"] == readout].set_index("seed")["CV"]
        common = sorted(set(a.index) & set(b.index))
        st = SSE.paired_stats((a.loc[common] - b.loc[common]).to_numpy())
        rows.append(dict(corpus="civil", comparison=name, metric="CV",
                         n_seeds=len(common), **st))
    rel = {seed: SSE.rel_degradation(curves[seed]) for seed in SEEDS
           if seed in curves}
    common = [s for s in SEEDS if s in rel
              and "FC-NTM" in rel[s] and "BTM" in rel[s]]
    st = SSE.paired_stats([rel[s]["BTM"] - rel[s]["FC-NTM"] for s in common])
    rows.append(dict(corpus="civil", comparison="occlusion_vs_BTM",
                     metric="rel", n_seeds=len(common), **st))

    table = pd.DataFrame(rows)
    path = RERUN / "civil_seed_extension_summary.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print("saved", path)
    show = table.copy()
    for column in ("mean", "sd", "ci_low", "ci_high", "sign_p", "wilcoxon_p"):
        show[column] = show[column].round(4)
    print(show.to_string(index=False))

    for fc_model, readout in (("FC-NTM-cTFIDF", "ctfidf"),
                              ("FC-NTM", "beta")):
        a = fc[fc["model"] == fc_model]
        print(f"{fc_model}: CV {a['CV'].mean():.3f}+-{a['CV'].std(ddof=1):.3f} "
              f"NPMI {a['NPMI'].mean():.3f}+-{a['NPMI'].std(ddof=1):.3f} "
              f"(n={len(a)})")
        b = btm[btm["readout"] == readout]
        print(f"BTM/{readout}: CV {b['CV'].mean():.3f}+-{b['CV'].std(ddof=1):.3f} "
              f"NPMI {b['NPMI'].mean():.3f}+-{b['NPMI'].std(ddof=1):.3f} "
              f"(n={len(b)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
