# -*- coding: utf-8 -*-
"""Ten-seed paired summary of the main read-outs and the occlusion protocol.

2026-10-11 review: the inputs are validated before any statistic is computed —
each corpus must cover the ten seeds for both read-outs, every requested
occlusion curve (FC-NTM and BTM, plus LDA on AG News where it is the best
3-seed baseline) must have all five rates with finite NLL, and files produced
before the identity sidecar existed are reported explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import occlusion_robustness as O
import result_identity as RI
from seed_identity import fair_readout_identity, occlusion_identity

OUT = Path(__file__).resolve().parent / "outputs"
RERUN = OUT / "rerun"
SEEDS = [11, 23, 37, 53, 71, 89, 101, 113, 127, 139]
NEW_SEEDS = [53, 71, 89, 101, 113, 127, 139]
K = 30
RATES = {0.0, 0.25, 0.5, 0.75, 1.0}


def fc_rows(corpus):
    parts = [pd.read_csv(OUT / f"fcntm_{corpus}_final.csv")]
    extra = RERUN / f"fcntm_{corpus}_seed53plus.csv"
    if extra.exists():
        parts.append(pd.read_csv(extra))
    frame = pd.concat(parts, ignore_index=True)
    return frame.drop_duplicates(subset=["model", "seed"], keep="first")


def btm_rows(corpus):
    parts = [pd.read_csv(OUT / f"fair_readout_{corpus}_3seed.csv")]
    extra = RERUN / f"fair_readout_{corpus}_newseeds.csv"
    if extra.exists():
        parts.append(pd.read_csv(extra))
    frame = pd.concat(parts, ignore_index=True)
    frame = frame[frame["model"] == "BTM"]
    return frame.drop_duplicates(subset=["readout", "seed"], keep="first")


def load_curves(corpus):
    """seed -> frame with all archived and new-seed occlusion rows."""
    curves = {}
    for seed in SEEDS:
        path = OUT / f"occlusion_robustness_{corpus}_seed{seed}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["seed"] = seed       # the archived per-seed files omit it
            curves[seed] = frame
    extra = RERUN / f"occlusion_robustness_{corpus}_newseeds.csv"
    if extra.exists():
        frame = pd.read_csv(extra)
        for seed, g in frame.groupby("seed"):
            curves[int(seed)] = g
    return curves


def rel_degradation(frame):
    """Relative degradation per model in a frame (files may hold several models)."""
    out = {}
    for model, g in frame[frame["rate"] > 0].groupby("model"):
        g = g.set_index("rate").sort_index()
        if 1.0 in g.index and 0.25 in g.index:
            out[str(model)] = float(g.loc[1.0, "hidden_NLL"]
                                    / g.loc[0.25, "hidden_NLL"])
    return out


def paired_stats(delta):
    delta = np.asarray(delta, dtype=float)
    n = delta.size
    mean, sd = float(delta.mean()), float(delta.std(ddof=1))
    se = sd / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
    pos = int((delta > 0).sum())
    sign_p = stats.binomtest(pos, n, 0.5).pvalue if n else float("nan")
    try:
        wil = stats.wilcoxon(delta).pvalue if np.any(delta != 0) else float("nan")
    except ValueError:
        wil = float("nan")
    return dict(n=n, mean=mean, sd=sd, ci_low=float(lo), ci_high=float(hi),
                positive=pos, sign_p=float(sign_p), wilcoxon_p=float(wil))


def verify_sidecars(corpus, state_dir="") -> list:
    """The new-seed files must be pinned to the current evaluation protocol.

    2026-10-12 review, item C4: checking completeness is not enough - the
    sidecars have to prove that every row comes from the same test split,
    mask protocol and checkpoint as the rest of the summary.
    """
    train, test = O.load_corpus(corpus)
    want = {
        f"fair_readout_{corpus}_newseeds.csv":
            fair_readout_identity(corpus, train, test, NEW_SEEDS, K),
        f"occlusion_robustness_{corpus}_newseeds.csv":
            occlusion_identity(
                corpus, train, test, NEW_SEEDS, K,
                checkpoint=(Path(state_dir) / f"fcntm_main_state_{corpus}.pt")
                if state_dir else None),
    }
    problems = []
    for name, ident in want.items():
        path = RERUN / name
        if not path.exists():
            problems.append(f"{name}: missing")
        elif RI.stored(path) is None:
            problems.append(f"{name}: no identity sidecar")
        elif not RI.matches(path, ident):
            problems.append(f"{name}: {RI.why_not(path, ident)}")
    return problems


def validate(corpus, fc, btm, curves):
    problems = []
    for model in ("FC-NTM", "FC-NTM-cTFIDF"):
        g = fc[fc["model"] == model]
        seeds = sorted(int(s) for s in g["seed"])
        missing = sorted(set(SEEDS) - set(seeds))
        if missing:
            problems.append(f"{corpus} {model}: missing seeds {missing}")
        if not np.all(np.isfinite(g["CV"])):
            problems.append(f"{corpus} {model}: non-finite CV")
        if "K_used" in g.columns and not np.all(g["K_used"] == 30):
            problems.append(f"{corpus} {model}: K_used != 30")
    for readout in ("beta", "ctfidf"):
        g = btm[btm["readout"] == readout]
        seeds = sorted(int(s) for s in g["seed"])
        missing = sorted(set(SEEDS) - set(seeds))
        if missing:
            problems.append(f"{corpus} BTM/{readout}: missing seeds {missing}")
        if not np.all(np.isfinite(g["CV"])):
            problems.append(f"{corpus} BTM/{readout}: non-finite CV")
    needed = ["FC-NTM", "BTM"] + (["LDA"] if corpus == "agnews" else [])
    for seed in SEEDS:
        frame = curves.get(seed)
        if frame is None:
            problems.append(f"{corpus} seed {seed}: no occlusion curves")
            continue
        for model in needed:
            g = frame[frame["model"] == model]
            if g.empty:
                problems.append(f"{corpus} seed {seed}: missing {model}")
                continue
            rates = {round(float(r), 2) for r in g["rate"]}
            if not RATES <= rates:
                problems.append(f"{corpus} seed {seed} {model}: rates "
                                f"{sorted(rates)}")
            scored = g[g["rate"] > 0]
            if not np.all(np.isfinite(scored["hidden_NLL"])):
                problems.append(f"{corpus} seed {seed} {model}: non-finite NLL")
    return problems


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="",
                    help="directory of the FC-NTM states the new-seed curves "
                         "were scored from (default: outputs/rerun)")
    args = ap.parse_args()
    data, problems = {}, []
    for corpus in ("github", "agnews"):
        fc = fc_rows(corpus)
        btm = btm_rows(corpus)
        curves = load_curves(corpus)
        data[corpus] = (fc, btm, curves)
        problems.extend(validate(corpus, fc, btm, curves))
        problems.extend(verify_sidecars(corpus, args.state_dir))
    if problems:
        print("input validation failed:", flush=True)
        for problem in problems:
            print("  " + problem, flush=True)
        return 2

    rows = []
    for corpus, (fc, btm, curves) in data.items():
        for fc_model, readout, name in (("FC-NTM-cTFIDF", "ctfidf", "main"),
                                        ("FC-NTM", "beta", "beta")):
            a = fc[fc["model"] == fc_model].set_index("seed")["CV"]
            b = btm[btm["readout"] == readout].set_index("seed")["CV"]
            common = sorted(set(a.index) & set(b.index))
            st = paired_stats((a.loc[common] - b.loc[common]).to_numpy())
            rows.append(dict(corpus=corpus, comparison=name, metric="CV",
                             n_seeds=len(common), **st))
        rel = {seed: rel_degradation(curves[seed]) for seed in SEEDS
               if seed in curves}
        common = [s for s in SEEDS if s in rel
                  and "FC-NTM" in rel[s] and "BTM" in rel[s]]
        st = paired_stats([rel[s]["BTM"] - rel[s]["FC-NTM"] for s in common])
        rows.append(dict(corpus=corpus, comparison="occlusion_vs_BTM",
                         metric="rel", n_seeds=len(common), **st))
        best_delta, winners = [], []
        for seed in common:
            baselines = {m: v for m, v in rel[seed].items()
                         if m != "FC-NTM"}
            if not baselines:
                continue
            winner = min(baselines, key=baselines.get)
            best_delta.append(baselines[winner] - rel[seed]["FC-NTM"])
            winners.append(winner)
        st = paired_stats(best_delta)
        rows.append(dict(corpus=corpus, comparison="occlusion_vs_best",
                         metric="rel", n_seeds=len(best_delta), **st))
        print(f"{corpus}: best baseline per seed "
              f"{ {w: winners.count(w) for w in set(winners)} }", flush=True)
    table = pd.DataFrame(rows)
    path = RERUN / "seed_extension_summary.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print("\nsaved", path)
    show = table.copy()
    for column in ("mean", "sd", "ci_low", "ci_high", "sign_p", "wilcoxon_p"):
        show[column] = show[column].round(4)
    print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())