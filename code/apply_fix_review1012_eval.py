# -*- coding: utf-8 -*-
"""2026-10-12 review: complete the evaluation identity (C4) and the manifest (C5).

Patches, all anchored:
  result_identity.py          -> CODE_FILES covers the new entry points
  baseline_fieldview.py       -> per-row eval/fit identity, identity-aware merge
  summarize_seed_extension.py -> the summariser verifies the sidecars
  results_manifest.py         -> split comment/path, track seed_identity.py
  tests_review_fixes.py       -> regression test for the evaluation identity
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
fails = []


def patch(name, pairs):
    path = ROOT / name
    raw = path.read_text(encoding="utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n") if crlf else raw
    for tag, old, new, count in pairs:
        n = text.count(old)
        if n != count:
            fails.append(f"{name}: {tag}: found {n}, want {count}")
            continue
        text = text.replace(old, new)
    if not any(f.startswith(name) for f in fails):
        out = text.replace("\n", "\r\n") if crlf else text
        path.write_text(out, encoding="utf-8", newline="")
        print(f"  patched {name}")


patch("result_identity.py", [
    ("code files",
     '''    "add_civil_btm_occlusion.py",
    "occlusion_controls.py",
)''',
     '''    "add_civil_btm_occlusion.py",
    "occlusion_controls.py",
    # seed-extension entries and the module that pins their *evaluation*
    # inputs (2026-10-12 review, item C4): a change in any of these files has
    # to move the identity of the caches and merged CSVs they write
    "seed_identity.py",
    "baseline_fieldview.py",
    "extend_seeds_btm.py",
    "extend_seeds_fcntm_occlusion.py",
    "extend_seeds_lda_occlusion.py",
    "verify_newseed_archives.py",
)''',
     1),
])

patch("baseline_fieldview.py", [
    ("imports",
     '''import metrics as M
import occlusion_robustness as O
import run_public_missing_field as RMF
from config import TOPN
from seed_identity import fit_identity, load_cached_npz, save_cached_npz''',
     '''import metrics as M
import occlusion_robustness as O
import result_identity as RI
import run_public_missing_field as RMF
from config import TOPN
from seed_identity import (fieldview_identity, fit_identity,
                           load_cached_npz, save_cached_npz)''',
     1),
    ("docstring",
     '''Every cached fit carries a full fitting identity; a cache that does not match
is refit instead of adopted (shared BTM/NMF fit files are never overwritten:
a separate field-view cache is written next to them).''',
     '''Every cached fit carries a full fitting identity; a cache that does not match
is refit instead of adopted (shared BTM/NMF fit files are never overwritten:
a separate field-view cache is written next to them).

Each row also records the *evaluation* identity of its corpus protocol and the
identity of the fit it was scored from (2026-10-12 review, item C4).  Fits
that predate the identity scheme are only used with ``--adopt-legacy-fits``,
in which case the row keeps a content digest (``legacy:<digest>``) instead of
a run identity.''',
     1),
    ("beta function",
     '''def baseline_beta(model: str, corpus: str, seed: int, K: int, train, doc_bow):
    """Topic-word matrix of one baseline (identity-checked cache)."""
    ident = fit_identity(f"baseline_fieldview.{model}", corpus, train, K, seed)
    shape = {"beta": (K, len(train["vocab"]))}
    if model == "LDA":
        return np.asarray(O.LDAAdapter(train, K, seed).beta, dtype=float)
    if model in ("NMF", "BTM"):
        shared = FIT / f"{model.lower()}_fit_{corpus}_seed{seed}.npz"
        cached = load_cached_npz(shared, ident, ("beta",), shape)
        if cached is not None:
            return np.asarray(cached["beta"], dtype=float)
        cache = FIT / f"fieldview_fit_{model}_{corpus}_seed{seed}.npz"
        cached = load_cached_npz(cache, ident, ("beta",), shape)
        if cached is not None:
            return np.asarray(cached["beta"], dtype=float)
        adapter = O.NMFAdapter(train, K, seed) if model == "NMF" \\
            else O.BTMAdapter(train, K, seed)
        beta = np.asarray(adapter.beta, dtype=float)
        save_cached_npz(cache, ident, beta=beta)
        return beta
    if model == "ProdLDA":
        cache = FIT / f"fieldview_fit_ProdLDA_{corpus}_seed{seed}.npz"
        cached = load_cached_npz(cache, ident, ("beta",), shape)
        if cached is not None:
            return np.asarray(cached["beta"], dtype=float)
        adapter = O.ProdLDAAdapter(train, K, seed, doc_bow)
        beta = np.asarray(adapter.model.get_beta(), dtype=float)
        save_cached_npz(cache, ident, beta=beta)
        return beta
    raise ValueError(model)''',
     '''def legacy_beta(path, K: int, n_vocab: int, adopt: bool):
    """Read an unpinned legacy fit and record its content digest.

    Fits written before the identity scheme existed carry no ``identity_id``;
    they are never adopted silently.  With ``adopt`` the matrix is read and
    every row keeps ``legacy:<digest>`` for it, so the exact numbers still
    have a content-addressed provenance (2026-10-12 review, item C4).
    """
    if not (adopt and path.exists()):
        return None
    raw = np.load(path, allow_pickle=True)
    if "beta" not in raw.files:
        return None
    beta = raw["beta"]
    beta = np.asarray(beta.item() if beta.ndim == 0 else beta, dtype=float)
    if beta.shape != (K, n_vocab):
        return None
    print(f"  adopting unpinned legacy fit {path.name}", flush=True)
    return beta, "legacy:" + RI.array_digest(beta)


def baseline_beta(model: str, corpus: str, seed: int, K: int, train, doc_bow,
                  adopt_legacy: bool = False):
    """``(beta, fit_id)`` of one baseline (identity-checked cache)."""
    ident = fit_identity(f"baseline_fieldview.{model}", corpus, train, K, seed)
    shape = {"beta": (K, len(train["vocab"]))}
    if model == "LDA":
        return np.asarray(O.LDAAdapter(train, K, seed).beta,
                          dtype=float), ident["id"]
    if model in ("NMF", "BTM"):
        shared = FIT / f"{model.lower()}_fit_{corpus}_seed{seed}.npz"
        cached = load_cached_npz(shared, ident, ("beta",), shape)
        if cached is not None:
            return np.asarray(cached["beta"], dtype=float), ident["id"]
        cache = FIT / f"fieldview_fit_{model}_{corpus}_seed{seed}.npz"
        cached = load_cached_npz(cache, ident, ("beta",), shape)
        if cached is not None:
            return np.asarray(cached["beta"], dtype=float), ident["id"]
        legacy = legacy_beta(shared, K, len(train["vocab"]), adopt_legacy)
        if legacy is not None:
            return legacy
        adapter = O.NMFAdapter(train, K, seed) if model == "NMF" \\
            else O.BTMAdapter(train, K, seed)
        beta = np.asarray(adapter.beta, dtype=float)
        save_cached_npz(cache, ident, beta=beta)
        return beta, ident["id"]
    if model == "ProdLDA":
        cache = FIT / f"fieldview_fit_ProdLDA_{corpus}_seed{seed}.npz"
        cached = load_cached_npz(cache, ident, ("beta",), shape)
        if cached is not None:
            return np.asarray(cached["beta"], dtype=float), ident["id"]
        legacy = legacy_beta(cache, K, len(train["vocab"]), adopt_legacy)
        if legacy is not None:
            return legacy
        adapter = O.ProdLDAAdapter(train, K, seed, doc_bow)
        beta = np.asarray(adapter.model.get_beta(), dtype=float)
        save_cached_npz(cache, ident, beta=beta)
        return beta, ident["id"]
    raise ValueError(model)''',
     1),
    ("flag",
     '''    ap.add_argument("--k", type=int, default=30)
    args = ap.parse_args()

    rows = []
    seeds = [int(s) for s in args.seeds.split(",")]
    for corpus in args.corpora.split(","):''',
     '''    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--adopt-legacy-fits", action="store_true",
                    help="use archived fits that carry no identity sidecar, "
                         "recording their content digest in every row")
    args = ap.parse_args()

    rows = []
    eval_ids = {}
    seeds = [int(s) for s in args.seeds.split(",")]
    models = [m for m in args.models.split(",") if m]
    for corpus in [c for c in args.corpora.split(",") if c]:''',
     1),
    ("eval identity",
     '''        doc_bow = torch.from_numpy(counts)
        for model in [m for m in args.models.split(",") if m]:
            for seed in seeds:
                beta = np.asarray(baseline_beta(model, corpus, seed, args.k,
                                                train, doc_bow), dtype=float)''',
     '''        doc_bow = torch.from_numpy(counts)
        eval_ident = fieldview_identity(corpus, train, test, seeds, args.k)
        eval_ids[corpus] = eval_ident["id"]
        for model in models:
            for seed in seeds:
                beta, fit_id = baseline_beta(
                    model, corpus, seed, args.k, train, doc_bow,
                    args.adopt_legacy_fits)
                beta = np.asarray(beta, dtype=float)''',
     1),
    ("row identity",
     '''                    rows.append({
                        "dataset": corpus, "model": model, "seed": seed,
                        "K": int(beta.shape[0]), "view": view,''',
     '''                    rows.append({
                        "dataset": corpus, "model": model, "seed": seed,
                        "K": int(beta.shape[0]), "view": view,
                        "eval_id": eval_ident["id"], "fit_id": fit_id,''',
     1),
    ("merge",
     '''    out = OUT / "baseline_fieldview.csv"
    frame = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        frame = pd.concat([old, frame], ignore_index=True)
        frame = frame.drop_duplicates(
            subset=["dataset", "model", "seed", "view"], keep="last")
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    print("\\nsaved", out)''',
     '''    out = OUT / "baseline_fieldview.csv"
    frame = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        for column in ("eval_id", "fit_id"):
            if column not in old.columns:
                old[column] = ""
        mapped = old["dataset"].map(eval_ids)
        superseded = mapped.notna() & \\
            (old["eval_id"].fillna("") != mapped.fillna(""))
        if superseded.any():
            backup = out.with_name(
                "backup_baseline_fieldview_pre_identity" + out.suffix)
            if not backup.exists():
                backup.write_bytes(out.read_bytes())
                print("previous rows backed up to", backup.name)
            old = old[~superseded]
        frame = pd.concat([old, frame], ignore_index=True)
        frame = frame.drop_duplicates(
            subset=["dataset", "model", "seed", "view", "K", "eval_id"],
            keep="last")
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    RI.write(out, RI.identity({
        "script": "baseline_fieldview.main", "k": args.k, "seeds": seeds,
        "corpora": sorted(eval_ids), "models": models, "eval_ids": eval_ids,
        "fit_ids": sorted({str(f) for f in frame["fit_id"].unique()}),
        "code": RI.code_version()}))
    print("\\nsaved", out)''',
     1),
])

patch("summarize_seed_extension.py", [
    ("imports",
     '''import numpy as np
import pandas as pd
from scipy import stats

import result_identity as RI''',
     '''import numpy as np
import pandas as pd
from scipy import stats

import occlusion_robustness as O
import result_identity as RI
from seed_identity import fair_readout_identity, occlusion_identity''',
     1),
    ("seeds",
     '''SEEDS = [11, 23, 37, 53, 71, 89, 101, 113, 127, 139]''',
     '''SEEDS = [11, 23, 37, 53, 71, 89, 101, 113, 127, 139]
NEW_SEEDS = [53, 71, 89, 101, 113, 127, 139]
K = 30''',
     1),
    ("sidecar check",
     '''def validate(corpus, fc, btm, curves):''',
     '''def verify_sidecars(corpus):
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
            occlusion_identity(corpus, train, test, NEW_SEEDS, K),
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


def validate(corpus, fc, btm, curves):''',
     1),
    ("call",
     '''        problems.extend(validate(corpus, fc, btm, curves))
        for name in (f"fair_readout_{corpus}_newseeds.csv",
                     f"occlusion_robustness_{corpus}_newseeds.csv"):
            path = RERUN / name
            if path.exists() and RI.stored(path) is None:
                print(f"  warning: {name} has no identity sidecar "
                      f"(produced before this check)", flush=True)''',
     '''        problems.extend(validate(corpus, fc, btm, curves))
        problems.extend(verify_sidecars(corpus))''',
     1),
])

patch("results_manifest.py", [
    ("checkpoint comment",
     '''    # reusable baseline fits used by the fair read-out and field-view tables    "outputs/rerun/btm_fit_github_seed11.npz",''',
     '''    # reusable baseline fits used by the fair read-out and field-view tables
    "outputs/rerun/btm_fit_github_seed11.npz",''',
     1),
    ("code entry",
     '''    "extend_seeds_lda_occlusion.py",
    "summarize_seed_extension.py",
]''',
     '''    "extend_seeds_lda_occlusion.py",
    "summarize_seed_extension.py",
    "seed_identity.py",
    "verify_newseed_archives.py",
]''',
     1),
])

patch("tests_review_fixes.py", [
    ("eval identity test",
     '''def main() -> None:
    print("review-fix checks")''',
     '''def test_eval_identity_covers_test_split() -> None:
    """The evaluation identity must move with the test side (C4)."""
    import seed_identity as SI

    def side(tokens):
        return {"vocab": ["a", "b", "c"], "doc_tokens": tokens,
                "reason_tokens": tokens, "item_tokens": tokens}

    train = side([["a", "b"], ["c", "a"]])
    test_a = side([["a", "c"], ["b", "c"]])
    test_b = side([["a", "c"], ["b", "b"]])
    seeds = [11, 23]
    base = SI.corpus_identity("unit", "github", train, test_a, seeds, 3)
    assert base["payload"]["rates"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert base["payload"]["mask"], "mask protocol missing"
    assert base["payload"]["test_data"], "test split missing"
    assert SI.corpus_identity("unit", "github", train, test_b, seeds,
                              3)["id"] != base["id"], "test split not covered"
    assert SI.corpus_identity("unit", "github", train, test_a, [11],
                              3)["id"] != base["id"], "seed list not covered"
    occ = SI.corpus_identity("unit", "github", train, test_a, seeds, 3,
                             extra={"state_digest": "deadbeef"})
    other = SI.corpus_identity("unit", "github", train, test_a, seeds, 3,
                               extra={"state_digest": "cafebabe"})
    assert occ["id"] != other["id"], "checkpoint digest not covered"
    print("  ok  evaluation identity covers test split, seeds and checkpoint")


def main() -> None:
    print("review-fix checks")''',
     1),
    ("register",
     '''    test_reuse_details_identity()
    test_ablation_entry_smoke()''',
     '''    test_reuse_details_identity()
    test_ablation_entry_smoke()
    test_eval_identity_covers_test_split()''',
     1),
])

sys.exit(1 if fails else 0)
