# -*- coding: utf-8 -*-
"""Factor the details-reuse identity check into a testable helper and update
the resume regression test for the smaller-budget rule."""

from pathlib import Path
import sys

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
        path.write_text(out, encoding="utf-8")
        print(f"  patched {name}")


helper = '''def reusable_full_record(rec, want_fit, expected_full, k, n_vocab,
                         n_docs):
    """(beta, theta, why) for a details record that may serve as reuse.

    Every field of the fitting identity must match: a correct config hash
    plus matching matrix shapes is not enough (2026-10-11 review, item 2).
    """
    import numpy as np
    if rec.get("model") != "FC-NTM":
        return None, None, ["not an FC-NTM record"]
    beta = np.asarray(rec["beta"], dtype=float)
    theta = (np.asarray(rec["theta"], dtype=float)
             if rec.get("theta") is not None else None)
    why = []
    if rec.get("config_hash") != expected_full:
        why.append("configuration hash does not match the full configuration")
    fit = rec.get("fit_identity")
    if not fit:
        why.append("details carry no fit identity")
    else:
        payload = fit.get("payload", {})
        for key, value in want_fit.items():
            if payload.get(key) != value:
                why.append(f"fit identity differs in {key!r}")
    if beta.ndim != 2 or beta.shape != (int(k), int(n_vocab)):
        why.append("beta shape is not (K, n_vocab)")
    if theta is None:
        why.append("details carry no theta for the cTFIDF read-out")
    elif theta.ndim != 2 or theta.shape != (int(n_docs), int(k)):
        why.append("theta shape is not (n_docs, K)")
    return beta, theta, why


def complete_pairs'''
# note: helper goes right before complete_pairs in run_ablation_public
a = [
    ("helper",
     "def complete_pairs",
     helper,
     1),
    ("loop uses helper",
     '''            for rec in json.loads(p.read_text(encoding="utf-8")):
                if rec.get("model") != "FC-NTM":
                    continue
                seed = int(rec["seed"])
                stored = rec.get("config_hash")
                beta = np.asarray(rec["beta"], dtype=float)
                theta = (np.asarray(rec["theta"], dtype=float)
                         if rec.get("theta") is not None else None)
                why = []
                if stored != expected_full:
                    why.append("configuration hash does not match the "
                               "full configuration of this run")
                fit = rec.get("fit_identity")
                if not fit:
                    why.append("details carry no fit identity (produced "
                               "before the identity column)")
                else:
                    payload = fit.get("payload", {})
                    for key, value in want_fit.items():
                        if payload.get(key) != value:
                            why.append(f"fit identity differs in {key!r}")
                if beta.ndim != 2 or beta.shape != (int(args.k),
                                                    int(train["n_vocab"])):
                    why.append("beta shape is not (K, n_vocab)")
                if theta is None:
                    why.append("details carry no theta for the cTFIDF read-out")
                elif theta.ndim != 2 or theta.shape != (int(train["n_docs"]),
                                                        int(args.k)):
                    why.append("theta shape is not (n_docs, K)")
                if why:
                    refused.append((seed, "; ".join(why)))
                    continue
                reused[seed] = (beta, theta, stored)''',
     '''            for rec in json.loads(p.read_text(encoding="utf-8")):
                if rec.get("model") != "FC-NTM":
                    continue
                seed = int(rec["seed"])
                beta, theta, why = reusable_full_record(
                    rec, want_fit, expected_full, args.k,
                    train["n_vocab"], train["n_docs"])
                if why:
                    refused.append((seed, "; ".join(why)))
                    continue
                reused[seed] = (beta, theta, rec.get("config_hash"))''',
     1),
]
patch("run_ablation_public.py", a)

t = [
    ("resume budget test",
     '''        # a matching identity does resume: asking for fewer epochs than the
        # stored state reports it continued (a fresh run would report 2)
        run(path, payload_b, 100.0, epochs=3)
        info_resume, _ = run(path, payload_b, 100.0, epochs=2)
        assert int(info_resume["epochs_run"]) == 3, info_resume["epochs_run"]''',
     '''        # a matching identity resumes when the budget grows: after the
        # 1-epoch fresh run above, asking for 4 epochs continues to 4
        info_resume, _ = run(path, payload_b, 100.0, epochs=4)
        assert int(info_resume["epochs_run"]) == 4, info_resume["epochs_run"]

        # a smaller budget must not inherit the 4-epoch state: it archives it
        # and restarts from scratch at 2 (2026-10-11 review, item 5)
        info_short, _ = run(path, payload_b, 100.0, epochs=2)
        assert int(info_short["epochs_run"]) == 2, info_short["epochs_run"]
        assert list(Path(tmp).glob("state.pt.stale*")), \\
            "stale state not archived for the smaller budget"''',
     1),
    ("reuse identity test",
     "def main() -> None:\n    print(\"review-fix checks\")",
     '''def test_reuse_details_identity() -> None:
    """A reused full-config dump must match the complete fitting identity."""
    import run_ablation_public as A
    import result_identity as RI

    want = {"k": 3, "n_vocab": 4, "n_docs": 6, "code": "abc"}
    base = {"model": "FC-NTM", "seed": 11, "config_hash": "h",
            "beta": [[0.0] * 4 for _ in range(2)],
            "theta": [[0.5, 0.5]] * 6}
    # the review's counter-example: 2 topics accepted for a K=3 request
    _beta, _theta, why = A.reusable_full_record(base, want, "h", 3, 4, 6)
    assert any("fit identity" in w for w in why), why
    assert any("beta shape" in w for w in why), why

    rec = dict(base)
    rec["beta"] = [[0.0] * 4 for _ in range(3)]
    rec["theta"] = [[0.5, 0.5, 0.0]] * 6
    rec["fit_identity"] = RI.identity(want)
    _beta, _theta, why = A.reusable_full_record(rec, want, "h", 3, 4, 6)
    assert why == [], why
    # a changed protocol field is refused even with matching shapes
    _beta, _theta, why = A.reusable_full_record(
        rec, dict(want, code="other"), "h", 3, 4, 6)
    assert any("code" in w for w in why), why
    print("  ok  details reuse requires the complete fitting identity")


def main() -> None:
    print("review-fix checks")''',
     1),
    ("call",
     "    test_ablation_resume_identity()\n",
     "    test_ablation_resume_identity()\n    test_reuse_details_identity()\n",
     1),
]
patch("tests_review_fixes.py", t)

sys.exit(1 if fails else 0)