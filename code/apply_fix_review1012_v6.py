# -*- coding: utf-8 -*-
"""2026-10-12 v6 review: the scored checkpoint must be the recorded one.

seed_identity.occlusion_identity() always hashed the *default* state path, so a
run with --state-dir recorded a digest of a file it never scored.  The helper
now takes the actual checkpoint, the FC-NTM entry passes it, the summariser
accepts an explicit --state-dir, and a new entry test builds two real state
files (default / alternate) and checks which digest lands in the identity.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
fails = []


def patch(name, pairs):
    path = ROOT / name
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    for tag, old, new, count in pairs:
        n = text.count(old)
        if n != count:
            fails.append(f"{name}: {tag}: found {n}, want {count}")
            continue
        text = text.replace(old, new)
    if not any(f.startswith(name) for f in fails):
        path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
        print(f"  patched {name}")


patch("seed_identity.py", [
    ("occlusion identity",
     '''def occlusion_identity(corpus, train, test, seeds, K) -> dict:
    return corpus_identity(
        "newseeds_occlusion", corpus, train, test, seeds, K,
        extra={"state_digest": RI.file_digest(state_path(corpus))})''',
     '''def occlusion_identity(corpus, train, test, seeds, K,
                       checkpoint=None) -> dict:
    """Identity of one occlusion protocol over one checkpoint.

    ``checkpoint`` is the state file that was actually scored; it defaults to
    the delivered ``outputs/rerun`` state.  2026-10-12 v6 review: recording
    the default path while scoring a ``--state-dir`` file would pin the wrong
    source and let two different checkpoints merge.
    """
    state = Path(checkpoint) if checkpoint is not None else state_path(corpus)
    return corpus_identity(
        "newseeds_occlusion", corpus, train, test, seeds, K,
        extra={"state_digest": RI.file_digest(state)})''',
     1),
])

patch("extend_seeds_fcntm_occlusion.py", [
    ("identity call",
     '''        merge_pinned(
            O.OUT_DIR / "rerun" / f"occlusion_robustness_{corpus}_newseeds.csv",
            rows, occlusion_identity(corpus, train, test, seeds, args.k),
            ["model", "seed", "rate"])''',
     '''        merge_pinned(
            O.OUT_DIR / "rerun" / f"occlusion_robustness_{corpus}_newseeds.csv",
            rows, occlusion_identity(corpus, train, test, seeds, args.k,
                                     checkpoint=state),
            ["model", "seed", "rate"])''',
     1),
])

patch("summarize_seed_extension.py", [
    ("verify signature",
     '''def verify_sidecars(corpus):''',
     '''def verify_sidecars(corpus, state_dir="") -> list:''',
     1),
    ("verify call",
     '''        f"occlusion_robustness_{corpus}_newseeds.csv":
            occlusion_identity(corpus, train, test, NEW_SEEDS, K),
    }''',
     '''        f"occlusion_robustness_{corpus}_newseeds.csv":
            occlusion_identity(
                corpus, train, test, NEW_SEEDS, K,
                checkpoint=(Path(state_dir) / f"fcntm_main_state_{corpus}.pt")
                if state_dir else None),
    }''',
     1),
    ("main args",
     '''def main() -> int:
    data, problems = {}, []''',
     '''def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="",
                    help="directory of the FC-NTM states the new-seed curves "
                         "were scored from (default: outputs/rerun)")
    args = ap.parse_args()
    data, problems = {}, []''',
     1),
    ("main call",
     '''        problems.extend(verify_sidecars(corpus))''',
     '''        problems.extend(verify_sidecars(corpus, args.state_dir))''',
     1),
])

patch("verify_newseed_archives.py", [
    ("identity call",
     '''        pin(occ_path, occlusion_identity(corpus, train, test, SEEDS, args.k),
            fresh, OCC_KEYS)''',
     '''        pin(occ_path,
            occlusion_identity(corpus, train, test, SEEDS, args.k,
                               checkpoint=state_path(corpus)),
            fresh, OCC_KEYS)''',
     1),
])

patch("tests_review_fixes.py", [
    ("entry test",
     '''def main() -> None:
    print("review-fix checks")''',
     '''def test_state_dir_identity_uses_the_actual_checkpoint() -> None:
    """--state-dir must be the file whose digest lands in the identity (v6)."""
    import sys
    import tempfile
    from pathlib import Path

    import extend_seeds_fcntm_occlusion as E
    import result_identity as RI

    class FakeDict(dict):
        token2id = {"w": 0}

    def fake_corpus():
        return {"vocab": ["w"], "n_vocab": 1, "n_docs": 2,
                "doc_tokens": [["w"], ["w"]],
                "reason_tokens": [["w"], ["w"]],
                "item_tokens": [[], []], "dictionary": FakeDict()}

    captured = {}

    class FakeAdapter:
        def __init__(self, path, vocab, k, seed=None):
            captured["state"] = Path(path)

    def fake_merge(path, rows, ident, keys):
        captured["ident"] = ident
        captured["rows"] = list(rows)

    old = (E.O.load_corpus, E.O.FCNTMAdapter, E.O.run_model, E.masks_for,
           E.merge_pinned, E.verify_checkpoint, E.state_path)
    checked = {}
    E.O.load_corpus = lambda corpus: (fake_corpus(), fake_corpus())
    E.O.FCNTMAdapter = FakeAdapter
    E.O.run_model = lambda *a, **k: [{"model": "FC-NTM", "rate": 0.25,
                                      "hidden_NLL": 1.0, "n_docs": 1}]
    E.masks_for = lambda test, seed: ({0.25: [["w"]]}, {0.25: [["w"]]})
    E.merge_pinned = fake_merge
    E.verify_checkpoint = lambda path, train, seeds, k: checked.update(
        path=Path(path), seeds=list(seeds), k=k)
    argv = sys.argv[:]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            default = tmp / "default.pt"
            alt = tmp / "alt" / "fcntm_main_state_github.pt"
            alt.parent.mkdir()
            default.write_bytes(b"default-state")
            alt.write_bytes(b"alternate-state")
            E.state_path = lambda corpus: default
            sys.argv = ["extend_seeds_fcntm_occlusion.py", "--corpora",
                        "github", "--seeds", "53", "--state-dir",
                        str(alt.parent)]
            E.main()
    finally:
        (E.O.load_corpus, E.O.FCNTMAdapter, E.O.run_model, E.masks_for,
         E.merge_pinned, E.verify_checkpoint, E.state_path) = old
        sys.argv = argv
    assert checked.get("path") == alt, checked
    assert captured.get("state") == alt, captured.get("state")
    payload = captured["ident"]["payload"]
    assert payload["state_digest"] == RI.file_digest(alt), payload
    assert payload["state_digest"] != RI.file_digest(default), payload
    print("  ok  --state-dir digest follows the file that was scored")


def main() -> None:
    print("review-fix checks")''',
     1),
    ("register",
     '''    test_eval_identity_covers_test_split()''',
     '''    test_eval_identity_covers_test_split()
    test_state_dir_identity_uses_the_actual_checkpoint()''',
     1),
])

import sys
if fails:
    print("unmatched anchors:")
    for f in fails:
        print("  ", f)
sys.exit(1 if fails else 0)
