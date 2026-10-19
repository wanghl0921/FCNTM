# -*- coding: utf-8 -*-
"""2026-10-12 review: fix the ablation entry and the seed-consistency check."""
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


patch("run_ablation_public.py", [
    ("local import removed",
     '''            refused = []
            import result_identity as RI
            want_fit = {''',
     '''            refused = []
            want_fit = {''',
     1),
    ("seed consistency",
     '''    else:
        payload = fit.get("payload", {})
        for key, value in want_fit.items():''',
     '''    else:
        payload = fit.get("payload", {})
        if payload.get("seed") != rec.get("seed"):
            why.append("record seed does not match the fit identity seed")
        for key, value in want_fit.items():''',
     1),
])

patch("tests_review_fixes.py", [
    ("seed mismatch case",
     '''    # a changed protocol field is refused even with matching shapes
    _beta, _theta, why = A.reusable_full_record(
        rec, dict(want, code="other"), "h", 3, 4, 6)
    assert any("code" in w for w in why), why
    print("  ok  details reuse requires the complete fitting identity")''',
     '''    # a changed protocol field is refused even with matching shapes
    _beta, _theta, why = A.reusable_full_record(
        rec, dict(want, code="other"), "h", 3, 4, 6)
    assert any("code" in w for w in why), why
    # the outer record seed must equal the seed inside the fit identity
    wrong_seed = dict(rec)
    wrong_seed["seed"] = 12
    _beta, _theta, why = A.reusable_full_record(
        wrong_seed, want, "h", 3, 4, 6)
    assert any("seed" in w for w in why), why
    print("  ok  details reuse requires the complete fitting identity")''',
     1),
    ("entry smoke test",
     "def main() -> None:\n    print(\"review-fix checks\")",
     '''def test_ablation_entry_smoke() -> None:
    """run_ablation_public.main() must run for all three reuse branches."""
    import json
    import sys
    import tempfile
    from pathlib import Path

    import result_identity as RI
    import run_ablation_public as A

    vocab = ["w0", "w1", "w2", "w3"]
    n_docs, k = 6, 3

    def fake_corpus(n):
        toks = [[vocab[(i + j) % 4] for j in range(2)] for i in range(n)]
        return {
            "n_vocab": 4, "n_docs": n, "vocab": vocab,
            "reason_bow": np.ones((n, 4), dtype=np.float32),
            "item_bow": np.ones((n, 4), dtype=np.float32),
            "reason_tokens": toks, "item_tokens": toks,
            "doc_tokens": [a + b for a, b in zip(toks, toks)],
        }

    def fake_load(name, abstract_words=0):
        return {"train": fake_corpus(n_docs), "test": fake_corpus(2)}

    trained = {"flag": False}

    def fake_train(train, K, seed, overrides, lr, max_epochs=700):
        trained["flag"] = True
        return (np.ones((k, 4), dtype=float),
                np.full((n_docs, k), 1.0 / k))

    def fake_evaluate(topics, mat, corpus):
        return {"CV": 0.5, "CV_n": 1.0, "NPMI": 0.1, "NPMI_n": 1.0,
                "UCI": 0.0, "UCI_n": 1.0, "UMass": 0.0, "UMass_n": 1.0,
                "EXCL": 0.0, "EXCL_KADJ": 0.0, "IRBO": 0.0}

    def details_record(lr):
        train = fake_corpus(n_docs)
        import hashlib
        payload = {
            "k": k, "n_vocab": 4, "n_docs": n_docs, "seed": 11,
            "vocab_digest": hashlib.sha1(
                "\\x1f".join(vocab).encode("utf-8")).hexdigest()[:16],
            "data": RI.text_digest([train["doc_tokens"]]),
            "fields": RI.text_digest([train["reason_tokens"],
                                      train["item_tokens"]]),
            "code": RI.code_version(),
            "field_dropout": float(A.FULL_CONFIG["field_dropout"]),
            "fusion": "gate",
            "fusion_norm": A.FULL_CONFIG["fused_recon_norm"],
            "fusion_weight": float(A.FULL_CONFIG["fused_recon_weight"]),
            "lr": lr, "epochs_budget": 2,
        }
        return {"model": "FC-NTM", "seed": 11,
                "config_hash": A.config_hash(dict(A.FULL_CONFIG)),
                "beta": np.ones((k, 4)).tolist(),
                "theta": np.full((n_docs, k), 1.0 / k).tolist(),
                "fit_identity": RI.identity(payload)}

    old = (A.load_dataset, A.train_model, A.evaluate)
    argv = sys.argv[:]
    A.load_dataset, A.train_model, A.evaluate = (
        fake_load, fake_train, fake_evaluate)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # (a) no details: trains
            trained["flag"] = False
            sys.argv = ["run_ablation_public.py", "--dataset", "github",
                        "--seeds", "11", "--k", str(k),
                        "--configs", "no-cross", "--max-epochs", "2",
                        "--out", str(tmp / "nodetails.csv")]
            A.main()
            assert trained["flag"], "no-details branch did not train"
            # (b) valid details for the full config: reused, no training
            valid = tmp / "valid.json"
            valid.write_text(json.dumps([details_record(5e-4)]),
                             encoding="utf-8")
            trained["flag"] = False
            sys.argv = ["run_ablation_public.py", "--dataset", "github",
                        "--seeds", "11", "--k", str(k), "--lr", "5e-4",
                        "--configs", "full", "--max-epochs", "2",
                        "--reuse-details", str(valid),
                        "--out", str(tmp / "valid.csv")]
            A.main()
            assert not trained["flag"], "valid details were not reused"
            # (c) rejected details (wrong lr): trains fresh
            bad = tmp / "bad.json"
            bad.write_text(json.dumps([details_record(0.123)]),
                           encoding="utf-8")
            trained["flag"] = False
            sys.argv = ["run_ablation_public.py", "--dataset", "github",
                        "--seeds", "11", "--k", str(k), "--lr", "5e-4",
                        "--configs", "full", "--max-epochs", "2",
                        "--reuse-details", str(bad),
                        "--out", str(tmp / "bad.csv")]
            A.main()
            assert trained["flag"], "bad details were not rejected"
    finally:
        A.load_dataset, A.train_model, A.evaluate = old
        sys.argv = argv
    print("  ok  ablation main() runs for no/valid/rejected details")


def main() -> None:
    print("review-fix checks")''',
     1),
    ("call",
     "    test_reuse_details_identity()\n",
     "    test_reuse_details_identity()\n    test_ablation_entry_smoke()\n",
     1),
])
sys.exit(1 if fails else 0)