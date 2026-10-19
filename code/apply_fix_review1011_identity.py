# -*- coding: utf-8 -*-
"""2026-10-11 review fixes, part 1: details reuse identity + resume budget.

- run_fcntm_train.py writes a full ``fit_identity`` into every details record
  (K, vocab digest and order, data/field digests, protocol, lr, budget, code).
- run_ablation_public.py refuses a reused full-config dump unless that identity
  matches field by field and the shapes are (K, n_vocab) / (n_docs, K).
- models.py refuses to continue a resume state whose stored epoch exceeds the
  requested budget (extension is still allowed).
"""

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


# ------------------------------------------------------------ run_fcntm_train
r = [
    ("details fit identity",
     '''                    "resumed_from": (resume_file if resume_file
                                     and os.path.exists(resume_file) else None),
                })''',
     '''                    "resumed_from": (resume_file if resume_file
                                     and os.path.exists(resume_file) else None),
                    # full fitting identity: reuse must match every field
                    # (2026-10-11 review, item 2)
                    "fit_identity": RI.identity({
                        "script": "run_fcntm_train.details",
                        "seed": int(seed),
                        "k": int(K),
                        "n_vocab": int(corpus_train["n_vocab"]),
                        "n_docs": int(corpus_train["n_docs"]),
                        "vocab_digest": hashlib.sha1(
                            "\\x1f".join(corpus_train["vocab"]).encode(
                                "utf-8")).hexdigest()[:16],
                        "data": RI.text_digest(
                            [corpus_train.get("doc_tokens")]),
                        "fields": RI.text_digest(
                            [corpus_train.get("reason_tokens"),
                             corpus_train.get("item_tokens")]),
                        "code": RI.code_version(),
                        "field_dropout": float(field_dropout),
                        "fusion": fusion,
                        "fusion_norm": fusion_norm,
                        "fusion_weight": float(fusion_weight),
                        "lr": float(lr or 0.0),
                        "epochs_budget": (int(max_epochs) if max_epochs
                                          else None),
                    }),
                })''',
     1),
]
patch("run_fcntm_train.py", r)

# -------------------------------------------------------- run_ablation_public
a = [
    ("strict reuse",
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
                if beta.ndim != 2 or beta.shape[1] != train["n_vocab"]:
                    why.append("beta shape does not match the current "
                               "vocabulary")
                if theta is not None and theta.shape[0] != train["n_docs"]:
                    why.append("theta does not match the current training "
                               "documents")
                if why:
                    refused.append((seed, "; ".join(why)))
                    continue
                reused[seed] = (beta, theta, stored)''',
     '''            import result_identity as RI
            want_fit = {
                "k": int(args.k),
                "n_vocab": int(train["n_vocab"]),
                "n_docs": int(train["n_docs"]),
                "vocab_digest": __import__("hashlib").sha1(
                    "\\x1f".join(train["vocab"]).encode("utf-8")
                ).hexdigest()[:16],
                "data": RI.text_digest([train.get("doc_tokens")]),
                "fields": RI.text_digest([train.get("reason_tokens"),
                                          train.get("item_tokens")]),
                "code": RI.code_version(),
                "field_dropout": float(FULL_CONFIG["field_dropout"]),
                "fusion": "gate",
                "fusion_norm": FULL_CONFIG["fused_recon_norm"],
                "fusion_weight": float(FULL_CONFIG["fused_recon_weight"]),
                "lr": float(args.lr),
                "epochs_budget": int(args.max_epochs),
            }
            for rec in json.loads(p.read_text(encoding="utf-8")):
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
     1),
]
patch("run_ablation_public.py", a)

# ------------------------------------------------------------------ models.py
m = [
    ("budget reduction guard",
     '''        elif ck is not None and stored_id != want_id:
            print(f"    [resume] identity mismatch (stored {stored_id!r}, "
                  f"current {want_id!r}); the old state cannot be reused",
                  flush=True)
            ck = None''',
     '''        elif ck is not None and stored_id != want_id:
            print(f"    [resume] identity mismatch (stored {stored_id!r}, "
                  f"current {want_id!r}); the old state cannot be reused",
                  flush=True)
            ck = None
        elif ck is not None and int(ck.get("epoch", 0)) > int(num_epochs):
            # a smaller budget must not silently inherit a longer run
            # (2026-10-11 review, item 5)
            print(f"    [resume] requested budget ({num_epochs}) is smaller "
                  f"than the stored epoch ({ck.get('epoch')}); restarting "
                  f"from scratch", flush=True)
            ck = None''',
     1),
]
patch("models.py", m)

sys.exit(1 if fails else 0)