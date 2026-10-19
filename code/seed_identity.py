# -*- coding: utf-8 -*-
"""Shared fitting and evaluation identities for the seed-extension entries.

2026-10-11 review: caches and merged CSVs needed an identity at all.
2026-10-12 review: the *evaluation* identity must also cover the test split,
the mask/scoring protocol, the checkpoint that produced a curve and the code
of the new entry points, and the summariser must verify those sidecars.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import result_identity as RI

RATES = [0.0, 0.25, 0.5, 0.75, 1.0]
MASK_ALGO = "doc_rng_seed_plus7"
ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "outputs" / "rerun"


def vocab_digest(train) -> str:
    return hashlib.sha1(
        "\x1f".join(train["vocab"]).encode("utf-8")).hexdigest()[:16]


def fit_identity(script, corpus, train, K, seed, extra=None) -> dict:
    payload = {
        "script": script, "corpus": corpus, "k": int(K), "seed": int(seed),
        "code": RI.code_version(), "vocab_digest": vocab_digest(train),
        "data": RI.text_digest([train.get("doc_tokens")]),
        "fields": RI.text_digest([train.get("reason_tokens"),
                                  train.get("item_tokens")]),
    }
    if extra:
        payload.update(extra)
    return RI.identity(payload)


def corpus_identity(script, corpus, train, test, seeds, K, extra=None) -> dict:
    payload = {
        "script": script, "corpus": corpus,
        "seeds": [int(s) for s in seeds], "k": int(K),
        "code": RI.code_version(), "vocab_digest": vocab_digest(train),
        "data": RI.text_digest([train.get("doc_tokens")]),
        "fields": RI.text_digest([train.get("reason_tokens"),
                                  train.get("item_tokens")]),
        # evaluation inputs must be pinned too (2026-10-12 review, item C4)
        "test_data": RI.text_digest([test.get("doc_tokens")]),
        "test_fields": RI.text_digest([test.get("reason_tokens"),
                                       test.get("item_tokens")]),
        "rates": list(RATES), "mask": MASK_ALGO,
    }
    if extra:
        payload.update(extra)
    return RI.identity(payload)


def state_path(corpus: str) -> Path:
    return STATE_DIR / f"fcntm_main_state_{corpus}.pt"


def occlusion_identity(corpus, train, test, seeds, K,
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
        extra={"state_digest": RI.file_digest(state)})


def fair_readout_identity(corpus, train, test, seeds, K) -> dict:
    return corpus_identity(
        "newseeds_fair_readout", corpus, train, test, seeds, K,
        extra={"models": ["BTM"]})


def fieldview_identity(corpus, train, test, seeds, K) -> dict:
    return corpus_identity("baseline_fieldview", corpus, train, test, seeds, K)


def verify_checkpoint(path, train, seeds, K) -> None:
    """Refuse a checkpoint whose recorded fit identity is not this corpus."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    want = {
        "k": int(K), "n_vocab": int(train["n_vocab"]),
        "vocab_digest": vocab_digest(train),
        "data": RI.text_digest([train.get("doc_tokens")]),
        "fields": RI.text_digest([train.get("reason_tokens"),
                                  train.get("item_tokens")]),
    }
    for seed in seeds:
        rec = ck.get(f"fielddrop_on:{seed}") or {}
        payload = (rec.get("identity") or {}).get("payload", {})
        bad = [key for key, value in want.items()
               if payload.get(key) != value]
        if bad:
            raise SystemExit(f"{Path(path).name} seed {seed}: checkpoint "
                             f"identity differs in {bad}")


def load_cached_npz(path, ident, arrays, shapes):
    """Return the cached arrays when the identity and shapes both match."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=True)
    except Exception:  # noqa: BLE001
        print(f"  cannot read {path.name}; refitting", flush=True)
        return None
    stored = str(z["identity_id"]) if "identity_id" in z.files else None
    if stored != ident["id"]:
        print(f"  {path.name} is not pinned to this fit "
              f"(stored {stored!r}); refitting", flush=True)
        return None
    out = {}
    for name in arrays:
        if name not in z.files:
            return None
        out[name] = z[name]
    for name, shape in shapes.items():
        if tuple(out[name].shape) != tuple(shape):
            print(f"  {path.name}: {name} shape {out[name].shape} != "
                  f"{tuple(shape)}; refitting", flush=True)
            return None
    return out


def save_cached_npz(path, ident, **arrays) -> None:
    path = Path(path)
    np.savez_compressed(
        path, identity_id=ident["id"],
        identity_json=json.dumps(ident, ensure_ascii=False, default=str),
        **{k: np.asarray(v) for k, v in arrays.items()})


def merge_pinned(path, rows, ident, keys):
    """Merge rows into a CSV pinned to ``ident`` (archive on mismatch)."""
    path = Path(path)
    frame = pd.DataFrame(rows)
    if path.exists():
        if RI.matches(path, ident):
            old = pd.read_csv(path)
            frame = pd.concat([old, frame], ignore_index=True)
            frame = frame.drop_duplicates(subset=keys, keep="last")
        else:
            print(f"  {path.name} is not pinned to this configuration "
                  f"({RI.why_not(path, ident)}); archiving it", flush=True)
            stash = RI.stash_unpinned(path)
            if stash:
                print(f"  old rows moved to {stash}", flush=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    RI.write(path, ident)
    print("saved", path, flush=True)