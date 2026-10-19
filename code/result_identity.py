# -*- coding: utf-8 -*-
"""Pin a result file to the exact run that produced it.

The review found that the "skip this seed, it is already scored" checks looked
at a couple of keys only: a CSV produced by a *different* configuration, a
different seed mapping or a different data split could be silently reused, and
a stale occlusion file could never be corrected because the skip was keyed on
the largest masking rate already present.

This module gives every writer a cheap identity for its output:

    ident = RI.identity({...script, config, seeds, data digest...})
    if RI.matches(out_path, ident):   # reuse is safe
        ...
    RI.write(out_path, ident)         # called once the file is complete

An output without a matching sidecar (``<file>.identity.json``) is treated as
*unverified* and recomputed rather than reused.  Nothing here touches the data
itself; it only decides whether an old file may be trusted.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# the modules that define how a result is produced; a change in any of them
# changes the identity of every subsequent run
CODE_FILES = (
    "config.py",
    "models.py",
    "baselines.py",
    "new_baselines.py",
    "metrics.py",
    "experiment.py",
    "run_fcntm_train.py",
    "run_public_missing_field.py",
    "occlusion_robustness.py",
    "benchmark_public.py",
    "fair_readout.py",
    "dev_split.py",
    "public_data.py",
    "public_data_github.py",
    "embed_public.py",
    # entry points and measures that the 2026-10-07 review found were missing
    # from the code digest even though they define results
    "public_data_civil.py",
    "run_ablation_public.py",
    "run_core_ablations.py",
    "summarize_ladder.py",
    "field_complementarity.py",
    "field_redundancy.py",
    "add_civil_btm_occlusion.py",
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
    # 2026-10-15 v3 review: the Concat entry point defines its training
    # closure and hyper-parameters, and the archival entry defines the
    # scoring protocol, so both belong in the code digest
    "run_concat_control.py",
    "civil_title_archival.py",
)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_digest(path) -> str:
    """Short content hash of an artifact (e.g. a trained checkpoint)."""
    path = Path(path)
    if not path.exists():
        return "missing"
    return sha256_file(path)[:16]


def code_version(files=CODE_FILES) -> str:
    """Digest of the implementation that writes this kind of result."""
    h = hashlib.sha256()
    for name in files:
        path = ROOT / name
        h.update(name.encode())
        h.update(sha256_file(path).encode() if path.exists() else b"missing")
    return h.hexdigest()[:16]


def text_digest(splits) -> str:
    """Digest of the tokenised corpus actually used by a run.

    ``splits`` is a sequence of token lists (or of lists of token lists); the
    full text is hashed so a different preprocessing / split / vocabulary
    source is visible in the identity, while the cost is a few megabytes.
    """
    h = hashlib.sha256()
    for split in splits:
        if split is None:
            h.update(b"|none|")
            continue
        for doc in split:
            if isinstance(doc, str):
                h.update(doc.encode("utf-8"))
            else:  # a token list
                h.update(" ".join(doc).encode("utf-8"))
            h.update(b"\n")
        h.update(b"||")
    return h.hexdigest()[:16]


def array_digest(*arrays) -> str:
    """Digest of the numeric state a result row was scored from.

    For a trained topic model that state is exactly ``(beta, theta)``, so the
    digest identifies the checkpoint without having to serialise it again; any
    change in the scored state (a different seed, epoch or implementation)
    changes the hash.
    """
    import numpy as np

    h = hashlib.sha256()
    for arr in arrays:
        if arr is None:
            h.update(b"|none|")
            continue
        a = np.asarray(arr, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def identity(payload: dict) -> dict:
    """Canonical, hashable description of one run."""
    body = json.dumps(payload, sort_keys=True, default=str,
                      ensure_ascii=False)
    return {"id": hashlib.sha256(body.encode("utf-8")).hexdigest()[:16],
            "payload": payload}


def sidecar_path(result_path) -> Path:
    return Path(str(result_path) + ".identity.json")


def write(result_path, ident: dict) -> None:
    sidecar_path(result_path).write_text(
        json.dumps(ident, indent=1, ensure_ascii=False), encoding="utf-8")


def stored(result_path):
    path = sidecar_path(result_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def matches(result_path, ident: dict) -> bool:
    old = stored(result_path)
    return bool(old) and old.get("id") == ident["id"]


def why_not(result_path, ident: dict) -> str:
    """Human-readable reason for refusing to reuse a stored result."""
    if not Path(result_path).exists():
        return "no stored result"
    old = stored(result_path)
    if old is None:
        return "stored result carries no identity sidecar (produced before " \
               "identities were pinned)"
    if old.get("id") == ident["id"]:
        return "identity matches"
    old_payload = old.get("payload", {})
    diff = [k for k in set(old_payload) | set(ident["payload"])
            if old_payload.get(k) != ident["payload"].get(k)]
    return "identity differs in: " + ", ".join(sorted(diff))


def stash_unpinned(result_path):
    """Move an existing result aside before it is rewritten.

    When the stored rows cannot be reused the writer starts from scratch, which
    would otherwise destroy the only copy of the old numbers if the recompute
    is interrupted.  The old file is renamed (never deleted) so both versions
    survive; the returned path is the backup, or ``None`` if there was nothing
    to move.
    """
    path = Path(result_path)
    if not path.exists():
        return None
    for i in range(100):
        candidate = path.with_name(
            path.stem + (".unpinned" if i == 0 else f".unpinned{i}")
            + path.suffix)
        if not candidate.exists():
            os.replace(path, candidate)
            return candidate
    raise RuntimeError(f"could not find a free backup name for {path}")
