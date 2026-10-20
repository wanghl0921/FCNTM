# -*- coding: utf-8 -*-
"""English sentence/word embeddings for the AG News public benchmark.

Uses sentence-transformers/all-MiniLM-L6-v2 (384-d), a standard lightweight
English sentence embedding model.  Results are cached next to the other outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

from config import OUT_DIR

EMB_MODEL = os.environ.get(
    "FCNTM_EN_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
# the exact commit the archived runs used; override with
# FCNTM_EN_EMBEDDING_REVISION when reproducing with another snapshot
EMB_REVISION = os.environ.get(
    "FCNTM_EN_EMBEDDING_REVISION",
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
EMB_MAX_LENGTH = int(os.environ.get("FCNTM_EN_EMBEDDING_MAX_LEN", "128"))
# long-document control: with FCNTM_EN_EMBEDDING_CHUNKED=1 every document is
# split into EMB_MAX_LENGTH-token windows and the window vectors are averaged,
# so the encoder is no longer limited to the first EMB_MAX_LENGTH tokens.
# The archived baseline numbers were produced with this switch *off*.
EMB_CHUNK_LONG = os.environ.get(
    "FCNTM_EN_EMBEDDING_CHUNKED", "0").strip().lower() in ("1", "true", "yes")
# bump when the windowing / pooling implementation changes: the value is
# part of both the cache key and the identity sidecar, so vectors produced
# by another algorithm can never be reused (2026-10-07 review, items 3-4)
EMB_ALGO = "tokwin-v2"
_MODEL = None


# Offline mode is opt-in: set FCNTM_HF_OFFLINE=1 (or HF_HUB_OFFLINE=1) to use
# the cached copy only.  By default the pinned revision is downloaded on first
# use and cached by huggingface_hub.
_OFFLINE = os.environ.get("FCNTM_HF_OFFLINE", "").strip().lower() in (
    "1", "true", "yes")
if _OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _get_model():
    global _MODEL
    if _MODEL is None:
        from transformers import AutoModel, AutoTokenizer

        _MODEL = {
            "tokenizer": AutoTokenizer.from_pretrained(
                EMB_MODEL, revision=EMB_REVISION,
                local_files_only=_OFFLINE),
            "model": AutoModel.from_pretrained(
                EMB_MODEL, revision=EMB_REVISION,
                local_files_only=_OFFLINE).eval(),
        }
    return _MODEL


def _encode(texts: List[str], batch_size: int = 64) -> np.ndarray:
    model = _get_model()
    tokenizer = model["tokenizer"]
    encoder = model["model"]
    all_vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=EMB_MAX_LENGTH,
                return_tensors="pt",
            )
            out = encoder(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp_min(1.0)
            all_vectors.append((summed / counts).numpy())
    return np.vstack(all_vectors).astype(np.float32)


def _with_specials(tokenizer, window: Sequence[int]) -> List[int]:
    """Add the tokenizer's special tokens around one window of token ids.

    Newer ``transformers`` versions no longer expose
    ``build_inputs_with_special_tokens`` on every tokenizer, so the special
    ids are taken from the tokenizer attributes (BERT-family) or, failing
    that, from the encoding of an empty string.
    """
    cls = getattr(tokenizer, "cls_token_id", None)
    sep = getattr(tokenizer, "sep_token_id", None)
    if cls is not None and sep is not None:
        return [int(cls)] + [int(t) for t in window] + [int(sep)]
    special = list(tokenizer("", add_special_tokens=True)["input_ids"])
    half = len(special) // 2
    return special[:half] + [int(t) for t in window] + special[half:]


def _encode_windows(windows: List[List[int]],
                    batch_size: int = 64) -> np.ndarray:
    """Encode pre-tokenised windows, adding the special tokens per window."""
    model = _get_model()
    tokenizer = model["tokenizer"]
    encoder = model["model"]
    all_vectors = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            ids = [_with_specials(tokenizer, w) for w in batch]
            encoded = tokenizer.pad(
                {"input_ids": ids,
                 "attention_mask": [[1] * len(x) for x in ids]},
                padding=True, return_tensors="pt")
            out = encoder(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp_min(1.0)
            all_vectors.append((summed / counts).numpy())
    return np.vstack(all_vectors).astype(np.float32)


def token_windows(ids: Sequence[int], max_length: int,
                  n_special: int) -> List[List[int]]:
    """Split token ids into windows that each fit the encoder's limit.

    ``n_special`` is the number of tokens the tokenizer adds around an
    input ([CLS]/[SEP] for BERT-family models), so each window carries at
    most ``max_length - n_special`` ids.  Every input id lands in exactly
    one window; an empty input still yields one (empty) window.
    """
    window = max(1, int(max_length) - int(n_special))
    parts = [list(ids[start : start + window])
             for start in range(0, len(ids), window)]
    return parts or [[]]


def _encoder_identity(kind: str, chunk_long: bool) -> dict:
    """Everything that can change the vector for the same input text."""
    return {
        "algo": EMB_ALGO,
        "model": EMB_MODEL,
        "revision": EMB_REVISION,
        "max_length": int(EMB_MAX_LENGTH),
        "kind": kind,
        "pooling": "token-window-mean" if chunk_long else "truncated-mean",
    }


def _cache_path(kind: str, items: Sequence[str],
                chunk_long: bool = False) -> Path:
    model_short = EMB_MODEL.split("/")[-1]
    payload = "\x1f".join(items)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    ident = json.dumps(_encoder_identity(kind, chunk_long), sort_keys=True)
    key = hashlib.sha1(ident.encode("utf-8")).hexdigest()[:12]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"emb_{model_short}_{kind}_{key}_{digest}.npy"


def chunk_and_pool(texts: Sequence[str]) -> np.ndarray:
    """Mean-pool MiniLM embeddings over every token window of each document.

    The contextual baselines could only see the first ``max_length`` tokens
    while the bag-of-words models saw the whole text - a real disadvantage on
    GitHub, whose issues can exceed 20k tokens.  The earlier control split on
    whitespace and counted special tokens against the window, so window tails
    could still be truncated.  This version tokenises with the model's own
    tokenizer, reserves room for the special tokens and windows the token ids,
    so *every* token of the document enters exactly one window.
    """
    texts = list(texts)
    model = _get_model()
    tokenizer = model["tokenizer"]
    n_special = int(tokenizer.num_special_tokens_to_add(pair=False))
    chunks, owners = [], []
    n_tokens = n_covered = 0
    for i, text in enumerate(texts):
        ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        n_tokens += len(ids)
        for part in token_windows(ids, EMB_MAX_LENGTH, n_special):
            n_covered += len(part)
            chunks.append(part)
            owners.append(i)
    if n_covered != n_tokens:
        raise RuntimeError(
            f"token windows lost {n_tokens - n_covered} of {n_tokens} tokens")
    print(f"  [embed] token windows: {len(texts)} docs -> {len(chunks)} "
          f"windows, {n_covered}/{n_tokens} encoder tokens covered",
          flush=True)
    vectors = _encode_windows(chunks)
    out = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
    weights = np.zeros(len(texts), dtype=np.float32)
    for vector, owner in zip(vectors, owners):
        out[owner] += vector
        weights[owner] += 1.0
    return out / np.maximum(weights[:, None], 1.0)


def embed_texts(texts: Sequence[str], kind: str = "doc",
                chunk_long: bool = False) -> np.ndarray:
    """Encode ``texts``; ``FCNTM_EN_EMBEDDING_CHUNKED=1`` forces window pooling.

    The cache key names the encoder, its revision, the maximum length, the
    pooling method and the algorithm version; the sidecar repeats that
    identity and the array shape.  An unusable cache is moved aside instead
    of being reused (2026-10-07 review, item 4).
    """
    texts = list(texts)
    chunk_long = chunk_long or EMB_CHUNK_LONG
    path = _cache_path(kind, texts, chunk_long)
    sidecar = path.with_name(path.stem + ".identity.json")
    identity = {"encoder": _encoder_identity(kind, chunk_long),
                "n_items": len(texts)}
    if path.exists():
        try:
            arr = np.load(path)
            meta = (json.loads(sidecar.read_text(encoding="utf-8"))
                    if sidecar.exists() else None)
            if (meta is not None
                    and meta.get("identity") == identity
                    and list(arr.shape) == list(meta.get("shape", []))
                    and arr.shape[0] == len(texts)):
                return arr
            print(f"  cached embeddings {path.name} do not match this "
                  f"encoder (sidecar missing or different); recomputing",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  cannot reuse cached embeddings {path.name}: {exc}; "
                  f"recomputing", flush=True)
        stamp = time.strftime("%Y%m%d%H%M%S")
        for stale in (path, sidecar):
            if stale.exists():
                os.replace(stale, stale.with_name(
                    stale.name + f".stale{stamp}"))
    arr = chunk_and_pool(texts) if chunk_long else _encode(texts)
    np.save(path, arr)
    sidecar.write_text(json.dumps(
        {"identity": identity, "shape": list(arr.shape)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return arr


def embed_words(vocab: List[str]) -> np.ndarray:
    return embed_texts(vocab, kind="word")
