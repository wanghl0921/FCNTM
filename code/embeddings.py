# -*- coding: utf-8 -*-
"""m3e contextual / word embeddings with local caching."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

from config import EMBEDDING_MODEL_NAME, OUT_DIR


_model = None


def _get_model():
    global _model
    if _model is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoModel, AutoTokenizer

        _model = {
            "tokenizer": AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME),
            "model": AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).eval(),
        }
    return _model


def _encode(texts: List[str], batch_size: int = 32) -> np.ndarray:
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
                max_length=128,
                return_tensors="pt",
            )
            out = encoder(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp_min(1.0)
            all_vectors.append((summed / counts).numpy())
    return np.vstack(all_vectors).astype(np.float32)


def _cache_path(kind: str, items: Sequence[str]) -> Path:
    payload = "\x1f".join(items)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"emb_{kind}_{digest}.npy"


def embed_texts(texts: Sequence[str], kind: str = "doc", batch_size: int = 32) -> np.ndarray:
    """Compute (and cache) m3e embeddings for a list of texts."""
    texts = list(texts)
    path = _cache_path(kind, texts)
    if path.exists():
        return np.load(path)
    arr = _encode(texts, batch_size=batch_size)
    np.save(path, arr)
    return arr


def embed_words(vocab: List[str]) -> np.ndarray:
    return embed_texts(vocab, kind="word", batch_size=64)
