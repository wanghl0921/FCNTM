# -*- coding: utf-8 -*-
"""Topic evaluation metrics (independent implementations on top of gensim primitives)."""

from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from scipy.optimize import linear_sum_assignment

from config import CV_WINDOW, NPMI_UCI_WINDOW, TOPN


def topics_from_matrix(
    beta: np.ndarray,
    vocab: List[str],
    topn: int = TOPN,
) -> List[List[str]]:
    """beta: (K, V) non-negative or logit matrix -> top words per topic."""
    beta = np.asarray(beta, dtype=float)
    if beta.ndim != 2 or beta.shape[1] != len(vocab):
        raise ValueError(f"beta shape {beta.shape} incompatible with vocab {len(vocab)}")
    if np.any(beta < -1e-9):
        beta = beta - beta.max(axis=1, keepdims=True)
        exp = np.exp(beta)
        beta = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)
    topics = []
    for row in beta:
        idx = np.argsort(-row)[:topn]
        topics.append([vocab[int(i)] for i in idx])
    return topics


def filter_topics_to_vocab(
    topics: List[List[str]],
    reference_vocab: set,
    topn: int = TOPN,
) -> List[List[str]]:
    """Keep only topic words that the reference field actually uses.

    A field view restricts a topic to one field's vocabulary and re-ranks it.
    When the model puts almost no mass on that field, the surviving words are
    near-zero-probability noise that never occurs in the field's reference
    texts.  Those words have an all-zero context vector, so gensim's c_v
    cosine similarity is 0/0 -> NaN and silently poisons the mean.  Dropping
    words the reference field never uses removes the degeneracy and makes the
    score a statement about words the field really contains.  Topics left with
    fewer than two words are dropped rather than scored off a single pair.
    """
    kept: List[List[str]] = []
    for topic in topics:
        words = [w for w in topic if w in reference_vocab][:topn]
        if len(words) >= 2:
            kept.append(words)
    return kept


def coherence_scores(
    topics: List[List[str]],
    token_lists: List[List[str]],
    corpus: List[List[Tuple[int, int]]],
    dictionary: Dictionary,
    topn: int = TOPN,
    reference_vocab: Optional[set] = None,
) -> Dict[str, float]:
    """CV / NPMI / UCI / UMass over the same corpus.

    When ``reference_vocab`` is given the topics are restricted to it first
    (see :func:`filter_topics_to_vocab`) and ``<metric>_n`` reports how many
    topics actually produced a finite score, so a thin cell shows up as low
    support instead of as a plausible-looking average over a handful of topics.
    """
    topics = [list(t) for t in topics]
    out: Dict[str, float] = {"n_topics": float(len(topics))}
    specs = {
        "CV": {"coherence": "c_v", "texts": token_lists, "window": CV_WINDOW, "corpus": None},
        "NPMI": {"coherence": "c_npmi", "texts": token_lists, "window": NPMI_UCI_WINDOW, "corpus": None},
        "UCI": {"coherence": "c_uci", "texts": token_lists, "window": NPMI_UCI_WINDOW, "corpus": None},
        "UMass": {"coherence": "u_mass", "texts": None, "window": None, "corpus": corpus},
    }
    for metric, spec in specs.items():
        scored = topics
        if reference_vocab is not None:
            scored = filter_topics_to_vocab(topics, reference_vocab, topn)
        if not scored:
            out[metric] = float("nan")
            out[metric + "_n"] = 0.0
            continue
        kwargs = dict(
            topics=scored,
            dictionary=dictionary,
            coherence=spec["coherence"],
            topn=topn,
            processes=1,
        )
        if spec["corpus"] is not None:
            kwargs["corpus"] = spec["corpus"]
        else:
            kwargs["texts"] = spec["texts"]
            kwargs["window_size"] = spec["window"]
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = CoherenceModel(**kwargs)
            per_topic = np.asarray(cm.get_coherence_per_topic(), dtype=float)
        per_topic[~np.isfinite(per_topic)] = np.nan
        finite = int(np.isfinite(per_topic).sum())
        out[metric + "_n"] = float(finite)
        if finite:
            out[metric] = float(np.nanmean(per_topic))
        else:
            out[metric] = float("nan")
    return out


def exclusivity(beta: np.ndarray, topn: int = TOPN) -> float:
    """Bischof & Airoldi-style exclusivity over per-topic top words."""
    beta = np.asarray(beta, dtype=float)
    if np.any(beta < 0):
        beta = beta - beta.max(axis=1, keepdims=True)
        exp = np.exp(beta)
        beta = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)
    beta = beta / (beta.sum(axis=1, keepdims=True) + 1e-12)
    col_sum = beta.sum(axis=0, keepdims=True)
    ratio = beta / (col_sum + 1e-12)
    vals = []
    for k in range(beta.shape[0]):
        top = np.argsort(-beta[k])[:topn]
        vals.append(float(np.mean(ratio[k, top])))
    return float(np.mean(vals)) if vals else float("nan")


def exclusivity_kadj(beta: np.ndarray, topn: int = TOPN) -> float:
    raw = exclusivity(beta, topn=topn)
    k = int(np.asarray(beta).shape[0])
    if k <= 1 or not np.isfinite(raw):
        return raw
    base = 1.0 / k
    return float((raw - base) / (1.0 - base))


def rbo(
    list_a: Sequence[str],
    list_b: Sequence[str],
    p: float = 0.9,
    tail: str = "extrapolate",
) -> float:
    """Rank-biased overlap between two (truncated) topic word lists.

    ``tail="trunc"`` returns the truncated sum
    ``(1-p) * sum_{d=1..L} p^(d-1) A_d``.  That quantity is a *lower bound*
    of RBO and is not normalized: two identical 10-word lists score
    ``1 - p^10 = 0.651`` rather than 1, so ``1 - RBO_trunc`` (IRBO = 0.349)
    overstates the diversity of any topic list.

    ``tail="extrapolate"`` adds the residual tail term ``p^L * A_L``.  For
    equal-length lists this is the standard finite extension that makes
    identical lists score exactly 1 and disjoint lists score 0, which is what
    a "diversity" of ``1 - RBO`` requires.  This is the default; the truncated
    historical definition is still available for comparison.
    """
    depth = min(len(list_a), len(list_b))
    if depth == 0:
        return 0.0
    set_a: set = set()
    set_b: set = set()
    overlap = 0.0
    for d in range(1, depth + 1):
        set_a.add(list_a[d - 1])
        set_b.add(list_b[d - 1])
        overlap += p ** (d - 1) * len(set_a & set_b) / d
    score = (1.0 - p) * overlap
    if tail == "extrapolate":
        score += p ** depth * (len(set_a & set_b) / depth)
    elif tail != "trunc":
        raise ValueError(f"unknown RBO tail handling: {tail}")
    return float(score)


def rbo_ext(list_a: Sequence[str], list_b: Sequence[str], p: float = 0.9) -> float:
    """RBO with the residual tail term (see :func:`rbo`)."""
    return rbo(list_a, list_b, p=p, tail="extrapolate")


def irbo_diversity(
    topics: List[List[str]],
    p: float = 0.9,
    tail: str = "extrapolate",
) -> float:
    """Inter-topic diversity, ``1 - mean RBO`` over all topic pairs."""
    if len(topics) < 2:
        return 0.0
    scores = [rbo(topics[i], topics[j], p, tail=tail)
              for i, j in combinations(range(len(topics)), 2)]
    return float(1.0 - float(np.mean(scores)))


def topic_jaccard(list_a: Sequence[str], list_b: Sequence[str]) -> float:
    a, b = set(list_a), set(list_b)
    if not a and not b:
        return 1.0
    return float(len(a & b) / len(a | b))


def aligned_topic_jaccard(
    topics_a: List[List[str]],
    topics_b: List[List[str]],
) -> float:
    """Mean Jaccard after Hungarian alignment of topic lists."""
    k = min(len(topics_a), len(topics_b))
    if k == 0:
        return float("nan")
    cost = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            cost[i, j] = -topic_jaccard(topics_a[i], topics_b[j])
    row, col = linear_sum_assignment(cost)
    return float(np.mean([-cost[r, c] for r, c in zip(row, col)]))


def summarize_metric_list(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0}
