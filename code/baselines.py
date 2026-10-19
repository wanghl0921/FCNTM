# -*- coding: utf-8 -*-
"""Classical baselines: LDA (gensim) and GSDMM (self-written Gibbs sampler)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from gensim.models import LdaModel
from scipy.special import gammaln
from sklearn.decomposition import NMF as SklearnNMF

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

from config import TOPN
from metrics import topics_from_matrix


def run_lda(
    corpus: List[List],
    dictionary,
    vocab: List[str],
    k: int,
    seed: int,
    passes: int = 50,
    iterations: int = 120,
) -> Dict[str, object]:
    model = LdaModel(
        corpus=corpus,
        num_topics=k,
        id2word=dictionary,
        alpha="symmetric",
        eta="auto",
        passes=passes,
        iterations=iterations,
        random_state=seed,
        minimum_probability=0.0,
        dtype=np.float64,
        eval_every=0,
    )
    beta = np.asarray(model.get_topics(), dtype=float)
    topics = topics_from_matrix(beta, vocab, TOPN)
    theta = np.zeros((len(corpus), k), dtype=float)
    for i, bow in enumerate(corpus):
        for topic, p in model.get_document_topics(bow, minimum_probability=0.0):
            theta[i, topic] = p
    return {
        "model": model,
        "beta": beta,
        "topics": topics,
        # document-topic matrix, exposed so that the *same* class-TF-IDF
        # read-out can be applied to every model that has one
        "theta": theta,
        "topic_count": len(topics),
        "log_perplexity": float(model.log_perplexity(corpus)),
    }


def run_gsdmm(
    token_lists: List[List[str]],
    dictionary,
    vocab: List[str],
    k: int,
    seed: int,
    alpha: float = 0.1,
    beta: float = 0.01,
    max_iterations: int = 300,
) -> Dict[str, object]:
    """Gibbs-sampled Dirichlet-multinomial mixture (GSDMM, Yin & Wang 2014)."""
    rng = np.random.RandomState(seed)
    word2id = dictionary.token2id
    docs = [[word2id[w] for w in ts if w in word2id] for ts in token_lists]
    docs = [d for d in docs if d]
    n_docs = len(docs)
    V = len(vocab)

    # deterministic round-robin initialization keeps every cluster non-empty
    d_z = np.array([i % k for i in range(n_docs)], dtype=np.int64)
    m_z = np.zeros(k, dtype=np.float64)
    n_z = np.zeros(k, dtype=np.float64)
    n_zw = np.zeros((k, V), dtype=np.float64)
    for d, doc in enumerate(docs):
        z = int(d_z[d])
        m_z[z] += 1
        n_z[z] += len(doc)
        for w in doc:
            n_zw[z, w] += 1

    def doc_log_likelihood(doc):
        counts = {}
        for w in doc:
            counts[w] = counts.get(w, 0) + 1
        words = np.asarray(sorted(counts), dtype=int)
        counts_arr = np.asarray([counts[w] for w in words], dtype=float)
        ll = np.zeros(k)
        for idx, (w, c) in enumerate(zip(words, counts_arr)):
            ll += gammaln(n_zw[:, w] + beta + c) - gammaln(n_zw[:, w] + beta)
        length = len(doc)
        ll -= gammaln(n_z + V * beta + length) - gammaln(n_z + V * beta)
        return ll

    change_history: List[int] = []
    for iteration in range(1, max_iterations + 1):
        changes = 0
        for d, doc in enumerate(docs):
            old = int(d_z[d])
            m_z[old] -= 1
            n_z[old] -= len(doc)
            for w in doc:
                n_zw[old, w] -= 1

            logp = np.log(m_z + alpha) + doc_log_likelihood(doc)
            logp -= np.max(logp)
            p = np.exp(logp)
            p = p / (p.sum() + 1e-12)
            if not np.all(np.isfinite(p)) or p.sum() <= 0:
                p = np.full(k, 1.0 / k)
            new = int(rng.choice(k, p=p))
            if new != old:
                changes += 1
            d_z[d] = new
            m_z[new] += 1
            n_z[new] += len(doc)
            for w in doc:
                n_zw[new, w] += 1
        change_history.append(changes)
        if iteration >= 40 and changes <= max(1, int(0.01 * n_docs)):
            break

    phi = (n_zw + beta) / (n_z[:, None] + V * beta)
    occupied = [z for z in range(k) if m_z[z] > 0]
    order = sorted(occupied, key=lambda z: -m_z[z])
    beta_out = phi[order]
    topics_out = [
        [vocab[idx] for idx in np.argsort(-beta_out[i])[:TOPN]]
        for i in range(len(order))
    ]
    return {
        "beta": beta_out,
        "topics": topics_out,
        "topic_count": len(order),
        "occupied": len(order),
        "iterations_run": len(change_history),
        "change_history": change_history,
        "labels": d_z.tolist(),
    }


def run_nmf(
    count_matrix: np.ndarray,
    vocab: List[str],
    k: int,
    seed: int,
) -> Dict[str, object]:
    """NMF on raw counts with KL beta-divergence (topic-factorization baseline)."""
    model = SklearnNMF(
        n_components=k,
        init="nndsvda",
        beta_loss="kullback-leibler",
        solver="mu",
        max_iter=600,
        random_state=seed,
        tol=1e-4,
    )
    w = model.fit_transform(count_matrix)   # document-topic weights
    h = np.asarray(model.components_, dtype=float)  # (K, V)
    topics = [ [vocab[i] for i in np.argsort(-h[j])[:TOPN]] for j in range(k) ]
    # NMF reconstructs W H, which is invariant to rescaling column k of W and
    # row k of H by reciprocal factors; a raw W is therefore not a document
    # mixture, and normalising W alone would depend on that arbitrary gauge.
    # Return the topic-word distribution and the mixture for which
    # ``theta @ beta == normalize(W H)``.
    beta = h / np.maximum(h.sum(axis=1, keepdims=True), 1e-12)
    mass = h.sum(axis=1)
    mix = np.asarray(w, dtype=float) * mass[None, :]
    mix = mix / np.maximum(mix.sum(axis=1, keepdims=True), 1e-12)
    return {"beta": beta, "topics": topics, "theta": mix,
            "topic_count": k, "n_iter": int(model.n_iter_)}


def _biterms_from_tokens(
    token_lists,
    dictionary,
    window: int = 12,
    max_pairs: int = 150,
    seed: int = 0,
    per_document_rng: bool = False,
):
    """Collect word pairs that co-occur inside a sliding window.

    The pairs kept are ``(i, j)`` with ``1 <= j - i < window`` (the window
    parameter is therefore the *exclusive* upper bound on the distance).

    ``window`` and ``max_pairs`` are substantive modelling choices (the
    biterm sampler sees at most ``max_pairs`` pairs per document); they are
    part of the reported protocol and are recorded in the returned record.

    With ``per_document_rng=True`` every document draws its kept pairs from a
    generator derived from *its own text* and the experiment seed, so the pairs
    do not depend on which other documents share the batch (and a document gets
    the same pairs in any process).  The default (``False``) consumes a single
    random stream in document order, which is how the published Gibbs fits were
    trained; that behaviour is kept so the fitted counts stay reproducible.
    Inference uses the per-document generator.
    """
    import zlib

    word2id = dictionary.token2id
    out = []
    owners = []          # document index of every emitted biterm
    shared_rng = None if per_document_rng else np.random.RandomState(seed)
    for d, tokens in enumerate(token_lists):
        ids = [word2id[w] for w in tokens if w in word2id]
        n = len(ids)
        if n < 2:
            continue
        pairs = []
        for i in range(n):
            for j in range(i + 1, min(n, i + window)):
                pairs.append((ids[i], ids[j]))
        if len(pairs) > max_pairs:
            if per_document_rng:
                stable = zlib.crc32(" ".join(tokens).encode("utf-8"))
                rng = np.random.RandomState([int(seed), int(stable)])
            else:
                rng = shared_rng
            pairs = [pairs[i] for i in
                     rng.choice(len(pairs), size=max_pairs, replace=False)]
        out.extend(pairs)
        owners.extend([d] * len(pairs))
    return out, owners


def check_simplex(matrix, name: str = "distribution", tol: float = 1e-6):
    """Refuse to score anything that is not a proper probability matrix.

    Every document-level prediction in the occlusion protocol is
    ``p(w|d) = theta_d @ beta``, so a row that is not a normalised,
    non-negative distribution silently turns into ``-log(1e-12) = 27.63``
    per token.  The review found two such rows (BTM without biterms, NMF with
    zero visible mass); this check makes that class of bug impossible to score
    without noticing.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"{name}: expected a 2-D matrix, got {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{name}: contains non-finite values")
    if np.any(m < -tol):
        raise ValueError(f"{name}: contains negative probabilities")
    row = m.sum(axis=1)
    if row.size and not np.allclose(row, 1.0, atol=tol):
        bad = int(np.argmax(np.abs(row - 1.0)))
        raise ValueError(
            f"{name}: row {bad} sums to {row[bad]:.6g}, not 1")
    return m


def btm_doc_topic(biterms, owners, n_docs, n_bz, n_wz, alpha, beta, V,
                  form: str = "point"):
    """Document-topic posterior of BTM, ``p(z|d) = sum_b p(b|d) p(z|b)``.

    ``form="point"`` (the default) follows the equation printed in Yan et al.
    (2013), §3.3: the point estimates of the fitted parameters are plugged into
    ``p(z=k|b) ∝ π_k φ_{k,w_i} φ_{k,w_j}`` with
    ``φ_kw = (n_kw + beta) / (2 n_k + V beta)`` and ``π_k ∝ n_k + alpha``, and
    each biterm of the document carries equal weight ``p(b|d)``.

    ``form="predictive"`` instead integrates the same Dirichlet posterior out,
    which yields the ``1/(S_k(S_k+1))`` denominator and the ``+1`` term for
    ``w_i == w_j``.  That is the form of the *Gibbs conditional* used while
    sampling, not the equation of §3.3; it is kept as a sensitivity check.

    A document whose visible text yields no biterm has no ``p(b|d)`` to average
    over; it falls back to the corpus-level topic prior ``π`` (the same
    quantity ``form="point"`` normalises), so every row is a distribution.
    """
    if form not in ("point", "predictive"):
        raise ValueError(f"unknown BTM posterior form: {form!r}")
    K = n_bz.shape[0]
    theta = np.zeros((n_docs, K), dtype=float)
    for b_idx, (w1, w2) in enumerate(biterms):
        if form == "point":
            logp = (np.log(n_bz + alpha)
                    + np.log(n_wz[:, w1] + beta)
                    + np.log(n_wz[:, w2] + beta)
                    - 2.0 * np.log(2.0 * n_bz + V * beta))
        else:
            same = 1.0 if w1 == w2 else 0.0
            logp = (np.log(n_bz + alpha)
                    + np.log(n_wz[:, w1] + beta)
                    + np.log(n_wz[:, w2] + beta + same)
                    - np.log(2.0 * n_bz + V * beta)
                    - np.log(2.0 * n_bz + V * beta + 1.0))
        logp -= logp.max()
        p = np.exp(logp)
        total = p.sum()
        if total > 0:
            theta[owners[b_idx]] += p / total
    row = theta.sum(axis=1, keepdims=True)
    prior = (n_bz + alpha) / np.sum(n_bz + alpha)
    empty = row[:, 0] <= 0
    if np.any(empty):
        theta[empty] = prior
        row = theta.sum(axis=1, keepdims=True)
    return check_simplex(theta / row, "BTM p(z|d)")


def _btm_conditional(
    n_bz, n_wz, w1, w2, alpha, beta, V
):
    """log p(z_b = k | z_-b) for every topic k (Yan et al. 2013, eq. 4).

    After integrating out the topic-word Dirichlet,

        p(w_i, w_j | k) = (n_kw_i + eta)(n_kw_j + eta + [w_i = w_j])
                          / [(2 n_k + V eta)(2 n_k + V eta + 1)]

    so the conditional is

        p(z_b = k | -b) \\propto (n_k^-b + alpha) * p(w_i, w_j | k).

    The ``+1`` in the second denominator factor and the ``[w_i = w_j]``
    indicator are both required for the collapsed Dirichlet-multinomial
    posterior; omitting them (as an earlier revision of this file did)
    makes the sampler an approximation rather than BTM.
    """
    same = 1.0 if w1 == w2 else 0.0
    return (
        np.log(n_bz + alpha)
        + np.log(n_wz[:, w1] + beta)
        + np.log(n_wz[:, w2] + beta + same)
        - np.log(2.0 * n_bz + V * beta)
        - np.log(2.0 * n_bz + V * beta + 1.0)
    )


def _btm_gibbs_numpy(biterms, k, V, alpha, beta, iterations, seed):
    """Fallback pure-numpy BTM Gibbs (used only if numba unavailable)."""
    rng = np.random.RandomState(seed)
    n_b = len(biterms)
    z = rng.randint(0, k, size=n_b).astype(np.int64)
    n_bz = np.zeros(k, dtype=np.float64)
    n_wz = np.zeros((k, V), dtype=np.float64)
    for idx, (w1, w2) in enumerate(biterms):
        topic = int(z[idx])
        n_bz[topic] += 1
        n_wz[topic, w1] += 1
        n_wz[topic, w2] += 1
    for _ in range(iterations):
        for idx in range(n_b):
            w1 = biterms[idx][0]
            w2 = biterms[idx][1]
            old = int(z[idx])
            n_bz[old] -= 1
            n_wz[old, w1] -= 1
            n_wz[old, w2] -= 1
            logp = _btm_conditional(n_bz, n_wz, w1, w2, alpha, beta, V)
            logp -= np.max(logp)
            p = np.exp(logp)
            p = p / (p.sum() + 1e-12)
            if not np.all(np.isfinite(p)) or p.sum() <= 0:
                p = np.full(k, 1.0 / k)
            new = int(rng.choice(k, p=p))
            z[idx] = new
            n_bz[new] += 1
            n_wz[new, w1] += 1
            n_wz[new, w2] += 1
    return z, n_bz, n_wz


if njit is not None:
    @njit(cache=True)
    def _btm_gibbs_jit(biterms, k, V, alpha, beta, iterations, seed):
        np.random.seed(seed)
        n_b = biterms.shape[0]
        z = np.random.randint(0, k, size=n_b).astype(np.int64)
        n_bz = np.zeros(k, dtype=np.float64)
        n_wz = np.zeros((k, V), dtype=np.float64)
        for idx in range(n_b):
            topic = int(z[idx])
            w1 = int(biterms[idx, 0])
            w2 = int(biterms[idx, 1])
            n_bz[topic] += 1
            n_wz[topic, w1] += 1
            n_wz[topic, w2] += 1
        for _ in range(iterations):
            for idx in range(n_b):
                w1 = int(biterms[idx, 0])
                w2 = int(biterms[idx, 1])
                old = int(z[idx])
                n_bz[old] -= 1
                n_wz[old, w1] -= 1
                n_wz[old, w2] -= 1
                # [w1 == w2] enables the squared Dirichlet moment when both
                # words of a biterm are the same vocabulary item.
                same = 1.0 if w1 == w2 else 0.0
                max_logp = -1e300
                for topic in range(k):
                    value = (
                        np.log(n_bz[topic] + alpha)
                        + np.log(n_wz[topic, w1] + beta)
                        + np.log(n_wz[topic, w2] + beta + same)
                        - np.log(2.0 * n_bz[topic] + V * beta)
                        - np.log(2.0 * n_bz[topic] + V * beta + 1.0)
                    )
                    if value > max_logp:
                        max_logp = value
                weights = np.zeros(k, dtype=np.float64)
                total = 0.0
                for topic in range(k):
                    value = (
                        np.log(n_bz[topic] + alpha)
                        + np.log(n_wz[topic, w1] + beta)
                        + np.log(n_wz[topic, w2] + beta + same)
                        - np.log(2.0 * n_bz[topic] + V * beta)
                        - np.log(2.0 * n_bz[topic] + V * beta + 1.0)
                    )
                    weights[topic] = np.exp(value - max_logp)
                    total += weights[topic]
                if total <= 0 or not np.isfinite(total):
                    weights[:] = 1.0 / k
                    total = 1.0
                else:
                    weights /= total
                rnd = np.random.random()
                cum = 0.0
                new = k - 1
                for topic in range(k):
                    cum += weights[topic]
                    if rnd < cum:
                        new = topic
                        break
                z[idx] = new
                n_bz[new] += 1
                n_wz[new, w1] += 1
                n_wz[new, w2] += 1
        return z, n_bz, n_wz


def _btm_gibbs(biterms, k, V, alpha, beta, iterations, seed):
    arr = np.asarray(biterms, dtype=np.int64)
    if njit is not None:
        return _btm_gibbs_jit(arr, k, V, alpha, beta, iterations, seed)
    return _btm_gibbs_numpy(biterms, k, V, alpha, beta, iterations, seed)


def run_btm(
    token_lists: List[List[str]],
    dictionary,
    vocab: List[str],
    k: int,
    seed: int,
    alpha: float = None,
    beta: float = 0.01,
    iterations: int = 300,
    window: int = 12,
    max_pairs: int = 150,
) -> Dict[str, object]:
    """Biterm topic model (Yan et al. 2013), collapsed Gibbs implementation.

    Sampling choices that are part of the protocol and are reported with the
    result: biterms are all word pairs within a ``window`` of 12 tokens, at
    most ``max_pairs`` sampled pairs per document, ``alpha = 50/K``,
    ``beta = 0.01`` and ``iterations`` full Gibbs sweeps.
    """
    if alpha is None:
        alpha = 50.0 / k
    biterms, owners = _biterms_from_tokens(
        token_lists, dictionary, window, max_pairs=max_pairs, seed=seed)
    if not biterms:
        raise ValueError("BTM: no biterms extracted")
    n_b = len(biterms)
    V = len(vocab)
    z, n_bz, n_wz = _btm_gibbs(biterms, k, V, alpha, beta, iterations, seed)
    n_wz_sum = n_wz.sum(axis=1, keepdims=True)
    phi = (n_wz + beta) / (n_wz_sum + V * beta)
    topics = [[vocab[i] for i in np.argsort(-phi[j])[:TOPN]] for j in range(k)]
    theta = btm_doc_topic(biterms, owners, len(token_lists), n_bz, n_wz,
                          alpha, beta, V)
    return {
        "beta": phi,
        "topics": topics,
        # document-topic posterior (Yan et al. §3.3) and the fitted counts,
        # exposed so BTM can join the class-TF-IDF read-out and the occlusion
        # protocol like every other model
        "theta": theta,
        "n_bz": n_bz,
        "n_wz": n_wz,
        "doc_owners": owners,
        "topic_count": k,
        "n_biterms": n_b,
        "iterations_run": iterations,
        "window": int(window),
        "max_pairs": int(max_pairs),
        "alpha": float(alpha),
        "beta_prior": float(beta),
    }
