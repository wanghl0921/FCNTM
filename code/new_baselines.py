# -*- coding: utf-8 -*-
"""Additional published baselines: BERTopic, Top2Vec, CombinedTM (official), NeuralLDA."""
from __future__ import annotations

import os
import tempfile
import contextlib
import io
from typing import Dict, List

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="nbcache_"))

from config import NUM_EPOCHS, PATIENCE, TOPN  # noqa: E402
from metrics import topics_from_matrix  # noqa: E402


def _topic_sums(bow, assignments: np.ndarray, topic_ids: List[int]) -> np.ndarray:
    """Sum the bag-of-words of the documents assigned to every topic.

    Kept sparse on the way in: densifying a 6000 x 8000 float matrix costs a
    few hundred MB and this box only has 8 GB.
    """
    from scipy import sparse as sp

    assignments = np.asarray(assignments).astype(int)
    index = {t: i for i, t in enumerate(topic_ids)}
    docs = np.nonzero(assignments >= 0)[0]
    row_of = np.array([index[int(assignments[d])] for d in docs], dtype=int)
    n_topics = len(topic_ids)
    if sp.issparse(bow):
        indicator = sp.csr_matrix(
            (np.ones(len(docs), dtype=np.float32), (row_of, docs)),
            shape=(n_topics, bow.shape[0]),
        )
        return np.asarray((indicator @ bow).todense(), dtype=float)
    counts = np.asarray(bow, dtype=float)
    sums = np.zeros((n_topics, counts.shape[1]), dtype=float)
    np.add.at(sums, row_of, counts[docs])
    return sums


def class_tfidf_readout(
    bow,
    assignments: np.ndarray,
    vocab: List[str],
    topn: int = TOPN,
):
    """Dense topic-word read-out on a shared vocabulary.

    BERTopic and Top2Vec report a handful of representative words from their
    *own* (unfiltered) vocabulary, which makes their topic word lists a
    different length from every other model's top-``TOPN`` list and therefore
    not comparable.  This builds the same read-out for them: aggregate the
    bag-of-words of every document assigned to a topic, apply the class based
    c-TF-IDF normalisation, and take the top ``topn`` words of each row.  The
    result is a dense (n_topics, V) score matrix over the evaluation
    vocabulary, exactly like the other models produce.

    ``assignments`` holds the topic id of every training document; -1 marks
    unassigned (outlier) documents and is dropped.
    """
    assignments = np.asarray(assignments).astype(int)
    topic_ids = [t for t in sorted(set(assignments.tolist())) if t >= 0]
    S = _topic_sums(bow, assignments, topic_ids)
    row_sum = S.sum(axis=1)
    tf = np.zeros_like(S)
    keep = row_sum > 1e-12
    tf[keep] = S[keep] / row_sum[keep, None]
    df = S.sum(axis=0)
    avg = float(row_sum[keep].mean()) if keep.any() else 1.0
    idf = np.log(avg / np.maximum(df, 1e-12) + 1.0)
    scores = tf * idf
    return scores, topic_ids, topics_from_matrix(scores, vocab, topn)


def run_bertopic(
    docs_for_embedding: List[str],
    docs_for_bow: List[str],
    embeddings: np.ndarray,
    vocab: List[str],
    k: int,
    seed: int,
    umap_kwargs: dict | None = None,
    bow=None,
    topn: int = TOPN,
    readout: str = "ctfidf",
) -> Dict[str, object]:
    """BERTopic (Grootendorst 2022) with fixed UMAP/HDBSCAN seeds."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    # Restrict the internal bag-of-words to the shared evaluation vocabulary.
    # The old unfiltered vectorizer built a c-TF-IDF matrix over every token in
    # the corpus, which on the long GitHub issue bodies needed more memory than
    # this 8 GB box has (MemoryError on all three seeds) -- and it was also the
    # source of the inconsistent read-out, since the topic word lists were
    # drawn from a different vocabulary than every other model's.
    vectorizer = CountVectorizer(lowercase=True, vocabulary=list(vocab))
    umap_opts = {
        "n_neighbors": 12,
        "n_components": 5,
        "min_dist": 0.0,
        "metric": "cosine",
        "random_state": seed,
    }
    if umap_kwargs:
        umap_opts.update(umap_kwargs)
    umap_model = UMAP(**umap_opts)
    hdbscan_model = HDBSCAN(
        min_cluster_size=8, min_samples=5, metric="euclidean", gen_min_span_tree=True
    )
    model = BERTopic(
        language="multilingual",
        embedding_model=None,
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=8,
        calculate_probabilities=False,
        verbose=False,
    )
    # small corpora can leave words with zero topic-level document frequency,
    # which makes representative-document similarity contain NaN; skip that step
    model._save_representative_docs = lambda documents: None
    _, _ = model.fit_transform(docs_for_bow, embeddings=np.asarray(embeddings, dtype=np.float32))
    assignments = model.topics_
    actual = len(set(assignments)) - (1 if -1 in set(assignments) else 0)
    if actual > k:
        try:
            model.reduce_topics(docs_for_bow, nr_topics=k)
        except Exception:  # noqa: BLE001
            pass
        assignments = model.topics_
    info = model.get_topic_info()
    topic_sizes = dict(zip(info["Topic"], info["Count"]))
    info = info[info["Topic"] != -1].sort_values("Topic")
    rows = info["Topic"].tolist()
    topic_words = model.get_topics()
    if readout == "ctfidf" and bow is not None:
        # Uniform read-out: aggregate the training bag-of-words by topic and
        # apply the class c-TF-IDF, so every model reports the same number of
        # topic words over the same vocabulary.  Without this, BERTopic emits
        # only the handful of its own vectorizer's words that survive the
        # vocabulary filter, and a 2-3 word topic list is not comparable with
        # a 10 word one under C_V.
        beta, rows, word_topics = class_tfidf_readout(
            bow, assignments, vocab, topn)
    else:
        vocab_index = {w: i for i, w in enumerate(vocab)}
        beta = np.zeros((len(rows), len(vocab)), dtype=float)
        word_topics = []
        for row_idx, topic_id in enumerate(rows):
            kept = []
            for w, score in topic_words.get(topic_id, []):
                if w in vocab_index:
                    beta[row_idx, vocab_index[w]] = float(score)
                    kept.append(w)
                if len(kept) >= topn:
                    break
            if not kept:
                # no overlap with the evaluation vocab: fall back to the row's
                # own strongest vocabulary entries so the topic stays
                # interpretable
                row = beta[row_idx]
                if row.sum() <= 0:
                    kept = vocab[:topn]
                else:
                    idx = np.argsort(-row)[:topn]
                    kept = [vocab[int(i)] for i in idx]
            word_topics.append(kept)
    feats = list(model.vectorizer_model.get_feature_names_out())
    debug_info = {
        "raw_topic_words": [topic_words.get(tid, [])[:10] for tid in rows[:3]],
        "feat_sample": feats[:10],
        "feat_count": len(feats),
        "ctfidf_shape": None if model.c_tf_idf_ is None else list(model.c_tf_idf_.shape),
        "row_sums": [float(model.c_tf_idf_[i].sum()) for i in range(model.c_tf_idf_.shape[0])],
        "topic_sizes": [int(topic_sizes.get(t, -1)) for t in rows[:6]],
        "assign_counts": {int(k): int(v) for k, v in zip(*np.unique(np.asarray(assignments), return_counts=True))},
    }
    return {
        "beta": beta,
        "topics": word_topics,
        "topic_count": len(rows),
        "n_outlier_docs": int((np.asarray(assignments) == -1).sum()),
        "debug": debug_info,
    }


def run_top2vec(
    docs_for_bow: List[str],
    vocab: List[str],
    seed: int,
    workers: int = 1,
    umap_kwargs: dict | None = None,
    bow=None,
    topn: int = TOPN,
    readout: str = "ctfidf",
) -> Dict[str, object]:
    """Top2Vec (Angelov 2020) with doc2vec embeddings (no external LM download)."""
    from top2vec import Top2Vec

    umap_opts = {"random_state": seed}
    if umap_kwargs:
        umap_opts.update(umap_kwargs)
    model = Top2Vec(
        documents=docs_for_bow,
        embedding_model="doc2vec",
        min_count=2,
        speed="learn",
        workers=workers,
        topic_merge_delta=0.2,
        verbose=False,
        tokenizer=lambda x: x.split(),
        use_embedding_model_tokenizer=False,
        hdbscan_args={"min_cluster_size": 8, "min_samples": 5},
        umap_args=umap_opts,
    )
    num_topics = model.get_num_topics()
    if readout == "ctfidf" and bow is not None and hasattr(model, "doc_top"):
        # same uniform read-out as BERTopic: topic-level word counts over the
        # shared vocabulary, so top_n is identical across models
        beta, topic_ids, topics = class_tfidf_readout(
            bow, np.asarray(model.doc_top), vocab, topn)
        return {
            "beta": beta,
            "topics": topics,
            "topic_count": len(topic_ids),
            "debug": {"readout": "ctfidf", "top2vec_topics": int(num_topics)},
        }
    returned = model.get_topics(num_topics)
    topic_words, topic_scores = returned[0], returned[1]
    vocab_index = {w: i for i, w in enumerate(vocab)}
    beta = np.zeros((num_topics, len(vocab)), dtype=float)
    for t in range(num_topics):
        for w, score in zip(topic_words[t], topic_scores[t]):
            if w in vocab_index:
                beta[t, vocab_index[w]] = float(score)
    return {
        "beta": beta,
        "topics": topics_from_matrix(beta, vocab, TOPN),
        "topic_count": num_topics,
    }


def run_combinedtm(
    bow_csr,
    contextual_embeddings: np.ndarray,
    vocab: List[str],
    K: int,
    seed: int,
    num_epochs: int = NUM_EPOCHS,
    val_fraction: float = 0.2,
    patience: int = PATIENCE,
) -> Dict[str, object]:
    """Official contextualized-topic-models CombinedTM.

    Trained with the same protocol as the other neural models: a seeded 20%
    held-out split used for early stopping and ``num_epochs`` as the epoch
    budget.  An earlier revision trained for a fixed 300 epochs with no
    validation set, which is why the paper reports the two protocols
    separately.
    """
    import torch

    # NumPy 2 removed the ``np.Inf`` alias that
    # contextualized_topic_models.utils.early_stopping still assigns to, so the
    # official CombinedTM import dies on NumPy >= 2.0 with AttributeError.
    # Restore just that alias for this library (no effect on NumPy 1.x).
    if not hasattr(np, "Inf"):
        np.Inf = np.inf  # type: ignore[attr-defined]

    from contextualized_topic_models.datasets.dataset import CTMDataset
    from contextualized_topic_models.models.ctm import CombinedTM

    torch.manual_seed(seed)
    np.random.seed(seed)
    embeddings = np.asarray(contextual_embeddings, dtype=np.float32)
    n_docs = bow_csr.shape[0]
    rng = np.random.RandomState(seed + 12345)
    perm = rng.permutation(n_docs)
    n_val = max(1, int(round(n_docs * val_fraction)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    dataset = CTMDataset(embeddings[train_idx], bow_csr[train_idx],
                         idx2token=vocab)
    val_dataset = CTMDataset(embeddings[val_idx], bow_csr[val_idx],
                             idx2token=vocab)
    model = CombinedTM(
        bow_size=bow_csr.shape[1],
        contextual_size=contextual_embeddings.shape[1],
        n_components=K,
        num_epochs=int(num_epochs),
        lr=0.002,
        batch_size=64,
        dropout=0.2,
        activation="softplus",
        learn_priors=True,
        model_type="prodLDA",
        num_data_loader_workers=0,
    )
    # The library's early stopper (created inside ``fit``) falls back to the
    # *relative* path "checkpoint.pt" when no save directory is given, and its
    # experimental, never-overwriting ``save`` then writes a ~37 MB
    # ``epoch_N.pth`` into a fresh sub-directory on every validation
    # improvement.  During the 3-seed rerun that left 87 files (3.1 GB) in the
    # working directory and filled the C: drive.  Nothing in this project ever
    # loads those checkpoints, so neutralise the saver itself; the early
    # stopping logic (counter/patience/best loss) is untouched.
    model.save = lambda *args, **kwargs: None  # noqa: ARG005
    # the official training loop reports every epoch through tqdm regardless of
    # `verbose`; silence that progress stream while keeping model.fit intact
    with contextlib.redirect_stderr(io.StringIO()):
        model.fit(dataset, validation_dataset=val_dataset, verbose=False,
                  patience=int(patience), delta=0.0,
                  do_train_predictions=False)
    # ``get_topic_word_matrix`` returns the *current* network's topic-word
    # matrix, but the early stopper keeps the best validation epoch in
    # ``best_components`` (its ``__call__`` clones ``model.model.beta`` whenever
    # the validation loss improves).  With early stopping the two differ, and
    # the paper's protocol selects on the validation loss, so score the best
    # state -- otherwise CombinedTM would silently report the last epoch while
    # every other neural baseline reports its best one.
    best = getattr(model, "best_components", None)
    if best is not None:
        beta = np.asarray(best.detach().cpu(), dtype=float)
    else:
        beta = np.asarray(model.get_topic_word_matrix(), dtype=float)
    return {
        "beta": beta,
        "topics": topics_from_matrix(beta, vocab, TOPN),
        "topic_count": beta.shape[0],
    }


def run_neural_lda(corpus, prep, K, seed, bow_key="doc_bow"):
    """NeuralLDA (Srivastava & Sutton 2017 AVITM style), single-stream linear-decoder VAE."""
    from experiment import _run_prodlda

    return _run_prodlda(corpus, K, seed, prep[bow_key])
