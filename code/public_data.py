# -*- coding: utf-8 -*-
"""AG News public benchmark data preparation (dual-field: title + description).

The official AG News CSV repository is cloned separately (train.csv / test.csv).
We draw a balanced, seeded subsample so that every model (including the neural
ones) can be trained on a plain CPU within a reasonable time.  The split is
written to disk so the exact rows used in the paper are reproducible.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary

import config

# Raw AG News CSVs.  The default points at the local checkout used for this
# study; override with the AGNEWS_SOURCE environment variable (the only thing
# required is a directory holding the official train.csv / test.csv).
AGNEWS_SOURCE = Path(
    os.environ.get(
        "AGNEWS_SOURCE",
        str(Path(__file__).resolve().parent / "agnews_repo" / "data"
            / "ag_news_csv"),
    )
)
PUBLIC_DIR = config.DATA_DIR / "agnews"

CLASS_NAMES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

_EN_TOKEN = re.compile(r"[a-z]+")

EN_STOPWORDS = set(
    """
    the a an and or of to in for on with at from by as is are was were be been being
    it its this that these those they them their he she his her we us our you your
    i me my not no do does did will would can could should may might must have has
    had there here when where which who whom what why how more most much many each
    other another such same only own even still just too very than then so but yet
    also new said say says according about after before between during over under
    into out up down off against among through across within without via per while
    since until though although however therefore if because nor once both either
    reuters ap afp xinhua press associated agency quot amp com www http https html
    nbsp target href
    """.split()
)


def en_tokenize(text: object) -> List[str]:
    out = []
    for w in _EN_TOKEN.findall(str(text).lower()):
        if len(w) > 2 and w not in EN_STOPWORDS:
            out.append(w)
    return out


def _load_raw(kind: str) -> pd.DataFrame:
    df = pd.read_csv(
        AGNEWS_SOURCE / f"{kind}.csv",
        header=None,
        names=["label", "title", "description"],
    )
    df["label"] = df["label"].astype(int)
    return df


def prepare_splits(
    train_per_class: int = 1500,
    test_per_class: int = 1900,
    seed: int = 20260905,
    force: bool = False,
) -> None:
    """Create (or reuse) balanced subsample files for train and test."""
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    out_train = PUBLIC_DIR / "train.csv"
    out_test = PUBLIC_DIR / "test.csv"
    if out_train.exists() and out_test.exists() and not force:
        # 2026-10-15 v3 review C4: a cached split must match the requested
        # sizes instead of silently serving a different sample
        n_train = len(pd.read_csv(out_train))
        n_test = len(pd.read_csv(out_test))
        want_train = 4 * int(train_per_class)
        want_test = 4 * int(test_per_class)
        if (n_train, n_test) != (want_train, want_test):
            raise ValueError(
                f"cached AG News splits have {n_train}/{n_test} rows but "
                f"{want_train}/{want_test} rows were requested; pass "
                "force_splits=True to rebuild them")
        return
    rng = random.Random(seed)
    train_rows, test_rows = [], []
    for kind, out, per_class in (("train", out_train, train_per_class), ("test", out_test, test_per_class)):
        df = _load_raw(kind)
        rows = []
        for label in sorted(df["label"].unique()):
            idx = df.index[df["label"] == label].tolist()
            chosen = rng.sample(idx, per_class)
            sub = df.loc[chosen]
            sub = sub.copy()
            sub["split"] = kind
            sub["row_id"] = [f"{kind}-{label}-{i}" for i in range(per_class)]
            rows.append(sub)
        frame = pd.concat(rows, ignore_index=True)
        frame.to_csv(out, index=False, encoding="utf-8")
    print("saved", out_train, "rows:", len(pd.read_csv(out_train)))
    print("saved", out_test, "rows:", len(pd.read_csv(out_test)))


def _build_corpus(
    df: pd.DataFrame,
    train_dictionary: Dictionary,
    train_vocab: List[str],
    need_bow: bool = True,
    need_doc_bow: bool = True,
) -> Dict[str, object]:
    """Turn a split frame into the corpus dict used by models/metrics.

    Field roles follow the same convention as the GitHub corpus so that both
    public benchmarks describe one document the same way:

        reason_* (field 1) = news description  (background / "why")
        item_*   (field 2) = news title        (concise / "what")

    Both fields are always present in AG News, so has_cause is True everywhere.
    """
    title_tokens = [en_tokenize(t) for t in df["title"]]
    desc_tokens = [en_tokenize(d) for d in df["description"]]
    doc_tokens = [a + b for a, b in zip(title_tokens, desc_tokens)]

    vocab_set = set(train_vocab)
    word2id = train_dictionary.token2id
    doc_tokens = [[w for w in ts if w in vocab_set] for ts in doc_tokens]
    title_tokens = [[w for w in ts if w in vocab_set] for ts in title_tokens]
    desc_tokens = [[w for w in ts if w in vocab_set] for ts in desc_tokens]

    reason_bow = item_bow = doc_bow = None
    if need_bow:
        reason_bow = np.zeros((len(doc_tokens), len(train_vocab)), dtype=np.float32)
        item_bow = np.zeros_like(reason_bow)
        if need_doc_bow:
            doc_bow = np.zeros_like(reason_bow)
        for d, (desc, title, dt) in enumerate(zip(desc_tokens, title_tokens, doc_tokens)):
            for w in desc:
                reason_bow[d, word2id[w]] += 1.0
            for w in title:
                item_bow[d, word2id[w]] += 1.0
            if need_doc_bow:
                for w in dt:
                    doc_bow[d, word2id[w]] += 1.0

    doc_corpus = [train_dictionary.doc2bow(ts) for ts in doc_tokens]
    return {
        "df": df,
        "labels": df["label"].astype(int).tolist(),
        "reason_texts": df["description"].tolist(),
        "item_texts": df["title"].tolist(),
        "doc_texts": (df["title"] + ". " + df["description"]).tolist(),
        "doc_tokens": doc_tokens,
        "reason_tokens": desc_tokens,
        "item_tokens": title_tokens,
        "dictionary": train_dictionary,
        "vocab": train_vocab,
        "corpus": doc_corpus,
        "reason_bow": reason_bow,
        "item_bow": item_bow,
        "doc_bow": doc_bow,
        "n_docs": len(doc_tokens),
        "n_vocab": len(train_vocab),
    }


def load_agnews(
    train_per_class: int = 1500,
    test_per_class: int = 1900,
    no_below: int = 5,
    no_above: float = 0.7,
    keep_n: int = 6000,
    force_splits: bool = False,
    need_train_bow: bool = True,
    need_doc_bow: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Load prepared splits and return {"train": ..., "test": ...}.

    ``need_train_bow=False`` skips the (6000 x V) count matrices for models
    that only consume ``doc_tokens``/``corpus`` (LDA, BTM), which keeps the
    memory footprint of the classical baselines small.
    """
    prepare_splits(train_per_class, test_per_class, force=force_splits)
    train_df = pd.read_csv(PUBLIC_DIR / "train.csv")
    test_df = pd.read_csv(PUBLIC_DIR / "test.csv")

    all_tokens = [en_tokenize(t) + en_tokenize(d) for t, d in zip(train_df["title"], train_df["description"])]
    dictionary = Dictionary(all_tokens)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dictionary.compactify()
    vocab = [dictionary[i] for i in range(len(dictionary))]

    return {
        "name": "AGNews",
        "meta": {
            "train_per_class": int(train_per_class),
            "test_per_class": int(test_per_class),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "split_seed": 20260905,
            "no_below": no_below,
            "no_above": no_above,
            "keep_n": keep_n,
            "source": "AG News (official train/test CSVs)",
        },
        "train": _build_corpus(
            train_df, dictionary, vocab, need_bow=need_train_bow,
            need_doc_bow=need_doc_bow),
        "test": _build_corpus(test_df, dictionary, vocab, need_bow=False),
    }


def stats(corpus: Dict[str, object]) -> Dict[str, float]:
    doc_tokens = corpus["doc_tokens"]
    return {
        "n_docs": len(doc_tokens),
        "mean_tokens": float(np.mean([len(t) for t in doc_tokens])),
        "median_tokens": float(np.median([len(t) for t in doc_tokens])),
        "mean_desc": float(np.mean([len(t) for t in corpus["reason_tokens"]])),
        "mean_title": float(np.mean([len(t) for t in corpus["item_tokens"]])),
    }


if __name__ == "__main__":
    data = load_agnews()
    print(data["meta"])
    print("train", stats(data["train"]))
    print("test", stats(data["test"]))
    print("vocab", data["train"]["n_vocab"])
