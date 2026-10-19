# -*- coding: utf-8 -*-
"""Small-scale versions of the two public corpora.

Full-scale runs cost ~45 min per FC-NTM fit on this CPU, which makes design
iteration impossible.  These loaders subsample the cached splits and rebuild the
dictionary on the subsample, so a whole fit takes minutes.  They are for
iteration and sanity checks only - the paper tables use the full splits.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary

import config
import public_data as PD
import public_data_github as PGH

SMALL_DIR = config.DATA_DIR / "small"

DEFAULTS = {
    "github": dict(n_train=1000, n_test=400, seed=20260910,
                   no_below=3, no_above=0.9, keep_n=3000),
    "agnews": dict(n_train=1000, n_test=400, seed=20260910,
                   no_below=3, no_above=0.9, keep_n=3000),
}


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.reset_index(drop=True)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(df)), n))
    return df.iloc[idx].reset_index(drop=True)


def _write_cache(corpus: str, split: str, df: pd.DataFrame) -> Path:
    SMALL_DIR.mkdir(parents=True, exist_ok=True)
    path = SMALL_DIR / f"{corpus}_{split}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def load_small_github(n_train=None, n_test=None, seed=None, no_below=None,
                      no_above=None, keep_n=None):
    cfg = dict(DEFAULTS["github"])
    for key, val in (("n_train", n_train), ("n_test", n_test), ("seed", seed),
                     ("no_below", no_below), ("no_above", no_above),
                     ("keep_n", keep_n)):
        if val is not None:
            cfg[key] = val
    full_train = pd.read_csv(PGH.GITHUB_DIR / "train.csv")
    full_test = pd.read_csv(PGH.GITHUB_DIR / "test.csv")
    train_df = _subsample(full_train, cfg["n_train"], cfg["seed"])
    test_df = _subsample(full_test, cfg["n_test"], cfg["seed"] + 1)

    tokens = [PGH.en_tokenize_issue(t) + PGH.en_tokenize_issue(PGH.clean_issue_text(b))
              for t, b in zip(train_df["title"], train_df["body"])]
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=cfg["no_below"], no_above=cfg["no_above"],
                               keep_n=cfg["keep_n"])
    dictionary.compactify()
    vocab = [dictionary[i] for i in range(len(dictionary))]

    train = PGH._build_corpus(train_df, dictionary, vocab)
    test = PGH._build_corpus(test_df, dictionary, vocab, need_bow=False)
    train["name"] = "GitHubIssues-small"
    _write_cache("github", "train", train_df)
    _write_cache("github", "test", test_df)
    return {"name": "GitHubIssues-small", "train": train, "test": test,
            "meta": cfg}


def load_small_agnews(n_train=None, n_test=None, seed=None, no_below=None,
                      no_above=None, keep_n=None):
    cfg = dict(DEFAULTS["agnews"])
    for key, val in (("n_train", n_train), ("n_test", n_test), ("seed", seed),
                     ("no_below", no_below), ("no_above", no_above),
                     ("keep_n", keep_n)):
        if val is not None:
            cfg[key] = val
    full_train = pd.read_csv(config.DATA_DIR / "agnews" / "train.csv")
    full_test = pd.read_csv(config.DATA_DIR / "agnews" / "test.csv")
    train_df = _subsample(full_train, cfg["n_train"], cfg["seed"])
    test_df = _subsample(full_test, cfg["n_test"], cfg["seed"] + 1)

    tokens = [PD.en_tokenize(t) + PD.en_tokenize(d)
              for t, d in zip(train_df["title"], train_df["description"])]
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=cfg["no_below"], no_above=cfg["no_above"],
                               keep_n=cfg["keep_n"])
    dictionary.compactify()
    vocab = [dictionary[i] for i in range(len(dictionary))]

    train = PD._build_corpus(train_df, dictionary, vocab)
    test = PD._build_corpus(test_df, dictionary, vocab, need_bow=False)
    train["name"] = "AGNews-small"
    _write_cache("agnews", "train", train_df)
    _write_cache("agnews", "test", test_df)
    return {"name": "AGNews-small", "train": train, "test": test, "meta": cfg}


LOADERS = {"github": load_small_github, "agnews": load_small_agnews}


def load_small(corpus: str, **kwargs):
    return LOADERS[corpus](**kwargs)


def mix_fields(train, rho: float, seed: int):
    """Blend the summary field into the background field by a controlled amount.

    Cross-field reconstruction should only pay off while the two fields carry
    different information.  Comparing GitHub (complementary) with AG News
    (redundant) confounds field structure with domain, document length and
    vocabulary, so build the knob directly instead: replace a fraction ``rho``
    of the summary tokens with the *leading* tokens of that document's own
    background, copied verbatim.  ``rho=0`` keeps the original summary;
    ``rho=1`` makes the summary a verbatim prefix of the background, which is
    the structure AG News already has - its description opens by restating the
    headline - and which makes the background directly recoverable from the
    summary.

    The copy is deterministic on purpose.  An earlier version drew the
    replacement tokens at random from the background, which lowers the measured
    cross-view redundancy instead of raising it: conditional on the background,
    a random subset is still unpredictable, so that knob injects noise rather
    than shared content.  Length, vocabulary and domain stay fixed either way,
    so any change in the value of cross-field reconstruction is attributable to
    the redundancy knob.
    """
    out = dict(train)
    n = len(train["doc_tokens"])
    item_tokens = list(train["item_tokens"])
    rng = random.Random(seed)
    replaced = 0
    total = 0
    for i in range(n):
        body = train["reason_tokens"][i]
        title = list(train["item_tokens"][i])
        total += len(title)
        if not body or not title or rho <= 0:
            continue
        n_replace = int(round(rho * len(title)))
        if n_replace <= 0:
            continue
        take = body[:n_replace]
        item_tokens[i] = title[: len(title) - len(take)] + take
        replaced += len(take)
    out["item_tokens"] = item_tokens
    out["doc_tokens"] = [t + b for t, b in zip(item_tokens, train["reason_tokens"])]

    word2id = train["dictionary"].token2id
    item_bow = np.zeros((n, train["n_vocab"]), dtype=np.float32)
    for d, toks in enumerate(item_tokens):
        for w in toks:
            item_bow[d, word2id[w]] += 1.0
    out["item_bow"] = item_bow
    reason_bow = train["reason_bow"]
    out["doc_bow"] = (reason_bow if reason_bow is not None
                      else np.zeros_like(item_bow)) + item_bow
    out["corpus"] = [train["dictionary"].doc2bow(t) for t in out["doc_tokens"]]
    out["rho"] = rho
    out["replaced_tokens"] = replaced
    out["total_item_tokens"] = total
    return out


def mask_background(train, fraction: float, seed: int):
    """Empty the background field for ``fraction`` of the training documents.

    Experiments under artificial field loss need a corpus whose background
    field is genuinely absent, not merely dropped at encoding time: the point
    is to measure what happens when that evidence never exists.  Everything a
    bag-of-words baseline would see (tokens, BoW, gensim corpus) is masked too,
    so the comparison against LDA/BTM stays fair.
    """
    out = dict(train)
    n = len(train["doc_tokens"])
    out["reason_bow"] = train["reason_bow"].copy()
    out["reason_tokens"] = list(train["reason_tokens"])
    out["item_tokens"] = list(train["item_tokens"])
    out["doc_tokens"] = list(train["doc_tokens"])
    out["reason_texts"] = list(train["reason_texts"])
    out["corpus"] = list(train["corpus"]) if train.get("corpus") else None
    out["masked_flag"] = [False] * n
    if fraction <= 0:
        out["masked_docs"] = 0
        return out
    rng = random.Random(seed)
    chosen = rng.sample(range(n), int(round(fraction * n)))
    for i in chosen:
        out["reason_bow"][i, :] = 0.0
        out["reason_tokens"][i] = []
        out["reason_texts"][i] = ""
        out["doc_tokens"][i] = list(train["item_tokens"][i])
        out["masked_flag"][i] = True
    if out["corpus"] is not None:
        dictionary = train["dictionary"]
        out["corpus"] = [dictionary.doc2bow(t) for t in out["doc_tokens"]]
    if train.get("doc_bow") is not None:
        out["doc_bow"] = out["reason_bow"] + train["item_bow"]
    out["masked_docs"] = len(chosen)
    return out


if __name__ == "__main__":
    for name in ("github", "agnews"):
        d = load_small(name)
        tr, te = d["train"], d["test"]
        print(f"[{name}] train={tr['n_docs']} test={te['n_docs']} vocab={tr['n_vocab']}")
        print("   reason mean toks",
              round(float(np.mean([len(t) for t in tr["reason_tokens"]])), 1),
              "| item mean toks",
              round(float(np.mean([len(t) for t in tr["item_tokens"]])), 1))
