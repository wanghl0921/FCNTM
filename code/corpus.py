# -*- coding: utf-8 -*-
"""Corpus preparation for the FC-NTM paper (self-contained)."""

from __future__ import annotations

import ast
import re
from collections import Counter
from typing import Dict, List, Tuple

import jieba
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary

import config
from config import CLEANED_CORPUS, DOMAIN_USER_WORDS


# ---------------------------------------------------------------------------
# tokenization
# ---------------------------------------------------------------------------

STOPWORDS = set(
    """
工程 增加 变更 调整 取消 设置 用于 根据 进行 以及 要求 部位 内容 情况 处理 相关
说明 记录 明确 如下 详见 详图 具体 方式 包括 主要 部分 改造 采用 使用 单位 本次
其他 完成 需要 可以 符合 是否 经 后 中 上 下 内 外 前 后 需 应 请 按 见 图 号 编号
""".split()
)

_CJK = re.compile(r"[\u4e00-\u9fff]")
_domain_loaded = False


def ensure_jieba_ready() -> None:
    global _domain_loaded
    if _domain_loaded:
        return
    if DOMAIN_USER_WORDS.exists():
        for line in DOMAIN_USER_WORDS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                jieba.add_word(line)
    _domain_loaded = True


def tokenize(text: object) -> List[str]:
    """Tokenize one field or document; keeps Chinese compounds only."""
    ensure_jieba_ready()
    out: List[str] = []
    for w in jieba.cut(str(text)):
        w = w.strip()
        if len(w) < 2:
            continue
        if not _CJK.search(w):
            continue
        if any(ch.isdigit() or ch.isascii() for ch in w):
            continue
        if w in STOPWORDS:
            continue
        out.append(w)
    return out


# ---------------------------------------------------------------------------
# corpus assembly
# ---------------------------------------------------------------------------


def load_corpus() -> Dict[str, object]:
    """Load raw change-order rows and build document/field token lists."""
    df = pd.read_excel(CLEANED_CORPUS)
    df["tokens"] = df["tokens"].map(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    reason_raw = df["变更原因"].fillna("")
    item_raw = df["变更事项"].astype(str)
    reason_tokens = [tokenize(r) for r in reason_raw]
    item_tokens = [tokenize(i) for i in item_raw]
    doc_tokens = [r + i for r, i in zip(reason_tokens, item_tokens)]

    keep = [i for i, t in enumerate(doc_tokens) if t]
    df = df.iloc[keep].reset_index(drop=True)
    reason_tokens = [reason_tokens[i] for i in keep]
    item_tokens = [item_tokens[i] for i in keep]
    doc_tokens = [doc_tokens[i] for i in keep]

    dictionary = Dictionary(doc_tokens)
    dictionary.filter_extremes(
        no_below=config.DICT_NO_BELOW,
        no_above=config.DICT_NO_ABOVE,
        keep_n=config.DICT_KEEP_N,
    )
    dictionary.compactify()

    vocab = [dictionary[i] for i in range(len(dictionary))]
    vocab_index = {w: i for i, w in enumerate(vocab)}

    reason_bow = np.zeros((len(doc_tokens), len(vocab)), dtype=np.float32)
    item_bow = np.zeros_like(reason_bow)
    doc_bow = np.zeros_like(reason_bow)
    for d, (rt, it, dt) in enumerate(zip(reason_tokens, item_tokens, doc_tokens)):
        for w in rt:
            if w in vocab_index:
                reason_bow[d, vocab_index[w]] += 1.0
        for w in it:
            if w in vocab_index:
                item_bow[d, vocab_index[w]] += 1.0
        for w in dt:
            if w in vocab_index:
                doc_bow[d, vocab_index[w]] += 1.0

    doc_corpus = [dictionary.doc2bow(t) for t in doc_tokens]
    return {
        "dataframe": df,
        "reason_texts": reason_raw.iloc[keep].tolist(),
        "item_texts": item_raw.iloc[keep].tolist(),
        "doc_tokens": doc_tokens,
        "reason_tokens": reason_tokens,
        "item_tokens": item_tokens,
        "dictionary": dictionary,
        "vocab": vocab,
        "corpus": doc_corpus,
        "reason_bow": reason_bow,
        "item_bow": item_bow,
        "doc_bow": doc_bow,
        "n_docs": len(doc_tokens),
        "n_vocab": len(vocab),
    }


def vocabulary_stats(corpus: Dict[str, object]) -> Dict[str, float]:
    reason_tokens = corpus["reason_tokens"]
    item_tokens = corpus["item_tokens"]
    doc_tokens = corpus["doc_tokens"]
    return {
        "n_docs": len(doc_tokens),
        "n_vocab": len(corpus["vocab"]),
        "mean_tokens": float(np.mean([len(t) for t in doc_tokens])),
        "median_tokens": float(np.median([len(t) for t in doc_tokens])),
        "docs_with_reason": int(sum(1 for t in reason_tokens if t)),
        "mean_reason_tokens": float(np.mean([len(t) for t in reason_tokens])),
        "mean_item_tokens": float(np.mean([len(t) for t in item_tokens])),
    }


if __name__ == "__main__":
    corpus = load_corpus()
    print(corpus["n_docs"], corpus["n_vocab"])
    print(vocabulary_stats(corpus))
    common = Counter(w for ts in corpus["doc_tokens"] for w in ts)
    print(common.most_common(20))
