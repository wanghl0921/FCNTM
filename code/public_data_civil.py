# -*- coding: utf-8 -*-
"""Civil-engineering literature corpus (OpenAlex, 标题 + 摘要 两个字段).

来源：OpenAlex（CC0，无需 API key）中的 *Civil and Structural Engineering* 与
*Building and Construction* 两个子领域，2000--2023 年、英文、带摘要的
article / review / conference-paper，按年分层采样（见
``civil_corpus/collect_openalex.py``）。字段映射沿用论文约定：

    reason_* = 摘要（背景：为什么、在什么条件下）
    item_*   = 标题（概要：是什么）

``--max-abstract-words`` 可以把摘要截断到前 N 个词，用来构造"字段长度比"不同的
受控变体（同一个语料、同一个词表，只改字段不对称度），这是论文里那条
"承重组件随字段不对称度变化"曲线的实验材料。
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary

import config
from public_data import _EN_TOKEN, EN_STOPWORDS

CIVIL_DIR = config.DATA_DIR / "civil_openalex"
# the collector writes its raw dump here; a copy lives with the data
RAW_CANDIDATES = [
    CIVIL_DIR / "raw.csv",
    Path(__file__).resolve().parent / "civil_corpus" / "civil_openalex_cc.csv",
    Path(__file__).resolve().parent / "civil_corpus" / "civil_openalex_raw.csv",
    # fallbacks for a checkout that still keeps the collector next to the code
    Path(r"C:\Users\Administrator\Downloads\civil_corpus\civil_openalex_cc.csv"),
    Path(r"C:\Users\Administrator\Downloads\civil_corpus\civil_openalex_raw.csv"),
]

# 学术摘要的样板词：这些词在两个字段里都高频出现但不携带领域信息。
# 只列明显是"写作套路"的词，模型/数据/设计这类在土木语料里仍是内容词，保留。
CIVIL_EXTRA_STOPWORDS = """
paper papers study studies article articles research present presents presented
propose proposes proposed proposing method methods methodology approach
approaches result results show shows showed shown also however therefore thus
moreover furthermore respectively obtained observed investigated discussed
described examined analysis analyse analyze based due using used use first
second third finally new novel different various several many can could may
might must should would consequently additionally overall
""".split()
CIVIL_STOPWORDS = EN_STOPWORDS | set(CIVIL_EXTRA_STOPWORDS)


def clean_civil_text(text: object) -> str:
    """Strip LaTeX fragments, URLs and non-letter noise from an abstract."""
    t = str(text)
    t = re.sub(r"\$[^$]{0,80}\$", " ", t)          # inline math
    t = re.sub(r"\\[a-zA-Z]+", " ", t)             # latex commands
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^A-Za-z\-']+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def en_tokenize_civil(text: object) -> List[str]:
    out = []
    for w in _EN_TOKEN.findall(clean_civil_text(text).lower()):
        if len(w) > 2 and w not in CIVIL_STOPWORDS:
            out.append(w)
    return out


def _raw_path() -> Path:
    for path in RAW_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "civil_openalex_raw.csv not found; run civil_corpus/collect_openalex.py")


def prepare_civil_splits(
    train_n: int = 6000,
    test_n: int = 2000,
    seed: int = 20260919,
    max_abstract_words: int = 0,
    tag: str = "",
    force: bool = False,
) -> Dict[str, object]:
    """Deterministic split; ``max_abstract_words`` builds a field-ratio variant.

    A truncation variant is derived from the *full* split's document IDs
    instead of re-sampling: the earlier implementation filtered on the
    truncated field first and then reshuffled, so abs60/abs30 shared only
    61% / 53% of the full training records and the ladder changed the
    sample as well as the field length (2026-10-07 review, item 5).
    """
    CIVIL_DIR.mkdir(parents=True, exist_ok=True)
    suffix = tag or ("full" if not max_abstract_words else f"abs{max_abstract_words}")
    out_train = CIVIL_DIR / f"train_{suffix}.csv"
    out_test = CIVIL_DIR / f"test_{suffix}.csv"
    if max_abstract_words:
        base = {"train": CIVIL_DIR / "train_full.csv",
                "test": CIVIL_DIR / "test_full.csv"}
        if not all(p.exists() for p in base.values()):
            raise FileNotFoundError(
                "the truncation ladder needs the full split first "
                "(run load_civil() once with max_abstract_words=0)")
        frames = {}
        for name, path in base.items():
            df = pd.read_csv(path)
            df["abstract"] = df["abstract_full"].astype(str).map(
                lambda s: " ".join(str(s).split()[:max_abstract_words]))
            too_thin = int(sum(len(en_tokenize_civil(a)) < 10
                               for a in df["abstract"]))
            if too_thin:
                print(f"  warning: {path.name}: {too_thin} abstracts have "
                      f"fewer than 10 tokens at {max_abstract_words} words; "
                      f"they are kept to preserve the fixed sample",
                      flush=True)
            frames[name] = df
        # 2026-10-15 v4 review R1: the ladder keeps the exact documents of
        # the fixed full split; a request with different sizes must fail
        # instead of silently ignoring the requested sample size
        n_base_train = len(frames["train"])
        n_base_test = len(frames["test"])
        if (n_base_train, n_base_test) != (int(train_n), int(test_n)):
            raise ValueError(
                f"the truncation ladder is derived from the fixed "
                f"{n_base_train}/{n_base_test} full split, but "
                f"{train_n}/{test_n} rows were requested; the ladder keeps "
                "the same documents and does not re-sample")
        frames["train"].to_csv(out_train, index=False, encoding="utf-8")
        frames["test"].to_csv(out_test, index=False, encoding="utf-8")
        print(f"fixed-ID ladder {suffix}: {out_train.name} / "
              f"{out_test.name} "
              f"({len(frames['train'])}+{len(frames['test'])} docs from "
              f"the full split)", flush=True)
        return {"train": str(out_train), "test": str(out_test),
                "cached": False, "fixed_ids": True}
    if out_train.exists() and out_test.exists() and not force:
        # 2026-10-15 v3 review C4: the cached split must match the requested
        # sizes; force_splits now actually rebuilds, and other requests fail
        n_train = len(pd.read_csv(out_train))
        n_test = len(pd.read_csv(out_test))
        if (n_train, n_test) != (int(train_n), int(test_n)):
            raise ValueError(
                f"cached civil splits have {n_train}/{n_test} rows but "
                f"{train_n}/{test_n} rows were requested; pass "
                "force_splits=True to rebuild them")
        return {"train": str(out_train), "test": str(out_test),
                "cached": True, "train_n": n_train, "test_n": n_test}

    df = pd.read_csv(_raw_path())
    df["abstract_full"] = df["abstract"].astype(str)
    if max_abstract_words:
        df["abstract"] = df["abstract_full"].map(
            lambda s: " ".join(str(s).split()[:max_abstract_words]))
    # drop records whose (possibly truncated) fields are too thin to model
    keep = []
    for title, abstract in zip(df["title"], df["abstract"]):
        keep.append(len(en_tokenize_civil(title)) >= 2
                    and len(en_tokenize_civil(abstract)) >= 10)
    df = df[keep].reset_index(drop=True)
    if len(df) < train_n + test_n:
        raise ValueError(f"only {len(df)} usable records for {train_n}+{test_n}")

    rng = random.Random(seed)
    order = list(range(len(df)))
    rng.shuffle(order)
    train_idx = sorted(order[:train_n])
    test_idx = sorted(order[train_n:train_n + test_n])
    cols = ["openalex_id", "doi", "year", "venue", "license", "title", "abstract"]
    df.loc[train_idx, cols + ["abstract_full"]].assign(split="train").to_csv(
        out_train, index=False, encoding="utf-8")
    df.loc[test_idx, cols + ["abstract_full"]].assign(split="test").to_csv(
        out_test, index=False, encoding="utf-8")
    print(f"saved {out_train} ({len(train_idx)}) | {out_test} ({len(test_idx)})")
    return {"train": str(out_train), "test": str(out_test), "cached": False}


def _build_corpus(df: pd.DataFrame, train_dictionary: Dictionary,
                  train_vocab: List[str], need_bow: bool = True,
                  need_doc_bow: bool = True) -> Dict[str, object]:
    abstract_tokens = [en_tokenize_civil(a) for a in df["abstract"]]
    title_tokens = [en_tokenize_civil(t) for t in df["title"]]
    vocab_set = set(train_vocab)
    word2id = train_dictionary.token2id
    abstract_tokens = [[w for w in ts if w in vocab_set] for ts in abstract_tokens]
    title_tokens = [[w for w in ts if w in vocab_set] for ts in title_tokens]
    doc_tokens = [t + a for t, a in zip(title_tokens, abstract_tokens)]

    reason_bow = item_bow = doc_bow = None
    if need_bow:
        reason_bow = np.zeros((len(doc_tokens), len(train_vocab)), dtype=np.float32)
        item_bow = np.zeros_like(reason_bow)
        if need_doc_bow:
            doc_bow = np.zeros_like(reason_bow)
        for d, (rt, it, dt) in enumerate(zip(abstract_tokens, title_tokens, doc_tokens)):
            for w in rt:
                reason_bow[d, word2id[w]] += 1.0
            for w in it:
                item_bow[d, word2id[w]] += 1.0
            if need_doc_bow:
                for w in dt:
                    doc_bow[d, word2id[w]] += 1.0

    clean_abs = df["abstract"].map(clean_civil_text).tolist()
    return {
        "df": df,
        "labels": df["venue"].tolist(),
        "reason_texts": clean_abs,
        "item_texts": df["title"].tolist(),
        "doc_texts": (df["title"] + ". " + pd.Series(clean_abs)).tolist(),
        "doc_tokens": doc_tokens,
        "reason_tokens": abstract_tokens,
        "item_tokens": title_tokens,
        "dictionary": train_dictionary,
        "vocab": train_vocab,
        "corpus": [train_dictionary.doc2bow(ts) for ts in doc_tokens],
        "reason_bow": reason_bow,
        "item_bow": item_bow,
        "doc_bow": doc_bow,
        "n_docs": len(doc_tokens),
        "n_vocab": len(train_vocab),
    }


def load_civil(
    train_n: int = 6000,
    test_n: int = 2000,
    no_below: int = 5,
    no_above: float = 0.7,
    keep_n: int = 8000,
    max_abstract_words: int = 0,
    force_splits: bool = False,
    need_train_bow: bool = True,
    need_doc_bow: bool = True,
) -> Dict[str, object]:
    meta = prepare_civil_splits(train_n, test_n,
                                max_abstract_words=max_abstract_words,
                                force=force_splits)
    suffix = "full" if not max_abstract_words else f"abs{max_abstract_words}"
    train_df = pd.read_csv(CIVIL_DIR / f"train_{suffix}.csv")
    test_df = pd.read_csv(CIVIL_DIR / f"test_{suffix}.csv")

    # the ladder keeps one vocabulary across all three field-lengths: it is
    # built on the untruncated full split, so only the field length changes
    # between variants (2026-10-07 review, item 5)
    vocab_df = (pd.read_csv(CIVIL_DIR / "train_full.csv")
                if max_abstract_words else train_df)
    all_tokens = [en_tokenize_civil(t) + en_tokenize_civil(a)
                  for t, a in zip(vocab_df["title"], vocab_df["abstract"])]
    dictionary = Dictionary(all_tokens)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dictionary.compactify()
    vocab = [dictionary[i] for i in range(len(dictionary))]

    mean_title = float(np.mean([len(en_tokenize_civil(t)) for t in train_df["title"]]))
    mean_abs = float(np.mean([len(en_tokenize_civil(a)) for a in train_df["abstract"]]))
    return {
        "name": "CivilEng-OpenAlex",
        "meta": {
            "source": "OpenAlex subfields 2205 + Building and Construction, "
                      "2000-2023, English, with abstract",
            "train_n": int(train_n), "test_n": int(test_n),
            "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
            "split_seed": 20260919,
            "max_abstract_words": int(max_abstract_words),
            "fixed_ids": bool(meta.get("fixed_ids", False)),
            "fixed_vocabulary": bool(max_abstract_words),
            "mean_title_words": round(mean_title, 2),
            "mean_abstract_words": round(mean_abs, 2),
            "field_length_ratio": round(mean_abs / max(mean_title, 1e-9), 2),
            "no_below": no_below, "no_above": no_above, "keep_n": keep_n,
            "splits": meta,
        },
        "train": _build_corpus(train_df, dictionary, vocab,
                               need_bow=need_train_bow, need_doc_bow=need_doc_bow),
        "test": _build_corpus(test_df, dictionary, vocab, need_bow=False),
    }


if __name__ == "__main__":
    data = load_civil(force_splits=True)
    print(data["meta"])
    for split in ("train", "test"):
        c = data[split]
        print(split, c["n_docs"], "vocab", c["n_vocab"],
              "| mean item toks",
              round(float(np.mean([len(t) for t in c["item_tokens"]])), 1),
              "| mean reason toks",
              round(float(np.mean([len(t) for t in c["reason_tokens"]])), 1))
        print("   sample:", c["item_texts"][0][:70], "||",
              " ".join(c["reason_tokens"][0][:12]))
