# -*- coding: utf-8 -*-
"""GitHub issues public benchmark data (multi-repo, title + body fields).

Source: open-index/open-github-issues (Hugging Face), which stores GitHub API
issue records per repository as parquet files.  We use four mature open-source
repositories from different software domains:

    moby/moby (container runtime), microsoft/TypeScript (compiler / language
    service), duckdb/duckdb (analytics database), redis/redis (cache).

Field mapping mirrors the change-order records: field 1 (reason_*) = issue
body (context/details/"why"), field 2 (item_*) = issue title (concise
"what").  Titles are essentially always present; a non-trivial share of
issues lack a usable body, which we report as the natural missing-field rate.
For the primary train/test benchmark we keep records where both fields carry
enough tokens so that every model is evaluated on the same complete records.
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

GITHUB_DIR = config.DATA_DIR / "github_issues"
SRC_DIR = config.DATA_DIR / "github_issues_src"

REPO_FILES = {
    "duckdb": "duckdb__duckdb.parquet",
    "typescript": "microsoft__TypeScript.parquet",
    "moby": "moby__moby.parquet",
    "redis": "redis__redis.parquet",
}

ISSUE_EXTRA_STOPWORDS = """
issue issues bug describe reproduce reproduction repro steps step expected actual
behavior behaviour version versions environment context information additional
relevant include including included tested latest tried happens happening
affiliation yes no full configuration build built example examples playground
link search terms code result results output please thanks thank wanted would
could should may might however everything nothing something anything anyone
always never sometimes still much many get got going make makes making using
used use need needs want wants like likes find found look looking seem seems
appears appeared hello hi dear thanks
""".split()
ISSUE_STOPWORDS = EN_STOPWORDS | set(ISSUE_EXTRA_STOPWORDS)

_PUNCT = "".join(c for c in "#*_~>|`[]():;,.!?\"'/-\\" if c.strip())


def clean_issue_text(text: object) -> str:
    t = str(text)
    t = re.sub(r"```.*?```", " ", t, flags=re.S)  # fenced code / logs
    t = re.sub(r"http[s]?\S+", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = t.replace("\r", " ")
    for ch in _PUNCT:
        t = t.replace(ch, " ")
    return t


def en_tokenize_issue(text: object) -> List[str]:
    out = []
    for w in _EN_TOKEN.findall(str(text).lower()):
        if len(w) > 2 and w not in ISSUE_STOPWORDS:
            out.append(w)
    return out


def _load_repo_issues(repo: str) -> pd.DataFrame:
    path = SRC_DIR / REPO_FILES[repo]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run download script first")
    df = pd.read_parquet(path, columns=["is_pull_request", "title", "body"])
    if "is_pull_request" in df.columns:
        df = df[~df["is_pull_request"]]
    out = pd.DataFrame(
        {
            "repo": repo,
            "title": df["title"].astype(str),
            "body": df["body"].fillna("").astype(str),
        }
    )
    return out


def _select_repo_records(
    repo: str,
    per_repo_train: int,
    per_repo_test: int,
    rng: random.Random,
    min_body_tokens: int = 8,
) -> Dict[str, object]:
    """Memory-lean selection: count tokens row by row, keep only chosen rows."""
    df = _load_repo_issues(repo)
    valid_flags = []
    empty_body = 0
    for title, body in zip(df["title"], df["body"]):
        body_s = str(body)
        if body_s.strip() == "":
            empty_body += 1
        n_t = len(en_tokenize_issue(title))
        n_b = len(en_tokenize_issue(clean_issue_text(body_s)))
        valid_flags.append(n_t >= 2 and n_b >= min_body_tokens)

    valid_positions = [i for i, v in enumerate(valid_flags) if v]
    n_usable = len(valid_positions)
    if n_usable < per_repo_train + per_repo_test:
        raise ValueError(f"{repo}: only {n_usable} usable records")

    chosen = rng.sample(valid_positions, per_repo_train + per_repo_test)
    sel = df.iloc[chosen]
    records = []
    for _, row in sel.iterrows():
        clean_body = clean_issue_text(row["body"])
        records.append(
            {
                "repo": repo,
                "title": str(row["title"]),
                "body": str(row["body"]),
                "clean_body": clean_body,
                "doc_text": f"{row['title']}. {clean_body}",
            }
        )
    return {
        "records": records,
        "issues": int(len(df)),
        "empty_body": int(empty_body),
        "too_short": int(len(df) - empty_body - n_usable),
        "usable": int(n_usable),
    }


def prepare_github_splits(
    per_repo_train: int = 1500,
    per_repo_test: int = 500,
    min_body_tokens: int = 8,
    seed: int = 20260905,
    force: bool = False,
) -> Dict[str, object]:
    """Balanced per-repo train/test split written to CSV for reproducibility."""
    GITHUB_DIR.mkdir(parents=True, exist_ok=True)
    out_train = GITHUB_DIR / "train.csv"
    out_test = GITHUB_DIR / "test.csv"
    if out_train.exists() and out_test.exists() and not force:
        # 2026-10-15 v3 review C4: cached splits must match the request
        n_train = len(pd.read_csv(out_train))
        n_test = len(pd.read_csv(out_test))
        want_train = int(per_repo_train) * len(REPO_FILES)
        want_test = int(per_repo_test) * len(REPO_FILES)
        if (n_train, n_test) != (want_train, want_test):
            raise ValueError(
                f"cached GitHub splits have {n_train}/{n_test} rows but "
                f"{want_train}/{want_test} rows were requested; pass "
                "force=True to rebuild them")
        return {}

    rng = random.Random(seed)
    train_rows, test_rows = [], []
    missing_stats = {}
    for repo in REPO_FILES:
        info = _select_repo_records(
            repo,
            per_repo_train,
            per_repo_test,
            rng,
            min_body_tokens=min_body_tokens,
        )
        missing_stats[repo] = {
            "issues": info["issues"],
            "empty_body": info["empty_body"],
            "too_short": info["too_short"],
            "usable": info["usable"],
        }
        for pos, rec in enumerate(info["records"]):
            rec["split"] = "train" if pos < per_repo_train else "test"
            (train_rows if rec["split"] == "train" else test_rows).append(rec)
        del info
    train_frame = pd.DataFrame(train_rows)
    test_frame = pd.DataFrame(test_rows)
    cols = ["repo", "split", "title", "body"]
    train_frame[cols].to_csv(out_train, index=False, encoding="utf-8")
    test_frame[cols].to_csv(out_test, index=False, encoding="utf-8")
    print("saved", out_train, len(train_frame), "|", out_test, len(test_frame))
    return {"missing_stats": missing_stats}


def _build_corpus(
    df: pd.DataFrame,
    train_dictionary: Dictionary,
    train_vocab: List[str],
    need_bow: bool = True,
    need_doc_bow: bool = True,
) -> Dict[str, object]:
    body_tokens = [en_tokenize_issue(clean_issue_text(b)) for b in df["body"]]
    title_tokens = [en_tokenize_issue(t) for t in df["title"]]
    vocab_set = set(train_vocab)
    word2id = train_dictionary.token2id
    body_tokens = [[w for w in ts if w in vocab_set] for ts in body_tokens]
    title_tokens = [[w for w in ts if w in vocab_set] for ts in title_tokens]
    doc_tokens = [t + b for t, b in zip(title_tokens, body_tokens)]

    reason_bow = item_bow = doc_bow = None
    if need_bow:
        reason_bow = np.zeros((len(doc_tokens), len(train_vocab)), dtype=np.float32)
        item_bow = np.zeros_like(reason_bow)
        if need_doc_bow:
            doc_bow = np.zeros_like(reason_bow)
        for d, (rt, it, dt) in enumerate(zip(body_tokens, title_tokens, doc_tokens)):
            for w in rt:
                reason_bow[d, word2id[w]] += 1.0
            for w in it:
                item_bow[d, word2id[w]] += 1.0
            if need_doc_bow:
                for w in dt:
                    doc_bow[d, word2id[w]] += 1.0

    doc_corpus = [train_dictionary.doc2bow(ts) for ts in doc_tokens]
    clean_body = df["body"].map(lambda x: clean_issue_text(x)).tolist()
    return {
        "df": df,
        "labels": df["repo"].tolist(),
        "reason_texts": clean_body,
        "item_texts": df["title"].tolist(),
        "doc_texts": (df["title"] + ". " + pd.Series(clean_body)).tolist(),
        "doc_tokens": doc_tokens,
        "reason_tokens": body_tokens,
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


def load_github(
    per_repo_train: int = 1500,
    per_repo_test: int = 500,
    no_below: int = 5,
    no_above: float = 0.7,
    keep_n: int = 8000,
    force_splits: bool = False,
    need_train_bow: bool = True,
    need_doc_bow: bool = True,
) -> Dict[str, object]:
    meta = prepare_github_splits(
        per_repo_train,
        per_repo_test,
        force=force_splits,
    )
    train_df = pd.read_csv(GITHUB_DIR / "train.csv")
    test_df = pd.read_csv(GITHUB_DIR / "test.csv")

    all_tokens = []
    for t, b in zip(train_df["title"], train_df["body"]):
        all_tokens.append(en_tokenize_issue(t) + en_tokenize_issue(clean_issue_text(b)))
    dictionary = Dictionary(all_tokens)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dictionary.compactify()
    vocab = [dictionary[i] for i in range(len(dictionary))]

    missing_rate = {}
    if meta:
        total = sum(v["issues"] for v in meta["missing_stats"].values())
        missing = sum(v["empty_body"] + v["too_short"] for v in meta["missing_stats"].values())
        missing_rate = {
            "total_issues": int(total),
            "empty_or_too_short_body": int(missing),
            "natural_missing_rate": float(missing / max(1, total)),
        }

    return {
        "name": "GitHubIssues-4repo",
        "meta": {
            "repos": list(REPO_FILES),
            "per_repo_train": int(per_repo_train),
            "per_repo_test": int(per_repo_test),
            "split_seed": 20260905,
            "no_below": no_below,
            "no_above": no_above,
            "keep_n": keep_n,
            "source": "open-index/open-github-issues (moby, TypeScript, duckdb, redis)",
            **missing_rate,
        },
        "train": _build_corpus(train_df, dictionary, vocab,
                               need_bow=need_train_bow,
                               need_doc_bow=need_doc_bow),
        "test": _build_corpus(test_df, dictionary, vocab, need_bow=False),
    }


if __name__ == "__main__":
    data = load_github(force_splits=True)
    print(data["meta"])
    for split in ("train", "test"):
        c = data[split]
        print(split, c["n_docs"], "vocab", c["n_vocab"], "labels", pd.Series(c["labels"]).value_counts().to_dict())
        print("  mean toks", round(float(np.mean([len(t) for t in c["doc_tokens"]])), 1))
