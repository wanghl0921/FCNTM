# -*- coding: utf-8 -*-
"""How complementary are a corpus's two fields?

FC-NTM only has something to gain when the two fields carry different
information.  If the summary field already contains the background field's
content, cross-field reconstruction has nothing to transfer.  This script
measures that, so the comparison between corpora rests on numbers instead of on
an assertion about the datasets.

Statistics (all model-free, computed on the cached splits):

  jaccard          mean per-document |B n S| / |B u S|
  bg_novelty       mean fraction of background tokens that do not occur in the
                   summary field of the same document  -> high = complementary
  sm_novelty       the same in the other direction
  js_divergence    Jensen-Shannon divergence between the two fields' unigram
                   distributions over the shared vocabulary -> high = the fields
                   talk about different things
  vocab_cover_r/s  share of a field's vocabulary also used by the other field

Usage: python field_complementarity.py [--corpus github,agnews,civil] [--split train]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import config
from public_small import load_small

OUT = Path(__file__).resolve().parent / "outputs"


def load_full(name: str):
    """The corpus as it is used by the paper (6000 train documents)."""
    if name == "agnews":
        import public_data as PD
        return PD.load_agnews(1500, 1900, need_train_bow=False)
    if name == "civil":
        import public_data_civil as PC
        return PC.load_civil(6000, 2000, need_train_bow=False,
                             need_doc_bow=False)
    import public_data_github as PGH
    return PGH.load_github(1500, 500, need_train_bow=False)


def js_divergence(p: Counter, q: Counter) -> float:
    """JS divergence (base 2) between two count vectors over their union."""
    vocab = set(p) | set(q)
    pv = np.array([p.get(w, 0) for w in vocab], dtype=float)
    qv = np.array([q.get(w, 0) for w in vocab], dtype=float)
    pv /= pv.sum() or 1.0
    qv /= qv.sum() or 1.0
    m = 0.5 * (pv + qv)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))

    return 0.5 * kl(pv, m) + 0.5 * kl(qv, m)


def per_doc_novelty(bg_tokens, sm_tokens):
    """Type- and token-weighted novelty of one field against the other.

    ``bg_n`` / ``sm_n`` are the share of *distinct word types* that the other
    field does not contain (the paper names this the per-document de-duplicated
    type novelty).  ``bg_n_tok`` / ``sm_n_tok`` count every occurrence, so a
    background token whose type appears in the summary is not counted as novel
    (2026-10-07 review, item 8).
    """
    bg_n, sm_n, bg_n_tok, sm_n_tok, jac = [], [], [], [], []
    for b, s in zip(bg_tokens, sm_tokens):
        B, S = set(b), set(s)
        if not B:
            continue
        bg_n.append(len(B - S) / len(B))
        sm_n.append(len(S - B) / len(S) if S else 0.0)
        bc, sc = Counter(b), Counter(s)
        # token-weighted: the share of *occurrences* whose word type does not
        # appear in the other field at all; matching types are not "novel",
        # however often they occur on this side
        bg_missing = sum(cnt for w, cnt in bc.items() if w not in S)
        sm_missing = sum(cnt for w, cnt in sc.items() if w not in B)
        bg_n_tok.append(bg_missing / max(1, sum(bc.values())))
        sm_n_tok.append(sm_missing / max(1, sum(sc.values())))
        jac.append(len(B & S) / len(B | S) if (B | S) else 0.0)
    return (float(np.mean(bg_n)), float(np.mean(sm_n)),
            float(np.mean(bg_n_tok)), float(np.mean(sm_n_tok)),
            float(np.mean(jac)))


def stats_for(data, split):
    corpus = data[split]
    bg = corpus["reason_tokens"]
    sm = corpus["item_tokens"]
    bg_n, sm_n, bg_n_tok, sm_n_tok, jac = per_doc_novelty(bg, sm)
    bg_counts = Counter(w for ts in bg for w in ts)
    sm_counts = Counter(w for ts in sm for w in ts)
    bg_vocab, sm_vocab = set(bg_counts), set(sm_counts)
    rec = {
        "split": split,
        "n_docs": len(bg),
        "mean_bg_len": float(np.mean([len(t) for t in bg])),
        "mean_sm_len": float(np.mean([len(t) for t in sm])),
        "jaccard": jac,
        "bg_novelty": bg_n,
        "sm_novelty": sm_n,
        "bg_novelty_token": bg_n_tok,
        "sm_novelty_token": sm_n_tok,
        "js_divergence": js_divergence(bg_counts, sm_counts),
        "bg_vocab_also_in_sm": len(bg_vocab & sm_vocab) / max(1, len(bg_vocab)),
        "sm_vocab_also_in_bg": len(sm_vocab & bg_vocab) / max(1, len(sm_vocab)),
    }
    # per-document novelty is needed again by the stratified experiment
    corpus["bg_novelty_per_doc"] = [
        (len(set(b) - set(s)) / len(set(b))) if b else float("nan")
        for b, s in zip(bg, sm)
    ]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="github,agnews")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--source", choices=["small", "full"], default="full",
                    help="small: the 1000/400 development sample; full: the "
                         "6000/2000(7600) splits actually used in the paper")
    args = ap.parse_args()

    table = {}
    for name in args.corpus.split(","):
        data = (load_small(name) if args.source == "small"
                else load_full(name))
        table[name] = {}
        for split in args.splits.split(","):
            table[name][split] = stats_for(data, split)

    rows = []
    for name, splits in table.items():
        for split, rec in splits.items():
            rows.append({"corpus": name, **rec})
    df = pd.DataFrame(rows)
    cols = ["corpus", "split", "n_docs", "mean_bg_len", "mean_sm_len", "jaccard",
            "bg_novelty", "sm_novelty", "bg_novelty_token", "sm_novelty_token",
            "js_divergence", "bg_vocab_also_in_sm", "sm_vocab_also_in_bg"]
    print(df[cols].round(4).to_string(index=False))
    (OUT / "field_complementarity.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    df[cols].to_csv(OUT / "field_complementarity.csv", index=False,
                    encoding="utf-8-sig")
    print("\nsaved outputs/field_complementarity.{json,csv}")


if __name__ == "__main__":
    main()
