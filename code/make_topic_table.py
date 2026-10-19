# -*- coding: utf-8 -*-
"""Generate the paper's topic-sample table from a trained model, not by hand.

The reviewed manuscript listed a topic table (seed=11, K=30) containing words
that the released vocabulary cannot produce (``build`` is in the GitHub extra
stop-word list; ``columnar`` and ``vectorized`` are not in the 5175-word
evaluation vocabulary at all).  This script rebuilds that table from the
stored theta/beta dump so the words are exactly what the model outputs.

For every topic it prints
  * the cTF-IDF read-out top words (the read-out used in the paper),
  * the top words restricted to the background-field vocabulary,
  * the top words restricted to the summary-field vocabulary,
  * the corpus label with the largest theta mass (a hint for the topic name).

Usage:
  python make_topic_table.py --corpus github --details outputs/fcntm_github_final_details.json --seed 11
  python make_topic_table.py --corpus agnews --details outputs/fcntm_agnews_final_details.json --seed 11

The ``*_final_details.json`` files are the models the paper's tables are built
from; the older ``*_gate_details.json`` dumps belong to the development runs and
must not be used to regenerate a published table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config as cfg
from run_fcntm_train import class_tfidf

OUT = Path(__file__).resolve().parent / "outputs"


def load_corpus(name: str):
    if name == "agnews":
        import public_data as PD
        data = PD.load_agnews(1500, 1900, need_train_bow=False)
    else:
        import public_data_github as PGH
        data = PGH.load_github(1500, 500, need_train_bow=False)
    return data["train"], data["test"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["agnews", "github"], default="github")
    ap.add_argument("--details", default="")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--topn", type=int, default=cfg.TOPN)
    ap.add_argument("--out", default="")
    ap.add_argument("--select", default="",
                    help="1-based topic ids to emit as LaTeX rows, e.g. 1,2,3")
    ap.add_argument("--names", default="",
                    help="comma separated human names for the selected topics")
    ap.add_argument("--rows-out", default="",
                    help="write the selected rows to this .tex file")
    args = ap.parse_args()

    # default to the *final* model the paper reports; the dev-run gate dumps
    # would silently regenerate a table from the wrong model
    path = Path(args.details) if args.details else (
        OUT / f"fcntm_{args.corpus}_final_details.json")
    if not path.exists():
        raise SystemExit(
            f"{path} not found: pass --details with the model the table should "
            f"come from (the paper uses fcntm_<corpus>_final_details.json)")
    records = json.loads(path.read_text(encoding="utf-8"))
    rec = next(r for r in records if int(r["seed"]) == args.seed)
    beta = np.asarray(rec["beta"], dtype=float)
    theta = np.asarray(rec["theta"], dtype=float)

    train, _ = load_corpus(args.corpus)
    vocab = train["vocab"]
    labels = np.asarray(train["labels"])
    if theta.shape[0] != len(train["doc_tokens"]):
        raise SystemExit(f"theta has {theta.shape[0]} rows but the corpus has "
                         f"{len(train['doc_tokens'])} documents")

    # S_k = sum_d theta_dk * x_d, accumulated token by token so that the
    # (6000 x V) count matrix never has to be materialised
    word2id = train["dictionary"].token2id
    soft_counts = np.zeros((theta.shape[1], len(vocab)), dtype=np.float64)
    for d, tokens in enumerate(train["doc_tokens"]):
        row = theta[d]
        for w in tokens:
            j = word2id.get(w)
            if j is not None:
                soft_counts[:, j] += row
    scores = class_tfidf(soft_counts)
    # hoist the field vocabularies out of the loop: building them inside the
    # comprehension re-scans the whole corpus once per vocabulary entry
    bg_vocab = set(w for ts in train["reason_tokens"] for w in ts)
    sm_vocab = set(w for ts in train["item_tokens"] for w in ts)
    bg_ids = [i for i, w in enumerate(vocab) if w in bg_vocab]
    sm_ids = [i for i, w in enumerate(vocab) if w in sm_vocab]

    def top(mat, ids, n):
        sub = mat[:, ids]
        idx = np.argsort(-sub, axis=1)[:, :n]
        return [[vocab[ids[j]] for j in row] for row in idx]

    ctfidf_words = top(scores, list(range(len(vocab))), args.topn)
    bg_words = top(scores, bg_ids, args.topn)
    sm_words = top(scores, sm_ids, args.topn)
    mass = theta.sum(axis=0)
    dominant = {}
    for k in range(theta.shape[1]):
        if labels.size == theta.shape[0]:
            w = theta[:, k]
            order = np.argsort(-w)[: max(1, int(0.05 * len(w)))]
            vals, cnts = np.unique(labels[order], return_counts=True)
            dominant[k] = str(vals[int(np.argmax(cnts))])
        else:
            dominant[k] = ""

    lines = [f"% {args.corpus} seed={args.seed} K={beta.shape[0]} "
             f"best_epoch={rec.get('best_epoch')}",
             "% topic & summary-side core words & background-side core words"]
    rows = []
    for k in range(len(ctfidf_words)):
        lines.append(
            f"T{k + 1:02d} [{dominant[k]}, mass={mass[k] / mass.sum():.3f}] "
            f"& {' '.join(sm_words[k])} & {' '.join(bg_words[k])} \\\\")
        lines.append(f"%   full cTF-IDF read-out: {' '.join(ctfidf_words[k])}")
        # directly \input-able row: only the topic name is left for the author
        rows.append(
            f"T{k + 1:02d}（待命名） & {' '.join(sm_words[k])} & "
            f"{' '.join(bg_words[k])} \\\\  % label={dominant[k]} "
            f"mass={mass[k] / mass.sum():.3f}")
    text = "\n".join(lines)
    print(text)
    selected = ([int(x) for x in args.select.split(",") if x.strip()]
                if args.select else list(range(1, len(rows) + 1)))
    names = [x.strip() for x in args.names.split(",")] if args.names else []
    picked = []
    for pos, tid in enumerate(selected):
        row = rows[tid - 1]
        if pos < len(names) and names[pos]:
            row = row.replace(f"T{tid:02d}（待命名）", names[pos], 1)
        picked.append(f"% T{tid:02d} ({args.corpus}, seed {args.seed})\n{row}")
    rows_path = Path(args.rows_out) if args.rows_out else (
        OUT / f"topic_table_{args.corpus}_seed{args.seed}_rows.tex")
    if not rows_path.is_absolute():
        rows_path = cfg.ROOT / rows_path
    def drop_last_terminator(rows_list):
        """Leave the final row unterminated on purpose.

        A row terminator as the last token of an \\input-ed file makes the
        caller's \\bottomrule fail with "Misplaced \\noalign"; main.tex
        therefore supplies the terminator for the last row.
        """
        if not rows_list:
            return rows_list
        out = list(rows_list)
        out[-1] = out[-1].replace("\\\\", "", 1)
        return out

    rows_path.write_text("\n".join(drop_last_terminator(rows)) + "\n",
                         encoding="utf-8")
    if args.select:
        rows_path.write_text("\n".join(drop_last_terminator(picked)) + "\n",
                             encoding="utf-8")
    print("\nsaved", rows_path, "(\\input-able rows)")
    if args.out:
        p = Path(args.out)
        if not p.is_absolute():
            p = cfg.ROOT / p
        p.write_text(text + "\n", encoding="utf-8")
        print("\nsaved", p)
    else:
        default = OUT / f"topic_table_{args.corpus}_seed{args.seed}.tex"
        default.write_text(text + "\n", encoding="utf-8")
        print("\nsaved", default)


if __name__ == "__main__":
    main()
