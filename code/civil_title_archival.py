# -*- coding: utf-8 -*-
"""Title-only archival task on the civil-engineering corpus.

Realistic engineering setting: some literature records have only a title
(the abstract is missing).  Given the title alone, can the topics of a
record be routed to a civil subfield?  The ground truth is OpenAlex's own
primary-topic assignment (an independent source, fetched by
``fetch_openalex_topics.py``), not any label produced by the evaluated
models.

Protocol
    * labels: the N most frequent OpenAlex primary topics on the training
      split; test records whose label is outside the set are reported as
      coverage but not scored;
    * each model assigns every title a topic via its own inference path
      (title-only input); the topic->label mapping is estimated on the
      training split from the same title-only inference;
    * a document is correct when the mapped label equals the OpenAlex label;
      accuracy and macro-F1 are reported per model and seed;
    * a supervised TF-IDF logistic-regression classifier on the same titles
      is included as a reference, and FC-NTM with the full fields (abstract
      present) is reported as an upper reference for the same pipeline.

Usage:
    python civil_title_archival.py --seeds 11,23,37 --top-labels 15
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import baselines as BL
import config
import models as NN
import occlusion_robustness as O
import seed_identity as SI

OUT = config.OUT_DIR
TOPIC_CACHE = OUT / "civil_openalex_topics.csv"


class ConcatAdapter(O.Adapter):
    """Title-only (and full-field) inference for the concatenated control."""

    name = "FC-NTM-concat"

    def __init__(self, state_path: Path, vocab_size: int, K: int, seed: int,
                 train: dict):
        ck = torch.load(state_path, map_location="cpu", weights_only=False)
        key = f"seed:{int(seed)}"
        if key not in ck:
            raise KeyError(f"{state_path} has no state for {key}")
        rec = ck[key]
        # 2026-10-14 review C5: a same-width vocabulary is not the same
        # corpus, so the state must match this fit before reuse
        ident = rec.get("identity") or {}
        if not ident:
            raise SystemExit(
                f"{state_path.name} seed {seed} carries no identity; "
                "retrain the control with the current run_concat_control.py")
        want = {"vocab_digest": SI.vocab_digest(train),
                "n_vocab": int(train["n_vocab"]), "k": int(K)}
        bad = [key2 for key2, value in want.items()
               if ident.get(key2) != value]
        if bad:
            raise SystemExit(
                f"{state_path.name} seed {seed}: identity differs in {bad}")
        self.model = NN.FCNTMConcat(vocab_size, K, **rec["kwargs"])
        self.model.load_state_dict(rec["model_state"])
        self.model.eval()

    def theta(self, visible_A, field_B):
        A = O._bow(visible_A, self.model.vocab_size, self.word2id)
        B = O._bow(field_B, self.model.vocab_size, self.word2id)
        has_cause = torch.tensor([len(t) > 0 for t in visible_A],
                                 dtype=torch.bool)
        with torch.no_grad():
            theta, _ = self.model.encode_doc_theta(
                NN._normalize_rows(torch.from_numpy(A)),
                NN._normalize_rows(torch.from_numpy(B)),
                has_cause)
        return theta.numpy()

    def set_word2id(self, word2id):
        self.word2id = word2id


def title_only_theta(adapter, n: int, titles):
    return adapter.theta([[] for _ in range(n)], titles)


def full_field_theta(adapter, abstracts, titles):
    return adapter.theta(list(abstracts), titles)


def topic_label_map(theta_train, labels_train, K: int) -> np.ndarray:
    """P(label | topic) estimated on the training split (argmax topics).

    2026-10-14 review C4: topics that take no training record have an empty
    row; instead of an arbitrary alphabetical first label they now fall back
    to the global training-label prior (returned alongside the table).
    """
    z = theta_train.argmax(axis=1)
    votes = defaultdict(Counter)
    for zi, lab in zip(z, labels_train):
        votes[int(zi)][lab] += 1
    labels = sorted({l for c in votes.values() for l in c})
    table = np.zeros((K, len(labels)))
    for k, counter in votes.items():
        for lab, cnt in counter.items():
            table[k, labels.index(lab)] = cnt
    counts = Counter(labels_train)
    prior = np.array([counts[l] for l in labels], dtype=float)
    prior = prior / prior.sum()
    return labels, table, prior


def score(theta_test, labels_test, mapping) -> tuple:
    labels, table, prior = mapping
    row_sums = table.sum(axis=1)
    prob = table / np.maximum(row_sums, 1e-12)[:, None]
    empty = row_sums == 0
    prob[empty] = prior
    z = theta_test.argmax(axis=1)
    fallback = int(empty[z].sum())
    pred_idx = prob[z].argmax(axis=1)
    pred = [labels[i] for i in pred_idx]
    correct = np.array([p == t for p, t in zip(pred, labels_test)])
    acc = float(correct.mean()) if len(correct) else float("nan")
    # macro F1 over the label set
    f1s = []
    y_true = np.array(labels_test)
    y_pred = np.array(pred)
    for lab in labels:
        tp = int(((y_true == lab) & (y_pred == lab)).sum())
        fp = int(((y_true != lab) & (y_pred == lab)).sum())
        fn = int(((y_true == lab) & (y_pred != lab)).sum())
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")
    return acc, macro_f1, pred, fallback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,23,37")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--top-labels", type=int, default=15)
    ap.add_argument("--fcntm-state",
                    default=str(OUT / "civil_missing_state.pt"))
    ap.add_argument("--concat-state",
                    default=str(OUT / "rerun" / "concat_state_civil.pt"))
    ap.add_argument("--models", default="FC-NTM,Concat,BTM,LDA,TFIDF")
    ap.add_argument("--out", default="rerun/civil_title_archival.csv")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    models = [m for m in args.models.split(",") if m]

    import public_data_civil as PC
    data = PC.load_civil(6000, 2000)
    train, test = data["train"], data["test"]
    vocab = test["vocab"]
    word2id = test["dictionary"].token2id

    topics = pd.read_csv(TOPIC_CACHE).set_index("openalex_id")
    tr = train["df"].copy()
    te = test["df"].copy()
    tr["label"] = tr["openalex_id"].map(topics["topic"]).fillna("")
    te["label"] = te["openalex_id"].map(topics["topic"]).fillna("")
    counts = tr["label"].value_counts()
    counts = counts[counts.index != ""]
    keep = [lab for lab in counts.index[:args.top_labels] if lab]
    tr = tr[tr["label"].isin(keep)].reset_index(drop=True)
    coverage = float(te["label"].isin(keep).mean())
    te = te[te["label"].isin(keep)].reset_index(drop=True)
    print(f"label set: {len(keep)} topics; test coverage {coverage:.3f} "
          f"({len(te)} / {test['n_docs']} records)")
    print("label counts (train):")
    print(tr["label"].value_counts().to_string())

    # positions (into the corpus arrays) of the records that survive the filter
    tr_lab = train["df"]["openalex_id"].map(topics["topic"]).fillna("")
    te_lab = test["df"]["openalex_id"].map(topics["topic"]).fillna("")
    tr_pos = np.nonzero(tr_lab.isin(keep).to_numpy())[0]
    te_pos = np.nonzero(te_lab.isin(keep).to_numpy())[0]
    tr_labels = tr_lab.iloc[tr_pos].tolist()
    te_labels = te_lab.iloc[te_pos].tolist()

    rows = []
    for seed in seeds:
        adapters = {}
        if "FC-NTM" in models:
            ad = O.FCNTMAdapter(Path(args.fcntm_state), test["n_vocab"],
                                args.k, seed=seed)
            ad.set_word2id(word2id)
            adapters["FC-NTM"] = ad
        if "Concat" in models:
            state = Path(args.concat_state)
            if not state.exists():
                raise SystemExit(
                    f"Concat requested but {state} is missing; refusing to "
                    "silently drop the model from the comparison")
            ad = ConcatAdapter(state, test["n_vocab"], args.k, seed, train)
            ad.set_word2id(word2id)
            adapters["Concat"] = ad
        if "LDA" in models:
            ad = O.LDAAdapter(train, args.k, seed)
            ad.dictionary = train["dictionary"]
            adapters["LDA"] = ad
        if "BTM" in models:
            fit_path = OUT / "rerun" / f"btm_fit_civil_seed{seed}.npz"
            rec = None
            if fit_path.exists():
                z = np.load(fit_path, allow_pickle=True)
                stored = (str(z["identity_id"]) if "identity_id" in z.files
                          else None)
                want_id = SI.fit_identity("extend_seeds_btm.fit", "civil",
                                          train, args.k, seed)["id"]
                if stored != want_id:
                    raise SystemExit(
                        f"{fit_path.name} is not pinned to the current civil "
                        f"fit (stored {stored!r}); refit before scoring")
                rec = {k: (z[k].item() if z[k].ndim == 0 else z[k])
                       for k in z.files
                       if k not in ("identity_id", "identity_json")}
            ad = O.BTMAdapter(train, args.k, seed, rec=rec)
            ad.dictionary = train["dictionary"]
            adapters["BTM"] = ad

        titles_tr = [train["item_tokens"][i] for i in tr_pos]
        titles_te = [test["item_tokens"][i] for i in te_pos]
        for name, ad in adapters.items():
            theta_tr = title_only_theta(ad, len(titles_tr), titles_tr)
            theta_te = title_only_theta(ad, len(titles_te), titles_te)
            mapping = topic_label_map(theta_tr, tr_labels, args.k)
            acc, f1, _, fallback = score(theta_te, te_labels, mapping)
            rows.append({"model": name, "seed": seed, "setting": "title_only",
                         "accuracy": acc, "macro_f1": f1,
                         "coverage": coverage, "n_eval": len(te_labels),
                         "mapping_fallback": fallback})
            print(f"  {name} seed {seed} title-only: acc {acc:.4f} "
                  f"macroF1 {f1:.4f} fallback {fallback}", flush=True)
            if name == "FC-NTM":
                ab_tr = [train["reason_tokens"][i] for i in tr_pos]
                ab_te = [test["reason_tokens"][i] for i in te_pos]
                th_tr = full_field_theta(ad, ab_tr, titles_tr)
                th_te = full_field_theta(ad, ab_te, titles_te)
                mapping_f = topic_label_map(th_tr, tr_labels, args.k)
                acc_f, f1_f, _, fallback_f = score(th_te, te_labels, mapping_f)
                rows.append({"model": "FC-NTM", "seed": seed,
                             "setting": "full_field", "accuracy": acc_f,
                             "macro_f1": f1_f, "coverage": coverage,
                             "n_eval": len(te_labels),
                             "mapping_fallback": fallback_f})
                print(f"  FC-NTM seed {seed} full-field: acc {acc_f:.4f} "
                      f"macroF1 {f1_f:.4f}", flush=True)

        if "TFIDF" in models:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                  min_df=2)
            X_tr = vec.fit_transform(
                [train["df"]["title"].iloc[i] for i in tr_pos])
            X_te = vec.transform(
                [test["df"]["title"].iloc[i] for i in te_pos])
            clf = LogisticRegression(max_iter=2000, C=4.0,
                                     random_state=seed)
            clf.fit(X_tr, tr_labels)
            pred = clf.predict(X_te)
            acc = float(np.mean(pred == np.array(te_labels)))
            y_true = np.array(te_labels)
            f1s = []
            for lab in sorted(set(tr_labels) | set(te_labels)):
                tp = int(((y_true == lab) & (pred == lab)).sum())
                fp = int(((y_true != lab) & (pred == lab)).sum())
                fn = int(((y_true == lab) & (pred != lab)).sum())
                if tp + fp + fn == 0:
                    continue
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                f1s.append(0.0 if prec + rec == 0
                           else 2 * prec * rec / (prec + rec))
            rows.append({"model": "TFIDF-LR", "seed": seed,
                         "setting": "title_only", "accuracy": acc,
                         "macro_f1": float(np.mean(f1s)),
                         "coverage": coverage, "n_eval": len(te_labels)})
            print(f"  TFIDF-LR seed {seed}: acc {acc:.4f}", flush=True)

    frame = pd.DataFrame(rows)
    out_path = OUT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    frame.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(frame.round(4).to_string(index=False))
    print("saved", out_path)


if __name__ == "__main__":
    main()
