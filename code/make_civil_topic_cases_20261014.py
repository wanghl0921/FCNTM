# -*- coding: utf-8 -*-
"""Civil topic case studies for the English supplementary (2026-10-14 review).

Uses only archived artifacts (seed-11 model state, details dump, the frozen
OpenAlex topic cache) to build, for four readable civil topics and one
confusing topic:

  * the topic words of the beta read-out,
  * the three highest-mass representative records (titles),
  * the OpenAlex (external automatic) label of each record,
  * the title-only routing decision and the full-field decision under the
    same majority-vote mapping protocol as civil_title_archival.py.

Writes outputs/rerun/civil_topic_cases_20261014.md and the LaTeX rows
topic_table_civil_seed11_paper_en.tex for the supplementary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import civil_title_archival as CTA
import occlusion_robustness as O
import public_data_civil as PC

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PAPER = (ROOT.parent / "paper") if (ROOT.parent / "paper").exists() \
    else (ROOT.parent / "FCNTM完整论文")
SEED = 11
TOPICS = {
    0: "Asphalt mixture design and performance",
    3: "Seismic response of tunnels and piles",
    8: "BIM and construction management",
    9: "Structural mechanics: shear, buckling, composite beams",
    6: "Mixed laboratory-materials topic (failure case)",
}


def predict(theta_row, mapping) -> tuple:
    labels, table, prior = mapping
    row_sum = table.sum(axis=1)
    prob = table / np.maximum(row_sum, 1e-12)[:, None]
    prob[row_sum == 0] = prior
    z = int(np.argmax(theta_row))
    return labels[int(np.argmax(prob[z]))], z


def clean(text: str) -> str:
    out = "".join(ch for ch in str(text) if 32 <= ord(ch) < 127)
    return " ".join(out.split())


def main() -> None:
    data = PC.load_civil(6000, 2000)
    train, test = data["train"], data["test"]
    vocab_size = test["n_vocab"]

    topics_cache = pd.read_csv(OUT / "civil_openalex_topics.csv")
    topics_cache = topics_cache.set_index("openalex_id")
    tr = train["df"].copy()
    tr["label"] = tr["openalex_id"].map(topics_cache["topic"]).fillna("")
    counts = tr["label"].value_counts()
    counts = counts[counts.index != ""]
    keep = [lab for lab in counts.index[:15] if lab]
    tr_lab = tr["label"].to_numpy()
    tr_pos = np.nonzero(np.isin(tr_lab, keep))[0]
    tr_labels = tr_lab[tr_pos].tolist()

    details = json.load(open(OUT / "fcntm_civil_final_details.json",
                             encoding="utf-8"))
    rec = next(r for r in details if int(r["seed"]) == SEED
               and r["model"] == "FC-NTM")
    full_theta = np.asarray(rec["theta"], dtype=float)
    final = pd.read_csv(OUT / "fcntm_civil_final.csv")
    frow = final[(final["model"] == "FC-NTM")
                 & (final["seed"] == SEED)].iloc[0]
    topic_words = json.loads(frow["topics"])

    adapter = O.FCNTMAdapter(OUT / "civil_missing_state.pt", vocab_size, 30,
                             seed=SEED)
    adapter.set_word2id(test["dictionary"].token2id)
    titles = train["item_tokens"]
    title_theta = adapter.theta([[] for _ in range(len(titles))], titles)

    map_t = CTA.topic_label_map(title_theta[tr_pos], tr_labels, 30)
    map_f = CTA.topic_label_map(full_theta[tr_pos], tr_labels, 30)

    te = test["df"].copy()
    te["label"] = te["openalex_id"].map(topics_cache["topic"]).fillna("")
    te_lab = te["label"].to_numpy()
    te_pos = np.nonzero(np.isin(te_lab, keep))[0]
    te_labels = te_lab[te_pos].tolist()
    test_titles = test["item_tokens"]
    test_title_theta = adapter.theta(
        [[] for _ in range(len(test_titles))], test_titles)
    test_full_theta = adapter.theta(list(test["reason_tokens"]), test_titles)
    t_pred_all = [predict(test_title_theta[i], map_t)[0] for i in te_pos]
    f_pred_all = [predict(test_full_theta[i], map_f)[0] for i in te_pos]
    acc_t = float(np.mean([p == l for p, l in zip(t_pred_all, te_labels)]))
    acc_f = float(np.mean([p == l for p, l in zip(f_pred_all, te_labels)]))

    lines = []
    tex_rows = []
    for k, name in TOPICS.items():
        mass = full_theta[:, k].copy()
        order = tr_pos[np.argsort(-mass[tr_pos])]
        picks = order[:3]
        lines.append(f"## Topic {k}: {name}")
        lines.append("top words: " + ", ".join(topic_words[k]))
        for i in picks:
            title = clean(train["df"]["title"].iloc[i])
            label = tr_lab[i]
            t_pred, t_topic = predict(title_theta[i], map_t)
            f_pred, f_topic = predict(full_theta[i], map_f)
            lines.append(
                f"- [{label}] {title!r} | title-only T{t_topic} -> {t_pred}"
                f" ({'ok' if t_pred == label else 'miss'}) | full-field"
                f" T{f_topic} -> {f_pred}"
                f" ({'ok' if f_pred == label else 'miss'})")
        i0 = picks[0]
        title0 = clean(train["df"]["title"].iloc[i0])
        label0 = tr_lab[i0]
        t_pred0, _ = predict(title_theta[i0], map_t)
        f_pred0, _ = predict(full_theta[i0], map_f)
        oid0 = str(train["df"]["openalex_id"].iloc[i0])
        # 2026-10-15 v3 review: full titles instead of a silent truncation,
        # with the OpenAlex ID so every case is traceable
        tex_rows.append((name, ", ".join(topic_words[k][:10]),
                         f"{title0} ({oid0})",
                         str(label0), str(t_pred0), str(f_pred0)))
        lines.append("")

    lines.append(f"test protocol check on {len(te_pos)} labelled records: "
                 f"title-only acc {acc_t:.3f}, full-field acc {acc_f:.3f}")
    fails = [j for j, (p, l) in enumerate(zip(t_pred_all, te_labels))
             if p != l]
    fixed = [j for j in fails if f_pred_all[j] == te_labels[j]]
    hard = [j for j in fails if f_pred_all[j] != te_labels[j]]
    picked = []
    for j in (fixed[:1] + hard[:1] + fails):
        if j not in picked:
            picked.append(j)
        if len(picked) >= 2:
            break
    for j in picked:
        i = te_pos[j]
        title = clean(test["df"]["title"].iloc[i])
        z_t = int(np.argmax(test_title_theta[i]))
        lines.append(
            f"## Failure case: {title!r} | true [{te_labels[j]}] | "
            f"title-only -> {t_pred_all[j]} (topic {z_t}) | full-field -> "
            f"{f_pred_all[j]}")
        oid = str(test["df"]["openalex_id"].iloc[i])
        tex_rows.append((
            "Failure case: title-only misrouted",
            ", ".join(topic_words[z_t][:10]),
            f"{title} ({oid})",
            str(te_labels[j]), str(t_pred_all[j]), str(f_pred_all[j])))
    lines.append("")
    md = "\n".join(lines) + "\n"
    (OUT / "rerun" / "civil_topic_cases_20261014.md").write_text(
        md, encoding="utf-8")
    print(md)

    tex = []
    for name, words, title, label, t_pred, f_pred in tex_rows:
        tex.append(
            f"{name} & {words} & {title} & {label} & "
            f"{'correct' if t_pred == label else 'miss'} & "
            f"{'correct' if f_pred == label else 'miss'} \\\\")
    (PAPER / "topic_table_civil_seed11_paper_en.tex").write_text(
        "\n".join(tex) + "\n", encoding="utf-8")
    print("case table written to",
          PAPER / "topic_table_civil_seed11_paper_en.tex")


if __name__ == "__main__":
    main()
