# -*- coding: utf-8 -*-
"""Verify every number printed in the paper against the stored artifacts.

The 2026-09-17 review showed why a checker that keeps its *own* copy of the
expected numbers cannot protect the paper: editing a value in the paper's table
left the check green.  Two rules follow, and this file implements both.

1. Every table value is **parsed out of main.tex** (every numbered table,
   the clean-test table and the field/redundancy tables).  ``--self-test`` copies
   main.tex, tampers with one printed number and requires the checker to fail.
2. Numbers are compared **as printed**: the artifact value is formatted with the
   same number of decimals as the LaTeX cell, so printing 0.674 where the value
   rounds to 0.675 is an error rather than "within tolerance".

Only prose percentages in the running text are still constants in this file;
they are labelled ``text:`` when they fail.

    python check_paper_numbers.py
    python check_paper_numbers.py --self-test
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from make_result_tables import table_rows

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PAPER = (ROOT.parent / "paper") if (ROOT.parent / "paper").exists() \
    else (ROOT.parent / "FCNTM完整论文")
PAPER_TEX = PAPER / "main.tex"          # overridden by --self-test

FAILURES: list[str] = []
CHECKS = 0
NUM_RE = re.compile(r"-?\d+\.\d+")


# ---------------------------------------------------------------------------
# LaTeX parsing
# ---------------------------------------------------------------------------

SUPP_TEX = PAPER / "supplementary.tex"


def paper_lines(path: Path | None = None) -> list[str]:
    lines = (path or PAPER_TEX).read_text(encoding="utf-8").splitlines()
    if path is None and SUPP_TEX.exists():
        lines += SUPP_TEX.read_text(encoding="utf-8").splitlines()
    return lines


def strip_comments(line: str) -> str:
    """Drop a LaTeX comment, keeping escaped ``\\%``.

    The review's counter-example appended ``% audit`` to a tampered row; the
    row then no longer ended with ``\\\\`` and was skipped entirely.  Comments
    are legal LaTeX and must not hide a number from the check.
    """
    out = []
    i = 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if line[i] == "%":
            break
        out.append(line[i])
        i += 1
    return "".join(out)


def table_lines(label: str, path: Path | None = None) -> list[str]:
    """Raw lines from the table's ``\\label{...}`` to its ``\\end{tabular}``."""
    lines = [strip_comments(l) for l in paper_lines(path)]
    start = next((i for i, l in enumerate(lines) if label in l), None)
    if start is None:
        return []
    out = []
    for line in lines[start:]:
        if line.strip().startswith(r"\end{tabular}"):
            break
        out.append(line)
    return out


def data_rows(label: str, path: Path | None = None) -> list[str]:
    """Logical rows of a table: physical lines joined until ``\\\\``.

    A row may be split over several lines (the paper's multirow tables do
    this), so the accumulator joins them; a row that never terminates is still
    returned, so a truncated table cannot silently lose its numbers.
    """
    rows, buffer = [], ""
    for raw in table_lines(label, path):
        if "&" not in raw and not buffer:
            continue                      # rules, captions, multirow labels
        buffer = f"{buffer} {raw.strip()}".strip() if buffer else raw.strip()
        if buffer.endswith("\\\\"):
            rows.append(buffer)
            buffer = ""
    if buffer:
        rows.append(buffer)
    return [r for r in rows if NUM_RE.search(r)]


def require_rows(what: str, parsed, expected) -> None:
    """Fail when a table lost rows or models (the review deleted a table)."""
    global CHECKS
    CHECKS += 1
    parsed = list(parsed)
    missing = sorted(set(expected) - set(parsed))
    extra = sorted(set(parsed) - set(expected))
    if missing or extra:
        FAILURES.append(f"{what}: rows missing {missing} / unexpected {extra}")
        print(f"  MISSING  {what}: missing rows {missing}, unexpected {extra}")


def model_name(cell: str) -> str:
    """Column-1 label -> the name used by the result files."""
    c = cell.strip()
    if c.startswith(r"\fcntm{}-cTFIDF"):
        return "FC-NTM-cTFIDF"
    if c.startswith(r"\fcntm{}"):
        return "FC-NTM"
    c = re.sub(r"（.*?）", "", c).strip()
    return {"CTM-lite": "CTMLite", "CTM-lite ": "CTMLite"}.get(c, c)


def _strip_markup(cell: str) -> str:
    c = cell.replace(r"\textbf{", "").replace(r"\emph{", "")
    c = c.replace(r"\shortstack[l]", "").replace("$-$", "-")
    c = c.replace("$+$", "+").replace("\\", "")
    return c.replace("{", "").replace("}", "").strip()


def printed(cell: str) -> str | None:
    """The number exactly as printed, e.g. ``'0.675'`` or ``'-4.19'``."""
    m = NUM_RE.search(_strip_markup(cell))
    return m.group(0) if m else None


def printed_pm(cell: str) -> tuple[str | None, str | None] | None:
    """``'0.600$\\pm$0.011'`` -> ``('0.600', '0.011')``."""
    if r"$\pm$" not in cell:
        return None
    head, tail = cell.split(r"$\pm$")[:2]
    return printed(head), printed(tail)


def printed_slash(cell: str) -> tuple[str | None, str | None] | None:
    """``'\\textbf{1.032} / 0.358'`` -> ``('1.032', '0.358')``."""
    if "/" not in cell:
        return None
    head, tail = cell.split("/")[:2]
    return printed(head), printed(tail)


def fmt(value: float, like: str) -> str:
    """Format ``value`` with the same number of decimals as ``like``."""
    decimals = len(like.split(".")[1]) if "." in like else 0
    text = f"{float(value):.{decimals}f}"
    if text.startswith("-") and float(text) == 0:
        text = text[1:]                     # never print "-0.000"
    return text


def cmp(what: str, value, shown: str | None) -> None:
    """Compare one printed number with the artifact value."""
    global CHECKS
    CHECKS += 1
    if shown is None or value is None:
        FAILURES.append(f"{what}: nothing to compare (artifact {value!r})")
        print(f"  MISSING  {what}: artifact {value!r}")
        return
    want = fmt(value, shown)
    if shown != want:
        FAILURES.append(f"{what}: artifact {float(value):.6f} prints as "
                        f"{want}, paper has {shown}")
        print(f"  MISMATCH {what}: artifact {float(value):.6f} prints as "
              f"{want}, paper has {shown}")


def cmp_pm(what: str, mean, sd, cell: str) -> None:
    parts = printed_pm(cell)
    if parts is None:
        FAILURES.append(f"{what}: no '\\pm' pair found in {cell!r}")
        print(f"  MISSING  {what}: {cell!r}")
        return
    cmp(f"{what} mean", mean, parts[0])
    cmp(f"{what} sd", sd, parts[1])


# ---------------------------------------------------------------------------
# artifact side
# ---------------------------------------------------------------------------

def main_table(tag: str) -> pd.DataFrame:
    """Per-model means/stds **plus** the mean number of topics actually kept."""
    frame = table_rows(tag).set_index("model")
    counts = topic_counts(tag)
    frame["K"] = [counts.get(m, float("nan")) for m in frame.index]
    return frame


def topic_counts(tag: str) -> dict:
    """Mean ``topic_count`` of the same source each table row is built from."""
    import make_result_tables as MRT

    name, baseline_path, fcntm_path = MRT.DATASETS[tag]
    if MRT.FINAL_FCNTM.get(tag) and MRT.FINAL_FCNTM[tag].exists():
        fcntm_path = MRT.FINAL_FCNTM[tag]
    base = pd.read_csv(baseline_path)
    btm = pd.read_csv(OUT / f"rerun_btm_{tag}.csv")
    etm = MRT.load_etm(tag)
    fcntm = pd.read_csv(fcntm_path)
    out = {}
    for model in ("BTM", "CombinedTM", "FC-NTM-cTFIDF", "FC-NTM", "ETM",
                  "ProdLDA", "Top2Vec", "NMF", "BERTopic", "CTMLite", "LDA"):
        if model == "BTM":
            g = btm
        elif model == "ETM":
            g = etm
        elif model.startswith("FC-NTM"):
            g = fcntm[fcntm["model"] == model]
        else:
            g = base[base["model"] == MRT.ALIASES.get(model, model)]
        key = ("topic_count" if "topic_count" in g.columns
               else "K_used" if "K_used" in g.columns else None)
        if not g.empty and key:
            out[model] = float(g[key].mean())
    return out


def fair_table(tag: str) -> pd.DataFrame:
    return pd.read_csv(OUT / f"fair_readout_{tag}_3seed.csv")


def final_table(tag: str) -> pd.DataFrame:
    return pd.read_csv(OUT / f"fcntm_{tag}_final.csv")


def occlusion_seed11() -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(OUT / "occlusion_robustness_github_seed11.csv")
    pivot = frame.pivot(index="model", columns="rate", values="hidden_NLL")
    agree = frame[frame["rate"] == 1.0].set_index("model")["agreement"]
    return pivot, agree


def occlusion_cross() -> pd.DataFrame:
    return pd.read_csv(OUT / "occlusion_comparison.csv")


def fieldview() -> pd.DataFrame:
    parts = [pd.read_csv(OUT / "public_missing_field_fieldview.csv"),
             pd.read_csv(OUT / "public_missing_field_civil_fieldview.csv")]
    frame = pd.concat(parts, ignore_index=True)
    frame["setting"] = frame["setting"].astype(str)
    return frame[frame["setting"] == "fielddrop_on"]


def missing_occlusion() -> pd.DataFrame:
    frame = pd.read_csv(OUT / "public_missing_field_occlusion.csv")
    civil = pd.read_csv(OUT / "public_missing_field_civil_occlusion.csv")
    frame = pd.concat([frame, civil], ignore_index=True)
    frame["setting"] = frame["setting"].astype(str)
    return frame


def ablation(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / f"ablation_public_github_{name}.csv")


def readout(frame: pd.DataFrame, which: str) -> pd.DataFrame:
    return frame[frame["readout"] == which]


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def check_main_tables() -> None:
    """Tables 1/2: K, CV+-sd, NPMI+-sd, UMass, EXCL, IRBO per model."""
    print("tab:agnews / tab:github (main results)")
    for label, tag in ((r"\label{tab:agnews}", "agnews"),
                       (r"\label{tab:github}", "github")):
        art = main_table(tag)
        rows = data_rows(label)
        require_rows(f"{tag} main table", [model_name(l.split("&")[0])
                                           for l in rows], list(art.index))
        for line in rows:
            cells = [c.strip() for c in line.split("&")]
            model = model_name(cells[0])
            if model not in art.index:
                FAILURES.append(f"{tag} {model}: not in the artifacts")
                print(f"  MISSING  {tag} {model}")
                continue
            row = art.loc[model]
            cmp(f"{tag} {model} K", row["K"], cells[1])
            cmp_pm(f"{tag} {model} CV", row["CV"], row["CVsd"], cells[2])
            cmp_pm(f"{tag} {model} NPMI", row["NPMI"], row["NPMIsd"], cells[3])
            cmp(f"{tag} {model} UMass", row["UMass"], printed(cells[4]))
            cmp(f"{tag} {model} EXCL", row["EXCL_KADJ"], printed(cells[5]))
            cmp(f"{tag} {model} IRBO", row["IRBO"], printed(cells[6]))


def check_fair_table() -> None:
    """tab:fair: the two read-outs (beta / class TF-IDF) per model."""
    print("tab:fair (read-out comparison)")
    fair = {t: fair_table(t) for t in ("github", "agnews")}
    final = {t: final_table(t) for t in ("github", "agnews")}
    # the corpus of each row comes from the preceding multirow label, which is
    # why this table is scanned line by line rather than through ``data_rows``
    entries, parsed, corpus = [], {"github": [], "agnews": []}, None
    for line in table_lines(r"\label{tab:fair}"):
        if "multirow" in line:
            corpus = "github" if "GitHub" in line else (
                "agnews" if "AG News" in line else None)
            continue
        if corpus is None or "&" not in line:
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 4:
            continue
        parsed[corpus].append(model_name(cells[1]))
        entries.append((corpus, model_name(cells[1]), cells[2], cells[3]))
    for tag in ("github", "agnews"):
        require_rows(f"{tag} fair table", parsed[tag],
                     sorted(set(fair[tag]["model"]) | {"FC-NTM"}))

    for corpus, model, beta_cell, ct_cell in entries:
        if model.startswith("FC-NTM"):
            b = final[corpus][final[corpus]["model"] == "FC-NTM"]
            c = final[corpus][final[corpus]["model"] == "FC-NTM-cTFIDF"]
            b_cv, b_npmi = b["CV"].mean(), b["NPMI"].mean()
            c_cv, c_npmi = c["CV"].mean(), c["NPMI"].mean()
        else:
            fb = readout(fair[corpus], "beta")
            fc = readout(fair[corpus], "ctfidf")
            fb = fb[fb["model"] == model]
            fc = fc[fc["model"] == model]
            b_cv, b_npmi = fb["CV"].mean(), fb["NPMI"].mean()
            c_cv, c_npmi = fc["CV"].mean(), fc["NPMI"].mean()
        b = printed_slash(beta_cell)
        c = printed_slash(ct_cell)
        if b:
            cmp(f"{corpus} {model} beta CV", b_cv, b[0])
            cmp(f"{corpus} {model} beta NPMI", b_npmi, b[1])
        if c:
            cmp(f"{corpus} {model} ctfidf CV", c_cv, c[0])
            cmp(f"{corpus} {model} ctfidf NPMI", c_npmi, c[1])


def check_occlusion_seed11() -> None:
    """tab:occl-robust: the GitHub seed-11 partial-occlusion curve."""
    print("tab:occl-robust (partial occlusion, seed 11)")
    pivot, agree = occlusion_seed11()
    rows = data_rows(r"\label{tab:occl-robust}")
    require_rows("tab:occl-robust (occlusion)", [model_name(l.split("&")[0])
                                         for l in rows], list(pivot.index))
    for line in rows:
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 6:
            FAILURES.append(f"tab:occl-robust row has {len(cells)} cells: {line!r}")
            print(f"  MISSING  tab:occl-robust row malformed: {line[:60]!r}")
            continue
        model = model_name(cells[0])
        if model not in pivot.index:
            continue
        for rate, cell in zip((0.25, 0.5, 0.75, 1.0), cells[1:5]):
            cmp(f"github seed11 {model} NLL@{rate}", pivot.loc[model, rate],
                printed(cell))
        rel = printed_slash(cells[5])
        if rel:
            cmp(f"github seed11 {model} rel", pivot.loc[model, 1.0]
                / pivot.loc[model, 0.25], rel[0])
            cmp(f"github seed11 {model} agree", agree[model], rel[1])


def check_occlusion_cross() -> None:
    """tab:occl-cross: relative degradation and agreement over the three seeds."""
    print("tab:occl-cross (cross-corpus occlusion)")
    comp = occlusion_cross()
    parsed = {"github": [], "agnews": [], "civil": []}
    corpus = None
    for line in table_lines(r"\label{tab:occl-cross}"):
        if "multirow" in line:
            corpus = "github" if "GitHub" in line else (
                "agnews" if "AG News" in line else (
                    "civil" if "土木" in line else None))
            continue
        if corpus is None or "&" not in line or not line.rstrip().endswith("\\\\"):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 4:
            continue
        parsed[corpus].append(model_name(cells[1]))
    for tag in ("github", "agnews", "civil"):
        require_rows(f"{tag} cross-corpus table", parsed[tag],
                     sorted(comp[comp["corpus"] == tag]["model"].unique()))
    corpus = None
    for line in table_lines(r"\label{tab:occl-cross}"):
        if "multirow" in line:
            corpus = "github" if "GitHub" in line else (
                "agnews" if "AG News" in line else (
                    "civil" if "土木" in line else None))
            continue
        if corpus is None or "&" not in line or not line.rstrip().endswith("\\\\"):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 4:
            continue
        model = model_name(cells[1])
        sub = comp[(comp["corpus"] == corpus) & (comp["model"] == model)]
        if sub.empty:
            continue
        cmp_pm(f"{corpus} {model} rel", sub["rel_degradation"].mean(),
               sub["rel_degradation"].std(ddof=1), cells[2])
        cmp(f"{corpus} {model} agree", sub["agreement@100"].mean(),
            printed(cells[3]))


def check_ladder_table() -> None:
    """New table: the within-corpus field-ratio ladder (2026-10-04 results)."""
    print("table (field-ratio ladder)")
    art = pd.read_csv(OUT / "rerun" / "ladder_summary.csv")
    variant = {"原始": "full", "abs60": "abs60", "abs30": "abs30"}
    readout = {r"$\beta$": "beta", "类 TF-IDF": "cTFIDF"}
    seen = 0
    for line in data_rows(r"\label{tab:ladder}"):
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 5:
            continue
        tag = next((v for key, v in variant.items()
                    if cells[0].startswith(key)), None)
        ro = readout.get(cells[1])
        if tag is None or ro is None:
            FAILURES.append(f"ladder row not understood: {line!r}")
            print(f"  MISSING  ladder row: {line[:60]!r}")
            continue
        seen += 1
        sub = art[(art["variant"] == tag) & (art["readout"] == ro)]
        if sub.empty:
            FAILURES.append(f"ladder {tag}/{ro}: no artifact rows")
            print(f"  MISSING  ladder {tag}/{ro}")
            continue
        cmp(f"ladder {tag} {ro} full CV", sub["full_CV"].iloc[0], cells[2])
        for config, cell in zip(("no-cross", "no-mask"), cells[3:5]):
            row = sub[sub["config"] == config]
            if row.empty:
                FAILURES.append(f"ladder {tag}/{ro}/{config}: missing artifact")
                continue
            clean = cell.replace("$-$", "-").replace("$+$", "+")
            m = re.search(r"(-?\d+\.\d+)\s*\((\d+)/3\)", clean)
            if not m:
                FAILURES.append(f"ladder {tag}/{ro}/{config}: unparsable {cell!r}")
                print(f"  MISSING  ladder cell {cell!r}")
                continue
            cmp(f"ladder {tag} {ro} {config} dCV",
                row["delta_CV"].iloc[0], m.group(1))
            cmp(f"ladder {tag} {ro} {config} positives",
                row["delta_CV_positive"].iloc[0], m.group(2))
    if seen != 6:
        FAILURES.append(f"ladder table: parsed {seen} rows, expected 6")
        print(f"  MISSING  ladder table: parsed {seen}/6 rows")


def check_chunked_table() -> None:
    """New table: 128-token window pooling for the sentence-embedding baselines."""
    print("table (long-document control)")
    trunc = {t: table_rows(t).set_index("model") for t in ("github", "agnews")}
    chunk = {t: pd.read_csv(OUT / "rerun" / f"chunked_v2_benchmark_{t}.csv")
             .set_index("model") for t in ("github", "agnews")}
    seen = 0
    for line in data_rows(r"\label{tab:chunked}"):
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 5:
            continue
        model = model_name(cells[0])
        if model not in trunc["github"].index:
            continue
        seen += 1
        for i, tag in enumerate(("github", "agnews")):
            cmp_pm(f"chunked {model} {tag} truncated CV",
                   trunc[tag].loc[model, "CV"], trunc[tag].loc[model, "CVsd"],
                   cells[1 + 2 * i])
            pooled = cells[2 + 2 * i]
            if "不适用" in pooled:
                continue        # Top2Vec (Doc2Vec): window pooling does not apply
            cmp_pm(f"chunked {model} {tag} pooled CV",
                   chunk[tag].loc[model, "CV_mean"],
                   chunk[tag].loc[model, "CV_std"], pooled)
    if seen != 3:
        FAILURES.append(f"chunked table: parsed {seen} rows, expected 3")
        print(f"  MISSING  chunked table: parsed {seen}/3 rows")


def control_curve(corpus: str, control: str):
    """(mean, sd, mean NLL@100) of one control's relative degradation."""
    frame = pd.read_csv(OUT / f"occlusion_control_{control}_{corpus}.csv")
    rel, nll = [], []
    for _seed, g in frame[frame["rate"] > 0].groupby("seed"):
        g = g.set_index("rate")
        rel.append(float(g.loc[1.0, "hidden_NLL"] / g.loc[0.25, "hidden_NLL"]))
        nll.append(float(g.loc[1.0, "hidden_NLL"]))
    rel, nll = np.asarray(rel), np.asarray(nll)
    return float(rel.mean()), float(rel.std(ddof=1)), float(nll.mean())


def check_occlusion_controls() -> None:
    """The unigram / shuffled-title controls for the relative-degradation metric."""
    print("table (occlusion controls)")
    comp = occlusion_cross()
    labels = {"github": "GitHub", "agnews": "AG News", "civil": "土木文献"}
    seen = 0
    for line in data_rows(r"\label{tab:occl-controls}"):
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 4:
            continue
        corpus = next((c for c, name in labels.items()
                       if cells[0].startswith(name)), None)
        if corpus is None:
            continue
        seen += 1
        for col, source in enumerate(("unigram", "shuffled", "main")):
            parts = [p.strip() for p in cells[col + 1].split("/")]
            if len(parts) != 2:
                FAILURES.append(f"occlusion control {corpus} {source}: "
                                f"unparsable cell {cells[col + 1]!r}")
                continue
            if source == "main":
                g = comp[(comp["corpus"] == corpus)
                         & (comp["model"] == "FC-NTM")]
                mean, sd = g["rel_degradation"].mean(), \
                    g["rel_degradation"].std(ddof=1)
                nll = g["NLL@100"].mean()
            else:
                mean, sd, nll = control_curve(corpus, source)
            cmp_pm(f"occlusion control {corpus} {source}",
                   mean, sd, parts[0])
            cmp(f"occlusion control {corpus} {source} NLL100",
                nll, printed(parts[1]))
    if seen != 3:
        FAILURES.append(f"occlusion controls: parsed {seen} rows, expected 3")
        print(f"  MISSING  occlusion controls: parsed {seen}/3 rows")


def check_fielddrop_control() -> None:
    """The fielddrop_on/off partial-occlusion comparison."""
    print("table (field-dropout control)")
    comp = occlusion_cross()
    labels = {"github": "GitHub", "agnews": "AG News", "civil": "土木文献"}
    seen = 0
    for line in data_rows(r"\label{tab:fielddrop-control}"):
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 3:
            continue
        corpus = next((c for c, name in labels.items()
                       if cells[0].startswith(name)), None)
        if corpus is None:
            continue
        seen += 1
        g = comp[(comp["corpus"] == corpus) & (comp["model"] == "FC-NTM")]
        cmp_pm(f"fielddrop control {corpus} on", g["rel_degradation"].mean(),
               g["rel_degradation"].std(ddof=1), cells[1])
        mean, sd, _ = control_curve(corpus, "fielddrop_off")
        cmp_pm(f"fielddrop control {corpus} off", mean, sd, cells[2])
    if seen != 3:
        FAILURES.append(f"fielddrop control: parsed {seen} rows, expected 3")
        print(f"  MISSING  fielddrop control: parsed {seen}/3 rows")


def check_beta_readout_table() -> None:
    """Table: raw-B vs BN generative read-out sensitivity (review item 11)."""
    print("table (beta read-out sensitivity)")
    art = pd.read_csv(OUT / "bn_readout_sensitivity.csv")
    labels = {"github": "GitHub", "agnews": "AG News", "civil": "土木文献"}
    columns = (("raw_CV", "raw CV"), ("bn_CV", "bn CV"),
               ("raw_NPMI", "raw NPMI"), ("bn_NPMI", "bn NPMI"),
               ("mean_top10_jaccard", "top10 jaccard"),
               ("top1_change_rate", "top1 change"))
    seen = 0
    for line in data_rows(r"\label{tab:beta-readout}"):
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 7:
            continue
        corpus = next((c for c, name in labels.items()
                       if cells[0].startswith(name)), None)
        if corpus is None:
            continue
        seen += 1
        g = art[art["corpus"] == corpus]
        for (column, name), cell in zip(columns, cells[1:]):
            cmp(f"beta readout {corpus} {name}", g[column].mean(),
                printed(cell))
    if seen != 3:
        FAILURES.append(f"beta readout table: parsed {seen} rows, expected 3")
        print(f"  MISSING  beta readout table: parsed {seen}/3 rows")


def check_seed_paired_table() -> None:
    """The 10-seed paired comparison (FC-NTM vs BTM)."""
    print("table (10-seed paired comparison)")
    art = pd.read_csv(OUT / "rerun" / "seed_extension_summary.csv")
    keys = [("github", "main"), ("github", "beta"),
            ("github", "occlusion_vs_best"), ("agnews", "main"),
            ("agnews", "beta"), ("agnews", "occlusion_vs_BTM"),
            ("agnews", "occlusion_vs_best")]
    rows = data_rows(r"\label{tab:seed-paired}")
    if len(rows) != len(keys):
        FAILURES.append(f"seed-paired table: parsed {len(rows)} rows, "
                        f"expected {len(keys)}")
        print(f"  MISSING  seed-paired table: parsed {len(rows)}/{len(keys)}")
        return
    for (corpus, comparison), line in zip(keys, rows):
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 8:
            FAILURES.append(f"seed-paired {corpus}/{comparison}: "
                            f"{len(cells)} cells")
            continue
        sub = art[(art["corpus"] == corpus)
                  & (art["comparison"] == comparison)]
        if sub.empty:
            FAILURES.append(f"seed-paired {corpus}/{comparison}: no artifact")
            continue
        row = sub.iloc[0]
        cmp(f"seed-paired {corpus} {comparison} mean", row["mean"],
            printed(cells[2]))
        cmp(f"seed-paired {corpus} {comparison} ci low", row["ci_low"],
            printed(cells[3]))
        cmp(f"seed-paired {corpus} {comparison} ci high", row["ci_high"],
            printed(cells[4]))
        match = re.search(r"(\d+)/10", cells[5])
        if match:
            cmp(f"seed-paired {corpus} {comparison} positives",
                row["positive"], match.group(1))
        cmp(f"seed-paired {corpus} {comparison} sign p", row["sign_p"],
            printed(cells[6]))
        cmp(f"seed-paired {corpus} {comparison} wilcoxon p",
            row["wilcoxon_p"], printed(cells[7]))


def check_fieldview_table() -> None:
    """tab:fieldview: two tables were merged, so the FC-NTM rows are prose."""
    print("table (field view, FC-NTM rows in prose)")
    fc = fieldview()
    art = fc
    quote = r"(\d+\.\d+)\$\\pm\$\d+\.\d+"
    for corpus, label, marker in (("github", "GitHub", "在 GitHub 上为"),
                                  ("agnews", "AG News", "在 AG News 上为"),
                                  ("civil", "土木文献", "在土木文献上为")):
        pattern = (marker + r"\s*" + quote + r"\s*/\s*" + quote +
                   r"\s*\n?（NPMI\s*" + quote + r"\s*/\s*\$-\$(\d+\.\d+)")
        m = re.search(pattern, prose_text())
        if not m:
            FAILURES.append(f"field-view prose for {corpus} not found")
            print(f"  MISSING  field-view prose ({corpus})")
            continue
        bg_cv, sm_cv, bg_npmi, sm_npmi = m.groups()
        sm_npmi = "-" + sm_npmi
        for view, cv, npmi in (("background", bg_cv, bg_npmi),
                               ("summary", sm_cv, sm_npmi)):
            g = fc[(fc["dataset"] == corpus) & (fc["view"] == view)]
            cmp(f"{label} {view} CV", g["CV"].mean(), cv)
            cmp(f"{label} {view} NPMI", g["NPMI"].mean(), npmi)
    support = re.search(
        r"概要视角的可评分主题数为 GitHub\s*(\d+)--(\d+)、AG News\s*"
        r"(\d+)--(\d+)、土木文献\s*(\d+)--(\d+)", prose_text())
    if not support:
        FAILURES.append("field-view summary support range not found")
        print("  MISSING  field-view support range")
    else:
        got = [int(v) for v in support.groups()]
        for corpus, lo, hi in (("github", got[0], got[1]),
                               ("agnews", got[2], got[3]),
                               ("civil", got[4], got[5])):
            g = art[(art["dataset"] == corpus) & (art["view"] == "summary")]
            n = (int(g["CV_n"].min()), int(g["CV_n"].max()))
            global CHECKS
            CHECKS += 1
            if (lo, hi) != n:
                FAILURES.append(f"{corpus} summary support: artifact {n} "
                                f"vs paper {(lo, hi)}")
                print(f"  MISMATCH {corpus} summary support: {n} vs {(lo, hi)}")


def check_fieldview_baselines() -> None:
    """The cross-model field-restricted read-out (review item 10)."""
    print("table (field-view baselines)")
    fc = fieldview()
    base = pd.read_csv(OUT / "baseline_fieldview.csv")
    labels = {"GitHub": "github", "AG News": "agnews", "土木": "civil"}
    corpus, seen = None, 0
    for line in table_lines(r"\label{tab:fieldview-baselines}"):
        if "multirow" in line:
            corpus = next((c for key, c in labels.items() if key in line),
                          None)
            continue
        if corpus is None or "&" not in line \
                or not line.rstrip().endswith("\\\\"):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 4:
            continue
        model = model_name(cells[1])
        seen += 1
        for offset, view in ((2, "background"), (3, "summary")):
            if model == "FC-NTM":
                g = fc[(fc["dataset"] == corpus) & (fc["view"] == view)]
            else:
                g = base[(base["dataset"] == corpus)
                         & (base["model"] == model)
                         & (base["view"] == view)]
            cmp_pm(f"fieldview baseline {corpus} {model} {view}",
                   g["CV"].mean(), g["CV"].std(ddof=1), cells[offset])
    if seen != 15:
        FAILURES.append(f"fieldview baselines: parsed {seen} rows, "
                        f"expected 15")
        print(f"  MISSING  fieldview baselines: parsed {seen}/15 rows")


def check_conditional_table() -> None:
    """The redundancy / gain table: R^2, CCA, Delta C_V and the BTM gaps."""
    print("redundancy table (held-out measures + BTM gaps)")
    red = json.loads((OUT / "field_redundancy.json").read_text(encoding="utf-8"))
    occl = missing_occlusion()
    rows = data_rows(r"\label{tab:conditional}")
    require_rows("redundancy table", list(range(len(rows))), list(range(2)))
    for label, line in zip(("github", "agnews"), rows):
        cells = [c.strip() for c in line.split("&")]
        cmp(f"{label} Ridge R2", red[label]["ridge_R2_mean"], printed(cells[1]))
        cmp(f"{label} CCA", red[label]["cca_sq_mean"], printed(cells[2]))
        g = occl[(occl["dataset"] == label)
                 & (occl["setting"] == "fielddrop_on")]
        cmp_pm(f"{label} delta CV", g["delta_CV"].mean(),
               g["delta_CV"].std(ddof=1), cells[3])
        gap = printed_slash(cells[4])
        fair = fair_table(label)
        final = final_table(label)
        ctfidf = final[final["model"] == "FC-NTM-cTFIDF"]["CV"].mean()
        btm_ct = fair[(fair["model"] == "BTM")
                      & (fair["readout"] == "ctfidf")]["CV"].mean()
        btm_beta = fair[(fair["model"] == "BTM")
                        & (fair["readout"] == "beta")]["CV"].mean()
        if gap:
            cmp(f"{label} gap (matched)", ctfidf - btm_ct, gap[0])
            cmp(f"{label} gap (beta)", ctfidf - btm_beta, gap[1])


def check_fusion_table() -> None:
    """tab:fusion: the three-step fusion ablation, including the paired signs."""
    print("tab:fusion (fusion ablation)")
    abl = {n: ablation(n) for n in ("nofusion", "fusedmean")}
    final = final_table("github")
    stats = {
        ("beta", "nofusion"): readout(abl["nofusion"], "beta"),
        ("beta", "fusedmean"): readout(abl["fusedmean"], "beta"),
        ("beta", "gate"): final[final["model"] == "FC-NTM"],
        ("cTFIDF", "nofusion"): readout(abl["nofusion"], "cTFIDF"),
        ("cTFIDF", "fusedmean"): readout(abl["fusedmean"], "cTFIDF"),
        ("cTFIDF", "gate"): final[final["model"] == "FC-NTM-cTFIDF"],
    }
    rows = [l for l in data_rows(r"\label{tab:fusion}") if r"$\pm$" in l]
    require_rows("tab:fusion (fusion)", list(range(len(rows))), list(range(6)))
    order = ["nofusion", "fusedmean", "gate"]
    for group, chunk in (("beta", rows[:3]), ("cTFIDF", rows[3:6])):
        previous = None
        for key, line in zip(order, chunk):
            cells = [c.strip() for c in line.split("&")]
            frame = stats[(group, key)].sort_values("seed")
            cmp_pm(f"tab:fusion {group} {key} CV", frame["CV"].mean(),
                   frame["CV"].std(ddof=1), cells[2])
            cmp_pm(f"tab:fusion {group} {key} NPMI", frame["NPMI"].mean(),
                   frame["NPMI"].std(ddof=1), cells[3])
            cmp(f"tab:fusion {group} {key} UMass", frame["UMass"].mean(),
                printed(cells[5]))
            if previous is not None:
                cv_pos = int(np.sum(frame["CV"].to_numpy()
                                    - previous["CV"].to_numpy() > 0))
                npmi_pos = int(np.sum(frame["NPMI"].to_numpy()
                                      - previous["NPMI"].to_numpy() > 0))
                match = re.search(r"(\d+)/(\d+), (\d+)/(\d+)", _strip_markup(cells[4]))
                global CHECKS
                CHECKS += 1
                if match:
                    want = (int(match.group(1)), int(match.group(2)),
                            int(match.group(3)), int(match.group(4)))
                    got = (cv_pos, len(frame), npmi_pos, len(frame))
                    if want != got:
                        FAILURES.append(f"tab:fusion {group} {key} paired "
                                        f"signs: paper {want}, artifacts {got}")
                        print(f"  MISMATCH tab:fusion {group} {key} paired "
                              f"signs: paper {want}, artifacts {got}")
            previous = frame


def check_core_ablation_table() -> None:
    """tab:core-ablation: four component ablations on both corpora (paired deltas)."""
    print("tab:core-ablation (core-mechanism ablation)")
    configs = ["no-cross", "no-consistency", "no-mask", "no-fielddrop"]
    rows = [l for l in data_rows(r"\label{tab:core-ablation}") if printed(l)]
    require_rows("tab:core-ablation (core ablation)", list(range(len(rows))),
                 list(range(6)))
    for line in rows:
        cells = [c.strip() for c in line.split("&")]
        corpus = ("github" if "GitHub" in cells[0]
                  else "agnews" if "AG News" in cells[0]
                  else "civil" if "土木" in cells[0] else None)
        which = "beta" if "beta" in cells[1] else "cTFIDF"
        if corpus is None:
            FAILURES.append(f"tab:core-ablation: unknown corpus in {cells[0]!r}")
            print(f"  MISSING  tab:core-ablation corpus {cells[0]!r}")
            continue
        model = "FC-NTM" if which == "beta" else "FC-NTM-cTFIDF"
        abl = pd.read_csv(OUT / f"ablation_public_{corpus}_core.csv")
        full = final_table(corpus)
        full = full[full["model"] == model].sort_values("seed")
        for j, config in enumerate(configs):
            frame = readout(abl[abl["config"] == config], which).sort_values("seed")
            diff = frame["CV"].to_numpy() - full["CV"].to_numpy()
            cmp(f"tab:core-ablation {corpus} {which} {config} delta CV",
                float(diff.mean()), printed(cells[2 + j]))
            match = re.search(r"\((\d+)/(\d+)\)", _strip_markup(cells[2 + j]))
            global CHECKS
            CHECKS += 1
            if match is None:
                FAILURES.append(f"tab:core-ablation {corpus} {which} {config}: "
                                f"no sign count in {cells[2 + j]!r}")
                print(f"  MISSING  tab:core-ablation {corpus} {which} {config} signs")
                continue
            want = (int(match.group(1)), int(match.group(2)))
            got = (int(np.sum(diff > 0)), len(frame))
            if want != got:
                FAILURES.append(f"tab:core-ablation {corpus} {which} {config} "
                                f"signs: paper {want}, artifacts {got}")
                print(f"  MISMATCH tab:core-ablation {corpus} {which} {config} "
                      f"signs: paper {want}, artifacts {got}")

    # the caption quotes the four main-model values as well
    lines = paper_lines()
    start = next((i for i, l in enumerate(lines)
                  if "主模型绝对值为" in l), None)
    if start is None:
        FAILURES.append("tab:core-ablation caption: main-model values not found")
        print("  MISSING  tab:core-ablation caption values")
        return
    caption = " ".join(lines[start - 2:start + 4])
    # the six main-model values come right after this phrase; later numbers in
    # the same caption are the field-length ratios, which are not checked here
    tail = _strip_markup(caption).split("主模型绝对值为", 1)[-1]
    quoted = NUM_RE.findall(tail)[:6]
    expected = []
    for corpus, model in (("github", "FC-NTM"), ("github", "FC-NTM-cTFIDF"),
                          ("agnews", "FC-NTM"), ("agnews", "FC-NTM-cTFIDF"),
                          ("civil", "FC-NTM"), ("civil", "FC-NTM-cTFIDF")):
        frame = final_table(corpus)
        expected.append(frame[frame["model"] == model]["CV"].mean())
    for value, shown in zip(expected, quoted):
        cmp("tab:core-ablation caption main CV", value, shown)


def check_weight_table() -> None:
    """tab:weight: the lambda_f sweep (both read-outs)."""
    print("tab:weight (fusion weight sweep)")
    abl = {n: ablation(n) for n in ("nofusion", "gate_w0p25", "gate_w1p0")}
    final = final_table("github")
    sweep = {
        "0": (readout(abl["nofusion"], "beta"), readout(abl["nofusion"], "cTFIDF")),
        "0.25": (readout(abl["gate_w0p25"], "beta"),
                 readout(abl["gate_w0p25"], "cTFIDF")),
        "0.5": (final[final["model"] == "FC-NTM"],
                final[final["model"] == "FC-NTM-cTFIDF"]),
        "1.0": (readout(abl["gate_w1p0"], "beta"),
                readout(abl["gate_w1p0"], "cTFIDF")),
    }
    wrows = data_rows(r"\label{tab:weight}")
    require_rows("tab:weight (weight sweep)", list(range(len(wrows))),
                 list(range(4)))
    for line in wrows:
        cells = [c.strip() for c in line.split("&")]
        key = _strip_markup(cells[0])
        if key not in sweep:
            continue
        beta_frame, ct_frame = sweep[key]
        cmp_pm(f"tab:weight {key} beta CV", beta_frame["CV"].mean(),
               beta_frame["CV"].std(ddof=1), cells[1])
        cmp_pm(f"tab:weight {key} beta NPMI", beta_frame["NPMI"].mean(),
               beta_frame["NPMI"].std(ddof=1), cells[2])
        cmp_pm(f"tab:weight {key} ctfidf CV", ct_frame["CV"].mean(),
               ct_frame["CV"].std(ddof=1), cells[3])
        cmp_pm(f"tab:weight {key} ctfidf NPMI", ct_frame["NPMI"].mean(),
               ct_frame["NPMI"].std(ddof=1), cells[4])


def check_whole_field_table() -> None:
    """tab:occlusion: whole-field occlusion for both corpora and both settings."""
    print("tab:occlusion (whole-field occlusion)")
    art = missing_occlusion()
    pairs = [("github", "fielddrop_on"), ("github", "fielddrop_off"),
             ("civil", "fielddrop_on"), ("civil", "fielddrop_off"),
             ("agnews", "fielddrop_on"), ("agnews", "fielddrop_off")]
    rows = [l for l in data_rows(r"\label{tab:occlusion}") if printed(l)]
    require_rows("tab:occlusion (whole-field)", list(range(len(rows))),
                 list(range(6)))
    for (corpus, setting), line in zip(pairs, rows):
        cells = [c.strip() for c in line.split("&")]
        g = art[(art["dataset"] == corpus) & (art["setting"] == setting)]
        for j, (name, value) in enumerate((
                ("full CV", g["full_CV"].mean()),
                ("occ CV", g["occ_CV"].mean()),
                ("delta CV", g["delta_CV"].mean()),
                ("delta NLL", g["delta_NLL_token"].mean()),
                ("agree", g["top1_agreement"].mean()),
                ("JS", g["mean_js"].mean()))):
            cmp(f"tab:occlusion {corpus} {setting} {name}", value,
                printed(cells[2 + j]))


def check_clean_test_table() -> None:
    """The development-free re-scoring table."""
    print("development-free re-scoring table")
    frame = pd.read_csv(OUT / "clean_test_scores.csv")
    final = {t: final_table(t) for t in ("github", "agnews")}
    fair = {t: fair_table(t) for t in ("github", "agnews")}
    # the table has no \label of its own; find it by its *header row*
    # (a prose sentence also mentions "无开发重叠测试集", so the search has to
    # anchor on the column header itself)
    lines = paper_lines()
    start = next((i for i, l in enumerate(lines)
                  if "无开发重叠测试集" in l
                  and re.search(r"语料\s*&\s*模型\s*&", l)), None)
    if start is None:
        FAILURES.append("clean-test table: header row not found")
        print("  MISSING  clean-test table header")
        return
    parsed_rows = []
    for line in lines[start:]:
        line = strip_comments(line)
        if line.strip().startswith(r"\end{tabular}"):
            break
        if "&" not in line or not line.rstrip().endswith("\\\\"):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 4:
            continue
        if not (printed(cells[2]) and printed(cells[3])):
            continue                      # the column header, not a data row
        parsed_rows.append(cells)
    require_rows("clean-test table", list(range(len(parsed_rows))),
                 list(range(8)))
    for cells in parsed_rows:
        corpus = "github" if "GitHub" in cells[0] else (
            "agnews" if "AG News" in cells[0] else None)
        if corpus is None:
            continue
        label = model_name(cells[1])
        if "类 TF-IDF" in cells[1]:
            model, which = "BTM", "ctfidf"
        elif "原生" in cells[1]:
            model, which = "BTM", "beta"
        elif label == "FC-NTM-cTFIDF":
            model, which = "FC-NTM-cTFIDF", None
        else:
            model, which = "FC-NTM", None
        if model == "BTM":
            full = fair[corpus]
            full = full[(full["model"] == "BTM") & (full["readout"] == which)]
            got_full = full["CV"].mean()
            sub = frame[(frame["corpus"] == corpus) & (frame["model"] == "BTM")
                        & (frame["readout"].fillna("") == which)]
            got_clean = sub["CV"].mean()
        else:
            got_full = final[corpus][final[corpus]["model"] == model]["CV"].mean()
            sub = frame[(frame["corpus"] == corpus) & (frame["model"] == model)]
            sub = sub[sub["source"].str.contains(f"fcntm_{corpus}_final")]
            got_clean = sub["CV"].mean()
        cmp(f"{corpus} clean-test {cells[1]} full", got_full, printed(cells[2]))
        cmp(f"{corpus} clean-test {cells[1]} subset", got_clean,
            printed(cells[3]))


# ---------------------------------------------------------------------------
# prose numbers (the only constants kept in this file)
# ---------------------------------------------------------------------------

def prose_text() -> str:
    """The whole paper as one string, so prose numbers can be parsed too."""
    return " ".join(paper_lines())


def find_number(pattern: str, group: int = 1) -> str | None:
    m = re.search(pattern, prose_text(), flags=re.S)
    return m.group(group) if m else None


def check_prose() -> None:
    """Percentages written in the running text (parsed out of main.tex)."""
    print("prose percentages")
    comp = occlusion_cross()

    def rel(corpus, model):
        g = comp[(comp["corpus"] == corpus) & (comp["model"] == model)]
        return (g["rel_degradation"].mean() - 1.0) * 100.0

    # abstract: "遮挡 25%->100% 时 NLL 仅上升 X%"
    cmp("text: abstract GitHub FC-NTM NLL rise", rel("github", "FC-NTM"),
        find_number(r"100\\% 时 NLL 仅上升\s*(\d+\.\d+)\\"))
    # abstract: "BTM 为 X%，NMF 为 Y%" (three-seed means)
    cmp("text: abstract BTM 3-seed NLL rise", rel("github", "BTM"),
        find_number(r"BTM 为\s*(\d+\.\d+)\\"))
    cmp("text: abstract NMF 3-seed NLL rise", rel("github", "NMF"),
        find_number(r"NMF 为\s*(\d+)\\"))

    # section 5.5: the seed-11 list, in the order the sentence prints it
    # the capture has to *include* the NMF number: taking only the text before
    # it let a tampered "NMF 99.2%" pass unnoticed
    segment = re.search(r"每 token NLL 只上升(.*?NMF\s*\d+\.\d+\\%)",
                        prose_text())
    if segment:
        clean = re.sub(r"（[^）]*）", "", segment.group(1))
        shown = re.findall(r"(\d+\.\d+)\\%", clean)
        pivot, _ = occlusion_seed11()
        for model, text_value in zip(("FC-NTM", "LDA", "ProdLDA", "ETM",
                                      "BTM", "NMF"), shown):
            cmp(f"text: GitHub seed11 {model} NLL rise",
                (pivot.loc[model, 1.0] / pivot.loc[model, 0.25] - 1.0) * 100.0,
                text_value)
    else:
        FAILURES.append("text: the seed-11 percentage list was not found")
        print("  MISSING  text: seed-11 percentage list")

    # every "改变者占 X%" / "改变最可能主题" claim: the GitHub share
    art = missing_occlusion()
    share = (1.0 - art[(art["dataset"] == "github")
                       & (art["setting"] == "fielddrop_on")]
             ["top1_agreement"].mean()) * 100.0
    shown = re.findall(r"改变者占\s*(\d+)\\", prose_text())
    if not shown:
        FAILURES.append("text: no '改变者占 X%' claim found")
        print("  MISSING  text: '改变者占' claim")
    for one in shown:
        cmp("text: GitHub share of documents that change topic", share, one)
    share_ag = (1.0 - art[(art["dataset"] == "agnews")
                          & (art["setting"] == "fielddrop_on")]
                ["top1_agreement"].mean()) * 100.0
    cmp("text: AG News share of documents that change topic", share_ag,
        find_number(r"与\s*(\d+)\\%（AG News）的文档改变了最可能主题"))

    # the clean-test sentence in section 5.3: the fifth review found that a
    # wrong figure there (0.591 instead of 0.590) was outside the checker's
    # coverage, so the two matched-read-out pairs are parsed and compared here
    clean = pd.read_csv(OUT / "clean_test_scores.csv")

    def clean_value(corpus, model, readout, column):
        sub = clean[(clean["corpus"] == corpus) & (clean["model"] == model)]
        sub = sub[sub["source"].str.contains("rerun_btm") if model == "BTM"
                  else sub["source"].str.contains(f"fcntm_{corpus}_final")]
        if readout:
            sub = sub[sub["readout"].fillna("") == readout]
        return sub[column].mean()

    for corpus, pattern in (
            ("github", r"在 GitHub 上为\s*(\d+\.\d+)\s*对\s*(\d+\.\d+)\s*"
                       r"（NPMI\s*(\d+\.\d+)\s*对\s*(\d+\.\d+)）"),
            ("agnews", r"在 AG News 上为\s*(\d+\.\d+)\s*对\s*(\d+\.\d+)\s*"
                       r"（NPMI\s*(\d+\.\d+)\s*对\s*(\d+\.\d+)）")):
        match = re.search(pattern, prose_text())
        if not match:
            FAILURES.append(f"text: the clean-test sentence for {corpus} "
                            f"was not found")
            print(f"  MISSING  text: clean-test sentence ({corpus})")
            continue
        ours_cv, btm_cv, ours_npmi, btm_npmi = match.groups()
        cmp(f"text: {corpus} clean-test FC-NTM-cTFIDF CV",
            clean_value(corpus, "FC-NTM-cTFIDF", None, "CV"), ours_cv)
        cmp(f"text: {corpus} clean-test BTM ctfidf CV",
            clean_value(corpus, "BTM", "ctfidf", "CV"), btm_cv)
        cmp(f"text: {corpus} clean-test FC-NTM-cTFIDF NPMI",
            clean_value(corpus, "FC-NTM-cTFIDF", None, "NPMI"), ours_npmi)
        cmp(f"text: {corpus} clean-test BTM ctfidf NPMI",
            clean_value(corpus, "BTM", "ctfidf", "NPMI"), btm_npmi)
        beta = re.search(r"若改用\s*BTM 的原生 \$\\beta\$ 读出，"
                         r"GitHub 变成\s*(\d+\.\d+)\s*对\s*(\d+\.\d+)、"
                         r"AG News 变成\s*(\d+\.\d+)\s*对\s*(\d+\.\d+)",
                         prose_text())
        if beta:
            for tag, ours, theirs in (("github", beta.group(1), beta.group(2)),
                                      ("agnews", beta.group(3), beta.group(4))):
                cmp(f"text: {tag} clean-test FC-NTM-cTFIDF CV (beta sentence)",
                    clean_value(tag, "FC-NTM-cTFIDF", None, "CV"), ours)
                cmp(f"text: {tag} clean-test BTM beta CV",
                    clean_value(tag, "BTM", "beta", "CV"), theirs)

    # 2026-10-04 revision: the seed-matched civil pairing quoted in the abstract
    adv = []
    for _seed, g in comp[comp["corpus"] == "civil"].groupby("seed"):
        rel = g.set_index("model")["rel_degradation"]
        adv.append(float(rel.drop("FC-NTM").min() - rel["FC-NTM"]))
    adv = float(np.mean(adv))
    m = re.search(r"配对差\s*(?:(\$-\$))?(\d+\.\d+)\$\\pm\$0\.001，0/3 种子",
                  prose_text())
    if m:
        shown = ("-" if m.group(1) else "") + m.group(2)
        cmp("text: civil pairing advantage", adv, shown)
    else:
        FAILURES.append("text: civil pairing sentence not found")
        print("  MISSING  text: civil pairing sentence")


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def run_all(path: Path | None = None) -> int:
    global PAPER_TEX, FAILURES, CHECKS
    if path is not None:
        PAPER_TEX = path
    if not PAPER_TEX.exists():
        print("paper-vs-artifact number check: the Chinese manuscript source "
              f"({PAPER_TEX.name}) is not shipped in this repository "
              "(paper/ holds the ASCE source only); skipped.")
        return 0
    FAILURES = []
    CHECKS = 0
    print(f"paper-vs-artifact number check on {PAPER_TEX.name}")
    check_main_tables()
    check_fair_table()
    check_prose()
    check_conditional_table()
    check_fieldview_table()
    check_fieldview_baselines()
    check_occlusion_seed11()
    check_occlusion_cross()
    check_ladder_table()
    check_chunked_table()
    check_occlusion_controls()
    check_fielddrop_control()
    check_beta_readout_table()
    check_seed_paired_table()
    check_fusion_table()
    check_core_ablation_table()
    check_weight_table()
    check_whole_field_table()
    check_clean_test_table()
    if FAILURES:
        print(f"\n{len(FAILURES)} mismatches out of {CHECKS} comparisons")
        return 1
    print(f"\nall {CHECKS} printed numbers match their artifacts")
    return 0


def self_test() -> int:
    """Tamper with copies of main.tex and require the checker to catch each one.

    The cases are the ones the review used to get past the previous version:
    a plain table value, the same value with a legal trailing LaTeX comment,
    a whole table deleted, and a percentage in the running text.
    """
    if not PAPER_TEX.exists():
        print("self-test: Chinese manuscript source not shipped; skipped")
        return 0
    print("self-test: tampering with copies of main.tex")
    text = PAPER_TEX.read_text(encoding="utf-8")
    # the tamper targets are derived from the current text, not hard-coded, so
    # the self-test keeps working after the paper's numbers change
    prose = re.search(r"BTM\s*(\d+\.\d+)\\%", text)
    cases = [("table value", "BTM & 6.630", "BTM & 99.630")]
    row_match = re.search(r"^BTM & .*\\\\$", text, flags=re.M)
    if row_match:
        cells = row_match.group(0).split("&")
        cells[4] = " 99.972 "
        cases.append(("table value + comment",
                      row_match.group(0),
                      "&".join(cells).rstrip() + " % audit comment"))
    if prose:
        cases.append(("prose percentage", prose.group(0),
                      f"BTM {prose.group(1)[:1]}9.1\\%"))
    failures = 0
    for name, target, replacement in cases:
        if target not in text:
            print(f"  cannot find {target!r} to tamper with ({name})")
            failures += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "main.tex"
            copy.write_text(text.replace(target, replacement, 1),
                            encoding="utf-8")
            rc = run_all(copy)
        if rc == 0:
            print(f"  FAILED: {name} was not detected")
            failures += 1
        else:
            print(f"  ok: {name} detected")

    # a table whose data rows are all gone must fail as well
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if r"\label{tab:occl-robust}" in l), None)
    if start is not None:
        out, dropping = [], False
        for i, line in enumerate(lines):
            if i == start:
                dropping, out = True, out
                out.append(line)
                continue
            if dropping:
                if line.strip().startswith(r"\end{tabular}"):
                    dropping = False
                    out.append(line)
                continue
            out.append(line)
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "main.tex"
            copy.write_text("\n".join(out), encoding="utf-8")
            rc = run_all(copy)
        if rc == 0:
            print("  FAILED: a table with no data rows was not detected")
            failures += 1
        else:
            print("  ok: deleted table rows detected")
    if failures:
        return 1
    print("  ok: every tampered copy was rejected")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    return run_all()


if __name__ == "__main__":
    raise SystemExit(main())
