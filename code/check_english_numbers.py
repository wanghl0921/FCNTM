# -*- coding: utf-8 -*-
"""Row- and column-level number checker for the English paper.

2026-10-15 review: the earlier version searched formatted numbers anywhere
inside a table block, so a wrong sign, a wrong Macro-F1 or a swapped method
name still passed.  This version parses every data row of the affected
tables into keyed cells, recomputes the expected values from the per-seed
CSV sources (paired differences from the unrounded seed values), requires
each key exactly once with the exact signed, rounded value, and ships a
``--self-test`` that replays the three tamperings reported in the review.

    python check_english_numbers.py [--self-test]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

import summarize_seed_extension as SSE

ROOT = Path(__file__).resolve().parent
PAPER = (ROOT.parent / "paper") if (ROOT.parent / "paper").exists() \
    else (ROOT.parent / "FCNTM完整论文")
OUT = ROOT / "outputs"
RERUN = OUT / "rerun"

TOL = 5.1e-4


def block(tex: str, label: str) -> str:
    i = tex.index("\\label{%s}" % label)
    start = tex.rindex("\\begin{table}", 0, i)
    end = tex.index("\\end{table}", i)
    return tex[start:end]


def clean_cell(cell: str) -> str:
    t = cell.strip()
    t = re.sub(r"\\multirow\{\d+\}\{\*\}\{([^}]*)\}", r"\1", t)
    t = t.replace("\\fcntm{}-cTFIDF", "FC-NTM-cTFIDF")
    t = t.replace("\\fcntm{}", "FC-NTM")
    t = t.replace("\\textbf{", "")
    t = t.replace("}", "")
    t = t.replace("$+$", "+").replace("$-$", "-")
    t = t.replace("$\\pm$", "±").replace("$", "")
    t = t.replace("\\ ", " ")
    return re.sub(r"\s+", " ", t).strip()


def parse_num(text: str):
    t = text.strip()
    if "±" in t:
        a, b = t.split("±")
        return float(a), float(b)
    return float(t), None


def _strip_spec(body: str) -> str:
    """Drop the leading ``{...}`` column spec, braces counted with depth."""
    if not body.startswith("{"):
        return body
    depth = 0
    for idx, ch in enumerate(body):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return body[idx + 1:]
    return body


def table_rows(tex: str, label: str):
    blk = block(tex, label)
    body = blk.split("\\begin{tabular}", 1)[1]
    body = body.split("\\end{tabular}", 1)[0]
    body = _strip_spec(body)
    rows = []
    for seg in body.split("\\\\"):
        seg = re.sub(r"\\(?:top|mid|bottom)rule", "", seg)
        seg = re.sub(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}", "", seg).strip()
        if "&" not in seg:
            continue
        rows.append([clean_cell(c) for c in seg.split("&")])
    return rows


def close(a, b) -> bool:
    return abs(float(a) - float(b)) <= TOL


def paired_deltas(fc_model: str, corpus: str, readout: str):
    fc = pd.concat([pd.read_csv(OUT / f"fcntm_{corpus}_final.csv"),
                    pd.read_csv(RERUN / f"fcntm_{corpus}_seed53plus.csv")],
                   ignore_index=True)
    btm = pd.concat(
        [pd.read_csv(OUT / f"fair_readout_{corpus}_3seed.csv"),
         pd.read_csv(RERUN / f"fair_readout_{corpus}_newseeds.csv")],
        ignore_index=True)
    btm = btm[btm["model"] == "BTM"]
    a = fc[fc["model"] == fc_model].set_index("seed")["CV"]
    b = btm[btm["readout"] == readout].set_index("seed")["CV"]
    common = sorted(set(a.index) & set(b.index))
    return (a.loc[common] - b.loc[common]).to_numpy()


def occlusion_stats(corpus: str, baseline: str = "BTM"):
    curves = SSE.load_curves(corpus)
    rel = {s: SSE.rel_degradation(curves[s]) for s in curves}
    common = [s for s in SSE.SEEDS
              if s in rel and "FC-NTM" in rel[s] and baseline in rel[s]]
    return SSE.paired_stats(
        [rel[s][baseline] - rel[s]["FC-NTM"] for s in common])


def check_civil_main(tex: str, problems: list) -> None:
    rows = table_rows(tex, "tab:civil")
    exp = {}
    fc = pd.read_csv(OUT / "fcntm_civil_final.csv")
    for model in ("FC-NTM", "FC-NTM-cTFIDF"):
        g = fc[fc["model"] == model]
        exp[model] = (int(round(g["K_used"].mean())),
                      (g["CV"].mean(), g["CV"].std(ddof=1)),
                      (g["NPMI"].mean(), g["NPMI"].std(ddof=1)),
                      g["UMass"].mean(), g["EXCL_KADJ"].mean(),
                      g["IRBO"].mean())
    fair = pd.read_csv(OUT / "fair_readout_civil_3seed.csv")
    for model in ("BTM", "NMF", "LDA", "ProdLDA", "ETM", "CTMLite"):
        g = fair[(fair["model"] == model) & (fair["readout"] == "beta")]
        exp["CTM-lite" if model == "CTMLite" else model] = (
            30, (g["CV"].mean(), g["CV"].std(ddof=1)),
            (g["NPMI"].mean(), g["NPMI"].std(ddof=1)),
            g["UMass"].mean(), g["EXCL_KADJ"].mean(), g["IRBO"].mean())
    bench = pd.read_csv(OUT / "rerun_main_baselines_civil.csv")
    for model in ("CombinedTM", "BERTopic", "Top2Vec"):
        g = bench[bench["model"] == model]
        exp[model] = (int(round(g["topic_count"].mean())),
                      (g["CV"].mean(), g["CV"].std(ddof=1)),
                      (g["NPMI"].mean(), g["NPMI"].std(ddof=1)),
                      g["UMass"].mean(), g["EXCL_KADJ"].mean(),
                      g["IRBO"].mean())
    seen = set()
    for r in rows:
        if len(r) != 7:
            problems.append(f"tab:civil: unexpected row {r}")
            continue
        if r[0] == "Model":
            continue
        model = r[0]
        if model not in exp:
            problems.append(f"tab:civil: unknown model {model!r}")
            continue
        if model in seen:
            problems.append(f"tab:civil: duplicate row for {model}")
            continue
        seen.add(model)
        e = exp[model]
        try:
            topics = int(float(r[1]))
            cv = parse_num(r[2]); npmi = parse_num(r[3])
            umass, _ = parse_num(r[4]); excl, _ = parse_num(r[5])
            irbo, _ = parse_num(r[6])
        except ValueError as exc:
            problems.append(f"tab:civil {model}: unparsable ({exc})")
            continue
        if topics != e[0]:
            problems.append(f"tab:civil {model}: topics {topics} != {e[0]}")
        for name, got, want, tol in (
                ("CV", cv, e[1], TOL), ("NPMI", npmi, e[2], TOL),
                ("UMass", (umass, None), (e[3], None), 5.1e-3),
                ("EXCL", (excl, None), (e[4], None), TOL),
                ("IRBO", (irbo, None), (e[5], None), TOL)):
            ok = abs(got[0] - want[0]) <= tol and (
                got[1] is None or abs(got[1] - want[1]) <= tol)
            if not ok:
                problems.append(f"tab:civil {model} {name}: {got} vs {want}")
    if set(exp) - seen:
        problems.append(f"tab:civil: missing rows {sorted(set(exp) - seen)}")


def check_seed_paired(tex: str, problems: list) -> None:
    rows = table_rows(tex, "tab:seed-paired")
    exp = {}
    for corpus, label in (("github", "GitHub"), ("agnews", "AG News"),
                          ("civil", "Civil corpus")):
        exp[(label, "Class TF--IDF")] = SSE.paired_stats(
            paired_deltas("FC-NTM-cTFIDF", corpus, "ctfidf"))
        exp[(label, "\\beta read-out")] = SSE.paired_stats(
            paired_deltas("FC-NTM", corpus, "beta"))
        exp[(label, "Occlusion (vs. BTM)")] = occlusion_stats(corpus)
    exp[("AG News", "Occlusion (vs. LDA)")] = occlusion_stats("agnews",
                                                               "LDA")
    seen = set()
    for r in rows:
        if len(r) != 8 or r[0] == "Corpus":
            continue
        key = (r[0], r[1])
        if key not in exp:
            problems.append(f"tab:seed-paired: unexpected row {key}")
            continue
        if key in seen:
            problems.append(f"tab:seed-paired: duplicate row {key}")
            continue
        seen.add(key)
        st = exp[key]
        try:
            got = dict(mean=float(r[2]), ci_low=float(r[3]),
                       ci_high=float(r[4]), positive=r[5],
                       sign_p=float(r[6]), wilcoxon_p=float(r[7]))
        except ValueError as exc:
            problems.append(f"tab:seed-paired {key}: unparsable ({exc})")
            continue
        for name in ("mean", "ci_low", "ci_high", "sign_p", "wilcoxon_p"):
            if not close(got[name], st[name]):
                problems.append(
                    f"tab:seed-paired {key} {name}: {got[name]} vs {st[name]}")
        n = int(st.get("n", st.get("n_seeds", 0)))
        want_pos = f"{int(st['positive'])}/{n}"
        if got["positive"] != want_pos:
            problems.append(f"tab:seed-paired {key} positive: "
                            f"{got['positive']} vs {want_pos}")
    if set(exp) - seen:
        problems.append(f"tab:seed-paired: missing rows "
                        f"{sorted(set(exp) - seen)}")


def check_concat(tex: str, problems: list) -> None:
    rows = table_rows(tex, "tab:concat")
    exp = {}
    for corpus, label in (("github", "GitHub"), ("agnews", "AG News"),
                          ("civil", "Civil corpus")):
        cc = pd.read_csv(RERUN / f"concat_control_{corpus}.csv")
        fc = pd.read_csv(OUT / f"fcntm_{corpus}_final.csv")
        for readout, model, cell in (("beta", "FC-NTM", "\\beta"),
                                     ("ctfidf", "FC-NTM-cTFIDF",
                                      "Class TF--IDF")):
            a = cc[cc["readout"] == readout].set_index("seed")["CV"]
            b = fc[fc["model"] == model].set_index("seed")["CV"]
            common = sorted(set(a.index) & set(b.index))
            d = (b.loc[common] - a.loc[common]).to_numpy()
            exp[(label, cell)] = (
                (b.mean(), b.std(ddof=1)), (a.mean(), a.std(ddof=1)),
                d.mean(), int((d > 0).sum()), len(d))
    seen = set()
    for r in rows:
        if len(r) != 5 or r[0] == "Corpus":
            continue
        key = (r[0], r[1])
        if key not in exp:
            problems.append(f"tab:concat: unexpected row {key}")
            continue
        if key in seen:
            problems.append(f"tab:concat: duplicate row {key}")
            continue
        seen.add(key)
        fc_v, cc_v, delta, pos, n = exp[key]
        try:
            got_fc = parse_num(r[2]); got_cc = parse_num(r[3])
            m = re.match(r"^([+-]?\d\.\d+) \((\d+)/(\d+)\)$", r[4])
            got_delta = float(m.group(1))
            got_pos = (int(m.group(2)), int(m.group(3)))
        except (ValueError, AttributeError) as exc:
            problems.append(f"tab:concat {key}: unparsable ({exc})")
            continue
        for name, got, want in (("FC", got_fc, fc_v),
                                ("Concat", got_cc, cc_v)):
            if not (close(got[0], want[0]) and close(got[1], want[1])):
                problems.append(f"tab:concat {key} {name}: {got} vs {want}")
        if not close(got_delta, delta):
            problems.append(f"tab:concat {key} delta: {got_delta} vs {delta}")
        if got_pos != (pos, n):
            problems.append(f"tab:concat {key} pairs: {got_pos} vs {(pos, n)}")
    if set(exp) - seen:
        problems.append(f"tab:concat: missing rows {sorted(set(exp) - seen)}")


def check_archival(tex: str, problems: list) -> None:
    rows = table_rows(tex, "tab:archival")
    arch = pd.read_csv(RERUN / "civil_title_archival.csv")
    exp = {}
    for model, setting, label in (
            ("TFIDF-LR", "title_only",
             ("TF--IDF logistic regression (supervised)", "title")),
            ("FC-NTM", "full_field", ("FC-NTM", "title + abstract")),
            ("FC-NTM", "title_only", ("FC-NTM", "title only")),
            ("Concat", "title_only", ("Concatenated control", "title only")),
            ("BTM", "title_only", ("BTM", "title only")),
            ("LDA", "title_only", ("LDA", "title only"))):
        g = arch[(arch["model"] == model) & (arch["setting"] == setting)]
        if model == "TFIDF-LR":
            exp[label] = (g["accuracy"].mean(), None,
                          g["macro_f1"].mean(), None)
        else:
            exp[label] = (g["accuracy"].mean(), g["accuracy"].std(ddof=1),
                          g["macro_f1"].mean(), g["macro_f1"].std(ddof=1))
    seen = set()
    for r in rows:
        if len(r) != 4 or r[0] == "Method":
            continue
        key = (r[0], r[1])
        if key not in exp:
            problems.append(f"tab:archival: unexpected row {key}")
            continue
        if key in seen:
            problems.append(f"tab:archival: duplicate row {key}")
            continue
        seen.add(key)
        acc, acc_sd, f1, f1_sd = exp[key]
        try:
            got_acc = parse_num(r[2]); got_f1 = parse_num(r[3])
        except ValueError as exc:
            problems.append(f"tab:archival {key}: unparsable ({exc})")
            continue
        if not close(got_acc[0], acc) or (
                acc_sd is not None and not close(got_acc[1], acc_sd)):
            problems.append(f"tab:archival {key} accuracy: {got_acc} vs "
                            f"{(acc, acc_sd)}")
        if not close(got_f1[0], f1) or (
                f1_sd is not None and not close(got_f1[1], f1_sd)):
            problems.append(f"tab:archival {key} macro-F1: {got_f1} vs "
                            f"{(f1, f1_sd)}")
    if set(exp) - seen:
        problems.append(f"tab:archival: missing rows "
                        f"{sorted(set(exp) - seen)}")
    blk = block(tex, "tab:archival")
    for token in ("857", "42.9"):
        if token not in blk:
            problems.append(f"tab:archival caption: missing {token}")


def check_fair_civil(tex: str, problems: list) -> None:
    rows = table_rows(tex, "tab:fair")
    fc = pd.read_csv(OUT / "fcntm_civil_final.csv")
    fair = pd.read_csv(OUT / "fair_readout_civil_3seed.csv")
    beta, ct = {}, {}
    g = fc[fc["model"] == "FC-NTM"]
    beta["FC-NTM"] = (g["CV"].mean(), g["NPMI"].mean())
    g2 = fc[fc["model"] == "FC-NTM-cTFIDF"]
    ct["FC-NTM"] = (g2["CV"].mean(), g2["NPMI"].mean())
    for model in ("NMF", "ETM", "BTM", "LDA", "ProdLDA", "CTMLite"):
        gb = fair[(fair["model"] == model) & (fair["readout"] == "beta")]
        gc = fair[(fair["model"] == model) & (fair["readout"] == "ctfidf")]
        beta[model] = (gb["CV"].mean(), gb["NPMI"].mean())
        ct[model] = (gc["CV"].mean(), gc["NPMI"].mean())
    current = None
    seen = set()
    for r in rows:
        if len(r) != 4 or r[0] == "Corpus":
            continue
        if r[0]:
            current = r[0]
        if current != "Civil corpus":
            continue
        model = r[1].split("(")[0].strip().replace("CTM-lite", "CTMLite")
        if model not in beta:
            problems.append(f"tab:fair civil: unknown model {r[1]!r}")
            continue
        if model in seen:
            problems.append(f"tab:fair civil: duplicate row for {model}")
            continue
        seen.add(model)
        try:
            got_beta = tuple(float(x) for x in r[2].split("/"))
            got_ct = tuple(float(x) for x in r[3].split("/"))
        except ValueError as exc:
            problems.append(f"tab:fair civil {model}: unparsable ({exc})")
            continue
        for name, got, want in (("beta", got_beta, beta[model]),
                                ("ctfidf", got_ct, ct[model])):
            if not (close(got[0], want[0]) and close(got[1], want[1])):
                problems.append(
                    f"tab:fair civil {model} {name}: {got} vs {want}")
    if set(beta) - seen:
        problems.append(f"tab:fair civil: missing rows "
                        f"{sorted(set(beta) - seen)}")


ABSTRACT_CHECKS = [
    ("+0.049", "class TF", 160),
    ("+0.061", "native", 160),
    ("0.66", "labels", 220),
    ("42.9", "split", 220),
]


def check_abstract(tex: str, problems: list) -> None:
    abstract = tex[tex.index("\\begin{abstract}"):tex.index("\\end{abstract}")]
    for needle, anchor, span in ABSTRACT_CHECKS:
        js = [m.start() for m in re.finditer(re.escape(needle), abstract)]
        ks = [m.start() for m in re.finditer(re.escape(anchor), abstract)]
        if not js or not ks or min(
                abs(j - k) for j in js for k in ks) > span:
            problems.append(f"abstract: {needle!r} not linked to "
                            f"{anchor!r} within {span} chars")


def check(tex: str) -> list:
    problems = []
    check_civil_main(tex, problems)
    check_seed_paired(tex, problems)
    check_concat(tex, problems)
    check_archival(tex, problems)
    check_fair_civil(tex, problems)
    check_abstract(tex, problems)
    return problems


def self_test(tex: str, problems: list) -> None:
    tampers = []
    tampers.append(("macro-F1 tamper",
                    tex.replace("0.624$\\pm$0.013", "0.999$\\pm$0.013", 1)))
    tampers.append(("sign tamper",
                    tex.replace("& $+$0.049 &", "& $-$0.049 &", 1)))
    blk = block(tex, "tab:archival")
    swapped = blk.replace("\\fcntm{} & title only", "TMPX", 1)
    swapped = swapped.replace("Concatenated control & title only",
                              "\\fcntm{} & title only", 1)
    swapped = swapped.replace("TMPX", "Concatenated control & title only", 1)
    tampers.append(("name swap", tex.replace(blk, swapped, 1)))
    for name, tampered in tampers:
        if tampered == tex:
            problems.append(f"self-test: {name} did not change the source")
            continue
        if not check(tampered):
            problems.append(f"self-test: {name} passed the checker")
        else:
            print(f"  ok  self-test: {name} is rejected")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    tex = (PAPER / "main_en.tex").read_text(encoding="utf-8")
    problems = check(tex)
    if problems:
        print("english number check FAILED:")
        for p in problems:
            print("  " + p)
        return 1
    print("english number check: every keyed value matches the per-seed CSVs")
    if args.self_test:
        problems = []
        self_test(tex, problems)
        if problems:
            for p in problems:
                print("  " + p)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
