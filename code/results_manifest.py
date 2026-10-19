# -*- coding: utf-8 -*-
"""One manifest for every number that ends up in the paper.

The review asked for a single results list that pins down *which* file each
table and figure came from, instead of a directory of same-named CSVs whose
provenance has to be guessed.  This script

  1. hashes every corpus file, checkpoint and result artifact that the paper
     reads (SHA-256 + size),
  2. declares, per table/figure label, exactly which artifacts feed it,
  3. records the protocol constants ($K$, seeds, read-outs) next to them,

and writes

    outputs/RESULTS_MANIFEST.json   machine-readable
    outputs/RESULTS_MANIFEST.md     readable table

``python results_manifest.py --verify`` re-hashes the current directory and
reports drifted / missing / new artifacts, so a stale CSV cannot silently stay
in a table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def _paper_dir() -> Path:
    # the open-source repository keeps the paper in a sibling ``paper/``
    # folder; the internal delivery tree uses the Chinese folder name
    for name in ("paper", "FCNTM完整论文"):
        candidate = ROOT.parent / name
        if candidate.exists():
            return candidate
    return ROOT.parent / "FCNTM完整论文"


PAPER = _paper_dir()

K = 30
SEEDS = [11, 23, 37]

# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

# corpus files: the raw splits that every number is computed from
CORPUS = [
    "data/agnews/train.csv",
    "data/agnews/test.csv",
    "data/github_issues/train.csv",
    "data/github_issues/test.csv",
    # civil corpus: the full split and the fixed-ID field-ratio ladder
    "data/civil_openalex/raw.csv",
    "data/civil_openalex/train_full.csv",
    "data/civil_openalex/test_full.csv",
    "data/civil_openalex/train_abs60.csv",
    "data/civil_openalex/test_abs60.csv",
    "data/civil_openalex/train_abs30.csv",
    "data/civil_openalex/test_abs30.csv",
    # original sources that now ship inside the code folder
    "civil_corpus/collect_openalex.py",
    "civil_corpus/civil_openalex_cc.csv",
    "agnews_repo/data/ag_news_csv/train.csv",
    "agnews_repo/data/ag_news_csv/test.csv",
]

# trained parameters: the final FC-NTM checkpoints actually used by the tables
CHECKPOINTS = [
    "outputs/fcntm_github_final_details.json",
    "outputs/fcntm_agnews_final_details.json",
    "outputs/fcntm_civil_final.csv",
    "outputs/fcntm_civil_final_details.json",
    "outputs/github_missing_state.pt",
    "outputs/agnews_missing_state.pt",
    "outputs/github_fielddrop_off.pt",
    "outputs/agnews_fielddrop_off.pt",
    "outputs/civil_missing_state.pt",
    # extra-seed FC-NTM states (required by the 2026-10-11 review)
    "outputs/rerun/fcntm_main_state_github.pt",
    "outputs/rerun/fcntm_main_state_agnews.pt",
    "outputs/rerun/fcntm_main_state_civil.pt",
    # concatenated-field control states (2026-10-14 review C1 retrain)
    "outputs/rerun/concat_state_github.pt",
    "outputs/rerun/concat_state_agnews.pt",
    "outputs/rerun/concat_state_civil.pt",
    # reusable baseline fits used by the fair read-out and field-view tables
    "outputs/rerun/btm_fit_github_seed11.npz",
    "outputs/rerun/btm_fit_github_seed23.npz",
    "outputs/rerun/btm_fit_github_seed37.npz",
    "outputs/rerun/btm_fit_github_seed53.npz",
    "outputs/rerun/btm_fit_github_seed71.npz",
    "outputs/rerun/btm_fit_github_seed89.npz",
    "outputs/rerun/btm_fit_github_seed101.npz",
    "outputs/rerun/btm_fit_github_seed113.npz",
    "outputs/rerun/btm_fit_github_seed127.npz",
    "outputs/rerun/btm_fit_github_seed139.npz",
    "outputs/rerun/btm_fit_agnews_seed11.npz",
    "outputs/rerun/btm_fit_agnews_seed23.npz",
    "outputs/rerun/btm_fit_agnews_seed37.npz",
    "outputs/rerun/btm_fit_agnews_seed53.npz",
    "outputs/rerun/btm_fit_agnews_seed71.npz",
    "outputs/rerun/btm_fit_agnews_seed89.npz",
    "outputs/rerun/btm_fit_agnews_seed101.npz",
    "outputs/rerun/btm_fit_agnews_seed113.npz",
    "outputs/rerun/btm_fit_agnews_seed127.npz",
    "outputs/rerun/btm_fit_agnews_seed139.npz",
    "outputs/rerun/nmf_fit_github_seed11.npz",
    "outputs/rerun/nmf_fit_github_seed23.npz",
    "outputs/rerun/nmf_fit_github_seed37.npz",
    "outputs/rerun/nmf_fit_agnews_seed11.npz",
    "outputs/rerun/nmf_fit_agnews_seed23.npz",
    "outputs/rerun/nmf_fit_agnews_seed37.npz",]

# result files: exactly the CSVs/JSONs the tables and figures read
RESULTS = [
    # main benchmark runs (one row per model/seed, with the topic lists)
    "outputs/public_benchmark_runs_agnews.csv",
    "outputs/public_benchmark_runs_githubtrain6000_test2000.csv",
    # re-runs after the review fixes
    "outputs/rerun_btm_agnews.csv",
    "outputs/rerun_btm_github.csv",
    "outputs/rerun_btm_agnews_readouts.csv",
    "outputs/rerun_btm_github_readouts.csv",
    "outputs/rerun_etm_agnews_fixed.csv",
    "outputs/rerun_etm_github_fixed.csv",
    "outputs/rerun_combinedtm_agnews_fixed.csv",
    "outputs/rerun_combinedtm_github_fixed.csv",
    # FC-NTM main results and the development runs
    "outputs/fcntm_agnews_final.csv",
    "outputs/fcntm_github_final.csv",
    "outputs/codev_agnews_gate.csv",
    "outputs/codev_github_gate_bal.csv",
    # read-out matched comparison
    "outputs/fair_readout_agnews_3seed.csv",
    "outputs/fair_readout_github_3seed.csv",
    # partial occlusion curves + their summary
    "outputs/occlusion_robustness_github_seed11.csv",
    "outputs/occlusion_robustness_github_seed23.csv",
    "outputs/occlusion_robustness_github_seed37.csv",
    "outputs/occlusion_robustness_agnews_seed11.csv",
    "outputs/occlusion_robustness_agnews_seed23.csv",
    "outputs/occlusion_robustness_agnews_seed37.csv",
    "outputs/occlusion_comparison.csv",
    # development-free re-scoring
    "outputs/clean_test_scores.csv",
    # fusion ablation + weight sweep
    "outputs/ablation_public_github_nofusion.csv",
    "outputs/ablation_public_github_fusedmean.csv",
    "outputs/ablation_public_github_fusedmean_w0.csv",
    "outputs/ablation_public_github_gate_w0p25.csv",
    "outputs/ablation_public_github_gate_w1p0.csv",
    # final-protocol core-mechanism ablations (fifth review, item 3)
    "outputs/ablation_public_github_core.csv",
    "outputs/ablation_public_agnews_core.csv",
    "outputs/ablation_public_civil_core.csv",
    # whole-field occlusion and field views
    "outputs/public_missing_field_occlusion.csv",
    "outputs/public_missing_field_fieldview.csv",
    "outputs/public_missing_field_gate.csv",
    "outputs/public_missing_field_agnews_gate.csv",
    # redundancy measures, diversity, topic sample
    "outputs/field_redundancy.json",
    "outputs/field_complementarity.csv",
    "outputs/field_complementarity.json",
    "outputs/irbo_summary.csv",
    "outputs/topic_table_github_seed11.tex",
    # boundary-case evidence (third review round)
    "outputs/rerun/btm_empty_biterm_counts.csv",
    "outputs/rerun/empty_summary_behavior.csv",
    "outputs/rerun/btm_empty_fallback_counts.csv",
    # civil occlusion: partial curves, whole-field occlusion and field views
    "outputs/occlusion_robustness_civil_seed11.csv",
    "outputs/occlusion_robustness_civil_seed23.csv",
    "outputs/occlusion_robustness_civil_seed37.csv",
    "outputs/public_missing_field_civil_occlusion.csv",
    "outputs/public_missing_field_civil_fieldview.csv",
    # fixed-ID field-ratio ladder (abs60 / abs30 + the full reference)
    "outputs/ablation_civil_abs60.csv",
    "outputs/ablation_civil_abs30.csv",
    "outputs/rerun/ladder_summary.csv",
    # long-document controls for the sentence-embedding baselines
    "outputs/rerun/chunked_runs_github.csv",
    "outputs/rerun/chunked_benchmark_github.csv",
    "outputs/rerun/chunked_runs_agnews.csv",
    "outputs/rerun/chunked_benchmark_agnews.csv",
    "outputs/rerun/chunked_v2_runs_github.csv",
    "outputs/rerun/chunked_v2_benchmark_github.csv",
    "outputs/rerun/chunked_v2_runs_agnews.csv",
    "outputs/rerun/chunked_v2_benchmark_agnews.csv",
    # review item 6 controls: constant unigram / shuffled item field /
    # fielddrop_off partial-occlusion curves
    "outputs/occlusion_control_unigram_github.csv",
    "outputs/occlusion_control_unigram_agnews.csv",
    "outputs/occlusion_control_unigram_civil.csv",
    "outputs/occlusion_control_shuffled_github.csv",
    "outputs/occlusion_control_shuffled_agnews.csv",
    "outputs/occlusion_control_shuffled_civil.csv",
    "outputs/occlusion_control_fielddrop_off_github.csv",
    "outputs/occlusion_control_fielddrop_off_agnews.csv",
    "outputs/occlusion_control_fielddrop_off_civil.csv",
    # beta read-out sensitivity: raw parameter rows vs BN generative rows
    "outputs/bn_readout_sensitivity.csv",
    # cross-model field-restricted read-out (review item 10)
    "outputs/baseline_fieldview.csv",
    # 10-seed extension of FC-NTM and BTM (review: paired significance)
    "outputs/rerun/fcntm_github_seed53plus.csv",
    "outputs/rerun/fcntm_agnews_seed53plus.csv",
    "outputs/rerun/fcntm_github_seed53plus_details.json",
    "outputs/rerun/fcntm_agnews_seed53plus_details.json",
    "outputs/rerun/fair_readout_github_newseeds.csv",
    "outputs/rerun/fair_readout_agnews_newseeds.csv",
    "outputs/rerun/occlusion_robustness_github_newseeds.csv",
    "outputs/rerun/occlusion_robustness_agnews_newseeds.csv",
    "outputs/rerun/seed_extension_summary.csv",
    # civil ten-seed chain and the 2026-10-14 additions (concat control,
    # title-only routing, OpenAlex label cache)
    "outputs/rerun/fcntm_civil_seed53plus.csv",
    "outputs/rerun/fcntm_civil_seed53plus_details.json",
    "outputs/rerun/fair_readout_civil_newseeds.csv",
    "outputs/rerun/occlusion_robustness_civil_newseeds.csv",
    "outputs/rerun/civil_seed_extension_summary.csv",
    "outputs/fair_readout_civil_3seed.csv",
    "outputs/rerun_main_baselines_civil.csv",
    "outputs/rerun_main_baselines_civil_summary.csv",
    "outputs/civil_main_table.csv",
    "outputs/rerun/concat_control_github.csv",
    "outputs/rerun/concat_control_agnews.csv",
    "outputs/rerun/concat_control_civil.csv",
    "outputs/rerun/civil_title_archival.csv",
    "outputs/civil_openalex_topics.csv",
    "outputs/civil_openalex_topics_meta.json",
    "outputs/rerun/civil_topic_cases_20261014.md",
    "outputs/rerun/concat_fix_numbers_20261014.txt",
]

# code that produced the numbers (so the manifest pins the implementation too)
CODE = [
    "config.py",
    "metrics.py",
    "public_data.py",
    "public_data_github.py",
    "models.py",
    "baselines.py",
    "new_baselines.py",
    "experiment.py",
    "benchmark_public.py",
    "run_fcntm_train.py",
    "run_public_missing_field.py",
    "occlusion_robustness.py",
    "fair_readout.py",
    "rescore_clean_test.py",
    "dump_btm_readout_topics.py",
    "add_btm_doc_topic.py",
    "recompute_irbo.py",
    "run_ablation_public.py",
    "run_core_ablations.py",
    "embed_public.py",
    "public_data_civil.py",
    "make_result_tables.py",
    "make_topic_table.py",
    "compare_occlusion.py",
    "check_paper_numbers.py",
    "result_identity.py",
    "rerun_btm_nmf_boundary_fix.py",
    "check_empty_summary_behavior.py",
    "check_btm_empty_biterm_counts.py",
    "check_missing_decomposition.py",
    "results_manifest.py",
    "field_redundancy.py",
    "field_complementarity.py",
    "summarize_ladder.py",
    "add_civil_btm_occlusion.py",
    "occlusion_controls.py",
    "bn_readout_sensitivity.py",
    "baseline_fieldview.py",
    "extend_seeds_btm.py",
    "extend_seeds_fcntm_occlusion.py",
    "extend_seeds_lda_occlusion.py",
    "summarize_seed_extension.py",
    "seed_identity.py",
    "verify_newseed_archives.py",
    "repin_fieldview_archive.py",
    "finish_fieldview_regen.py",
    "run_concat_control.py",
    "civil_title_archival.py",
    "summarize_civil_extension.py",
    "fetch_openalex_topics.py",
    "make_civil_table.py",
    "make_civil_topic_cases_20261014.py",
    "check_english_numbers.py",
]

# --------------------------------------------------------------------------
# which artifact feeds which table / figure
# --------------------------------------------------------------------------

GITHUB_MAIN = [
    "outputs/public_benchmark_runs_githubtrain6000_test2000.csv",
    "outputs/rerun_btm_github.csv",
    "outputs/rerun_etm_github_fixed.csv",
    "outputs/rerun_combinedtm_github_fixed.csv",
    "outputs/fcntm_github_final.csv",
]
AGNEWS_MAIN = [
    "outputs/public_benchmark_runs_agnews.csv",
    "outputs/rerun_btm_agnews.csv",
    "outputs/rerun_etm_agnews_fixed.csv",
    "outputs/rerun_combinedtm_agnews_fixed.csv",
    "outputs/fcntm_agnews_final.csv",
]
FAIR = [
    "outputs/fair_readout_github_3seed.csv",
    "outputs/fair_readout_agnews_3seed.csv",
    "outputs/fcntm_github_final.csv",
    "outputs/fcntm_agnews_final.csv",
]
OCCL = [f"outputs/occlusion_robustness_{c}_seed{s}.csv"
        for c in ("github", "agnews", "civil") for s in SEEDS]

TABLES = {
    "fig:seedpaired": [
        "outputs/fcntm_civil_final.csv",
        "outputs/rerun/fcntm_civil_seed53plus.csv",
        "outputs/fair_readout_civil_3seed.csv",
        "outputs/rerun/fair_readout_civil_newseeds.csv",
        "outputs/fcntm_github_final.csv",
        "outputs/rerun/fcntm_github_seed53plus.csv",
        "outputs/fair_readout_github_3seed.csv",
        "outputs/rerun/fair_readout_github_newseeds.csv",
        "outputs/fcntm_agnews_final.csv",
        "outputs/rerun/fcntm_agnews_seed53plus.csv",
        "outputs/fair_readout_agnews_3seed.csv",
        "outputs/rerun/fair_readout_agnews_newseeds.csv",
    ],
    "tab:agnews": AGNEWS_MAIN,
    "tab:github": GITHUB_MAIN,
    "tab:fair": FAIR,
    "fig:main": AGNEWS_MAIN + GITHUB_MAIN,
    "tab:occl-robust": ["outputs/occlusion_robustness_github_seed11.csv"],
    "fig:occlusion": ["outputs/occlusion_robustness_github_seed11.csv"],
    "tab:occl-cross": ["outputs/occlusion_comparison.csv"] + OCCL,
    "fig:occl-cross": OCCL,
    "tab:occlusion": [
        "outputs/public_missing_field_occlusion.csv",
        "outputs/public_missing_field_civil_occlusion.csv",
        "outputs/civil_missing_state.pt",
    ],
    "tab:fieldview": [
        "outputs/public_missing_field_fieldview.csv",
        "outputs/public_missing_field_civil_fieldview.csv",
    ],
    "tab:fieldview-baselines": [
        "outputs/baseline_fieldview.csv",
        "outputs/public_missing_field_fieldview.csv",
        "outputs/public_missing_field_civil_fieldview.csv",
    ],
    "tab:fusion": [
        "outputs/ablation_public_github_nofusion.csv",
        "outputs/ablation_public_github_fusedmean.csv",
        "outputs/ablation_public_github_fusedmean_w0.csv",
        "outputs/fcntm_github_final.csv",
    ],
    "tab:weight": [
        "outputs/ablation_public_github_nofusion.csv",
        "outputs/ablation_public_github_gate_w0p25.csv",
        "outputs/ablation_public_github_gate_w1p0.csv",
        "outputs/fcntm_github_final.csv",
    ],
    "tab:core-ablation": [
        "outputs/ablation_public_github_core.csv",
        "outputs/ablation_public_agnews_core.csv",
        "outputs/ablation_public_civil_core.csv",
        "outputs/fcntm_github_final.csv",
        "outputs/fcntm_agnews_final.csv",
        "outputs/fcntm_civil_final.csv",
    ],
    "tab:conditional": [
        "outputs/field_redundancy.json",
        "outputs/field_complementarity.csv",
    ] + FAIR,
    "fig:cond": [
        "outputs/field_complementarity.csv",
        "outputs/field_redundancy.json",
    ] + FAIR,
    "tab:topics": [
        "outputs/topic_table_github_seed11.tex",
        "outputs/fcntm_github_final_details.json",
    ],
    "tab:ladder": [
        "outputs/rerun/ladder_summary.csv",
        "outputs/ablation_civil_abs60.csv",
        "outputs/ablation_civil_abs30.csv",
        "outputs/ablation_public_civil_core.csv",
        "outputs/fcntm_civil_final.csv",
        "data/civil_openalex/train_full.csv",
        "data/civil_openalex/train_abs60.csv",
        "data/civil_openalex/train_abs30.csv",
    ],
    "tab:chunked": [
        "outputs/rerun/chunked_v2_benchmark_github.csv",
        "outputs/rerun/chunked_v2_benchmark_agnews.csv",
    ] + GITHUB_MAIN + AGNEWS_MAIN,
    "tab:occl-controls": [
        "outputs/occlusion_control_unigram_github.csv",
        "outputs/occlusion_control_unigram_agnews.csv",
        "outputs/occlusion_control_unigram_civil.csv",
        "outputs/occlusion_control_shuffled_github.csv",
        "outputs/occlusion_control_shuffled_agnews.csv",
        "outputs/occlusion_control_shuffled_civil.csv",
        "outputs/occlusion_comparison.csv",
    ],
    "tab:fielddrop-control": [
        "outputs/occlusion_control_fielddrop_off_github.csv",
        "outputs/occlusion_control_fielddrop_off_agnews.csv",
        "outputs/occlusion_control_fielddrop_off_civil.csv",
        "outputs/occlusion_comparison.csv",
    ],
    "tab:beta-readout": ["outputs/bn_readout_sensitivity.csv"],
    "tab:seed-paired": [
        "outputs/rerun/seed_extension_summary.csv",
        "outputs/rerun/fcntm_github_seed53plus.csv",
        "outputs/rerun/fcntm_agnews_seed53plus.csv",
        "outputs/rerun/fair_readout_github_newseeds.csv",
        "outputs/rerun/fair_readout_agnews_newseeds.csv",
        "outputs/rerun/occlusion_robustness_github_newseeds.csv",
        "outputs/rerun/occlusion_robustness_agnews_newseeds.csv",
    ],
}

# the clean-test table is written into the main-results section
TABLES["tab:clean-test"] = ["outputs/clean_test_scores.csv"]
TABLES["tab:civil"] = [
    "outputs/fcntm_civil_final.csv",
    "outputs/fair_readout_civil_3seed.csv",
    "outputs/rerun_main_baselines_civil_summary.csv",
]
TABLES["tab:concat"] = [
    "outputs/rerun/concat_control_github.csv",
    "outputs/rerun/concat_control_agnews.csv",
    "outputs/rerun/concat_control_civil.csv",
    "outputs/fcntm_github_final.csv",
    "outputs/fcntm_agnews_final.csv",
    "outputs/fcntm_civil_final.csv",
]
TABLES["tab:archival"] = [
    "outputs/rerun/civil_title_archival.csv",
    "outputs/civil_openalex_topics.csv",
]
TABLES["tab:civil-cases"] = [
    "outputs/rerun/civil_topic_cases_20261014.md",
    "outputs/fcntm_civil_final_details.json",
    "outputs/civil_missing_state.pt",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(rel: str) -> dict:
    path = ROOT / rel
    if not path.exists():
        return {"path": rel, "status": "missing"}
    stat = path.stat()
    return {
        "path": rel,
        "status": "ok",
        "sha256": sha256(path),
        "bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
    }


def build() -> dict:
    groups = {"corpus": CORPUS, "checkpoint": CHECKPOINTS,
              "result": RESULTS, "code": CODE}
    artifacts = {name: [entry(rel) for rel in rels]
                 for name, rels in groups.items()}
    paper = {}
    for rel in ("main.tex", "main.pdf", "supplementary.tex",
                "supplementary.pdf", "main_en.tex", "main_en.pdf",
                "supplementary_en.tex", "supplementary_en.pdf",
                "main_asce.tex", "main_asce.pdf", "main_asce.bib",
                "FC-NTM_manuscript.docx", "FC-NTM_manuscript_ASCE.docx",
                "FC-NTM_manuscript_zh.docx",
                "ascelike-new.cls", "ascelike-new.bst",
                "make_paper_figures.py",
                "topic_table_github_seed11_paper.tex",
                "fig_main_comparison.pdf", "fig_occlusion.pdf",
                "fig_occlusion_cross_corpus.pdf", "fig_complementarity.pdf",
                "fig_architecture_en.pdf", "fig_seed_paired_en.pdf"):
        path = PAPER / rel
        if path.exists():
            stat = path.stat()
            paper[rel] = {
                "sha256": sha256(path),
                "bytes": stat.st_size,
                "mtime": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            }
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "protocol": {"K": K, "seeds": SEEDS, "topn": 10,
                     "readouts": ["beta", "ctfidf"]},
        "artifacts": artifacts,
        "tables": TABLES,
        "paper": paper,
    }


def write_markdown(doc: dict, path: Path) -> None:
    lines = ["# RESULTS MANIFEST", "",
             f"- generated: {doc['generated_utc']}",
             f"- python: {doc['python']} on {doc['platform']}",
             f"- protocol: K={doc['protocol']['K']}, "
             f"seeds={doc['protocol']['seeds']}, "
             f"topn={doc['protocol']['topn']}, "
             f"read-outs={doc['protocol']['readouts']}", ""]
    for group, items in doc["artifacts"].items():
        lines += [f"## {group}", "",
                  "| path | sha256 | bytes |", "|---|---|---|"]
        for it in items:
            if it["status"] != "ok":
                lines.append(f"| {it['path']} | **MISSING** | |")
            else:
                lines.append(f"| {it['path']} | `{it['sha256'][:16]}` | "
                             f"{it['bytes']} |")
        lines.append("")
    lines += ["## table / figure -> artifacts", "",
              "| label | artifacts |", "|---|---|"]
    for label, rels in doc["tables"].items():
        lines.append(f"| {label} | " + "<br>".join(rels) + " |")
    lines += ["", "## paper files", "",
              "| file | sha256 | bytes |", "|---|---|---|"]
    for rel, it in doc["paper"].items():
        lines.append(f"| {rel} | `{it['sha256'][:16]}` | {it['bytes']} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def verify(path: Path) -> int:
    doc = json.loads(path.read_text(encoding="utf-8"))
    bad = missing = new = 0
    for group, items in doc["artifacts"].items():
        for it in items:
            p = ROOT / it["path"]
            if not p.exists():
                print(f"MISSING  {it['path']}")
                missing += 1
                continue
            if it.get("status") != "ok":
                print(f"NEW      {it['path']} (was missing when written)")
                new += 1
                continue
            now = sha256(p)
            if now != it["sha256"]:
                print(f"DRIFT    {it['path']}\n"
                      f"         manifest {it['sha256'][:16]}\n"
                      f"         current  {now[:16]}")
                bad += 1
    # the paper itself is part of the result chain: a table edited after the
    # manifest was written must show up here (the review found that the paper
    # group was recorded but never verified)
    for rel, it in doc.get("paper", {}).items():
        p = PAPER / rel
        if not p.exists():
            print(f"MISSING  {rel} (paper)")
            missing += 1
            continue
        now = sha256(p)
        if now != it["sha256"]:
            print(f"DRIFT    {rel} (paper)\n"
                  f"         manifest {it['sha256'][:16]}\n"
                  f"         current  {now[:16]}")
            bad += 1
    # every data-driven table/figure in the paper must name its artifacts
    import re as _re
    allow_unmapped = {"fig:arch"}     # static architecture drawing, no numbers
    labels = set()
    for name in ("main.tex", "main_en.tex", "supplementary.tex",
                 "supplementary_en.tex"):
        tex = (PAPER / name).read_text(encoding="utf-8")
        labels |= set(_re.findall(r"\\label\{((?:tab|fig):[^}]+)\}", tex))
    unmapped = sorted(labels - set(doc.get("tables", {})) - allow_unmapped)
    for label in unmapped:
        print(f"UNMAPPED {label} (no artifact mapping in the manifest)")
    bad += len(unmapped)
    print(f"[verify] drift={bad} missing={missing} new={new} "
          f"(manifest {path.name})")
    return 1 if (bad or missing) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check the current tree against the manifest")
    args = ap.parse_args()
    js = OUT / "RESULTS_MANIFEST.json"
    if args.verify:
        if not js.exists():
            print("no manifest yet; run without --verify first")
            return 2
        return verify(js)
    doc = build()
    js.write_text(json.dumps(doc, indent=1, ensure_ascii=False),
                  encoding="utf-8")
    write_markdown(doc, OUT / "RESULTS_MANIFEST.md")
    n = sum(len(v) for v in doc["artifacts"].values())
    miss = [it["path"] for items in doc["artifacts"].values() for it in items
            if it["status"] != "ok"]
    print(f"wrote {js.name} + RESULTS_MANIFEST.md ({n} artifacts)")
    if miss:
        print("missing:", *miss, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
