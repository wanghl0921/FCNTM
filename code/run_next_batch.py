# -*- coding: utf-8 -*-
"""Unattended chain: 遮挡协议 + 不对称度阶梯 + chunked 基线（用户 2026-09-21 批准）。

Everything is sequential on purpose (8 GB box, one training at a time).
Each step is resumable and the chain stops on the first failure instead of
skipping ahead:

  1. civil 整字段遮挡：训练 fielddrop_on/off × 3 种子并保存模型状态
     -> outputs/public_missing_field_civil_occlusion.csv (+ fieldview)
     -> outputs/civil_missing_state.pt
  2. civil 部分遮挡曲线：每个种子跑一次 occlusion_robustness（含 LDA/NMF/
     ProdLDA/ETM 基线，FC-NTM 复用第 1 步的状态）
     -> outputs/occlusion_robustness_civil_seed{11,23,37}.csv
  3. 不对称度阶梯：把土木文献的摘要截断到前 60 / 30 词（字段长度比 ≈8 / ≈4），
     每个变体跑 full / no-cross / no-mask × 3 种子
     -> outputs/ablation_civil_abs{60,30}.csv + rerun/ladder_summary.csv
  4. chunked 基线：用 FCNTM_EN_EMBEDDING_CHUNKED=1 重跑上下文相关基线
     -> outputs/rerun/chunked_*_summary.csv

    python run_next_batch.py                 # 全跑
    python run_next_batch.py --only 3        # 只跑第 3 步
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PY = sys.executable
SEEDS = "11,23,37"


def child_env(**extra) -> dict:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[var] = "4"
    env.update(extra)
    return env


def run(cmd, tag, env=None) -> int:
    print(f"\n=== [{time.strftime('%H:%M:%S')}] {tag} ===\n"
          f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    rc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env).returncode
    print(f"=== [{time.strftime('%H:%M:%S')}] {tag} exit={rc} ===", flush=True)
    return rc


def step1() -> int:
    state = OUT / "civil_missing_state.pt"
    cmd = [PY, "-u", "run_public_missing_field.py", "--dataset", "civil",
           "--seeds", SEEDS, "--k", "30", "--lr", "5e-4",
           "--fusion", "gate", "--fusion-norm", "balanced",
           "--fusion-weight", "0.5",
           "--settings", "fielddrop_on,fielddrop_off",
           "--dump-state", "outputs/civil_missing_state.pt",
           "--resume-every", "50",
           "--out", "outputs/public_missing_field_civil_occlusion.csv",
           "--field-view-out", "outputs/public_missing_field_civil_fieldview.csv"]
    if state.exists():
        cmd += ["--reuse-state", "outputs/civil_missing_state.pt"]
    return run(cmd, "1/4 civil 整字段遮挡", child_env())


def step2() -> int:
    for seed in SEEDS.split(","):
        cmd = [PY, "-u", "occlusion_robustness.py", "--corpus", "civil",
               "--seed", seed, "--k", "30",
               "--models", "FC-NTM,LDA,NMF,ProdLDA,ETM",
               "--rates", "0,0.25,0.5,0.75,1.0",
               "--fcntm-state", "outputs/civil_missing_state.pt"]
        rc = run(cmd, f"2/4 civil 部分遮挡 seed={seed}", child_env())
        if rc != 0:
            return rc
    return 0


def step3() -> int:
    for words, tag in ((60, "abs60"), (30, "abs30")):
        csv = OUT / f"ablation_civil_{tag}.csv"
        cmd = [PY, "-u", "run_ablation_public.py", "--dataset", "civil",
               "--abstract-words", str(words),
               "--configs", "full,no-cross,no-mask",
               "--seeds", SEEDS, "--k", "30", "--lr", "5e-4",
               "--max-epochs", "700", "--resume",
               "--out", f"ablation_civil_{tag}.csv"]
        rc = run(cmd, f"3/4 阶梯 {tag}（full / no-cross / no-mask）", child_env())
        if rc != 0:
            return rc
    rc = run([PY, "-u", "summarize_ladder.py"], "3/4 阶梯汇总", child_env())
    return rc


def step4() -> int:
    env = child_env(FCNTM_EN_EMBEDDING_CHUNKED="1")
    for corpus in ("github", "agnews"):
        cmd = [PY, "-u", "benchmark_public.py", "--dataset", corpus,
               "--models", "BERTopic,Top2Vec,CombinedTM",
               "--seeds", SEEDS, "--k", "30", "--topn", "10",
               "--readout", "ctfidf",
               "--out", f"rerun/chunked_runs_{corpus}.csv",
               "--summary", f"rerun/chunked_benchmark_{corpus}.csv"]
        rc = run(cmd, f"4/4 chunked 基线 {corpus}", env)
        if rc != 0:
            return rc
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="1234",
                    help="which steps to run, e.g. '3' or '34'")
    args = ap.parse_args()
    for step in args.only:
        if step not in "1234":
            raise SystemExit(f"unknown step {step!r}")
        rc = {"1": step1, "2": step2, "3": step3, "4": step4}[step]()
        if rc != 0:
            print(f"\n[batch] step {step} failed (exit={rc}); the chain stops "
                  f"here. 已完成的产物都保留，重跑本脚本会继续。", flush=True)
            return rc
    print("\n[batch] all requested steps DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
