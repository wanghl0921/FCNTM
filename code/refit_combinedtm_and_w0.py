# -*- coding: utf-8 -*-
"""After the ETM refit: CombinedTM best-state refit + the lambda_f = 0 control.

Two review follow-ups that need training, queued behind the ETM chain so only
one model trains at a time:

  1. CombinedTM now scores its best-validation state (``best_components``)
     instead of the last epoch of the early-stopped run, so its two rows have
     to be regenerated and merged back into the main tables.
  2. ``fused-w0`` keeps the fused branch enabled with its loss multiplied by
     zero, separating "one more objective" from "one more decoder forward"
     (BatchNorm running stats and RNG consumption).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OUT = ROOT / "outputs"
RUNS = {
    "agnews": OUT / "public_benchmark_runs_agnews.csv",
    "github": OUT / "public_benchmark_runs_githubtrain6000_test2000.csv",
}
SUMMARIES = {
    "agnews": OUT / "public_benchmark_summary_agnews.csv",
    "github": OUT / "public_benchmark_summary_github.csv",
}
DEADLINE_HOURS = 5.0


def run(cmd, tag):
    print(f"\n=== [{time.strftime('%H:%M:%S')}] {tag} ===\n"
          f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    rc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT)).returncode
    print(f"=== [{time.strftime('%H:%M:%S')}] {tag} exit={rc} ===", flush=True)
    return rc


def main() -> int:
    log = OUT / "rerun" / "refit_etm.log"
    deadline = time.time() + DEADLINE_HOURS * 3600
    while time.time() < deadline:
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        if "ETM refit DONE" in text:
            break
        if "exit=1" in text or "Traceback" in text:
            print("[chain] the ETM chain failed; continuing anyway", flush=True)
            break
        time.sleep(120)
    else:
        print("[chain] timed out waiting for the ETM chain", flush=True)
        return 1

    print("[chain] ETM chain finished; refitting CombinedTM", flush=True)
    for corpus, runs in RUNS.items():
        rc = run([PY, "-u", "benchmark_public.py", "--dataset", corpus,
                  "--models", "CombinedTM", "--seeds", "11,23,37", "--k", "30",
                  "--no-resume",
                  "--out", f"rerun_combinedtm_{corpus}_fixed.csv",
                  "--summary", f"rerun_combinedtm_{corpus}_fixed_summary.csv"],
                 f"CombinedTM refit {corpus}")
        if rc != 0:
            return rc
        rc = run([PY, "-u", "merge_benchmark_rows.py", runs,
                  OUT / f"rerun_combinedtm_{corpus}_fixed.csv",
                  "--summary", SUMMARIES[corpus]],
                 f"merge CombinedTM {corpus}")
        if rc != 0:
            return rc

    rc = run([PY, "-u", "run_ablation_public.py", "--dataset", "github",
              "--seeds", "11,23,37", "--k", "30", "--lr", "5e-4",
              "--configs", "fused-w0",
              "--out", "ablation_public_github_fusedmean_w0.csv"],
             "ablation fused-w0 (branch kept, weight 0)")
    if rc != 0:
        return rc
    print("\n[chain] CombinedTM + w0 DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
