# -*- coding: utf-8 -*-
"""Unattended master chain for the three remaining paper gaps.

    1. finish_baseline_reruns.py          CombinedTM / ProdLDA / CTM-lite,
                                         3 seeds x 2 corpora, merged into the
                                         main benchmark tables
    2. AG News fielddrop_off training     the missing half of the 2x2 in
                                         Table "遮挡背景字段" (the control that
                                         is currently GitHub-only)
    3. fair_readout.py, 3 seeds, 2 corpora  the read-out confound table is
                                         single-seed today

Everything is sequential because the box has ~5 GB of usable memory and two
concurrent trainings have OOMed before.  Each step's exit code is propagated
(the 2026-09-13 chain silently continued after an OOM), and a disk guard aborts
before a step can fill the volume the way the CTM checkpoint leak did.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OUT = ROOT / "outputs"


def free_gb() -> float:
    return shutil.disk_usage(str(ROOT)).free / (1024 ** 3)


def run(cmd, tag):
    print(f"\n=== [{time.strftime('%H:%M:%S')}] {tag} "
          f"(free {free_gb():.2f} GB) ===\n$ {' '.join(str(c) for c in cmd)}",
          flush=True)
    rc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT)).returncode
    print(f"=== [{time.strftime('%H:%M:%S')}] {tag} exit={rc} "
          f"(free {free_gb():.2f} GB) ===", flush=True)
    return rc


def main() -> int:
    # The baselines step is already running in a separate background chain
    # (``baseline_reruns3.log``); wait for it rather than restarting the
    # 40-minute CombinedTM fit that is currently in flight.  If that chain
    # failed, fall through and let this chain re-run the step (it resumes).
    external = OUT / "rerun" / "baseline_reruns3.log"
    deadline = time.time() + 12 * 3600
    while time.time() < deadline:
        text = (external.read_text(encoding="utf-8", errors="replace")
                if external.exists() else "")
        if "merge github rows exit=0" in text:
            print("[chain] external baseline chain finished; taking over",
                  flush=True)
            break
        if "exit=1" in text or "exit=2" in text:
            print("[chain] external baseline chain failed; the baselines step "
                  "will be re-run (and resumed) below", flush=True)
            break
        time.sleep(120)
    else:
        print("[chain] timed out waiting for the external baseline chain",
              flush=True)

    steps = []

    # 1) main-table baselines
    steps.append(("baselines", [PY, "-u", "finish_baseline_reruns.py"]))

    # 2) AG News "no field dropout" control (Table: whole-field occlusion)
    steps.append(("agnews fielddrop_off", [
        PY, "-u", "run_public_missing_field.py",
        "--dataset", "agnews", "--seeds", "11,23,37",
        "--settings", "fielddrop_off",
        "--dump-state", "outputs/agnews_fielddrop_off.pt",
        "--resume-every", "50",
    ]))

    # 3) fair read-out table on three seeds per corpus
    for corpus in ("agnews", "github"):
        steps.append((f"fair_readout {corpus}", [
            PY, "-u", "fair_readout.py",
            "--corpus", corpus, "--seeds", "11,23,37", "--k", "30",
            "--models", "ETM,ProdLDA,CTMLite,LDA,NMF",
            "--out", f"fair_readout_{corpus}_3seed.csv",
        ]))

    for tag, cmd in steps:
        if free_gb() < 0.5:
            print(f"[abort] only {free_gb():.2f} GB free before step '{tag}'",
                  flush=True)
            return 1
        rc = run(cmd, tag)
        if rc != 0:
            print(f"[chain] step '{tag}' failed with {rc}; stopping", flush=True)
            return rc
    print("\n[chain] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
