# -*- coding: utf-8 -*-
"""Unattended chain to finish the civil revision experiments (2026-10-12).

The 2026-10-12 review asks for the civil corpus to become a main experiment
(ten seeds, paired statistics), for a concatenated-field single-encoder
control and for a title-only archival task.  The first two jobs are already
running (civil BTM extension, civil FC-NTM seed extension); this chain waits
for them and then runs every remaining step in order on the single 8 GB box.

Order:
  0. wait for the running jobs to exit;
  0.5 train the civil FC-NTM seed extension if it is not on disk yet
      (the first attempt in the morning of 2026-10-12 was killed by OOM and
      the evening retry by a native torch crash; everything stays serial);
  1. stamp the per-seed ``fit_identity`` from the extension details into
     ``fcntm_main_state_civil.pt`` (the delivered github/agnews states carry
     exactly these stamps, and ``verify_checkpoint`` needs them);
  2. ``extend_seeds_fcntm_occlusion --corpora civil`` (also re-pins the
     merged new-seed occlusion file to the final checkpoint digest);
  3. ``extend_seeds_btm --corpora civil`` re-run: pinned fits are reused and
     the BTM rows are merged under the same final identity;
  4. ``extend_seeds_lda_occlusion --corpora civil``;
  5. ``run_concat_control`` on github, agnews and civil (seeds 11/23/37);
  6. ``fair_readout`` for civil (seeds 11/23/37, the main-table baselines);
  7. ``benchmark_public`` civil (BERTopic, Top2Vec, CombinedTM);
  8. ``civil_title_archival`` (title-only OpenAlex topic routing);
  9. ``make_civil_table`` and ``summarize_civil_extension``.

Every step is a fresh subprocess with its own log under ``outputs/rerun``;
the chain stops on the first non-zero exit and never skips ahead.

Usage:  python run_civil_chain_20261012.py [--wait-pids 11424]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
RERUN = ROOT / "outputs" / "rerun"
LOG = RERUN / "civil_chain_20261012.log"
SEEDS = "53,71,89,101,113,127,139"
TRAIN_CMD = [PY, "-u", "run_fcntm_train.py", "--corpus", "civil",
             "--seeds", SEEDS, "--k", "30", "--fusion", "gate",
             "--fusion-norm", "balanced", "--fusion-weight", "0.5",
             "--field-dropout", "0.5", "--max-epochs", "700", "--lr", "5e-4",
             "--recover-nan", "--resume-every", "50",
             "--out", "outputs/rerun/fcntm_civil_seed53plus.csv",
             "--dump-details", "outputs/rerun/fcntm_civil_seed53plus_details.json",
             "--dump-state", "outputs/rerun/fcntm_main_state_civil.pt"]


def say(text: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {text}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def pid_alive(pid: int) -> bool:
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True, encoding="gbk", errors="replace",
    ).stdout
    return str(pid) in out


def wait_pids(pids) -> None:
    pids = [int(p) for p in pids if p]
    alive = [p for p in pids if pid_alive(p)]
    say(f"waiting for PIDs {alive}")
    last = 0.0
    while True:
        alive = [p for p in alive if pid_alive(p)]
        if not alive:
            say("all waited jobs finished")
            return
        if time.time() - last > 600:
            last = time.time()
            say(f"still running: {alive}")
        time.sleep(60)


def child_env() -> dict:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[var] = "4"
    return env


def run(cmd, tag: str) -> int:
    out = RERUN / f"chain_{tag}.log"
    err = RERUN / f"chain_{tag}.err"
    say(f"=== {tag}: {' '.join(str(c) for c in cmd)}")
    with open(out, "w", encoding="utf-8") as fo, \
            open(err, "w", encoding="utf-8") as fe:
        rc = subprocess.run(
            [str(c) for c in cmd], cwd=str(ROOT), env=child_env(),
            stdout=fo, stderr=fe,
        ).returncode
    say(f"=== {tag} exit={rc}")
    return rc


def check_inputs() -> None:
    import pandas as pd

    csv = RERUN / "fcntm_civil_seed53plus.csv"
    state = RERUN / "fcntm_main_state_civil.pt"
    want = {int(s) for s in SEEDS.split(",")}
    if not csv.exists() or not state.exists():
        raise SystemExit("extension outputs missing; not continuing")
    frame = pd.read_csv(csv)
    got = {int(s) for s in frame[frame["model"] == "FC-NTM"]["seed"]}
    if not want <= got:
        raise SystemExit(f"FC-NTM extension incomplete: {sorted(want - got)}")
    import torch

    ck = torch.load(state, map_location="cpu", weights_only=False)
    missing = [s for s in sorted(want) if f"fielddrop_on:{s}" not in ck]
    if missing:
        raise SystemExit(f"state file missing seeds {missing}")
    say("extension outputs verified for all 7 new seeds")


def stamp_identities() -> None:
    import torch

    details = RERUN / "fcntm_civil_seed53plus_details.json"
    state = RERUN / "fcntm_main_state_civil.pt"
    records = json.load(open(details, encoding="utf-8"))
    ck = torch.load(state, map_location="cpu", weights_only=False)
    changed = 0
    for rec in records:
        ident = rec.get("fit_identity")
        if not ident:
            continue
        key = f"fielddrop_on:{int(rec['seed'])}"
        entry = ck.get(key)
        if entry is None:
            continue
        if entry.get("identity_id") == ident.get("id"):
            continue
        entry["identity"] = ident
        entry["identity_id"] = ident.get("id")
        changed += 1
    if changed:
        torch.save(ck, state)
    say(f"stamped fit identities into the state: {changed} records")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pids", default="11424")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    say("=== civil chain start ===")
    wait_pids(args.wait_pids.split(","))
    try:
        check_inputs()
        say("FC-NTM extension outputs already complete")
    except SystemExit as exc:
        say(f"FC-NTM extension incomplete ({exc}); training it now")
        rc = run(TRAIN_CMD, "fcntm_ext_training")
        if rc != 0:
            say(f"chain stops: fcntm_ext_training failed (exit={rc})")
            return rc
        check_inputs()
    stamp_identities()

    steps = [
        ("fcntm_occlusion_ext",
         [PY, "-u", "extend_seeds_fcntm_occlusion.py", "--corpora", "civil"]),
        ("btm_rerun",
         [PY, "-u", "extend_seeds_btm.py", "--corpora", "civil",
          "--seeds", SEEDS]),
        ("lda_occlusion_ext",
         [PY, "-u", "extend_seeds_lda_occlusion.py", "--corpora", "civil"]),
        ("concat_control",
         [PY, "-u", "run_concat_control.py", "--corpora", "github,agnews,civil",
          "--seeds", "11,23,37", "--k", "30"]),
        ("fair_readout_civil",
         [PY, "-u", "fair_readout.py", "--corpus", "civil",
          "--seeds", "11,23,37", "--k", "30",
          "--models", "BTM,ETM,ProdLDA,CTMLite,LDA,NMF",
          "--out", "fair_readout_civil_3seed.csv"]),
        ("benchmark_civil",
         [PY, "-u", "benchmark_public.py", "--dataset", "civil",
          "--models", "BERTopic,Top2Vec,CombinedTM",
          "--seeds", "11,23,37", "--k", "30", "--topn", "10",
          "--readout", "ctfidf",
          "--out", "rerun_main_baselines_civil.csv",
          "--summary", "rerun_main_baselines_civil_summary.csv"]),
        ("title_archival",
         [PY, "-u", "civil_title_archival.py", "--seeds", "11,23,37",
          "--top-labels", "15"]),
        ("make_civil_table", [PY, "-u", "make_civil_table.py"]),
        ("summarize_civil_extension",
         [PY, "-u", "summarize_civil_extension.py"]),
        # re-bind the five new-seed archives to the current code identity:
        # the civil file just produced, and the github/agnews files whose
        # sidecars predate today's later code edits
        ("verify_newseed_archives",
         [PY, "-u", "verify_newseed_archives.py",
          "--corpora", "civil,github,agnews"]),
    ]
    for tag, cmd in steps:
        rc = run(cmd, tag)
        if rc != 0:
            say(f"chain stops: {tag} failed (exit={rc})")
            return rc
    say("=== civil chain DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
