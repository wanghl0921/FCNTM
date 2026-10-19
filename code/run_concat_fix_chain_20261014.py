# -*- coding: utf-8 -*-
"""Unattended chain for the 2026-10-14 review (C1 fix).

Steps (sequential, stop on the first failure, logs in outputs/rerun):

  0. copy fcntm_main_state_{github,agnews,civil}.pt back from
     ``Downloads/FCNTM_检查点_20261014`` (the verification entries expect
     them under outputs/rerun);
  1. retrain the concatenated-field control on all three corpora -- the
     validation mode fix (models.py, review C1) changes early stopping, so
     re-scoring old states is not enough;
  2. re-run the title-only archival task (new Concat states + C4 label
     fallback);
  3. re-pin the new-seed sidecars with verify_newseed_archives (the code
     digest changed with the review fixes);
  4. refresh the civil ten-seed summary;
  5. recompute the affected table values (concat control, archival, civil
     unified read-out) into ``outputs/rerun/concat_fix_numbers_20261014.txt``;
  6. move the three main states back out to the checkpoint folder.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RERUN = ROOT / "outputs" / "rerun"
CKPT = Path(r"C:\Users\Administrator\Downloads\FCNTM_检查点_20261014")
LOG = RERUN / "concat_fix_chain_20261014.log"
PY = sys.executable
MAIN_STATES = ("fcntm_main_state_github.pt", "fcntm_main_state_agnews.pt",
               "fcntm_main_state_civil.pt")


def say(text: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {text}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def child_env() -> dict:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[var] = "4"
    return env


def run(cmd, tag: str) -> int:
    out = RERUN / f"chain_fix_{tag}.log"
    err = RERUN / f"chain_fix_{tag}.err"
    say(f"=== {tag}: {' '.join(str(c) for c in cmd)}")
    with open(out, "w", encoding="utf-8") as fo, \
            open(err, "w", encoding="utf-8") as fe:
        rc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT),
                            env=child_env(), stdout=fo, stderr=fe).returncode
    say(f"=== {tag} exit={rc}")
    return rc


def copy_states_in() -> None:
    for name in MAIN_STATES:
        src = CKPT / "outputs" / "rerun" / name
        dst = RERUN / name
        if not src.exists():
            raise SystemExit(f"checkpoint folder is missing {src}")
        if not dst.exists():
            shutil.copy2(src, dst)
            say(f"copied {name} back into outputs/rerun")


def move_states_out() -> None:
    for name in MAIN_STATES:
        src = RERUN / name
        dst = CKPT / "outputs" / "rerun" / name
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            say(f"moved {name} back to the checkpoint folder")


def recompute_numbers() -> None:
    import pandas as pd

    lines = []
    base = ROOT / "outputs"
    for corpus in ("github", "agnews", "civil"):
        cc = pd.read_csv(base / "rerun" / f"concat_control_{corpus}.csv")
        fc = pd.read_csv(base / f"fcntm_{corpus}_final.csv")
        for readout, model in (("beta", "FC-NTM"),
                               ("ctfidf", "FC-NTM-cTFIDF")):
            a = cc[cc["readout"] == readout].set_index("seed")["CV"]
            b = fc[fc["model"] == model].set_index("seed")["CV"]
            common = sorted(set(a.index) & set(b.index))
            d = (b.loc[common] - a.loc[common]).to_numpy()
            lines.append(
                "concat %-7s %-7s FC %.4f+-%.4f Concat %.4f+-%.4f "
                "dCV %+.4f (pos %d/%d)"
                % (corpus, readout, b.mean(), b.std(ddof=1), a.mean(),
                   a.std(ddof=1), d.mean(), int((d > 0).sum()), len(d)))
    arch = pd.read_csv(RERUN / "civil_title_archival.csv")
    for (model, setting), g in arch.groupby(["model", "setting"]):
        lines.append(
            "archival %-12s %-11s acc %.3f+-%.3f F1 %.3f+-%.3f fallback %s"
            % (model, setting, g["accuracy"].mean(), g["accuracy"].std(ddof=1),
               g["macro_f1"].mean(), g["macro_f1"].std(ddof=1),
               list(g.get("mapping_fallback", pd.Series(dtype=int)))))
    fair = pd.read_csv(base / "fair_readout_civil_3seed.csv")
    fc = pd.read_csv(base / "fcntm_civil_final.csv")
    for name, rows in (("FC-NTM", fc[fc["model"] == "FC-NTM"]),
                       ("FC-NTM-cTFIDF",
                        fc[fc["model"] == "FC-NTM-cTFIDF"])):
        lines.append("civil %-12s beta %.4f/%.4f"
                     % (name, rows["CV"].mean(), rows["NPMI"].mean()))
    for name in ("NMF", "ETM", "BTM", "LDA", "ProdLDA", "CTMLite"):
        g = fair[fair["model"] == name]
        b = g[g["readout"] == "beta"]
        c = g[g["readout"] == "ctfidf"]
        lines.append("civil %-12s beta %.4f/%.4f ctfidf %.4f/%.4f"
                     % (name, b["CV"].mean(), b["NPMI"].mean(),
                        c["CV"].mean(), c["NPMI"].mean()))
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    (RERUN / "concat_fix_numbers_20261014.txt").write_text(
        text, encoding="utf-8")
    say("numbers written to concat_fix_numbers_20261014.txt")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    say("=== concat fix chain start ===")
    copy_states_in()
    steps = [
        ("concat_retrain",
         [PY, "-u", "run_concat_control.py", "--corpora", "github,agnews,civil",
          "--seeds", "11,23,37", "--k", "30"]),
        ("title_archival",
         [PY, "-u", "civil_title_archival.py", "--seeds", "11,23,37",
          "--top-labels", "15"]),
        ("verify_newseed_archives",
         [PY, "-u", "verify_newseed_archives.py", "--corpora",
          "github,agnews,civil"]),
        ("summarize_civil_extension",
         [PY, "-u", "summarize_civil_extension.py"]),
    ]
    for tag, cmd in steps:
        rc = run(cmd, tag)
        if rc != 0:
            say(f"chain stops: {tag} failed (exit={rc}); the main states stay "
                "in outputs/rerun for debugging")
            return rc
    try:
        recompute_numbers()
    except Exception as exc:  # noqa: BLE001
        say(f"number recomputation failed: {exc!r}")
        return 1
    move_states_out()
    say("=== concat fix chain DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
