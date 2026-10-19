import subprocess, sys, time
from pathlib import Path
ROOT = Path(r"C:\Users\Administrator\Downloads\FCNTM完整代码")
WATCH = ROOT / "outputs" / "rerun" / "verify_newseed4.log"
LOG = ROOT / "outputs" / "rerun" / "repin_after_v6.log"

def run(*args):
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(args)}\n")
        fh.flush()
        subprocess.run(args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)

deadline = time.time() + 3 * 3600
while time.time() < deadline:
    try:
        text = WATCH.read_text(encoding="utf-8")
    except OSError:
        text = ""
    if "pinned fair_readout_agnews_newseeds.csv" in text:
        break
    time.sleep(30)
else:
    raise SystemExit(2)
time.sleep(10)
run(sys.executable, "summarize_seed_extension.py")
run(sys.executable, "results_manifest.py")
run(sys.executable, "results_manifest.py", "--verify")
run(sys.executable, "check_paper_numbers.py")
