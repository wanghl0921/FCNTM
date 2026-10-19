# -*- coding: utf-8 -*-
"""Point every in-tree path at the consolidated layout (no sibling folder)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
fails = []


def patch(name, pairs):
    path = ROOT / name
    raw = path.read_text(encoding="utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n") if crlf else raw
    for tag, old, new, count in pairs:
        n = text.count(old)
        if n != count:
            fails.append(f"{name}: {tag}: found {n}, want {count}")
            continue
        text = text.replace(old, new)
    if not any(f.startswith(name) for f in fails):
        out = text.replace("\n", "\r\n") if crlf else text
        path.write_text(out, encoding="utf-8")
        print(f"  patched {name}")


patch("results_manifest.py", [(
    "checkpoint paths",
    '''    # extra-seed FC-NTM states are delivered outside the code folder (the
    # 2026-10-11 review asked for them; they exceed the 512 MB budget in-tree)
    "../FCNTM_10seed_checkpoints/fcntm_main_state_github.pt",
    "../FCNTM_10seed_checkpoints/fcntm_main_state_agnews.pt",''',
    '''    # extra-seed FC-NTM states (required by the 2026-10-11 review)
    "outputs/rerun/fcntm_main_state_github.pt",
    "outputs/rerun/fcntm_main_state_agnews.pt",''',
    1)])

patch("extend_seeds_fcntm_occlusion.py", [(
    "default state dir",
    '''DEFAULT_STATE_DIR = Path(__file__).resolve().parent.parent / \\
    "FCNTM_10seed_checkpoints"''',
    '''DEFAULT_STATE_DIR = Path(__file__).resolve().parent / "outputs" / "rerun"
EXTERNAL_STATE_DIR = Path(__file__).resolve().parent.parent / \\
    "FCNTM_10seed_checkpoints"''',
    1), (
    "fallback dir",
    '''        state = Path(args.state_dir) / f"fcntm_main_state_{corpus}.pt"
        if not state.exists():
            alt = O.OUT_DIR / "rerun" / f"fcntm_main_state_{corpus}.pt"
            if alt.exists():
                state = alt''',
    '''        state = Path(args.state_dir) / f"fcntm_main_state_{corpus}.pt"
        if not state.exists():
            alt = EXTERNAL_STATE_DIR / f"fcntm_main_state_{corpus}.pt"
            if alt.exists():
                state = alt''',
    1)])

patch("public_data.py", [(
    "agnews source",
    '''AGNEWS_SOURCE = Path(
    os.environ.get(
        "AGNEWS_SOURCE",
        r"C:/Users/Administrator/Downloads/agnews_repo/data/ag_news_csv",
    )
)''',
    '''AGNEWS_SOURCE = Path(
    os.environ.get(
        "AGNEWS_SOURCE",
        str(Path(__file__).resolve().parent / "agnews_repo" / "data"
            / "ag_news_csv"),
    )
)''',
    1)])

patch("public_data_civil.py", [(
    "civil raw candidates",
    '''RAW_CANDIDATES = [
    CIVIL_DIR / "raw.csv",
    Path(r"C:\\Users\\Administrator\\Downloads\\civil_corpus\\civil_openalex_cc.csv"),
    Path(r"C:\\Users\\Administrator\\Downloads\\civil_corpus\\civil_openalex_raw.csv"),
]''',
    '''RAW_CANDIDATES = [
    CIVIL_DIR / "raw.csv",
    Path(__file__).resolve().parent / "civil_corpus" / "civil_openalex_cc.csv",
    Path(__file__).resolve().parent / "civil_corpus" / "civil_openalex_raw.csv",
    # fallbacks for a checkout that still keeps the collector next to the code
    Path(r"C:\\Users\\Administrator\\Downloads\\civil_corpus\\civil_openalex_cc.csv"),
    Path(r"C:\\Users\\Administrator\\Downloads\\civil_corpus\\civil_openalex_raw.csv"),
]''',
    1)])

patch("README.md", [(
    "10seed paths",
    '''# 新增 7 个种子（53/71/89/101/113/127/139）；检查点写在代码目录外以保持 <512 MB
python run_fcntm_train.py --corpus github --seeds 53,71,89,101,113,127,139 --k 30 \\
    --fusion gate --fusion-norm balanced --fusion-weight 0.5 --field-dropout 0.5 \\
    --max-epochs 700 --lr 5e-4 --recover-nan --resume-every 50 \\
    --out ../FCNTM_10seed_checkpoints/fcntm_github_seed53plus.csv \\
    --dump-details ../FCNTM_10seed_checkpoints/fcntm_github_seed53plus_details.json \\
    --dump-state ../FCNTM_10seed_checkpoints/fcntm_main_state_github.pt
# agnews 同理；随后补 BTM / FC-NTM 遮挡 / LDA 遮挡并汇总
python extend_seeds_btm.py --corpora github,agnews --seeds 53,71,89,101,113,127,139
python extend_seeds_fcntm_occlusion.py --state-dir ../FCNTM_10seed_checkpoints
python extend_seeds_lda_occlusion.py''',
    '''# 新增 7 个种子（53/71/89/101/113/127/139），产物都在 outputs/rerun 下
python run_fcntm_train.py --corpus github --seeds 53,71,89,101,113,127,139 --k 30 \\
    --fusion gate --fusion-norm balanced --fusion-weight 0.5 --field-dropout 0.5 \\
    --max-epochs 700 --lr 5e-4 --recover-nan --resume-every 50 \\
    --out outputs/rerun/fcntm_github_seed53plus.csv \\
    --dump-details outputs/rerun/fcntm_github_seed53plus_details.json \\
    --dump-state outputs/rerun/fcntm_main_state_github.pt
# agnews 同理；随后补 BTM / FC-NTM 遮挡 / LDA 遮挡并汇总
python extend_seeds_btm.py --corpora github,agnews --seeds 53,71,89,101,113,127,139
python extend_seeds_fcntm_occlusion.py        # 默认读取 outputs/rerun 下的状态
python extend_seeds_lda_occlusion.py''',
    1), (
    "identity paths",
    '''`outputs/*.pt`（3 种子）与 `../FCNTM_10seed_checkpoints/*.pt`（新增 7 种子）''',
    '''`outputs/*.pt`（3 种子）与 `outputs/rerun/fcntm_main_state_*.pt`（新增 7 种子）''',
    1)])
sys.exit(1 if fails else 0)