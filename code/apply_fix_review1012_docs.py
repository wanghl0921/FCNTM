# -*- coding: utf-8 -*-
"""2026-10-12 review: update REVISION_NOTES.md and 进度记录.md."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
fails = []


def patch(name, pairs, append=None):
    path = ROOT / name
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    for tag, old, new, count in pairs:
        n = text.count(old)
        if n != count:
            fails.append(f"{name}: {tag}: found {n}, want {count}")
            continue
        text = text.replace(old, new)
    if append:
        text = text.rstrip("\n") + "\n" + append
    if not any(f.startswith(name) for f in fails):
        out = text.replace("\n", "\r\n") if crlf else text
        path.write_bytes(out.encode("utf-8"))
        print(f"  patched {name}")


SECTION = """
## 8. 2026-10-12 新 review 修复（已完成）

- **消融入口（C1，P1）**：删除 `run_ablation_public.py` 中重复的局部 `import result_identity`
  （它使 `main()` 在任何参数组合下都抛 `UnboundLocalError`）；回归测试新增
  `test_ablation_entry_smoke`，用 6 文档桩数据执行 `main()` 的无 details / 合法 details /
  被拒 details 三条分支；`reusable_full_record()` 增加外层记录与拟合身份 payload 的 seed
  一致性检查（C2 的补充项）。
- **评测身份（C4，P2）**：`seed_identity.py` 的 `corpus_identity()` 现在覆盖测试集两个字段的
  内容摘要、遮挡率列表与掩码算法版本，`occlusion_identity()` 另加检查点内容哈希；
  `verify_checkpoint()` 在评分前核对检查点内每种子的 K、词表顺序、数据与字段身份。
  `result_identity.CODE_FILES` 与清单 CODE 纳入 `seed_identity.py`、`baseline_fieldview.py`
  与三个 `extend_seeds_*` 入口；`summarize_seed_extension.py` 在统计前核对四个新种子产物
  （`fair_readout_*_newseeds.csv`、`occlusion_robustness_*_newseeds.csv`）的边车，
  缺失或不一致直接报错而不是打印警告。
- **归档重绑定（C4 的交付侧）**：新增 `verify_newseed_archives.py`，从交付的检查点与归档
  拟合重算锚点种子 53 的全部模型行再逐行比对：GitHub 遮挡 15 行 max|Δ| = 9.7e-17、
  AG News 遮挡 15 行 9.0e-17，读出 2+2 行分别 8.3e-17 与 5.6e-17（均为机器精度），
  随后把四个 CSV 重新绑定到当前评测身份，旧文件备份为
  `outputs/rerun/backup_*_pre_pin.csv`；汇总器在新身份下重跑，数字与论文一致。
- **字段视角来源（C4）**：`baseline_fieldview.py` 的每一行新增 `eval_id`（语料评测协议身份）
  与 `fit_id`（拟合身份），去重键改为 `(dataset, model, seed, view, K, eval_id)`，被替换的
  行先写备份；没有身份的历史拟合在显式 `--adopt-legacy-fits` 下才被采用，并以
  `legacy:<矩阵内容摘要>` 记录来源，否则重新拟合。归档 CSV 本身未重算，仍由唯一清单固定。
- **清单（C5，P2）**：拆开被注释吞掉的 `btm_fit_github_seed11.npz`（此前实际只追踪 19 个
  BTM 拟合），CODE 增加 `seed_identity.py` 与 `verify_newseed_archives.py`；清单重建后为
  **176 个产物**，`--verify` = drift 0 / missing 0 / new 0。
- **论文（P2）**：消融小结不再写"GitHub 上可省"，改为"当前三种子未观测到稳定的额外质量
  收益"；局限七区分读出（FC-NTM/BTM）与遮挡（另含 LDA）的覆盖；局限九分开陈述"3 种子
  on/off 曲线已齐"与"新增 7 种子只扩展置零设置"；`tab:seed-paired` 的行名与表注改为预先
  选定的 BTM/LDA；原始参数 $\\beta$ 读出那一句写明主语（FC-NTM 的 $\\beta$ 对 BTM 的
  原生 $\\beta$）。
- **表映射**：README 的旧表号（表 1--11、图 2/3/5）全部换成 LaTeX 标签，
  `check_paper_numbers.py` 的进度标题同步改为标签，避免复现者按旧号查错实验。
- **验证**：`check_paper_numbers.py` 777 项印刷数字全过；`tests_review_fixes.py` 25 项全过；
  `results_manifest.py --verify` drift 0 / missing 0 / new 0（176 个产物，含论文与四张图
  PDF）；`main.pdf` 35 页、0 error / 0 undefined / 0 overfull。
"""

patch("REVISION_NOTES.md", [
    ("external checkpoint path",
     "- **交付（P1）**：在代码目录外 `../FCNTM_10seed_checkpoints/` 重训 7 个新种子的 FC-NTM",
     "- **交付（P1）**：重训 7 个新种子的 FC-NTM", 1),
    ("external distribution",
     "  检查点（11:34 启动，github→agnews，预计 ~14 h），补齐 review 指出的缺失；清单已加入该\n"
     "  外置检查点、20 个 BTM 拟合、`tab:occlusion` 的 civil 来源，以及 `main.pdf` 与四张图 PDF 的\n"
     "  发布校验（重训完成后重建清单）。",
     "  检查点（11:34 启动，github→agnews，预计 ~14 h），补齐 review 指出的缺失；清单已加入\n"
     "  `outputs/rerun/fcntm_main_state_*.pt`、20 个 BTM 拟合、`tab:occlusion` 的 civil 来源，\n"
     "  以及 `main.pdf` 与四张图 PDF 的发布校验。", 1),
    ("checkpoint delivered path",
     "- 新增 7 种子的 FC-NTM 检查点已在 `../FCNTM_10seed_checkpoints/` 交付",
     "- 新增 7 种子的 FC-NTM 检查点已在 `outputs/rerun/` 交付", 1),
    ("manifest count",
     "- 清单重建：**169 个产物**，`--verify` = drift 0 / missing 0 / new 0；其中包含外置检查点、\n"
     "  20 个 BTM 拟合、6 个 NMF 拟合、`tab:occlusion` 的 civil 来源，以及 `main.pdf` 与四张图\n"
     "  PDF 的发布校验。",
     "- 清单重建：**176 个产物**（2026-10-12 review 后再次重建），`--verify` = drift 0 /\n"
     "  missing 0 / new 0；其中包含 `outputs/rerun/fcntm_main_state_*.pt`、20 个 BTM 拟合、\n"
     "  6 个 NMF 拟合、`tab:occlusion` 的 civil 来源，以及 `main.pdf` 与四张图 PDF 的发布校验。", 1),
    ("tests count",
     "- `check_paper_numbers.py` 777 项全过；`tests_review_fixes.py` 23 项全过。",
     "- `check_paper_numbers.py` 777 项全过；`tests_review_fixes.py` 当时 23 项全过\n"
     "  （2026-10-12 增至 25 项）。", 1),
], append=SECTION)

patch("进度记录.md", [
    ("external checkpoint path",
     "- **交付（P1）**：在代码目录外 `../FCNTM_10seed_checkpoints/` 重训 7 个新种子的 FC-NTM",
     "- **交付（P1）**：重训 7 个新种子的 FC-NTM", 1),
    ("external distribution",
     "  检查点（11:34 启动，github→agnews，预计 ~14 h），补齐 review 指出的缺失；清单已加入该\n"
     "  外置检查点、20 个 BTM 拟合、`tab:occlusion` 的 civil 来源，以及 `main.pdf` 与四张图 PDF 的\n"
     "  发布校验（重训完成后重建清单）。",
     "  检查点（11:34 启动，github→agnews，预计 ~14 h），补齐 review 指出的缺失；清单已加入\n"
     "  `outputs/rerun/fcntm_main_state_*.pt`、20 个 BTM 拟合、`tab:occlusion` 的 civil 来源，\n"
     "  以及 `main.pdf` 与四张图 PDF 的发布校验。", 1),
    ("checkpoint delivered path",
     "- 新增 7 种子的 FC-NTM 检查点已在 `../FCNTM_10seed_checkpoints/` 交付",
     "- 新增 7 种子的 FC-NTM 检查点已在 `outputs/rerun/` 交付", 1),
    ("manifest count",
     "- 清单重建：**169 个产物**，`--verify` = drift 0 / missing 0 / new 0；其中包含外置检查点、\n"
     "  20 个 BTM 拟合、6 个 NMF 拟合、`tab:occlusion` 的 civil 来源，以及 `main.pdf` 与四张图\n"
     "  PDF 的发布校验。",
     "- 清单重建：**176 个产物**（2026-10-12 review 后再次重建，见 §8），`--verify` =\n"
     "  drift 0 / missing 0 / new 0；其中包含 `outputs/rerun/fcntm_main_state_*.pt`、20 个 BTM\n"
     "  拟合、6 个 NMF 拟合、`tab:occlusion` 的 civil 来源，以及 `main.pdf` 与四张图 PDF 的发布校验。", 1),
], append=SECTION)

import sys
if fails:
    print("unmatched anchors:")
    for f in fails:
        print("  ", f)
sys.exit(1 if fails else 0)
