# -*- coding: utf-8 -*-
"""Append the field-view regeneration outcome to both revision documents."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BLOCK = """
### 8.1 字段视角归档表的重算（2026-10-12 同日追加）

- `outputs/baseline_fieldview.csv` 已用当前代码重算并替换：三段共约 1.9 小时（github+agnews
  的 LDA/NMF/BTM、三语料 ProdLDA、civil 的 LDA/NMF/BTM）；civil 的 BTM/NMF 是唯一需要
  重新拟合的部分，其缓存已按当前拟合身份落盘，以后可直接复用。
- 重算与归档表**逐位一致**：72/72 行键匹配，`n_topics`/`C_V`/`C_V_n`/`NPMI`/`NPMI_n`
  的最大差异为 0.0；旧表备份为 `outputs/backup_baseline_fieldview_pre_identity.csv`
  （4059 字节），逐行比对报告见 `outputs/rerun/fieldview_regen_report.json`。
- 新表带 `eval_id`（每语料评测协议身份）与 `fit_id`（36 个拟合身份，其中 21 个为
  `legacy:<矩阵摘要>`：采用的历史拟合矩阵）；重算入口 `repin_fieldview_archive.py`
  只有在逐位一致时才替换归档，收尾脚本 `finish_fieldview_regen.py` 负责重建清单与复验。
- 同时修正边车的描述口径：由多段调用合并成的表，边车改为从合并结果推导语料、模型与
  `eval_id`（原先只记录最后一次调用，边车与文件内容不符）；因此 4 个新种子评测边车按新
  代码摘要重新绑定（锚点种子 53 复核：GitHub 遮挡 9.7e-17、AG News 遮挡 9.0e-17，
  读出 8.3e-17 / 5.6e-17）。
- 验证更新：清单 **178 个产物**（新增 `repin_fieldview_archive.py` 与
  `finish_fieldview_regen.py` 的代码追踪），`--verify` = drift 0 / missing 0 / new 0；
  `check_paper_numbers.py` 777 项、`tests_review_fixes.py` 25 项全过；
  `summarize_seed_extension.py` 在重绑后的身份下重跑，数字与论文一致。
"""

for name in ("REVISION_NOTES.md", "进度记录.md"):
    path = ROOT / name
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n").rstrip("\n") + "\n" + BLOCK
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    print(f"  appended {name}")
