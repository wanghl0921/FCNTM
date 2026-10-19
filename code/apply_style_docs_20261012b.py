# -*- coding: utf-8 -*-
"""Document the compression pass and rebuild the manifest."""
from pathlib import Path
import subprocess, sys

ROOT = Path(r"C:\Users\Administrator\Downloads\FCNTM完整代码")
BLOCK = """
## 11. 论文压缩与打磨（2026-10-12）

- **结构**：新增附录 A「附加实验与对照」，把长文档截断控制（`tab:chunked`）、长度比阶梯
  （`tab:ladder`）、遮挡协议对照（`tab:occl-controls`、`tab:fielddrop-control`）与
  $\\beta$ 读出敏感性（`tab:beta-readout`）移入附录；正文各留一句指针。
- **文字压缩**：逐节精简主结果（GitHub/AG News）、读出公平性、多种子配对、部分遮挡、跨语料
  对比、字段视角、融合消融、核心组件消融、缺失推断与冗余度，以及讨论与局限（九条合并为
  七条）；删除正文内嵌的 clean-test 小表（数字保留在对账器解析的那句话里）。
- **规模**：PDF 从 35 页降到 31 页（其中附录约 3 页），全部表格、图、数字与结论不变。
- **对账**：压缩过程中对账器曾抓到两处问题（一张内嵌表被误删导致 clean-test 表头缺失、
  字段视角两张表被整段替换带走），均已恢复；`check_paper_numbers.py` 777 项与
  `tests_review_fixes.py` 26 项全部通过。
- **下一步**：若要继续压到 22--25 页，可再压缩数据节与模型节、把 `tab:conditional` 与
  `tab:topics` 移入附录，并把四张 \\textwidth 图缩小一档。
"""
for name in ("REVISION_NOTES.md", "进度记录.md"):
    path = ROOT / name
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n").rstrip("\n") + "\n" + BLOCK
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    print("appended", name)
for cmd in (["results_manifest.py"], ["results_manifest.py", "--verify"]):
    subprocess.run([sys.executable] + cmd, cwd=ROOT)
