from pathlib import Path
import subprocess, sys
ROOT = Path(r"C:\Users\Administrator\Downloads\FCNTM完整代码")
BLOCK = """
## 12. 土木定位与正文/补充材料拆分（2026-10-12）

- **土木领域定位**：新增 10 篇英文文献（JCCE 与 Automation in Construction 上的文本挖掘、
  主题模型先例，如 DUET 的 LDA、建筑领域神经主题模型），新增相关工作小节「工程文档的文本挖掘
  与主题建模」，并在摘要、引言与讨论中把「工程文献标题—摘要」写成应用背景。
- **正文/补充材料拆分**：附录整体抽成独立文件 `supplementary.tex` / `supplementary.pdf`
  （长文档控制、长度比阶梯、遮挡对照、$\\beta$ 敏感性、冗余度表与图、主题样例，以及数据说明
  与指标推导），正文只保留文字指针；`check_paper_numbers.py` 改为同时解析两份文件。
- **规模**：正文 33 → **28 页**，补充材料独立（约 5 页）；`check_paper_numbers.py` 764 项
  全部一致，两本 PDF 均 0 error / 0 undefined。
- **下一步**：英文重写（按 ASCE JCCE 或 EAAI 模板，目标 ≈9,000--10,000 英文词），逐种子表格
  随补充材料提交。
"""
for name in ("REVISION_NOTES.md", "进度记录.md"):
    p = ROOT / name
    raw = p.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n").rstrip("\n") + "\n" + BLOCK
    p.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    print("appended", name)
for cmd in (["results_manifest.py"], ["results_manifest.py", "--verify"]):
    subprocess.run([sys.executable] + cmd, cwd=ROOT)
