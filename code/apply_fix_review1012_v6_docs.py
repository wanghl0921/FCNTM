# -*- coding: utf-8 -*-
"""2026-10-12 v6 review: document the --state-dir identity fix."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BLOCK = """
## 9. 2026-10-12 v6 review 修复（已完成）

- **`--state-dir` 的记录来源（P2，非默认路径）**：`seed_identity.occlusion_identity()` 增加
  `checkpoint` 参数，`state_digest` 改为按**实际被评分的状态文件**计算；
  `extend_seeds_fcntm_occlusion.py` 传入解析后的 `state`，`verify_newseed_archives.py`
  显式传入默认状态。此前用 `--state-dir` 评分时会记录默认检查点的摘要，两个来源可能被
  合并进同一张表。
- **汇总器可显式选择来源**：`summarize_seed_extension.py --state-dir <dir>` 用同一来源计算
  期望身份；来源不同则身份不匹配、汇总被拒绝（默认路径下两种写法得到相同身份；本轮实测
  过代码摘要变化时四个文件均报 `identity differs in: code` 并拒绝汇总）。
- **新增入口测试**：`test_state_dir_identity_uses_the_actual_checkpoint` 在临时目录同时创建
  内容不同的 default / alternate 状态文件，用桩替换模型推断后执行
  `extend_seeds_fcntm_occlusion.main()`，断言检查与评分都用 alternate 文件，且写出的身份中
  `state_digest` 等于 alternate 的摘要、不等于 default 的摘要。回归测试 25 → **26 项**。
- **重绑定与复验**：4 个新种子评测 CSV 按新代码摘要重新绑定（锚点种子 53 复核：GitHub
  遮挡 9.7e-17、AG News 遮挡 9.0e-17，读出 8.3e-17 / 5.6e-17）；
  `summarize_seed_extension.py` 重跑数字与论文一致；清单 **178 个产物** drift 0 /
  missing 0 / new 0；`check_paper_numbers.py` 777 项、`tests_review_fixes.py` 26 项全过。
- **论文按 v6 结论保持不动**：语料、读出与机制边界不变；独立确认语料与单编码器/拼接对照
  仍作为将来的实验，不属于本轮修复。下一步是按目标期刊整理语言、摘要、篇幅与附录。
"""

for name in ("REVISION_NOTES.md", "进度记录.md"):
    path = ROOT / name
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n").rstrip("\n") + "\n" + BLOCK
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    print(f"  appended {name}")

path = ROOT / "README.md"
raw = path.read_bytes().decode("utf-8")
crlf = "\r\n" in raw
text = raw.replace("\r\n", "\n")
old = "`summarize_seed_extension.py` 直接报错，而不是打印警告继续。"
new = ("`summarize_seed_extension.py` 直接报错，而不是打印警告继续。若遮挡曲线是从别的检查点\n"
       "目录评分的，用 `summarize_seed_extension.py --state-dir <dir>` 指定同一来源：汇总器按该\n"
       "目录下的状态文件计算期望身份，来源不同就拒绝汇总。")
assert text.count(old) == 1
text = text.replace(old, new)
path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
print("  patched README.md")
