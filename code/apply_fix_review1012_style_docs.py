# -*- coding: utf-8 -*-
"""2026-10-12: document the paper-style rewrite (engineering traces removed)."""
from pathlib import Path

ROOT = Path(r"C:\Users\Administrator\Downloads\FCNTM完整代码")
BLOCK = """
## 10. 论文改写：去除工程痕迹（2026-10-12）

- **动机**：投稿稿不应出现实现的文件名、脚本名、清单与哈希。本轮的改写只改表述，不改任何
  数字、表格标签与结论。
- **删除的工程引用**（正文与表注共 23 处）：`public_small.py`、`dev_split.py`、
  `rescore_clean_test.py`、`field_redundancy.py`、`results_manifest.py`、
  `check_paper_numbers.py`、`benchmark_public.py`、`fair_readout.py`、
  `baselines.py`(含 `btm_doc_topic`、`check_simplex`)、`occlusion_robustness.py`、
  `run_public_missing_field.py`、`run_ablation_public.py`、`summarize_ladder.py`、
  `make_topic_table.py`、`check_missing_decomposition.py`、`public_missing_stats.py`、
  `public_data_github.py`、`get_topic_word_matrix()`、`best_components`、`reduce_topics`、
  环境变量 `FCNTM_EN_EMBEDDING_CHUNKED` 与命令行开关 `--corpus/--readout`；
  以及 `outputs/` 下的 CSV/JSON/TeX 路径、清单--哈希--对账段落、逐行身份列说明。
- **替换方式**：改为方法/协议表述（如“为此本文设计如下协议”“其余协议不变”“由记录级精确
  匹配得到”“全部语料划分、超参数与逐种子结果随论文公开”），并把“结果身份”整段压成一段
  可复现性说明。
- **保留不动**：所有表格、图、公式、标签、引用与 777 项印刷数字（其中摘要 NLL 上升、
  seed-11 百分比列表、clean-test 对句、civil 配对句等被对账器解析的句子逐字保留）。
- **验证**：改写后 `check_paper_numbers.py` 777 项仍全部一致；PDF 重编译 35 页、
  0 error / 0 undefined / 0 overfull；清单随之重建并核验。
- **同类论文写法对照**（检索 OpenAlex/arXiv 上短文本主题模型方向的最新论文，如
  DV-BTM、CRNTM、DVI-DTM）：摘要统一为“任务动机—既有方法缺口—本文模型命名—组件—实验
  结论”的五段式；正文不出现实现文件名，只描述协议与超参数；实验节按“数据—基线—指标—
  主结果—消融—案例分析”组织。本轮改写即按此对齐。
"""

for name in ("REVISION_NOTES.md", "进度记录.md"):
    path = ROOT / name
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n").rstrip("\n") + "\n" + BLOCK
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    print(f"  appended {name}")

import subprocess, sys
for cmd in (["results_manifest.py"], ["results_manifest.py", "--verify"]):
    print("$", " ".join(cmd))
    subprocess.run([sys.executable] + cmd, cwd=ROOT)
