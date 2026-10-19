# -*- coding: utf-8 -*-
"""2026-10-12 review, paper P2: sync the residual wording and the table map.

main.tex:  the ablation summary must not read "optional on GitHub"; the two
           limitations must separate read-out/occlusion coverage and the
           on/off vs new-seed pairing; the seed-paired rows are named after
           the pre-selected baselines; the beta comparison names its subject.
README.md: table numbers are replaced by LaTeX labels (the printed numbers
           moved when the ladder/chunked/seed tables were added).
check_paper_numbers.py: the same stale numbers in the progress titles.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent / "FCNTM完整论文"
fails = []


def patch(path, pairs):
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    for tag, old, new, count in pairs:
        n = text.count(old)
        if n != count:
            fails.append(f"{path.name}: {tag}: found {n}, want {count}")
            continue
        text = text.replace(old, new)
    if not any(f.startswith(path.name) for f in fails):
        out = text.replace("\n", "\r\n") if crlf else text
        path.write_bytes(out.encode("utf-8"))
        print(f"  patched {path.name}")


patch(PAPER / "main.tex", [
    ("ablation summary",
     "\\emph{跨字段重构与词掩码}在 GitHub 上可省、在 AG News 与土木文献上承重；",
     "\\emph{跨字段重构与词掩码}在 GitHub 上当前三种子未观测到稳定的额外质量收益、"
     "在 AG News 与土木文献上承重；",
     1),
    ("limitation 7",
     "节）；其余基线仍为 3 种子，土木文献为 3 种子，因此只有 \\fcntm{}--BTM 的读出与遮挡对比\n"
     "做了多种子推断。",
     "节）；其余基线仍为 3 种子，土木文献为 3 种子，因此多种子推断的读出对比覆盖 \\fcntm{} 与\n"
     "BTM，遮挡对比在此基础上另覆盖 LDA（AG News）。",
     1),
    ("limitation 9",
     "固定使用置零训练（fielddrop\\_on）的检查点，因此它在三个语料上各覆盖 11/23/37 三个种子，\n"
     "三语料的置零/不置零曲线都已补齐（表 \\ref{tab:fielddrop-control}），两套设置的相对退化差异\n"
     "很小；但字段视角与部分遮挡曲线的配对仍只覆盖置零训练的检查点，扩展到更多训练设置是后续\n"
     "工作。",
     "固定使用置零训练（fielddrop\\_on）的检查点：三个语料上各覆盖 11/23/37 三个种子，置零与\n"
     "不置零两套曲线都已补齐（表 \\ref{tab:fielddrop-control}），两套设置的相对退化差异很小；但\n"
     "新增的 7 个种子只扩展了置零设置，因此字段视角与部分遮挡曲线的多种子配对仍只覆盖置零训练\n"
     "的检查点，扩展到更多训练设置是后续工作。",
     1),
    ("seed section intro",
     "另按\"同种子最好的基线\"配对。读出对比只覆盖 \\fcntm{} 与 BTM；遮挡对比另加入 LDA",
     "另按同种子配对，遮挡行使用预先选定的对照基线（AG News 另含其 3 种子最好基线 LDA）。\n"
     "读出对比只覆盖 \\fcntm{} 与 BTM；遮挡对比另加入 LDA",
     1),
    ("seed table github row",
     "GitHub & 部分遮挡（对最好基线 BTM） &",
     "GitHub & 部分遮挡（对 BTM） &",
     1),
    ("seed table agnews lda row",
     "AG News & 部分遮挡（对最好基线 LDA） &",
     "AG News & 部分遮挡（对 LDA） &",
     1),
    ("seed table caption",
     "更稳定。CI 为 95\\% $t$ 区间，符号检验与 Wilcoxon 均为双侧。}",
     "更稳定。遮挡行的对照是预先选定的基线（GitHub 为 BTM，AG News 为 BTM 与 LDA），\n"
     "不是对所有基线的完整竞赛。CI 为 95\\% $t$ 区间，符号检验与 Wilcoxon 均为双侧。}",
     1),
    ("limitation 8 beta subject",
     "与 BTM 的原生 $\\beta$ 读出比较时 \\fcntm{} 在两个语料上都更低（$-$0.014 与 $-$0.027）。",
     "\\fcntm{} 的 $\\beta$ 读出（模型原始参数矩阵，与类 TF-IDF 读出相对）与 BTM 的原生\n"
     "$\\beta$ 读出相比，在两个语料上都更低（$-$0.014 与 $-$0.027）。",
     1),
])

patch(ROOT / "README.md", [
    ("tree core ablation",
     "├── run_core_ablations.py         # 最终协议下的核心组件消融（PDF 表 8，可断点续跑）",
     "├── run_core_ablations.py         # 最终协议下的核心组件消融（\\ref{tab:core-ablation}，可断点续跑）",
     1),
    ("tree identity entry",
     "├── seed_identity.py              # 种子扩展脚本共用的拟合身份与合并校验",
     "├── seed_identity.py              # 种子扩展脚本共用的拟合身份与评测身份\n"
     "├── verify_newseed_archives.py    # 重算新种子锚点行并核对 / 绑定评测身份边车",
     1),
    ("section 1",
     "### 1. FC-NTM 两行（表 1、表 2）",
     "### 1. FC-NTM 两行（表 \\ref{tab:agnews}、\\ref{tab:github}）",
     1),
    ("section 2",
     "### 2. 基线（表 1、表 2）",
     "### 2. 基线（表 \\ref{tab:agnews}、\\ref{tab:github}）",
     1),
    ("section 3",
     "### 3. 遮挡互补字段（表 5）",
     "### 3. 遮挡互补字段（表 \\ref{tab:occlusion}、\\ref{tab:fieldview}）",
     1),
    ("section 4",
     "### 4. 互补性与增益（表 6、图 3）",
     "### 4. 互补性与增益（表 \\ref{tab:conditional}、图 \\ref{fig:cond}）",
     1),
    ("section 5",
     "### 5. 核心组件消融（表 8）",
     "### 5. 核心组件消融（表 \\ref{tab:core-ablation}）",
     1),
    ("baseline command",
     "python baseline_fieldview.py --models LDA,NMF,BTM,ProdLDA --corpora github,agnews,civil\n```",
     "python baseline_fieldview.py --models LDA,NMF,BTM,ProdLDA --corpora github,agnews,civil\n```\n\n"
     "归档包里的 `*_fit_*.npz` 产生于身份机制之前：复现归档数字时加 `--adopt-legacy-fits`，\n"
     "脚本会把每个矩阵的内容摘要写进该行的 `fit_id` 列（`legacy:<digest>`），不加该开关则\n"
     "重新拟合。每行的 `eval_id` 是该语料评测协议的身份，去重键为\n"
     "`(dataset, model, seed, view, K, eval_id)`。",
     1),
    ("seed extension note",
     "`summarize_seed_extension.py` 在计算前校验每个语料的十个种子、两种读出、五个遮挡率与\n"
     "有限指标；缓存与合并 CSV 都带完整拟合身份（`seed_identity.py`）。",
     "`summarize_seed_extension.py` 在计算前校验每个语料的十个种子、两种读出、五个遮挡率与\n"
     "有限指标，并核对四个新种子产物（`fair_readout_*_newseeds.csv`、\n"
     "`occlusion_robustness_*_newseeds.csv`）的身份边车：拟合身份覆盖语料、词表与代码版本，\n"
     "评测身份另外覆盖测试集两个字段的内容、遮挡率与掩码算法版本、以及 FC-NTM 检查点的内容\n"
     "哈希；边车缺失或与当前协议不一致时拒绝汇总（`seed_identity.py`）。归档行的绑定过程见\n"
     "`verify_newseed_archives.py`（重算锚点种子并逐行核对后再写边车）。",
     1),
    ("artifact github main",
     "| `outputs/fcntm_github_final.csv` | GitHub 主表（PDF 表 2）的 FC-NTM 两行，逐种子 |",
     "| `outputs/fcntm_github_final.csv` | GitHub 主表（表 \\ref{tab:github}）的 FC-NTM 两行，逐种子 |",
     1),
    ("artifact agnews main",
     "| `outputs/fcntm_agnews_final.csv` | AG News 主表（PDF 表 1）同上 |",
     "| `outputs/fcntm_agnews_final.csv` | AG News 主表（表 \\ref{tab:agnews}）同上 |",
     1),
    ("artifact fair",
     "| `outputs/fair_readout_{github,agnews}_3seed.csv` | 读出对照表（PDF 表 3） |",
     "| `outputs/fair_readout_{github,agnews}_3seed.csv` | 读出对照表（表 \\ref{tab:fair}） |",
     1),
    ("artifact occlusion",
     "| `outputs/public_missing_field_occlusion.csv` | 整字段遮挡（PDF 表 10） |",
     "| `outputs/public_missing_field_occlusion.csv` | 整字段遮挡（表 \\ref{tab:occlusion}） |",
     1),
    ("artifact fieldview",
     "| `outputs/public_missing_field_fieldview.csv` | 字段受限读出（PDF 表 6） |",
     "| `outputs/public_missing_field_fieldview.csv` | 字段受限读出（表 \\ref{tab:fieldview}） |",
     1),
    ("artifact partial occlusion",
     "| `outputs/occlusion_robustness_*_seed*.csv` | 部分遮挡曲线（PDF 表 4、图 2） |",
     "| `outputs/occlusion_robustness_*_seed*.csv` | 部分遮挡曲线（表 \\ref{tab:occl-robust}、图 \\ref{fig:occlusion}） |",
     1),
    ("artifact cross occlusion",
     "| `outputs/occlusion_comparison.csv` | 跨语料遮挡汇总（PDF 表 5、图 3） |",
     "| `outputs/occlusion_comparison.csv` | 跨语料遮挡汇总（表 \\ref{tab:occl-cross}、图 \\ref{fig:occl-cross}） |",
     1),
    ("artifact fusion",
     "| `outputs/ablation_public_github_{nofusion,fusedmean,*_w*}.csv` | 融合消融与权重扫描（PDF 表 7、表 9） |",
     "| `outputs/ablation_public_github_{nofusion,fusedmean,*_w*}.csv` | 融合消融与权重扫描（表 \\ref{tab:fusion}、\\ref{tab:weight}） |",
     1),
    ("artifact core",
     "| `outputs/ablation_public_{github,agnews}_core.csv` | 核心组件消融，两个语料（PDF 表 8） |",
     "| `outputs/ablation_public_{github,agnews}_core.csv` | 核心组件消融，两个语料（表 \\ref{tab:core-ablation}） |",
     1),
    ("artifact redundancy",
     "| `outputs/field_redundancy.json` / `field_complementarity.csv` | 冗余度度量（PDF 表 11 的左两列、图 5） |",
     "| `outputs/field_redundancy.json` / `field_complementarity.csv` | 冗余度度量（表 \\ref{tab:conditional} 的左两列、图 \\ref{fig:cond}） |",
     1),
    ("legacy sidecars",
     "仍有少量 2026-10 之前的历史拟合缓存没有边车",
     "仍有少量 2026-10 之前的历史拟合缓存没有边车（`outputs/rerun/*_fit_*.npz`）：入口不会\n"
     "静默复用它们，而是**重新拟合**；`baseline_fieldview.py` 只有在显式给出\n"
     "`--adopt-legacy-fits` 时才改用它们，并把该矩阵的内容摘要（`legacy:<digest>`）写进\n"
     "每一行的 `fit_id` 列。10 种子扩展的评测 CSV 已按同一协议重算锚点行、核对后绑定边车，\n"
     "缺失或不匹配的边车会让 `summarize_seed_extension.py` 直接报错。",
     0),
    ("paper tables wording",
     "- 论文表 1--9 所用的结果文件大多产生于该机制之前，**本身没有这些列**，它们由唯一清单固定；",
     "- 论文各表所用的结果文件大多产生于该机制之前，**本身没有这些列**，它们由唯一清单固定；",
     1),
    ("identity entries",
     "  以及 `benchmark_public.py` / `run_fcntm_train.py` 逐行写出的 `actual_readout`、`state_hash`、\n"
     "  `code_hash`。",
     "  以及 `benchmark_public.py` / `run_fcntm_train.py` 逐行写出的 `actual_readout`、`state_hash`、\n"
     "  `code_hash`。\n"
     "- 种子扩展的三个入口（`extend_seeds_*.py`）与 `baseline_fieldview.py` 写出的边车是\n"
     "  **评测身份**：除语料、词表与代码版本外，还包含测试集两个字段的内容摘要、遮挡率列表、\n"
     "  掩码算法版本与所用 FC-NTM 检查点的内容哈希（`seed_identity.py`）；修改测试集或更换\n"
     "  检查点会让身份变化，旧行不会被合并。",
     1),
])

patch(ROOT / "check_paper_numbers.py", [
    ("main results", 'print("tables 1/2 (main results)")',
     'print("tab:agnews / tab:github (main results)")', 1),
    ("fair doc", '"""Table 3: the two read-outs (beta / class TF-IDF) per model."""',
     '"""tab:fair: the two read-outs (beta / class TF-IDF) per model."""', 1),
    ("fair print", 'print("table 3 (read-out comparison)")',
     'print("tab:fair (read-out comparison)")', 1),
    ("occl doc", '"""Table 5: the GitHub seed-11 partial-occlusion curve."""',
     '"""tab:occl-robust: the GitHub seed-11 partial-occlusion curve."""', 1),
    ("occl print", 'print("table 5 (partial occlusion, seed 11)")',
     'print("tab:occl-robust (partial occlusion, seed 11)")', 1),
    ("occl require", 'require_rows("table 5 (occlusion)",',
     'require_rows("tab:occl-robust (occlusion)",', 1),
    ("occl cells", 'FAILURES.append(f"table 5 row has {len(cells)} cells: {line!r}")',
     'FAILURES.append(f"tab:occl-robust row has {len(cells)} cells: {line!r}")', 1),
    ("occl malformed", 'print(f"  MISSING  table 5 row malformed: {line[:60]!r}")',
     'print(f"  MISSING  tab:occl-robust row malformed: {line[:60]!r}")', 1),
    ("cross doc", '"""Table 6: relative degradation and agreement over the three seeds."""',
     '"""tab:occl-cross: relative degradation and agreement over the three seeds."""', 1),
    ("cross print", 'print("table 6 (cross-corpus occlusion)")',
     'print("tab:occl-cross (cross-corpus occlusion)")', 1),
    ("fieldview doc", '"""Table 4: field-restricted read-out plus the support range."""',
     '"""tab:fieldview: field-restricted read-out plus the support range."""', 1),
    ("fieldview print", 'print("table 4 (field-restricted read-out)")',
     'print("tab:fieldview (field-restricted read-out)")', 1),
    ("fieldview require", 'require_rows("table 4 (field view)",',
     'require_rows("tab:fieldview (field view)",', 1),
    ("fusion doc", '"""Table 7: the three-step fusion ablation, including the paired signs."""',
     '"""tab:fusion: the three-step fusion ablation, including the paired signs."""', 1),
    ("fusion print", 'print("table 7 (fusion ablation)")',
     'print("tab:fusion (fusion ablation)")', 1),
    ("fusion require", 'require_rows("table 7 (fusion)",',
     'require_rows("tab:fusion (fusion)",', 1),
    ("core doc", '"""Table 8: four component ablations on both corpora (paired deltas)."""',
     '"""tab:core-ablation: four component ablations on both corpora (paired deltas)."""', 1),
    ("core print", 'print("table 8 (core-mechanism ablation)")',
     'print("tab:core-ablation (core-mechanism ablation)")', 1),
    ("core require", 'require_rows("table 8 (core ablation)",',
     'require_rows("tab:core-ablation (core ablation)",', 1),
    ("weight doc", '"""Table 9: the lambda_f sweep (both read-outs)."""',
     '"""tab:weight: the lambda_f sweep (both read-outs)."""', 1),
    ("weight print", 'print("table 9 (fusion weight sweep)")',
     'print("tab:weight (fusion weight sweep)")', 1),
    ("weight require", 'require_rows("table 9 (weight sweep)",',
     'require_rows("tab:weight (weight sweep)",', 1),
    ("whole doc", '"""Table 10: whole-field occlusion for both corpora and both settings."""',
     '"""tab:occlusion: whole-field occlusion for both corpora and both settings."""', 1),
    ("whole print", 'print("table 10 (whole-field occlusion)")',
     'print("tab:occlusion (whole-field occlusion)")', 1),
    ("whole require", 'require_rows("table 10 (whole-field)",',
     'require_rows("tab:occlusion (whole-field)",', 1),
])

import sys
if fails:
    print('unmatched anchors:')
    for f in fails:
        print('  ', f)
sys.exit(1 if fails else 0)
