# FC-NTM：论文配套代码

本目录是论文《面向双字段短文本的字段协同神经主题模型》的实验代码。论文正文与 PDF
在 `../FCNTM完整论文/`，本目录只保留复现论文结果所需的脚本与产物。

## 模型与协议

两个模型共享同一份掩码训练结果，只读出口径不同：

- **FC-NTM**：取共享解码器 $\beta$ 每行 softmax 后概率最高的 10 个词；
- **FC-NTM-cTFIDF**：按文档—主题后验软归属聚合训练词频，做类级 TF-IDF 后取 10 个词。

训练协议（论文第 4 节）：背景字段词掩码 0.30、概要字段 0.40，重建目标始终为完整词频；
背景字段整体置零概率 0.5；KL 权重前 20 轮线性预热至上界 0.05；跨字段重构权重
$\gamma_c=2.0$、$\gamma_i=1.0$；一致性权重 0.5；正交正则权重 0.2；文档级门控融合
表示参与训练，融合重建项权重 0.5，且各流项保持与不含融合项时相同的尺度
（`--fusion-norm balanced`）；20% 留出验证集早停（patience 150）。

公开语料的 FC-NTM 使用 `--lr 5e-4 --max-epochs 700 --recover-nan`（默认学习率
$2\times10^{-3}$ 在该规模上会发散）。

## 统一读出

跨模型比较必须固定词表与主题词数量。`new_baselines.class_tfidf_readout` 把
BERTopic / Top2Vec 的文档—主题归属在同一评测词表上做与 FC-NTM-cTFIDF 相同的类级
c-TF-IDF，使其输出同样长度的 10 词主题；`benchmark_public.py --topn 10
--readout ctfidf` 为默认行为，`--readout native` 可还原各模型的原生读出。

## 目录

```text
FCNTM完整代码/
├── models.py                     # FC-NTM 与各神经基线模型
├── config.py / metrics.py        # 超参与一致性、多样性、排他性指标
├── corpus.py                     # 私有语料加载（不随包分发，见下）
├── public_data.py                # AG News 划分与词典
├── public_data_github.py         # GitHub issues 划分与词典
├── embeddings.py / embed_public.py
├── baselines.py                  # LDA / BTM / NMF
├── new_baselines.py              # BERTopic / Top2Vec / CombinedTM / NeuralLDA
├── experiment.py                 # 各神经模型的训练封装
├── run_fcntm_train.py            # 主入口：一次训练输出两种读出
├── benchmark_public.py           # 公开基线批量运行（支持断点续跑）
├── merge_benchmark_rows.py       # 把重跑行并回主表并刷新汇总
├── compare_fusion.py             # 两组逐种子结果配对比较
├── run_public_missing_field.py   # 遮挡互补字段实验（公开语料）
├── occlusion_robustness.py       # 部分遮挡曲线（每个模型走自己的推断路径）
├── finish_agnews_occlusion.py    # 补齐遮挡链条的编排脚本（可重入）
├── watch_agnews37_then_finish.py # 等训练日志写出 saved 行后自动接力上一步
├── field_complementarity.py      # 跨字段互预测 / 冗余度度量
├── field_redundancy.py           # CCA 与 Ridge 跨视图 R²
├── public_missing_stats.py       # 自然缺失字段比例统计
├── run_small.py / public_small.py / report_small.py
│                                 # 小规模受控实验（早期 1000/400 协议；正文已不再报告）
├── run_variant_screen_public.py / summarize_variant_public.py
├── run_ablation.py / run_ablation_public.py
├── run_core_ablations.py         # 最终协议下的核心组件消融（\ref{tab:core-ablation}，可断点续跑）
├── fair_readout.py               # 每个模型两种读出（β / 类 TF-IDF）的对照
├── rescore_clean_test.py         # 剔除开发记录后的重评分敏感性分析
├── add_btm_doc_topic.py          # 给 BTM 补文档—主题后验并重跑相关表
├── dump_btm_readout_topics.py    # 保存 BTM 两种读出的主题词表（供 clean-test 复用）
├── occlusion_controls.py         # 常量 unigram / 概要错排 / fielddrop_off 遮挡对照
├── bn_readout_sensitivity.py     # raw β 参数读出 vs BN 生成分布读出
├── baseline_fieldview.py         # LDA/BTM/NMF/ProdLDA 的字段受限读出对照
├── add_civil_btm_occlusion.py    # civil 遮挡补 BTM 行（不重算已有五模型）
├── summarize_ladder.py           # 固定文献与词表的长度比阶梯汇总
├── seed_identity.py              # 种子扩展脚本共用的拟合身份与评测身份
├── verify_newseed_archives.py    # 重算新种子锚点行并核对 / 绑定评测身份边车
├── repin_fieldview_archive.py    # 重算字段视角归档表并绑定逐行身份（一致才替换）
├── finish_fieldview_regen.py     # 等重算结束后重建清单并复验
├── extend_seeds_btm.py / extend_seeds_fcntm_occlusion.py / extend_seeds_lda_occlusion.py
├── summarize_seed_extension.py   # 10 种子配对统计（CI / 符号检验 / Wilcoxon）
├── result_identity.py            # 结果身份：配置 + 数据 + 代码哈希，决定旧结果能否复用
├── results_manifest.py           # 唯一结果清单：产物哈希 + 表/图 → 产物映射
├── check_paper_numbers.py        # 解析 main.tex 表格，与产物重新计算值逐项对账
├── check_empty_summary_behavior.py  # 空概要记录的实际推断行为（实测差异）
├── check_btm_empty_biterm_counts.py # 无词对记录的计数（无需模型）
├── rerun_btm_nmf_boundary_fix.py    # 边界修复后重跑 BTM/NMF 的读出与遮挡
├── data/                         # AG News 与 GitHub issues 固定划分
├── outputs/                      # 论文全部数字的逐种子产物与汇总
└── figures/                      # 私有语料论文时期的插图（保留备查）
```

## 复现论文表格

### 1. FC-NTM 两行（表 \ref{tab:agnews}、\ref{tab:github}）

```bash
python run_fcntm_train.py --corpus github --seeds 11,23,37 --k 30 \
    --max-epochs 700 --lr 5e-4 --recover-nan \
    --fusion gate --fusion-norm balanced --fusion-weight 0.5 \
    --dump-details outputs/github_gate_details.json \
    --out outputs/codev_github_gate_bal.csv

python run_fcntm_train.py --corpus github --seeds 11,23,37 --k 30 \
    --max-epochs 700 --lr 5e-4 --recover-nan \
    --out outputs/codev_github_base.csv          # 消融：不训练融合表示
```

AG News 同理，把 `--corpus github` 换成 `--corpus agnews`。
`--dump-details` 写出的 $\beta/\theta$ 只够重算主题词读出；遮挡实验需要在推理期重新
编码，因此它复用 `run_public_missing_field.py --dump-state` 保存的完整模型状态
（`--reuse-state`），不能用 details 文件代替。

### 2. 基线（表 \ref{tab:agnews}、\ref{tab:github}）

```bash
python benchmark_public.py --dataset github \
    --models LDA,BTM,NMF,NeuralLDA,ETM,CTMLite,CombinedTM,BERTopic,Top2Vec \
    --seeds 11,23,37 --k 30 --topn 10 --readout ctfidf \
    --out public_benchmark_runs_githubtrain6000_test2000.csv \
    --summary public_benchmark_summary_github.csv
```

`NeuralLDA` 是历史文件名，正式实现与论文名称为 **ProdLDA**（`--models ProdLDA` 等价）；
CSV 中沿用 `NeuralLDA` 以保持与既有逐种子文件可合并。ETM 现在是标准概率混合解码；
CombinedTM 与其它神经模型一样使用 20% 留出验证与 700 轮预算。

注意：`public_benchmark_runs_*.csv` 里的 `readout` 列记录的是**命令行请求值**，
而不是每个模型实际使用的读出。`--readout` 只影响 BERTopic / Top2Vec；
其余模型（LDA、BTM、NMF、ProdLDA、ETM、CTM-lite、CombinedTM）始终按各自的
$\beta$ 矩阵取词。各模型在同一读出下的对照见 `fair_readout.py`。

注意 `--out` / `--summary` 是相对 `outputs/` 的文件名。重跑个别模型后可把结果并回主表：

```bash
python merge_benchmark_rows.py outputs/public_benchmark_runs_githubtrain6000_test2000.csv \
    outputs/ro_github_bertopic.csv \
    --summary outputs/public_benchmark_summary_github.csv
```

### 3. 遮挡互补字段（表 \ref{tab:occlusion}、\ref{tab:fieldview}）

```bash
# 训练并保存完整模型状态（每个 setting×seed 一个 key）；中断可自动续跑
python run_public_missing_field.py --dataset github --seeds 11,23,37 --k 30 --lr 5e-4 \
    --fusion gate --fusion-norm balanced --fusion-weight 0.5 \
    --settings fielddrop_on,fielddrop_off \
    --dump-state outputs/github_missing_state.pt

# 已有模型状态时只做评测（不重新训练）
python run_public_missing_field.py --dataset github --seeds 11 --k 30 \
    --reuse-state outputs/github_missing_state.pt \
    --out outputs/public_missing_field_occlusion.csv \
    --field-view-out outputs/public_missing_field_fieldview.csv
```

输出两套指标：真正的遮挡推断（`*_occlusion.csv`，含 ΔC_V、ΔNLL、top-1 一致率、JS）和
字段受限读出（`*_fieldview.csv`，词表截断到某字段并在该字段文本上评测）。二者定义不同，
不要混用。`--no-resume` 表示忽略已有训练状态从头训练（仍会写新的检查点）。

### 4. 互补性与增益（表 \ref{tab:conditional}、图 \ref{fig:cond}）

```bash
python field_complementarity.py     # 词重叠、JS 散度、词表交叉
python field_redundancy.py          # CCA / Ridge 跨视图 R²
python compare_fusion.py outputs/codev_github_base.csv outputs/codev_github_gate_bal.csv
```

### 5. 核心组件消融（表 \ref{tab:core-ablation}）

```bash
# 四档组件 × 11/23/37 种子，与主模型同一 700 轮预算；行边跑边写，可随时中断重跑
python run_core_ablations.py --corpus github            # -> outputs/ablation_public_github_core.csv
python run_core_ablations.py --corpus agnews            # -> outputs/ablation_public_agnews_core.csv
# 单档、少线程：python run_core_ablations.py --corpus github --configs no-cross --threads 2
```

每档配置相对主模型只改一处（`no-cross` / `no-consistency` / `no-mask` / `no-fielddrop`），
配对比较写入 `outputs/rerun/core_ablation_summary.csv`。脚本会给每个子拟合固定
BLAS/OpenMP 线程数（这台机器上并发训练会 OOM，见 `进度记录.md` §2）。

### 6. 机制对照与敏感性

```bash
python occlusion_controls.py --control unigram        # 不读输入的常量 unigram 对照
python occlusion_controls.py --control shuffled       # 概要错排对照
python occlusion_controls.py --control fielddrop-off  # 三语料置零 / 不置零曲线
python bn_readout_sensitivity.py                      # raw β vs softmax(BN(B))
python baseline_fieldview.py --models LDA,NMF,BTM,ProdLDA --corpora github,agnews,civil
```

归档包里的 `*_fit_*.npz` 产生于身份机制之前：复现归档数字时加 `--adopt-legacy-fits`，
脚本会把每个矩阵的内容摘要写进该行的 `fit_id` 列（`legacy:<digest>`），不加该开关则
重新拟合。每行的 `eval_id` 是该语料评测协议的身份，去重键为
`(dataset, model, seed, view, K, eval_id)`。归档包里的 `baseline_fieldview.csv` 由
`python repin_fieldview_archive.py` 重算：逐行与现表比对，完全一致才替换（替换前备份），
随后 `finish_fieldview_regen.py` 重建清单并复验。

对应论文中的 `tab:occl-controls`、`tab:fielddrop-control`、`tab:beta-readout` 与
`tab:fieldview-baselines`。前四个对照都不重新训练主模型（unigram 与 BN 敏感性不需要
检查点，错排与 fielddrop-off 只做前向评分）。

### 7. 10 种子扩展（`tab:seed-paired`）

```bash
# 新增 7 个种子（53/71/89/101/113/127/139），产物都在 outputs/rerun 下
python run_fcntm_train.py --corpus github --seeds 53,71,89,101,113,127,139 --k 30 \
    --fusion gate --fusion-norm balanced --fusion-weight 0.5 --field-dropout 0.5 \
    --max-epochs 700 --lr 5e-4 --recover-nan --resume-every 50 \
    --out outputs/rerun/fcntm_github_seed53plus.csv \
    --dump-details outputs/rerun/fcntm_github_seed53plus_details.json \
    --dump-state outputs/rerun/fcntm_main_state_github.pt
# agnews 同理；随后补 BTM / FC-NTM 遮挡 / LDA 遮挡并汇总
python extend_seeds_btm.py --corpora github,agnews --seeds 53,71,89,101,113,127,139
python extend_seeds_fcntm_occlusion.py        # 默认读取 outputs/rerun 下的状态
python extend_seeds_lda_occlusion.py
python summarize_seed_extension.py   # -> outputs/rerun/seed_extension_summary.csv
python verify_newseed_archives.py   # 重算锚点种子，核对后绑定评测身份边车
```

`summarize_seed_extension.py` 在计算前校验每个语料的十个种子、两种读出、五个遮挡率与
有限指标，并核对四个新种子产物（`fair_readout_*_newseeds.csv`、
`occlusion_robustness_*_newseeds.csv`）的身份边车：拟合身份覆盖语料、词表与代码版本，
评测身份另外覆盖测试集两个字段的内容、遮挡率与掩码算法版本、以及 FC-NTM 检查点的内容
哈希；边车缺失或与当前协议不一致时拒绝汇总（`seed_identity.py`）。归档行的绑定过程见
`verify_newseed_archives.py`（重算锚点种子并逐行核对后再写边车）。

## 数据

`data/agnews/`（6000 训练 / 7600 测试）来自 AG News 官方划分；`data/github_issues/`
（6000 / 2000）由四个仓库的原始 parquet 生成，字段映射为正文=背景、标题=概要；为控制
交付体积，原始 parquet 已在清理时移除，已处理划分仍在 `data/github_issues/`。从原始数据
重建需先重新下载那四个仓库，`public_data_github.py` 会在缺失时报错。词典仅由训练集构建。私有工程变更单语料不随代码分发；
如需复现私有部分，在 `config.py` 中填写清理后表格路径后运行
`python run_fcntm_train.py --corpus private`。

## 结果文件与论文对应

| 文件 | 论文内容 |
| --- | --- |
| `outputs/fcntm_github_final.csv` | GitHub 主表（表 \ref{tab:github}）的 FC-NTM 两行，逐种子 |
| `outputs/fcntm_agnews_final.csv` | AG News 主表（表 \ref{tab:agnews}）同上 |
| `outputs/public_benchmark_runs_*.csv` + `outputs/rerun_btm_*.csv` + `outputs/rerun_etm_*_fixed.csv` + `outputs/rerun_combinedtm_*_fixed.csv` | 两个主表的基线行 |
| `outputs/fair_readout_{github,agnews}_3seed.csv` | 读出对照表（表 \ref{tab:fair}） |
| `outputs/public_missing_field_occlusion.csv` | 整字段遮挡（表 \ref{tab:occlusion}） |
| `outputs/public_missing_field_fieldview.csv` | 字段受限读出（表 \ref{tab:fieldview}） |
| `outputs/occlusion_robustness_*_seed*.csv` | 部分遮挡曲线（表 \ref{tab:occl-robust}、图 \ref{fig:occlusion}） |
| `outputs/occlusion_comparison.csv` | 跨语料遮挡汇总（表 \ref{tab:occl-cross}、图 \ref{fig:occl-cross}） |
| `outputs/ablation_public_github_{nofusion,fusedmean,*_w*}.csv` | 融合消融与权重扫描（表 \ref{tab:fusion}、\ref{tab:weight}） |
| `outputs/ablation_public_{github,agnews}_core.csv` | 核心组件消融，两个语料（表 \ref{tab:core-ablation}） |
| `outputs/field_redundancy.json` / `field_complementarity.csv` | 冗余度度量（表 \ref{tab:conditional} 的左两列、图 \ref{fig:cond}） |
| `outputs/clean_test_scores.csv` | 无开发重叠测试集复核表 |

开发阶段用过的 `outputs/codev_*.csv` 仍保留以备追溯，但**论文的 FC-NTM 行取自
`fcntm_*_final.csv`**；引用旧文件会得到未定稿的数字。全部产物与论文文件的当前哈希见
`outputs/RESULTS_MANIFEST.json`，其中也包含 `main.pdf` 与四张图 PDF 的发布校验。

2026-10 修订新增的结果：`outputs/rerun/ladder_summary.csv`（`tab:ladder`）、
`outputs/rerun/chunked_v2_benchmark_*.csv`（`tab:chunked`）、
`outputs/occlusion_control_*.csv`（`tab:occl-controls`、`tab:fielddrop-control`）、
`outputs/bn_readout_sensitivity.csv`（`tab:beta-readout`）、
`outputs/baseline_fieldview.csv`（`tab:fieldview-baselines`）、
`outputs/rerun/seed_extension_summary.csv`（`tab:seed-paired`）。

## 环境

归档运行使用 Python 3.13.15（`requirements.txt` 由 `dump_environment.py` 生成，含
`torch==2.13.0+cpu`、`transformers==5.16.1`、`sentence-transformers==6.0.0`、
`gensim==4.4.0`、`scikit-learn==1.9.0` 等；可选依赖 `bertopic`、`top2vec`、
`contextualized-topic-models`、`jieba`、`pyarrow` 单列在文件末尾）。从 parquet 重建
GitHub 划分需要 `pyarrow`；私有语料需要 `jieba` 与 `openpyxl`，公开路径不需要
（`check_public_imports.py` 会在屏蔽 `jieba` 的环境下逐个 import 公开模块来验证这一点）。

句向量编码器固定为 `sentence-transformers/all-MiniLM-L6-v2` 的
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，最大长度 128，并默认离线加载。首次准备
权重需要联网一次（之后代码会以 `HF_HUB_OFFLINE=1` 离线加载）：

```bash
python -c "from transformers import AutoModel, AutoTokenizer; \
  AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'); \
  AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')"
```

CPU 版 PyTorch 按官方 CPU 轮子安装（归档环境为 `torch==2.13.0+cpu`，
<https://pytorch.org/get-started/locally/>）。
两个环境变量用于敏感性检查：`FCNTM_EN_EMBEDDING_MAX_LEN` 改窗口长度，
`FCNTM_EN_EMBEDDING_CHUNKED=1` 打开"长文档按 128 token 切窗后平均池化"
（`embed_public.chunk_and_pool`）。**注意：论文报告的基线数字是默认（截断）设置产出的；
开启 chunked 后必须重跑对应基线并单独报告，不能与已归档的数字混用。**

论文 PDF 由 XeLaTeX 编译：`cd ../FCNTM完整论文 && xelatex main.tex`（跑两遍）。

## 结果身份与唯一清单

旧版本的“已经评估过就跳过”只看少量键，配置、数据或代码版本变化时可能默默复用旧行。
主入口（`run_fcntm_train.py`、`run_ablation_public.py`、`benchmark_public.py`、
`occlusion_robustness.py`、`run_public_missing_field.py`、`fair_readout.py` 与种子扩展
脚本）现在都会写身份边车（`<file>.identity.json`），`--reuse-details` 还要求 details 里
记录完整拟合身份。仍有少量 2026-10 之前的历史拟合缓存没有边车
（`outputs/rerun/*_fit_*.npz`）：入口不会静默复用它们，而是**重新拟合**；
`baseline_fieldview.py` 只有在显式给出 `--adopt-legacy-fits` 时才改用它们，并把该
矩阵的内容摘要（`legacy:<digest>`）写进每一行的 `fit_id` 列。10 种子扩展的评测 CSV
已按同一协议重算锚点行、核对后绑定边车；缺失或不匹配的边车会让
`summarize_seed_extension.py` 直接报错，而不是打印警告继续。若遮挡曲线是从别的检查点
目录评分的，用 `summarize_seed_extension.py --state-dir <dir>` 指定同一来源：汇总器按该
目录下的状态文件计算期望身份，来源不同就拒绝汇总。

三条复现路径要区分：(i) \emph{核验归档数字}——`check_paper_numbers.py` 与
`results_manifest.py --verify`，不需要权重；(ii) \emph{从检查点重评分}——需要
`outputs/*.pt`（3 种子）与 `outputs/rerun/fcntm_main_state_*.pt`（新增 7 种子）；
(iii) \emph{从头训练}——按上文命令重训，所有新入口都要求匹配的实验身份。

```bash
python results_manifest.py            # 写出 outputs/RESULTS_MANIFEST.{json,md}
python results_manifest.py --verify   # 重新计算哈希，报告漂移 / 缺失 / 新增
python check_paper_numbers.py         # 论文表格与正文数字 vs 产物（解析 main.tex）
python check_paper_numbers.py --self-test   # 篡改论文里的一个数字，必须报错
python tests_review_fixes.py                # 25 项回归测试（含消融入口冒烟与身份自检）
```

对账器**不保存期望值副本**：表格与正文里的数字都从 `main.tex` 解析出来，再按显示精度
（三位小数就比三位、百分比按原样）与产物重算值逐位比较；`--self-test` 会把一个印刷数字
故意改错并要求校验失败，这条已纳入回归测试。

## 边界情形：没有任何可见证据时

遮挡协议下有两种"该文档没有任何可见证据"的情况，两者的处理必须写成合法概率分布：

| 情形 | 回退 | 说明 |
| --- | --- | --- |
| BTM 可见文本不产生任何词对（背景被整体遮挡、概要为空、或只剩一个词） | 训练语料主题先验 $\pi_k\propto n_k+\alpha$ | 100% 遮挡下受影响记录 GitHub 34/2000、AG News 576/7599；旧实现会给全零 θ，被遮挡 token 的 NLL 虚高到 27.63 |
| NMF 可见词袋投影到 $W=0$ | 训练文档的平均混合 | 同上；受影响记录 100% 遮挡下 GitHub 1 条、AG News 98 条 |

`baselines.check_simplex` 对每个 $p(z\mid d)$ 与预测词分布强制检查非负且行和为 1，
并保证单篇文档的推断与批次划分无关（回归测试覆盖）。

BTM 的文档—主题后验按原文第 3.3 节的**点估计**形式
$p(k\mid b)\propto\pi_k\phi_{ki}\phi_{kj}$ 计算（`btm_doc_topic(form="point")`，默认）；
把同一 Dirichlet 后验积掉的变体（对应采样条件概率，`form="predictive"`）只作敏感性检查，
不能当作原文的文档后验公式。

清单记录每个语料文件、检查点、结果 CSV 与关键源码的 SHA-256，声明固定协议
（`K=30`、种子 11/23/37、读出方式），并逐表逐图给出所引用的产物。没有匹配边车的既有
结果会被视为“未固定身份”，下次运行会重新计算而不是跳过。

**这套机制的覆盖范围要说清楚**（第四轮复审要求）：

- 边车由下面这几个入口写出：`run_fcntm_train.py`（文件级）、`occlusion_robustness.py`
  （文件级，含检查点内容哈希）、`run_public_missing_field.py`（**逐分区**：dataset+setting+seed），
  以及 `benchmark_public.py` / `run_fcntm_train.py` 逐行写出的 `actual_readout`、`state_hash`、
  `code_hash`。
- 种子扩展的三个入口（`extend_seeds_*.py`）与 `baseline_fieldview.py` 写出的边车是
  **评测身份**：除语料、词表与代码版本外，还包含测试集两个字段的内容摘要、遮挡率列表、
  掩码算法版本与所用 FC-NTM 检查点的内容哈希（`seed_identity.py`）；修改测试集或更换
  检查点会让身份变化，旧行不会被合并。
- 论文各表所用的结果文件大多产生于该机制之前，**本身没有这些列**，它们由唯一清单固定；
  `rerun_btm_nmf_boundary_fix.py`、`rerun_btm_inference_sampling_fix.py` 这类重算脚本直接用
  `to_csv` 写文件，也不写边车。
- `result_identity.matches()` 只比较身份摘要，**不校验 CSV 内容**：身份相符但数字被改过时它
  仍返回 True。行级别的完整性由各入口自己的检查负责（主训练要求 CV/NPMI 有限，遮挡要求
  每个 rate 齐全，缺失字段要求整体行 + 两条字段视图齐全）。
- `state_hash` 标识的是**模型状态** $(\beta,\theta)$，不是评价代码：同一个模型换一套评价
  方法会改变分数而不改变这个哈希。模型状态、评估代码、结果内容三者分别由
  `state_hash`、`code_hash`、唯一清单里的文件哈希覆盖。

逐种子的结果 CSV 每行还写出三列身份信息：

| 列 | 含义 |
| --- | --- |
| `readout` | 命令行**请求**的读出（历史列，保持兼容） |
| `actual_readout` | 实际生效的读出：只有 BERTopic／Top2Vec 会被 `--readout` 重新读出，其余模型一律用自己的 $\beta$ 矩阵 |
| `state_hash` | 该行所用模型状态 $(\beta,\theta)$ 的摘要，数字变了哈希就变 |
| `code_hash` | 产出该行时的实现版本摘要（`result_identity.code_version()`） |
