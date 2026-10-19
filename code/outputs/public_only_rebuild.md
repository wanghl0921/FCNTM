# 公开语料版论文重建说明

## 目标

论文正文与图表**不再出现任何私有语料**（原工程变更单语料，577 条 / 285 条缺失字段 /
49.4% 等），全部实证只使用公开数据集。

## 语料

| 语料 | 来源 | 划分 | 字段1（背景） | 字段2（概要） |
|---|---|---|---|---|
| AG News | 官方 train/test CSV（镜像 mhjabreel/CharCnn_Keras） | 6000 / 7600 | 描述 | 标题 |
| GitHub issues | open-index/open-github-issues parquet，4 仓库 | 6000 / 2000 | 议题正文 | 议题标题 |

AG News 原始 CSV 在本机已丢失（`public_data.py` 指向的 `agnews_repo` 目录不存在），
已重新下载：

- `C:\Users\Administrator\Downloads\agnews_repo\data\ag_news_csv\train.csv`（29,470,338 B）
- `C:\Users\Administrator\Downloads\agnews_repo\data\ag_news_csv\test.csv`（1,857,427 B）

缓存切分：`data/agnews/{train,test}.csv`（seed 20260905）。

## 代码改动

1. **字段映射统一**（`public_data.py`）：AG News 原来把标题当 `cause`、描述当 `item`，
   与 GitHub 语料（正文=`cause`、标题=`item`）相反。现已统一为
   **背景字段 = 描述/正文，概要字段 = 标题**，两个语料同义。
2. **AG News 接入训练脚本**（`run_fcntm_train.py`）：新增 `--corpus agnews`。
3. **β/θ 落盘**：`run_fcntm_train.py --dump-details`、`benchmark_public.py --details-out`，
   用于字段视角分析（需要完整主题—词矩阵，仅存 top-10 词不够）。
4. 新增脚本：
   - `public_missing_stats.py` — GitHub 语料的天然字段缺失率
   - `fieldview_public.py` — 两个语料的字段视角一致性
   - `run_public_missing_field.py` — 缺失字段鲁棒性（公开语料版）
   - `run_ablation_public.py` — 公开语料上的组件消融
5. **论文生成器** `build_public_paper.py`：所有数字从 `outputs/` 的运行产物读取，
   缺失的产物显示 `（待补）` 而不是编造数值。输出
   `C:\Users\Administrator\Downloads\FCNTM完整论文\FCNTM_公开语料版.docx`。

## 公开语料上的关键统计

GitHub issues 天然缺失字段（正文为空或过短）：**7110/80509 = 8.83%**
（duckdb 5.5%、TypeScript 9.7%、moby 7.7%、redis 10.8%）。
这是私有语料“285/577 = 49.4%”的公开替代，但**数量级低得多**：公开语料的字段缺失
远没有私有语料严重，因此“缺失字段鲁棒性”这一卖点在公开数据上会明显弱化。

## 任务流水线（严格串行，8GB 内存限制）

- `run_all_public.ps1`（PID 15044）— 主流水线，可重复执行，按产物行数跳过已完成任务
- `run_all_public2.ps1`（PID 13036）— 跨字段词预测实验

设置 `OPENBLAS_NUM_THREADS=4 / OMP_NUM_THREADS=4`：8 线程的 OpenBLAS 内存池
加上第二个训练进程会直接分配失败（已实测报错 `OpenBLAS error: Memory allocation
still failed after 10 retries`）。

任务顺序：

| # | 任务 | 产物 | 预计 |
|---|---|---|---|
| J1 | AG News FC-NTM 3 seeds | `outputs/refresh_agnews_linear.csv` | ~2h |
| J2 | GitHub FC-NTM seeds 23,37 | `outputs/refresh_public_linear.csv` | ~1.5h |
| J3 | GitHub 基线（含 β） | `outputs/public_benchmark_runs_github_v2.csv` | ~4.7h |
| J4 | AG News 基线（含 β） | `outputs/public_benchmark_runs_agnews.csv` | ~4h |
| J5 | 字段视角 | `outputs/public_fieldview_{github,agnews}.csv` | ~15min |
| J6 | 缺失字段鲁棒性 | `outputs/public_missing_field_eval.csv` | ~2.2h |
| J7 | 组件消融 | `outputs/ablation_public_github.csv` | ~6.6h |
| J8 | 跨字段词预测 | `outputs/public_imputation_multi.csv` | ~4.5h |

合计约 25 小时后台 CPU 时间。

## 重新生成论文

```powershell
cd C:\Users\Administrator\Downloads\FCNTM完整代码
$env:PYTHONIOENCODING='utf-8'
python build_public_paper.py
```

随时可跑；有多少产物就填多少表格，其余显示 `（待补）`。

## 必须如实说明的风险

1. **跨字段重构的公开证据目前很弱。** 1 个种子的 GitHub 结果：开启跨字段重构
   recall 0.0711 / precision 0.2334，关闭 0.0727 / 0.2255，**没有区别**。
   原来 p=0.002 的强证据来自私有语料（10 seeds）。J8 就是在公开语料上重做这个检验；
   若仍为 null，论文的核心卖点必须重新定位（转向字段视角与缺失字段鲁棒性）。
2. **缺失字段比例差异大。** 公开语料 8.83% vs 私有语料 49.4%，结论的适用范围变窄，
   已在第 6 节局限性中写明。
3. **FC-NTM 在通用 C_V 上并不全面胜出**，GitHub 上 BTM 仍最强；论文不能声称全面占优。
4. 历史发现（`outputs/verification_findings.md`）中“FC-NTM-Emb 击败 BTM”是三类偏差
   叠加的假象，公开版已不使用该配置（只用线性解码器 + β/cTFIDF 读出）。
