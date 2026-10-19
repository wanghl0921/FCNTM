# 修订说明（当前版本）

本文件记录论文当前版本相对早期版本的实质变更，供审阅时核对。
更细的逐条过程记录见 `进度记录.md`（§14--§16）；产物与论文文件的当前哈希见
`outputs/RESULTS_MANIFEST.json`。旧的私有语料修订草稿已归档到
`outputs/rerun/archive_scripts/paper_revision_notes_private_draft.md`。

## 1. 影响结论的修复

| 问题 | 处理 | 结果变化 |
| --- | --- | --- |
| seed 23 的部分遮挡曲线误用 seed 11 检查点 | 用正确的检查点重算 6 组曲线 | GitHub 相对退化 1.0304→**1.0285**、AG News 1.0183→**1.0180** |
| ETM 词嵌入张量与调用方 numpy 数组共享内存 | `models.py` 改为独立复制后重训 3 种子 | 仅每个进程内第二个种子（seed 37）变化：AG News $C_V$ 0.502→0.490、GitHub NPMI 0.201→0.213 |
| NMF 概率适配器不保持原分解 | 改为按主题质量配平，使 `theta @ beta == normalize(W H)` | 遮挡退化 ×1.471→**×1.472**、×1.434→**×1.435**（后随第 3 项再变） |
| CombinedTM 未导出最佳验证状态 | 改用早停保存的 `best_components` 并重训 3 种子 | AG News 0.513→**0.526**、GitHub 0.620→**0.643**（IRBO 0.987→0.983） |
| BTM 对无词对的文档返回全零主题分布 | 逐文档回退到语料主题先验，并强制校验分布合法性 | 遮挡退化 GitHub 1.104→**1.052**、AG News 1.254→**1.044**；旧值主要是 27.63 的虚高 NLL |
| NMF 零输入投影到 $W=0$ 时返回非法分布 | 回退到按训练重构质量聚合的主题先验 | GitHub 1.472→**1.470**、AG News 1.435→**1.408** |
| BTM 文档后验公式与原文 §3.3 不一致 | 默认改为原文的点估计形式；积分变体仅作敏感性检查 | 两种形式给出的 top-10 词表逐位相同，主题质量读数不变 |
| BTM 推断的词对抽样依赖批次组成 | 推断阶段按文档文本与种子独立派生随机数（训练采样保持不变） | 单篇推断与批次无关；遮挡曲线按新抽样重算 |

## 2. 计量口径与可复现性

- 新增 `result_identity.py`：结果文件钉在（脚本版本、数据摘要、字段划分、超参）上；
  遮挡身份含**检查点内容哈希**；缺失字段表改为**逐分区**身份（dataset+setting+seed）。
- 新增 `results_manifest.py`：记录语料、检查点、结果与源码的 SHA-256 及逐表逐图的产物映射，
  `--verify` 同时核验论文文件。
- 新增 `check_paper_numbers.py`：从 `main.tex` **解析**表 1--9、clean-test 表与正文百分比，
  按显示精度逐位比对；`--self-test` 覆盖"改数值、加合法注释、删整表行、改正文百分比"四类篡改。
- 保存可复用的拟合状态：`outputs/rerun/btm_fit_*.npz`、`nmf_fit_*.npz` 与依赖版本
  `outputs/rerun/baseline_fit_env.json`，避免每次审计只能重新拟合。

## 3. 论文表述的更正

- 读出匹配口径：BTM 进入同一类 TF-IDF 读出后 FC-NTM-cTFIDF 的领先为 **+0.027／+0.019**
  $C_V$（不匹配口径为 +0.014／+0.012），并披露逐种子结构（GitHub 3/3、AG News 2/3，
  NPMI 1/3）。
- GitHub 上最好的基线由 LDA 变为 BTM，配对优势相应改为 **+0.024±0.004**；
  AG News 上只有 NMF 不在 1.005--1.072 区间内。
- 修复后 BTM 在 AG News 上不再是最差模型；正文改为记录"早期 $\times$1.254 是边界 bug 的假象"。
- 空概要记录：明确概要编码器仍参与融合，并给出实测差异（GitHub 1 条 0.115、AG News 98 条
  最大 0.261、其中 3 条最可能主题改变）。
- 部分遮挡只覆盖置零训练的三个种子；λ_f=0 对照按同一均值融合配置配对（NPMI 3/3、
  $C_V$ 2/3、$\beta$ 两项 0/3）。
- 论文明确写出结果身份列的**覆盖范围**：表 1--9 的结果文件产生于该机制之前，不含这些列。

## 4. 其他

- 字段视角表的概要支撑度 27--30→**29--30**；8.8%→**7.9%** 的分解引用
  `check_missing_decomposition.py`（并修掉该脚本把 `tot.empty` 当作计数的 bug）。
- 文献：BCK-TM 2022, 65(3)；RBO article 20；JMVAE 改为 2016 预印本 + ICLR 2017 投稿记录；
  BigARTM 改为通用加性正则化框架表述。
- 主题样例由 `make_topic_table.py` 从最终模型（`fcntm_*_final_details.json`）生成，
  默认来源与表头词数（前 10 个）均已对齐。

## 5. 2026-10-04/07 续跑与三语料遮挡收尾

- **剩余实验全部完成**（10-04 10:14--21:25 无人值守链）：abs30 阶梯 8 次拟合、civil 部分遮挡
  补 BTM（3 种子）、chunked 句向量基线（github + agnews）。
- **阶梯汇总修正**：`summarize_ladder.py` 原先只读 `ablation_public_civil_core.csv`，漏掉 full
  参照行；现在 full 取自 `fcntm_civil_final.csv`，得到 11.2→3.9→2.0 三档完整表。
- **civil BTM 补行**：新增 `add_civil_btm_occlusion.py`（不能直接 `--models BTM`，否则会被身份
  校验判为不匹配并归档重算）。最好基线由 ProdLDA 1.0272 变为 BTM 1.0067，FC-NTM 1.0100
  不再占优（配对差 −0.0033±0.0010，0/3）。
- **判据降级**：三个语料里只有 GitHub 遮挡占优（+0.0236±0.0038，3/3）；土木文献长度比 11.2、
  背景独有 0.894（都接近或超过 GitHub 的 12.9 / 0.877）却不占优。论文摘要、表 6、§5.3、结论
  与局限性同步改写为否定结果。
- **chunked 公平性控制**：窗口池化把 GitHub BERTopic 0.540→0.581、CombinedTM 0.643→0.652，
  Top2Vec 不变；AG News +0.005 / 0 / +0.012。仍低于 FC-NTM（β 0.701/0.586，cTFIDF
  0.727/0.612）。
- **论文**：新增长度比阶梯小节（表）、长文档控制表，表 6 与表 10 扩为三语料；PDF 33 页、
  0 error / 0 undefined / 0 overfull。
- **对账**：`check_paper_numbers.py` 扩到 **622** 项全部一致（含阶梯、chunked、civil 遮挡与
  civil 配对差），四类篡改自检仍全拦；`results_manifest.py --verify` = 90 个产物、
  drift=0 missing=0 new=0。
- **仍未做**：种子数 10；未参与设计的独立评测数据。（部分遮挡的置零/不置零曲线已补齐。）
## 6. 2026-10-08/09 更新：审阅问题修复、重跑与对照实验

- **chunked v2（tokenizer 真实切窗）**：GitHub BERTopic 0.540→0.582±0.009、CombinedTM
  0.643→0.656±0.018；AG News BERTopic 0.488→0.518±0.011、CombinedTM 0.526→0.488±0.030；
  Top2Vec 使用 Doc2Vec，窗口池化不适用。
- **固定样本阶梯**（同一批 6000/2000 文献 ID + 同一 7685 词表）：full β 0.633 / cTFIDF 0.636；
  abs60 β 0.595 / 0.645；abs30 β 0.534 / 0.606。β 的 −跨字段重构配对差
  −0.022→+0.006(3/3)→−0.006(1/3)，无单调趋势；词掩码代价在 abs30 最大
  （β −0.027、cTFIDF −0.025），abs60 最小。
- **遮挡对照**：常量 unigram 退化比 0.999/1.001/1.000（三语料）；打乱概要
  1.091/1.096/1.084（真实概要 1.029/1.018/1.010）；fielddrop_off 曲线
  1.028/1.020/1.012，与置零训练几乎相同。
- **代码加固**：续训状态写实验身份（不匹配即归档重开）、消融续跑按行/按文件双重校验、
  句向量缓存键含模型/revision/max_length/池化/算法版本、tokenizer 切窗零丢失、
  清单覆盖 civil/阶梯/chunked/对照并在论文表图未映射时报错、CODE_FILES 扩到关键入口。
  回归测试 `tests_review_fixes.py` 22 项全过。
- **论文**：新增「遮挡协议的对照分析」小节（机制对照表 + 训练设置对照表）；阶梯小节按固定
  样本新数字重写为否定结果；词型/token 两种独有率口径写明；"公平比较/不可省略"等过强措辞
  收窄。
- **验证**：`check_paper_numbers.py` 657 项印刷数字全部一致（自检四类篡改全拦）；
  `main.pdf` 33 页、0 error / 0 undefined / 0 overfull；`results_manifest.py --verify`
  131 个产物 drift=0 missing=0 new=0。
- **BN 读出敏感性（§三.11）已完成**：raw 参数读出与 BN 生成分布读出的 top-10 平均 Jaccard
  只有 0.36--0.42，C_V 从 0.701/0.586/0.633 降到 0.625/0.541/0.584（NPMI 同向）；
  逐词平移不改变生成分布却使 100% 主题的 raw 词表改变；论文新增小节与表 tab:beta-readout。
- **基线字段视角对照（§三.10）已完成**：LDA/BTM/NMF/ProdLDA 均按同一字段受限读出评测；
  背景视角上 BTM 在 GitHub/AG News 略高于 FC-NTM（0.710/0.586 对 0.697/0.566），FC-NTM
  只在土木文献领先（0.637）；概要视角上 LDA/ProdLDA 在三个语料都高于 FC-NTM。论文已把该
  读出改写为描述性证据并收窄独特性表述（新增表 \texttt{tab:fieldview-baselines}）。
- **10 种子配对（§五.6 的种子部分）已完成**：FC-NTM 与 BTM 在 GitHub/AG News 各扩到
  10 种子（新增 53/71/89/101/113/127/139）。GitHub 同口径读出 +0.028（95% CI
  [+0.012, +0.043]，9/10）、遮挡 +0.024（10/10，p=0.002）得到支持；AG News 读出
  −0.009（CI 含 0，3/10）、遮挡对同种子最好基线 LDA 为 −0.013（0/10）——3 种子时观测到的
  AG News 小幅优势未复现。论文新增"多种子配对比较"小节与表 tab:seed-paired，摘要/结论改为
  只在 GitHub 上主张多种子支持的增益。
- **摘要压缩已完成**：从 2663 字压到 977 字（保留问题、方法、关键结果与适用边界，
  以及必要的校验句）；PDF 35 页、0 error / 0 undefined / 0 overfull；对账 785 项全部一致，
  清单 148 个产物 drift=0。
- **review 待办已清空**：独立确认语料按用户决定不做；其余条目（续训/消融/缓存身份、
  tokenizer 切窗、chunked v2、固定样本阶梯、遮挡对照、BN 读出敏感性、字段视角基线对照、
  10 种子配对、正文更正与摘要压缩）均已完成。
## 7. 2026-10-11 新 review 修复（进行中）

- **代码身份（P1）**：`run_fcntm_train.py` 在每个 details 记录里写入完整拟合身份
  （K、词表摘要与顺序、数据/字段摘要、协议、lr、预算、代码版本）；`run_ablation_public.py`
  的 `--reuse-details` 逐字段校验该身份并检查 `beta=(K,V)`、`theta=(n_docs,K)`，缺一项即拒绝
  重训；新扩展脚本统一走 `seed_identity.py`，拟合缓存与合并 CSV 都绑定身份（不匹配则归档
  重来）；`models.py` 在"请求预算小于已执行 epoch"时归档旧状态并从头训练。回归测试 23 项全过。
- **交付（P1）**：重训 7 个新种子的 FC-NTM
  检查点（11:34 启动，github→agnews，预计 ~14 h），补齐 review 指出的缺失；清单已加入
  `outputs/rerun/fcntm_main_state_*.pt`、20 个 BTM 拟合、`tab:occlusion` 的 civil 来源，
  以及 `main.pdf` 与四张图 PDF 的发布校验。
- **论文同步**：引言改用 10 种子结果；方法把 BN 敏感性标为已完成并引 `tab:beta-readout`；
  clean-test 段删除"完全排除开发重叠"的表述；多种子节与局限区分读出（FC-NTM/BTM）与遮挡
  （另含 LDA）的种子覆盖，注明候选池差异与 p 值为探索性；阶梯"最小代价"更正；
  局限"方向与幅度接近""没有不置零曲线"两处更正；VALTA 按原文改为 user–item 编码、
  方面权重与句级方面分配；三处论证边界（内容使用≠结构更优、不显著≠可省略、种子区间只
  覆盖训练随机性）写入正文。当前 PDF 35 页、0/0/0，777 项对账全过。
- **README**：新增对照与 10 种子链的运行说明、表映射、原始 parquet 移除说明、MiniLM 首次
  准备与 CPU torch 来源、三条复现路径（核验归档 / 从检查点重评分 / 从头训练）。
### 7.1 本条目的完成状态（2026-10-12）

- 新增 7 种子的 FC-NTM 检查点已在 `outputs/rerun/` 交付
  （`fcntm_main_state_github.pt` 62.5 MB、`fcntm_main_state_agnews.pt` 57.3 MB），
  并写入逐种子的 `fit_identity`；重训 CSV 与归档的新种子 CSV **逐位一致**
  （max |ΔC_V| = max |ΔNPMI| = 0），证明检查点确为原曲线的模型。
- 续训文件已清理（479 MB）；检查点目录当前只保留状态、CSV 与 details。
- 清单重建：**176 个产物**（2026-10-12 review 后再次重建），`--verify` = drift 0 /
  missing 0 / new 0；其中包含 `outputs/rerun/fcntm_main_state_*.pt`、20 个 BTM 拟合、
  6 个 NMF 拟合、`tab:occlusion` 的 civil 来源，以及 `main.pdf` 与四张图 PDF 的发布校验。
- `check_paper_numbers.py` 777 项全过；`tests_review_fixes.py` 当时 23 项全过
  （2026-10-12 增至 25 项）。
- 代码目录仍 <512 MB；C 盘空闲 11.26 GB。
- 2026-10-11 review 的可执行条目至此全部完成；唯一未做项为 review 自己也不强制要求的
  独立确认语料（由用户决定不做）。

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
  选定的 BTM/LDA；原始参数 $\beta$ 读出那一句写明主语（FC-NTM 的 $\beta$ 对 BTM 的
  原生 $\beta$）。
- **表映射**：README 的旧表号（表 1--11、图 2/3/5）全部换成 LaTeX 标签，
  `check_paper_numbers.py` 的进度标题同步改为标签，避免复现者按旧号查错实验。
- **验证**：`check_paper_numbers.py` 777 项印刷数字全过；`tests_review_fixes.py` 25 项全过；
  `results_manifest.py --verify` drift 0 / missing 0 / new 0（176 个产物，含论文与四张图
  PDF）；`main.pdf` 35 页、0 error / 0 undefined / 0 overfull。

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

## 11. 论文压缩与打磨（2026-10-12）

- **结构**：新增附录 A「附加实验与对照」，把长文档截断控制（`tab:chunked`）、长度比阶梯
  （`tab:ladder`）、遮挡协议对照（`tab:occl-controls`、`tab:fielddrop-control`）与
  $\beta$ 读出敏感性（`tab:beta-readout`）移入附录；正文各留一句指针。
- **文字压缩**：逐节精简主结果（GitHub/AG News）、读出公平性、多种子配对、部分遮挡、跨语料
  对比、字段视角、融合消融、核心组件消融、缺失推断与冗余度，以及讨论与局限（九条合并为
  七条）；删除正文内嵌的 clean-test 小表（数字保留在对账器解析的那句话里）。
- **规模**：PDF 从 35 页降到 31 页（其中附录约 3 页），全部表格、图、数字与结论不变。
- **对账**：压缩过程中对账器曾抓到两处问题（一张内嵌表被误删导致 clean-test 表头缺失、
  字段视角两张表被整段替换带走），均已恢复；`check_paper_numbers.py` 777 项与
  `tests_review_fixes.py` 26 项全部通过。
- **下一步**：若要继续压到 22--25 页，可再压缩数据节与模型节、把 `tab:conditional` 与
  `tab:topics` 移入附录，并把四张 \textwidth 图缩小一档。

## 12. 土木定位与正文/补充材料拆分（2026-10-12）

- **土木领域定位**：新增 10 篇英文文献（JCCE 与 Automation in Construction 上的文本挖掘、
  主题模型先例，如 DUET 的 LDA、建筑领域神经主题模型），新增相关工作小节「工程文档的文本挖掘
  与主题建模」，并在摘要、引言与讨论中把「工程文献标题—摘要」写成应用背景。
- **正文/补充材料拆分**：附录整体抽成独立文件 `supplementary.tex` / `supplementary.pdf`
  （长文档控制、长度比阶梯、遮挡对照、$\beta$ 敏感性、冗余度表与图、主题样例，以及数据说明
  与指标推导），正文只保留文字指针；`check_paper_numbers.py` 改为同时解析两份文件。
- **规模**：正文 33 → **28 页**，补充材料独立（约 5 页）；`check_paper_numbers.py` 764 项
  全部一致，两本 PDF 均 0 error / 0 undefined。
- **下一步**：英文重写（按 ASCE JCCE 或 EAAI 模板，目标 ≈9,000--10,000 英文词），逐种子表格
  随补充材料提交。
