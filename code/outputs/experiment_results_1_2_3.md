# 实验 1–3 结果（2026-09-10）

## 结论速览

| 实验 | 结果 | 能否支撑论文主张 |
| --- | --- | --- |
| 1. 公开规模跨字段填补（GitHub 6000/2000） | recon2 与 noxrec **无差别** | ❌ 不能 |
| 2. 私有语料 10 种子配对检验 | recon2 显著优于 noxrec（p=0.0019） | ✅ 可以 |
| 3. 缺失字段分组评估 | FC-NTM 在无原因字段文档上 C_V 掉 0.12；先验回退不优于强制编码 | ❌ 与原主张相反 |

---

## 1. 公开规模跨字段填补（GitHub issues）

**任务**：在 6000 篇训练，2000 篇留出（全部有正文）上，用标题（事项字段）预测正文（原因字段）的词。

**修复**：`run_public_imputation.py` 原来在测试集上崩 `TypeError: expected np.ndarray (got NoneType)`，
因为 `public_data_github.load_github` 对 test 用 `need_bow=False`，test 无 BoW。
现在按训练词表在脚本内重建 test 的 BoW。

**训练不稳定**：公开语料上 FC-NTM 会在 ~epoch 300 出现 `loss is not finite`（线性解码器也会，
不只是 embedding 解码器）。已给 `models.py` 加可选数值恢复（回滚到上一个干净 epoch + 学习率减半 +
熔断），并把公开实验的学习率降到 `5e-4`。

**结果**（seed 11, K=30, 700 epoch 上限, 评估 2000 篇留出文档）：

| config | item_only recall@10 | item_only precision@10 | cause_oracle recall@10 |
| --- | --- | --- | --- |
| recon2 (cause_recon_weight=2.0) | 0.0711 | 0.2334 | 0.0505 |
| noxrec (无跨字段重建) | 0.0727 | 0.2255 | 0.0444 |

→ **两者实质相同**（recall −0.0016，precision +0.008）。只有 1 个种子，
但方向上不支持"跨字段重建提升公开语料填补能力"。可能原因：GitHub 正文很长且主题发散，
"标题→正文"与"事项→原因"不是同一类任务。

复现：`python run_public_imputation.py --seeds 11 --configs recon2,noxrec --k 30
--max-epochs 700 --lr 5e-4 --recover-nan --out outputs/public_imputation.csv`
（每个 config 约 50 分钟）

---

## 2. 私有语料跨字段填补：10 种子配对检验（核心正面结果）

指标：`item_recall@10` —— 把原因字段置为缺失，只用事项字段推断主题分布，
再预测该文档真实的原因字段词（80/20 划分，评估留出的成对文档）。

| config | item_recall 均值 | 标准差 | item-cause 余弦 |
| --- | --- | --- | --- |
| recon2 | **0.1407** | 0.0453 | 0.650 |
| base | 0.1094 | 0.0378 | 0.655 |
| noxrec | 0.0730 | 0.0380 | 0.448 |

配对检验（10 个种子，逐种子配对）：

| 对比 | 平均差 | t | p (paired t) | p (Wilcoxon) |
| --- | --- | --- | --- | --- |
| recon2 − noxrec | **+0.0677** | +4.33 | **0.0019** | 0.0039 |
| base − noxrec | +0.0364 | +2.49 | 0.0345 | 0.0371 |

→ **跨字段重建把缺失字段填补能力提升约 1.9 倍，10 个种子方向一致且统计显著。**
这是目前唯一站得住的方法论主张。

产物：`outputs/cross_field_sweep_merged.csv`（30 行 = 10 种子 × 3 配置）

---

## 3. 缺失字段分组评估（5 种子）

把模型的全局主题分别用三组文档作为参考语料算一致性：
`all`(577) / `with_cause`(292) / `without_cause`(285)。模型只在 80% 上训练。

**C_V / NPMI（5 种子均值）**

| 文档组 | 模型 | C_V | NPMI |
| --- | --- | --- | --- |
| with_cause | FC-NTM-Emb | 0.522 | −0.008 |
| with_cause | FC-NTM-Emb-ForceEnc | 0.513 | +0.026 |
| with_cause | BTM | 0.476 | +0.040 |
| with_cause | NMF | 0.532 | +0.027 |
| without_cause | FC-NTM-Emb | **0.403** | −0.151 |
| without_cause | FC-NTM-Emb-ForceEnc | 0.567 | −0.536 |
| without_cause | BTM | 0.600 | +0.015 |
| without_cause | NMF | 0.477 | −0.418 |
| all | FC-NTM-Emb | 0.496 | −0.057 |
| all | FC-NTM-Emb-ForceEnc | 0.539 | −0.155 |
| all | BTM | 0.484 | +0.074 |
| all | NMF | 0.498 | −0.011 |

**两个问题**：
1. FC-NTM-Emb 在缺失原因字段的文档上 C_V 从 0.522 掉到 0.403，说明缺失字段确实损伤了它的主题质量。
2. 论文里的"缺失时回退到先验，从而对部分记录更鲁棒"这一设计主张**没有得到支持**：
   强制编码空字段（ForceEnc）在 `without_cause` 上 C_V 反而更高（0.567 vs 0.403），
   虽然它的 NPMI 更差（−0.536 vs −0.151）。两种方案各有胜负。

产物：`outputs/missing_field_eval.csv`

---

## 4. 对论文的直接影响

**可以写的**：
- 跨字段重建显著提升缺失字段填补能力（私有语料 10 种子，p≈0.002）——作为主贡献。
- 公开基准上全局一致性达到神经主题模型最好水平（0.683/0.222 vs CombinedTM 0.618/0.176）。
- 如实报告：公开 C_V/NPMI 略低于 BTM（0.706/0.230）。

**不能写的**：
- ❌ "FC-NTM-Emb 在 C_V/NPMI 上超过 BTM"（in-sample + 温度锐化的产物）。
- ❌ "缺失字段先验回退带来鲁棒性"（实验 3 反证）。
- ❌ "跨字段重建在公开规模上提升填补"（实验 1 无效果）。

**必须补的**：
- 公开实验只有 1 个种子，若要用作证据需补种子；否则只作为"规模可行性"陈述。
- 缺失字段鲁棒性要么放弃该主张，要么改设计（例如让缺失流走"可学习的先验查询"而不是零向量）。
