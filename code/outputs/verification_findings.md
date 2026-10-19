# FC-NTM-Emb "赢了 BTM 吗？" 核查记录（2026-09-09）

## 结论（先给答案）
**不是伪造式的作弊，但目前这套证据不足以支持"FC-NTM-Emb 真的赢了 BTM"。**
原来的私有语料胜利有三个可证实的偏置：in-sample 评价、未归一化词向量 + 温度锐化、以及被削弱的 ETM 基线；
真正干净的公开留出集实验（GitHub 2000 测试文档）在 20:22 因 embedding decoder 发散而崩溃，所以目前**没有**可信的泛化证据。

## 证据一：私有语料是 in-sample 评价
- `run_fcntm_train.py` 私有分支调用 `run_corpus(data, data, ...)`，训练语料 == 评价语料（577 篇）。
- `experiment.py:evaluate_run` 也用同一份 `corpus` 计算 C_V/NPMI/UMass。
- BTM/NMF 同样是 in-sample，但神经模型容量更大，更容易记住语料共现；in-sample 指标天然偏向神经模型。

## 证据二：私有语料太小，留出集 C_V/NPMI 大量为 NaN
严格 80/20 留出（462 训练 / 115 留出，K=30），主题只在训练集上学习，指标在留出文档上计算。

seed 11：
| 模型 | in-sample C_V | in-sample NPMI | held-out C_V | held-out NPMI | held-out UMass |
| --- | --- | --- | --- | --- | --- |
| FC-NTM-Emb | 0.610 | +0.158 | NaN | NaN | −9.92 |
| FC-NTM-Emb-Shuf | 0.551 | +0.024 | NaN | NaN | −10.94 |
| ETM-Frozen | 0.519 | +0.098 | NaN | NaN | −11.70 |
| FC-NTM-Linear | 0.502 | +0.023 | NaN | NaN | −11.72 |
| BTM | 0.514 | +0.129 | NaN | NaN | −13.56 |
| NMF | 0.522 | +0.008 | NaN | NaN | −15.30 |

seed 23：
| 模型 | in-sample C_V | in-sample NPMI | held-out C_V | held-out NPMI | held-out UMass |
| --- | --- | --- | --- | --- | --- |
| FC-NTM-Emb | 0.523 | +0.022 | 0.451 | −0.137 | −11.45 |
| ETM-Frozen | 0.503 | +0.028 | 0.390 | −0.052 | −11.16 |
| BTM | 0.473 | +0.068 | **0.536** | **+0.115** | −13.86 |
| NMF | 0.476 | −0.004 | **0.634** | **+0.172** | −14.97 |

要点：
- 115 篇文档下，top 词对几乎不共现，gensim c_v/c_npmi 除零 → 大量 NaN；**私有语料规模根本不足以做可信的留出集一致性评价**。
- 能算出来的 seed 23 上，**BTM/NMF 在留出集 C_V/NPMI 上反超 FC-NTM-Emb**；seed 11 只有 UMass 可算且 FC-NTM-Emb 更好。结论：证据矛盾，不能宣称稳赢。
- in-sample 与 held-out 的差距本身说明 in-sample 数字被高估。

## 证据三：embedding decoder 的"外挂"与不稳定
1. **词向量未归一化**：`embeddings.embed_words` 返回 mean-pooled hidden state，M3E 范数≈20，MiniLM≈7；`EmbeddingDecoder` 没有 BatchNorm/LayerNorm，`EMB_DECODER_TEMPERATURE=0.35` 又把 logits 放大 2.9 倍。
   - 私有 880 词表侥幸收敛；公开 GitHub 5175 词表 **训练发散**：`FloatingPointError: loss is not finite`（20:22，跑了约 2 小时）。
2. **打乱词向量对照**：FC-NTM-Emb-Shuf（行打乱）in-sample NPMI 从 +0.158 掉到 +0.024（seed 11）→ 大部分 in-sample 增益来自 **M3E 的外部语义先验**，而不是字段协同结构。
3. **C_V 对稀有词/短文本不敏感**：Shuf 的 in-sample C_V 仍有 0.551，高于 BTM 0.514，但 NPMI 只有 +0.024 → 私有语料上的 C_V 不能单独作为"赢"的证据。

## 证据四：ETM 基线被削弱
- `experiment.py:_run_entm` 对 ETM 基线固定 `trainable_embeddings=True`：880×768≈67.5 万词向量参数在 577 篇文档上训练，`best_epoch` 平均仅 154，明显过拟合。
- FC-NTM-Emb 使用冻结词向量。二者不是同一起跑线；公平对照应是 ETM-Frozen（本次留出实验已加入）。

## 证据五：超参选择偏差
- decoder 类型、biterm λ、K 等是在私有语料的 C_V/NPMI 上挑选的，私有数字本身带有选择偏差。
- 唯一干净的检验是公开语料的 train/test 拆分（6000/2000）。

## 下一步（按优先级）
1. **修稳定化后重跑公开留出集**：词向量 L2 归一化 + EmbeddingDecoder 加 BatchNorm（或降 LR / 温度=1.0），GitHub 6000/2000，K=30，种子 11/23/37。这是"是否真赢"的唯一定论实验。
2. **公平基线**：ETM-Frozen、单流 ProdLDA、BTM、NMF 同协议。
3. **字段结构对照**：打乱字段分配（把原因/事项互换或随机配错）后重训；若指标不降，说明字段协同没起作用。
4. **私有部分只报 UMass + 字段视角指标**，不再用 in-sample C_V/NPMI 宣称超过 BTM。
5. 若公开留出集上 FC-NTM-Emb 不能稳定超过 BTM：论文定位改为"字段视角可解释性 + 缺失字段鲁棒性"，全局一致性报告为与 BTM 相当，而不是超过。

## 补充：3 种子完整留出结果（seed 11 大量 NaN，仅 seed 23/37 可算）
held-out（115 篇文档）均值：
| 模型 | held-out C_V | held-out NPMI | held-out UMass | in-sample C_V | in-sample NPMI |
| --- | --- | --- | --- | --- | --- |
| ETM-Frozen | 0.588 | +0.228 | −11.46 | 0.508 | +0.034 |
| NMF | 0.446 | −0.068 | −14.78 | 0.497 | −0.003 |
| FC-NTM-Emb | 0.422 | −0.103 | −11.13 | 0.542 | +0.018 |
| FC-NTM-Emb-Shuf | 0.416 | +0.134 | −11.18 | 0.509 | −0.003 |
| BTM | 0.395 | −0.056 | −13.86 | 0.493 | +0.095 |
| FC-NTM-Linear | NaN | NaN | −13.03 | 0.456 | −0.022 |

逐种子波动极大（BTM held-out NPMI：seed 23 为 +0.115，seed 37 为 −0.228；ETM-Frozen seed 37 为 +0.507）。
**115 篇留出集上的 C_V/NPMI 噪声过大，无法据此判定任何模型胜出。**

## 补充：运行层面的两个新事实
1. 归一化 + 温度=1.0 之后，公开语料 embedding decoder 不再报 NaN，但在同时运行两个 GitHub 任务时因**内存不足**崩溃（`DefaultCPUAllocator: not enough memory`，仅 2.6MB 分配都失败）→ 本机内存受限，公开实验必须串行、单独跑。
2. 因此"公开留出集能否赢 BTM"这个唯一定论实验目前**仍未完成**。

## 最终判定
- 私有语料上的 FC-NTM-Emb > BTM 是 **in-sample 结果**，且 held-out 证据矛盾/不可用。
- 公开留出集实验因发散和内存问题尚未完成。
- **现阶段只能说：私有上的"胜利"不可信，不能写进论文作为"超过 BTM"的证据。**

## 补充：温度/尺度敏感性实验（私有语料，seed 11，词向量 L2 归一化）
"有效锐化尺度" = 1/temperature（越大越尖锐）：

| 有效尺度 | β C_V | β NPMI | β UMass | β EXCL_KADJ | cTFIDF C_V | cTFIDF NPMI |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 (temp=1.0) | 0.440 | −0.044 | −10.93 | 0.582 | 0.431 | −0.000 |
| 2.86 (temp=0.35) | 0.434 | −0.065 | −11.59 | 0.641 | 0.427 | −0.033 |
| 20 (temp=0.05) | 0.491 | +0.030 | −10.20 | 0.678 | 0.441 | +0.008 |
| 57 (temp=0.0175) | 0.454 | +0.055 | −10.09 | 0.779 | 0.482 | +0.064 |
| 57（原始未归一化，temp=0.35） | 0.514 | +0.107 | −9.41 | 0.754 | 0.466 | +0.056 |

规律：**NPMI 和 EXCL_KADJ 随锐化尺度单调上升**；C_V 在尺度 20 附近最高。原始的 +0.107 需要"未归一化词向量（范数≈20）× 温度 0.35"这种极端锐化才能达到。
在标准配置（归一化 + 温度 1.0）下，FC-NTM-Emb 三个种子一致跌到 **C_V 0.441 / NPMI −0.091**，**低于 BTM 的 0.470 / +0.075**。

## 最终核查结论
1. **"FC-NTM-Emb 超过 BTM"不成立**。它是三重偏置叠加的产物：
   - in-sample 评价（train=test）；
   - 未归一化词向量 + 低温度造成的极端锐化（锐化本身会抬高 NPMI/排他性）；
   - 锐化强度/词向量几何是任意超参，且在私有评价指标上被隐式选中。
2. **公开留出集（唯一干净证据）**：线性 FC-NTM 已经是 C_V 0.683 / NPMI 0.222，低于 BTM 的 0.706 / 0.230；embedding 版本在公开语料上先发散、后因内存不足崩溃，未能完成。
3. **可以保留的贡献**：字段协同结构 + 缺失字段先验回退 + 字段视角评价协议（组合层面是新的；49.4% 缺失率是真实动机）。在公开语料上 FC-NTM-cTFIDF 是**最好的神经主题模型**（0.683/0.222，高于 CombinedTM 0.618/0.176、NeuralLDA 0.591/0.124、ETM 0.503/−0.391、NMF 0.614/0.164），但仍低于 BTM。
4. **论文定位建议**：不再宣称"全局一致性超过 BTM"；改为"面向缺失字段的字段协同神经主题模型 + 字段视角评价"，全局一致性如实报告为接近但略低于 BTM，把贡献放在字段级可解释性、缺失字段鲁棒性与公开基准上对其它神经主题模型的领先。

## 数据完整性提醒
- 温度扫描脚本会覆盖 `outputs/biterm_private_embedding_k30_details.json`（原 5 种子版本已被 seed 11 版本覆盖）。逐种子指标仍在 `outputs/biterm_private_emb_seed11.csv`，字段视角结果仍在 `outputs/emb_field_view.csv`；如需原始 details 可重跑 5 种子 embedding 配置恢复。
