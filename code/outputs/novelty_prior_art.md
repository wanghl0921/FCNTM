# FC-NTM 新颖性核查备忘（2026-09-09）

## 一句话结论
- 以 "field-collaborative neural topic model" / "字段协同神经主题模型" 为名的模型：**未检索到**，模型名可以保留。
- 但"多字段 / 多视角 + 主题模型"并非空白：VALTA/VALDA、Multi-field CTM、SubNTM、FVAE、DV-BTM、Multi-source NVI 已分别覆盖部分要素。
- 可发表的新颖性必须收窄为以下**组合**：同一文档内的字段级条件变分流 + 共享解码器 + 跨字段重建 + JS 字段一致性 + **缺失字段先验回退** + 字段级词掩码 + 字段视角评价协议。其中"缺失字段"是最强、最难被现有工作覆盖的差异化点。
- **biterm 正则不是新颖点**：BigARTM `biterms_phi` regularizer、BCK-TM、GraphBTM、TRNMF 都已做过。只能作为消融，不能作为主贡献。
- **embedding decoder 不是新颖点**：ETM / CombinedTM 的 embedding decoder 已是标准做法。只能定位为"配置改进"，不能作为方法创新。

## 必须引用并逐条区分的最近工作

| 工作 | 出处 | 做了什么 | 与 FC-NTM 的重叠 | 必须写清的差异 |
| --- | --- | --- | --- | --- |
| VALTA / VALDA | Esmaeili et al., AISTATS 2019 / arXiv:1812.05035, 1812.05035 | 把评论按"方面(aspect)"结构化，用 aspect-specific topics 构造 log-linear 似然 | **最接近**：方面≈字段，per-aspect 主题 | VALTA 需要句级 aspect 标注/分配，字段齐全；FC-NTM 字段是文档级记录字段，且允许字段整体缺失并用先验回退 |
| Multi-field Correlated Topic Modeling | SDM 2009 | CTM 扩展，多字段（离散+实值）联合建模 | 多字段联合主题 | 非神经、无缺失字段处理、无跨字段重建/一致性约束 |
| SubNTM | Nguyen et al., Knowledge-Based Systems 2025 | 文档切成 sub-document，文档级+子文档级主题 | 文档内结构建模 | 子文档是切分而非语义字段；无字段缺失问题；无字段级一致性 |
| FVAE (Field-aware VAE) | Fan et al., IEEE 2022 | 多字段用户数据，每个字段独立 multinomial | 字段级 VAE、多字段 | 目标是用户表征/推荐，不是主题一致性；无跨字段一致性、无缺失字段回退 |
| DV-BTM | Liu et al., IEEE DSCA 2022 | 双视角 biterm 主题模型 | "双视角"、biterm | 两个视角是 local+global 语料信息，不是同一文档的两个语义字段 |
| Multi-source / Multi-view NVI | Kurle et al., AAAI 2019；Gupta et al., NAACL 2021 | 多源/多视角变分推断迁移 | 多变分编码器 | 视角是不同语料/域，不是同一文档内的字段；无缺失字段机制 |
| PTM (Parallel Topic Modeling) | IBM 专利 US20110202484, 2010 | problem/solution 字段 + 主题转移概率 | 原因/事项字段的工业建模 | 非神经、无缺失字段回退、无字段一致性约束；作为专利需引用区分 |
| FICM / Multi-field document clustering | Miyamoto & Mamitsuka 等 | 多字段文档聚类 | 多字段 | 前神经时代、聚类而非主题发现 |
| GraphBTM | Zhu et al., EMNLP 2018 | 用图神经网络做 BTM 推断 | biterm + 神经 | 证明 biterm 神经化已有工作；FC-NTM 不以此为主贡献 |
| BCK-TM | The Computer Journal 2020 | 用 biterm 相关性知识正则化短文本主题 | biterm 正则 | 与我们的 biterm 项高度重叠，必须引用并说明只作消融 |
| BigARTM biterms_phi | Vorontsov et al., 2017 | biterm/共现 phi 正则 | biterm 正则 | 同上，明确 biterm 非本文创新 |
| ETM / CombinedTM | Dieng et al., ICML 2020；Bianchi et al., NAACL 2021 | 预训练词向量解码器 | embedding decoder | 明确 FC-NTM-Emb 的改进来自字段协同，而非解码器 |
| Neural Models for Documents with Metadata | Card et al., ACL 2018 | 元数据感知 NTM | 元数据/文档结构 | 元数据是文档级协变量，不是字段级文本；无缺失字段回退 |
| STM | Roberts et al. 2019 | 内容协变量主题模型 | 字段对主题的影响 | 频度学派、协变量而非字段级神经编码；无缺失字段机制 |
| JMVAE | Suzuki et al., AAAI 2017 | 联合多模态 VAE，可补缺失模态 | 缺失视角的思想 | 多模态 VAE 中已有；但未用于字段级主题建模与字段视角一致性评价 |

## 建议的安全新颖性表述（可直接改写进 Introduction）

1. **问题层面**：首次把施工变更单建模为"原因字段 + 变更事项字段"的字段级结构化短文本问题，并给出该语料中 **285/577（49.4%）文档缺失原因字段** 的实证统计，把"字段缺失"确立为一等公民。
2. **方法层面**：提出字段协同神经主题模型 FC-NTM——字段级条件变分编码器共享一个主题-词解码器，训练目标同时包含字段重建、跨字段重建与 JS 字段一致性；当某字段缺失时，该流回退到先验而非强制编码空文本。上述**缺失字段先验回退 + 跨字段重建 + JS 一致性**的组合，在现有 aspect/multi-view topic model 文献中未见。
3. **评价层面**：提出**字段视角主题一致性协议**（分别在原因词表与事项词表上计算 C_V/NPMI/UMass，并报告字段间主题对齐度），用于诊断"全局指标好看但某一字段主题塌陷"的问题。

## 风险与对策
- 风险：审稿人指出 VALTA 已做 aspect-specific neural topic model。对策：在 Related Work 用一小段专门对比，强调"字段可整体缺失"与"字段是记录结构而非句级方面"，并用缺失字段消融证明回退机制的必要性。
- 风险：embedding decoder 被指 ETM 已有。对策：不把解码器当创新，只报告"字段协同 + 预训练词向量"的配置增益，并保留 linear decoder 作对照。
- 风险：biterm 被指已有。对策：主文不主张 biterm 创新，只在消融中报告"embedding decoder 下 biterm 反而有害"。

## 投稿前必须补齐的证据
- 公开数据集（GitHub issues）上 FC-NTM-Emb 的 3 种子结果：必须与 BTM/NMF/LDA/ETM/CTM/CombinedTM/BERTopic 同表可比。
- 组件消融：cross-reconstruction 开/关、consistency_weight=0/0.5、cause/item dropout 开/关、缺失字段回退 vs 强制编码空文本。
- K 敏感性：K=20/30/40。
- 缺失字段分组评价：有原因字段 vs 无原因字段文档分别报告主题质量。
- 私有语料为 in-sample 评价，必须在论文中明确说明；泛化证据以公开数据集的 train/test 拆分为准。
