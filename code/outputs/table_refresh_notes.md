# 主表刷新记录（2026-09-10）

用新默认值 `CAUSE_RECON_WEIGHT=2.0` + `FIELD_DROPOUT=0.5` 重跑论文主表。
除这两个默认值外，协议、种子、K、数据划分全部保持不变，因此差异可归因到默认值本身。

## 表 A（私有变更单语料，K=30，5 种子）

| 模型 | 读出 | CV（旧 → 新） | NPMI（旧 → 新） |
| --- | --- | --- | --- |
| FC-NTM-Emb | beta | 0.536±0.025 → **0.541±0.027** | +0.098±0.037 → **+0.073±0.024** |
| FC-NTM-Emb | cTFIDF | 0.481±0.021 → **0.506±0.031** | +0.067±0.018 → **+0.079±0.030** |
| BTM | — | 0.470±0.013（未变） | +0.075±0.024（未变） |
| FC-NTM | cTFIDF | 0.449±0.017 → **0.470±0.021** | +0.003±0.022 → **+0.017±0.031** |
| NMF | — | 0.477±0.008（未变） | −0.045±0.008（未变） |
| ETM | — | 0.466±0.041（未变） | −0.084±0.050（未变） |
| FC-NTM | beta | 0.446±0.027 → 0.439±0.047 | −0.038±0.020 → −0.019±0.049 |
| LDA | — | 0.439±0.019（未变） | −0.158±0.019（未变） |
| CTMLite | — | 0.420±0.009（未变） | −0.075±0.008（未变） |

**新增对照行**：`FC-NTM-Emb(norm, L=1)` —— M3E 词向量按行归一化、解码温度 1.0
（即不做任何锐化）：

| 读出 | CV | NPMI |
| --- | --- | --- |
| beta | **0.461±0.035** | **−0.076±0.057** |
| cTFIDF | 0.443±0.051 | +0.004±0.036 |

### 解读
1. 新默认值让线性 FC-NTM 的 cTFIDF 读出明显改善：CV 0.449→0.470（追平 BTM），
   NPMI +0.003→+0.017。这是真实的、可解释的改进（跨字段重建权重提高 + 字段级缺失增强）。
2. FC-NTM-Emb 的 cTFIDF 读出也提升到 0.506 / +0.079。
3. **但"超过 BTM"只在未归一化词向量 + 温度 0.35 的锐化设置下成立**：
   归一化 + 温度 1.0 时 FC-NTM-Emb 是 0.461 / −0.076，仍然低于 BTM 的 0.470 / +0.075。
   锐化设置本身是可疑的（见 `verification_findings.md`），所以主表必须同时报告这一行。

## 产物
- `outputs/refresh_private_emb.csv`（M3E 未归一化，温度 0.35）
- `outputs/refresh_private_emb_norm.csv`（归一化，温度 1.0）
- `outputs/refresh_private_linear.csv`（线性解码器 + cTFIDF）
- `outputs/table_a_private_refreshed.csv`（合并后的表 A）
- `refresh_table_a.py`（合并脚本）

## 代码改动
- `config.py`：`CAUSE_RECON_WEIGHT=2.0`、新增 `FIELD_DROPOUT=0.5`
- `models.py`：`field_dropout` / `fusion_gate` / `fused_recon` 三个开关；
  训练循环数值恢复；重建损失 inf 防护
- `experiment.py` / `run_fcntm_train.py` / `run_fcntm_biterm.py`：
  接入 `field_dropout`，`run_fcntm_train.py` 另加 `--max-epochs` / `--lr` / `--recover-nan`

## 表 B（GitHub 公开语料）—— 进行中
`run_fcntm_train.py --corpus github --seeds 11,23,37 --k 30 --max-epochs 700 --lr 5e-4 --recover-nan`
输出到 `outputs/refresh_public_linear.csv`。基线行（BTM/NMF/ETM/LDA/CombinedTM 等）
不受默认值影响，沿用 `outputs/public_benchmark_summary_github.csv`。

⚠️ 公开语料需要 `--max-epochs 700 --lr 5e-4 --recover-nan` 才能稳定收敛
（默认 2e-3 会在 ~epoch 300 发散），因此公开表的 FC-NTM 行与基线之间存在
训练预算差异，需要在论文里说明。

### 表 B 进展（seed 11 已完成，seed 23/37 运行中）

| 模型 | 读出 | CV（旧 → 新，seed 11） | NPMI（旧 → 新，seed 11） |
| --- | --- | --- | --- |
| FC-NTM | beta | 0.617 → **0.696** | +0.160 → **+0.186** |
| FC-NTM | cTFIDF | 0.691 → 0.668 | +0.231 → +0.209 |
| BTM（基线，未变） | — | 0.706 | +0.230 |

新默认值让 FC-NTM 的 beta 读出在公开语料上明显改善（+0.079 CV），
但仍低于 BTM（0.696 vs 0.706）。
