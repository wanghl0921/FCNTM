# 题名归档：从交付检查点包重评的复现证据（2026-10-15）

本目录是应 2026-10-15 复审意见第 2 条运行的端到端复现证据：

1. **状态来源**：直接从 `FCNTM_检查点_20261014.zip` 解出
   `outputs/civil_missing_state.pt` 与 `outputs/rerun/concat_state_civil.pt`，
   SHA-256 记录在 `state_hash_check.txt`；两者与交付清单
   `outputs/RESULTS_MANIFEST.json` 及复审报告给出的预期哈希一致
   （concat_state_civil = `a706754efc7e7897...`）。
2. **运行方式**（隔离临时目录，未改动任何交付件）：

   ```
   python civil_title_archival.py --seeds 11,23,37 --top-labels 15 \
       --fcntm-state <extracted>/outputs/civil_missing_state.pt \
       --concat-state <extracted>/outputs/rerun/concat_state_civil.pt \
       --out <tmp>/pkg_verify_civil_title_archival.csv
   ```

   运行时间 2026-10-15 07:01 → 08:48（约 1 小时 47 分），无报错；
   完整日志见 `archival_pkg_verify.log`。
3. **结果对比**：`pkg_verify_civil_title_archival.csv`（本目录）与论文表 15
   的来源文件 `outputs/rerun/civil_title_archival.csv` 按
   (model, seed, setting) 对键比较，**18/18 行全部匹配**；
   Accuracy、Macro-F1、覆盖率的逐行最大绝对差为 **0.0**，
   `n_eval` 与映射回退计数也完全一致。Concat 三种子准确率为
   0.6488 / 0.6441 / 0.6663，与论文一致。

因此，"交付包内检查点 → 从头评分 → 论文题名归档表"的链条在本包内闭合。
