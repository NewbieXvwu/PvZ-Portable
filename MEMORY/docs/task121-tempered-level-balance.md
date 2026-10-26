# task121：关卡再平衡改为温和版（幂次 + 倍数上限）

主控 2026-10-27 派给 Mac CodeBuddy（cbp-task121）。分支 `rebalance2`，基于 `origin/rebalance`（7592889a，deploy14g 在役代码）。

## 问题

deploy14g 的 `source_weight.level_balance`（`python/pvz_training_rows.py` 约 30–67 行）按行数**完全反比**加权，结果各关有效份额几乎相同（第 21 轮：L7 948 万行，份额 2.0%、倍数 0.035；L40 只有 276 行，倍数 1190，份额 2.3%）。几百行的关被放大上千倍，梯度被极少样本主导，噪声大。

## 要做的

1. 配方加两个可选参数（缺省时行为与现在逐位相同）：
   - `source_weight.level_balance_power`：p ∈ [0,1]，质量 = 原权重 × inverse_rows^p。p=1 等于现行，p=0 等于不再平衡。
   - `source_weight.level_balance_max_multiplier`：每关训练行倍数的上限（超出部分不再分配，总训练权重可以因此略小于原值，日志记下）。
   - 25% 封顶逻辑保留。
2. 新配方 `recipes/dw4i2_rebalanced_p05.json`：在 `dw4i2_rebalanced.json` 基础上 `level_balance_power: 0.5`、`level_balance_max_multiplier: 20`。不改旧配方文件。
3. 单元测试：缺省逐位不变；p=0.5 时用第 21 轮 `level_shares.json` 的行数（在台式机 `~/pvz-logs/curriculum_state.pipeline/round_000021/seeds/seed_000002/student/level_shares.json`，可 scp 一份到测试夹具，几 KB）算出份额表，写进报告。
4. 只跑相关测试（AGENTS §6.5）。不写部署脚本，不动台式机 qc。推送分支，结果写在本文件末尾。
