# task40 报告：推演“胜负已定就收手”（confident cutoff）

任务书：`MEMORY/docs/task40-confident-cutoff.md`。本文件记结果、命令、种子与输入 SHA；任务书末尾“结果”有逐族全表。

## 第 1 步：能省多少（只读测量）

### 口径

- root：`/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2` 四族真实搜索点，每族 16 档 × 4 个决策点（帧数 15/40/65/85%）= 64 root。
  - day L7：seeds 1800000–1800015，卡组 `0,1,2,3,4,5,7`
  - night L14：seeds 1930000–1930015，卡组 `0,1,2,3,4,5,7,8,9`（存档实际卡组；无墓碑吞噬者 11）
  - pool L24：seeds 2030000–2030015，卡组 `0,1,2,3,4,5,7,16,19`
  - 原版传送带 L10：seeds 1880258–1880273，无卡组
- 每 root 9 个未来 seed；检查点 t ∈ {500,1000,1500,2000} 与终点 2500；共 11520 条推演。
- 学生 `dw4i_s1`，`--rollout-sim-skip 61`，`MAX_ACTIONS=3000`；叶值用 `value_net/v3`（`search_play._leaf_values` 同一套 featurize + context + sigmoid），终端叶按 `_score_terminal_or_value` 取 0/1。
- 前缀校验：同 root 同 seed，t 推演的引擎 trace 与 2500 推演逐决策行一致 32/32；t 叶与 2500 推演“下一决策”观测一致 32/32（按决策序号配对，不按 tick）。

### 输入 SHA256

| 输入 | SHA256 |
|---|---|
| `artifacts/student/dw4i_s1/model.pt` | `122703803804ab7e1e5aeda4ec382420d45acc349d74fe85344b859a8d760375` |
| `artifacts/value_net/v3/model.pt` | `5171db54a31ab1795ce3cc136a25684a5bf6237cfd0e49a3b3380956dc26374a` |
| `artifacts/hard/value_branch1/model.pt` | `2485841036cf3a8738174affb0dadc0201a925ca3ea0ebe6e3eb02e74621e314` |
| 引擎 `.claude/worktrees/cq/build/pvz-portable` | `7508e840ec19d0ef6f91d55f4052cf17846a82df7eb24e235fab2aa9b1aefe65` |
| root 清单 `artifacts/task40/roots.json` | `7ef100b49de30be214dc8421451db4ecc87a8a3bf7867fb2e72254de7b06d3cf` |
| 推演结果 `artifacts/task40/rollouts.jsonl` | `5592ca5f556c732691bf210ee18898e1d2c8d1d33e6e08770032a57c251243ed` |
| 汇总 `artifacts/task40/summary.json` | `a7c711a9cb225cfbc8a5b5bc1af40419b14c125bc2eeaa2a346766e11485d0b1` |

### 命令

```bash
cd /home/newbiexvwu/PvZ-search/.claude/worktrees/cutoff
~/.venvs/ml/bin/python3 scripts/confident_cutoff.py manifest --scan-seeds 100000
~/.venvs/ml/bin/python3 scripts/confident_cutoff.py prefix
~/.venvs/ml/bin/python3 scripts/confident_cutoff.py measure
~/.venvs/ml/bin/python3 scripts/confident_cutoff.py report
```

## 结果（第 1 步）

- 64 格（4 t × 4 θ，每族与全族）：**翻转率全部 0.0%**；同 root 同 seed 的 (t, 2500) 配对里没有一对胜负分类改变。
- 自然的实现口径（每 500 tick 查一次，第一次自信就停）四族加权：

| θ | 自信停止占比 | 省 tick（全部） | day | night | 传送带 | pool | 翻转 | abs差 p95 |
|---|---|---|---|---|---|---|---|---|
| 0.90 | 59.9% | 46.1% | 58.1% | 32.9% | 78.3% | 15.1% | 0 | 0.0466 |
| 0.95 | 52.3% | 39.9% | 56.7% | 20.2% | 76.8% | 5.7% | 0 | 0.0297 |
| **0.98** | **44.1%** | **31.9%** | **45.7%** | **7.2%** | **71.2%** | **3.6%** | **0** | **0.0124** |
| 0.99 | 35.4% | 26.0% | 38.5% | 1.0% | 64.4% | 0.3% | 0 | 0.0071 |

- **建议实现参数：每 500 tick 检查，θ=0.98**。全族省 31.9% tick（≥15% 门槛），价值扰动 p95 0.0124 小于老师的覆盖余量 MARGIN=0.02，零翻转。θ=0.95 是激进备选（39.9%，p95 0.0297 已高于 MARGIN）；θ=0.99 最保守（26.0%）。
- 省 tick 主要来自 day 与传送带；night 与 pool 在 θ=0.98 只省 7.2% / 3.6%，即“输赢未定就不砍”的推演基本跑满。

逐族、逐检查点全表见 `MEMORY/docs/task40-confident-cutoff.md` 的“结果”节。

## 第 2 步：实现（待续）

- task37 的 `vcpp` 分支不可依赖：没有 `origin/vcpp`，工作树有未提交改动，且 C++ 价值网前向有未消除的 CPU 回退。按任务书走 Python 侧验收慢版本（只量质量，不量速度）。
- 验收设计（待预登记后执行）：同 root 同 decision_seed 跑三条臂——原版（2500）、收手版（检查点每 500、θ=0.98）、原版换随机未来（噪声底线）；`scripts/search_agreement_stats.py` 的整局成组 bootstrap 口径。
