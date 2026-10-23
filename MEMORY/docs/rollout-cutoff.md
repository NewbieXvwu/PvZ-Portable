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

## 第 2 步：实现（Python 验收慢版本）

- task37 的 `vcpp` 分支不可依赖：没有 `origin/vcpp`，工作树有未提交改动，且 C++ 价值网前向有未消除的 CPU 回退。按任务书走 Python 侧验收慢版本（只量质量，不量速度）。
- `scripts/cutoff_agreement.py` 只在一次搜索期间替换 `search_play._rollout_policy` / `_score_terminal_or_value`，搜索代码本身不改。
- cutoff 臂：按 500/1000/1500/2000 顺序各跑一次引擎短推演（每个检查点一次往返），首次价值 v≥0.98 或 v≤0.02 即收手、叶值取该检查点；短推演先终局取 0/1；都不自信才跑满 2500、取叶值。
- 自检：day_l7 root 137/365/593/776 × seed 0–2 共 12 例，cutoff 值与第 1 步逐位一致——不自信 root 等于 2500 值（如 root 137 seed0 `0.765421`），自信 root 在 500 收手且等于 t=500 值（如 root 593 seed0 `0.990201`）；日志 `artifacts/task40/cutoff_check.log`。先试过“跑满 2500 + 引擎 trace 再在 Python 打分”的版本，值相同但慢 10–20 倍，弃用。

## 第 3 步：验收（预登记，正式跑）

- 三臂同 root、同未来：`exact`（2500）、`cutoff`（每 500 检查、θ=0.98）、`noise`（未来偏移 `+7,777,777`）；臂顺序在每个搜索点轮换。
- 参数：`samples 3`、`per-card 1`、`student-candidates 2`、`verify-samples 9`、halving、`horizon 2500`、`MARGIN 0.02`、`rollout-sim-skip 61`、`max-actions 3000`、`playthrough 2`；搜索点同 task8（有合法植物且学生想种植或距上次 ≥300 tick），三臂共用同一批决策点；学生打完整局（`dagger_beta=0`）。
- 局：4 族 × 任务清单 `seeds` 30000–30015（每族 16 局，共 64 局），2 workers：

| 族 | 清单 | task_id |
|---|---|---|
| day_l7 | `/home/newbiexvwu/PvZ-search/artifacts/stage4/day_l7_hard.json` | `day_l7_z20_eval` |
| night_l14 | `/home/newbiexvwu/PvZ-search/artifacts/stage4/night_panel_n10.json` | `night_l14_eval_n10` |
| pool_24 | `/home/newbiexvwu/PvZ-search/artifacts/stage4/pool_panel.json` | `pool_l24_eval_p9k` |
| original_1_10 | `/home/newbiexvwu/PvZ-search/.claude/worktrees/conveyor/artifacts/stage4/day_panel.json` | `day_l10_orig_eval` |

- 判据：`search_agreement_stats.agreement_intervals`，整局成组 bootstrap 20000 次、seed 20261019、margin 0.03；gate 要求 cutoff − noise 的一致率差 95% 区间下界 ≥ −0.03，且 64 局齐全；分族 16 局的一致率与区间只报告。

| 输入 | SHA256 |
|---|---|
| `scripts/cutoff_agreement.py` | `68e32842102ee4832f29a60378274b41cf4b2d716418bdb70726fd3ff8868dd2` |
| `scripts/search_play.py` | `d04eaa3643477c86f749df25515a395686b36819d85c69c4302777a45754c340` |
| `scripts/search_agreement_stats.py` | `5353107d891b8701bbef8d25e759bbada755f3fe932016b575db8820ec27a4bb` |
| `scripts/render_episode.py` | `e64f1c249f4946a9bb2da56bbbf926095e654488ebe44c4f551e272e3c278b75` |
| `python/pvz_env.py` | `2d53cd18b5390b1d5b1997bcbeb97a4ff069f962fbb94538f128936a8d568e69` |
| `python/pvz_event_env.py` | `027dfd9c0771b5b1a69de2e47f9500bf8dc61a778d97a03d4c9c578efa64fd89` |
| `artifacts/stage4/day_l7_hard.json` | `8b5f16704b3c8fe29b1981d01bd35d6df8b18c346a12cb4260c75b59bdd2a941` |
| `artifacts/stage4/night_panel_n10.json` | `f62258d7c0d254a47999120ba2456a8c02f7b7b8fbea48e42c0ef8a9b91d4ed1` |
| `artifacts/stage4/pool_panel.json` | `10ebc1b404a15aa96e44e705747c99fdb6b3aaf90a9d930ac9f51f61cda1b0fb` |
| conveyor `day_panel.json` | `01e3ae4f437b7673b3278280e3891bfcb0f65e92ba8bddaeb7b6321713fcdc27` |
| 学生 / 价值网 / 分叉网 / 引擎 | 同第 1 步表（`122703…` / `5171db…` / `248584…` / `7508e8…`） |

```bash
PVZ_NO_POPUP=1 ~/.venvs/ml/bin/python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/cutoff/scripts/cutoff_agreement.py \
  --workers 2 \
  --executable /home/newbiexvwu/PvZ-search/.claude/worktrees/cutoff/artifacts/task40/engine/pvz-portable
# -> artifacts/task40/agreement.json、artifacts/task40/agreement.log
```

> 引擎必须用本工作树冻结的 `artifacts/task40/engine/pvz-portable`（`7508e8…`，BuildNum 2101）：共享路径 `cq/build/pvz-portable` 已于 10:55 被重编为 `b64261…`。跑前按 SHA 校验，别只信路径。

> 2026-10-23 11:02 首跑在第 6 局因引擎瞬时退出（`SimulatorExited` → `BrokenPipe`）整池失败，前 5 局明细未落盘（汇总见 `artifacts/task40/agreement_run1_crash.log`）。11:09 起加固为单局重试（`--retries 2`）+ 每局落盘 `artifacts/task40/agreement_parts/` + 断点续跑；三臂、种子、参数、判据均不变。上表脚本 SHA 为新版本。

> 11:10 第二次跑被 12:00 WSL 重启打断（0 局、仅 0 字节残件，日志 `agreement_run2_wsl_restart.log`）；12:17 第三次跑被 12:20 前后的第二次重启打断（日志止于预热、0 part，`agreement_run3_wsl_restart.log`）。两次都不进入统计。12:34 第四次跑用冻结引擎、同样参数与断点续跑启动；完成的局以 `agreement_parts/<family>_<seed>.json` 为准。

## 第 4 步：强度关（预登记 2026-10-23 18:45，质量关通过后才跑）

- 目的：确认收手版不降低老师的实际胜率，而不只是决策一致。`beta=1`：老师（搜索）执行自己的决策；学生只提供 fast 动作来触发表搜索，与生产 `search_play` 的 `fast_search` 循环一致（`search_play.py:302-310`：有合法植物且学生想种植或距上次 ≥300 tick 才搜索）。
- 两臂：`exact`（生产搜索、视界 2500）与 `cutoff`（θ=0.98、检查点 500/1000/1500/2000、首次自信收手；与第 2 步同一个 Python 慢版本）。每局同种子、同 `decision_seed = seed*10000+decisions`，各执行本臂决策；两臂轨迹会分叉，按种子配对（不共用一局）。
- 族与种子：同第 3 步 4 族，每族 `seeds` 30000–30063 共 64 局（4×64 = 256 对，512 次对局），2 workers。
- 判据：按种子配对的胜负差（cutoff − exact），整局（种子）成组 bootstrap 20000 次、seed 20261019；**95% 区间下界 ≥ −3pp** 且 256 对齐全；分族 64 对只报告。脚本 `scripts/cutoff_strength.py`（静态检查通过，尚未实战）。
- 成本实测估计：历史 beta=1 老师跑（`artifacts/day_panel/teacher_l7_d7_64`，64 局）决策 1,054/局、实际搜索 203/局；第 3 步 33 局 part 实测 exact 0.96 s/搜索、cutoff 2.40 s/搜索（qc 满载下）。按搜索点估 4 族 × 64 seed 约 60–65 h 游戏时间，2 workers ≈ 30–33 h 墙钟。
- 输入 SHA256：`scripts/cutoff_strength.py` `04a190e24e7efbde23d3aca8490a6be341dbcf9d7b87624eb65d1bcd04a4cfcb`；学生/价值网/分叉网/引擎/清单同第 3 步表（引擎 `7508e8…`）。实现与生产 `search_play` 的 beta=1 循环逐点对齐，包括搜索后写 `fast_state["action"] = 执行动作`（下一帧特征与搜索触发门都读到执行动作）。

```bash
# 校准（质量关结束后先跑 1 个种子 = 2 局，量单局耗时；临时 parts 不污染正式结果）
PVZ_NO_POPUP=1 ~/.venvs/ml/bin/python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/cutoff/scripts/cutoff_strength.py \
  --families day_l7 --seeds 1 --workers 1 \
  --parts /tmp/strength_calib_parts --out /tmp/strength_calib.json \
  --executable /home/newbiexvwu/PvZ-search/.claude/worktrees/cutoff/artifacts/task40/engine/pvz-portable
# 正式（2 个游戏进程）
PVZ_NO_POPUP=1 ~/.venvs/ml/bin/python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/cutoff/scripts/cutoff_strength.py \
  --workers 2 \
  --executable /home/newbiexvwu/PvZ-search/.claude/worktrees/cutoff/artifacts/task40/engine/pvz-portable
# -> artifacts/task40/strength.json、artifacts/task40/strength_parts/
```

## 结果（第 3 步：质量关，2026-10-23 21:00）

第四次正式跑 12:34 启动、20:54 收尾（`cutoff_agreement.py --workers 2`，冻结引擎 `7508e8…`），**64/64 局齐全、`errors=[]`、全程无引擎崩溃/无重启**；14,720 个搜索点，墙钟 29,931.2 s。`agreement.json` SHA256 `9ee9d700b735be3ec4386234de554c4eaa19c12c65c2335cc2b4b6b165ba7039`。

整局成组 bootstrap（20000 次、seed 20261019、margin 0.03，`minimum_games=64`）：

| 口径 | cutoff 一致率 | noise 一致率 | 差（cutoff−noise） | 95% 区间 | 门禁 |
|---|---|---|---|---|---|
| 64 局汇总 | 0.98376 [0.97813, 0.98862] | 0.78505 [0.75433, 0.81964] | **+0.19871** | **[0.16207, 0.23131]** | **通过**（下界 ≥ −0.03） |
| day_l7 (16) | 0.96087 | 0.82337 | +0.13750 | [0.09993, 0.17873] | 只报告 |
| night_l14 (16) | 0.99297 | 0.68062 | +0.31235 | [0.28865, 0.33392] | 只报告 |
| pool_24 (16) | 0.99941 | 0.76404 | +0.23536 | [0.21238, 0.26090] | 只报告 |
| original_1_10 (16) | 0.98421 | 0.96479 | +0.01941 | [0.00861, 0.03278] | 只报告 |

- 胜局 36/64：day 8/16、night 12/16、pool 0/16、original 16/16。pool 全负是该族既有难度（搜索策略本就不胜），与收手无关——该族 cutoff 与 exact 的一致率 0.9994。
- cutoff 臂 293,283 条推演的收手分布：≤500 tick 前终局 2.6%、500 收手 23.4%、1000 收手 1.5%、1500 1.0%、2000 1.0%、跑满 2500 的 70.6%。与第 1 步同向：day 与传送带大量早停，night 与 pool 多数跑满。
- 搜索耗时（Python 慢版本，**只作诊断、不作门槛**）：cutoff/exact = 2.03×（day 1.92×、night 2.93×、pool 2.95×、original 0.50×），noise/exact = 1.00×；exact 14,718 s、cutoff 29,916 s。慢在每检查点一次引擎往返；引擎内实现才有收益，所以速度不作为本关判据。
- 结论：质量关通过。按 18:45 预登记继续第 4 步强度关（先 1 种子校准，再 2 workers 正式跑）。

