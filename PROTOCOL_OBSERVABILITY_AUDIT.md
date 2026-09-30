# 协议与可观测性审计

日期：2026-09-30
分支：`pvz-env`
测量环境：macOS arm64，`build/pvz-portable`，资源目录
`~/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN`，解释器
`/Users/newbiexvwu/.local/share/mise/installs/python/3.14/bin/python3`。

这份文档回答三个问题，全部用实测数据，不用推测：

1. **很多数据真的有必要存或者传吗？**
2. **真正便于调试、观察和测试的数据，真的存下来了吗？**
3. **项目跑的时候能看到任何动态吗，还是只能死等？**

---

## 0. 三句话结论

| 问题 | 结论 |
|---|---|
| 传的数据有必要吗 | **字段几乎都有必要**（未读字段只占 **1.0%**），**但形式极贵**：`legal_actions` 里 **78%** 是重复键名，`cells` 里 54 个空格子花掉 4,375 B，`loadout_context`/`player_profile`/`defenses` 每步重传**完全不变**的 669 B。观测 12,163 B 里约 **一半是表示开销**。 |
| 便于调试的数据存了吗 | **训练器把最有用的两类数据算了就扔**：每局的 `profile_seconds`（耗时分解）从不聚合；每局的 `env.episode`（完整可回放操作序列）从不落盘。反而每 update 把 **70.3 KiB 的 episode digest** 写进 `training_state.json`（`training_state.json` 单次 102 KiB，**69% 是哈希**），而 checkpoint 里已经有一份。 |
| 跑的时候能看见动态吗 | **不能。** 一个 update 约 3 分钟，其中 **184 s（92%）的 PPO 更新完全静默**，1,280 局的评估也完全静默。整个仓库没有 tensorboard / wandb / tqdm / rich / matplotlib，只有**每个 update 一行 `print`**，而且那行的 `wins=0/2000` 在早期恒为 0。 |

---

## 1. 测量方法

新增探针 `scripts/protocol_payload_probe.py`。它把 `PvZEnv._read_message`
（每个响应唯一的必经之路）子类化，记录每行的字节数与解码耗时，因此
"传输/解码"和"模拟"可以分开定价。

```bash
PY=/Users/newbiexvwu/.local/share/mise/installs/python/3.14/bin/python3
$PY scripts/protocol_payload_probe.py --decisions 120 --privileged --field-budget 200
```

`--field-budget N` 会采样 N 个真实观测，对每个顶层字段计价，并用正则扫描
`python/*.py` 与 `scripts/*.py`（排除测试与 `pvz_env.py` 自身）里的
`observation["x"]` / `observation.get("x")` 读取点，标出**没有任何读取者**的字段。

**不要把长跑探针用 `| tail -N` 包住**——会丢流式输出。

---

## 2. 传了什么：观测字段预算

200 个真实观测（每步 `WAIT 30`），平均 **12,163 B**：

| 字段 | 字节 | 占比 | 有读取者 |
|---|---:|---:|---|
| `legal_actions` | 5,309.2 | **43.6%** | 是 |
| `cells` | 4,375.0 | **36.0%** | 是 |
| `zombies` | 608.5 | 5.0% | 是 |
| `packets` | 581.4 | 4.8% | 是 |
| `defenses` | 231.0 | 1.9% | 是 |
| `loadout_context` | 230.0 | 1.9% | 是 |
| `player_profile` | 208.0 | 1.7% | 是 |
| `grid` | 121.0 | 1.0% | **无** |
| 其余 24 个字段合计 | ~110 | ~0.9% | 是 |
| `coins` / `playthrough` / 3 个版本号 | 5.0 | 0.0% | **无** |
| **未被读取的字段合计** | **127.0** | **1.0%** | |

**结论：不存在"传了一堆没人要的字段"这种问题。** `grid`（121 B）是唯一
值得注意的死字段——它是 5×9 的 `mGridSquareType` 矩阵，而 `cells` 里每格
已经带了 `terrain`。

真正的问题在下一节。

---

## 3. 传输的形式成本：数据没错，拼写太贵

### 3.1 `legal_actions`：78% 是键名

`legal_actions` 是 `{"plants": [...], "shovels": [...], "wait": bool}`。
`plants` 有 **90 个条目**，每条形如：

```json
{"packet":0,"col":0,"row":0}
```

**29 B 里 22 B 是键名。**

| 编码 | 字节 | 相对现状 |
|---|---:|---:|
| 现状 `[{"packet":0,"col":0,"row":0}, ...]` | 2,611 | 100% |
| 行式 `[[0,0,0], ...]` | 721 | 27.6% |
| 列式 `{"packet":[...],"col":[...],"row":[...]}` | **568** | **21.8%** |
| 信息量下限（3 字节/条 × 90） | 270 | 10.3% |

列式省 **2,043 B = 整个观测的 16.8%**，而且**不丢任何信息**——模型侧
`legal_summary()` 本来就要把它展开成 mask。

### 3.2 `cells`：54 个空格子，每格 80 B

54 格（`MAX_GRID_SIZE_X * MAX_GRID_SIZE_Y`）**全部为空**，仍然花 4,375 B。
一格的原文：

```json
{"row":0,"col":0,"terrain":1,"row_type":1,"plant_types":[],"grid_item_types":[]}
```

**80 B 承载 6 个小整数**，其中键名 `row`/`col`/`terrain`/`row_type`/
`plant_types`/`grid_item_types` 占 55 字符。`row` 和 `col` 还是**数组下标**，
纯冗余。

按 `terrain` + `row_type` 的二维表 + 稀疏的非空格子列表编码，同样信息
约 **400 B**，省 **~4,000 B = 观测的 33%**。

### 3.3 每步重传完全不变的数据：669 B（5.5%）

`loadout_context`（230 B）、`player_profile`（208 B）、`defenses`（231 B）
在 `reset` 之后就固定了，但**每个 decision 重传一次**。`defenses` 里
`x`/`y` 还是从 `row` 推出来的（`y = 103 + 100*row`），属于派生数据显式传输。

### 3.4 `zombies`：23 个字段，静态属性混在动态里

每只僵尸 356 B、23 个字段。其中 `body_max_health`、`helm_max_health`、
`shield_max_health`、`type` 是**按僵尸种类固定**的，可以查表；每步真正变化的
只有 `x`/`y`/`body_health`/`phase` 等少数几个。

### 3.5 合计可省多少

| 项 | 可省字节 | 占观测 |
|---|---:|---:|
| `legal_actions` 改列式 | ~2,040 | 16.8% |
| `cells` 改稀疏 + 去下标 | ~3,975 | 32.7% |
| 静态字段移出每步观测 | ~669 | 5.5% |
| `grid` 删除 | 121 | 1.0% |
| **合计** | **~6,800** | **~56%** |

**但请注意第 4 节：这不会带来同比例的加速。**

---

## 4. 每次决策的往返账

`env.step` 一次 = 一条 `PLANT`/`SHOVEL`/`WAIT` 命令 + 一条完整观测响应。
`CRITIC_INPUTS` 是**每个决策额外的一次同步往返**。

120 个决策实测：

| 项 | 耗时 | 占一个决策 |
|---|---:|---:|
| `env.step`（含模拟 + 序列化 + 传输 + 解码） | 0.108 ms | 90.4% |
| ├─ 其中 `json.loads` | 0.037 ms | **31.1%** |
| `env.critic_inputs` | 0.012 ms | 9.6% |
| `_canonical_events` | 0.001 ms | 0.6% |

**`json.loads` 是 `env.step` 里最大的单项（34%），也是整个决策的 31%。**
（`WAIT 1` 下；载荷变大时升到 **43%**。）

### 4.1 `CRITIC_INPUTS` 有一半是纯冗余——已证实

```
critic_inputs['wave_timer'] == observation['wave_timer']:  120/120 decisions
```

`CRITIC_INPUTS` 返回 `{"wave_timer": mZombieCountDown, "wave_zombies": [...]}`
（`src/main.cpp:534`），而**公开观测里已经有同一个 `wave_timer`**
（`src/LawnApp.cpp:1524`）。更讽刺的是：

- `pvz_agent_model.py:190` 用 `observation["wave_timer"] / 6000` 算
  `next_wave_distance` 喂给 actor；
- `pvz_agent_model.py:950` 用 `critic_inputs["wave_timer"] / 6000` 算
  `privileged_extra[0]` 喂给 critic。

**同一个数被取两次、算两遍、喂给两个头。** `CRITIC_INPUTS` 唯一不可替代的
信息只有 `wave_zombies`（当前波僵尸名单）。

### 4.2 但环境不是瓶颈

单 worker / 单线程 CPU，每局 0.5128 s（`artifacts/t5/perf/worker_single_core_current.json`）：

| 项 | 每局 | 占比 |
|---|---:|---:|
| `model` | 0.4373 s | **85.3%** |
| `environment` | 0.0529 s | 10.3% |
| `tokenization` | 0.0179 s | 3.5% |
| `persistence` | 0.0130 s | 2.5% |
| `critic_inputs` | 0.0038 s | 0.75% |

把观测缩小一半，只能让 `environment` 降约 20%，即**每局省 ~2%**。
**协议瘦身是清洁度收益，不是性能收益。** 真正的墙是 `model`。

（例外：GPU 上 `model` 会大幅缩小，`environment` 的占比会上升——那时这项
才变成性能问题。届时应重新测这张表。）

---

## 5. `PRIV`：协议里最大的一块死重

`PRIV` 响应 **57,907 B**，其中：

| 项 | 字节 | 占比 |
|---|---:|---:|
| `hidden.reanimations` | 39,737 | **68.6%** |
| `hidden.rand_state_hex` | 5,002 | 8.6% |
| `hidden.zombies_in_wave` | 253 | 0.4% |
| 其余 | ~12,915 | 22.3% |

`reanimations` 是**整条动画系统**（每轨道的 `attachment`/`blend_counter`/
`blend_time`/`render_group`），`rand_state_hex` 是 RNG 状态。

**训练路径一次都不发 `PRIV`。** `debug_replay` 默认 `False`
（`pvz_env.py:178`），`privileged_state()` 只在 `debug_replay=True`
（`verify_env_equivalence.py`）和三个基准脚本
（`test_agent_model.py:264`、`trajectory_storage_benchmark.py:54`、
`branch_benchmark.py:86`、`snapshot_hash_compare.py:69`）里被调用。

代价实测 **0.229 ms/次**（含 57.9 KiB 序列化 + 传输 + 解码）。它是
`verify_env_equivalence.py` 的**证据基础**，所以不能删——但它是"协议里
存在一个巨大的、几乎无人使用的命令"的典型。

---

## 6. 存了什么：落盘产物清单

| 产物 | 时机 | 内容 | 问题 |
|---|---|---|---|
| `{output_dir}/training_state.json` | **每 update** | `runs` / `evaluations` / `learning_curve` / `recent_passes` / `last_training_debug` / `last_update` | **单次 ~102 KiB，69% 是 2,000 个 episode digest**；`atomic_json(indent=2)` 未压缩；`evaluations` 与 `learning_curve` 累积 |
| `{output_dir}/learning_curve.json` | 每次评估（每 5,000 局或每 30 分钟） | 损失 + 三套 pass rate + 每任务滚动 64 胜率 | 只存**损失与评估**，**不存任何进程统计**（时间/吞吐/RSS） |
| `runs/run_N/gameplay_model_v1_ppo.pt` | **每 update** | `state_dict` + `provenance`（含同一批 digest）+ `losses` | 全量重写；`provenance.trajectory_sha256` 与 `training_state.json` 里的 digest **重复** |
| `runs/run_N/.seed_jobs/update_NNNN/<sha>/seed_*.npz` | 每局 | 完整 episode（tokens/legal/critic_extra/reward） | 可复现的基础，保留（见 `TRAINING_HOTSPOT_ANALYSIS.md` §10.8.2） |
| `{output_dir}/evaluations/heldout_NNNNNNN.json.gz` | 每次评估 | 每 seed 的完整评估记录 | 合理 |
| `gates/T5.json` | run 结束时 | 门禁文档 | 合理 |

**没有 `training_state.json` 也没有 `learning_curve.json` 存在于仓库里**——
训练从未跑完过（GPU 掉线），所以这些格式**从未被真实数据检验过**。

### 6.1 算了就扔的两类数据

1. **`profile_seconds`**：`collect_task_episode`（`train_pvz_ppo.py:202`）
   **每局**都返回
   `{"model", "environment", "critic_inputs", "tokenization"}` 四个耗时，
   而 T5 训练器**从不聚合它**。只有 `t5_throughput.py`、
   `t5_worker_batch_benchmark.py`、`training_hotspot_profile.py` 三个
   **离线脚本**在用。也就是说：训练时每局都量了体温，然后把体温计扔了。

2. **`env.episode`**：`pvz_env.py:711` 的 `_record_operation` 在**每个
   decision** 都往 `self.episode` 追加一条
   `{request, action, ticks_advanced, state}`，并在 `reset` 时写入
   `initial_state`。这是**唯一能离线回放/可视化的数据**。但
   `collect_task_episode`（T5 训练路径）**不调 `save_replay`**——
   只有旧的 `collect_episode`（`train_pvz_ppo.py:83`）、
   `train_pvz_agent.py`、`benchmark_pvz_agent.py` 会存。

   **每局都在累积完整回放，然后随 episode dict 一起被垃圾回收。**

### 6.2 写得太重的一项

`training_state.json` 每 update 全量重写 102 KiB，其中
**70.3 KiB（69%）是 `last_update.losses.rollout_episode_hashes`**——
2,000 个 32-hex 的 `episode_digest`。它的唯一用途是溯源，而
`checkpoint.provenance.trajectory_sha256` **已经存了同一批值**。

时间成本可忽略（<1 KB/s），但**信噪比很差**：想看一眼损失曲线，得从
70 KiB 的哈希里把它挖出来。

---

## 7. 跑的时候能看见什么

### 7.1 主循环只有一个 `print`

`train_pvz_ppo_task_family.py:1079`，每个 update 一行：

```
run=1 update=1 episodes=2000 wins=0/2000 policy_loss=... value_loss=... entropy=...
```

**`wins=0/2000` 在训练早期恒为 0，这是零信息量的一行。**

### 7.2 各阶段的可见性

| 阶段 | 时长 | 输出 |
|---|---|---|
| rollout（2,000 局，18 workers） | ~11 s | `run_seed_jobs` 每 16 局一行 `T5 run 1 update 1 2000/2000` |
| **PPO 更新（GPU）** | **~184 s** | **完全静默** |
| 评估（1,280 局，多 worker） | 数十秒 | **完全静默**（`_run_evaluation_jobs` 无进度打印） |
| 保存 checkpoint + state | 若干 ms | 无 |

一个 update ≈ 3 分钟，**约 92% 的时间没有任何输出**。

### 7.3 已经算好但没有出口的数据

- `state["recent_passes"]`：**每 update 落盘**，含**每个任务的滚动 64 局胜率**。
  这正是早期唯一有信号的指标（`wins=0/2000` 不是），但它**只进文件，
  不进 stdout**，而且只在 `stop_reason == "stage0_no_signal"` 的诊断块里
  被引用一次（`_curve_row` 在评估时才把它写进 `learning_curve`）。
- `losses["mean_abs_shaping_reward_recent"]`、
  `losses["terminal_outcome_mean_recent"]`：都在 `last_training_debug` 里，
  但不在 `print` 行里。
- `TASK_RECENT` 已经在驱动采样权重（`_sampling_weights`），所以它是**活的
  信号**，只是不可见。

### 7.4 没有的东西

- 无 tensorboard / wandb / SummaryWriter / tqdm / rich / matplotlib
  （`requirements.txt` 只有 `torch` + `numpy`）
- 无进度条、无 ETA、无 GPU 利用率、无 RSS
- 无"实时"通道：想观察只能 `tail -f training_state.json`，而它每 3 分钟才变

---

## 8. 改造方案

分两档。**A 档不改协议语义、不改任何哈希、不影响已有证据**；**B 档会改变
协议版本或摘要，必须等到明确节点**。

### A 档：可立即做（建议）

| # | 改动 | 位置 | 收益 |
|---|---|---|---|
| A1 | 把每局的 `profile_seconds` 聚合进 `last_training_debug`（model / environment / critic_inputs / tokenization 的均值），并把 rollout 与 update 的墙钟耗时也记进去 | `train_pvz_ppo_task_family.py:962` | 每 update 免费得到一张体检表；回答"为什么变慢了" |
| A2 | `print` 行换成有信号的：每任务滚动胜率（min/median/max）、`terminal_outcome_mean_recent`、`mean_abs_shaping_reward_recent`、update 耗时、episodes/hour | 同上 1079 | 把 `wins=0/2000` 换成早期唯一有变化的量 |
| A3 | `training_state.json` 里**删掉 `last_update.losses.rollout_episode_hashes`**（checkpoint 的 `provenance` 已有同一批），或改成单独文件 | 同上 961 | 单次 102 KiB → ~32 KiB |
| A4 | `_run_evaluation_jobs` 加进度打印（每 64 局一行） | 同上 255 | 消灭 1,280 局的静默段 |
| A5 | `train_update` 加每 epoch 一行（`ppo_epochs` 通常 3–4） | `train_pvz_ppo.py:229` | 把 184 s 切成 3–4 段可见 |
| A6 | 记录 `_state_sha256` 与 `torch.save` 的耗时（现在每 update 都跑但**耗时未记录**） | 同上 906 / 1078 | 补上 F 段缺的两个数 |
| A7 | `learning_curve.json` 增补进程统计（update 耗时、episodes/hour、RSS） | 同上 `_curve_row` 426 | 让曲线带上成本 |
| A8 | 新增一个**只读**的 `scripts/watch_training.py`：`tail` `training_state.json` + `learning_curve.json`，在终端画滚动胜率与损失 | 新文件 | 不碰训练代码就能"看见动态" |

A1–A7 全是**纯增量的观测改造**，不动协议、不动数值、不产生新的哈希。

### B 档：需要动协议（等到明确节点）

| # | 改动 | 代价 | 前置 |
|---|---|---|---|
| B1 | `CRITIC_INPUTS` 只返回 `wave_zombies`，`wave_timer` 从 `observation` 取 | 改 `src/main.cpp:534` + `pvz_agent_model.py:948`；`test_agent_model.py:264` 的对照测试要改 | 无——**可以独立做，收益 9.6% 的决策时间** |
| B2 | `legal_actions` 改列式编码 | 改 `src/LawnApp.cpp` + `legal_summary()`；省 16.8% 观测字节 | 需要 `OBSERVATION_VERSION` 递增 |
| B3 | `cells` 改稀疏 + 去 `row`/`col` | 省 32.7% 观测字节 | 同上 |
| B4 | `loadout_context`/`player_profile` 移出每步观测，只在 `reset` 返回 | 省 5.5% | 同上 |
| B5 | 删 `grid` | 省 1.0% | 同上 |
| B6 | `PRIV` 拆成 `PRIV_CORE`（wave_timer/zombies_in_wave/rand_state）与 `PRIV_ANIM`（reanimations） | `verify_env_equivalence` 改调两个 | 无——但 `reanimations` 是那套证据的基础，拆分而非删除 |

**B1 是唯一"改了就能直接省时间"的一项，而且不需要动 `OBSERVATION_VERSION`**，
因为它只改 `CRITIC_INPUTS` 的响应体。

### 8.1 关于 `env.episode`

如果要回答"便于调试、观察和测试的数据存了吗"，最直接的答案是**存 `env.episode`**：
它每局已经算好了，落盘成本是每局约 3–30 KB（取决于决策数），
比 `profile_seconds` 更值得。可以先只对**失败的局**存
（`won == False` 且 `wave` 达到某个阈值），把体积压到可接受范围。

---

## 9. 结论

1. **"传了没用的字段"是个错觉**——未读字段只有 1.0%。但"用了最贵的形式"
   是真的：**约 56% 的观测字节是表示开销**（重复键名、数组下标、
   每步重传的静态数据）。这是清洁度问题，不是性能问题。
2. **"存了没用的东西"是真的**：每 update 102 KiB 的 `training_state.json`
   里 69% 是重复的哈希；而**每局都算了、却从不落盘的 `profile_seconds`
   与 `env.episode`** 才是真正该存的。
3. **"跑的时候只能死等"是真的**：一个 update 92% 的时间静默，
   `wins=0/2000` 零信息量，而**每任务滚动胜率这个唯一有信号量已经算好了
   但只进文件、不进终端**。
4. **改的顺序应该是先 A 档再 B 档**：A 档让你**看得见**，然后才知道
   B 档值不值得做。在看不见的情况下重设计协议，是在猜。

---

## 10. 复现命令

```bash
PY=/Users/newbiexvwu/.local/share/mise/installs/python/3.14/bin/python3

# 每次决策的往返账 + 观测字段预算 + PRIV 计价
$PY scripts/protocol_payload_probe.py --decisions 120 --privileged --field-budget 200

# 每局耗时分解（单 worker 单线程基线）
$PY scripts/t5_throughput.py --resource-dir ~/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN \
    --minutes 1 --output /tmp/throughput.json

# 全流程必要性审计（内存 / 编解码 / 分片）
$PY scripts/pipeline_necessity_audit.py --episodes 4
```
