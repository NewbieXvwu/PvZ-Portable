# T5 彻夜工作令（阶段 0 → 阶段 1）

> **本文件是给执行 Agent 的指令，优先级高于 `TODO.md` §4 的一般描述。冲突时以本文件为准。**
> 适用机器：台式机 WSL（i7-12700F + RTX 5080）。起点 HEAD：`514340a` 或更新。
> 读取顺序：本文件 → `TODO.md` §1 铁律 / §2 门禁规则 / §4 T5 / §5.2 缺口说明 → `MEMORY_BUDGET.md` §8。

---

## 0. 目标

补完阶段 0 的三个缺口（§3）→ 跑阶段 0 学习信号验证门（§4）→ **只有门通过**才进阶段 1 正式训练（§5）。

**本次无人值守的边界**：阶段 1 跑到门禁通过或触发硬停止条件后，**立即停止并等待确认，不得开始 T6**。

---

## 1. 红线（违反任一条 = 本次任务失败）

| 编号 | 红线 |
|---|---|
| **R1** | 禁止修改门禁、评估器（`t4_capability_profile.py`）、任务定义（`artifacts/task_family/*.json`）、冻结 seed、`gates/T4.json`。见铁律 §2.4。 |
| **R2** | 禁止为了让测试通过而放宽、跳过或删除测试。测试红了就改实现，不是改测试。 |
| **R3** | 禁止桩实现。派生特征必须从**真实 observation** 计算，禁止常数填充、占位值、`pass`。lane token 必须**真的进入注意力计算**，禁止"加了参数但前向里不用"。 |
| **R4** | 禁止事后修改判据。§4.1 的判据文件必须在**开跑之前** commit。 |
| **R5** | 任何证据（**含失败结果**）都必须 commit 并 `git push origin pvz-env`。见铁律 §2.3。 |
| **R6** | 不得开始 T6。 |
| **R7** | 网络结构改动后必须**显式**处理 seed-0 初始化校验（§3.3），禁止静默绕过、禁止注释掉校验。 |
| **R8** | 禁止用"样本还不够""再跑一轮看看"作为继续加量的理由。见铁律 §3。 |

---

## 2. 硬阻断机制（本工作令的核心）

**背景**：上一轮的执行 Agent 在阶段 0 零胜率的情况下仍然机械地往下推进。
**因此本工作令不依赖执行者的自觉，所有关键约束都做成可执行的代码/退出码。**

### M1 · 阶段 0 的早期零信号硬停止（必须写进训练脚本）

在训练脚本里实现：**每当产生一次评估**（现有逻辑：每 5,000 局或 30 分钟），若同时满足

- `stage0_set.pass_rate == 0.0`（见 §3.2 的新评估字段），**且**
- 全部 5 个 cap1 训练任务的滚动 64 局胜率均为 `0.0`

则立即置 `stop = True`、`state["stop_reason"] = "stage0_no_signal"`、落盘、退出。
**理由**：在最简单任务上 5,000 局仍零胜率，问题一定在奖励 / 塑形 / 探索 / 网络四者之一，
继续加样本只是把同样的失败放大。此时**禁止**跑满 10,000 局，**禁止**进阶段 1。

### M2 · 阶段 0 门禁的代码级阻断（必须写进训练脚本）

新增 `scripts/t5_stage0_gate.py`：

- 读 `artifacts/t5/training_state.json`，取**最后一次**评估的 `stage0_set.pass_rate`；
- 判据：`pass_rate >= 0.50` 且样本数 `>= 5 × 64` → 写 `artifacts/t5/stage0_gate.json`
  （含 `result: "pass"|"fail"`、`pass_rate`、`sample_count`、`cumulative_episodes`、`commit`），
  **`pass` 时退出码 0，否则退出码 1**。

训练脚本里加硬检查：**当训练集为全部 20 个任务（阶段 1 模式）时**，启动前必须读
`artifacts/t5/stage0_gate.json`，若文件不存在或 `result != "pass"`，**直接 `raise RuntimeError`**，
消息写明"阶段 0 未通过，禁止进入阶段 1"。

唯一逃生口：显式传 `--ignore-stage0-gate` **且** 同时传 `--motivation` 说明原因；
此时必须把该事实写入 `run_config` 与最终报告。**默认不存在这条路径。**

### M3 · 每个缺口的变异测试（必须做，不是可选）

每完成一个缺口，必须做一次**变异测试**并记录到报告：

1. 故意破坏该缺口的实现（例如把派生特征改成常数、把 lane token 从注意力里摘掉、把过滤器改成恒等）；
2. 跑新增的测试，**确认它变红**；
3. 恢复实现，确认测试变绿；
4. 把"破坏方式 → 哪个测试变红"写进报告。

若测试在变异后仍然全绿，说明测试是空转的，**必须重写测试**。

### M4 · 不许跳步

严格按 §3 → §4 → §5 顺序。每个缺口完成即 commit + push。
**不得**在缺口未完成时先开跑训练（会浪费正式 run 名额，见 §3.4）。

### M5 · 更新设备硬检查（2026-09-30 追加，起因见下）

**事故**：首次阶段 0 跑到 75 分钟仍没有落下第一个 optimizer 更新、`cumulative_episodes`
始终为 0，而按 §8 的预算第一个更新应在约 6 分钟内完成。根因是 WSL 长开机后 GPU 掉线
（已知 bug），而 `resolve_device("auto")` 在 `torch.cuda.is_available() == False` 时
**静默回退到 CPU**（`python/pvz_agent_model.py:70-73`），训练因此没有崩、只是退化：
rollout 照旧在 18 个 CPU worker 上跑，状态文件只在更新落盘后才重写，所以进程看起来还活着。

实测代价（本次测得）：同一批 2,000 局、`chunks=16`、`seq=16` 的更新
在 RTX 5080 上 **184.4 s**，在 CPU 上 **1,072.6 s**（10,889.8 µs/transition），**慢 5.8 倍**。
按 §8 的 5 次更新算，阶段 0 会从约 1 h 变成数小时，且大概率在中途撞上 3 小时止损线，
既拿不到学习信号、也白烧一整晚。

**规格**：新增 `_check_update_device()`（`python/train_pvz_ppo_task_family.py`）并在 `main()`
解析出 `device` 之后立刻调用：

- 若 `device.type == "cuda"` → 通过，把 `device_name` 写进 `run_config`；
- 若 `device.type != "cuda"` **且** `artifacts/t5/perf/ppo_update_2000_flex_saved.json`
  记录的 `device == "cuda"`（即超参是对着 GPU 调出来的）→ **`raise RuntimeError`，直接退出**；
- 唯一逃生口：显式传 `--allow-cpu-update` **且** 同时传 `--motivation`，
  此时把 `override` 写进 `run_config`；只给 `--allow-cpu-update` 不给 `--motivation`
  → `raise ValueError`。

判定记录写入 `run_config["update_device"]` 与 `run_config["update_device_check"]`。
**这条检查必须在任何昂贵工作之前触发**（当前位于 `_baseline()` 之后、资源哈希与 rollout 之前）。

**变异测试**（已执行）：
- 破坏 1：把 `if record["benchmark_device"] != "cuda":` 改成恒真 → 
  `test_cpu_update_is_blocked_when_the_benchmark_was_cuda` 与
  `test_cpu_override_requires_motivation` 变红；
- 破坏 2：把 `if not motivation:` 改成恒假 → `test_cpu_override_requires_motivation` 变红；
- 恢复后 16 项测试全绿（`python/test_t5_overnight.py`）。

**顺带教训**：执行者当时**已经把根因写进日志**（"当前环境已明确缺少 GPU 设备，无法恢复
原 RTX 训练通路"），却选择"沿预注册流程观察到状态写出或三小时止损线"。这是把
"按流程办事"凌驾于"报告异常"之上。**凡遇到与预算不符的静默，先报告，再等。**

---

## 3. 三个缺口的实现规格

### 3.1 缺口 1：课程起点（cap1-only 训练）

**现状**：`_task_family()`（`train_pvz_ppo_task_family.py:66-72`）硬性断言训练集恰好 20 个任务；
`TRAIN_PATH` 是模块常量；无任何课程过滤参数。

**必须注意的坑**：`_baseline()`（第 88-90 行）会校验
`T4.json["metrics"]["manifests"]["train_sha256"] == sha256_file(TRAIN_PATH)`。
**因此不能通过"换一个 cap1 清单文件"来实现课程**——那会让 `_baseline()` 直接报错。
**课程必须做成过滤器，`TRAIN_PATH` 指向的文件保持不变。**

**规格**：

- 新增参数 `--curriculum`，取值 `all`（默认，行为与现在**完全一致**）或 `cap1`。
- `cap1` 的定义：`train.json` 中 `wave_cap == 1` 且 `zombie_count_multiplier == 1.0` 的
  全部任务（当前为 5 个：`train_day_1`、`train_day_2`、`train_day_3`、`train_day_4`、
  `train_fog_1`——以实际清单为准，**不要硬编码这 5 个 id**，按条件筛）。
- 过滤只作用于**采样输入**：`_assignments()` 的 `tasks` 参数（第 591 行）与
  `_curve_row()` 的 `train_tasks` 参数（第 684 行）。
- `_task_family()` 的 20 任务断言**保持不变**（清单本身仍是 20 个，只是采样时子集）。
- 筛出 0 个任务 → 报错退出。
- 把实际使用的课程任务 id 列表写入 `run_config` 与学习曲线每一行。

**验收**：`--curriculum cap1` 时，`state["last_update"]["task_counts"]` 只含这 5 个任务，
其余 15 个计数为 0（或不出现在字典里，二选一并写清）；`--curriculum all` 与改动前行为逐字段一致。

### 3.2 缺口 2：阶段 0 的独立评估路径

**现状**：`_evaluate()`（第 188-234 行）只评估 heldout 的 cap3×1.0 子集；
`_curve_rises()` 也只看 `heldout_cap3_x1_pass_rate`。**没有 cap1 的独立评估路径。**

**裁定（已定，不要再设计）**：阶段 0 的评估集 = **`train.json` 中 cap1 且 1.0× 的那 5 个任务，
各用其清单里自带的全部 seeds**（T4 实测每任务 64 个 seed，基线 pass rate **全部为 0.0000**——
已核对 `gates/T4.json`，所以 TODO §4 里写的"<5%"是保守说法，真实基线是 **0%**）。

**规格**：

- 在 `_evaluate()` 的返回字典里**新增** `stage0_set` 字段，与现有 `gate_set`/`reference_set` 并列。
  结构与 `gate_set` 相同（`task_count`/`sample_count`/`passes`/`pass_rate`/
  `terminal_wave_histogram`/`failure_terminal_wave_histogram`/`per_task`）。
- **不得改动 `gate_set` 的语义与计算方式**（它是阶段 1 的门禁裁判，且要与 T4 基线可比）。
- 学习曲线行（`_curve_row`）新增 `stage0_pass_rate` 字段。
- `_evaluate` 的现有调用点（第 678-681 行）传入的 `tasks` 参数是 heldout 全集；
  cap1 评估需要**额外**跑训练清单里的 5 个任务——复用同一个 env 实例，不要新建进程。

**验收**：一次评估后，`training_state.json` 里同时存在 `gate_set` 与 `stage0_set`，
且 `stage0_set.sample_count == 5 × 64 = 320`。

### 3.3 缺口 3：网络要求（派生特征 + lane 聚合 token）

**现状**：`pvz_agent_model.py:448` 的 `aux_lane_threat` 是 T4 时代的**辅助预测头**，
既不是 lane 聚合 token，也没有下列派生特征。

**规格（来自 TODO §4，已定）**：

1. **显式派生特征**（状态的可计算函数，不是硬编码策略）：
   - 每行：僵尸威胁度、最近僵尸距离、该行射手数、该行植物总血量
   - 全局：阳光收入速率（滑动窗口）、经济植物数与火力植物数之比、波次进度、距下一波进度
2. **lane 级聚合 token**：每行为一个 token 聚合该行实体，让注意力直接在"行"这一层发生。

**连带后果（必须处理，这是 R7 的来源）**：

- 改动会改变 `state_dict` 结构 → 必须递增 `MODEL_ARCHITECTURE_VERSION`；
- 观测语义变了 → 必须递增 `OBSERVATION_VERSION`；
- **`train_pvz_ppo_task_family.py` 第 519 行附近有一段校验**：当 `--initialization-seed 0` 时，
  要求 `t4_capability_profile._state_sha256(model.state_dict())` 等于
  `gates/T4.json["metrics"]["checkpoint"]["state_sha256"]`。**改了网络结构后这个校验必然失败。**

**处理方案（已定，按此实现）**：

- 保留该校验逻辑本身；当哈希不匹配时，若 `MODEL_ARCHITECTURE_VERSION` 已递增，
  则**不静默放行**，而是要求显式传入 `--initialization-note "<为什么 T4 基线已被取代>"`；
- 未传 note → `raise`，消息写明"网络结构已变更，seed-0 初始化不再与 T4 基线一致"；
- 传入 note 时，把**实际哈希**与 T4 哈希**同时**写入 `run_config` 与 `provenance`
  （字段建议 `initialization_baseline`：`{"status": "superseded", "actual": ..., "t4": ..., "note": ...}`）；
- **报告中必须明确写出**："T4 的 0% 基线是在旧网络结构上测的，新结构下阶段 0 的基线不再与之逐位可比"。

**验收**：模型前向对含新特征的 observation 正常工作；`--initialization-seed 0` 不传 note 时
按预期报错，传 note 时正常启动且 provenance 里有两个哈希。

### 3.4 关于正式 run 名额（**先读这条再动手**）

- 每次调用 `train_pvz_ppo_task_family.py` 都会消耗一个正式 run 名额
  （`run_number = state["formal_runs"] + 1`，`MAX_FORMAL_RUNS = 8`）。
- run > 1 必须给 `--motivation`，且**默认从上一 run 的 `runs/run_{N-1}/gameplay_model_v1_ppo.pt` 续训**；
  网络结构一变就因 `model_architecture_version` 不匹配而报错。
- **因此顺序必须是：先做完 §3.1–§3.3 并 commit，再开 run 1。**
  绝不允许"先跑一轮看看，回头再改网络"——那会白白烧掉 8 个名额里的第 1 个。

---

## 4. 阶段 0：学习信号验证门

### 4.1 判据预注册（**开跑之前** commit，R4）

在跑阶段 0 **之前**创建 `gates/T5_stage0_criteria.json` 并 commit，内容为字面数字：

```json
{
  "stage": "T5-stage0",
  "preregistered_at": "<ISO 8601 时间戳>",
  "curriculum": "cap1",
  "eval_set": {
    "source": "artifacts/task_family/train.json",
    "filter": {"wave_cap": 1, "zombie_count_multiplier": 1.0},
    "task_ids": ["<按实际清单填写>"],
    "seeds_per_task": 64
  },
  "baseline_pass_rate": 0.0,
  "baseline_evidence": "gates/T4.json metrics.per_task.train.*.pass_rate == 0.0000 (n=64 each)",
  "target_pass_rate": 0.5,
  "episode_budget": 10000,
  "early_stop": {
    "checked_at_episodes": 5000,
    "condition": "stage0_set.pass_rate == 0.0 AND all cap1 rolling-64 train win rates == 0.0",
    "action": "stop and write diagnosis; do NOT run to 10000; do NOT start stage 1"
  }
}
```

**这个文件在跑之前 commit，事后修改会在 git 历史里可见 = 违规（R4）。**

### 4.2 启动命令

```bash
cd ~/PvZ-Portable
git fetch origin && git pull --ff-only origin pvz-env
git log --oneline -1                       # 确认包含 §3 的三个缺口提交
rm -f artifacts/t5/training_state.json     # 必须是干净的 run 1（见 §3.4）
python python/train_pvz_ppo_task_family.py \
  --device cuda --curriculum cap1 \
  --max-episodes-per-run 10000 \
  --minibatch-chunks 16 --learning-rate 1e-4 --attention-backend auto \
  --output-dir artifacts/t5
```

**配置说明**：性能优先（已裁定）。`chunks 16 / lr 1e-4 / attention auto` 是脚本当前默认值，
CUDA 吞吐实测最优（2000 局更新 404 s → 188 s，2.15×）。但**该组合只验证过吞吐、
从未验证过学习信号**——若阶段 0 零胜率且 M1 已触发，则按 §6 处理，
**不要在无人值守的情况下自动改回 `chunks 1` 重跑**。

### 4.3 通过条件

`stage0_set.pass_rate >= 0.50`（5 任务 × 64 seed = 320 局评估）。
通过后：写 `gates/T5_stage0_gate.json`（M2）、commit、push，然后进 §5。

---

## 5. 阶段 1：正式训练

**前置**：`artifacts/t5/stage0_gate.json` 存在且 `result == "pass"`（M2 会在代码级强制检查）。

```bash
python python/train_pvz_ppo_task_family.py \
  --device cuda --curriculum all \
  --motivation "阶段 0 通过（stage0 pass_rate=<实际值>）；扩展到全部 20 个任务" \
  --max-episodes-per-run 20000 \
  --minibatch-chunks 16 --learning-rate 1e-4 --attention-backend auto \
  --output-dir artifacts/t5
```

**门禁**（`gates/T5.json`，字段见 TODO §2.3）：

| 指标 | 阈值 |
|---|---|
| 端到端训练吞吐（含更新） | ≥ 5,000 局/小时 |
| heldout ∩ (cap3, ×1.0) pass rate | ≥ 90% |
| 学习曲线 | 必须上升（`_curve_rises`） |
| 单核 rollout | ≥ 5,000 局/小时（已有 6,273.8，直接引用） |

**硬停止条件已实现**（`HARD_STOP_EPISODES = 20_000`，pass rate < 20% 且近 5,000 局改善 < 5pp）。
触发即停止、写诊断、**不得"再跑一轮看看"**（R8）。

**调参纪律**：正式 run 最多 8 次。若阶段 1 未过门禁且仍有剩余名额，
**只在有明确、可陈述的假设时**才允许发起下一次 run，且必须给 `--motivation` 记录
"改了什么、为什么、与上一轮曲线的差异"。**禁止无假设地反复重跑。**

---

## 6. 失败时的行为（**这是最容易偷懒的地方，逐条执行**）

任一门未通过时：

1. **立即停止**，不要继续跑、不要自动改超参重跑、不要进下一个阶段。
2. 写诊断报告，按 TODO §4「常见失败模式的诊断顺序」逐条排查，**不要跳步**：
   1) pass rate 卡 0% → 查奖励信号（终局 ±1 是否真进入回报、Φ 塑形是否在量纲上淹没主信号、
      advantage 是否归一化）；
   2) 学到但低于阈值 → 把失败 seed 的终局波次分布与脚本失败 seed 对比，判断是同一批难 seed
      还是策略缺陷；
   3) 崩溃 / NaN → 查梯度裁剪、value loss 与 policy loss 的量级比；
   4) 吞吐不足 → profile 环境推进与模型推理各占多少，**用数据说话，不要猜**。
3. 把失败证据（`training_state.json`、学习曲线、评估原始结果）commit 并 push（R5）。
4. **报告里必须回答**："如果继续加样本会发生什么"——若答案只是"同样的失败被放大"，就明确写出来。

**允许在失败后做的事**：写诊断、修实现 bug、补测试。
**不允许**：改判据、改门禁、改任务定义、无假设地重跑。

---

## 7. 报告要求

结束时（无论成功失败）写 `T5_OVERNIGHT_REPORT.md`，**必须包含**：

1. **三个缺口各自**：改了什么文件、加了什么测试、**变异测试的"破坏方式 → 哪个测试变红"**；
2. **判据预注册文件的 commit 哈希**（证明判据是事前定的）；
3. 阶段 0：启动命令原文、每 5,000 局的 `stage0_set.pass_rate` 序列、终止原因、实际局数；
4. 阶段 1（若进入）：完整超参、学习曲线序列、门禁各指标实测值、`gates/T5.json`；
5. **未采用的方案清单**——凡是实测后放弃的思路，写清"实测数据 + 为什么放弃"
   （沿用 `T5_PERFORMANCE_REPORT.md` 的诚实取舍做法）；
6. **明确写出仍未验证的假设**（例如"新网络结构下的基线未与 T4 逐位可比"）；
7. 本次消耗的正式 run 名额数（`state["formal_runs"]`）与剩余名额。

---

## 8. 时间预算参考（用于判断是否卡住）

| 阶段 | 预估 | 依据 |
|---|---|---|
| 补三个缺口 + 测试 + 变异测试 | 1–3 h | 纯实现工作 |
| 阶段 0（10,000 局） | 约 1 h | rollout 13.5 min（44,565.9 局/h）+ 更新约 16 min（188 s/2000 局 × 5）+ 评估 |
| 阶段 1（≤20,000 局） | 2–6 h | 取决于何时触发门禁或硬停止 |

**若某一阶段耗时超过上表 3 倍仍未结束，视为卡住**：停止、写清楚卡在哪、push 证据。
不要在无进展的情况下让它空转到天亮。
