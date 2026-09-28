# PvZAgent Python 训练层代码评审 · 修复报告

评审范围：`python/`（PvZ1 模拟器魔改出的 AI 训练环境新增代码）
首次评审快照：2026-09-28 19:33　|　修复完成：2026-09-28 19:56　|　性能优化：2026-09-28 20:20　|　Apple Silicon 设备策略：2026-09-28 20:35　|　线程数策略：2026-09-28 21:10
运行环境：系统 Python 3.14（torch 2.13.0、numpy 2.5.1）；`pytest` 未安装，改用 `python3 -m unittest discover -p "test_*.py"`
静态检查：`pyflakes`（安装在隔离 venv `~/.workbuddy-ai/binaries/python/envs/default`）
硬件：Apple M5 Pro，15 核（5 P + 10 E），arm64

> 评审期间工作区被并发改写多次（19:30 / 19:32 / 19:36 / 19:37 / 19:44）。本报告的所有结论都以**当前磁盘内容**重新实测为准。

---

## 0. 结论

**全部 17 条生产问题（P1–P17）与 8 条测试问题（T1–T8）已修复；随后完成一轮搜索吞吐优化（见 §5），修正了一个会让训练慢 1.55× 的设备默认值（见 §6.5），并把复现性用的线程数 pin 改成了参数（见 §6.6）。**

| 指标 | 修复前 | 修复后 | 性能优化后 | 设备修复后 | 线程数放开后 |
|---|---|---|---|---|---|
| 测试数 | 26 | 121 | 139 | 141 | **184** |
| 测试耗时 | 0.007s | 0.99s | 1.09s | 1.30s | **1.58s** |
| `pyflakes` 未定义名称 | 7 处（`hashlib`） | **0** | **0** | **0** | **0** |
| `pyflakes` 未使用导入 | 有 | **0** | **0** | **0** | **0** |
| 变异测试捕获率 | — | **24 / 25** | **14 / 17**（3 项为构造性等价变异） | 同上 | 同上 |
| `advice()`（参考模拟器） | — | 30.0 ms | **8.6 ms（3.49×）** | 同 | 同 |
| `advice()`（真实 C++ 模拟器） | — | 186.5 ms | **164.6 ms（1.13–1.17×）** | 同 | 同 |
| `search_value_features`（真实观测） | — | 122.8 µs | **23.2 µs（5.3×）** | 同 | 同 |
| `--device auto` 在 Apple Silicon 上 | — | mps（慢 1.71×） | mps | **cpu** | cpu |
| CPU 线程数 | — | 1（硬编码） | 1 | 1 | **4（`--threads`，可调）** |

> 测试数从 141 跳到 184，是因为工作树里还有三批**已写完但未提交**的改动（搜索预算记账、策略分布蒸馏、搜索审计工具），它们各自带来了新用例；见 §7。

**验证方式**：不只跑测试。对生产代码注入了 25 个人工缺陷（把 `elapsed_ticks` 恒置 0、让 transposition 忽略 `state_hash`、让快照永不释放、让 GAE 丢掉 tick 比率、取消越界保护、恢复 P8/P9 回归……），确认测试能捕获其中的 24 个。修复前的测试套件对**全部 25 个变异都无反应**。性能优化阶段另注入 17 个缺陷，捕获 14 个。

**性能优化的等价性**：`search_value_features` 的向量化被证明为**逐位相等**（257 万对抗性 double 上的转换等价 + 语料差分测试），并在**真实 `pvz-portable` 模拟器**上对 4 个 seed × 6 轮搜索验证了**搜索决策完全一致**（24/24）。详见 §5.2。

**清理时的一个附带发现**：删除 `/tmp/pvz_search.bak` 前比对发现，工作树的
`advice()` 根候选上限是 `min(max(1, budget // 2), max(32, root_candidate_limit * 4))`，
而备份里是 `min(budget, ...)`。核查确认**不是残留变异**（4 个变异脚本都用 `try/finally`
还原，且无一触碰该行），而是 19:55 那轮修复里刻意加的预算保护 —— 它符合
`test_root_candidate_limit_reserves_budget_for_depth` 表达的「为深度展开预留预算」意图。
但**测试此前无法区分两个版本**（140 个用例两边都绿），故补了
`test_root_screening_limit_never_consumes_the_whole_budget`，并用 3 个变异
（回退到备份版本 / 完全去掉预算保护 / 放宽到 90%）验证它能捕获，3/3 全中。

---

## 1. 最严重的发现：测试通过，但通过得毫无意义

修复前 26 个测试全绿，然而：

- **测试桩的 tick 语义是错的**：桩把「等待时长」直接写进 `observation["tick"]`，而搜索是按 `observation["tick"]` 差分推算推进量的。实测一次 `advice()`：24 个分支里有 **16 个** 的 `advanced == 0`。也就是说 `horizon_ticks` 在实际测试里从未成为约束，`elapsed_ticks` 几乎不增长。
- **transposition 去重路径不可能被触发**：`state_hash` 由全局递增的 snapshot 编号派生，任意两个分支的 hash 必然不同。而 `TODO.md` 把「相同状态只保留更高分的路径」列为验收条件——该条件实际零覆盖。
- **桩从不返回终局**：`terminal` 恒为 `False`，胜负路径、`_transition_reward`、`terminal_outcome` 在真实循环中从未执行。
- **7 处断言恒真**：`assertGreaterEqual(value, -1.0)`（网络末层就是 `nn.Tanh`）、`assertFalse(any("GameplayModel" in type(m).__name__ ...))`、`assertEqual(ENV_PROTOCOL_VERSION, 3)`（复述常量）、`assert 16 <= 16`。
- **覆盖严重倾斜**：`PvZEnv`、`GameplayModelV1`、`observation_tokens`、`select_action`、`hard_behavior_cloning_loss`、`predict_action`、`pvz_imitation.train`、`benchmark_pvz_agent` —— **所有真正跟模拟器交互的代码和全部模型代码，一行测试都没有**。

---

## 2. 生产代码：P1–P17

### 2.1 本次修复中**新发现**的缺陷（原评审未列出）

| # | 严重度 | 问题 | 修复 |
|---|---|---|---|
| **P14** | **致命** | `pvz_env.py` 在 **5 处** 使用 `hashlib`（`_debug_state_sha256`、`_manifest_data`、`_write_manifest` ×2、`_check_manifest` ×2）却**从未导入它**。`save_replay()` 必定抛 `NameError`，整条 replay 存取链路是死的。 | 把摘要计算下沉到 `pvz_common.sha256_bytes` / `canonical_digest`，`pvz_env` 不再需要 `hashlib` |
| **P15** | **致命** | `train_pvz_ppo.episode_hash()` 同样使用未导入的 `hashlib`，PPO 每次 update 都会 `NameError` | 改用 `pvz_common.canonical_digest` |
| **P16** | 低 | `benchmark_pvz_agent._checkpoint_metadata` 与 `train_pvz_agent._search_value_metadata` 是逐字相同的重复实现 | 合并为 `pvz_training_artifacts.checkpoint_metadata` |
| **P17** | — | 曾怀疑 `pvz_search_diagnostics.compare_label_samples` 会把「某个 horizon 无样本」的簇计为匹配簇。**实测证伪**：`defaultdict` 的内层桶只在有样本时才创建，因此该分支不可达。已撤回改动，未留下不可观测的死守卫。 | 无改动 |

> P14/P15 是最值得记录的一条：**它们是「测试通过」这件事本身的产物**。删掉看起来冗余的 `import hashlib` 时，没有任何测试变红，因为没有任何测试调用过 `save_replay` / `episode_hash`。这正是 T6（覆盖倾斜）的直接后果。

### 2.2 原评审问题

| # | 问题 | 修复 | 验证 |
|---|---|---|---|
| P1 | `replay_record` 的 tick 校验恒真（`info.get(...)` 默认值 = 写入时的同一个表达式） | `step()` 改为按 `observation["tick"]` 差分计算真实推进量并放进 `info["ticks_advanced"]`；`replay_record` 比对真实值；5 处 `info.get(...)` 回退复制全部消除 | 变异「elapsed 恒为 0」被捕获 |
| P2 | `observation_version` / `task_version` 是散落的 15 处字面量 `2` | 新建无依赖底层模块 `pvz_common.py`，集中 `ENV_PROTOCOL_VERSION` / `REPLAY_FORMAT_VERSION` / `OBSERVATION_VERSION` / `TASK_VERSION` / `TRAINING_SEED` / `VALUE_RANGE` | `assertIs` 校验 `pvz_env` 只是重导出 |
| P3 | `provenance()` 硬编码 `17` 与 `[-1, 1]` | 改用 `TRAINING_SEED` / `VALUE_RANGE` | — |
| P4 | `sha256_file` ×3、git 元数据 ×3、`TaskSpec` 组装 ×3、`contiguous_seeds` 纯别名 | 各收敛为 1 份（`pvz_common.sha256_file` / `git_metadata` / `pvz_env.training_task`），删除别名。**统一了 dirty 规则**：三份 git 元数据现在共用同一套「忽略生成物」过滤 | — |
| P5 | `search_value_features` 守卫不对称（只挡超出、不挡不足） | 改为**预分配定长**再填充，结尾 `cursor != SEARCH_VALUE_FEATURES` 双向校验 | 宽度 4116 实测一致 |
| P6 | 根节点 `_leaf_value` 是死计算（每次决策白跑一次 value 前向） | 根节点 `score` 改为 `0.0` 并注释说明 | — |
| P7 | `_successive_halving` 排序键对 `_estimate_root` 重复求值 | 提取 `_root_rank`，每个元素只求值一次 | — |
| P8 | `PvZEnv.__init__` 立即哈希 `main.pak`，裸 `FileNotFoundError` 先于友好提示 | 改为惰性 + 缓存 | 新增 `test_construction_does_not_touch_the_resource_directory`；变异「构造期立即哈希」被捕获（15 个错误） |
| P9 | `summarize` 缺空集保护；决策上限硬编码 | 空集抛 `ValueError`；新增 `--max-actions` / `DEFAULT_MAX_ACTIONS` | 变异「取消空集保护」被捕获 |
| P10 | `--train-only` 用 CLI 默认 dagger 区间做互斥校验；BC/最终 provenance 自相矛盾 | `--train-only` 从磁盘读取真实 dagger 种子；两处 `provenance(...)` 改关键字参数 | — |
| P11 | 未使用导入（`MODEL_CONFIG` 等） | 清理干净 | `pyflakes` 退出码 0 |
| P12 | 特征层对未知类型静默丢弃，「没有植物」与「有未建模植物」不可区分 | 每类增加一个 dedicated unknown 槽（`SEARCH_VALUE_FEATURES` 4048 → **4116**）；`_slot()` 统一映射；imitater 判断改为 `is not None and >= 0` | 新增两个测试；变异「未知槽合并进 0 号槽」被捕获 |
| P13 | `episode_targets` 未校验 `zombie["row"]`，越界直接 IndexError | 加 `0 <= row < LANE_COUNT` 边界检查 | 变异「取消越界保护」被捕获 |

---

## 3. 测试代码：T1–T8

### T1 / T2 / T3【高】协议桩重写

`_FastProtocolEnv` → **`_FakeSimulator`**，它是一个**有状态的确定性模拟器**，忠实复现真实 `BRANCH_SNAPSHOT_FAST` 契约的三条要点：

1. **绝对 tick**：维护每个 snapshot 的棋盘状态，`observation["tick"]` 是动作**之后**的绝对 tick（tick 120 发 60 tick 等待 → 报 180，而不是 60）；
2. **内容派生 `state_hash`**：`f"t{board['tick']}|s{board['sun']}|p{board['plants']}"`，因此两条不同动作序列到达同一棋盘时会**碰撞**，transposition 表真的被走到；
3. **终局分支**：`terminal=True` 时不带 `snapshot_id` / `state_hash`，与 `_branch_result` 的非终局守卫一致。

同时它还会记录 `issued` / `dropped` 快照 id，从而可以断言**搜索不泄漏快照**。

新增的针对性用例：

- `test_branch_tick_is_absolute_and_not_the_wait_duration` —— 直接钉死 T1 的缺陷；
- `test_different_action_orders_that_reach_the_same_board_transpose` —— 两个分支 snapshot 不同、`state_hash` 相同 → 转置键必须相等；并额外构造一个「elapsed / decision_count / same_tick 全相同、只有棋盘不同」的对照，**只有 `state_hash` 能区分**（这是让变异验证通过的关键断言）；
- `test_terminal_branch_reports_its_outcome_and_has_no_snapshot` / `..._can_report_a_loss`；
- `test_nonterminal_branch_without_snapshot_is_rejected` —— T5 指出的「负例不存在」已补齐；
- `test_search_releases_every_snapshot_it_creates`。

### T4【中】7 处空转断言

| 原断言 | 处理 |
|---|---|
| `assertEqual(ENV_PROTOCOL_VERSION, 3)` 等复述常量 | 替换为 `test_env_module_reexports_the_shared_protocol_versions`（`assertIs` 校验单一来源） |
| `assertFalse(hasattr(CandidateGenerator(), "model_proposals"))` | **删除**（负向属性检查无信息量） |
| `inspect.signature` 检查参数名 | **删除**，替换为 `test_search_is_reproducible_from_the_simulator_alone`（同一状态两次决策必须完全一致）+ `test_advice_only_proposes_actions_legal_in_the_root_observation` |
| `assertLessEqual(root_candidate_limit, 16)`（等价于 `16 <= 16`） | 改为断言**精确公式值**并加多组配置的 `root_candidate_limit * 2 <= simulation_budget` 性质 |
| `assertFalse(any("GameplayModel" in type(m).__name__ ...))` | 替换为 `test_model_input_width_matches_the_feature_vector`（`first_layer.in_features == SEARCH_VALUE_FEATURES`） |
| `assertGreaterEqual(value, -1.0)` / `assertLessEqual(value, 1.0)`（Tanh 恒真） | 替换为 `test_value_model_output_depends_on_its_input`（**恒返回同一数值的模型也会通过原断言**） |
| `test_provenance_path_values_are_jsonable`（单用例） | 扩展为覆盖 `Path` / `tuple` / `str` / `int` / `float` / `bool` / `None`，并断言转换结果能通过 `json.dumps` |

### T5【中】私有实现耦合

- 新增 `_node()` 关键字工厂，**消除 `_SearchNode` 的 10 个位置参数构造**（字段重排不再静默改义）；
- 冗余的 `self.assertIsNotNone(x)` + 裸 `assert x is not None` 改为 `assertIsNotNone(msg)` + `cast(...)`（一条断言、一条类型收窄）；
- 不再手工写 `searcher._snapshots = {1, 2}`，改为用 `_FakeSimulator` 真实产生的快照 id。

### T6【中】补齐零覆盖（这是本次改动量最大的部分）

| 新文件 | 用例数 | 覆盖 |
|---|---|---|
| `test_env_protocol.py` | 21 | `PvZEnv` 构造惰性、reset 的全部参数校验（level/seed/task/profile/deck/forced_seeds）、生命周期守卫、坐标校验、replay 版本校验、**实验 manifest 的写入/校验/防篡改**、共享常量与 `training_task` 工厂 |
| `test_agent_model.py` | 23 | `observation_tokens`（形状 + cell/packet 索引约定 + 类型钳制）、`GameplayModelV1.step`、`select_action`（强制动作 / 非法动作拒绝 / 采样合法性 / 确定性）、`predict_action`、`hard_behavior_cloning_loss`（**含动作类型头梯度非零**与一步优化降低损失）、`episode_targets`（含越界）、`pvz_imitation.train` |
| `test_shared_helpers.py` | 18 | `pvz_common` 全部 helper（含跨 1 MiB 分块的 `sha256_file`、非 git 目录的降级）、`visible_state_key`、`benchmark_pvz_agent.summarize`（空集 / Wilson 区间 / 失败波次分布 / 搜索指标条件汇总） |

`test_env_protocol.py` 里的 manifest 测试**正是 P14 的回归守卫**：它在修复前会以 `NameError` 失败。

### T7【低】重复用例

`test_zero_tick_transition_does_not_decay_trace` 与 `test_time_based_lambda_matches_reference_ticks` 保留（两者分别钉住 `lambda ** 0 == 1` 与 `lambda ** 1`，语义不同），并新增 `test_bootstrapped_values_enter_the_advantage`：用 `lambda=1.0` 把递推化成闭式，期望值（`0.79` / `0.4801` / `0.99` / `0.9801`）可手工推导，不再是实现代码的镜像。另加 `test_returns_are_advantages_plus_the_state_value`。

### T8【低】

`import inspect` 已删除，全仓库不再有签名反射。

---

## 4. 验证证据

```
$ python3 -m unittest discover -p "test_*.py"
Ran 139 tests in 1.085s
OK

$ pyflakes python/*.py scripts/real_env_equivalence.py
(无输出，退出码 0)
```

变异验证（对生产代码注入缺陷，确认测试变红）：

```
捕获  elapsed 恒为 0（忽略 tick 差分）              3 failures
捕获  transposition 忽略 state_hash                 1 failure
捕获  _release 不释放快照                           2 failures
捕获  结果键把失败排在未完成之上                    2 failures
捕获  horizon 不裁剪超时动作                        1 failure, 1 error
捕获  same_tick_actions 永不累加                    1 failure
捕获  GAE trace 忽略 tick 比率                      1 failure
捕获  GAE 丢弃 bootstrap value                      1 failure
捕获  未知类型静默映射到槽 0                        1 failure
捕获  seed 角色不再检查重叠                         1 failure
捕获  CLI tuple 不再转 list                         1 failure
捕获  checkpoint_metadata 直接返回原字典            1 failure
捕获  cell 索引约定被改坏                           1 failure
捕获  select_action 不再过滤非法卡槽                1 failure
捕获  BC 损失忽略示范动作                           2 failures
捕获  lane 目标错位一行                             2 failures
捕获  lane 目标取消越界保护                         1 error
捕获  P8 回归：构造期立即哈希资源                   15 errors
捕获  manifest 摘要不再重算                         1 failure, 1 error
捕获  P9 回归：summarize 不再拒绝空集               1 error
捕获  sha256_file 只读首块                          1 failure
捕获  visible_state_key 不再对 tick 分桶            1 failure
捕获  未知类型槽被合并进 0 号槽                     1 failure
捕获  checkpoint 校验不再比对 task signature        1 failure
SKIP  诊断把空桶计为匹配簇                          （该分支不可达，见 P17）
```

---

## 5. 性能优化（本轮）

### 5.1 基线 profile 与结论

合成棋盘（30 plants / 15 zombies / 10 projectiles / 288 合法落点，`torch.set_num_threads(1)`）：

| 项目 | 优化前 | 优化后 |
|---|---|---|
| `search_value_features` | 152.7 µs | **44.1 µs**（合成）／**23.2 µs**（真实观测，122.8 → 23.2，**5.3×**） |
| `SearchValueModel.predict` | 173.7 µs | **67.4 µs** |
| `SearchTeacher.advice()`（参考模拟器，budget=256） | 30025.8 µs | **8603.6 µs（3.49×）** |
| `advice()`（真实 C++ 模拟器，budget=256） | 186.5 ms | **164.6 ms（1.13–1.17×）** |

`advice()` 的 cProfile（50 次，`sort_stats("cumulative")`）从 2.242 s 降到 1.025 s：

```
优化前                                         优化后
2.242  advice()                                1.025  advice()
1.961  ├ _branch_result                        0.799  ├ _branch_result
1.863  │  ├ predict                            0.707  │  ├ predict
1.341  │  │  ├ search_value_features           0.369  │  │  ├ search_value_features
0.837  │  │  │  └ torch.tensor(list)  ← 最大单项  ───  │  │  │  └ (已消除)
0.240  │  │  └ nn.Module 调度（_wrapped_call）  0.240  │  │  └ nn.Module 调度
0.087  │  │     └ torch._C._nn.linear          0.087  │  │     └ torch._C._nn.linear
```

**注意**：`torch.tensor(4116 python floats)` 一项就占了原 `advice()` 的 **37%**（0.837/2.242），
比真正的矩阵乘法（0.087 s）贵 9.6 倍。这是纯粹的 Python 对象拆箱开销。

### 5.2 改动一：`search_value_features` 改用 numpy float64 累加器

**改动**：`features = [0.0] * N` → `np.zeros(N, dtype=np.float64)`；五个稀疏子块改为
**基础切片视图**（`features[a:b]` 是 view，累加直接落回主向量）；结尾
`torch.tensor(features, dtype=torch.float32)` → `torch.from_numpy(features.astype(np.float32))`。

**等价性论证**（逐位相等，非近似）：

1. Python `float` **就是** IEEE-754 binary64。写入累加器的每个值在两种实现下表示完全相同。
2. numpy 的 `arr[i] += x` 与 `list[i] += x` 都是 round-to-nearest-even 的 binary64 加法，逐位相同。
3. 唯一的 float64 → float32 转换仍在**同一批 binary64 输入上发生且只发生一次**。
   `torch.tensor(list, float32)` 与 `torch.from_numpy(arr.astype(float32))` 已在
   **257 万个对抗性 double** 上比对（非规格化数、float32 tie-to-even 边界、量级两端、
   超出 float32 范围的饱和），**逐位一致**。
4. 累加**顺序未变**（`for plant in plants` / `for zombie in zombies` 的遍历次序保持原样），
   而浮点加法不满足结合律，所以这一点必须显式保证。

**等价性测试**：`python/test_search_value_features.py`

- `_reference_search_value_features` 是**冻结的原实现**（逐字保留 list 版本）。
- 语料 37 个观测覆盖：未知／负数／非数值／`None` 类型 id、越界行列、`squished`／零血／
  超血实体、imitater 卡（含 `-1`/`None`/越界）、包索引越界、空棋盘、无 `cells`、
  缺省可选键、极端 `sun`/`tick`、urgency 钳位边界，以及**同槽碰撞**（唯一会让加法顺序起作用的地方）。
- 断言逐位相等（`torch.Tensor.numpy().tobytes()` 比较），并有一条守卫测试防止语料退化成全零向量。
- `ConversionEquivalenceTests` 独立钉住上述第 3 条前提，包括一条
  `test_accumulating_in_float64_is_required`：证明 float32 累加器会给出**不同的** float32 结果
  （7 次 `+=0.2` → 1.4000000000000001 vs 1.4000000953674316），因此该契约不可被无声降级。

**真实模拟器端到端等价性**：`scripts/real_env_equivalence.py`
在 `pvz-portable` + PvZ GOTY `1.2.0.1073_EN` 上，对 4 个 development seed
（30000–30003）各交替跑 3 轮「优化版 / 冻结参考版」，共 24 次搜索：

```
seed 30000: identical=True  vectorised=  164.55 ms  reference=  186.46 ms  speedup= 1.13x
seed 30001: identical=True  vectorised=  161.70 ms  reference=  189.29 ms  speedup= 1.17x
seed 30002: identical=True  vectorised=  152.02 ms  reference=  178.57 ms  speedup= 1.17x
seed 30003: identical=True  vectorised=  152.02 ms  reference=  174.11 ms  speedup= 1.15x
4/4 seeds produced bit-identical search decisions across 6 runs each
```

「identical」比较的是动作、全部候选分数（12 位小数）、策略分布、`best_second_margin`、
`terminal_outcome`、`search_elapsed_ticks` 与 `simulation_count`。**24/24 完全一致。**

### 5.3 改动二：`SearchValueModel` 缓存设备

`predict` 每个叶子调用一次（每次决策约 200 次），而 `next(self.parameters()).device`
要构建生成器 + 参数字典，实测 **7 µs/次**（占 `advice()` 约 6.6%）。
改为 `register_buffer("_device_anchor", torch.zeros(1), persistent=False)`：
`Module.to` 会像搬参数一样搬它，而 `persistent=False` 让它**不进 `state_dict`**，
因此既有 checkpoint 仍可加载（`test_the_device_anchor_stays_out_of_the_state_dict` 钉住这一点）。

### 5.4 被否决的方案（有数据支撑）

**叶子估值批量化 —— 否决。** 原计划把 `_expand_actions` 里约 4.3 个非终局分支合并成一次前向。
实测批量前向成本曲线（`Linear(4116,128)`，1 线程）：

```
batch=1   15.07 us/row      batch=4   14.79 us/row      batch=16   3.73 us/row
batch=2   28.16 us/row      batch=8    7.49 us/row      batch=1024  1.05 us/row
```

**batch ≥ 2 存在约 44 µs 的固定开销**，把 ~4 行的批量收益完全吃掉。根因已定位：

```
F.linear(x, W, b)   batch=1   6.06 us      batch=4  47.59 us
x @ W.t().contiguous() + b     5.79 us               15.78 us
```

`F.linear` 走 `addmm(bias, x, W.t())`，而 `W.t()` 是**非连续转置视图**；M≥2 时该布局掉出
快速路径，比连续权重慢 **3.2 倍**（与 bias 无关：`F.linear(x, W)` 同样 48.67 µs；与线程数无关：
1/5 线程结果相同）。要吃到批量收益必须缓存连续转置权重，而权重每次 `optimizer.step()`
都会变，缓存失效是实打实的正确性风险。同时批量前向与单行前向**并非逐位相等**
（GEMM 与 GEMV 的累加顺序不同，实测偏差 ≤3.1e-7，稀疏真实特征 ≤1.3e-8）。

结论：收益≈0、需要引入缓存陈旧风险、且放弃逐位相等 —— **不做**。当前单行路径恰好走在快路径上。

**进一步向量化 `search_value_features` —— 暂缓。** 剩余 23–44 µs 主要是逐实体 Python 循环
（`sum`/`min`/`dict.get` 合计约 2 万次/次调用）。把它改成 numpy 逐行聚合属于重写，
在真实环境里该函数只占 `advice()` 的 13%，收益被模拟器往返淹没，性价比不足。

### 5.5 真实环境暴露的下一个瓶颈（未动 C++）

真实模拟器下 `advice()` = 164.6 ms，cProfile 显示 **`pvz_env._read_message` 独占 1.049 s / 1.582 s（66%）**
——即等待 C++ 进程返回。Python 侧全部加起来不到 20%。用协议命令分解 C++ 单分支成本：

```
OBS（完整观测序列化 + 往返）              89.8 us      观测 JSON 10 565 B
SNAPSHOT_FAST                            101.4 us
BRANCH_SNAPSHOT_FAST 1x W:0              305.5 us      → 每分支固定 ~290 us
BRANCH_SNAPSHOT_FAST 4x W:0             1165.7 us      → 291 us/分支（线性）
BRANCH_SNAPSHOT_FAST 8x W:0             2337.0 us      → 292 us/分支（线性）
BRANCH_SNAPSHOT_FAST 1x W:300            813.1 us      → 300 tick 约 508 us（1.7 us/tick）
BRANCH_SNAPSHOT_FAST 4x W:300           3166.5 us      → 791 us/分支 = 290 固定 + 501 模拟
```

即 **每分支约 290 µs 固定开销 ≈ `EnvironmentObservation` 90 µs + `LawnSaveGameToMemory` 101 µs
+ `LawnLoadGameFromMemory` + `EnvironmentSnapshotHash`**，与 TODO「保存/恢复或状态序列化」的判断一致。
`src/main.cpp` 的 `BRANCH_SNAPSHOT_FAST` 循环对**每个分支**都 `RestoreEnvironmentSnapshot` →
执行动作 → `EnvironmentObservation` → `SaveEnvironmentSnapshot` → `EnvironmentSnapshotHash`，
其中 `EnvironmentSnapshotHash` 是对整个 `snapshot.board` 字节向量做 FNV-1a。

**本轮未改 C++**：这块的成本落在棋盘序列化格式内部，且用户明确要求模拟器改动必须附等价性证明。
在没有先把 `verify_env_equivalence.py` 跑成自动化回归之前动它是不可控的。

### 5.6 变异验证（本轮新增）

对生产代码注入 17 个缺陷，确认新测试套件变红：

```
捕获  float64 累加器 -> float32                        3 failures
捕获  植物块视图 -> 拷贝（累加被丢弃）                   3 failures
捕获  僵尸块视图 -> 拷贝                                3 failures
捕获  卡片块视图 -> 拷贝                                2 failures
捕获  float32 转换 -> float16 往返                      5 failures
捕获  占位 1/COL_GROUPS -> 1/COL_COUNT                  3 failures
捕获  僵尸计数累加 -> 赋值                              3 failures
捕获  僵尸威胁累加 -> 赋值                              1 failure
捕获  植物血量累加 -> 赋值                              1 failure
捕获  imitater 槽偏移丢失                               2 failures
捕获  行块推进 8 -> 7（触发布局守卫）                   15 errors
捕获  植物块半宽 -> 全宽                                9 errors
捕获  卡片 active 标志丢失                              1 failure
捕获  urgency 钳位 1.5 -> 1.0                           1 failure
SKIP  _slot 边界 `<` -> `<=`                            （unknown == count，两式在所有调用点等价）
SKIP  布局守卫改为 `if False`                            （cursor 仅由模块常量决定，对所有输入恒成立）
SKIP  设备锚点改回 next(self.parameters()).device        （纯性能改动，无行为差异 —— 本就不该被捕获）
14/17 mutations detected
```

前两条 SKIP 是**构造性等价变异**，不是测试缺口：`UNKNOWN_PLANT_SLOT == PLANT_TYPE_COUNT`
（49==49）、`UNKNOWN_ZOMBIE_SLOT == ZOMBIE_TYPE_COUNT`（100==100），已穷举验证两式无差异；
`cursor` 只依赖模块常量，与观测无关。

---

## 6. 精度预算：容忍误差能不能换速度？

结论：**不能，而且不是因为误差预算不够，而是因为这份代码的时间根本不在精度敏感的算术上。**
实测脚本：`scripts/precision_budget.py`、`scripts/decision_margin.py`。

### 6.1 误差预算是多少

真实模拟器 level 7、seed 30000–30003、`budget=256`，逐步执行 **160 次真实决策**，统计
`best_second_margin`（top1 与 top2 的分数差）：

```
top1-top2 margin     n=160  min=0.000e+00  p10=0.000e+00  median=6.222e-04  p90=3.950e-02  max=1.043e-01
候选分数极差          n=160  min=3.634e-02  p10=4.006e-02  median=7.477e-02  p90=1.250e-01  max=1.784e-01

ε            margin < ε 的决策占比
1e-9              10.6%     ← 精确并列
1e-7              10.6%     ← float32 舍入量级
2.5e-5            15.0%     ← fp16 前向实测误差
6.6e-5            19.4%     ← bf16 前向实测误差
1e-3              56.2%     ← 宽松的近似值模型
1e-2              60.6%     ← 激进的近似值模型

排除精确并列后（143 次）：最小 3.868e-06，中位 8.069e-04
  其中 margin < 1e-7 的：0/143 (0.0%)
  其中 margin < 1e-5 的：2/143 (1.4%)
  其中 margin < 6.6e-5 的：14/143 (9.8%)
```

**读法**：有 10.6% 的决策 top1/top2 **精确同分**（margin 恰为 0），翻转它们没有质量损失。
排除这些之后，**没有任何一次决策的分数差低于 1e-7**，也就是说 float32 舍入量级的误差
（~1e-7）在 143 次决策里一次都翻不动。误差预算大致在 **1e-5 ~ 1e-4** 之间；
真正危险的是 1e-3 以上（会翻转超过一半的决策）。

### 6.2 花这个预算能买到什么

在 `SearchValueModel`（batch=1，即搜索叶子的真实形状，基线 15.72 µs）上实测：

| 方案 | 速度 | 输出误差 max\|d\| |
|---|---|---|
| 特征累加器 float32 | **1.00×（零收益）** | 0（该观测无碰撞；有碰撞时 ~1e-7） |
| fp16 权重 + fp16 输入 | **0.27×（慢 3.7 倍）** | 2.5e-05 |
| 稀疏首层（nonzero gather） | **0.18×（慢 5.6 倍）** | 3.7e-09 |
| bf16 autocast（CPU） | **0.12×（慢 8.3 倍）** | 6.6e-05 |
| 仅首层 bf16 | **0.09×（慢 11 倍）** | 1.9e-05 |

**全部四个方案都比全精度基线更慢。** 根因已定位（**本节曾用错误的证据，已更正**）：

> 最初这里写的是「本机不存在低精度向量化路径」，依据是
> `torch.backends.mkldnn.is_available() = False` 与
> `torch.backends.cpu.get_cpu_capability() = DEFAULT`。
> **这两个探针是 x86 专用的**：官方 macOS arm64 wheel 本来就 `USE_MKLDNN=OFF`，
> 而 `get_cpu_capability()` 在 ARM 上恒返回 `DEFAULT`，与硬件能力无关。
> 用它们推断 ARM 上有没有 bf16/fp16 路径是无效推理。

在 Apple Silicon 上按真实负载形状重测（`scripts/device_benchmark.py` 第 5 段），
结论**反而更强**：混合精度不是「没有路径」，而是**有路径但更慢**。

```
机器   Apple M5 Pro, 15 核 (5 P + 10 E), uname -m = arm64
torch  2.13.0 官方 Apple Silicon wheel（libtorch_cpu.dylib 为 Mach-O arm64）
构建   BLAS_INFO=accelerate  LAPACK_INFO=accelerate  USE_MKL=OFF
       USE_MKLDNN=OFF  USE_MPS=ON  AT_BUILD_ARM_VEC256_WITH_SLEEF  USE_OPENMP=ON

fp32 基线（手工函数式）        10.38 us
fp16 权重 + fp16 输入          43.11 us   0.24x   max|d|=6.951e-06
bf16 首层（fp32 其余）         31.29 us   0.33x   max|d|=1.741e-05
```

原因是硬件层面的：Apple 的矩阵协处理器（AMX）是 **fp32** 单元，
M 系列芯片没有为 fp16/bf16 提供比 fp32 更高的吞吐路径。
所以「用低精度换速度」在这台机器上是**结构性不可行**，与误差预算多宽松无关。

**CPU 线程数同样不是杠杆**（第 6 段）：threads=1/2/5/10/15 下 batch=1 恒为 15.1–15.3 µs，
batch=256 恒为 272.7–275.2 µs。Accelerate/AMX 已经在单线程里吃满了这条路径。
（顺带确认：`torch.get_num_threads()` 默认返回 5，正好是 M5 Pro 的 P 核数，不是配置错误。）

### 6.3 为什么精度本来就不是杠杆

| 环节 | 占 `advice()`（参考模拟器 8.6 ms） | 有精度杠杆吗 |
|---|---|---|
| `search_value_features` | ~43% | **没有** —— 全是 Python 循环/字典访问，已做到逐位精确 |
| 值网络前向 | ~27% | 有，但**每个降精度方案都更慢**（见 6.2） |
| 候选生成 | ~12% | 没有 —— 纯 Python 集合运算 |
| 记账/剪枝/去重 | ~18% | 没有 —— 纯 Python |

在**真实模拟器**下（`advice()` = 164.6 ms）更极端：**66% 是等待 C++ 进程返回**，Python 侧不到 20%。
即使把 Python 侧优化到零耗时，端到端也只能从 164.6 ms 降到约 56 ms（2.9×）——
而这 2.9× 与精度毫无关系，它是 §5.5 里那个 C++ 快照保存/恢复问题。

**另一个反直觉的数据点**：特征向量只有 **1.1%–3.3% 非零**（46–134 / 4116）。
直觉上「97% 是零，那就跳过零」应该快很多，但 `Linear(4116,128)` 在 batch=1 只要 6.59 µs
（2.1 MB 权重命中 L2/L3），而 `torch.nonzero` + fancy indexing 的稀疏版本要 88.57 µs
——索引构造的开销远超省下的乘加。**稀疏 ≠ 快，尤其在这种小批量、缓存驻留的形状上。**

### 6.4 真正该问的问题

不是「能容忍多少误差」，而是「**能不能少算一些**」。两者完全不同：

- **少算**（合法）：减少叶子估值次数、剪枝更激进、把 C++ 往返批得更狠 —— 这些改的是**算法**，
  不是数值精度，而且效果立竿见影（`BRANCH_SNAPSHOT_FAST` 已经证明批量下沉 C++ 是有效的）。
- **算粗**（无效）：降精度、跳零 —— 在这份代码里要么零收益，要么负收益。

唯一的例外是**训练**：`train_search_value` 用 batch=256 的前向，那个形状下 GEMM 才真正
FLOP-bound。但训练用的是同一套权重，降低精度等于训练另一个模型，属于「换模型」而非「容忍误差」。

### 6.5 Apple Silicon：真正的加速路径，与一个默认值缺陷

问题：「Apple Silicon 上没有更快的路径吗？是不是 torch 装错了？」
实测脚本：`scripts/device_benchmark.py`、`scripts/device_policy.py`。

**torch 没装错**。上面第 6.2 节的构建信息就是官方 Apple Silicon wheel 的正常形态：
`libtorch_cpu.dylib` 是 `Mach-O 64-bit ... arm64`，`sysconfig.get_platform()` 返回
`macosx-11.0-arm64`，BLAS 走 Accelerate。`get_num_threads() = 5` 也是对的（= P 核数）。

**MPS 在 batch=1 上全面更慢**，而这份代码里**每一次模型调用都是 batch=1**：

| 调用点 | 频率 | CPU | MPS | 结论 |
|---|---|---|---|---|
| `SearchValueModel.predict` | 每决策 ~250 次 | 68.16 µs | 451.49 µs | **MPS 慢 6.6×** |
| `GameplayModelV1.step` | 每决策 1 次 | 3864.83 µs | 4726.62 µs | MPS 慢 1.22× |
| value forward batch=256 | 训练内 | 274.54 µs | 187.01 µs | MPS 快 1.47× |
| encoder batch=64 | **训练路径里不存在** | 70022.50 µs | 28696.27 µs | MPS 快 2.44× |
| 学生网络 16 步前向+反向 | 训练内 | 13774 µs/step | 10664 µs/step | MPS 快 1.29× |

关键点：**「encoder batch=64 快 2.44×」在真实训练路径里拿不到。**
`pvz_imitation.train` 的 `for start in range(0, len(steps), 64)` 只是**梯度累积窗口**，
循环体里仍然是逐条 `model.step()`，GRU 隐状态串行传递 —— 张量维度始终是 1。

真实模拟器端到端（level 7、seed 30000、`budget=256`、104 次决策）：

```
一次完整 rollout 采集     CPU 13.22 s   MPS 22.56 s   →  MPS 慢 1.71×
单次 advice()            CPU 162.28 ms MPS 282.38 ms →  MPS 慢 1.74×
  其中叶子估值占比         8.5%          47.4%
  单次叶子估值            54.54 µs      526.70 µs    →  MPS 慢 9.66×
```

**净账**（一次完整训练：192 集 rollout + 8 epochs × 64 集训练）：

```
rollout  192 × 13.22 s  = 42.3 min  →  ×1.71  = 72.4 min     亏 +30.1 min
train    32768 × 13.774 ms = 7.5 min →  ×0.774 = 5.8 min     赚 −1.7 min
                                                            ─────────────
                                          净亏 +28.4 min（整体约 1.55× 慢）
```

**缺陷**：`resolve_device("auto")` 原本是 `cuda → mps → cpu`，于是
`train_pvz_agent.py:323` 的默认 `--device auto` 在 Apple Silicon 上**必然选中 MPS**，
而 `:372` 把 `SearchValueModel` 也绑到了同一个 `device` —— 搜索叶子估值直接慢 9.7×，
把 §5 里 Python 侧全部优化成果一次性吃掉，还倒贴 28 分钟。

**已修复**：`resolve_device("auto")` 改为 `cuda → cpu`，不再考虑 MPS。
MPS 仍可通过显式 `--device mps` 使用。理由与全部实测数字都写在 `pvz_agent_model.py`
的 docstring 里，并由 `test_agent_model.py::test_auto_never_selects_mps` 锁定
（该测试对旧实现会失败，已做变异验证）。

**数值影响**：MPS 与 CPU **不是逐位一致**。在真实模拟器上对比 24 个候选分数与 24 个
策略概率，最大绝对偏差 **7.3e-9**（相对 ~1e-7），而 §6.1 测出的误差预算是 1e-5 ~ 1e-4
—— 偏差比预算小 3 个数量级，选出的动作完全一致。但它确实破坏了「逐位等价」这条标准，
这也是把 MPS 移出默认路径的第二个理由。

**如果将来想把 MPS 用起来**，唯一有意义的做法是拆设备：值模型留 CPU（它占 94% 的时间），
学生模型放 MPS（训练阶段 1.29×）。但训练只占 7.5 min / 50 min，收益约 1.7 min（3.4%），
还要引入双设备参数与设备相关数值，**当前不值得**。真正的杠杆仍然是 §5.5 的 C++ 快照开销。

### 6.6 线程数：把复现性 pin 变成参数

问题：「不需要绝对精度一致，在尽量不影响研究的前提下，可以少量牺牲精度换速度。」
实测脚本：`scripts/thread_effect.py`（两台机器都跑过）。

三个训练入口此前都把 `torch.set_num_threads(1)` 硬编码在 `random.seed` /
`torch.manual_seed` **紧邻处** —— 意图明确是复现性，不是性能。但它的代价两台机器完全不同：

| 量（相对 `threads=1`） | i7-12700F + RTX 5080 | Apple M5 Pro |
|---|---|---|
| leaf `predict` | **2.0×（threads=4 最优）** | 平坦（Accelerate 已吃满单线程） |
| 64 步前向+反向 | **1.7×** | 平坦 |
| 一次完整训练运行 | 32.0 min → **21.3 min** | 无变化 |

线程数**确实会扰动数值**，所以这一步不是「无代价的开关」，必须量化：

| 量（相对 `threads=1`，相对误差） | i7-12700F | M5 Pro |
|---|---|---|
| value forward | 4.0e-07 | 0（逐位一致） |
| gameplay `step` | 2.3e-07 | 0（逐位一致） |
| 64 步训练窗口后的权重 | 6.1e-07 | 7.0e-10 |

对照 §6.1 实测的决策误差预算 **1e-5 ~ 1e-4**：全部偏差低 **2 个数量级以上**，
翻不动任何一次搜索决策。作为参照，同一次测量里「换设备（CPU↔MPS）」的偏差是 7.3e-9 ——
线程数的扰动比它还小一个量级。

**落地方式**：

* `pvz_agent_model.configure_torch_threads(requested)` 负责设置并**返回实际生效的值**，
  默认 `DEFAULT_TORCH_THREADS = 4`（`requested <= 0` 时取默认）。
* 四个入口（`train_pvz_agent.py`、`train_pvz_ppo.py`、`benchmark_pvz_agent.py`、
  `pvz_search_audit.py`）新增 `--threads`。`train_pvz_agent.py` 的 `provenance()`
  会把 `vars(args)` 整个记进 checkpoint，所以**线程数自动进溯源记录**，不需要额外字段。
* 需要与旧产物逐位对齐时用 `--threads 1`。
* 三个单元测试锁定契约：默认值与「`<=0` 即默认」的映射、显式值被原样应用且原样返回
  （防止 setter 变成 no-op 而返回值照旧撒谎）、同一线程数下两次前向逐位一致。

**为什么不是「无脑调大」**：`threads=16` 在 20 逻辑核的 i7-12700F 上比 `threads=4` 更慢 ——
batch-of-1 的形状下，线程同步开销超过了并行收益。4 是两台机器上实测的最优点，
不是「越多越好」的猜测。

---

## 7. 本轮一并落地的三项改动

这三项与性能无关，但和工作树里同一批未提交改动一起落地，且都直接影响后续实验的可解释性，
故一并记录。

1. **搜索预算记账（`SearchAdvice` 新增 5 个字段）**。`simulation_budget` **不是搜索深度**：
   每个根候选在任何一条线被展开前都要先花 1 次模拟，所以真正留给 `_successive_halving`
   的是 `simulation_budget - screening_simulations`。以前只记 `simulation_count`，
   于是「`budget=128` 的跑分」被误读成「256 的一半」，其实它是一个不同的工作点。
   现在 `screening_simulations` / `depth_simulations` / `effective_depth_budget` /
   `root_actions_generated` / `root_candidates_screened` 逐决策落盘。
2. **策略分布蒸馏（`pvz_imitation.SOFT_LABEL_WEIGHT`，默认 0.5）**。搜索教师本来就把整个
   根候选集上的分布写进了每一步，只对 argmax 做行为克隆等于把它扔掉。新增
   `soft_behavior_cloning_loss` 项，`--soft-label-weight 0` 精确还原旧行为。
   DAgger 步骤带同样的标签，所以旧数据只是被跳过，而不是要重采一轮。
3. **搜索审计工具（`python/pvz_search_audit.py`，18 个用例）**。值模型是在教师**访问过**的
   状态上训练的，却被查询在每个根候选的**反事实一步子节点**上 —— 训练集上的 MSE 检测不到
   这种错配。工具用 `sibling_ranking` 直接测「兄弟排序是否与更深搜索一致」，
   用 `candidate_recall` 把生成与筛选两级的召回损失分开测，并只允许 development 种子。

顺带修掉的两个对齐缺陷：`_expand_actions` 以前把 `zip(actions, ...)` 与自己的
`fit_action_to_remaining` 过滤配在一起，一旦某个动作因剩余 horizon 不足被丢弃，
**后面每个动作都会对上邻居的结果**；现在 `_expand_paired` 从一处同时返回两个对齐的列表。
`_branch_snapshot_fast` 现在显式拒绝超过 `BRANCH_BATCH_LIMIT = 128` 的批量
（`src/main.cpp` 把 `branchCount` 上界设为 128），审计工具按 128 分块展开全量合法动作。

---

## 8. 遗留事项

1. **CI 不跑 Python 测试**。`.github/workflows/ci.yml` 只构建 C++。P14/P15 这类「测试全绿但生产代码一调就炸」的缺陷，只要 CI 里有一步 `unittest discover` 就会当场暴露。建议加入。
2. **`test_env_protocol.py` 的 manifest 用例依赖 git 工作树**。它在非 git 检出里会自动 `skip`，在 CI 的浅克隆上需要确认 `git diff --binary` 可用。
3. **`verify_env_equivalence.py` 仍无自动化测试**——它需要真实的 `pvz-portable` 二进制与资源包，只能在集成环境里跑。已做的只是把它的 `digest()` 透传包装删掉、统一走 `pvz_common.canonical_digest`。
4. **真实模拟器下 `advice()` 的 66% 花在等待 C++ 返回**（见 5.5）。下一个吞吐瓶颈在 `src/main.cpp` 的
   `BRANCH_SNAPSHOT_FAST`：每分支 ~290 µs 固定开销，主要来自 `LawnSaveGameToMemory` /
   `LawnLoadGameFromMemory` 与对整块 `board` 字节做 FNV-1a 的 `EnvironmentSnapshotHash`。
   改这里必须先有可自动化的 `verify_env_equivalence.py` 回归（逐 tick 可视/无画面差分 + `debug_state_sha256` 全状态比对），
   再动序列化格式。
5. **设备处理只在 CPU 上被测过**。`test_predict_matches_an_explicit_device_transfer` 在 CUDA/MPS 主机上
   会自动变成有效回归，但 CI 是 CPU-only；`_device_anchor` 若被人为改坏成固定 `cpu`，在 CI 上不可见。
   `test_auto_never_selects_mps`（§6.5）同样只在 Apple Silicon 上有区分度 —— 在 CUDA 主机上
   `auto` 本来就走 `cuda`，该断言恒真。MPS 分支的实际行为由 `scripts/device_benchmark.py`
   与 `scripts/device_policy.py` 在真机上覆盖，这两个脚本**不参与 CI**。
6. **`test_pvz_search_teacher.py` / `test_pvz_training_semantics.py` 的删除已在 `769e44c` 落地**。历史上同一份测试曾同时存在于 4 个文件、39 个用例（见 T7）。提交信息里已记录这次合并，避免再次分裂。
7. **仓库此前没有任何依赖声明文件**，本轮新增了 `requirements.txt`（`torch`、`numpy`）。注意
   `numpy` **不是** torch 的依赖（已核对两个包的 `Requires-Dist`），是本次为性能优化显式引入的。
8. **线程数策略的验证是「两台机器 + 一次实测」，不是持续保证**。`--threads` 的三个单元测试
   （§6.6）是纯逻辑断言，在 CI 上完全有效；但「4 是最优点」这个**结论**只由
   `scripts/thread_effect.py` 在两台具体机器上支撑，该脚本不参与 CI。换 CPU 架构
   （更多 P 核、NUMA、AMD 的 CCD 拓扑）时应重跑一次，而不是照抄 4。
9. **`scripts/thread_effect.py` 与 `scripts/device_benchmark.py` 对同一件事给出的倍数略有差异**
   （线程数 leaf 2.2× vs 2.0×，训练 1.6× vs 1.7×）。原因是两者的测量路径不同：
   前者用合成观测、后者用真实训练内循环，且都没有做多轮取中位数。**结论方向一致，
   具体倍数不可当精确值引用。**
