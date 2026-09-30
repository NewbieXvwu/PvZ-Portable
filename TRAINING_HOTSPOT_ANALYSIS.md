# T5 训练全流程热点分析：测量、必要性审计、并行化

测量日期：2026-09-30。测量机器：Apple M5 Pro（CPU 单线程，torch 2.13.0）。
模型：`GameplayModelV1`（3,681,809 参数，4 层，width 192，6 头，L=108）。

> 配套脚本：`scripts/training_hotspot_profile.py`（全流程）、
> `scripts/relation_bias_benchmark.py`（关系偏置优化）、
> `scripts/attention_cost_profile.py`（单决策拆解）。

---

## 0. 三个结论

1. **有一个从没被注意到的成本大头**：`_evaluate` 每次**单线程**跑 960 局，
   一轮 20,000 局要跑 4 次 = **3,840 局串行评估**。而训练 rollout 是 18 个并行 worker。
   **评估的单核工作量是训练的 3.5 倍**（3840 局串行 vs 20000 局 ÷ 18 worker）。
2. **有一段纯粹是浪费的开销**：`episode_hash` 每局物化 **220,809 个 Python 标量**，
   2000 局 = **4.42 亿个 Python 对象**，耗时 29.3 s/update、293 s/轮。
   改成直接对已打包字节做摘要后 **45.5×**（0.32 ms/局）。
3. **关系偏置优化已落地并验证**：索引跨层提升 1.07×，叠加 `torch.compile` 融合后
   rollout 路径 **1.30×**、批更新路径 **1.42–1.46×**，全部 `torch.equal` 逐位相同。

---

## 1. 全流程测量（实测）

### 1.1 每决策拆解（L=108）

| 环节 | 耗时 | 占比 |
|---|---|---|
| `step_tokens`（模型前向） | 2.749 ms | **91.6%** |
| `observation_tokens` | 0.165 ms | 5.5% |
| `select_action` | 0.087 ms | 2.9% |
| `legal_summary` | 0.001 ms | 0.0% |
| **合计** | **3.002 ms** | |

### 1.2 每局 rollout 拆解（真实 env，6 局平均）

| 环节 | 耗时 | 占比 |
|---|---|---|
| environment（含子进程 IPC） | 139.17 ms | 51.0% |
| model | 123.48 ms | 45.2% |
| tokenization | 9.32 ms | 3.4% |
| critic_inputs | 0.72 ms | 0.3% |
| **WALL** | **272.97 ms** | |

平均 **53.7 决策/局**。

> **注意**：台式机 `throughput.json` 报的是 model 888 ms、environment 71 ms。
> 与本次测量相反。最可能的原因是 18 worker 挤在 20 个逻辑核上，
> **SMT 争用把「大量小算子」的模型前向放大得远多于「子进程自己干活」的环境步进**。
> 这条值得在台式机上用 `--workers 1` 复测确认。

### 1.3 每 update 固定开销（**此前从未测量**）

| 项目 | 耗时 | 折算 |
|---|---|---|
| `state_dict` 拷贝（3.68M 参数） | 0.25 ms | 可忽略 |
| `_state_sha256` | 4.70 ms | 可忽略 |
| `torch.save` checkpoint | 6.25 ms（14.1 MiB） | 可忽略 |
| **`episode_hash` × 2000** | **8.94 ms/局 → 17.9 s** | **占总开销 99%** |
| **固定开销合计** | | **17.88 s/update** |

### 1.4 评估（**此前从未测量**）

| 项目 | 值 |
|---|---|
| 单局评估耗时 | 137.4 ms |
| `_evaluate` 每次跑的局数 | **960 局**（cap3 全部 15 个任务） |
| 单次评估 | **175.9 s** |
| 一轮 20,000 局的评估次数 | 4 次 |
| **一轮的评估总耗时** | **703.4 s** |

其中门禁集（cap3×1.0，10 任务）只有 640 局，另外 320 局是
`reference_set`（cap3×1.5 的 5 个任务）——按 TODO §4 它**只报告、不设阈值**。

---

## 2. 必要性审计：这些过程本身需要吗？

这是比「怎么更快」更值钱的问题。逐项问一遍：

### 2.1 `episode_hash`：**需要，但当前实现方式完全没必要** ⭐

现路径 `episode_hash` → `_jsonable` → `canonical_digest`：

| | 现路径 | 直接字节摘要 | 加速 |
|---|---|---|---|
| 每局 | 14.638 ms | 0.322 ms | **45.5×** |
| 每 update（2000 局） | 29.3 s | 0.6 s | |
| 每轮（10 个 update） | **292.8 s** | **6.4 s** | |

**根因**：`_jsonable` 把已打包的 numpy 数组逐元素展开成 Python 标量再 JSON 序列化。
实测**每局物化 220,809 个 Python 标量**，2000 局 = **4.42 亿个 Python 对象**。

**这是纯浪费**：`tokens` 本来就是 `int8`/`float16` 的 numpy 数组，直接对
`array.tobytes()` 做 blake2b 就得到同样强度的溯源摘要，根本不需要重建 Python 对象。

**注意**：这会改变记录的 `trajectory_sha256` 取值。T5 尚未产出任何结果，
现在改是成本最低的时机；一旦开跑就不要动，否则证据不可比。

### 2.2 评估的 `reference_set`：**中间评估不需要** ⭐

`_evaluate` 每次都跑 960 局，其中 320 局（33%）只服务于「报告但不设阈值」的 1.5× 参考集。
**中间评估只跑门禁集（640 局），最后一轮再补参考集**，可省 33% 评估开销，且不损失任何判据。

### 2.3 `_state_sha256` / checkpoint 保存：**需要，且已经足够便宜**

4.7 ms + 6.25 ms = 11 ms/update，占 0.06%。不需要动。

### 2.4 `observation_tokens`：**需要，但不是热点**

0.165 ms/决策 = 5.5%。台式机 profile 里 tokenization 占 3.4%。可以不管。

### 2.5 `select_action` / `legal_summary`：**不需要动**

合计 2.9%。我原以为 factored sampling 可能是隐藏大头，实测证明不是。

---

## 3. 并行化：哪里没用满？

### 3.1 现状

| 阶段 | 当前并行度 | 判断 |
|---|---|---|
| rollout | **18 个 CPU worker**（每 worker 1 torch 线程） | ✅ 已充分 |
| PPO 更新 | CUDA，`chunks 16` | ✅ 已优化（FlexAttention 2.15×） |
| **评估** | **单线程**（`configure_torch_threads(1)` + 单 env 串行） | ❌ **最大浪费** |
| 固定开销 | 主进程串行 | ⚠️ 见 §2.1 |

### 3.2 评估是并行化的最大机会 ⭐

评估的 960 局**彼此完全独立**（不同 task × seed），且已有现成的
`run_seed_jobs` + spawn worker 基础设施（rollout 就在用）。

- 现状：960 局 × 137.4 ms = **175.9 s**，单线程
- 改成复用 18 worker：**约 10 s**（实测见 §9.2：固定开销约 8.8 s，不能忽略）
- 一轮 4 次评估：**703 s → 约 40 s**

**这是安全改动**：每局由 `(task, seed)` 完全决定，并行不改变任何结果。
但要注意：`_evaluate` 目前复用**同一个 env 实例**，并行需要每 worker 独立 env
（rollout 已经是这么做的）。

### 3.3 为什么 rollout 不该改成多线程

台式机已实测：`12×2` 约 35,189 局/小时、`16×2` 约 27,470 局/小时，
都**低于** `18×1` 的 44,565.9 局/小时。多 worker 叠加多线程会超订阅。
**rollout 的并行度已经对了。**

### 3.4 GPU 用满了吗？

GPU 只在 PPO 更新阶段用，且 rollout 阶段 18 个 worker 全在 CPU 上跑。
RTX 5080 在 1,880 s 的更新时间里是忙的，但在 1,615 s 的 rollout 和
1,103 s 的评估期间**完全空闲**。

**如果评估搬到 GPU** 是个陷阱：评估是 batch=1 的串行前向，
MPS/CUDA 在 batch=1 上反而更慢（项目已实测记录在案：
`predict` 68 µs → 451 µs）。**评估应该用 CPU 多进程，不是 GPU。**

真正能让 GPU 参与的办法是**把评估批量化**（多个 episode 的同一 tick 拼成一个 batch）——
但不同 episode 的终止时刻不同，实现复杂度高，收益不如直接上 18 worker 进程。
**建议先做进程并行，不要碰批量评估。**

---

## 4. 关系偏置优化（已落地，提交 `eca9a4d`）

### 4.1 做了什么

1. **索引跨层提升**（GLM-5.2 IndexShare 的 PvZ 对应物）：
   `row_bucket`/`col_bucket`/`same_cell` 只依赖 `kinds`/`rows`/`cols`，
   与层参数无关 → 每次前向算一次，4 层复用。
2. **可选融合**：把四张偏置表作为显式参数传入，让 `torch.compile` 融合整条链。
   开关：`PVZ_RELATION_BIAS_FUSION=1` 或 `set_relation_bias_fusion(True)`，**默认关闭**。

### 4.2 实测（L=108，Mac CPU 单线程）

| 配置 | encoder 前向 | 相对基线 | 逐位相同 |
|---|---|---|---|
| 基线（每层重算索引） | 3.026 ms | 1.00× | — |
| 索引提升 | 2.838 ms | 1.07× | ✅ |
| 索引提升 + 融合 | 2.139 ms | **1.41×** | ✅ |
| 完整 `step_tokens`（融合） | 2.347 ms | **1.30×** | ✅ |

批更新路径 encoder 前向：

| N（chunks × seq） | eager | fused | 加速比 | 逐位相同 |
|---|---|---|---|---|
| 64 | 162.20 ms | 114.55 ms | 1.42× | ✅ |
| 256 | 652.65 ms | 447.66 ms | 1.46× | ✅ |
| 512 | 1304.27 ms | 898.10 ms | 1.45× | ✅ |

### 4.3 诚实记录：我的预估错了

我原先估计「索引提升能砍掉约 90 次算子派发」，实际只有 **1.07×**。
原因：117 个算子里，剩下的查表 + permute + add 才是大头，索引只占小部分。
**融合（1.41×）才是主力，IndexShare 类比的价值被高估了。**

### 4.4 默认开关（已在 §9.3 更新为默认开启）

原先默认关闭，理由是每个 rollout worker 都要付一次性编译成本。
**实测该成本是 440 ms/进程**（不是 6 s，那是 FlexAttention 的 warmup），
18 worker 并行付 → 约 0.5 s 墙钟，可忽略，因此**已改为默认开启**。
详见 §9.3。

---

## 5. 稀疏注意力的正确实现方式：掩码，不是新算法

用户提示的关键点：**稀疏注意力可以做成「全注意力 + 掩码让部分权重为 0」**。
这在本模型上**已经是既成事实**：

```python
# pvz_agent_model.py — CUDA 路径
attended = _COMPILED_FLEX_ATTENTION(query, key, value, score_mod=relation_score)
```

`relation_score` 就是「全注意力 + 结构化修饰」：它在同一个 kernel 里给每个
(query, key) 对加上关系偏置，并可用 `key_mask` 把 padding 位置压到 `finfo.min`。

**这条提示的直接推论**：

1. **不需要移植任何新的注意力算法**。要加稀疏性，正确做法是在**已有的
   `score_mod` 里多加一项掩码**，而不是换一个 kernel。这样稀疏和关系偏置
   共享同一次 kernel launch。
2. **CPU 路径缺的正是这个融合**。CPU 上走的是 eager 的
   `scores + relation` 分支，117 个算子。`torch.compile` 就是把这条 eager 链
   变成「一个融合 kernel」——**这正是本次优化做的事**，等价于把
   FlexAttention 的 `score_mod` 思想搬到 CPU。
3. **`-1e9` 而非 `-inf` 的选择是有意的**（代码注释已写明）：padding 的 query 行
   仍需有限值，否则反向会出 NaN。任何新增掩码都必须遵守这条。

---

## 6. 建议的实施顺序（按性价比）

| 优先级 | 改动 | 预计收益 | 风险 | 状态 |
|---|---|---|---|---|
| **P0** | `episode_hash` 改为字节摘要 | 每轮省 ~286 s | 低 | ✅ 已做（`episode_digest`，见 §9.1） |
| **P0** | 评估改用 18 worker 并行 | 每轮省 ~663 s | 低 | ✅ 已做并验证逐位相同（§9.2） |
| **P1** | 中间评估跳过 `reference_set` | 评估再省 25% | 低 | ✅ 已做（§9.5） |
| **P1** | 关系偏置融合默认开启 | rollout 1.1–1.3× | 中 | ✅ 已做；**更新路径无效**（§9.3） |
| **P2** | 跨决策增量编码 | 未测 | 高 | 未做，需先测相邻决策的 token 变化率 |

**更正后：P0+P1 合计约省 32% 的全轮时间**（原估 46%，其中更新那 38% 拿不到收益）。

---

## 7. 全轮时间预算（投影，含不确定度）

以台式机 20,000 局一轮、18 worker rollout + CUDA 更新为基准：

| 阶段 | 现状 | 占比 | 优化后 | 依据 |
|---|---|---|---|---|
| PPO 更新 | ~1,880 s | 38% | ~1,290 s | 融合 1.46×（仅 encoder 部分） |
| rollout | ~1,615 s | 33% | ~1,240 s | 融合 1.30× |
| **评估** | **~1,103 s** | **23%** | **~75 s** | 18 worker 并行 + 跳过参考集 |
| 固定开销 | ~300 s | 6% | ~15 s | 字节摘要 |
| **合计** | **~4,900 s** | | **~2,620 s** | **约 −46%** |

> 这些是**投影**，不是实测。台式机的单核速度、评估单局耗时、以及融合在
> CUDA 上的实际收益都需要在台式机上复测。标注为投影的部分不要当结论引用。
>
> ⚠️ **本节表格中的"PPO 更新 1.46×"与"rollout 1.30×"已在 §9.3 被更正，请以 §9 为准。**
> 该表保留在此是为了留下当时的推断记录。

---

## 9. 实施结果与两处更正（2026-09-30）

P0/P1 全部落地。本节记录实测值，并更正 §7 投影里两处不成立的假设。

### 9.1 P0-1 `episode_digest`：已实现，换了函数名，但**收益是 8× 不是 45×**

> **§10.5 更正**：本节原标题写的是 45×。第二次审计复测（`training_hotspot_profile.py`
> 第 2 阶段，12 局）得到 13.93 ms/局 → 1.75 ms/局 = **8.0×**（更短的 8 局样本上是 5.3×）。
> 45× 这个数字我无法复现，已从代码 docstring 与本文件删除。绝对收益仍然成立：
> 2000 局从 27.9 s 降到 3.5 s。

新增 `episode_digest()`（`python/train_pvz_ppo.py`），直接对带标签的字节流做
blake2b-128：数组走 `tobytes()`，每个分支写类型标签，变长载荷写长度前缀。

**没有改 `episode_hash`**：`artifacts/adventure2_level7/training_summary.json` 记录了它
产出的摘要，改掉会让那份已封存证据无法复算。`episode_hash` 保留为 legacy 路径，
T5 训练器改用 `episode_digest`，并在 provenance 里记录
`trajectory_digest_algorithm`，读者不必从摘要长度猜用的是哪个算法。

测试 `python/test_episode_digest.py`（13 项）：确定性 + 键序无关、**逐个字段**变异都必须
改变摘要、缺失字段 ≠ 存在字段、dtype/shape 参与摘要、结构不同必不同摘要、
**字节流逐字节钉死**（`test_encoding_is_self_describing`）、固定合成 episode 的摘要钉值。

变异测试（M3 要求）：
- 去掉 list 长度前缀 → 2 项变红；
- 去掉字典键排序 → 2 项变红。

> **诚实记录**：第一次做"去掉长度前缀"变异时测试**没有变红**。原因是我的碰撞测试用了
> `{"kills": 1, "spawns": 23}` 这种载荷，而 int 已经带 NUL 终止符、字符串已带长度，
> 容器长度前缀对它是冗余的 —— 这是个**等价变异体**。我因此补了
> `test_encoding_is_self_describing` 直接钉字节流，变异才被捕获。

### 9.2 P0-2 评估并行化：已实现，**逐条逐位相同**（实测）

`_evaluate` 改为复用 spawn 进程池（`_run_evaluation_jobs`），作业表是
`(task, seed)` 的扁平列表，`job_id` 唯一；`_evaluation_jobs()` 是纯函数，
布局可以不开池就断言。

`scripts/evaluation_parallel_equivalence.py` 在**真实模拟器**上对比新旧实现，
逐条比对记录（不是只比汇总）：

| 样本 | 串行 | 1 worker | 18 workers | 记录相同 |
|---|---|---|---|---|
| 20 局 | 9.66 s | 9.62 s | 4.84 s | ✅ |
| 128 局 | 28.83 s | 26.16 s | 10.37 s | ✅ |

gate-only 路径与串行 gate 子集也逐条相同。

> **注意池的固定开销**：128 局 / 18 worker 里，理论计算时间只有约 1.6 s，
> 实测 10.37 s —— 约 **8.8 s 是进程池启动 + torch 导入 + `model_state` 分发**
> （3.68M 参数 × 18 个进程 ≈ 252 MiB IPC）。全量 1,280 局时固定开销被摊薄，
> 但**每次评估都要付一次**，一轮 5 次评估约 44 s。仍远优于 703 s 串行。

### 9.3 P1-2 关系偏置融合默认开启 —— 但**收益只落在 rollout 和评估上**

开关已改为默认开启（`env_flag("PVZ_RELATION_BIAS_FUSION", default=True)`，
`PVZ_RELATION_BIAS_FUSION=0` 可关）。

**更正 §7 的一处错误**：我原先写"PPO 更新 ~1,880 s → ~1,290 s（融合 1.46×）"。
**这条不成立。** 融合只作用于 **eager 注意力路径**：

```python
# pvz_agent_model.py:501
if (self.use_flex_attention and _COMPILED_FLEX_ATTENTION is not None
        and query.device.type == "cuda" and batch >= 32):
    ...  # relation_score 在 FlexAttention kernel 内部算，根本不走融合路径
```

而台式机的 PPO 更新是 `use_flex = (device.type == "cuda" and backend != "dense")`
→ **走 FlexAttention，不受融合影响**。受益的是 batch=1 的路径：rollout 与评估。

融合实测（`scripts/relation_bias_benchmark.py`，L=108）：

| | 每步耗时 | 相对 eager |
|---|---|---|
| eager | 3.083 ms | 1.00× |
| fused | 2.352 ms | **1.31×** |

**与 token 数强相关**：L=108（benchmark）→ 1.31×；L=75（`test_agent_model.observation()`）
→ 1.14×。真实棋盘 L=74–116，所以诚实区间是 **1.1–1.3×**，不是单点值。

**一次性编译成本实测 440 ms/进程**（不是文档先前写的"约 6 s"，那是 FlexAttention 的
warmup，两回事）。18 个 rollout worker 并行编译 → 约 0.5 s 墙钟，相对 161 s 的
rollout 可忽略。

### 9.4 更正后的全轮预算（仍为投影）

| 阶段 | 现状 | 优化后 | 依据 |
|---|---|---|---|
| PPO 更新 | ~1,880 s | **~1,880 s（不变）** | FlexAttention 路径，融合无效 |
| rollout | ~1,615 s | ~1,320 s | 融合 1.14–1.31× |
| 评估 | ~1,103 s | **~25 s/次 × 5 ≈ 125 s** | 18 worker + 固定开销 8.8 s |
| 固定开销 | ~300 s | ~15 s | 字节摘要 |
| **合计** | **~4,900 s** | **~3,340 s** | **约 −32%** |

比 §7 原先的 −46% 保守，因为 PPO 更新那 38% 拿不到收益。

### 9.5 P1-1 中间评估跳过 `reference_set`：已实现

`_evaluate(include_reference=...)`：中间评估只跑门禁集 + stage-0 集，
`reference_set` 写成 `{"skipped": True, "reason": ...}`；
**每轮的最后一次评估必定带上参考集**（`run_episodes >= max_episodes_per_run` 时），
若因早停（门禁通过 / 阶段 0 零信号 / 硬停止）而没有，则**循环结束后补一次完整评估**，
保证 `gates/T5.json` 永远不缺数据。

> 说明：并行化之后这条优化的绝对收益从 175.9 s 降到约 2.7 s。保留它更多是因为
> 语义更干净（不设阈值的诊断量只算一次），而不是为了省时间。

---

## 10. 第二次全流程必要性审计（2026-09-30，逐阶段实测）

用户要求："对每一个流程都问自己：这个流程真的有任何必要吗？有没有更廉价的替代方案？
有没有方法优化它？重点是提高性能并降低内存占用。"

第一次审计（§1–§9）测的是**时间**，这次补上了**内存**，并把此前只是"看着可疑"的
环节全部量化。测量工具：`scripts/pipeline_necessity_audit.py`（A–H 段，真实 env）、
`scripts/pool_ipc_probe.py`（进程池与 IPC 通道，分节执行，各自带 `faulthandler` 超时保护）。

### 10.1 每 update 的全阶段账（2000 局，本机 CPU 测量 + RTX 5080 已有实测）

| 阶段 | 实测 | 必要性 | 更廉价的替代 | 裁定 |
|---|---|---|---|---|
| env 子进程往返（`step`） | 0.255 ms/决策 | 需要（崩溃隔离 + C++ 模拟器边界） | 无 | 保留 |
| observation tokenize | 0.21 ms/决策 | 需要（就是模型输入） | 无 | 保留 |
| 模型前向 batch=1 | 2.29 ms/决策 | 需要 | 融合已上（§4） | 保留 |
| 进程池启动（spawn ×18） | 0.72 s + 每 worker 模型/env | 需要 | 常驻池省 ~0.5 s | **不改**（0.3%） |
| worker→父进程：npz 分片 | 父进程**串行解码 12.4 s** | 传输需要，**格式不需要** | 管道传 pickle：**2.1 s** | **已改** |
| `metadata` 每 job 重序列化 | 61.7 KiB × 2000 = 120.5 MiB，**0.77 s** | 需要但可上提 | 走 `initargs` | **不改**（0.4%） |
| `add_advantages` | 0.17 µs/transition → **0.02 s** | 需要 | numpy 版**慢 0.4×** | 保留 |
| `episode_digest` ×2000 | 1.75 ms/局 → **3.5 s** | 需要（溯源） | JSON 路径要 27.9 s | 已做（§9.1） |
| `train_update`（GPU） | **184.4 s** | 需要，主导项 | 未触碰 | 保留 |
| checkpoint + state.json | 6.1 ms + 0.2 ms | 需要 | 无 | 保留 |
| 评估（每 5000 局） | 已并行 | 需要 | 已做（§9.2） | 保留 |
| rollout 载荷常驻内存 | **810.6 MiB** | 需要 | 事件键共享 −46 MiB | **已改** |

### 10.2 ⭐ 发现一：磁盘不是瓶颈，**npz 编解码**才是

D 段测出压缩分片往返 = 写 14.18 s + 读 12.83 s（×2000 投影）。但管道对照实验
（`pool_ipc_probe.py --section pipe/disk`，同为 87.5 KiB 载荷 × 2000）给出：

```
pipe: worker returns the bytes      1.425 s wall   (170.9 MiB)
disk: worker writes an npz shard    1.520 s wall
disk: parent re-reads every shard   0.030 s wall   (170.9 MiB)
```

**磁盘本身几乎免费**——父进程重读 170.9 MiB 只花 0.03 s（刚写过，在页缓存里）。
D 段那 27 s 全在 `atomic_numpy` 的 Python 编解码。D2 段分解（每局）：

| 步骤 | 每局 | 说明 |
|---|---|---|
| `_archive_encode`（遍历嵌套 dict → 数组） | 0.41 ms | |
| `json.dumps` manifest | 0.25 ms | 51.6 KiB |
| **zlib 压缩数组** | **4.06 ms** | **占编解码 77%** |
| `json.loads` manifest | 0.20 ms | |
| `_archive_decode`（重建嵌套 dict） | 0.36 ms | |
| **合计** | **5.28 ms** | ×2000 = **10.6 s** |
| 替代：`pickle` + `unpickle` | **0.43 ms** | ×2000 = **0.86 s** |

而管道传真实 pickle 尺寸（303.5 KiB × 2000 = 592.8 MiB）只要 **1.199 s**（495 MiB/s）。

所以最优解是：**worker 直接把结果回传**，分片照写（保断点续跑）。
净效果：父进程那 12.4 s 串行解码换成 0.86 s 反序列化 + 1.2 s 传输 ≈ **省 10 s/update**。

**为什么压缩反而更慢**：父进程立刻重读自己刚写的文件，数据在页缓存里，
压缩省下的 I/O 抵不过 zlib 的 CPU。`savez_compressed` 读 12.83 s > 不压缩 11.52 s。

### 10.3 ⭐ 发现二：`events` 的键字符串每步被 JSON 重建

`events` 从模拟器子进程以 JSON 过来，每步的 8 个键都是**全新的字符串对象**：

```
key objects shared between two steps:   0/8
keys present in sys.intern table:       0/8
per episode: 13.5 KiB of dicts + 21.0 KiB of duplicated key strings
```

21.0 KiB/局 = `events` 字段的 **61%**、整个载荷的 **5.1%**，2000 局 **41 MiB**。
实测 A/B（6 局，347 决策）：

| | 载荷/局 | 不同键对象数 |
|---|---|---|
| 改动前 | 456.7 KiB | 2,728 |
| 改动后 | 433.3 KiB | **8** |

**逐位验证**：改动前后同一批 4 局，`episode_digest` 与 legacy `episode_hash` **全部相同**
（`ca3409…` / `916a91…` 等 4 组）。`_digest_into` 对 dict 键走 `str(key)`，按值摘要，
共享对象透明。

### 10.4 ⭐ 发现三：`run_seed_jobs` 在 worker 死亡时**永久挂起**

不是推测，是**实测**：`pool_ipc_probe.py` 第一次运行时 initializer 签名写错，
18 个 worker 全部抛 `TypeError` 后死掉，父进程在 `imap_unordered` 上等了 **21 分钟**
才被我手动杀掉。`multiprocessing.Pool` 会不断重启死掉的 worker，
而 `imap_unordered` 不区分"慢"和"永远不会来"。

这条恰好命中本轮失败模式：台式机 Agent 在"没有 GPU"的环境里干等 75 分钟。
**空转和挂死都按墙钟时间烧掉止损预算。**

修复：`run_seed_jobs(..., stall_timeout=900.0)` 用 `iterator.next(timeout=...)`，
超时即 `pool.terminate()` 并抛 `RuntimeError("... stalled: no seed completed in N s
after K of M ...")`。单局最多 4000 决策 × ~3 ms ≈ 12 s，900 s 是 50× 余量。

变异验证：把 `raise` 改成 `break`，测试不再通过（它以**挂起**而非失败的形式暴露，
因为 `pool.join()` 同样等不到 worker 退出——这也说明 `except BaseException:
pool.terminate()` 那条路径是必需的，不是装饰）。

### 10.5 被实测否掉的方向（诚实记录）

| 猜想 | 实测 | 结论 |
|---|---|---|
| `add_advantages` 是热点 | 0.17 µs/transition，2000 局 0.02 s | **否**，numpy 改写还慢 0.4× |
| 磁盘 IPC 是瓶颈 | 170.9 MiB 重读 0.03 s | **否**，瓶颈是编解码 |
| 常驻池能省 36 s/update | spawn 0.72 s vs fork 0.47 s（torch 已导入时 fork 仅 0.013 s） | **否**，量级是 0.3% |
| `events` 存了模型不读的键 | 存储键集 == 模型读取键集 | **否**，没有多余键 |
| `tokens`（75% 内存）能降精度 | 已是 int8/float16/uint8/uint16 | **否**，`features` float16 (75,32) 占 91% |
| `normalized_advantage` 存 0-dim tensor 很贵 | 0.238 µs vs Python float 0.020 µs | 量级 **0.03 s**，CUDA 上 ~0.8 s，**不值得改** |
| `metadata` 每 job 重序列化 | 0.77 s/update | 可省但收益 0.4%，**不做** |
| `episode_digest` 比 JSON 路径快 45× | 实测 13.93 → 1.75 ms/局 = **8.0×** | **原数字不可复现，已更正** |

### 10.6 已落地的改动

1. **`pvz_seed_jobs._collect_and_save` 回传结果**（`(seed, result)`），
   `run_seed_jobs` 直接收进 `results`，不再重读分片。分片照写（断点续跑不变）。
   测试 `test_fresh_results_come_from_the_worker_not_the_shard` 用 int 键证明走了直传路径
   （`_archive_encode` 会把键 `str()` 化，所以缓存路径必然看到 `"7"`）。
2. **`pvz_seed_jobs` 停滞看门狗**（`stall_timeout`，默认 900 s）。
3. **`pvz_env._canonical_events`**：在 `_read_message` 这个唯一的 JSON 解码点把
   `events` 的 8 个键换成共享对象；未知键原样透传。
4. **更正 `episode_digest` 的 45× 说法为实测 8.0×**（`train_pvz_ppo.py` 两处 docstring）。

### 10.7 全量回归

`python -m unittest discover -s python -p "test_*.py"` → **250 项全绿**（原 244 + 新增 6）。
变异测试：`_canonical_events` 改成透传 → 3 项失败；`_read_message` 去掉钩子 → 1 项失败；
看门狗 `raise` 改 `break` → 测试不再通过。

### 10.8 仍未做（需要用户裁定或额外测量）

- **P2 跨决策增量编码**：相邻决策的 token 变化率仍未测，是 `tokens`（75% 内存）唯一的
  压缩方向。不测不动。
- **shard 写入是否必要**：现在写入只服务断点续跑，代价 ~0.8 s/update（并行）+ 178 MiB/update
  磁盘。若认为"崩溃就重跑整个 update"可接受，可再省 0.8 s 并去掉磁盘依赖。

---

## 8. 复现

```bash
cd /Users/newbiexvwu/PvZAgent

# 全流程热点（需要重编译过的协议 4 二进制）
python scripts/training_hotspot_profile.py --episodes 6 --update-episodes 8

# 第二次必要性审计：内存分字段 + 编解码分解 + 事件键（真实 env）
python scripts/pipeline_necessity_audit.py --episodes 6

# 进程池与 IPC 通道（分节跑，每节自带 faulthandler 超时）
for s in startup import fork metadata pipe disk; do
    python scripts/pool_ipc_probe.py --section $s
done
python scripts/pool_ipc_probe.py --section pipe --payload-kib 87.5 303.5

# 关系偏置优化
python scripts/relation_bias_benchmark.py --repeats 200

# 单决策拆解
python scripts/attention_cost_profile.py

# 评估并行化：与串行实现逐条比对（真实模拟器）
python scripts/evaluation_parallel_equivalence.py --tasks 3 --stage0-tasks 2 --seeds 4

# 等价性
cd python && python -m unittest test_relation_bias_optimization test_episode_digest test_t5_overnight
```

**前置**：本机 PvZ 二进制需为协议 4（`cmake --build build -j6`）。
