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
- 改成复用 18 worker：**约 10 s**
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

### 4.4 为什么默认关闭

每个 rollout worker 都要付一次性编译成本（实测 flex 的编译 warmup 约 6 s）。
18 个 worker × 6 s 是并行的，摊到数千局可忽略，**但必须先验证**。
建议在台式机上先跑 `--workers 1` 对比开启/关闭的端到端吞吐，再决定是否默认开。

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

| 优先级 | 改动 | 预计收益 | 风险 |
|---|---|---|---|
| **P0** | `episode_hash` 改为字节摘要 | 每轮省 ~286 s（占全轮约 6%） | 低；会改 `trajectory_sha256` 取值，**必须在 T5 开跑前做** |
| **P0** | 评估改用 18 worker 并行 | 每轮省 ~663 s（占约 13%） | 低；每局由 (task, seed) 决定，结果不变 |
| **P1** | 中间评估跳过 `reference_set` | 评估再省 33% | 低；参考集不设阈值 |
| **P1** | 关系偏置融合默认开启 | rollout 1.30×、更新 1.42× | 中；需先验证编译开销 |
| **P2** | 跨决策增量编码 | 未测 | 高；需先测相邻决策的 token 变化率 |

**P0 两条加起来可省约 19% 的全轮时间，且都不改变任何训练语义或判据。**

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

---

## 8. 复现

```bash
cd /Users/newbiexvwu/PvZAgent

# 全流程热点（需要重编译过的协议 4 二进制）
python scripts/training_hotspot_profile.py --episodes 6 --update-episodes 8

# 关系偏置优化
python scripts/relation_bias_benchmark.py --repeats 200

# 单决策拆解
python scripts/attention_cost_profile.py

# 等价性
cd python && python -m unittest test_relation_bias_optimization
```

**前置**：本机 PvZ 二进制需为协议 4（`cmake --build build -j6`）。
