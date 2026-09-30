# PPO 更新的内部账目

**结论先说：一个 update 的 191 秒里，约 36% 是注意力里的关系偏置（relation bias），
63% 是反向传播，0.6% 是优化器，0.3% 是策略重放。主机侧的 Python 和 numpy 一共
占 0.4%。** 上一轮我怀疑"每个 minibatch 从 Python 对象重建张量"和"逐条 numpy
赋值"是瓶颈——**这个怀疑是错的，实测合计 0.38 ms / 620 ms**。真正的开销在张量
算术本身，而其中最大的一块是关系偏置。

本文记录全部测量，以及三条被推翻的假设，避免下一轮重复走这些路。

已据此实施了两项**位等价**优化（§6）：`replay_log_probs` 不再逐行写设备张量
（CPU 1.9x / MPS 8.8x），关系偏置的四次 permute 合成一次（前向 1.16–1.36x，
CPU update 每步 1.31x）。两者都不改模型语义，所以 `MODEL_ARCHITECTURE_VERSION`、
T4 的 seed-0 基线、阶段 0 的学习信号门都不受影响。

**另外，`rollout` 才是循环的另一半，而且它是模型受限的**（§5）：单核口径下每局
78–85% 的时间在模型前向，整个观测载荷只值 13.8%。**那份 09-29 的画像确实过期了**——
本轮用真实游戏资源（`~/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN`）重测，单核口径下
`model` 已经快了 **3.7x**（§5.1）。同时记录里 `workers=1`（6.84 ms/决策）与
`workers=18`（13.89 ms/决策）两个口径之前被当成一个用，实际差 2.03x 的纯 CPU 争抢
（§5）。

---

## 1. 基准事实（仓库里已有的记录）

来源 `artifacts/t5/perf/ppo_update_2000_flex_saved.json`，RTX 5080：

| 项 | 值 |
|---|---|
| rollout 规模 | 2,000 局 / 127,877 个 transition |
| optimizer step 数 | 1,120 |
| 总耗时 | 191.0 s |
| **每步** | **170.5 ms** |
| 吞吐 | 693.5 transitions/s |
| 每个 transition（前向+反向） | 0.75 ms |
| 分片字节 | 272.0 MB |
| Python 对象字节 | 1,111.1 MB（1.11 GB） |
| RSS 更新前 / 后 | 3,704.8 / 3,772.2 MB |
| CUDA 峰值 allocated / reserved | 2,073.3 / 5,878.0 MB |

每步从**一层**里取 `minibatch_chunks=16` 个 chunk，`sequence_length=16`。所以一步的名义
容量是 256 个 transition，但实际处理的是这 16 个 chunk 的真实长度之和。

**这里没有"凑整浪费"。** `forward_sequences` 用 `count = len(flat)`（真实 transition 数）
作为批维度，只有 token 维度会 pad 到批内最大值，GRU 走 `pack_padded_sequence`。所以短
chunk 就是真的便宜。把 1,120 步 × 256 当作"槽位"去算会得出 10.8% 的浪费，那是错的：
255,754 个 transition 访问（127,877 × 2 个 epoch）÷ 1,120 步 = **228 个/步**，比 256 少
的 11% 全部来自第 5 层的短 chunk——它们本来就不占那么多算力。**批处理维度已经是紧的，
不需要为它做序列打包。**

顺带纠正一个我自己写错的数字：早期笔记里的"每步 114 个 transition"是漏掉了 2 个 PPO
epoch 的算法，正确值是 228。

同一份数据里还有 dense 的对照：`ppo_update_2000_dense.json` 是 **404.4 s**
（316.2 transitions/s）。所以换到 FlexAttention 已经赚了 2.2 倍。

---

## 2. 每步的构成

在本机（单核 CPU，48 局 / 3,288 transition / 34 步，合成批次与基准同形状）跑真实的
`train_update`，在真实调用点包计时器：

| 阶段 | 每次调用 | 占墙钟 |
|---|---|---|
| `everything_else`（反向 + 损失 + 优化器） | 540.6 ms | **64.9%** |
| `forward_sequences` | 289.5 ms | **34.8%** |
| `replay_log_probs` | 2.7 ms | 0.3% |
| 每步合计 | 832.7 ms | |

再把 `everything_else` 拆开（包住 `optimizer.step` / `zero_grad` /
`torch.nn.utils.clip_grad_norm_`，残差即反向）：

| 成分 | 每步 | 占 `everything_else` |
|---|---|---|
| **反向 + 损失运算** | **619.6 ms** | **99.2%** |
| `optimizer.step`（AdamW） | 3.64 ms | 0.6% |
| `clip_grad_norm_` | 1.16 ms | 0.2% |
| `optimizer.zero_grad` | 0.12 ms | 0.0% |

所以一步 ≈ **36% 前向 + 63% 反向 + 0.6% 其余**。梯度的裁剪与优化器步进根本不值一提，
不必再动。

---

## 3. 被推翻的三条假设

### 3.1 "每个 minibatch 从 Python 对象重建张量很贵" —— 假

`train_update` 每个 minibatch、每个 epoch 都重建 `extras` / `old_log_prob` /
`advantage` / `returns`。实测（256 个 transition，取 5 次最小值）：

| 表达式 | ms |
|---|---|
| `cell_keys_stack`（256 个 (54,192) 栈起来） | 0.17 |
| `critic_extra_tensor`（256 × 16 个 float 的列表） | 0.09 |
| `advantage_stack`（256 个 0 维张量） | 0.03 |
| `belief_cat`（256 个切片） | 0.03 |
| `type_logits_stack` | 0.02 |
| `wait_logits_stack` | 0.02 |
| `old_log_prob_tensor` | 0.01 |
| `returns_tensor` | 0.01 |
| **合计** | **0.38 ms** |

占一步 620 ms 的 **0.06%**。把它们提出 epoch 循环是干净的改动，但**不会有可测的收益**，
不值得为它承担改错的风险。

### 3.2 "`forward_sequences` 里的逐条 numpy 赋值和逐个 `.to(device)` 是瓶颈" —— 假

把 `forward_sequences` 里的纯主机部分逐块复刻并计时（256 个 transition）：

| 块 | ms |
|---|---|
| numpy 组装（`ids`/`features`/`key_mask`/`packet_index`/`cell_index`） | 0.27 |
| 5 次 `torch.from_numpy(...).to(device)` | 0.06 |
| `_previous_action_batch` | 0.15 |
| `delta_index` 列表推导 | 0.02 |
| `_event_features_batch` | 0.14 |
| **`outputs` 逐条字典循环（14 个键 × 256 条）** | **0.89** |
| **合计** | **1.53 ms** |

对照 `forward_sequences` 整体 **353.5 ms**——**主机侧 Python 占 0.4%**。

那 0.89 ms 的字典循环是主机侧最大的一块，而它换来的是 `replay_log_probs` 里的
重新拼装（0.19 ms）。**"张量→256 个切片→256 个字典→再拼回张量"这条往返在 CPU 上
只值 1 ms 量级。** 把它改掉是设计上的整洁，不是性能上的必需。

### 3.3 "用 profiler 数算子能定位瓶颈" —— 假

`torch.profiler` 的 `count` 会把内部元数据读取（`as_strided` / `select` /
`resolve_conj`）也算成 ATen 调用。一次 `torch.bmm`（1536×84×28）报告 **12,293 次**
ATen 调用，其中真正的 `aten::bmm` 是 **1 次**；另外 4,608 次 `select` = 3 × 1536、
3,074 次 `resolve_conj` = 2 × 1536，都正比于 batch 元素数。

同一份数据里 `key_averages()` 的 `self_cpu_time_total` 汇总出 **5,637 秒**，而墙钟是
**5.7 秒**——差三个数量级。**本仓库此后不应再用 profiler 的算子计数或自计时下结论。**

---

## 4. 真正的大头：注意力里的关系偏置

> **⚠️ 本节已被 §10 更正，两条结论都反了。** 那张表里的 `dense_relation`（71.69 ms）
> 用的是**融合编译之前**的偏置组装；同一台机器、同样设置重跑，现在只有 **8.11 ms**
> （§10.2）。所以：
>
> * "FlexAttention 已经是正确的选择，不该回退" —— **反了**。当前代码下 dense 是
>   8.11 ms、flex 是 21.56 ms，**dense 快 2.66 倍**；512 局规模的完整 update 也证实
>   dense 快 1.54 倍（§10.4）。
> * "关系偏置在 CUDA 上以 score_mod 的形式逐元素求值……GPU 上则完全主导"（§7）——
>   **归因错了**。18.26 ms/层里有 **17.78 ms（97%）在反向**，前向只值 0.47 ms
>   （§10.3）。
>
> 表里另外 5 个变体都复现了（±15% 以内），只有 `dense_relation` 变了。所以下面这段
> 文字保留原样，是为了让下一个人看到"为什么当时的结论是对的、现在为什么不对"。

仓库里已经有一份针对**真实的 `encoder[0].attention` 模块、真实数据**的前向+反向基准
（`artifacts/t5/perf/attention_sweep.json`，RTX 5080，形状 256 帧 × 89 token × width 192）：

| 变体 | 前向+反向 | 相对"无偏置" |
|---|---|---|
| `dense_relation` | 71.69 ms | **21.7x** |
| `sdpa_relation` | 74.80 ms | 22.6x |
| **`flex_relation`（生产路径）** | **18.92 ms** | **5.7x** |
| `flex_local_relation`（窗口 16） | 9.29 ms | 2.8x |
| `sdpa_without_relation` | 3.31 ms | 1.0x |
| `linear_without_relation` | 3.43 ms | 1.0x |

读法：

* 去掉关系偏置，一层 attention 的前向+反向从 18.92 ms 掉到 3.31 ms。**关系偏置在
  FlexAttention 下值 15.61 ms/层**，四层 **62.4 ms/步**，占一步 170.5 ms 的
  **36.6%**。
* 这正是 dense 与 flex 差 2.2 倍的原因：dense 下关系偏置要 21.7 倍，flex 只要 5.7 倍。
  **FlexAttention 已经是正确的选择，不该回退。**
* 全部去掉关系偏置：一步 170.5 → 108 ms，update 191 → 121 s（**1.58x**）。
* 换成窗口 16 的局部关系偏置：一步 → 132 ms，update → 148 s（**1.29x**），
  但这是**语义改动**，不是优化。

放到整条训练循环上看。用**实测**的 rollout 时间（`worker_batch_sweep_2000.json`：
18 worker、2,000 局、161.6 s）而不是早期的估算：

| 场景 | rollout | update | 合计 | 相对现在 |
|---|---|---|---|---|
| 现状 | 161.6 s | 191.0 s | **352.6 s** | 1.00x |
| 只去掉 update 的关系偏置（flex 路径） | 161.6 s | 121.0 s | 282.6 s | **1.25x** |
| 再去掉 rollout 的关系偏置（eager 路径，bias ≈ 前向的 49%） | 73.0 s | 121.0 s | 194.0 s | **1.82x** |

两行都是**语义改动**（模型不再有这组偏置），不是优化：它会让
`MODEL_ARCHITECTURE_VERSION = 5` 与 T4 的 seed-0 基线脱钩，必须重跑阶段 0 的学习信号门。
列在这里是为了说明这组偏置总共值多少，不是为了建议这么做。

阶段 0 的 20,000 局 = 10 个 update：59 分钟 → 47 分钟（只去 update 侧）。

---

## 5. 每局的成本分布：rollout 是模型受限的

上面所有数字都是 update 的。update 只占整条循环的一半，另一半是 rollout。

**先说清楚一件事：记录里有两个不同口径的 rollout 画像，之前被混用了。**
`artifacts/t5/throughput.json` 的 6 个配置**全部是 1 线程/worker、cpu 设备**，区别只在
worker 数；`worker_single_core_current.json` 是其中 `workers=1` 那一档的单独记录。按
127,877 transition / 2,000 局 = **63.94 决策/局**折算成每决策成本：

| `throughput.json` 配置 | 秒/局 | model | environment | tokenization | critic | persistence |
|---|---|---|---|---|---|---|
| `workers=1`（= `worker_single_core_current.json`） | 0.5128 | **6.84** | 0.83 | 0.28 | 0.06 | 0.20 |
| `workers=12` | 0.7710 | 10.55 | 0.96 | 0.43 | — | — |
| `workers=14` | 0.8654 | 11.86 | 1.07 | 0.47 | — | — |
| `workers=16` | 0.9300 | 12.80 | 1.12 | 0.50 | — | — |
| **`workers=18`（生产选中）** | **0.9990** | **13.89** | 1.12 | 0.52 | — | — |
| `workers=20` | 1.1161 | 15.44 | 1.35 | 0.57 | — | — |

（单位 ms/决策。`persistence` 是 worker 落盘的耗时，不是 `collect_task_episode` 的四个
计时器之一，只有单核那一档单独记了。）

**`workers=1` 与 `workers=18` 差 2.03x，全部来自 CPU 争抢**，不是代码差异。所以：

* §5.1 里"记录里的 `model` 是 6.8 ms/决策"取的是 `workers=1` 那一行。
* `worker_batch_sweep_2000.json` 的 0.888 / 0.071 / 0.033 / 0.004（model 占 88.9%）
  是 `workers=18` 那一档。**0.888 / 63.94 = 13.89，不是 6.84** —— 这两个数字是同一张表
  的相邻两行，不能互相换算。

（单核记录里 model 占 0.4373/0.5128 = **85.3%**；18 worker 记录里占 0.8883/0.9990 =
**88.9%**。争抢对 model 的影响略小于对 environment 的影响，所以 worker 越多，model
占比反而越高。）

这条数据有两个直接后果：

* **rollout 的 85–89% 是模型前向，不是环境。** 所以优化 rollout 只有两条路：把模型
  前向变快，或者把每个决策的模型调用减少（例如一次前向处理多个候选动作）。
* **整个观测载荷（environment + tokenization）在单核口径下只值每局的 13.8%**
  （0.83 + 0.28 = 1.11 ms/决策）。`PROTOCOL_OBSERVABILITY_AUDIT.md` 里 B2–B6 那一整档
  （`legal_actions` 占 43.6% 的线上字节、`grid` 无人读取、`cells` 97.7% 是空的）
  **天花板就是它**，而 `environment` 里还有真正的游戏模拟。按整条循环（rollout 161.6 s
  + update 191 s = 352.6 s）算，这一档**最多值 4.8%**。它仍然值得做——但要知道它换不来
  数量级。

**所以两个真正的杠杆是同一个东西：模型前向。** 按 `workers=18` 口径，它在 rollout 的
161.6 s 里占 143.6 s（整循环的 40.7%）；在 update 里占前向 68.8 s。§6 记录的就是对它
已实施的两项优化。

### 5.1 真实环境实测：单核口径下代码已经快了 3.7x

§5 那张表是 **09-29 13:55** 的，而"关系偏置融合默认开启"是之后才做的。上一轮我据此
推测"画像可能过期"，但当时以为本机没有游戏资源、跑不了真实环境，就把这个疑点挂成了
"起跑后再测"。**那个判断是错的**——资源就在
`~/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN`。现在有了
`scripts/real_rollout_profile.py`，它驱动**生产的** `collect_task_episode`，四个阶段与
trainer 写进 `mean_episode_profile_seconds` 的完全同源，不是复制品。

**单核（1 线程、串行、无争抢；与 `workers=1` 那一行同构）**，20 局、`all` 课程、温机后
19 局（1,209–1,319 决策），独立跑三次：

| 阶段 | 记录 09-29 | 实测 09-30 | 变化 | 实测占比 |
|---|---|---|---|---|
| **model** | **6.84** | **1.770 / 1.889 / 1.902** | **3.6–3.9x** | **77.9–78.4%** |
| environment | 0.83 | 0.322 / 0.351 / 0.356 | 2.4x | 14.2–14.6% |
| tokenization | 0.28 | 0.183 / 0.184 | 1.5x | 7.5% |
| critic_inputs | 0.06 | 0.001 | — | 0.0% |

（ms/决策。`cap1` 课程同量级：model **1.729** ms/决策、54.9 决策/局，占比 80.1%——
cap1 每局决策更少、板面更小，所以每局更便宜，但**每决策**成本几乎一样。）

所以"记录过期"这个判断成立，而且比原先估的更厉害：原先按 2.10 ms 的合成 fixture 估的
是 3.2x，真实是 **3.6–3.9x**。差异主体就是 09-29 之后默认开启的关系偏置融合。

**融合在真实环境上值 1.27x**（同一批任务、同一单核配置，只切
`PVZ_RELATION_BIAS_FUSION`）：开 1.770 ms/决策，关 2.251 ms/决策。合成 fixture 上量到
的 1.33x 基本吻合，所以 §6.2 之外没有第二块隐藏的差异。

**冷启动只值一局，而且只在融合路径上。** 单核第一局：model 7.74–8.59 ms/决策、
environment 10.2–11.3 ms/决策；第二局起就降到 1.8 / 0.35。但**关掉融合时第一局的
model 只有 2.008 ms/决策**（没有 7.7 的尖峰）——所以那个尖峰是 `torch.compile` 的
warmup，是融合换来的代价，摊到一局上。这有一个直接后果：
`mean_episode_profile_seconds` 是**按局平均**的，所以只有局数很小（例如 T1 smoke 的
1–2 局）时它才会被冷启动主导；2,000 局的 rollout 里它占 0.05%。

### 5.2 并行口径不能跨机器比较

18 worker 是生产实际选择的配置（`selected_parallel_workers = 18`），所以它值得量。
本机（macOS，360 局，1 线程/worker，走生产的 `run_seed_jobs` 池）：

| 阶段 | 记录 09-29（18 worker） | 实测 09-30（18 worker） | 变化 |
|---|---|---|---|
| model | 13.89 | **9.03** | 1.54x |
| environment | 1.12 | 2.00 | 0.56x |
| tokenization | 0.52 | 0.27 | 1.9x |
| 每局秒数 | 0.999 | 0.791 | 1.26x |

**但这一行不能当结论用，因为它跨了机器，而且跨了操作系统**：记录的
`worker_single_core_current.json` 里 `resource_dir` 是
`/home/newbiexvwu/.cache/pvz-research-resources`，是 Linux/WSL；本轮全在 macOS 上。

争抢放大系数（同一台机器上 18 worker ÷ 单核）在两台机器上差了一倍多：

| 机器 | 单核 model | 18 worker model | 放大 |
|---|---|---|---|
| 记录（Linux） | 6.84 | 13.89 | **2.03x** |
| 本机（macOS） | 1.85 | 9.03 | **4.88x** |

放大系数的差异比代码改动本身还大。两个可能的原因，都指向"并行时瓶颈换了位置"：

* **每个 worker 其实是两个进程**（python worker + `pvz-portable` 子进程），所以 18
  worker = **36 个进程**争抢核。`environment` 阶段要跨管道往返，争抢下 IPC 延迟被放大
  ——这解释了为什么 environment 的放大（0.35 → 2.00，**5.7x**）比 model 的还大。
* 融合优化省掉的主要是**内存分配和跨步读**（§6.2 的反向 1.40x 就是这么来的）。争抢时
  内存带宽成为瓶颈，省掉的算术不再线性地变成墙钟。

**所以：代码优化的收益只在单核口径上可靠，生产（18 worker）上的收益必须在生产机器上
重测。** 本轮的 3.7x 不要直接外推到生产的 18 worker。

### 5.3 worker 启动开销：单次约 5.5 s

`run_seed_jobs` 每次调用都新建一个 pool，所以**每批 rollout 都要重新启动 18 个 worker，
每个都要加载 45 MB 的 `main.pak`**。用两个规模做线性拟合（本机 18 worker）：

| 局数 | 墙钟 |
|---|---|
| 180 | 10.24 s |
| 720 | 24.46 s |

斜率 = 26.3 ms/局，**截距 = 5.5 s**，就是 pool 启动。按生产默认
`--rollout-episodes 2000`（`HARD_STOP_EPISODES = 20_000`，每 5,000 局插一次评估，
所以大多数批次是满 2000 局）算：5.5 / (5.5 + 2000 × 0.0263) = **9.5%** 的 rollout
墙钟花在启动上。

在记录的机器上这个比例可能大得多——从 161.6 s 减去 111 s 的计算时间会得到约 50 s
的启动开销（31%），但那个减法假设了 `mean_episode_seconds` 可以线性外推，**不可靠**，
只能当作"需要在生产机器上确认"的线索。**如果成立，复用 pool 就是一个比 B2–B6 大得多
的杠杆。**


---

## 6. 已实施的两项优化

两项都是**位等价**的（`torch.equal` 逐位相同），都不改模型语义，因此
`MODEL_ARCHITECTURE_VERSION = 5` 不动、T4 的 seed-0 基线不动、不需要重跑阶段 0 的
学习信号门。

### 6.1 `replay_log_probs`：不再逐行写设备张量

原来每个 plant 行都要 `sel_pos[r] = ...`、`type_mask[row, 0] = ...`、
`packet_mask[r, j] = ...` 写一次设备张量，并且用 `int(sel_pos[r])` 回读——**读一个标量
就会同步设备**。现在全部先建成 Python 对象再一次拷进去。

| 设备 | 改前 | 改后 | 加速 |
|---|---|---|---|
| CPU | 2.91 ms | 1.55 ms | **1.9x** |
| MPS | 52.11 ms | 5.94 ms | **8.8x** |

（256 个 transition；三次采样，改前 2.85/2.91/3.10、改后 1.51/1.55/1.63。）
对一步的贡献只有 0.3%，所以这是"顺手赚到的"，不是大头。

### 6.2 关系偏置：四次 permute 合成一次

`relation_bias_from_indices` 原来把四个 `(B, L, L, H)` 的查表结果**各自** permute 成
`(H, B, L, L)` 再相加。`permute` 是 view、本身不搬数据——**代价在于随后那次加法必须按
跨步方式读这个 view**。所以三个加法各付一次跨步读，最后再 permute 回去。

改成全部在 `(B, L, L, H)` 里查表并相加（四次都连续），**只在最后 permute 一次**。
每个元素的加法次序完全没变，所以结果逐位相同。

| 形状 | 改前 | 改后 | 加速 |
|---|---|---|---|
| batch=1, L=108（一个 rollout 决策） | 0.316 ms/层 | 0.239 ms/层 | **1.32x** |
| batch=16, L=80 | 2.722 ms/层 | 2.092 ms/层 | **1.30x** |
| batch=256, L=80（一个 update minibatch） | 44.657 ms/层 | 32.745 ms/层 | **1.36x** |

`scripts/relation_bias_layout_bench.py` 里三个变体都过了 `torch.equal` 断言。

**端到端实测（单核 CPU，`git stash` 做 A/B）：**

| 路径 | 改前 | 改后 | 加速 |
|---|---|---|---|
| rollout `step_tokens`（batch=1, L=108） | 2.297 ms | 1.975 ms | **1.16x** |
| update `forward_sequences` | 290.7 ms | 248.0 ms | **1.17x** |
| update `everything_else`（反向+损失+优化器） | 545.2 ms | 390.1 ms | **1.40x** |
| **update 每步** | **838.7 ms** | **639.8 ms** | **1.31x** |

反向也赚 1.40x 是这次最意外的发现：旧写法在反向里要为三个被 permute 过的分支各生成一份
跨步梯度，每层约 236 MB 的中间梯度分配；新写法只要一份，约 118 MB。**所以关系偏置在
CPU 上是分配受限的，不只是算术受限。**

### 6.3 这两项在生产的哪个配置上生效

必须说清楚，因为生产的 rollout 和 update 跑在不同设备上
（`throughput.json`：`rollout_device=cpu`、`update_device=cuda`）：

| 路径 | 走哪个 kernel | 6.2 是否生效 |
|---|---|---|
| **rollout（CPU，batch=1）** | eager（flex 要求 batch ≥ 32） | **生效** |
| update（CUDA，batch=256） | **FlexAttention**，偏置在 `score_mod` 里算 | **不生效** |
| update（CPU / `--attention-backend dense`） | eager | 生效 |
| rollout（若跑在 CUDA 且 batch < 32） | eager | 生效 |

所以 **6.2 对生产循环的实际收益是 rollout 那一段**：模型前向 143.6 s × 14% =
**20.1 s**，整循环 352.6 → 332.5 s，**1.06x**。update 的 1.31x 只落在 CPU/dense 配置上
（也就是 `ppo_update_2000_dense.json` 那一档）。

**这也说明 §9 里的关系偏置优化（把 bias 预先算成 `(B,H,L,L)`）依然值得在 CUDA 上实测**：
生产 update 的 36.6% 仍然在那里，而且 6.2 碰不到它。

> **§10.4 已给出答案，而且推翻了上面这张表的最后一行。** 实测：`update（CUDA，batch=256）`
> 走 dense 是 **8.251 ms/层**，走 flex 是 **22.395 ms/层**，**dense 快 2.71 倍**；
> 512 局规模的完整 update 也证实 dense 快 1.54 倍。所以：
>
> * 把生产 update 改成 `--attention-backend dense`，**6.2 就开始对 update 生效了**。
> * 上面"6.2 对生产循环的实际收益只有 rollout 那 1.06x"这个结论随之作废：改成 dense 后，
>   6.2 同时吃到 rollout 和 update 两段。

### 6.4 顺手修掉的一个"测量会骗人"的问题

`scripts/attention_cost_profile.py` 里的 `build_relation` 原本是 `RelationAttention` 稠密
路径的**手抄副本**。副本会漂移：我改完生产代码后它仍在报旧的 0.301 ms/层，也就是**它一直
在测量已经不存在的代码**。现在它直接调用生产的
`relation_bias_indices` / `relation_bias_from_indices`，并把"pair 索引构造（每前向一次）"
和"bias 组装（每层一次）"分开报，免得两者再混在一起。

---

## 7. GPU 相对单核只有 5 倍，但这不等于"有 5 倍浪费"

同一份代码、同一批数据：

| 设备 | 每步 | `forward_sequences` | `everything_else` | `replay_log_probs` |
|---|---|---|---|---|
| 单核 CPU（本机） | 832.7 ms | 289.5 ms | 540.6 ms | 2.7 ms |
| MPS（本机） | 509.3 ms | 107.0 ms | 344.7 ms | **57.5 ms** |
| RTX 5080（记录） | 170.5 ms | — | — | — |

MPS 只比单核快 1.63 倍、RTX 5080 只快 4.9 倍。乍看像是"主机侧占了大头"，但 §3.2 已经
证明主机侧只占 0.4%。**所以这个比例是张量算术本身的特征，不是调度开销。**

~~原因是关系偏置在 CUDA 上以 `score_mod` 的形式**逐元素**求值：每个 (batch, head,
query, key) 元素要做 4 次查表（kind 对、行差、列差、同格）加 3 次加法，而 score 矩阵是
256 × 6 × 89 × 89 ≈ 12.2M 个元素 × 4 层 × 3（前向+反向）。CPU 上这几次查表几乎免费，
GPU 上则完全主导。~~ **这就是为什么 CPU 上的测量会系统性低估 GPU 上的关系偏置成本。**

（上面这段归因**是错的**，§10.3 直接量出来了：那 4 次查表加 3 次加法在前向里只值
**0.47 ms**，而整层偏置值 18.26 ms —— **97% 在反向**。真正的机制不是"逐元素求值贵"，
而是**查表反向要把 14.2M 个元素的梯度归约进极少数几个目标**：`same_cell_bias` 只有
**12 个输出**，它的反向单独就值 **460 ms**。所以结论的方向仍然成立——CPU 测量确实
系统性低估了 CUDA 上的偏置成本（CPU 上这个开关只值 1.30–1.41x，CUDA 上值 **177x**，
见 §10.2）——但**原因不是 CPU 那段文字说的那个**。）

`replay_log_probs` 在 MPS 上 57.5 ms/次、CPU 上 2.7 ms/次（21 倍），是另一类现象：
它按元素写张量（`sel_pos[r] = ...`、`type_mask[row, 0] = ...`、`packet_mask[r, j] = ...`），
并且每个 plant 行都重建一次 `dict(zip(...))`。在 CUDA 上这会变成约 12,000 次微算子，
按每次 3–8 µs 的发射成本算是 **19–62 ms/步（占 11–36%）**。这一块**可以位等价地修掉**，
而且不涉及任何模型语义。

---

## 8. 复现命令

```bash
PY=/Users/newbiexvwu/.local/share/mise/installs/python/3.14/bin/python3
RES=~/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN

# 每步的阶段分解（真实 train_update，计时器包在真实调用点上）
$PY scripts/ppo_update_profile.py --episodes 48 --skip-profiler --micro

# 换设备做对照
$PY scripts/ppo_update_profile.py --episodes 48 --skip-profiler --device mps

# 完整报告（含被证伪的 profiler 一节，保留是为了让下一个人不必重跑）
$PY scripts/ppo_update_profile.py --episodes 48 --profile-steps 5

# 真实游戏上的每局成本分布（§5.1）：单核串行，对应 throughput.json 的 workers=1
$PY scripts/real_rollout_profile.py --resource-dir $RES --episodes 20
$PY scripts/real_rollout_profile.py --resource-dir $RES --episodes 20 --curriculum cap1

# 生产的并行口径（§5.2，18 worker，走真实的 run_seed_jobs 池）
$PY scripts/real_rollout_profile.py --resource-dir $RES --episodes 360 --workers 18

# 融合开关在真实环境上值多少（§5.1）
PVZ_RELATION_BIAS_FUSION=0 $PY scripts/real_rollout_profile.py --resource-dir $RES --episodes 20

# rollout 侧（batch=1）的逐阶段分解，以及关系偏置的布局基准
$PY scripts/attention_cost_profile.py
$PY scripts/relation_bias_layout_bench.py
```

§10 的 CUDA 部分要在台式机上跑（本机 `cuda=False`）。连接见 `scripts/win_ssh.py`
（`--host/--port` 或 `PVZ_DESKTOP_HOST/PORT`），命令见 §10 开头的那一整块。

`scripts/ppo_update_profile.py` 用合成批次（token 形状与真实一致：每 transition 约 80
个 token、54 个 cell、6 个 lane、6 个 packet）复现基准的规模，因此 optimizer step 数与
张量形状都对得上；它跑的是**真实的 `train_update`**，不是复制品。

做 A/B 的正确姿势（本轮两项优化都是这么验的）：

```bash
git stash push -- python/pvz_agent_model.py     # 只回退模型文件，脚本保持新版
# ... 跑 3 次，记下数字 ...
git stash pop
```

---

## 9. 下一步的取舍

已经做完、不需要再碰的：

1. ~~`replay_log_probs` 的逐元素写设备张量~~ —— **已修**（§6.1，CPU 1.9x / MPS 8.8x）。
2. ~~关系偏置的 permute 布局~~ —— **已修**（§6.2，前向 1.16–1.36x、CPU update 每步 1.31x）。
3. ~~"rollout 画像可能过期"这个疑点~~ —— **已解决**（§5.1，真实环境实测，单核 3.7x）。
   顺带纠正了记录里两个口径被混用的问题（§5）。
4. §3 里被推翻的三条假设不必再试。

剩下的，按证据强度排序：

1. ~~**关系偏置的 CUDA 侧**~~ —— **已实测并已实施，见 §10。**
   结论与本节原判断相反：当前代码下 dense 比 flex 快 1.53–1.59x（512 局口径，
   §10.6 交替验证）。实施方式是改 `train_update` 里 `auto` 的解析，让它选 dense；
   `flex` 保留为显式选项。**不改模型语义、不改数值。**
2. **精度** —— **已实测完成，见 §10.5。3.85x 是测量假象，真实收益 1.03–1.06x。**
   等价性论证通过（ratio 在 ±2.3% 内，clip 是 ±20%），但没有东西可买。
3. **worker pool 复用（§5.3）** —— 本机实测每次 rollout 有 5.5 s 的 pool 启动开销，
   按生产默认的 2,000 局/批算是 rollout 墙钟的 9.5%；**在记录的机器上这个比例可能到
   31%**。不需要 CUDA，本机就能做完，而且**不涉及模型语义**。风险在于每批的权重不同，
   复用的 worker 必须重新 `load_state_dict`，还要处理 worker 崩溃与长跑内存增长——
   所以先在生产机器上把 5.5 s 这个截距确认一遍，再决定动不动手。如果确认，这是比
   B2–B6 大得多的杠杆。**§10 之后它成了第 1 项之后最大的杠杆。**
4. **观测载荷（B2–B6）** —— §5 已经量出天花板是整循环的 4.8%。值得做，但换不来数量级。

**必须在生产机器（Linux + CUDA）上重做的**：

* ~~第 2、3 项要 CUDA~~ —— **已在台式机（RTX 5080, sm_120, torch 2.14.0+cu132）上做完，
  §10**。
* **并行口径的画像**（§5.2）。本机 18 worker 的争抢放大是 4.88x，记录的机器是 2.03x
  ——差得比代码改动本身还多。所以本轮的 3.7x **不要直接外推到生产的 18 worker**；
  单核口径（`--workers 1`）才是可跨机器比较的那个。
* 第 3 项的 5.5 s 启动截距。

---

## 10. 生产机器上的实测（RTX 5080, sm_120, torch 2.14.0+cu132, WSL2）

§9 列的"必须在生产机器上重做"的两项已经做完。**两项的记录结论都错了**，而且错的方向
相反：关系偏置那项低估了自己（因为被融合编译掩盖），精度那项高估了自己（因为预热缺失）。

复现命令（在台式机 `~/PvZ-Portable` 上，venv `/home/newbiexvwu/.venvs/ml`）：

```bash
SHARDS=artifacts/t5/runs/run_2/.seed_jobs/update_0001/fcd5098ca3d43a7574e5f4dedb77b158d4ccc5caf666d2a74ae67d817df002d2
PY=/home/newbiexvwu/.venvs/ml/bin/python

# §10.1–10.3：注意力层的前向/反向分解、融合 A/B、逐项反向
$PY scripts/relation_bias_cuda_bench.py --data-dir $SHARDS \
    --episodes 16 --frames-per-episode 16 --repeats 20 --term-repeats 3 \
    --output artifacts/t5/perf/relation_bias_cuda.json

# §10.2：用记录的原始设置（repeats=2）重跑旧扫描，看哪些变了
$PY scripts/attention_benchmark.py --data-dir $SHARDS \
    --episodes 16 --frames-per-episode 16 --repeats 2 --device cuda \
    --output artifacts/t5/perf/attention_sweep_reproduced.json

# §10.4：512 局规模的 flex vs dense
$PY scripts/ppo_update_benchmark.py --data-dir $SHARDS --episodes 512 \
    --attention-backend flex --configurations 16:16:fp32,16:16:bf16 \
    --output artifacts/t5/perf/ppo_update_512_flex.json

# §10.5：fp32/bf16 的等价性论证（每种精度各自预热）
$PY scripts/precision_equivalence.py --data-dir $SHARDS --episodes 16 \
    --warmup-episodes 4 --held-out-offset 500 --held-out-episodes 8 \
    --attention-backend flex --output artifacts/t5/perf/precision_equivalence_flex.json
```

形状：256 帧（一个 update minibatch）× 96 token × width 192 × 6 head。`repeats=20`，
两次预热，中位数。

### 10.1 旧扫描的 6 个变体，5 个复现，1 个变了 8.8 倍

同一台机器、同样的 `--episodes 16 --frames-per-episode 16 --repeats 2`（即记录当时的
设置）：

| 变体 | 记录 | 复现 | 差 |
|---|---|---|---|
| `dense_relation` | 71.693 ms | **8.111 ms** | **8.8x 快** |
| `sdpa_relation` | 74.804 ms | 92.149 ms | 1.23x 慢 |
| `sdpa_without_relation` | 3.306 ms | 3.469 ms | 1.05x |
| `flex_relation` | 18.922 ms | 21.555 ms | 1.14x 慢 |
| `flex_local_relation` | 9.294 ms | 10.039 ms | 1.08x 慢 |
| `linear_without_relation` | 3.429 ms | 3.307 ms | 0.96x |

除 `dense_relation` 外的 1.0–1.2x 量级差异来自 token 数（96 vs 89，(96/89)² = 1.16；
`sdpa_relation` 偏 1.23x，因为它走的是 `attention_benchmark` 里那份手抄的 eagerly
偏置组装，见 §6.4）。**只有
`dense_relation` 动了 8.8 倍，因为 dense 路径的偏置组装现在是 `torch.compile` 的**
（§6.2 之后）。这不是"记录过期"，是**记录里的 dense 那一行测的是已经不存在的代码**。

### 10.2 融合开关在 CUDA 上值 177x，不是 1.3x

| 组装方式 | 前向 | 前向+反向 |
|---|---|---|
| `assembly_shipped_eager`（`relation_bias_from_indices` 直接调） | 1.761 ms | **906.353 ms** |
| `assembly_shipped_fused`（`fused_relation_bias()`） | 0.668 ms | **5.109 ms** |
| `assembly_pre_6_2_eager`（§6.2 之前的布局） | 2.474 ms | 820.891 ms |

**177x。** §6.2 在 CPU 上量到的是 1.30–1.41x，§2 的注释因此写着"CPU 测量会系统性低估
GPU 上的关系偏置成本"——低估了 **125 倍**。

整层效果：`dense_relation`（融合开）8.251 ms vs `dense_relation_eager`（融合关）
907.891 ms → **110x**。

### 10.3 97% 的成本在反向，而且能精确归因到 12 个输出

| 变体 | 前向 | 反向 | 前向+反向 |
|---|---|---|---|
| `flex_mask_only`（无偏置，只做 mask） | 0.945 | 3.195 | 4.140 |
| **`flex_relation`（生产）** | **1.418** | **20.976** | **22.395** |
| 差 = 偏置 | **0.473** | **17.782** | **18.255** |

**偏置的前向只值 0.47 ms，反向值 17.78 ms —— 97% 在反向。**

把组装的四个查表各自单独反传：

| 项 | 输出元素数 | 反向 |
|---|---|---|
| `kind_pair_bias` | 600 | 163.05 ms |
| `row_bias` | 72 | 115.32 ms |
| `col_bias` | 108 | 176.01 ms |
| **`same_cell_bias`** | **12** | **459.65 ms** |
| 合计 | | **914.0 ms** |

四项之和 914.0 ms ≈ 实测组装反向 906.4 ms，**没有余项**。成本与输出元素数**反相关**：
`same_cell_bias` 只有 12 个输出却占了 **50%**。这就是"查表反向 = 14.2M 个元素归约进极
少数目标"的原子竞争，不是算术。

**所以 §9 里那个"降低 score_mod 查表次数"的方向 (b) 是无效的**：
`flex_hoisted_buckets`（把桶索引搬到外面，score_mod 只留 4 次查表）22.651 ms vs
`flex_relation` 22.395 ms —— **一样，甚至略慢**。前向那 0.47 ms 里没有东西可省。

### 10.4 方向 (a) 是对的：预计算偏置，而且必须编译

| 变体 | 前向 | 前向+反向 | 相对 `flex_relation` |
|---|---|---|---|
| `flex_relation`（生产，score_mod 里算偏置） | 1.418 | 22.395 ms | 1.00x |
| `flex_hoisted_buckets` | 1.554 | 22.651 ms | 0.99x |
| `flex_precomputed_bias`（预计算，**eager** 组装） | 2.698 | 915.244 ms | 0.02x |
| **`flex_precomputed_bias_fused`（预计算，**编译**组装）** | **1.934** | **8.746 ms** | **2.56x** |
| `dense_relation`（eager 注意力 + 编译组装） | 2.074 | 8.251 ms | 2.71x |

§9 说方向 (a)"看起来是错的"，依据是 `sdpa_relation` 22.6x 比 `flex_relation` 5.7x 更慢。
**那个依据测的是 eager 组装，不是"预计算"这个想法。** 把组装编译掉之后：

* 预计算 + flex：22.395 → **8.746 ms**（**2.56x**）
* 预计算 + eager（= `dense` 路径）：22.395 → **8.251 ms**（**2.71x**）

两条路都落在 8–9 ms，而"在 score_mod 里现算"是 22.4 ms。**结论：不要把偏置放在
`score_mod` 里。**

`flex_precomputed_bias_fused` 的 `max_abs_output_error` 是 1.31e-6，与 `flex_relation`
的 1.19e-6 同一量级 —— 数值上等价。

**512 局规模的完整 update 证实了这一点：**

| 后端 | fp32 每步 | 相对 flex |
|---|---|---|
| `flex`（生产 `auto` 选中） | 125.8 ms | 1.00x |
| **`dense`** | **81.7 ms** | **1.54x 快** |

（512 局 / 28,822 transition / 129 步。16 局小规模同向：dense 79.4 vs flex 118.4 ms。）

**记录里 `ppo_update_2000_dense.json` 说 dense 是 361.1 ms/步、flex 是 164.6 ms/步
（flex 快 2.19x）—— 那个 dense 数字与当前代码不符（当前 81.7 ms，差 4.4 倍），
与 §10.1 里 `dense_relation` 的 8.8 倍偏差是同一个原因。flex 那侧则吻合
（164.6 vs 125.8，1.31x，与 token 数差异同量级）。**

**行动项：把生产 update 的 `--attention-backend` 从 `auto` 改成 `dense`。**
不动模型语义，不动数值（dense 就是 `attention_sweep.json` 里的参考实现，误差 0.0）。
顺带把 rollout（batch=1，本来就走 eager）和 update 统一到同一条路径上。

**已实施（§10.6）。**

### 10.5 精度：3.85x 是"冷 fp32 比热 bf16"，真实收益 1.03–1.06x

先看记录的来源。`artifacts/t5/perf/ppo_update_lr_1e4.json` 的顶层字段是：

```
['device','device_name','torch_version','episodes','ppo_epochs','learning_rate','data_dir','configurations']
```

**没有 `warmup`，也没有 `attention_backend`。** 而之后的 `ppo_update_2000_*.json` 两个
都有。也就是说 14.371 s vs 3.73 s 是**旧版脚本**产出的：它没有预热，于是 fp32 那一档
吸收了 `torch.compile` 的全部成本，bf16 那一档是热进程。3.85x 是这么来的。

`ppo_update_benchmark.py` 的预热本身也有同一个缺陷：它只预热 `configurations[0]`。
本轮已修（每个配置各自预热），修之前用它会得到 **bf16 慢 21.9x** 的结论 —— 同一个
bug 的另一个方向。

修好之后，512 局 / 129 步：

| 后端 | fp32 | bf16 | bf16 收益 |
|---|---|---|---|
| `flex` | 125.8 ms/步 | 118.6 ms/步 | **1.061x** |
| `dense` | 81.7 ms/步 | 79.6 ms/步 | **1.026x** |

**1.03–1.06x，不是 3.85x。** 原因：这个 update 不是 fp32 张量算术受限的。一层
attention 的偏置组装在 dense 路径下就值 5.11 ms（占该层 8.25 ms 的 62%），而它是
**整数索引 gather + scatter-add**，autocast 碰不到；剩下的步进成本是 GRU、损失、优化器
和主机侧 Python。所以"把 matmul 换成 bf16"能碰到的部分本来就不大。

**等价性论证（`precision_equivalence.py`，506 条留出 transition，两边都用 fp32 评估
更新后的策略）：**

| 指标 | flex | dense |
|---|---|---|
| `max abs Δlogp` | 0.02266 | 0.02274 |
| `mean abs Δlogp` | 0.01057 | 0.01052 |
| PPO ratio 区间 | [0.9776, 1.0186] | [0.9775, 1.0182] |
| **超出 ±0.2 clip 的决策数** | **0 / 506** | **0 / 506** |
| 前向噪声底（同权重，仅 autocast） | 0.00297 | 0.00357 |

**结论：bf16 是安全的（ratio 偏差 2.3%，clip 是 20%），但买不到东西。**
两个后端的漂移几乎一样（0.0227 vs 0.0227），说明漂移来自 bf16 算术本身、与注意力实现
无关；噪声底 0.0030 说明 update 把前向噪声放大了约 7.6 倍，仍在 clip 的十分之一以内。

**所以 `--precision bf16` 不值得开。** 要提速，先改 `--attention-backend dense`（1.54x，
零风险），再谈别的。

### 10.6 已实施：`auto` 现在解析成 dense，并在生产机器上验证

改动是 `train_pvz_ppo.train_update` 里的一行解析（`attention_backend != "dense"` →
`attention_backend == "flex"`），而不是改 CLI 默认值。这样所有调用者（包括
`train_update` 自身的 `"auto"` 默认）都跟着变，`--attention-backend flex` 保留为
可选项，将来想重测不用再改代码。

验证一（语义）：在 CUDA 上跑一次真实的 `train_update`，检查每层拿到的标志：

```
 auto -> use_flex_attention = {False}
dense -> use_flex_attention = {False}
 flex -> use_flex_attention = {True}
```

验证二（速度）：512 局 / 129 步，单档 fp32，**flex 与 auto 交替各跑两次**
（避免把运行间噪声当成结论）：

| 后端 | 第 1 次 | 第 2 次 |
|---|---|---|
| `flex` | 126.8 ms/步 | 131.1 ms/步 |
| **`auto`（改动后）** | **82.8 ms/步** | **82.5 ms/步** |

**1.53–1.59x**，与显式 `dense` 那次（81.7 ms/步）吻合。

代价与等价性：

* 峰值显存 **2174 → 2470 MB**（+296 MB，dense 要存 `(B,H,L,L)` 的偏置张量）。
  RTX 5080 有 16 GB，不是约束。
* `policy_loss` 在 8 位有效数字上一致（`0.1370359`–`0.1370360`），`peak_cuda_allocated_mb`
  与显式 `dense` 完全相同（2470.0）—— 同一条路径，不是巧合。

---

## 11. 已否定的方向：LLM 稀疏/压缩注意力移植（原 `ATTENTION_TRANSFER_ANALYSIS.md`）

> **合并说明（2026-09-30）**：原文是 2026-09-30 在 Apple M5 Pro（CPU 单线程，torch 2.13.0）
> 上做的一次调研，回答"LLM 的稀疏/压缩注意力能不能搬过来"。结论是否定的，而且**否定的
> 理由与硬件无关**（上下文规模差三个数量级），所以并入本文件作为"不要重复调研"的清单。
> 原文里的 CPU 加速比（融合 1.86×/1.30×）**不要外推到 CUDA**——§10.2 实测 CUDA 上是 177x。

**一句话**：那 7 个机制全部在解决"上下文长度 N 增长到 10⁵–10⁶ 时 KV Cache 与注意力算力的
爆炸"，而 PvZ 的上下文是 **L ≈ 74–116 个 token**，**没有 KV Cache**（每次决策把整条序列从头
重算），注意力只占单层算力的 **4.2%**。

### 11.1 明确不做的（避免以后重复讨论）

| 想法 | 为什么不做 |
|---|---|
| 加稀疏 indexer / Top-k 检索 | L=102，indexer 的算子开销 > 它省的 4.2% 算力，**负收益** |
| 线性注意力替换 softmax attention | 摧毁关系偏置；L=102 时二次成本不是瓶颈 |
| KV 压缩 / 跨层 KV 共享 | **没有 KV Cache 可压** |
| FP4/FP8/FP16 量化 | 台式机已实测 FP16/BF16/TF32 全部无收益或更慢（§10.5 更正：bf16 其实快 1.03–1.06x，但不足以采用） |
| 滑窗局部注意力 | L=102 < 任何合理窗口，等价于全注意力 |
| 改 GRU 为 GDN-2 | 性能无收益（GRU 占 1.1%）；能力评估要等 T5 基线 |

### 11.2 逐项判定（原 §4，结论摘录）

| 机制 | 判定 |
|---|---|
| CSA2（DeepSeek V4.1 Flash） | ❌ 不可移植 |
| QSA（Qwen3.8-Flash-Next） | ❌ 不可移植 |
| MSA（MiniMax M3） | ❌ 不可移植（但工程理念值得学） |
| SGA（A.X K2） | ⚠️ 只有 head gate 有一点价值，属能力而非性能 |
| **IndexShare / IndexCache（GLM-5.2 / Hy4）** | ✅ **有真正的对应物**，见下 |
| KDA + Gated MLA（Kimi K3） | ❌ 不可移植，方向值得记在 T6 |
| Gated DeltaNet-2 | ⚠️ 唯一可能对 PvZ 有真实价值的一条，但也是能力而非性能 |

### 11.3 唯一有对应物的那一条，已经做完了

GLM-5.2 的 IndexShare 洞察是"**索引比它索引的东西更贵 → 共享/缓存**"。PvZ 有同一个病：
一次关系偏置构造派发 **117 个算子**，算术量只有微秒级，纯调度开销——占 attention 时间的
**67.1%**、每决策总耗时的约 **35%**。

修法不是任何稀疏化算法，是**融合**：

* `torch.compile` 融合偏置组装（§6.2），CPU 口径 1.86×，`torch.equal` 逐位相同；
* 索引张量跨层提升（4 层算出完全相同的索引，`kinds/rows/cols` 与层参数无关）；
* CUDA 上这同一个开关值 **177x**（§10.2），而且它顺带改变了 dense 与 FlexAttention 的
  胜负（§10.6）。

复现：`scripts/attention_cost_profile.py`（单决策拆解）、`scripts/relation_bias_cuda_bench.py`
（CUDA 分解）。
