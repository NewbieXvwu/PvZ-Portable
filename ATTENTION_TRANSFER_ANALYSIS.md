# LLM 稀疏/压缩注意力能否移植到 PvZ 模型？—— 实测分析

分析日期：2026-09-30。测量机器：Apple M5 Pro（CPU，单线程，torch 2.13.0）。
模型：`GameplayModelV1`（3,681,809 参数，4 层，width 192，6 头，GRU 2×256）。

---

## 0. 结论先行

**这 7 个机制针对的是 PvZ 模型根本不存在的问题。** 它们全部在解决"上下文长度 N 增长到
10⁵–10⁶ 时 KV Cache 与注意力算力的爆炸"，而 PvZ 的上下文是 **L ≈ 74–116 个 token**，
没有 KV Cache（每次决策把整条序列从头重算），注意力只占单层算力的 **4.2%**。

但这次核查**发现了真正的瓶颈**，而且它和 GLM-5.2 的 IndexShare 是同一个洞察：

> **关系偏置（relation bias）的构造占 attention 时间的 67.1%、每决策总耗时的约 35%，
> 而它一次派发 117 个算子，算术量只有微秒级 —— 纯调度开销。**

已实测的修法：`torch.compile` 融合后 **1.86×，逐位相同**（`torch.equal == True`）。

---

## 1. 来源核实（先做这一步，因为你贴的内容带 ChatGPT 引用标记）

你给的材料带有 `:chatgpt-content-reference{index="N"}` 标记，说明是 ChatGPT 输出。
这类输出里的模型名、参数规模、具体数字都可能被编造，所以逐条核实：

| 机制 | 模型 | 核实结果 | 来源 |
|---|---|---|---|
| CSA2 压缩稀疏注意力 | DeepSeek V4.1 Flash | ✅ 真实。2026-09-10 发布，技术报告 09-17 | 中信证券研报、知乎图解、官方 HF 仓库镜像 |
| QSA | Qwen3.8-Flash-Next | ✅ 真实。2026-08-26 发布，官方定位 Qwen4 预览 | QwenLM GitHub、官方技术博客 |
| MSA | MiniMax M3 | ✅ 真实。论文 arXiv 2606.13392，kernel 开源 | arXiv 摘要（已读） |
| SGA | A.X K2 | ✅ 真实。688B MoE，arXiv 2608.30181 | arXiv / HF Papers |
| IndexShare | GLM-5.2 | ✅ 真实。2026-06-16 官方博客 | 智谱 HF 博客、架构拆解 |
| KDA + Gated MLA | Kimi K3 | ✅ 真实。2.78T 参数、104B 激活、1M 上下文 | 47 页技术报告（多篇解读） |
| Gated DeltaNet-2 | （研究） | ✅ 真实。arXiv 2605.22791，NVIDIA Research | arXiv / NVIDIA 官网 |
| Hy4 Preview | 腾讯 | ⚠️ 未独立核实到一手来源 | — |

**你材料里的数字与一手来源基本吻合**，但有两处要修正：

- **MSA 不是"每个 GQA group 有自己的 S_i"这么简单**：论文原文是 Index Branch 对
  **KV block** 打分、**为每个 GQA group 独立选 Top-k 子集**，Main Branch 只在选中的 block 上
  做 exact block-sparse attention。而且它明确是**与 GQA 同精度**（"performs on par with GQA"），
  不是"效果基本保持"——这个差别在评估移植风险时重要。
- **CSA2 的"跨层复用"在 PvZ 上没有对应物**（见 §4.1）：它复用的是一个**随上下文增长的
  KV Cache**，而 PvZ 每次决策把整条序列从头重算，压根没有 KV Cache。

---

## 2. PvZ 模型的实测画像

### 2.1 上下文规模

| 场景 | token 数 L |
|---|---|
| 开局（1 植物 / 1 僵尸） | 74 |
| **中局（18 植物 / 12 僵尸）** | **102** |
| 密集（24 植物 / 20 僵尸） | 116 |

L² = 10,404。对比：CSA2 / QSA / MSA 的目标是 L ~ 10⁵–10⁶，L² 达 10¹⁰–10¹²。
**差 6 个数量级。**

### 2.2 每决策耗时构成（实测）

| 环节 | 耗时 | 占比 |
|---|---|---|
| `step_tokens`（模型前向） | 2.749 ms | **91.6%** |
| `observation_tokens`（token 化） | 0.165 ms | 5.5% |
| `select_action`（采样） | 0.087 ms | 2.9% |
| `legal_summary` | 0.001 ms | 0.0% |
| **合计** | **3.002 ms** | |

### 2.3 `step_tokens` 内部拆解（实测）

| 环节 | 耗时 | 占 step_tokens |
|---|---|---|
| encoder（4 个 relation 层） | 2.589 ms | **93.3%** |
| ├─ attention（4 层合计） | 1.668 ms | **60.1%** |
| │  └─ **relation-bias 构造** | **1.060 ms** | **38.2%** |
| └─ FFN（4 层合计） | 0.696 ms | 25.1% |
| GRU | 0.031 ms | 1.1% |
| embeddings + 动作头 | 0.155 ms | 5.6% |

### 2.4 单层 attention 内部拆解（实测，0.396 ms）

| 环节 | 耗时 | 占比 |
|---|---|---|
| **relation-bias 构造** | **0.265 ms** | **67.1%** |
| softmax | 0.060 ms | 15.2% |
| QKV 投影 | 0.027 ms | 6.7% |
| score matmul QKᵀ | 0.020 ms | 5.1% |
| AV matmul | 0.012 ms | 3.0% |
| 输出投影 | 0.012 ms | 2.9% |

### 2.5 算力 vs 时间的错配（关键）

| | attention | FFN |
|---|---|---|
| 单层 MACs | 3,995,136 | 90,243,072 |
| **算力占比** | **4.2%** | 95.8% |
| **实测时间占比** | **70.5%** | 29.5% |

**attention 只占 4.2% 的算术量，却吃掉 70.5% 的时间。** 这不是注意力数学的问题，
是那些 unfused 的 gather/elementwise 操作的调度开销问题。

---

## 3. 根因定位：117 个算子

对 relation-bias 构造做算子计数（`torch.profiler`）：

```
一次 relation-bias 构造派发 117 个算子
     27 x aten::as_strided
     14 x aten::unsqueeze
      6 x aten::view
      6 x aten::select
      5 x aten::reshape
      5 x aten::__and__
      5 x aten::bitwise_and
      5 x aten::to
      5 x aten::add
      5 x aten::empty
      4 x aten::ge
      4 x aten::where
      4 x aten::permute
      3 x aten::embedding
```

而这套运算作用的张量规模：`(1, 102, 102, 38)` 的查表输出 + 几个 `(1, 102, 102)` 的整数索引。
**算术量在微秒级，117 次算子派发把它放大到 265 微秒。**

代码位置：`python/pvz_agent_model.py` 第 377–391 行（dense 路径）
与第 339–363 行（FlexAttention 的 `score_mod` 路径）。

### 实测修法：融合

```
eager      0.263 ms
compiled   0.142 ms   speedup 1.86x
max |eager - compiled| = 0.000e+00   (torch.equal: True)
```

`torch.compile(build_relation, dynamic=True)` → **1.86×，逐位相同**。
（`dynamic=True` 能覆盖 L 在 74–116 之间变化，不会因形状变化反复重编译。）

**投射到端到端**：4 层 × (0.263 − 0.142) = 0.484 ms，占每决策 3.002 ms 的 **16.1%**。
且因为是逐位相同，不触碰任何等价性保证。

---

## 4. 逐项移植性判定

### 4.1 CSA2（DeepSeek V4.1 Flash）—— ❌ 不可移植

| CSA2 组件 | PvZ 上的对应物 | 判定 |
|---|---|---|
| 压缩 KV Cache（890 B/token） | **没有 KV Cache**。每次决策从零重算整条序列 | 不适用 |
| 跨层 KV 共享（`kv_source_layer_ids=[2,8,14,20]`） | 没有可共享的缓存 KV | 不适用 |
| 两级稀疏 indexer → Top-512 | 需要新增 indexer（**更多算子**）去省 4.2% 的 MACs | **负收益** |
| 最近 128 token 滑窗 | L=102 < 128，滑窗覆盖全部 | 不适用 |
| FP4 量化 | 台式机已实测 FP16/BF16/TF32 全部无收益或更慢 | 不适用 |

**唯一有对应物的洞察**：CSA2 的核心是"哪些层需要拥有自己的记忆"。PvZ 的对应版本是
**"哪些层需要自己的关系偏置表"** —— 见 §4.5，但那是架构决策，不是免费优化。

### 4.2 QSA（Qwen3.8-Flash-Next）—— ❌ 不可移植

QSA = Gated DeltaNet（固定大小状态）+ 稀疏精确检索。它解决的问题是"纯线性注意力丧失
随机精确召回能力"。

PvZ 上：
- **不需要线性注意力**：L=102，L² = 10,404，softmax attention 的二次成本根本不是瓶颈
  （只占 4.2% MACs）。
- **线性状态会摧毁模型的核心归纳偏置**：现有 attention 的价值几乎全在
  `kind_pair_bias` / `row_bias` / `col_bias` / `same_cell_bias` 这 792 个参数构成的
  **成对关系偏置**（"第 2 行的僵尸应当注意第 2 行的植物"）。固定大小的递归状态
  **无法表达成对关系**。
- 模型**已经有**递归记忆：`nn.GRU(width+128 → 256, 2 层)`，占 1.1% 耗时。

### 4.3 MSA（MiniMax M3）—— ❌ 不可移植（但工程理念值得学）

MSA 的 per-GQA-group block 稀疏 + 开源 kernel 确实是目前工程最完整的。但：

- 它省的是 **1M 上下文下每 token 28.4× 的注意力算力**。PvZ 的注意力算力占比是 4.2%，
  **理论上限就是 4.2%**，而代价是新增一个 index branch（更多算子）。
- MSA 与 GQA **同精度**，但那是 109B 模型在长上下文任务上的结论，不能外推到 3.68M 模型。
- **值得学的不是算法，是"算法与 kernel 共同设计"这条方法论** —— 这正是台式机
  在 CUDA 上用 FlexAttention 做的事（2000 局更新 404 s → 188 s，2.15×）。

### 4.4 SGA（A.X K2）—— ⚠️ 只有 head gate 有一点价值，但属于能力而非性能

`O_h = σ(g_h(x)) · Attention_h(Q,K,V)` —— 每个 head 按输入动态决定输出强度，用于抑制
attention sink。

- **性能上**：增加算子，不省任何东西。
- **能力上**：理论上可能改善优化，成本极低（每 head 一个标量门）。
- **但**：本项目铁律规定"行为来源只有 RL"、"进度以能力不以 loss"。加一个门是
  归纳偏置（梯度能修正），合法；但它**不解决当前的问题**——当前问题是 RL 链路
  还没证明能学到东西（阶段 0 零胜率）。**在阶段 0 通过前不要碰架构。**

### 4.5 IndexShare / IndexCache（GLM-5.2 / Hy4）—— ✅ **这一条有真正的对应物**

GLM-5.2 的洞察：DSA/QSA/MSA 的隐藏成本是"便宜的 indexer × 几十层 × 1M token"，
indexer 本身最后也会变成大头 → 每 4 层共用一个 indexer，1M 上下文每 token FLOPs 再降 2.9×。

**PvZ 上的直接对应物**：relation-bias 构造里的**索引张量是层无关的**。

```python
# 这四行只依赖 kinds/rows/cols，与层的参数无关，4 层算出完全相同的索引
row_known = (rows[:, :, None] >= 0) & (rows[:, None, :] >= 0)
row_delta = (rows[:, :, None] - rows[:, None, :]).clamp(-5, 5) + 5
row_bucket = torch.where(row_known, row_delta, 11)
same_cell = (...).long()
```

只有**查表**（`kind_pair_bias` / `row_bias` / `col_bias` / `same_cell_bias`）是逐层不同的
（每层 792 个参数）。

**所以**：把索引计算提到层循环外，一次算好、4 层复用 —— **不改模型、不改语义**，
且直接砍掉 117 个算子中约 30 个索引算子 × 3 层 ≈ **90 次派发**。

这与 GLM-5.2 的 IndexShare 是**同一个洞察**（索引比它索引的东西更贵，所以要共享/缓存），
只是 GLM 共享的是跨层 indexer，这里共享的是跨层 index 张量。

### 4.6 KDA + Gated MLA（Kimi K3）—— ❌ 不可移植，但方向值得记在 T6

K3 证明"大多数层不需要传统 softmax attention"（93 层中 69 层 KDA）。

PvZ 上：
- 只有 4 层，全 attention 的算力占比 4.2%。**把 3 层换成线性层最多省 3% 算力。**
- 而且会牺牲关系偏置。
- **KDA 的价值在"长序列 + 大模型"，不在"4 层 + 102 token"。**

### 4.7 Gated DeltaNet-2 —— ⚠️ 唯一可能对 PvZ 有真实价值的一条，但不是性能

`S_t = Decay(S_{t-1}) − Erase_b(S_{t-1}) + Write_w(k_t,v_t)`，decay/erase/write 都是 channel-wise。

PvZ 的 `nn.GRU` 承担同样的角色（跨决策的信念状态），而且**GRU 的更新门是
element-wise 但耦合的**，没有显式的"擦除旧记忆"通路。

- **性能上**：GRU 只占 1.1% 耗时，换成 GDN-2 没有任何速度收益。
- **能力上**：这可能是**唯一值得在 T6 认真评估的架构改动** —— 因为 PvZ 的
  信念状态需要"僵尸死了要把它从记忆里擦掉"这种语义，而 GDN-2 的 channel-wise
  erase 正是为这个设计的。
- **但**：这是能力改动，必须在阶段 0 通过、有 T5 基线之后才能评估。

---

## 5. 真正该做的事（按性价比排序）

### 建议 1：融合 relation-bias 构造 —— 已实测，1.86×，逐位相同 ⭐ 最高优先

- 对 CPU rollout 路径的 `build_relation` 用 `torch.compile(dynamic=True)`。
- 收益：每决策 **−16.1%**；18 worker 各编译一次（秒级），摊到数千局可忽略。
- 风险：低。已实测 `torch.equal == True`。
- **注意**：这是 rollout 路径（18 个 CPU worker），与台式机已做的 CUDA FlexAttention
  是**互补**的 —— 那个优化的是 PPO 更新，这个优化的是 rollout。

### 建议 2：索引张量跨层提升（IndexShare 的 PvZ 对应物）⭐ 高优先

- 把 `row_bucket` / `col_bucket` / `same_cell` / `kind_pair` 索引的计算
  从 `RelationLayer` 循环里提到 `step_tokens` / `forward_sequences` 里算一次。
- 不改模型参数、不改语义、不改 checkpoint 格式。
- 预估收益：砍掉约 90 次算子派发（需实测确认）。
- **这条是本次分析中唯一"从 LLM 前沿机制里直接拿到可执行优化"的收获。**

### 建议 3：跨决策的增量编码（KV Cache 的真正对应物）⚠️ 需先测量

PvZ 每次决策把 102 个 token 从头重算 4 层。但两次决策之间：
- 54 个 cell token 的 `kinds/rows/cols` **完全不变**
- 只有实体 token 增删、以及 feature 变化

**候选**：缓存 `kind_embedding`/`category_embedding`/`variant_embedding`/
`row_embedding`/`col_embedding` 的贡献（它们只依赖离散元数据），只重算
`feature_projection`。

**但必须先测**：两次决策之间到底有多少 token 的内容真正变化。如果 90% 变了，
这个方向就没价值。**在没有测量之前不要动手。**

### 不建议做的事（明确记录，避免以后重复讨论）

| 想法 | 为什么不做 |
|---|---|
| 加稀疏 indexer / Top-k 检索 | L=102，indexer 的算子开销 > 它省的 4.2% 算力。**负收益** |
| 线性注意力替换 softmax attention | 摧毁关系偏置；L=102 时二次成本不是瓶颈 |
| KV 压缩 / 跨层 KV 共享 | **没有 KV Cache 可压** |
| FP4/FP8/FP16 量化 | 台式机已实测 FP16/BF16/TF32 全部无收益或更慢 |
| 滑窗局部注意力 | L=102 < 任何合理窗口，等价于全注意力 |
| 改 GRU 为 GDN-2 | 性能无收益（GRU 占 1.1%）；能力评估需等 T5 基线 |

---

## 6. 如果要做，验证顺序

1. **先做建议 1**（`torch.compile` 融合）：改动最小、已实测、逐位相同。
   验收：`torch.equal` 对比融合前后；端到端 rollout 吞吐对比。
2. **再做建议 2**（索引跨层提升）：验收同样是逐位等价 + 算子计数下降。
3. **建议 3 先测量再决定**：写一个探针统计"相邻决策之间变化 token 的比例"。
4. **所有改动必须在阶段 0 通过之后再进主分支** —— 当前 T5 阶段 0 还没跑通，
   架构/性能改动会污染"RL 链路能否学到东西"这个待验证的命题。

---

## 7. 一句话总结

> 你列的 7 个机制全部真实存在，但它们解决的是"上下文长到 10⁵–10⁶ 时的 KV Cache 与算力爆炸"。
> PvZ 的上下文是 **102 个 token、没有 KV Cache、注意力只占 4.2% 算力**。
> 唯一有直接对应物的是 **GLM-5.2 的 IndexShare**（索引比它索引的东西更贵 → 共享/缓存），
> 而这次核查已经证明 PvZ 存在同一个病：**一次关系偏置构造派发 117 个算子**。
> 修法不是任何稀疏化算法，是**融合**——实测 1.86×，逐位相同。

---

## 附录：复现

全部结论由仓库内脚本复现（不依赖 `/tmp`）：

```bash
cd /Users/newbiexvwu/PvZAgent
python scripts/attention_cost_profile.py
```

输出包含：L 随棋盘密度的变化、每决策耗时拆解、`step_tokens` 内部拆解、
单层 attention 六环节拆解、算力/时间错配、关系偏置的算子计数、
以及 eager vs `torch.compile` 的加速比与逐位等价性检查。

参考输出（Apple M5 Pro / CPU 单线程 / torch 2.13.0 / L=102）：

```
--- per-decision breakdown (L=102) ---
  step_tokens (model forward)     2.768 ms   91.5%
  observation_tokens              0.168 ms    5.5%
  select_action                   0.087 ms    2.9%
  legal_summary                   0.001 ms    0.0%
  TOTAL                           3.023 ms

--- inside one attention layer (L=102, heads=6, head_width=32) ---
  relation-bias build      0.266 ms   67.1%
  softmax                  0.061 ms   15.4%
  QKV projection           0.026 ms    6.6%
  score matmul QK^T        0.020 ms    5.0%
  AV matmul                0.012 ms    3.0%
  output projection        0.011 ms    2.9%

--- attention is time-bound, not compute-bound ---
  attention MACs/layer    3,995,136  (4.2% of layer MACs)
  FFN MACs/layer         90,243,072

--- operator count of one relation-bias build ---
  dispatched operators: 117

--- eager vs torch.compile on the relation-bias build ---
  eager       0.266 ms
  compiled    0.145 ms   speedup 1.83x
  torch.equal(eager, compiled) = True   max|diff| = 0.000e+00
  projected saving: 0.482 ms per decision  (16.0% of per-decision cost)
```

### 未完成的测量

**相邻决策之间有多少 token 的内容真正变化**（决定建议 3 是否可行）需要跑真实 episode。
本机 PvZ 二进制仍是**协议 3**，而 Python 侧已是协议 4（台式机提交 `5fb13b6` 同步升级了两侧），
因此 `PvZEnv` 启动即报
`did not enter protocol-v4 environment mode: {'ready': True, 'protocol_version': 3}`。
跑之前需先 `cmake --build build -j6` 重编译本机二进制。

