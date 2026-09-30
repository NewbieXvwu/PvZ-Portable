# 训练与模拟器性能优化任务清单

目标：只保留经实测能显著提速或降低内存、且能力回归没有明显恶化的改动。

本文件是**性能与内存的唯一总账**：正文是任务与结论，**附录 A** 是内存预算实测（原 `MEMORY_BUDGET.md`），**附录 B** 是 2026-09-29 那一轮优化的详细数据（原 `T5_PERFORMANCE_REPORT.md`）。附录 B 里有三处已被 [PPO_UPDATE_ANATOMY.md](PPO_UPDATE_ANATOMY.md) §10 推翻，读之前先看附录 B 开头的说明。

## 阶段 0：基线与现有改动

- [x] 使用 `/home/newbiexvwu/.venvs/ml` 确认 PyTorch、CUDA、GPU、CPU 和资源路径：PyTorch 2.14.0+cu132、RTX 5080、i7-12700F；WSL CUDA 可用。资源目录为 `~/.cache/pvz-research-resources`，指向 `~/main.pak`。
- [x] 检查原工作区改动和已有提交；保留经验证有价值的紧凑 token 表示、批量序列更新、worker 扫描、NPZ shard 和单波进度条修复；没有发现应因负收益删除的改动。
- [x] 固定任务 manifests、资源文件、随机初始化及基准输入；新增的 benchmark 报告写入 Git 忽略的 `artifacts/t5/perf/`。
- [x] 建立 rollout、shard 恢复、PPO 更新、attention 和 simulator 子阶段的计时与 RSS/CUDA 峰值记录。
- [x] 更新后的 CPU 单 worker 跑到 6,273.8 局/小时，正式训练要求的 5,000 单核门槛通过。

## 阶段 1：rollout、持久化与 simulator 协议

- [x] 扫描 CPU/CUDA worker、每 worker 线程和推理设备。以正式 2,000 局 `run_seed_jobs` 批次选 CPU 18×1：161.559 秒、44,565.9 局/小时；比 16×1 快 1.2%，比 20×1 快 8.1%。配置已写入 `artifacts/t5/throughput.json` 并供正式训练读取。
- [x] 按 2,000 局批次记录进程树 RSS：18×1 峰值 19,168 MB；12×1 峰值 15,126 MB 但吞吐 40,606.6 局/小时。选最快 worker 会多占约 4 GB RSS。
- [x] 比较 gzip JSON、压缩 NPZ 和未压缩 NPZ；压缩 NPZ round trip 快约 4.9×、恢复对象内存约减半，故用于 shard。记录磁盘空间增加约 5.3×的代价。
- [x] 将训练用 token ID/索引/feature 数组窄化到 int8/窄整数/FP16，训练时恢复 FP32；最大 feature 误差 3.26e-4，真实轨迹 60/60 动作一致。
- [x] 将 critic 的完整 `PRIV` 查询改为 `CRITIC_INPUTS`；50 个模拟器状态得到完全相同 16 维向量，响应时间快 8.26×、JSON 从约 32.6 KB 降至 69 B。协议升为 4。
- [x] profile 完整 simulator 响应和 observation tokenization：分别约 0.071 秒和 0.033 秒/局。实体 observation 也是策略输入；未采用有损删字段方案，critic 以外协议简化的潜在上限约 10.5%，不足以承担能力风险。
- [x] 比较短批与正式批、考虑跨更新持久 worker 池。18 worker 需约 19 GB 进程树 RSS；持久池会把该开销叠加到 PPO 更新期间，未证明启动节省值得内存代价，未实现。
- [x] 检查固定 simulator 与单波任务路径；单波进度条除零修复保留。C++ 构建和 100 tick 单波 smoke 通过。

## 阶段 2：PPO 更新与轨迹表示

- [x] 测量 batch/chunk 数与 truncated-BPTT 长度；选择 sequence 16 / minibatch_chunks 16。默认 rollout batch 设为 2,000、learning rate 设为 1e-4，以匹配真实更新批次并减少每局训练开销。
- [x] 修复 CUDA 序列回放中的窄索引 dtype 与 CPU mask/device 错误；新增 CUDA 回放覆盖。
- [x] 比较 FP32、TF32、BF16、FP16。TF32 无明显收益，BF16 更慢，FP16 提速不稳定且幅度约 2%；正式训练保持 FP32。
  - **2026-09-30 更正**：记录的"BF16 更慢"（14.371 s vs 3.73 s，3.85x）是测量假象——旧脚本无预热，`torch.compile` 按 dtype 各编译一次，变成冷 fp32 比热 bf16。每档各自预热后真实收益只有 **1.03–1.06x**（512 局：dense fp32 81.7 / bf16 79.6 ms每步）。等价性通过（Δlogp 最大 0.0227，0/506 超出 ±0.2 clip）但买不到东西，**结论"正式训练保持 FP32"不变，理由变了**。见 [PPO_UPDATE_ANATOMY.md](PPO_UPDATE_ANATOMY.md) §10.5。
- [x] 检查轨迹状态向量、合法动作和奖励构造；紧凑格式保留原字段，不删实体 token、不做 lane/global 汇总。
- [x] 用真实 2,000 局、2 epochs 更新对比 dense/Flex：Flex 为 184.391 秒，dense 为 404.420 秒；峰值 CUDA 分配 2.07/2.49 GB，reserved 5.88/6.44 GB。更新 loss 接近。
  - **2026-09-30 更正，结论反了**：那个 dense 数字出自关系偏置组装被 `torch.compile` 融合**之前**的代码。重测（512 局口径）dense 81.7 ms/步、flex 125.8 ms/步，**dense 快 1.54x**；交替 A/B 复测 1.53–1.59x。`train_update` 的 `--attention-backend auto` 已改为解析成 dense，`flex` 保留为显式选项。见 [PPO_UPDATE_ANATOMY.md](PPO_UPDATE_ANATOMY.md) §10.1 / §10.6。

## 阶段 3：attention 消融

- [x] 建立 dense relation attention 的 forward+backward 基线：256 帧×89 token×192 width，71.7 ms、547.8 MB。
  - **2026-09-30 更正**：71.7 ms 里绝大部分是**未编译**的关系偏置组装（eager 906 ms/层反向 vs 融合后 5.1 ms）。融合后的 dense 是 **8.25 ms**。旧扫描 6 个变体重跑，5 个在 ±15% 内复现，只有 `dense_relation` 变 8.8 倍（71.7 → 8.11 ms）。见 §10.1 / §10.2。
- [x] 测量 SDPA、保留 relation bias 的 FlexAttention、滑窗和线性 attention。Exact Flex 为 18.9 ms、330 MB、输出 RMS 误差 1.75e-7；纳入 CUDA 大 batch，保留 dense 小 batch/CPU fallback。
- [x] 量化近似 attention：滑窗 16、线性和无关系 SDPA 的输出 RMS 误差分别 0.434、0.296、0.199；均未采用。
- [x] 记录编译一次性成本和真实 2,000 局 PPO 更新；精确 Flex 更新约快 2.19×，峰值分配内存降低约 17%。
  - **2026-09-30 更正，2.19x 变为 0.65x（反向）**：FlexAttention 的成本 97% 在反向——`score_mod` 里 14.2M 个 score 元素的梯度要归约进 12/72/108/600 个表项，`same_cell_bias` 只有 12 个输出却占反向的 50%（原子竞争）。dense 路径让 `torch.compile` 一次核函数做完。逐项归因见 §10.3。

## 阶段 4：最终验收与提交

- [x] 每个保留项均有针对性基准或等价性检查；汇总改前/改后吞吐、内存、精度误差和能力回归证据于性能报告。
- [x] 完成 2,000 局 PPO 更新和正式 2,000 局 worker 批次测量。
- [x] 配对 held-out capability smoke：相同冻结 seeds 上初始与一次 2,000 局 Flex PPO 更新均为 0/20 胜出，终局 wave 分布相同；记录为无可见回退但能力证据有限，不替代正式 held-out 曲线门禁。
- [x] 更新最终证据并按 worker/storage、simulator protocol、PPO/CUDA 和 attention 分组提交；最终提交后复查工作区。

---

## 附录 A：内存预算与轨迹表示（原 `MEMORY_BUDGET.md`，2026-09-29 实测）

> **合并说明（2026-09-30）**：原文是"只做诊断与规格，不在本轮实现"的诊断稿，其中
> §7「实施结果」已经落地、§8 的台式机实测已回填。整份并入此处以保留实测数字。
> 与 [PPO_UPDATE_ANATOMY.md](PPO_UPDATE_ANATOMY.md) §10 冲突的地方以 §10 为准。


2026-09-29，针对"WSL 吃满 16 GB"的实测诊断。

**本文件只做诊断与规格，不在本轮实现。**
台式机上的 T5 正在跑，此时改 `python/` 会造成双方代码分歧；等那轮结束后按本文档合并。
（唯一例外是 §4 的配置层改动，它不碰仓库代码，可以现在做。）

---

### 1. 实测数据

全部在 Mac 上用 `python/train_pvz_ppo.py` 的真实调用路径跑出来（探针 `/tmp/pvz_mem_probe.py`），
任务是 `train.json` 第一个任务 `train_day_1`（wave_cap=1，倍率 1.0）：

| 量 | 实测值 | 备注 |
|---|---|---|
| 一个 `observation`（Python 对象） | **42.4 KiB** | `pickle` 后只有 4.0 KiB → **10.6× 膨胀** |
| 一个 `privileged_state`（Python 对象） | **76.9 KiB** | 比 observation 还大，多了 `hidden` 字段 |
| 一个 `transition` 合计 | **184.6 KiB** | 上面两项 + action/events/potential 等 |
| 一局决策数 | 79（tick 7889 终局） | wave_cap=1；cap=3/5 会更长 |
| 模拟器子进程 RSS | **222 MB** | 走完一局稳定在 222.7 MB，不增长 |
| `import torch` + 模型 | 224 MB | 模型本身仅 3.68 M 参数 = 14 MB fp32，AdamW 状态 28 MB |

**观测内部最大的两块**：`cells` 21.3 KiB（54 个 cell，每个是 dict）、`legal_actions` 9.3 KiB。

### 2. 内存账（8 workers 情形）

| 项 | 每单位 | 数量 | 小计 |
|---|---|---|---|
| 模拟器子进程 | 222 MB（实测） | 8 | 1.78 GB |
| Python worker + torch | 200–300 MB（Mac 实测 224；**Linux/cu132 待实测**） | 8 | 1.6–2.4 GB |
| 主进程 torch + 模型 + 优化器 | ~250 MB | 1 | 0.25 GB |
| **轨迹缓冲** | 184.6 KiB × 79 决策/局 × batch | — | **见下，是决定项** |

轨迹缓冲随 batch（= `--rollout-episodes`）线性增长：

| batch | 轨迹缓冲 |
|---|---|
| 100（当前默认值） | 1.39 GiB |
| 500 | 6.95 GiB |
| **2000（TODO 要求的每次更新样本量）** | **27.8 GiB** |

固定开销约 3.6–4.4 GB，**轨迹缓冲是唯一会爆掉的量**。
TODO §4 T5 要求"每次 PPO 更新的样本量不得低于 2,000 局"，按当前表示那需要 27.8 GiB ——
**这不是配置问题，是当前实现根本撑不住 2,000 局的批**。
吃满 16 GB 是必然结果，不是意外。

### 3. 三项可消除的浪费（按性价比排序）

### 3.1 `privileged_state` 存了 77 KiB，只用了其中 200 字节 —— 先做这条

`GameplayModelV1.privileged_value`（`pvz_agent_model.py:411`）实际消费的只有：

```python
hidden = privileged_state.get("hidden", {})
values[0] = _ratio(hidden.get("wave_timer", 0), 6000)
waves = hidden.get("zombies_in_wave", [])
current = min(max(0, output["wave_index"]), max(0, len(waves) - 1))
for zombie_type in waves[current][:15] if waves else []:
    if 0 <= zombie_type < 15:
        values[1 + zombie_type] += 0.1
```

即 **一个 int + 一个 ≤15 元素的小列表 → 16 维 `values` 数组**。
为这 16 个 float，每个 transition 存了 76.9 KiB 的完整状态字典，**浪费 99.7%**。

**改法**（数值完全等价，见 §5 验收）：

1. 在 `pvz_agent_model.py` 把 `privileged_value` 拆成两半：

   ```python
   def privileged_extra(self, privileged_state, wave_index) -> list[float]:   # 纯函数，无梯度
       ...返回 16 个 float...
   def privileged_value_from_extra(self, output, extra) -> Tensor:
       return self.privileged_critic(torch.cat(
           (output["belief"], self.privileged_features(torch.tensor([extra], ...))), dim=-1))
   ```

2. `collect_task_episode` 里：算出 `extra`，**只存 `transition["critic_extra"] = extra`**，
   **删掉 `transition["privileged_state"]`**。
3. `train_update`（`train_pvz_ppo.py:260`）改用
   `model.privileged_value_from_extra(output, transition["critic_extra"])`。
4. `episode_hash` 的字段清单里把 `privileged_state` 换成 `critic_extra`。

**收益**：每 transition 185 KiB → 108 KiB（−42%）；
同时省掉每个 PPO epoch 重新解析 77 KiB dict 的 CPU（ppo_epochs=2 就是两遍）。
**风险**：零。`values` 数组逐位相同，前向结果不变。

### 3.2 observation 以 Python 对象驻留，且每个 epoch 重新 token 化

`model.step` 第一步就是 `observation_tokens(observation)`（从 42 KiB 的 Python dict 建张量），
而 `train_update` 在每个 epoch 都跑一遍整条 chunk —— **ppo_epochs=2 就把整批轨迹 token 化两遍**。

**改法**：rollout 时 token 化一次，把**输入张量**存进 transition（不是 output，output 依赖 hidden 不能缓存，
但 token 化结果是无状态的，可以缓存）。训练时直接取用。

张量大小估计：54 cell + packet/global token，每 token 32 维特征 + 若干 int id，
fp32 下约 8–10 KiB —— 比 42 KiB 的 Python 对象小 4–5×，且能被后续 batch 化直接吃。

**收益**：−80% 内存（叠加 3.1 后每 transition 约 20 KiB）；
顺带砍掉训练侧最大的 Python 开销。batch=2000 从 27.8 GiB → 约 2.9 GiB。

### 3.3 训练循环是 batch=1 串行 —— 这条不只是内存问题

`model.step` 里 `x = x.unsqueeze(0)`、`hidden` 形状写死 `(gru_layers, 1, gru_width)`，
`train_update` 内层是：

```python
for transition in transitions[start:end]:        # 逐条，batch=1
    output = model.step(transition["observation"], hidden, ...)
```

`resolve_device` 的文档字符串自己承认了这件事：

> Every model invocation in this project is a batch-of-1 forward pass … MPS only wins once
> tensors are batched, **and no such path exists in the training loop**.

这条注释诚实，但它描述的是一个**设计缺陷，不是物理定律**。

**改法**：3.2 做完之后，chunk 内 64 步的 token 张量可以 stack 成一个 batch，**一次前向**；
GRU 的序列依赖仍在，但 token 化（纯 Python 侧开销）从"64 次 batch=1"变成"1 次 batch=64"。
跨 episode 还可以在 batch 维并行多个 chunk（hidden 各自独立）。

**收益**：这才是"更新占 95% 耗时"的真正解药。

### 4. 配置层（不碰仓库代码，可以现在做）

改 Windows 用户目录下的 `.wslconfig`（`%UserProfile%\.wslconfig`）：

```ini
[wsl2]
memory=24GB
swap=8GB
localhostForwarding=true
```

然后 `wsl --shutdown` 再重开。**前提是主机物理内存够**（≥32 GB 才建议给 24 GB）。

另外一个 WSL2 的已知行为：Linux 侧的 page cache 计入 Windows 任务管理器看到的 `vmmem`，
且**不会主动归还**，所以"看起来吃满"里有相当部分是缓存而非真实占用。
判断真实占用要看 WSL 内部的 `free -m`（减掉 `buff/cache` 那列），而不是 Windows 任务管理器。
**先确认是不是真吃满，再决定要不要改代码。**

### 5. 验收（硬要求）

1. **等价性（3.1 必须过）**：同一个 checkpoint、同一批 seed，改动前后
   每个 transition 的 `value` 必须**逐位相同**（`torch.equal`，不是近似）。
   3.1 只是把 16 个 float 提前算好，数学上恒等；不恒等说明实现错了。
   注意：`episode_hash` 会因此变化（字段换了），这是预期的一次性基线变更，
   必须在门禁证据的 `notes` 里写明，并保留改动前的 hash 作为对照。
2. **内存**：batch=2000 下实测主进程峰值 RSS + 全部子进程 RSS 之和，写进
   `artifacts/t5/memory.json`，目标 < 8 GB。
3. **吞吐**：端到端（rollout + 更新）局/小时，与改动前对照。不许只看 rollout。
4. **变异测试**（铁律）：注入"漏删一个 privileged_state"的缺陷，内存门禁必须报错。

### 6. 对既有判断的修正（重要）

TODO §5 里我写过：

> 本机实测是更新占 95%，批量矩阵运算正是 RTX 5080 的强项——故 T5 改在台式机执行。

**这个判断是错的，过度简化了。** 更新慢的真实原因是 **batch=1 串行 + 每 epoch 重新做
Python 侧 token 化**（§3.2/§3.3），不是"矩阵太大"。在 batch=1、每步都要从 42 KiB dict
建张量的负载下，GPU 大部分时间在等 kernel launch 和 CPU→GPU 搬运，**收益远低于预期**。

连带影响：i7-12700F 的单核性能约为此机的一半。若 GPU 红利不成立，
**台式机未必比本机快**——这个结论不能靠推理下，必须实测（§5.3 的耗时拆分要按机器分别测）。

**正确顺序是"先 batch 化，再选机器"，而不是"先迁机器，再优化"。**
当前台式机上那轮让它跑完不打断（它产出的是学习信号验证门的结果，与吞吐无关），
但**下一轮开始前必须先把 §3.1–§3.3 做完再重新测机器选择**。

---

### 7. 实施结果（2026-09-29 当天完成，提交 fe4d9e6 / 801dbc2 及后续）

### 已落地

| 改动 | 效果 | 语义 |
|---|---|---|
| §3.1 privileged_state → 16 维 critic_extra | transition 184.6 → 108 KiB | **逐位等价**（torch.equal 验证） |
| §3.2 observation → 打包 token（2 个 numpy 数组）+ legal bitmask | 184.6 → **13.1 KiB（−93%）** | 逐位等价 |
| §3.3 forward_chunk：chunk 内一次 encoder + GRU 序列化 + 批量重放 | 更新 3.5 → 0.375 s/局 | GEMM 容差级（≤4.8e-7） |
| 分层批 forward_sequences（跨 episode，hidden 沿层传递） | **CPU 上负优化**（852s vs 750s） | 语义变更（per-minibatch step） |

**实测教训（2026-09-29，Mac CPU）：**
1. 2000 局 × 51 步 × 2 epochs 批更新：逐条 773s → 单 chunk 批化 750s → 分层批(128) 852s。
   **batch 化在 CPU 上的收益到单 chunk 就到顶**——RelationAttention 的 gather/permute/softmax
   部分（(N, heads, L, L) 级别）随 batch 线性扩展、无摊薄，GEMM 摊开的 kernel 开销不是瓶颈。
2. 分层批 minibatch=128 时注意力激活约 767 MB/层、4 层反向保存 → 峰值 6.6 GB。
   **16 GB 机器上 minibatch_chunks × T × L² × heads × 4B ≈ scores 内存，建议 ≤ 32。**
3. **大 batch 的红利在 GPU**：softmax/matmul/gather 都是大核操作，batch 大才能摊 kernel launch。
   台式机开工时必须按 §5.3 的顺序实测：`--minibatch-chunks` 1 / 16 / 64 / 128 四档，
   `--device cuda`，报告每档的更新耗时与峰值显存，据此定正式配置。

### 台式机开工参数建议

```bash
# 安全基线（语义与旧版一致，lr 含义不变）
python python/train_pvz_ppo_task_family.py --device cuda --minibatch-chunks 1 ...
# GPU 批量红利（先跑 15 分钟吞吐实验再决定）
python python/train_pvz_ppo_task_family.py --device cuda --minibatch-chunks 64 --learning-rate 1e-4 ...
```

minibatch_chunks > 1 时优化器步数减少约 minibatch_chunks 倍，**等效学习率必须上调**
（建议从 1e-4 起测，阶段 0 学习信号验证门是最终裁判——10,000 局内 cap1/1.0 任务
pass rate 升不到 50% 就是配置错了，按 §3 停下诊断，不许加样本硬凑）。

---

### 8. 台式机实测回填（2026-09-29，已合入 HEAD `02125b9`）

**§7 教训 3 的预测被证实，但机制需要修正**：CUDA 大 batch 的红利确实存在，
**但来源是 attention kernel 的实现，不是"矩阵更大"**。

| 配置（RTX 5080，2,000 局，127,877 transitions，2 epochs） | 耗时 | 峰值分配 |
|---|---|---|
| dense relation attention，chunks 16 | 404.4 s | 2.49 GB |
| **exact FlexAttention，chunks 16** | **184.4 s（2.15×）** | **2.07 GB** |

同层 forward+backward（256 帧×89 token×192 width）：dense 71.7 ms / 547.8 MB →
exact Flex 18.9 ms / 330.0 MB，输出 RMS 误差 1.75e-7。**Flex 保留了 kind/row/column/same-cell
学习关系偏置**（不是靠简化注意力换速度）；loss 与梯度范数与 dense 一致
（policy 0.12451 vs 0.12455，grad norm 1.488 vs 1.489）。

**正式 T5 参数已定**：`--device cuda`、sequence 16、`--minibatch-chunks 16`、lr 1e-4、attention `auto`。

### WSL OOM 的第二条根因（§2 内存账的补充）

§2 只算了**轨迹缓冲**（27.8 GiB），台式机实测发现**另一条独立的大头**：

| 项 | 峰值 |
|---|---|
| 18 worker 进程树 RSS 合计 | **19,168 MB** |
| 训练进程 CPU RSS | 约 3.8 GB |
| CUDA 峰值预留 | 5.88 GB |

三者**同时存在**。进程池每次 2,000 局更新重建一次；报告明确**不提交持久池**
（跨更新保留省下的启动开销不值这份常驻内存）。→ 16 GB WSL 跑正式 T5 前必须先按 §4
把 `.wslconfig` 上限调够，否则这条根因会单独把机器打穿。

### 复现脚本

`scripts/t5_worker_batch_benchmark.py`、`scripts/ppo_update_benchmark.py`、
`scripts/attention_benchmark.py`、`scripts/trajectory_storage_benchmark.py`；
证据在 `artifacts/t5/perf/`（被 `.gitignore` 忽略，需随工作区一起搬）。
汇总报告：本文件（原 `T5_PERFORMANCE_REPORT.md` 已并入附录 B）。

---

## 附录 B：T5 训练与模拟器性能优化记录（原 `T5_PERFORMANCE_REPORT.md`，2026-09-29，RTX 5080）

> **合并说明（2026-09-30）**：这是 2026-09-29 那一轮优化的详细数据。其中三处已被
> [PPO_UPDATE_ANATOMY.md](PPO_UPDATE_ANATOMY.md) §10 推翻，读的时候一并看：
> ① "dense 404.420 秒 / Flex 184.391 秒"——dense 那个数字出自关系偏置融合之前，
> 现在 dense 更快 1.54x；② "BF16 慢于 FP32"——无预热造成的假象，真实快 1.03–1.06x；
> ③ "dense relation 71.7 ms"——融合后是 8.25 ms。


测量日期：2026-09-29。基准在 `/home/newbiexvwu/.venvs/ml` 执行：PyTorch 2.14.0+cu132、RTX 5080 16GB、i7-12700F（20 个逻辑 CPU）。游戏资源使用 `/home/newbiexvwu/.cache/pvz-research-resources`，其中 `main.pak` 指向用户主目录下的资源。性能数据与任务 manifest 固定在 `artifacts/t5/perf/`；该目录被 Git 忽略，避免提交大型 rollout 数据。

### 已采用的优化

| 项目 | 实测结果 | 采用方式 |
|---|---|---|
| PPO rollout worker | 正式 `run_seed_jobs` 的同一批 2,000 局：18×CPU/1 Torch 线程 161.559 秒、44,565.9 局/小时；20×1 为 174.667 秒、41,221.3 局/小时；16×1 为 163.545 秒、44,024.6 局/小时。18 worker 的进程树 RSS 合计峰值 19,168 MB。 | 正式 T5 默认 CPU/18 worker/每 worker 1 线程；配置写入 `artifacts/t5/throughput.json`。单核门禁重跑为 6,273.8 局/小时，高于 5,000 门槛。 |
| 紧凑 rollout shard | 同一 60 步轨迹：gzip JSON round trip 中位数 122.4 ms、19,343 B、恢复对象 956,321 B；压缩 NPZ 为 25.1 ms、101,763 B、恢复对象 495,809 B；未压缩 NPZ 为 19.7 ms、444,979 B。 | 压缩 NPZ 保留 NumPy dtype。它比 gzip JSON 多用约 5.3 倍磁盘，但序列化约快 4.9 倍，恢复对象内存约减半。完整 2,000 局 shard 为 272 MB，解码 episode 对象合计约 1.11 GB。 |
| token 与 observation 存储 | 将 token ID 存为 int8、索引存为窄整数、features 存为 FP16 后，60 步恢复对象从 956,321 B 降至 495,809 B。FP16 与原 FP32 特征最大绝对误差 0.000326；一条真实轨迹 60/60 次确定性动作相同，type-logit 最大变化 2.98e-7。 | 训练时将 FP16 features 转回 FP32；离散 observation、合法动作和 token 不变。 |
| critic 请求 | 在 50 个模拟器时刻逐项比较旧 `PRIV` 派生的 16 维 critic 输入与新接口，向量完全相同。中位响应时间从 0.284 ms 降到 0.034 ms（8.26×）；响应 JSON 从约 32,581 B 降到 69 B。 | 增加 `CRITIC_INPUTS` 协议命令，只返回 wave timer 和当前 wave 的 zombie 类型；协议版本升到 4。 |
| CUDA PPO 批处理 | RTX 5080，32 局、1 epoch：sequence 16/chunks 16 的 dense 更新 2.904 秒、峰值分配 2.30 GB；同配置 exact Flex 为 1.536 秒、1.94 GB。完整 2,000 局、2 epochs、127,877 transitions：Flex 两次为 184.391/190.998 秒（中位 187.695 秒），dense 为 404.420 秒；中位速度比 2.15×。Flex 峰值分配 2.07 GB/预留 5.88 GB，dense 为 2.49 GB/6.44 GB。第一组 loss 分别为 policy/value 0.12451/0.01556（Flex）和 0.12455/0.01555（dense）；复跑 Flex 的 gradient norm 为 1.494。 | T5 默认 2,000 局 rollout、sequence 16、chunks 16、学习率 1e-4、attention `auto`。较大的 minibatch 以较少 optimizer step 换取吞吐，更新级能力 smoke 结果见下文。 |
| 精确关系注意力 | 相同 256 帧×89 token×192 width 的 forward+backward：dense relation 71.7 ms、547.8 MB；exact Flex relation 18.9 ms、330.0 MB，输出 RMS 误差 1.75e-7。 | CUDA 大批次采用 FlexAttention，并保留 kind/row/column/same-cell 学习关系偏置；小批次和 CPU 回退到 dense。编译为进程级一次性开销，正式 2,000 局更新的更新耗时不含初次编译。 |

### 实测后未采用

- **worker 设备/线程：** 初筛中 CUDA 多 worker rollout 明显慢于 CPU（CUDA 4×1 约 11,682 局/小时、8×1 约 11,217 局/小时）。CPU 多线程也较差（12×2 约 35,189 局/小时、16×2 约 27,470 局/小时）。连续短测曾使 20×1 看起来最快；按正式 `run_seed_jobs` 跑同一完整 2,000 局批次后，18×1 胜出，故按正式路径选型。
- **worker 持久池：** 18 个 worker 会在 PPO 更新期间额外常驻约 19 GB 进程树 RSS（RSS 合计会重复计入共享页）。进程池当前每个 2,000 局更新创建一次；跨更新保留它可能减少启动开销，但会把这部分内存与训练的约 3.8 GB CPU RSS、约 2 GB CUDA 分配同时叠加。没有证据证明这一内存代价值得承担，因此不提交持久池实现。
- **TF32/FP16/BF16：** TF32 在 32 局测试中与 FP32 基本同速；BF16 慢于 FP32；FP16 在一组 32 局测量中只快约 2%，另一组小样本反而更慢。它们没有带来稳定的大幅速度收益，默认保持 FP32。
- **SDPA relation bias：** exact SDPA 约 74.8 ms，慢于 dense 的 71.7 ms；保留 dense 作为小批次路径。Flex 的 exact relation 结果才有足够收益。
- **滑窗与线性注意力：** Flex local-window16 输出 RMS 误差 0.434；无关系偏置线性注意力误差 0.296；无关系偏置 SDPA 误差 0.199。虽然它们更快，但会丢失已训练的实体关系信息，未采用。
- **完整 OBS/action JSON 与 tokenization：** 正式 2,000 局 profile 的均值为模型 0.888 秒/局、simulator 0.071 秒/局、critic 0.004 秒/局、tokenization 0.033 秒/局。删除 simulator 和 tokenization 全部耗时的理论上限约 10.5%，实际可安全削减的部分更小；现有策略使用这些实体字段，因此不以有损 observation 简化换取有限吞吐。critic 的独立小响应则已精简并精确核对。
- **删除实体 token、增加 lane/global 聚合：** 本轮没有移除策略可见实体或改变表征，避免未经能力评估的语义删减；速度收益也未实测证明。

### 能力与正确性验收

- 2,000 局同数据 PPO dense/Flex 更新的 loss 与 gradient norm 接近；精确 attention 单层输出误差在 1e-6 量级。一次完整 Flex 更新前后，在 10 个冻结 held-out cap-3 任务上各取 2 个相同 seed：初始和更新后均为 0/20 胜出，终局 wave 分布也都为 13 局到达 wave 3、7 局在 wave 2 结束。这个 smoke 没有显示该次更新让胜率下降，但样本与初始策略过弱，不能证明多轮训练的最终能力；正式训练仍由冻结 held-out 曲线门禁决定是否继续或停止。
- `PYTHONPATH=python mlpython -m unittest test_agent_model test_env_protocol test_training_semantics test_seed_jobs test_shared_helpers`：113 项通过（协议 4、CUDA replay mask/index、紧凑轨迹和 GAE）。
- `cmake --build build -j6` 成功；协议 4 下单波关卡 `wave_cap=1` 跑到 tick 100，精简 critic 查询成功。`Board.cpp` 原有单波进度条除零修复保留并已单独提交。

### 固定复现实验

- worker 选择：`scripts/t5_worker_batch_benchmark.py`，证据 `artifacts/t5/perf/worker_batch_sweep_2000*.json`。
- PPO 更新：`scripts/ppo_update_benchmark.py`，证据 `artifacts/t5/perf/ppo_update_2000_{dense,flex}.json`。
- 注意力消融：`scripts/attention_benchmark.py`，证据 `artifacts/t5/perf/attention_sweep.json`。
- shard 格式：`scripts/trajectory_storage_benchmark.py`，证据 `artifacts/t5/perf/trajectory_storage_{baseline,narrow}.json`。
- 正式 rollout 单核门禁和默认 worker 配置：`artifacts/t5/throughput.json`。
