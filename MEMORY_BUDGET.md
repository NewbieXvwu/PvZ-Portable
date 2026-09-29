# 内存预算与轨迹表示

2026-09-29，针对"WSL 吃满 16 GB"的实测诊断。

**本文件只做诊断与规格，不在本轮实现。**
台式机上的 T5 正在跑，此时改 `python/` 会造成双方代码分歧；等那轮结束后按本文档合并。
（唯一例外是 §4 的配置层改动，它不碰仓库代码，可以现在做。）

---

## 1. 实测数据

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

## 2. 内存账（8 workers 情形）

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

## 3. 三项可消除的浪费（按性价比排序）

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

## 4. 配置层（不碰仓库代码，可以现在做）

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

## 5. 验收（硬要求）

1. **等价性（3.1 必须过）**：同一个 checkpoint、同一批 seed，改动前后
   每个 transition 的 `value` 必须**逐位相同**（`torch.equal`，不是近似）。
   3.1 只是把 16 个 float 提前算好，数学上恒等；不恒等说明实现错了。
   注意：`episode_hash` 会因此变化（字段换了），这是预期的一次性基线变更，
   必须在门禁证据的 `notes` 里写明，并保留改动前的 hash 作为对照。
2. **内存**：batch=2000 下实测主进程峰值 RSS + 全部子进程 RSS 之和，写进
   `artifacts/t5/memory.json`，目标 < 8 GB。
3. **吞吐**：端到端（rollout + 更新）局/小时，与改动前对照。不许只看 rollout。
4. **变异测试**（铁律）：注入"漏删一个 privileged_state"的缺陷，内存门禁必须报错。

## 6. 对既有判断的修正（重要）

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

## 7. 实施结果（2026-09-29 当天完成，提交 fe4d9e6 / 801dbc2 及后续）

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

## 8. 台式机实测回填（2026-09-29，已合入 HEAD `02125b9`）

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
汇总报告：`T5_PERFORMANCE_REPORT.md`、`OPTIMIZATION_TASKS.md`。
