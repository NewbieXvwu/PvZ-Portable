# T5 训练与模拟器性能优化记录

测量日期：2026-09-29。基准在 `/home/newbiexvwu/.venvs/ml` 执行：PyTorch 2.14.0+cu132、RTX 5080 16GB、i7-12700F（20 个逻辑 CPU）。游戏资源使用 `/home/newbiexvwu/.cache/pvz-research-resources`，其中 `main.pak` 指向用户主目录下的资源。性能数据与任务 manifest 固定在 `artifacts/t5/perf/`；该目录被 Git 忽略，避免提交大型 rollout 数据。

## 已采用的优化

| 项目 | 实测结果 | 采用方式 |
|---|---|---|
| PPO rollout worker | 正式 `run_seed_jobs` 的同一批 2,000 局：18×CPU/1 Torch 线程 161.559 秒、44,565.9 局/小时；20×1 为 174.667 秒、41,221.3 局/小时；16×1 为 163.545 秒、44,024.6 局/小时。18 worker 的进程树 RSS 合计峰值 19,168 MB。 | 正式 T5 默认 CPU/18 worker/每 worker 1 线程；配置写入 `artifacts/t5/throughput.json`。单核门禁重跑为 6,273.8 局/小时，高于 5,000 门槛。 |
| 紧凑 rollout shard | 同一 60 步轨迹：gzip JSON round trip 中位数 122.4 ms、19,343 B、恢复对象 956,321 B；压缩 NPZ 为 25.1 ms、101,763 B、恢复对象 495,809 B；未压缩 NPZ 为 19.7 ms、444,979 B。 | 压缩 NPZ 保留 NumPy dtype。它比 gzip JSON 多用约 5.3 倍磁盘，但序列化约快 4.9 倍，恢复对象内存约减半。完整 2,000 局 shard 为 272 MB，解码 episode 对象合计约 1.11 GB。 |
| token 与 observation 存储 | 将 token ID 存为 int8、索引存为窄整数、features 存为 FP16 后，60 步恢复对象从 956,321 B 降至 495,809 B。FP16 与原 FP32 特征最大绝对误差 0.000326；一条真实轨迹 60/60 次确定性动作相同，type-logit 最大变化 2.98e-7。 | 训练时将 FP16 features 转回 FP32；离散 observation、合法动作和 token 不变。 |
| critic 请求 | 在 50 个模拟器时刻逐项比较旧 `PRIV` 派生的 16 维 critic 输入与新接口，向量完全相同。中位响应时间从 0.284 ms 降到 0.034 ms（8.26×）；响应 JSON 从约 32,581 B 降到 69 B。 | 增加 `CRITIC_INPUTS` 协议命令，只返回 wave timer 和当前 wave 的 zombie 类型；协议版本升到 4。 |
| CUDA PPO 批处理 | RTX 5080，32 局、1 epoch：sequence 16/chunks 16 的 dense 更新 2.904 秒、峰值分配 2.30 GB；同配置 exact Flex 为 1.536 秒、1.94 GB。完整 2,000 局、2 epochs、127,877 transitions：Flex 两次为 184.391/190.998 秒（中位 187.695 秒），dense 为 404.420 秒；中位速度比 2.15×。Flex 峰值分配 2.07 GB/预留 5.88 GB，dense 为 2.49 GB/6.44 GB。第一组 loss 分别为 policy/value 0.12451/0.01556（Flex）和 0.12455/0.01555（dense）；复跑 Flex 的 gradient norm 为 1.494。 | T5 默认 2,000 局 rollout、sequence 16、chunks 16、学习率 1e-4、attention `auto`。较大的 minibatch 以较少 optimizer step 换取吞吐，更新级能力 smoke 结果见下文。 |
| 精确关系注意力 | 相同 256 帧×89 token×192 width 的 forward+backward：dense relation 71.7 ms、547.8 MB；exact Flex relation 18.9 ms、330.0 MB，输出 RMS 误差 1.75e-7。 | CUDA 大批次采用 FlexAttention，并保留 kind/row/column/same-cell 学习关系偏置；小批次和 CPU 回退到 dense。编译为进程级一次性开销，正式 2,000 局更新的更新耗时不含初次编译。 |

## 实测后未采用

- **worker 设备/线程：** 初筛中 CUDA 多 worker rollout 明显慢于 CPU（CUDA 4×1 约 11,682 局/小时、8×1 约 11,217 局/小时）。CPU 多线程也较差（12×2 约 35,189 局/小时、16×2 约 27,470 局/小时）。连续短测曾使 20×1 看起来最快；按正式 `run_seed_jobs` 跑同一完整 2,000 局批次后，18×1 胜出，故按正式路径选型。
- **worker 持久池：** 18 个 worker 会在 PPO 更新期间额外常驻约 19 GB 进程树 RSS（RSS 合计会重复计入共享页）。进程池当前每个 2,000 局更新创建一次；跨更新保留它可能减少启动开销，但会把这部分内存与训练的约 3.8 GB CPU RSS、约 2 GB CUDA 分配同时叠加。没有证据证明这一内存代价值得承担，因此不提交持久池实现。
- **TF32/FP16/BF16：** TF32 在 32 局测试中与 FP32 基本同速；BF16 慢于 FP32；FP16 在一组 32 局测量中只快约 2%，另一组小样本反而更慢。它们没有带来稳定的大幅速度收益，默认保持 FP32。
- **SDPA relation bias：** exact SDPA 约 74.8 ms，慢于 dense 的 71.7 ms；保留 dense 作为小批次路径。Flex 的 exact relation 结果才有足够收益。
- **滑窗与线性注意力：** Flex local-window16 输出 RMS 误差 0.434；无关系偏置线性注意力误差 0.296；无关系偏置 SDPA 误差 0.199。虽然它们更快，但会丢失已训练的实体关系信息，未采用。
- **完整 OBS/action JSON 与 tokenization：** 正式 2,000 局 profile 的均值为模型 0.888 秒/局、simulator 0.071 秒/局、critic 0.004 秒/局、tokenization 0.033 秒/局。删除 simulator 和 tokenization 全部耗时的理论上限约 10.5%，实际可安全削减的部分更小；现有策略使用这些实体字段，因此不以有损 observation 简化换取有限吞吐。critic 的独立小响应则已精简并精确核对。
- **删除实体 token、增加 lane/global 聚合：** 本轮没有移除策略可见实体或改变表征，避免未经能力评估的语义删减；速度收益也未实测证明。

## 能力与正确性验收

- 2,000 局同数据 PPO dense/Flex 更新的 loss 与 gradient norm 接近；精确 attention 单层输出误差在 1e-6 量级。一次完整 Flex 更新前后，在 10 个冻结 held-out cap-3 任务上各取 2 个相同 seed：初始和更新后均为 0/20 胜出，终局 wave 分布也都为 13 局到达 wave 3、7 局在 wave 2 结束。这个 smoke 没有显示该次更新让胜率下降，但样本与初始策略过弱，不能证明多轮训练的最终能力；正式训练仍由冻结 held-out 曲线门禁决定是否继续或停止。
- `PYTHONPATH=python mlpython -m unittest test_agent_model test_env_protocol test_training_semantics test_seed_jobs test_shared_helpers`：113 项通过（协议 4、CUDA replay mask/index、紧凑轨迹和 GAE）。
- `cmake --build build -j6` 成功；协议 4 下单波关卡 `wave_cap=1` 跑到 tick 100，精简 critic 查询成功。`Board.cpp` 原有单波进度条除零修复保留并已单独提交。

## 固定复现实验

- worker 选择：`scripts/t5_worker_batch_benchmark.py`，证据 `artifacts/t5/perf/worker_batch_sweep_2000*.json`。
- PPO 更新：`scripts/ppo_update_benchmark.py`，证据 `artifacts/t5/perf/ppo_update_2000_{dense,flex}.json`。
- 注意力消融：`scripts/attention_benchmark.py`，证据 `artifacts/t5/perf/attention_sweep.json`。
- shard 格式：`scripts/trajectory_storage_benchmark.py`，证据 `artifacts/t5/perf/trajectory_storage_{baseline,narrow}.json`。
- 正式 rollout 单核门禁和默认 worker 配置：`artifacts/t5/throughput.json`。
