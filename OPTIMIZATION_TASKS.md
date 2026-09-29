# 训练与模拟器性能优化任务清单

目标：只保留经实测能显著提速或降低内存、且能力回归没有明显恶化的改动。详细数据见 [T5_PERFORMANCE_REPORT.md](T5_PERFORMANCE_REPORT.md)。

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
- [x] 检查轨迹状态向量、合法动作和奖励构造；紧凑格式保留原字段，不删实体 token、不做 lane/global 汇总。
- [x] 用真实 2,000 局、2 epochs 更新对比 dense/Flex：Flex 为 184.391 秒，dense 为 404.420 秒；峰值 CUDA 分配 2.07/2.49 GB，reserved 5.88/6.44 GB。更新 loss 接近。

## 阶段 3：attention 消融

- [x] 建立 dense relation attention 的 forward+backward 基线：256 帧×89 token×192 width，71.7 ms、547.8 MB。
- [x] 测量 SDPA、保留 relation bias 的 FlexAttention、滑窗和线性 attention。Exact Flex 为 18.9 ms、330 MB、输出 RMS 误差 1.75e-7；纳入 CUDA 大 batch，保留 dense 小 batch/CPU fallback。
- [x] 量化近似 attention：滑窗 16、线性和无关系 SDPA 的输出 RMS 误差分别 0.434、0.296、0.199；均未采用。
- [x] 记录编译一次性成本和真实 2,000 局 PPO 更新；精确 Flex 更新约快 2.19×，峰值分配内存降低约 17%。

## 阶段 4：最终验收与提交

- [x] 每个保留项均有针对性基准或等价性检查；汇总改前/改后吞吐、内存、精度误差和能力回归证据于性能报告。
- [x] 完成 2,000 局 PPO 更新和正式 2,000 局 worker 批次测量。
- [x] 配对 held-out capability smoke：相同冻结 seeds 上初始与一次 2,000 局 Flex PPO 更新均为 0/20 胜出，终局 wave 分布相同；记录为无可见回退但能力证据有限，不替代正式 held-out 曲线门禁。
- [x] 更新最终证据并按 worker/storage、simulator protocol、PPO/CUDA 和 attention 分组提交；最终提交后复查工作区。
