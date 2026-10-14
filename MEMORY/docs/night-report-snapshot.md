# RL 微调夜间验证报告

**状态：进行中（2026-10-10 11:38 CST）。A1–A8 已完成；A9 v3 初始点与 100k 双模式均完成 1,024 局，500k greedy 暂停于 742/1,024 shards。v3 100k 相对初始点的连续变化两模式均不显著。B1–B7 已结束；B7 100k 主指标两模式均不显著。B8 `bc_roll64_v1` 已训练到 50k decisions / 2 updates（4,715 秒），正在执行 50k 评估；400k 预算、10h timeout。磁盘可用约 883 GB；无 git 提交、制品删除或 HF 上传。源码两处 CPU worker eager-path 修复有备份，见 §D；未运行测试。

## 先看结论

- 当前主指标选 **平均推进波数比例** `terminal_wave / wave_count`：256 局时节点内 95% CI 半宽中位 **1.79 个百分点**，约 **212 局**即可把单节点均值精度压到 ±2 点；胜负率需要约 **1,062 局**。
- 100k 的 A/B 差异在胜负和连续主指标上都**不显著**。v3 连续曲线在 400k 明显回落、600k 后回升；到 1M 连续指标是 +2.14 点（未通过 10 节点 Holm 校正），胜率是 +8.20 点（McNemar Holm 校正后 p=.00678）。
- 目标 KL 放宽并未改善主指标的结论仍是“不显著”；现有证据不支持把 1–3 点差异写成策略优劣。
- B1/B3/B4 均在 TorchDynamo 访问 CUDA current_stream 时失败；B2 在首个 PPO epoch 显存不足；B5 已训练到 100k，但 100k CPU evaluation 在相同 current_stream 路径失败。B6 已完成且主指标不显著；B7 正在训练 update 3。

## §A 轨道 A

### A1 — 连续指标与噪声

主指标为 `seed_results.<mode>.validation_full_level7_v1_d1[].terminal_wave / wave_count`。把 v3 初始点+10 个训练节点与 A/B 两节点合并（13 个 eval 节点），单节点 95% CI 半宽中位 **0.0182**（范围 0.0149–0.0233）；达到 ±0.02 精度的正态近似样本量中位数 **212**（范围 142–349）。v3 10 个训练节点的胜负率 CI 半宽中位 **4.07 点**（初始节点 4.45 点），达到 ±2 点约需 **1,062 局**。

同节点 greedy/sampled 按种子配对，波数比例均值差绝对值中位 **0.90 点**、配对 SE 中位 **0.81 点**、95% CI 半宽中位 **1.60 点**。其他配对噪声（中位 SE）：终局 tick **967 tick**；峰值火力 **0.161**；杀敌 **2.039**；花费阳光 **68.90**；植物被吃 **0.701**；割草机触发 **0.053**；种植动作 **0.652**；等待动作 **16.13**；铲除动作全为 0。来源：v3 与 A/B `output/evaluations/update_*.json.gz` 的 `seed_results.greedy/sampled.validation_full_level7_v1_d1[]`（事件字段取 `events.*`、动作字段取 `action_counts.*`），按同 seed 配对；逐项结果在 `/tmp/night_a1_fullnoise.json`。Idle 控制 256 局推进比例均值 **0.2591**、95% CI 半宽 **0.00281**，花费阳光为 0；来源 `artifacts/research/bc_finetune_level7_v3/output/evaluations/idle_control.json.gz` 的 `seed_results.validation_full_level7_v1_d1[]`。

### A2 — 种子不变比例

v3 的 11 个已存节点（初始点 + 10 个训练点）中，**97/256 = 37.89%** 的种子胜负全程不变，且全是一直赢；没有一直输的种子。200k→1M 翻转 **39/256（15.23%）**，其中 34 个转胜、5 个转负；300k→1M 翻转 **49/256（19.14%）**，44 转胜、5 转负。不存在精确 250k 节点，故用 200k/300k 给边界。0→1M 净增 21 个胜局（+8.20 点）。稳定种子比例低于 60%，按规则不把 B8 提前。此比例是“结果不随训练节点改变”的代理量，不能单独把净优势拆成随机运气与策略效应。来源同 A1 的 v3 评估文件；种子序列逐行取 `seed`、`won`；统计在 `/tmp/night_analysis_fast.json:A2`。

### A3 — A/B 重判与 v3 曲线

A/B 100k 对 v3 100k：胜率差 **−3.125 点**，配对 McNemar 精确 p **.2559**；连续推进差 **−0.872 点**，95% 配对 CI **[−2.634,+0.889] 点**、p **.3293**；两者都不显著。采样模式推进差 −0.807 点，p=.3688。AB 的峰值火力 +8.469、花费阳光 +1,110.6 虽显著变化，但不代表推进提升。来源 `artifacts/research/bc_kl_ab_v1/output/evaluations/update_000008.json.gz` 对 `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，按 256 个相同 seed 配对，字段 `won`、`terminal_wave/wave_count`、`peak_offense`、`events.sun_spent`。

v3 对初始点的连续推进差在 100k **+1.20 点**（p=.132）、200k **+0.29**（p=.715）、300k **−1.91**（p=.0608）、400k **−4.19**（95% CI [−6.91,−1.48]，Holm 后 p=.0236）、500k **+1.11**（p=.179）、600k **+2.21**（p=.0143）、700k **+2.20**（p=.0365）、800k **+0.98**（p=.320）、900k **+1.56**（p=.171）、1M **+2.14**（p=.0419）。600k 是首次名义显著回升点；10 个训练节点校正后没有正向连续指标显著点。胜率在 400k 降 **16.02 点**（McNemar p=1.27×10⁻⁵，Holm 后 p=.000127），在 1M 增 **8.20 点**（McNemar p=.000753，Holm 后 p=.00678）。连续指标在 400k 下降、600k 回升，未比胜负曲线更早、更单调地显出趋势。来源 v3 `update_000000.json.gz` 与 `update_000008`…`update_000084`，上述字段按 seed 配对；统计 `/tmp/night_analysis_fast.json:A3`。

### A4 — Φ 塑形是否抵消

对 v3 训练分片中 **1,159** 个 shard 解码，分析 **1,101** 个完整 episode，排除 58 个截断 episode。未折扣 `Σ shaping − (−Φ₀)` 均值 **−0.9708**（按 update 聚类 bootstrap 95% CI [−0.9910,−0.9508]），但 PPO 回报里的实际折扣和 `Σ discount_before_t × shaping_t` 均值为 **−0.0166666666667**，与 `−Φ₀` 的残差均值 **1.97×10⁻¹⁷**、最大绝对误差 **2.53×10⁻¹⁵**，即在数值精度内严格望远镜相消。26–40、41–55、56+ 种植动作组的折扣残差都约为 0；原始未折扣残差依次 −0.923/−0.964/−0.993，不能据此称目标在教“少种”。来源 `artifacts/research/bc_finetune_level7_v3/output/training_state.json:update_history[*].shard_directory` 下每个 `seed_*.npz` 的 `episode.transitions[*].{potential,discount,shaping_reward,action}`；重算在 `/tmp/night_analysis_a4.json`。

### A5 — critic 来源与语义

BC 检查点有 4 个 `privileged_critic.*` 参数，但它们与 v3 初始 critic 权重完全相等；BC 训练脚本只有行为克隆动作损失、没有 value target，所以 critic 是未训练的初始化值，这能解释负的 value explained variance，但尚未证明因果。BC 与 v3 初始 checkpoint 的 `value_semantics` 都是 `undiscounted_terminal_v1`；更早 v1/v2 初始 checkpoint 标记 `research_explicit_return_v1`，但 v3 使用的是 BC checkpoint。来源 `/tmp/night_analysis_fast.json:A5` 对应的 checkpoint `state_dict`、`value_semantics`、BC 训练代码 `scripts/bc_pipeline/train_bc_gru_sequence.py`。据此已把 `bc_criticwarm_v1` 加到队尾：总预算 100k，前 20k 冻结策略只训 critic，后 80k 恢复 v3 的正常 PPO。

### A6 — clip_fraction 偏低

v3 84 次更新的 `clip_fraction` 中位 **0.005285**，p90 **0.01172**，最大 **0.03242**；`approx_kl` 中位 **0.00408**，76/84 次更新因 KL 停止。优势标准化会缩放 policy loss，但不会改变 `ratio=exp(new_log_prob−old_log_prob)` 或 `|ratio−1|>clip_epsilon` 判定，故不能直接解释低 clip fraction。`minibatch_chunks=2` 是否导致实际步长过小不能从旧记录分离判断，B2/B7 对应检验。首 minibatch `first_minibatch_log_prob_max_error` 中位 **4.77×10⁻⁷**、最大 **3.05×10⁻⁵**。来源 v3 `output/learning_curve.json` / `training_state.json:update_history[*].losses.{clip_fraction,approx_kl,first_minibatch_log_prob_max_error,max_log_prob_change}`；统计 `/tmp/night_analysis_fast.json:A6`。

### A7 — 行为侧写

v3 1M greedy 256 局：`action_counts` 为种植 **12,495**、等待 **210,200**、铲除 **0**（等待占 94.39%）；每局平均种 **48.81** 次、等待 **821.09** 次，单次等待 **59.86 tick**。均值产阳光 **6,042**、花费 **4,854**（花费/产出 81.18%，零花费率 0），峰值火力 **21.85**（零火力率 0），杀敌 **148.59**、植物被吃 **16.44**、推进 **28.92/30** 波。经济曲线 12k/24k/36k/48k tick 的均值余额为 87.7/134.8/1,180.3/1,539.8；并非只收不花，后段有大量余额。胜局/败局的峰值火力为 22.28/16.42、花费阳光 4,981/3,278、杀敌 157.32/39.68，均通过 7 项 Welch 均值差的 Holm 校正（调整 p 分别 3.01×10⁻⁷、3.01×10⁻⁷、1.88×10⁻¹¹）；种植动作 +7.64（调整 p=.00171）、等待动作 +325.5（调整 p=4.30×10⁻⁵）；植物被吃 +0.36 的 95% CI [−2.89,+3.61]、p=.818，不显著。胜/败分别 237/19 局；这些是按结果分组的描述性关联，不是因果证据。来源 v3 1M greedy raw `seed_results.greedy.validation_full_level7_v1_d1[]` 的 `action_counts`、`wait_records`、`economy_curve`、`events`、`peak_offense`、`won`；统计 `/tmp/night_analysis_fast.json:A7` 和 `/tmp/night_a7_tests.json`。保存的评估只有汇总动作计数，没有逐步动作轨迹，因此不能回答具体哪行哪列种了哪种植物。

### A8 — 代码结构结论

1. `python/train_pvz_ppo.py:311–405` 每个 minibatch 都用当前参数重建 prefix hidden；分片只在当前 forward 是 no-grad，prefix 状态仍进当前窗口。`first_minibatch...` 中位 4.77e−7，而每更新保存的全 minibatch `max_log_prob_change` 中位 **6.15**、p90 **10.61**、最大 **56.52**（其定义是最大 `|new_log_prob−old_log_prob|`）；所以后续/少数样本差异确实远大于首批，但该最大值没有批次索引，不能单独归因到 prefix 漂移，更新参数变化也混在内。
2. 实验新建 AdamW，BC 没有传 optimizer moments；resume 时恢复训练状态，不能把“从零状态”说成所有更新都重置。早期动量影响未单独测量。
3. 更新任务 seed 由 `random.Random(sampling_seed)` 控制，`rng.choice(task["seeds"])` 有放回；16 episode/update 的 84 个 v3 更新平均有 **13.40** 个不同 seed，相邻更新 seed 集合交集均值 **0.807**、范围 0–4。
4. `frontier_v1` 当前只有 `train_full_level7_v1_d1` 一个 task，课程 frontier 在任务选择上无可筛选对象，实际只是从该任务 256 个 seed 随机抽样。
5. 54 是 cell-score head 的掩码槽，不是 315 项扁平离散动作。动作按类型、合法卡片、合法格子因子化；合法 mask 限制位置。它可表示 5×9 地图上的合法植物/位置组合，但网络没有一个 315-way 独立 logits。
6. `sequence_length=128` 意味着梯度最多经过 128 个决策；更早前缀用 no-grad 重算 hidden，状态仍传进窗口，梯度不跨过截断点。
7. `scripts/t4_capability_profile.py:run_episode` 与 `python/train_pvz_ppo.py:collect_task_episode` 每步都把 `output["hidden"]` 赋回下一步；训练 rollout 没发现类似 `render_episode.py:436` 的记忆不推进问题。

来源：`python/train_pvz_ppo.py`、`python/pvz_research.py:_assign`、`python/pvz_curriculum.py:probabilities`、`scripts/t4_capability_profile.py:run_episode`，以及 v3 `training_state.json`、`learning_curve.json`。

### A9 — 1024 种子可信重评

已验证评估清单显式 seeds 为 30000–31023（1,024 个）；按 A1 敏感度选 5 个关键节点：v3 0/100k/500k/1M 与 A/B 100k，greedy+sampled。A9 已完成 v3 初始点与 100k 双模式各 1,024 局；500k greedy 因 B8 50k 评估暂停于 742/1,024 shards。临时 evaluator 使用 12 个 CPU workers，`PVZ_RELATION_BIAS_FUSION=0` 禁用 CPU fused compile；B8 回到 PPO 训练后恢复 A9，进入下一次 rollout 或评估时再暂停。缓存路径 `artifacts/research/bc_eval_1024_v1/output/<node>/.seed_jobs/`；进度 `/tmp/night_a9_1024_cached.log`，分片数按各节点缓存目录的 `seed_*.npz` 计。

已完成的 v3 初始节点有 1,024 seeds：greedy 推进均值 94.430%（95% CI [93.458%,95.402%]），胜 863/1,024；sampled 均值 94.785%（CI [93.867%,95.704%]），胜 868/1,024。新旧评估重叠 seed 30,000–30,255 的 256 局在两模式下胜负、终局波数和终局 tick 都逐局完全一致。来源 `artifacts/research/bc_eval_1024_v1/output/v3_000000/evaluations/update_000000.json.gz` 与旧 256 局 `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000000.json.gz`，字段 `seed_results.<mode>.validation_full_level7_v1_d1[].{seed,won,terminal_wave,wave_count,terminal_tick}`。

v3 100k 的 1,024 seeds：greedy 推进均值 95.003%、胜 882/1,024；相对初始点配对差 +0.573 点（95% CI [−0.164,+1.310]，p=.127），胜负 McNemar p=.0814。sampled 均值 94.183%、胜 852/1,024；相对初始点 −0.602 点（CI [−1.356,+0.152]，p=.118），McNemar p=.160。两模式连续指标与胜率变化均不显著。来源 1024 局 `v3_000000/evaluations/update_000000.json.gz` 与 `v3_100k/evaluations/update_000008.json.gz` 的上述字段，按 seed 配对；连续差 CI/p 为配对差正态近似，胜率 p 为精确 McNemar。

## §B GPU 队列

B1–B7 已结束（B5 在评估阶段失败）；B8 `bc_roll64_v1` 已训练到 50,000 decisions / 2 updates，训练用时 4,715s，正在做 50k 评估。首次 64 episode rollout 用 4,678.5s，得到 49,588 decisions；第二次只收集 1 episode 到 50,000。其初始胜率为 greedy 216/256（84.38%）、sampled 218/256（85.16%），来自 `/tmp/night_bc_roll64_v1.log` 的 `evaluation update=0`；主连续指标待计算。A9 已完成 v3 100k sampled，500k greedy 暂停于 742/1,024 shards。主指标为按 seed 配对的 `terminal_wave / wave_count` 差；95% CI 与双侧 p 按配对差正态近似，胜率另外用精确 McNemar 检验。各实验 100k 结果对 v3 100k，B2 对 A/B 100k。

| 顺序 / run | 状态、耗时 | 主指标相对基线 | 显著性 |
|---|---|---|---|
| B1 `bc_seed1_v1` | **failed**，4,161s；32,429 decisions / 2 updates，未到 50k 评估点 | 无 100k 候选评估 | 不可检验；失败栈为 CPU rollout worker 的 TorchDynamo `current_stream` CUDA error，完整日志 `/tmp/night_bc_seed1_v1.log`。失败时 A9 旧 CPU 评估也在跑且使用 TorchDynamo，可能相关但未能证明；旧评估已停，A9 恢复时禁用了该编译路径 |
| B2 `bc_mb16_v1` | **failed**，1,679s；初始评估 512 局完成（greedy .8438 / sampled .8516），首批 rollout 16/16 完成；首个 PPO epoch 在 `scores + relation` 分配 530 MiB 时 CUDA OOM，`training_state` 仍为 0 decisions / 0 updates | 无 100k 候选评估 | 不可检验；错误栈 `/tmp/night_bc_mb16_v1.log`；无评估效应可判显著性 |
| B3 `bc_lr10_v1` | **failed**，3,310s；14,786 decisions / 1 update，未到评估点；CPU rollout worker `torch.compile` 首次执行触发 CUDA current_stream busy/unavailable，详见 `/tmp/night_bc_lr10_v1.log` | 无 100k 候选评估 | 不可检验 |
| B4 `bc_gae95_v1` | **failed**，3,054s；14,786 decisions / 1 update，未到评估点；update 1 有非致命 203 MB allocator OOM 警告，update 2 rollout 随后在 CPU worker `torch.compile` 请求 CUDA current_stream 时失败；详见 `/tmp/night_bc_gae95_v1.log` | 无 100k 候选评估 | 不可检验 |
| B5 bc_ent_v1 | failed_rc1，6,228s；训练到 100,000 decisions / 10 updates；50k 评估文件已写出，100k 评估未写出，失败栈在 CPU evaluation worker 的 TorchDynamo current_stream；详见 /tmp/night_bc_ent_v1.log | 50k 对 v3 100k：greedy −0.951pp（配对 95% CI [−2.499,+0.598]，p=.229），sampled −1.120pp（CI [−2.887,+0.647]，p=.214） | 50k 主指标均不显著。次指标胜率差异显著（greedy 173/256 vs 217/256，McNemar p=1.05e−8；sampled 186/256 vs 209/256，p=.00379），但预算不等，只作描述；100k 无结果不可判定 |
| B6 `bc_ep1_v1` | **completed**，8,627s；100,000 decisions / 109 episodes / 10 updates；100k 评估于 07:15 完成 | greedy 0.000 点（95% CI [−1.359,+1.359]，p=1.000）；sampled +0.260 点（CI [−1.408,+1.929]，p=.760），对 v3 100k | 两模式均不显著；胜率 McNemar：greedy 217/256 vs 217/256，p=1.000；sampled 217/256 vs 209/256，p=.268 |
| B7 `bc_clip05_v1` | **completed**，11,034s；100,000 decisions / 115 episodes / 10 updates；50k 与 100k 评估均完成 | 50k：greedy −0.417 点（CI [−1.949,+1.116]，p=.594），sampled −0.625 点（CI [−2.234,+0.984]，p=.447）；100k：greedy −0.208 点（CI [−1.748,+1.332]，p=.791），sampled −0.299 点（CI [−1.929,+1.330]，p=.719） | 两节点、两模式的主指标变化均不显著；没有证据显示 clip_epsilon 0.05 改变推进表现 |
| B8 `bc_roll64_v1` | **running**；10:19:19 启动，静态校验通过，400k 决策，10h timeout；50k 决策 / 2 updates 已完成，50k 评估进行中 | 等待评估文件 | 待定 |
| B9 `bc_seq32_v1` | queued | — | — |
| B10 `bc_val2_v1` | queued | — | — |
| 条件项 `bc_criticwarm_v1` | queued；A5 条件已满足；100k 总预算（20k critic-only + 80k PPO） | — | — |

B6 100k 配对统计来源：`artifacts/research/bc_ep1_v1/output/evaluations/update_000010.json.gz` 对 `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，按 seed 配对；连续主指标字段 `seed_results.<mode>.validation_full_level7_v1_d1[].{terminal_wave,wave_count}`；胜率字段 `won`。95% CI 与双侧 p 使用配对差的正态近似；胜率差用精确 McNemar。

B6 50k 配对统计来源：`artifacts/research/bc_ep1_v1/output/evaluations/update_000005.json.gz` 对 `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，按 seed 配对；主指标字段 `seed_results.<mode>.validation_full_level7_v1_d1[].{terminal_wave,wave_count}`。95% CI 与双侧 p 使用配对差的正态近似。

B7 50k 配对统计来源：`artifacts/research/bc_clip05_v1/output/evaluations/update_000005.json.gz` 对 `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，按 256 个 seed 配对；主指标字段 `seed_results.<mode>.validation_full_level7_v1_d1[].{terminal_wave,wave_count}`、胜率字段 `won`。主指标双侧 p 与 95% CI 用配对差正态近似，胜率用精确 McNemar；greedy 215/256 vs 217/256（discordant 17/19，p=.868），sampled 208/256 vs 209/256（17/18，p=1.000）。

B7 100k 配对统计来源：`artifacts/research/bc_clip05_v1/output/evaluations/update_000010.json.gz` 对 `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，按 256 个 seed 配对；主指标字段 `seed_results.<mode>.validation_full_level7_v1_d1[].{terminal_wave,wave_count}`、胜率字段 `won`。greedy 胜率 213/256 vs 217/256（discordant 17/21，McNemar p=.627）；sampled 209/256 vs 209/256（23/23，p=1.000）。配对主指标 p 和 CI 用正态近似。

B5 50k 配对统计来源：`artifacts/research/bc_ent_v1/output/evaluations/update_000005.json.gz` 对 `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，按 `seed` 配对；主指标字段 `seed_results.<mode>.validation_full_level7_v1_d1[].{terminal_wave,wave_count}`，胜率字段 `won`。

队列由 `/tmp/night_queue.sh` 驱动，每个 run 独立日志 `/tmp/night_<run>.log`，进度 `/tmp/night_progress.txt`；配置静态校验通过后才启动。A1 判断连续指标足够敏感，因此 B1–B3 保持 100k 预算；A2 的不变 seed 比例为 37.9%，B8 不提前。B1 实际配置将 `initialization_seed` 与 `sampling_seed` 从 v3 的 0/1701 改成 1/1（sampling 源值与任务文字里的“0→1”不一致，已如实记录）；其它实验遵守单字段改动。磁盘每轮启动前复查，低于 30GB 会停止。

## §C 暂定总结

当前证实：Φ 的折扣后塑形项精确望远镜相消；A/B 与 v3 100k 差异不显著；B6 的 PPO epochs=1 与 v3 100k 两模式均不显著；B7 的 clip_epsilon 0.05 在 50k、100k 两节点和两模式下均未检测到显著主指标差异。critic 未接受价值监督；单任务 frontier sampler 无任务可筛。当前证伪：把 A/B 的 −3.1 点胜率差当成真实变差。B1/B3/B4 因 CPU rollout compile/current_stream 错误失败；B5 在 rollout 修复后训练到 100k，但 CPU evaluation 再次遇到同类错误，100k 结果缺失。B8 400k 大 batch 实验正进入 PPO epoch 1/2，A9 在其 PPO 阶段并行续跑。

## §D 源码修复（单列）

B1/B3/B4 的 CPU rollout worker 首次 torch.compile 调用 torch.accelerator.current_stream() 时收到 CUDA busy/unavailable。第一处修复在 _init_worker 中仅对 CPU worker 调用 set_relation_bias_fusion(False)；B5 因此越过 rollout 并训练到 100k、完成 50k 评估。B5 100k 评估仍在 _init_evaluation_worker 的同类 compile/current_stream 路径失败。第二处已在 _init_evaluation_worker 加入同样的 CPU-only eager 设置，CUDA learner 未改。pvz_agent_model.py 说明 eager/fused 关系偏置逐元素 bit-identical；CPU worker 可能较慢。

原始源码 SHA256 654397fe6660e8901002ef091d3c2d789f0bf6b13d070c326a9b41eecddec974 备份在 /tmp/night_code_change_backup/train_pvz_ppo_task_family.py.orig；第二处修复前版本（含 rollout 修复）备份在 /tmp/night_code_change_backup/train_pvz_ppo_task_family.py.pre_eval_fix，SHA256 e77954ae8aa6040de6e466792ee5c8753906efcffe2d9f1036dae8b8e99ebcea。完整回滚命令：cp /tmp/night_code_change_backup/train_pvz_ppo_task_family.py.orig /home/newbiexvwu/PvZ-Portable/python/train_pvz_ppo_task_family.py；仅回滚 evaluation 修复：cp /tmp/night_code_change_backup/train_pvz_ppo_task_family.py.pre_eval_fix /home/newbiexvwu/PvZ-Portable/python/train_pvz_ppo_task_family.py。回滚与修改说明在 /tmp/night_code_change_backup/README.txt。B6 的 512 个初始 evaluation episode 与 16 局 rollout 均完成且未复现该错误；未运行测试。
