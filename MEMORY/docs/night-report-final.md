# RL 微调夜间验证报告

**状态：完成（2026-10-10 21:51 CST）。A1–A9 与 B1–B10 均已核对；A9 的 5 个关键节点均有 greedy/sampled 1,024 局 raw。v3 1M 相对初始点连续推进经 6 项 Holm 校正仍显著；A/B 的连续主指标两模式经 Holm 后不显著，但 1,024 局胜率差显著偏低。B9、B10 与条件项 criticwarm 在 100k 显著退化；criticwarm 是单个训练种子，需独立复现。磁盘约 878GB；未提交、未删 artifacts、未上传 HF。临时 runner/evaluator 修改、备份和回滚见 §D。**

## 先看结论

- 主指标预先采用连续推进比例 `terminal_wave / wave_count`。256 局评估时单节点 95% CI 半宽中位 1.82pp，约 212 局可把均值精度压到 ±2pp；胜率需约 1,062 局。连续指标是主判据，胜率作为辅助终点单独报告。
- A9 用 1,024 个配对种子确认 v3 1M 相对初始点推进 +1.78pp（greedy）和 +1.39pp（sampled）；对三个关键训练节点、两种模式共 6 项检验做 Holm 校正后 p=.00154/.0348，均显著。100k、500k 均不显著。
- A/B 100k 的 1,024 局最终结果：连续主指标 greedy −0.758pp（两模式 Holm p=.0857）、sampled −0.238pp（p=.576），不显著；胜率分别下降 4.69pp、3.52pp，McNemar p=5.37e−5/.00520。结论依终点而异，不能把两者压成同一个“显著/不显著”。
- B9 `sequence_length=32` 与 B10 `value_coefficient=2.0` 在 100k 的连续推进显著下降。criticwarm 在 20k 冻结期逐局与初始策略完全一致，解冻后到 100k 崩塌 42.8/44.2pp；这是一个训练种子，因果仍需复现。
- B1/B3/B4 因 CPU worker `current_stream` 错误失败；B2 OOM；B5 100k 评估失败。B6/B7 主指标不显著；B8 只在 50k 显著下降，100k/400k 不显著。

## §A 轨道 A

### A1 — 连续指标与噪声

主指标为 `seed_results.<mode>.validation_full_level7_v1_d1[].terminal_wave / wave_count`。把 v3 初始点+10 个训练节点与 A/B 两节点合并（13 个 eval 节点），单节点 95% CI 半宽中位 **0.0182**（范围 0.0149–0.0233）；达到 ±0.02 精度的正态近似样本量中位数 **212**（范围 142–349）。v3 10 个训练节点的胜负率 CI 半宽中位 **4.07 点**（初始节点 4.45 点），达到 ±2 点约需 **1,062 局**。

同节点 greedy/sampled 按种子配对，波数比例均值差绝对值中位 **0.90 点**、配对 SE 中位 **0.81 点**、95% CI 半宽中位 **1.60 点**。其他配对噪声（中位 SE）：终局 tick **967 tick**；峰值火力 **0.161**；杀敌 **2.039**；花费阳光 **68.90**；植物被吃 **0.701**；割草机触发 **0.053**；种植动作 **0.652**；等待动作 **16.13**；铲除动作全为 0。来源：v3 与 A/B `output/evaluations/update_*.json.gz` 的 `seed_results.greedy/sampled.validation_full_level7_v1_d1[]`（事件字段取 `events.*`、动作字段取 `action_counts.*`），按同 seed 配对；逐项结果在 `/tmp/night_a1_fullnoise.json`。Idle 控制 256 局推进比例均值 **0.2591**、95% CI 半宽 **0.00281**，花费阳光为 0；来源 `artifacts/research/bc_finetune_level7_v3/output/evaluations/idle_control.json.gz` 的 `seed_results.validation_full_level7_v1_d1[]`。

### A2 — 种子不变比例

v3 的 11 个已存节点（初始点 + 10 个训练点）中，**97/256 = 37.89%** 的种子胜负全程不变，且全是一直赢；没有一直输的种子。200k→1M 翻转 **39/256（15.23%）**，其中 34 个转胜、5 个转负；300k→1M 翻转 **49/256（19.14%）**，44 转胜、5 转负。不存在精确 250k 节点，故用 200k/300k 给边界。0→1M 净增 21 个胜局（+8.20 点）。稳定种子比例低于 60%，按规则不把 B8 提前。此比例是“结果不随训练节点改变”的代理量，不能单独把净优势拆成随机运气与策略效应。来源同 A1 的 v3 评估文件；种子序列逐行取 `seed`、`won`；统计在 `/tmp/night_analysis_fast.json:A2`。

### A3 — A/B 重判与 v3 曲线
A/B 在 256 局时连续推进 greedy −0.872pp（95% CI [−2.634,+0.889]，p=.329）、sampled −0.807pp（p=.369），胜率 McNemar p=.2559，均不显著。A9 对相同 100k 检查点补足到 1,024 seeds 后，连续主指标 greedy −0.758pp（CI [−1.493,−0.024]，未校正 p=.0429；两模式 Holm p=.0857），sampled −0.238pp（CI [−1.070,+0.595]，p=.576）；按两模式校正后连续主指标均不显著。胜率则下降 4.69/3.52pp，McNemar p=5.37×10⁻⁵/.00520，两项 Holm 校正后 p=.000107/.00520，仍显著。故 256 seeds 的原结论只适用于当时的宽置信区间，1024 seeds 表明胜率差异可检出，但主指标未通过多模式校正。A/B 的峰值火力、阳光花费等行为字段不替代推进主指标。来源：A9 AB 1024 raw `artifacts/research/bc_eval_1024_v1/output/ab_100k/evaluations/update_000008.json.gz` 对 v3 100k `.../v3_100k/evaluations/update_000008.json.gz`，按 seed 配对；完整原始 256 局在 `artifacts/research/bc_kl_ab_v1/output/evaluations/update_000008.json.gz`，字段 `seed_results.<mode>.validation_full_level7_v1_d1[].{seed,won,terminal_wave,wave_count}`。

v3 对初始点的连续推进差在 100k **+1.20 点**（p=.132）、200k **+0.29**（p=.715）、300k **−1.91**（p=.0608）、400k **−4.19**（95% CI [−6.91,−1.48]，Holm 后 p=.0236）、500k **+1.11**（p=.179）、600k **+2.21**（p=.0143）、700k **+2.20**（p=.0365）、800k **+0.98**（p=.320）、900k **+1.56**（p=.171）、1M **+2.14**（p=.0419）。600k 是首次名义显著回升点；10 个训练节点校正后没有正向连续指标显著点。胜率在 400k 降 **16.02 点**（McNemar p=1.27×10⁻⁵，Holm 后 p=.000127），在 1M 增 **8.20 点**（McNemar p=.000753，Holm 后 p=.00678）。连续指标在 400k 下降、600k 回升，未比胜负曲线更早、更单调地显出趋势。来源 v3 `update_000000.json.gz` 与 `update_000008`…`update_000084`，上述字段按 seed 配对；统计 `/tmp/night_analysis_fast.json:A3`。

A9 对 v3 选定关键节点用 1,024 个 seed 重评后，100k 与 500k 相对初始点仍不显著；1M 推进分别 +1.777pp（greedy，95% CI [+0.825,+2.730]，p=.000256）和 +1.387pp（sampled，CI [+0.380,+2.394]，p=.00696）。对 100k/500k/1M × 两模式共 6 项配对主指标检验作 Holm 校正后，1M p=.00154/.0348，均显著。该子集重评补充了原 10 节点、256 局曲线的精度，不改变原曲线出现 400k 回落、600k 回升的非单调事实。来源见 §A9。

### A4 — Φ 塑形是否抵消

对 v3 训练分片中 **1,159** 个 shard 解码，分析 **1,101** 个完整 episode，排除 58 个截断 episode。未折扣 `Σ shaping − (−Φ₀)` 均值 **−0.9708**（按 update 聚类 bootstrap 95% CI [−0.9910,−0.9508]），但 PPO 回报里的实际折扣和 `Σ discount_before_t × shaping_t` 均值为 **−0.0166666666667**，与 `−Φ₀` 的残差均值 **1.97×10⁻¹⁷**、最大绝对误差 **2.53×10⁻¹⁵**，即在数值精度内严格望远镜相消。26–40、41–55、56+ 种植动作组的折扣残差都约为 0；原始未折扣残差依次 −0.923/−0.964/−0.993，不能据此称目标在教“少种”。来源 `artifacts/research/bc_finetune_level7_v3/output/training_state.json:update_history[*].shard_directory` 下每个 `seed_*.npz` 的 `episode.transitions[*].{potential,discount,shaping_reward,action}`；重算在 `/tmp/night_analysis_a4.json`。

### A5 — critic 来源与语义

BC 检查点有 4 个 `privileged_critic.*` 参数，但它们与 v3 初始 critic 权重完全相等；BC 训练脚本只有行为克隆动作损失、没有 value target，所以 critic 是未训练的初始化值，这能解释负的 value explained variance，但尚未证明因果。BC 与 v3 初始 checkpoint 的 `value_semantics` 都是 `undiscounted_terminal_v1`；更早 v1/v2 初始 checkpoint 标记 `research_explicit_return_v1`，但 v3 使用的是 BC checkpoint。来源 `/tmp/night_analysis_fast.json:A5` 对应的 checkpoint `state_dict`、`value_semantics`、BC 训练代码 `scripts/bc_pipeline/train_bc_gru_sequence.py`。据此已把 `bc_criticwarm_v1` 加到队尾：总预算 100k，前 20k 冻结策略只训 critic，后 80k 恢复 v3 的正常 PPO。

### A6 — clip_fraction 偏低

v3 84 次更新的 `clip_fraction` 中位 **0.005285**，p90 **0.01172**，最大 **0.03242**；`approx_kl` 中位 **0.00408**，76/84 次更新因 KL 停止。优势标准化会缩放 policy loss，但不会改变 `ratio=exp(new_log_prob−old_log_prob)` 或 `|ratio−1|>clip_epsilon` 判定，故不能直接解释低 clip fraction。`minibatch_chunks=2` 是否导致实际步长过小不能从旧记录分离判断，B2/B7 对应检验。首 minibatch `first_minibatch_log_prob_max_error` 中位 **4.77×10⁻⁷**、最大 **3.05×10⁻⁵**。来源 v3 `output/learning_curve.json` / `training_state.json:update_history[*].losses.{clip_fraction,approx_kl,first_minibatch_log_prob_max_error,max_log_prob_change}`；统计 `/tmp/night_analysis_fast.json:A6`。

### A7 — 行为侧写

v3 1M greedy 256 局：`action_counts` 为种植 **12,495**、等待 **210,200**、铲除 **0**（等待占 94.39%）；每局平均种 **48.81** 次、等待 **821.09** 次，单次等待 **59.86 tick**。均值产阳光 **6,042**、花费 **4,854**（花费/产出 81.18%，零花费率 0），峰值火力 **21.85**（零火力率 0），杀敌 **148.59**、植物被吃 **16.44**、推进 **28.92/30** 波。经济曲线 12k/24k/36k/48k tick 的均值余额为 87.7/134.8/1,180.3/1,539.8；并非只收不花，后段有大量余额。胜局/败局的峰值火力为 22.28/16.42、花费阳光 4,981/3,278、杀敌 157.32/39.68，均通过 7 项 Welch 均值差的 Holm 校正（调整 p 分别 3.01×10⁻⁷、3.01×10⁻⁷、1.88×10⁻¹¹）；种植动作 +7.64（调整 p=.00171）、等待动作 +325.5（调整 p=4.30×10⁻⁵）；植物被吃 +0.36 的 95% CI [−2.89,+3.61]、p=.818，不显著。胜/败分别 237/19 局；这些是按结果分组的描述性关联，不是因果证据。来源 v3 1M greedy raw `seed_results.greedy.validation_full_level7_v1_d1[]` 的 `action_counts`、`wait_records`、`economy_curve`、`events`、`peak_offense`、`won`；统计 `/tmp/night_analysis_fast.json:A7` 和 `/tmp/night_a7_tests.json`。保存的评估只有汇总动作计数，没有逐步动作轨迹，因此不能回答具体哪行哪列种了哪种植物。

criticwarm 100k 的描述性行为对照（每模式 256 局）：greedy 每局 plant/wait/shovel 57.58/590.20/1.92，对照 v3 100k 为 54.40/887.32/0；peak_offense 12.89 vs 19.46，zombies_killed 44.70 vs 144.14，plants_eaten 31.61 vs 22.91，sun_spent 3,782 vs 4,885。它不是“停止种植”，而是火力与杀敌明显不足、等待次数减少且铲除增多；这是描述性侧写，不能独立证明机制。来源 `artifacts/research/bc_criticwarm_v1/output/evaluations/update_000012.json.gz` 和 v3 100k update_000008 的 `action_counts`、`peak_offense`、`events.*`。

### A8 — 代码结构结论

1. `python/train_pvz_ppo.py:311–405` 每个 minibatch 都用当前参数重建 prefix hidden；分片只在当前 forward 是 no-grad，prefix 状态仍进当前窗口。`first_minibatch...` 中位 4.77e−7，而每更新保存的全 minibatch `max_log_prob_change` 中位 **6.15**、p90 **10.61**、最大 **56.52**（其定义是最大 `|new_log_prob−old_log_prob|`）；所以后续/少数样本差异确实远大于首批，但该最大值没有批次索引，不能单独归因到 prefix 漂移，更新参数变化也混在内。
2. 实验新建 AdamW，BC 没有传 optimizer moments；resume 时恢复训练状态，不能把“从零状态”说成所有更新都重置。早期动量影响未单独测量。
3. 更新任务 seed 由 `random.Random(sampling_seed)` 控制，`rng.choice(task["seeds"])` 有放回；16 episode/update 的 84 个 v3 更新平均有 **13.40** 个不同 seed，相邻更新 seed 集合交集均值 **0.807**、范围 0–4。
4. `frontier_v1` 当前只有 `train_full_level7_v1_d1` 一个 task，课程 frontier 在任务选择上无可筛选对象，实际只是从该任务 256 个 seed 随机抽样。
5. 54 是 cell-score head 的掩码槽，不是 315 项扁平离散动作。动作按类型、合法卡片、合法格子因子化；合法 mask 限制位置。它可表示 5×9 地图上的合法植物/位置组合，但网络没有一个 315-way 独立 logits。
6. `sequence_length=128` 意味着梯度最多经过 128 个决策；更早前缀用 no-grad 重算 hidden，状态仍传进窗口，梯度不跨过截断点。
7. `scripts/t4_capability_profile.py:run_episode` 与 `python/train_pvz_ppo.py:collect_task_episode` 每步都把 `output["hidden"]` 赋回下一步；训练 rollout 没发现类似 `render_episode.py:436` 的记忆不推进问题。

来源：`python/train_pvz_ppo.py`、`python/pvz_research.py:_assign`、`python/pvz_curriculum.py:probabilities`、`scripts/t4_capability_profile.py:run_episode`，以及 v3 `training_state.json`、`learning_curve.json`。

### A9 — 1,024 种子关键节点重评

评估清单显式种子为 30000–31023。按 A1 的 ±2pp 精度判断，仅重评 v3 初始、100k、500k、1M 与 AB 100k 五个关键节点，每个节点 greedy/sampled 各 1,024 局。v3 四节点 raw 位于 `artifacts/research/bc_eval_1024_v1/output/<node>/evaluations/update_*.json.gz`；AB raw 为 `artifacts/research/bc_eval_1024_v1/output/ab_100k/evaluations/update_000008.json.gz`。全部记录字段为 `seed_results.<mode>.validation_full_level7_v1_d1[]`，按 `seed` 配对；主指标 `terminal_wave / wave_count`，胜负辅助字段 `won`。

v3 初始节点：greedy 推进均值 94.430%（863/1,024 胜），sampled 94.785%（868/1,024 胜）。v3 100k：greedy 95.003%、882 胜，相对初始 +0.573pp（CI [−0.164,+1.310]，p=.127，McNemar p=.0814）；sampled 94.183%、852 胜，−0.602pp（CI [−1.356,+0.152]，p=.118，McNemar p=.160）。均不显著。v3 500k：greedy 94.967%、870 胜，+0.537pp（CI [−0.212,+1.286]，p=.160，McNemar p=.595）；sampled 94.941%、881 胜，+0.156pp（CI [−0.729,+1.042]，p=.729，McNemar p=.319）。均不显著。

v3 1M：greedy 96.208%、937 胜，相对初始 +1.777pp（95% CI [+0.825,+2.730]，配对正态近似 p=.000256；胜负 McNemar p=1.98×10⁻⁹）；sampled 96.172%、929 胜，+1.387pp（CI [+0.380,+2.394]，p=.00696；McNemar p=3.56×10⁻⁶）。对三个训练节点 × 两模式 6 项主指标检验作 Holm 校正后 p=.00154/.0348，均显著。旧 256 seeds 的 10 节点曲线结果仍作为高频轨迹；此处是预先按 A1 精度选的关键节点校验。

A/B 100k 对 v3 100k：greedy 推进均值 94.245% vs 95.003%，配对差 −0.758pp（95% CI [−1.493,−0.024]，原始 p=.0429；两模式 Holm p=.0857）；sampled 93.945% vs 94.183%，差 −0.238pp（CI [−1.070,+0.595]，p=.576）。连续主指标经两模式校正后均不显著。胜率 greedy 834/1,024 vs 882/1,024，差 −4.69pp（discordant 45/93，McNemar p=5.37×10⁻⁵，Holm p=.000107）；sampled 816/1,024 vs 852/1,024，差 −3.52pp（61/97，p=.00520，Holm p=.00520）；两模式均显著。来源同上 AB raw 对 v3 100k raw。

AB raw 的前 256 seeds 复用已有 `artifacts/research/bc_kl_ab_v1/output/evaluations/update_000008.json.gz`；该记录 `experiment_id=bc_kl_ab_v1_seed0`、decisions=100,000、种子 30000–30255。其 greedy/sampled 每条记录与新 raw 的前 256 条逐字段完全一致；新增评估 768 局覆盖 30256–31023。来源检查点为 `artifacts/research/bc_kl_ab_v1/output/runs/run_1/update_000008_evaluated_1791546515834898308.pt`，SHA256 `9c32841ee6efb5e218689c7c7d68fe5d2c92dde9d2ce5a80854e3fce284b2eb7`。

A9 evaluator 使用 CPU、`PVZ_RELATION_BIAS_FUSION=0`，缓存于各节点 `.seed_jobs/`，进度日志 `/tmp/night_a9_1024_cached.log`，完成标志 `/tmp/night_analysis_a9.json`。连续差的 CI/p 为按 seed 配对正态近似；多节点主指标按 6 项 Holm 校正；胜率 p 为精确 McNemar，AB 两模式 Holm 校正。每个 mode 原始记录均有 1,024 个唯一 seeds，范围 30000–31023。

## §B GPU 队列

B1–B10 均按序尝试；失败项保留在表内，不作效果结论。B8 完成 400k；条件项 criticwarm 完成 100k（18:59:58–21:16:13，8,175s），并在 20k/60k/100k 评估。各可评估结果的主指标均为按 seed 配对的 `terminal_wave / wave_count`；95% CI 与双侧 p 用配对差正态近似，胜率 `won` 用精确 McNemar。除另注外，100k 候选都对照 v3 100k；B2 对照 AB 100k。完整 queue 状态在 `/tmp/night_progress.txt`，各日志 `/tmp/night_<run>.log`。

| 顺序 / run | 状态、耗时 | 主指标相对基线 | 显著性 |
|---|---|---|---|
| B1 `bc_seed1_v1` | **failed**，4,161s；32,429 decisions / 2 updates，未到 50k 评估点 | 无 100k 候选评估 | 不可检验；失败栈为 CPU rollout worker 的 TorchDynamo `current_stream` CUDA error，完整日志 `/tmp/night_bc_seed1_v1.log`。失败时 A9 旧 CPU 评估也在跑且使用 TorchDynamo，可能相关但未能证明；旧评估已停，A9 恢复时禁用了该编译路径 |
| B2 `bc_mb16_v1` | **failed**，1,679s；初始评估 512 局完成（greedy .8438 / sampled .8516），首批 rollout 16/16 完成；首个 PPO epoch 在 `scores + relation` 分配 530 MiB 时 CUDA OOM，`training_state` 仍为 0 decisions / 0 updates | 无 100k 候选评估 | 不可检验；错误栈 `/tmp/night_bc_mb16_v1.log`；无评估效应可判显著性 |
| B3 `bc_lr10_v1` | **failed**，3,310s；14,786 decisions / 1 update，未到评估点；CPU rollout worker `torch.compile` 首次执行触发 CUDA current_stream busy/unavailable，详见 `/tmp/night_bc_lr10_v1.log` | 无 100k 候选评估 | 不可检验 |
| B4 `bc_gae95_v1` | **failed**，3,054s；14,786 decisions / 1 update，未到评估点；update 1 有非致命 203 MB allocator OOM 警告，update 2 rollout 随后在 CPU worker `torch.compile` 请求 CUDA current_stream 时失败；详见 `/tmp/night_bc_gae95_v1.log` | 无 100k 候选评估 | 不可检验 |
| B5 bc_ent_v1 | failed_rc1，6,228s；训练到 100,000 decisions / 10 updates；50k 评估文件已写出，100k 评估未写出，失败栈在 CPU evaluation worker 的 TorchDynamo current_stream；详见 /tmp/night_bc_ent_v1.log | 50k 对 v3 100k：greedy −0.951pp（配对 95% CI [−2.499,+0.598]，p=.229），sampled −1.120pp（CI [−2.887,+0.647]，p=.214） | 50k 主指标均不显著。次指标胜率差异显著（greedy 173/256 vs 217/256，McNemar p=1.05e−8；sampled 186/256 vs 209/256，p=.00379），但预算不等，只作描述；100k 无结果不可判定 |
| B6 `bc_ep1_v1` | **completed**，8,627s；100,000 decisions / 109 episodes / 10 updates；100k 评估于 07:15 完成 | greedy 0.000 点（95% CI [−1.359,+1.359]，p=1.000）；sampled +0.260 点（CI [−1.408,+1.929]，p=.760），对 v3 100k | 两模式均不显著；胜率 McNemar：greedy 217/256 vs 217/256，p=1.000；sampled 217/256 vs 209/256，p=.268 |
| B7 `bc_clip05_v1` | **completed**，11,034s；100,000 decisions / 115 episodes / 10 updates；50k 与 100k 评估均完成 | 50k：greedy −0.417 点（CI [−1.949,+1.116]，p=.594），sampled −0.625 点（CI [−2.234,+0.984]，p=.447）；100k：greedy −0.208 点（CI [−1.748,+1.332]，p=.791），sampled −0.299 点（CI [−1.929,+1.330]，p=.719） | 两节点、两模式的主指标变化均不显著；没有证据显示 clip_epsilon 0.05 改变推进表现 |
| B8 `bc_roll64_v1` | **completed**；400,000 decisions / 465 episodes / 11 updates，用时 14,595s；最终评估完成 | 50k：greedy −3.320 点（CI [−5.514,−1.127]，p=.00301）；sampled −5.026 点（CI [−7.397,−2.655]，p=3.26×10⁻⁵）。100k：greedy −1.211 点（CI [−2.852,+0.431]，p=.148）；sampled +0.573 点（CI [−1.231,+2.377]，p=.534）。400k 另见下文 | 50k 两模式显著下降；100k 两模式不显著；400k greedy 名义 p=.0245，但六项 Holm 后 p=.098，不显著；sampled p=.277 |
| B9 `bc_seq32_v1` | **completed**；100,000 decisions / 111 episodes / 8 updates；6,233s，50k 与 100k 评估完成；日志 `/tmp/night_bc_seq32_v1.log` | 50k：greedy −0.182pp（CI [−1.666,+1.302]，p=.810），sampled −0.286pp（CI [−1.894,+1.321]，p=.727）；100k：greedy −14.362pp（CI [−17.494,−11.230]，p=1.68×10⁻¹⁹），sampled −15.951pp（CI [−19.001,−12.900]，p=6.93×10⁻²⁵） | 50k 两模式不显著；100k 两模式显著下降。100k 胜率也显著下降：greedy 75/256 vs 217/256，McNemar p=4.65×10⁻³⁵；sampled 69/256 vs209/256，p=1.71×10⁻³⁰ |
| B10 `bc_val2_v1` | **completed**；100,000 decisions / 9 updates；训练与 50k/100k 评估完成，耗时约 10,170s；日志 `/tmp/night_bc_val2_v1.log` | 50k：greedy −0.104pp（p=.885），sampled +0.990pp（p=.256）。100k：greedy −9.375pp（CI [−12.296,−6.454]，p=3.17×10⁻¹⁰）；sampled −8.190pp（CI [−11.213,−5.167]，p=1.09×10⁻⁷），对 v3 100k | 50k 均不显著；100k 两模式显著下降；胜率 McNemar p=1.87×10⁻¹³ / 1.28×10⁻⁷ |
| 条件项 `bc_criticwarm_v1` | **completed**；100,000 decisions / 12 updates；用时 8,175s（18:59:58–21:16:13），20k/60k/100k 评估均完成；日志 `/tmp/night_bc_criticwarm_v1.log` | 60k：greedy −0.586pp（p=.414），sampled −0.781pp（p=.400），主指标不显著；100k：greedy −42.77pp（CI [−45.87,−39.67]，p=1.05×10⁻¹⁶²）；sampled −44.18pp（CI [−47.29,−41.07]，p=5.50×10⁻¹⁷²） | 60k 连续主指标不显著；100k 两模式均经 Holm 仍显著大幅退化。胜率 100k 亦显著下降；单个训练种子，需独立复现 |

criticwarm 20k actor-freeze 验证：`artifacts/research/bc_criticwarm_v1/output/evaluations/update_000000.json.gz` 与 `update_000002.json.gz` 在 greedy/sampled 各 256 个相同 seed 上，`won`、`terminal_wave`、`terminal_tick`、`action_counts`、`events`、`peak_offense` 均逐局完全一致；说明前 20k 只训 critic 时策略行为确实冻结。60k 统计来源 `artifacts/research/bc_criticwarm_v1/output/evaluations/update_000006.json.gz` 对 v3 100k `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，n=256：greedy 推进 94.883% vs 95.469%，差 −0.586pp（CI [−1.991,+0.819]，p=.414），胜 187 vs 217（McNemar p=5.21×10⁻⁵）；sampled 94.271% vs 95.052%，差 −0.781pp（CI [−2.599,+1.037]，p=.400），胜 180 vs 209（p=.000422）。此处主指标不显著而辅助胜率显著下降。

criticwarm 100k 统计来源 `artifacts/research/bc_criticwarm_v1/output/evaluations/update_000012.json.gz` 对同一 v3 100k 基线，n=256。Greedy 推进均值 52.695% vs 95.469%，差 −42.773pp（95% CI [−45.873,−39.673]，p=1.05×10⁻¹⁶²），胜 19/256 vs 217/256（候选单胜/基线单胜 2/200，McNemar p=6.38×10⁻⁵⁷）；sampled 50.872% vs 95.052%，差 −44.180pp（CI [−47.293,−41.066]，p=5.50×10⁻¹⁷²），胜 17 vs 209（discordant 3/195，p=6.44×10⁻⁵⁴）。两模式连续主指标及胜率均在多重校正后显著下降。结论仅针对单个初始化/采样种子；不同于评估 seeds 的配对统计，尚不能估计训练种子间方差。

100k 行为描述来源同一 candidate raw 与 v3 100k raw 的 `action_counts`、`peak_offense`、`events.*`：greedy 每局 plant/wait/shovel 平均 57.58/590.20/1.92，对照 54.40/887.32/0；peak_offense 12.89 vs 19.46，zombies_killed 44.70 vs 144.14，plants_eaten 31.61 vs 22.91，sun_spent 3,782 vs 4,885。该行为差用于描述策略退化，不另作机制因果检验。

B9 50k 配对统计来源：`artifacts/research/bc_seq32_v1/output/evaluations/update_000004.json.gz` 对 v3 100k `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，256 个相同 seed。连续主指标取 `seed_results.<mode>.validation_full_level7_v1_d1[].{terminal_wave,wave_count}` 的推进比例，配对差 95% CI 与 p 用正态近似；胜率取 `won`，用精确 McNemar。Greedy 推进均值 95.287% vs 95.469%，discordant wins 14/14；sampled 94.766% vs 95.052%，discordant 16/15。

B9 100k 配对统计来源：`artifacts/research/bc_seq32_v1/output/evaluations/update_000008.json.gz` 对 v3 100k `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，256 个相同 seed。主指标字段 `seed_results.<mode>.validation_full_level7_v1_d1[].{terminal_wave,wave_count}`，按 seed 配对；差值 CI/p 用正态近似，胜负字段 `won` 用精确 McNemar。Greedy 均值 81.107% vs 95.469%，胜 75 vs 217（discordant 8/150）；sampled 79.102% vs 95.052%，胜 69 vs 209（discordant 15/155）。

B10 50k 配对统计来源：`artifacts/research/bc_val2_v1/output/evaluations/update_000004.json.gz` 对 v3 100k `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，256 个相同 seed；连续主指标字段 `seed_results.<mode>.validation_full_level7_v1_d1[].{seed,terminal_wave,wave_count}`，按 seed 配对，差值 CI/p 使用正态近似。Greedy 推进均值 95.365% vs 95.469%，候选胜 225/256 vs 217/256（discordant 18/10，McNemar p=.185）；sampled 推进均值 96.042% vs 95.052%，候选胜 224/256 vs 209/256（discordant 28/13，McNemar p=.0275；该胜负次指标为未校正探索结果）。主指标两模式均不显著。
B10 100k 配对统计来源：`artifacts/research/bc_val2_v1/output/evaluations/update_000009.json.gz` 对 v3 100k `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，256 个相同 seed；字段 `seed_results.<mode>.validation_full_level7_v1_d1[].{seed,terminal_wave,wave_count,won}`。Greedy 推进均值 86.094% vs 95.469%，配对差 −9.375 点（95% CI [−12.296,−6.454]，p=3.17×10⁻¹⁰），胜 144/256 vs217/256（McNemar p=1.87×10⁻¹³）；sampled 推进均值 86.862% vs95.052%，差 −8.190 点（CI [−11.213,−5.167]，p=1.09×10⁻⁷），胜 152/256 vs209/256（p=1.28×10⁻⁷）。按 seed 配对，主指标差值 CI/p 为正态近似，胜率 p 为精确 McNemar；两模式下降经多重检验校正仍显著。
B6 100k 配对统计来源：`artifacts/research/bc_ep1_v1/output/evaluations/update_000010.json.gz` 对 `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，按 seed 配对；连续主指标字段 `seed_results.<mode>.validation_full_level7_v1_d1[].{terminal_wave,wave_count}`；胜率字段 `won`。95% CI 与双侧 p 使用配对差的正态近似；胜率差用精确 McNemar。

B6 50k 配对统计来源：`artifacts/research/bc_ep1_v1/output/evaluations/update_000005.json.gz` 对 `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，按 seed 配对；主指标字段 `seed_results.<mode>.validation_full_level7_v1_d1[].{terminal_wave,wave_count}`。95% CI 与双侧 p 使用配对差的正态近似。

B7 50k 配对统计来源：`artifacts/research/bc_clip05_v1/output/evaluations/update_000005.json.gz` 对 `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，按 256 个 seed 配对；主指标字段 `seed_results.<mode>.validation_full_level7_v1_d1[].{terminal_wave,wave_count}`、胜率字段 `won`。主指标双侧 p 与 95% CI 用配对差正态近似，胜率用精确 McNemar；greedy 215/256 vs 217/256（discordant 17/19，p=.868），sampled 208/256 vs 209/256（17/18，p=1.000）。

B7 100k 配对统计来源：`artifacts/research/bc_clip05_v1/output/evaluations/update_000010.json.gz` 对 `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，按 256 个 seed 配对；主指标字段 `seed_results.<mode>.validation_full_level7_v1_d1[].{terminal_wave,wave_count}`、胜率字段 `won`。greedy 胜率 213/256 vs 217/256（discordant 17/21，McNemar p=.627）；sampled 209/256 vs 209/256（23/23，p=1.000）。配对主指标 p 和 CI 用正态近似。

B5 50k 配对统计来源：`artifacts/research/bc_ent_v1/output/evaluations/update_000005.json.gz` 对 `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，按 `seed` 配对；主指标字段 `seed_results.<mode>.validation_full_level7_v1_d1[].{terminal_wave,wave_count}`，胜率字段 `won`。

B8 50k 配对统计来源：`artifacts/research/bc_roll64_v1/output/evaluations/update_000002.json.gz` 对 `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，按 256 个相同 seed 配对；主指标 `terminal_wave / wave_count`，胜负字段 `won`。主指标 p/CI 用配对差正态近似，胜率 p 用精确 McNemar。

B8 100k 配对统计来源：`artifacts/research/bc_roll64_v1/output/evaluations/update_000004.json.gz` 对 `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`，按 256 个相同 seed 配对；字段 `terminal_wave / wave_count`、`won`。主指标 p/CI 用配对差正态近似，胜率 p 用精确 McNemar。

B8 400k 最终评估的探索性配对结果：`artifacts/research/bc_roll64_v1/output/evaluations/update_000011.json.gz` 对 v3 100k `update_000008.json.gz`，同 256 seeds。greedy 推进均值 93.438% vs 95.469%，差 −2.031pp（95% CI [−3.801,−0.261]，未校正 p=.0245）；sampled 93.880% vs 95.052%，差 −1.172pp（CI [−3.286,+0.942]，p=.277）。对 B8 50k/100k/400k × greedy/sampled 六个主指标检验作 Holm 校正后，400k greedy p=.098，sampled p=.555；两者均不显著。胜负检验也均不显著（greedy 210/256 vs 217/256、McNemar p=.349；sampled 212/256 vs 209/256、p=.795）。这是程序在预算终点自动追加的评估节点，作为探索性结果报告。字段为 `seed_results.<mode>.validation_full_level7_v1_d1[].{seed,terminal_wave,wave_count,won}`；统计按 seed 配对。

队列由 `/tmp/night_queue.sh` 驱动，每个 run 独立日志 `/tmp/night_<run>.log`，进度 `/tmp/night_progress.txt`；配置静态校验通过后才启动。A1 判断连续指标足够敏感，因此 B1–B3 保持 100k 预算；A2 的不变 seed 比例为 37.9%，B8 不提前。B1 实际配置将 `initialization_seed` 与 `sampling_seed` 从 v3 的 0/1701 改成 1/1（sampling 源值与任务文字里的“0→1”不一致，已如实记录）；其它实验遵守单字段改动。磁盘每轮启动前复查，低于 30GB 会停止。

## §C 总结与下一步

已证实或得到有力支持：Φ 的折扣 shaping 在数值精度内严格望远镜相消；critic 参数来自未受 value target 训练的初始化权重；v3 1M 的关键节点提升在 1,024 seeds 与六项 Holm 校正后仍成立。B9 的 sequence_length=32、B10 的 value_coefficient=2.0 在 100k 均显著退化；criticwarm 的 100k 单种子 run 更严重退化。A/B 的 1,024 局胜率下降显著，但连续主指标经两模式校正后不显著；必须把这两个终点分开陈述。B6、B7 无主指标差异；B8 50k 显著下降，100k/400k 不显著。B1/B3/B4/B5/B2 的失败没有可解释的训练效果结论。

谨慎边界：1024 个 evaluation seeds 只量清同一个训练 checkpoint 的评估误差，不能代替多个训练 seeds。criticwarm 的冻结期验证通过，但单次解冻后崩塌不能单独证明“critic warm-up 导致”退化；训练种子间方差尚未测。A1 的连续主指标总体有更窄的均值置信区间，但 criticwarm 60k 与 AB 1024 都显示胜率有时比推进均值更敏感，应继续并列报告胜负终点，避免把一个指标当成所有效果的替代。

下一步优先级：

1. 用同一初始化 seed 做 criticwarm 与未 warm 的成对复现，至少多个训练 seeds；同时保留 20k actor-freeze 的逐局一致性检查，确认 100k 崩塌是否可重复。
2. 先补齐 B1/B3/B4/B5 的运行故障（CPU worker CUDA `current_stream` 路径与 B2 显存峰值），再运行它们的单变量实验；失败 run 不作为参数优劣证据。
3. 对 B9/B10 的显著退化做不同训练 seeds 复现，判断是否稳定；在复现前不扩展网格调参。
4. 以后保留连续推进为主指标，同时报告配对胜负 McNemar；256 局只适合较粗信号，关键结论按 1,024 seeds 或功效计算扩样。

## §D 源码修复（单列）

B1/B3/B4 的 CPU rollout worker 首次 torch.compile 调用 torch.accelerator.current_stream() 时收到 CUDA busy/unavailable。第一处修复在 _init_worker 中仅对 CPU worker 调用 set_relation_bias_fusion(False)；B5 因此越过 rollout 并训练到 100k、完成 50k 评估。B5 100k 评估仍在 _init_evaluation_worker 的同类 compile/current_stream 路径失败。第二处已在 _init_evaluation_worker 加入同样的 CPU-only eager 设置，CUDA learner 未改。pvz_agent_model.py 说明 eager/fused 关系偏置逐元素 bit-identical；CPU worker 可能较慢。

原始源码 SHA256 654397fe6660e8901002ef091d3c2d789f0bf6b13d070c326a9b41eecddec974 备份在 /tmp/night_code_change_backup/train_pvz_ppo_task_family.py.orig；第二处修复前版本（含 rollout 修复）备份在 /tmp/night_code_change_backup/train_pvz_ppo_task_family.py.pre_eval_fix，SHA256 e77954ae8aa6040de6e466792ee5c8753906efcffe2d9f1036dae8b8e99ebcea。完整回滚命令：cp /tmp/night_code_change_backup/train_pvz_ppo_task_family.py.orig /home/newbiexvwu/PvZ-Portable/python/train_pvz_ppo_task_family.py；仅回滚 evaluation 修复：cp /tmp/night_code_change_backup/train_pvz_ppo_task_family.py.pre_eval_fix /home/newbiexvwu/PvZ-Portable/python/train_pvz_ppo_task_family.py。回滚与修改说明在 /tmp/night_code_change_backup/README.txt。B6 的 512 个初始 evaluation episode 与 16 局 rollout 均完成且未复现该错误；未运行测试。

criticwarm temporary runner：首次在 18:56:39 因 `pvz_research.DEFAULT_RESOURCE_DIR` 不存在而 3 秒退出；改为 `train_pvz_ppo_task_family.DEFAULT_RESOURCE_DIR`。第二次在 multiprocessing spawn 时重跑模块顶层并二次调用 `run_experiment`，撞执行锁；本次无训练更新，`training_state.json` 为 0 decisions，`resume.json` 指向的 initial checkpoint SHA 校验通过。给调用与验证段加 `if __name__ == "__main__"`，retry wrapper 检测现存 state 并传 `--resume`；18:59:58 续跑进入初始化评估。脚本原版备份 `/tmp/night_code_change_backup/night_criticwarm.py.pre_resource_fix`（SHA256 `cc671bbce43aeae864a76ac2319eeedc741df8f58bd1294921aa33cda21d541d`），仅改资源路径的版本备份 `/tmp/night_code_change_backup/night_criticwarm.py.pre_main_guard`（SHA256 `d1387c7f85e8bcc3732a392c8ba610f26b6d2bf98fda645250e418f15978dc41`）；当前 runner SHA256 `5eb064d5eb1170aea9fbcc2d2bfe3acb3b62fa4077b06afb4d50e938867366de`。retry wrapper 改动前备份 `/tmp/night_code_change_backup/night_criticwarm_retry.sh.pre_resume`。回滚 runner：`cp /tmp/night_code_change_backup/night_criticwarm.py.pre_resource_fix /tmp/night_criticwarm.py`；回滚 retry wrapper：`cp /tmp/night_code_change_backup/night_criticwarm_retry.sh.pre_resume /tmp/night_criticwarm_retry.sh`。没有修改仓库源码来修这两个临时 runner 问题。


A9 临时 evaluator（只在 `/tmp`，未改仓库源码）：首次汇总 1M 时发现 `profile.summarize_episodes` 遍历全部 wait 事件过慢，改成只计算 A9 需要的 wins 与 terminal_wave 均值；raw 逐局数据未改。为给 GPU 队列留出 CPU，workers 从 12 调到 8，criticwarm 完成后 sampled 阶段调到 16。AB 100k 复用旧 raw 中同 checkpoint、同 seeds 30000–30255 的 256 局，新增跑 768 局；最后逐字段验证前 256 局完全一致。原脚本备份 `/tmp/night_eval1024_cached.py.pre_fast_summary` SHA256 `2e89e10c5371421f7d48731a935593e2a918eb3342135d1740bab46cc1ddb65b`；AB reuse 修改前备份 `/tmp/night_eval1024_cached.py.pre_ab_reuse` SHA256 `ceb1b9a46e4a59dd668c85eabcab11956d64068fefa915321d1c1aaf778abcd6`；当前脚本 SHA256 `21fe78e165459565930e0634c722e4928ecc9c50f4a4cccae4f1652ff661d64a`。回滚到最初版本：`cp /tmp/night_eval1024_cached.py.pre_fast_summary /tmp/night_eval1024_cached.py`；只撤销 AB reuse：`cp /tmp/night_eval1024_cached.py.pre_ab_reuse /tmp/night_eval1024_cached.py`。中断日志里的 KeyboardInterrupt 是为安全调并行度后按缓存续跑，不影响最终 raw。
