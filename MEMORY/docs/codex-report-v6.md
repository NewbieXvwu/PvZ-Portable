# 通宵任务 v6：熵守卫与崩溃率

更新：2026-10-13（Asia/Shanghai）。本报告按阶段追加；原始源码、配置与结果路径均列于各节。

## §A1 全部可用 run 的熵复核与分类器

独立逐个读取 `artifacts/research/bc_*/output/training_state.json`，对 `update_history[*].losses.entropy` 重算首值、末值、最大值和中位数。长度与预算取同一文件的 `update_history` / `counters.decisions`。`bc_eval_1024_v1` 只有评估文件，没有 `training_state.json`，因此熵不可得；`bc_roll64_v1` 和 `bc_seq32_v1` 均已纳入。表中“100k末熵”只取精确 100,000 决策节点或之前最后一个更新，供同预算分类；run 终点熵仍按其完整可用轨迹报告。

| run（state 文件） | updates / 决策数 | 熵首 | 熵末 | 熵最大 | 熵中位 | 100k末熵 | 100k 对 v3 分类 |
|---|---:|---:|---:|---:|---:|---:|---|
| `bc_clip05_v1` | 10 / 100,000 | 0.004874 | 0.001136 | 0.007952 | 0.003820 | 0.001136 | 不显著 |
| `bc_criticwarm_v1` | 12 / 100,000 | 0.004875 | 0.156473 | 0.333173 | 0.010504 | 0.156473 | 显著更差（两模式） |
| `bc_ent_v1` | 10 / 100,000 | 0.005760 | 0.029714 | 0.070232 | 0.011974 | 0.029714 | 无 100k 评估；50k 结果不作为同预算分类 |
| `bc_ep1_v1` | 10 / 100,000 | 0.004541 | 0.000126 | 0.008977 | 0.003166 | 0.000126 | 不显著 |
| `bc_finetune_level7_v1/output` | 3 / 50,000 | 0.035211 | 0.085856 | 0.085856 | 0.073871 | — | 无 100k 节点 |
| `bc_finetune_level7_v2/output`（旧 seed0 run） | 28 / 307,668 | 0.074658 | 0.022406 | 0.146784 | 0.058349 | 0.016666 | 显著更差（两模式） |
| `bc_finetune_level7_v3/output`（v3 对照，非候选） | 84 / 1,000,000 | 0.004703 | 0.006446 | 0.085099 | 0.009500 | 0.002496 | 基准 |
| `bc_finetune_level7_v3/output_workers2_partial2` | 5 / 75,106 | 0.004703 | 0.003269 | 0.004703 | 0.003328 | — | 未到 100k |
| `bc_gae95_v1` | 9 / 100,000 | 0.004408 | 0.017117 | 0.017418 | 0.007308 | 0.017117 | 显著更差（两模式） |
| `bc_kl001_night2` | 25 / 300,000 | 0.004385 | 0.019593 | 0.063513 | 0.008984 | 0.000428 | 不显著（100k） |
| `bc_kl_ab_v1` | 8 / 100,000 | 0.004703 | 0.000219 | 0.012992 | 0.007803 | 0.000219 | 不显著 |
| `bc_lr10_night2` | 0 / 0 | — | — | — | — | — | 仅初始评估，无训练更新 |
| `bc_lr10_v1/output`（C1早期前缀） | 1 / 14,786 | 0.005161 | 0.005161 | 0.005161 | 0.005161 | — | 未到 100k |
| `bc_lr10_v1/output_300k`（C1延长输出，256k timeout） | 20 / 256,226 | 0.005161 | 0.006743 | 0.007424 | 0.004519 | 0.000524 | 不显著（100k）；300k 未到 |
| `bc_mb16_v1` | 0 / 0 | — | — | — | — | — | 仅初始评估，无训练更新 |
| `bc_mb8_v1` | 0 / 0 | — | — | — | — | — | OOM，未提交更新 |
| `bc_roll64_v1` | 11 / 400,000 | 0.005373 | 0.059350 | 0.059350 | 0.006934 | 0.010850 | 100k 不显著；400k 熵不能冒充 100k 判据 |
| `bc_seed1_v1` | 9 / 100,000 | 0.006788 | 0.113177 | 0.113177 | 0.021840 | 0.113177 | 显著更差（两模式） |
| `bc_seq32_v1` | 8 / 100,000 | 0.003642 | 0.025451 | 0.025451 | 0.002596 | 0.025451 | 显著更差（两模式） |
| `bc_val2_v1` | 9 / 100,000 | 0.004451 | 0.017938 | 0.247687 | 0.006359 | 0.017938 | 显著更差（两模式） |
| `bc_eval_1024_v1` | 无 state | — | — | — | — | — | 只有评估，不可算训练熵 |

配对分类只纳入具备精确 100k 节点、同一 256 个 validation seed、greedy 与 sampled 结果完整的 12 个训练候选（排除 v3 自身；含 C1 的 `output_300k`）。每模式与 v3 100k 做双侧精确 McNemar；对 24 个 run×mode p 值统一 Holm 校正。run 标签定义为：至少一种模式 Holm 后显著更差；这 6 个负向 run 实际上两种模式都显著更差，其余 6 个都不显著。比较路径为各候选 `artifacts/research/<run>/output/evaluations/update_*.json.gz`（以 `counters.decisions=100000` 定位）和 v3 的 `.../bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`；结局字段为 `seed_results[mode].validation_full_level7_v1_d1[*].won`。精确 paired p 值与本轮 Holm 结果写入 `/tmp/n6_A1_entropy_results.json`。

在这个回顾性小样本中，用**100k 节点末次熵 ≥ 0.015**识别“显著变差”，混淆矩阵为 TP=6、FP=0、TN=6、FN=0，样本内准确率 12/12（100%）。相同 0.015 阈值下，中位熵为 TP=2、FP=0、TN=6、FN=4（8/12）；最大熵为 TP=6、FP=1、TN=5、FN=0（11/12）；连续至少 2 次超阈为 TP=4、FP=0、TN=6、FN=2（10/12）。因此把**末次熵 0.015**作为下一轮的工作预警阈值；这不是独立验证的准确率，阈值是在同一批历史结果上选择/检查，不能外推成崩溃概率。A2 仍按预先指定的 0.02 实验，避免事后改动主实验剂量。

复核源字段：每行熵来自各自 `training_state.json:update_history[*].losses.entropy`；100k末值由 `update_history` 项中的 `counters.decisions` 截断到 100,000；通关率和 run 分类来自对应节点 gzip 的 `seed_results`，基线为 v3 100k 文件。初始仅评估 run 没有训练熵，未用零值填充。

## §A4 熵尖峰与评估下降的时序

把“熵尖峰”操作化为 `losses.entropy > 0.015`，成绩下降按评估节点相对前一个已评估节点的 pass rate 下降记录；评估间隔较大时只能判断先后到节点级，不能把事件定位到单个 optimizer step。

| run | 首次熵越阈（state update / decisions） | 首次评估下降 | 时序判断 |
|---|---|---|---|
| C2 `bc_seed1_v1` | update 2 / 32,429，H=0.057423（提示中的零基 upd1） | update 4 / 50k：greedy 84.38→72.66%，sampled 85.16→71.48% | 熵尖峰先于首次评估下降；支持 H1，但中间没有评估，不能定位损害发生在哪一步 |
| B9 `bc_seq32_v1` | update 8 / 100k，H=0.025451 | greedy 在 100k 首次从 84.77→29.30%；sampled 已在 50k 从 85.16→82.03%，100k 再降至 26.95% | sampled 先降；greedy 与熵尖峰同一评估节点。不能主张熵普遍先于成绩 |
| B10 `bc_val2_v1` | update 6 / 74,131，H=0.247687 | update 9 / 100k：greedy 87.89→56.25%，sampled 87.50→59.38% | 熵越阈先于 100k 下滑；50k 评估仍高于起点 |
| `bc_criticwarm_v1` | update 8 / 83,942，H=0.284820 | update 6 / 60k：greedy 84.38→73.05%，sampled 85.16→70.31% | 成绩下降先于熵尖峰，削弱“熵是普遍先因” |
| B8 `bc_roll64_v1` | update 2 / 50k，H=0.023969 | 同一 update 2 / 50k：greedy 84.38→77.73%，sampled 85.16→71.09% | 同节点；随后 100k 回升到 86.33%/87.11%，是明显的暂时尖峰/暂时下降例 |
| B5 `bc_ent_v1` | update 6 / 63,946，H=0.015183 | update 5 / 50k：greedy 降至 67.58%，sampled 72.66% | 评估下降先于首次熵越阈；50k 对照检验按既有记录不显著 |

结论：六条轨迹的时序并不统一。C2、B10 中熵越阈先于后续评估下降；criticwarm、B5 中成绩先降；B8 同节点后又恢复。熵尖峰可能是部分训练失稳的先行信号，但仅凭历史序列不能钉死因果。

C2 前三次训练更新的数值独立读取 `artifacts/research/bc_seed1_v1/output/training_state.json:update_history[0:3].losses`（这里用人类可读零基 upd0/1/2，对应 state update 1/2/3）：

| 零基 upd（state update） | decisions / rollout wins | entropy | advantage_std | gradient_norm | clip_fraction | max_log_prob_change | optimizer_steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0（1） | 16,958 / 15 | 0.006788 | 0.410488 | 0.312892 | 0.003665 | 5.147358 | 127 |
| 1（2） | 32,429 / 13 | 0.057423 | 0.650184 | 2.753876 | 0.018229 | 3.790772 | 2 |
| 2（3） | 47,848 / 14 | 0.023776 | 0.543751 | 2.746456 | 0.005859 | 6.234941 | 1 |

首次熵尖峰所在的第二次训练更新，`gradient_norm` 约为前一更新的 8.8 倍、`clip_fraction` 约为 5 倍，`advantage_std` 也升高；但 `max_log_prob_change` 并非三次中最大。之后 KL 提前停止使更新步数骤减。该异常与熵尖峰同时出现，仍不足以区分共同梯度噪声与熵的因果作用。

## §A2 熵守卫实现与身份影响（G1 待跑）

### 实现记录

改前备份在 /tmp/n6_train_pvz_ppo.py.bak 和 /tmp/n6_pvz_research.py.bak。守卫逻辑位于 python/train_pvz_ppo.py：每个 optimizer.step() 后，在刚更新策略的同一 minibatch 上重算熵；超过 ppo.entropy_guard.max_entropy 时，恢复本次 train_update 开始时的模型参数与 AdamW 状态，并停止该次 PPO 更新。状态记录在本次 losses.entropy_guard，包括是否触发、观测熵、阈值、尝试步数和 rollback_scope=whole_train_update。没有该配置项时不做快照、不额外计算熵，默认关闭。

还在 python/pvz_research.py 对严格 PPO 字段校验中允许并验证 entropy_guard，并把配置传给 train_update；这两处是同一守卫功能的配置接线。改动前原校验会拒绝新增字段。py_compile 通过；对 G1、B1–B3 和 D 的配置逐份调用 pvz_research.load_config() 均通过。尚未用 CUDA 运行专门的微型训练验证；实际触发/回滚行为以 G1 的训练状态记录为准。

**身份影响：会改变配置指纹与续跑身份。** experiment_identity_digest() 纳入整个配置和源文件指纹，pvz_research.py 的 source_paths 包含 python/train_pvz_ppo.py 与 python/pvz_research.py；G1 的 JSON 也新增了 ppo.entropy_guard。因此本轮使用新的 bc_seed1_guard002_v1 目录从初始化权重启动，不续跑 bc_seed1_v1；旧 run 在当前源码下身份不匹配，不能直接续跑。其他旧配置虽仍默认关闭，但源码 hash 已改变，续跑身份同样改变。

回滚源码：cp -p /tmp/n6_train_pvz_ppo.py.bak python/train_pvz_ppo.py 与 cp -p /tmp/n6_pvz_research.py.bak python/pvz_research.py。这会移除守卫实现/配置接线；实验结果与报告继续保留。

### 预设判据

- **G1 显著优于 C2（seed1，无守卫）**：按任务预设，支持熵越阈对应的危险更新对崩溃有因果作用、阈值回滚有效。解释边界是“越阈后回滚整次更新”这一干预有效；它不能单独证明熵数值本身是唯一中介，因为同一回滚也减少了本次学习更新。
- **G1 与 C2 不显著**：守卫在 0.02 下未显示效果；可能阈值/测量窗口不合适，也可能熵主要是症状。写“不显著”，不据此否定所有阈值。
- **G2 显著优于 v3 seed0**：守卫在好种子上也有增益；若 G2 未跑则明确写“未估计”。
- **G2 显著差于 v3 seed0**：守卫妨碍正常学习；结论指向阈值限制过强，应按 A3 已列阈值扫描解释，不外推成熵假设整体无效。

G1 使用 experiments/t7/bc_seed1_guard002_v1/seed1.json，初始化/采样种子均为 1，训练预算 100k，评估节点 50k/100k；唯一策略干预是 ppo.entropy_guard={"enabled":true,"max_entropy":0.02}。输出隔离在 artifacts/research/bc_seed1_guard002_v1/output/。参考 C2 为 artifacts/research/bc_seed1_v1/output/evaluations/update_000009.json.gz；v3 100k 为 artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz。G1 100k 按 256 个相同 validation seed 做 greedy/sampled 精确 McNemar，并对两个 G1 模式统一 Holm 校正。结果待 G1 完成。


### G1 运行进度（首次更新）

截至 2026-10-13 20:51（本地时区），G1 的首次 16,958 决策 PPO 更新已触发守卫：`artifacts/research/bc_seed1_guard002_v1/output/training_state.json:update_history[0].losses.entropy_guard` 记录 `triggered=true`、`max_entropy=0.02`、`post_step_entropy=0.0934735015`、`attempted_optimizer_steps=61`、`rollback_scope=whole_train_update`。这一更新的常规聚合 `losses.entropy=0.00519408` 是回滚后的记录值，不能据此误判为未触发。初始策略验证为 greedy 216/256、sampled 218/256；本轮 rollout 15/16 胜（训练日志 `update 1` 行）。G1 仍在继续到 100k；目前只确认守卫实际触发并执行了整次更新回滚，尚无结果性因果判定。

G1 随后在第二次训练更新也触发守卫：`update_history[1].losses.entropy_guard` 为 `triggered=true`、`post_step_entropy=0.0262795109`、`attempted_optimizer_steps=14`；回滚后的聚合熵为 `0.0046266802`。训练日志 `update 2` 累计到 31,037 决策、rollout 13/16 胜。当前只记录到回滚后聚合熵，尚无回滚前完整更新的聚合熵，不能比较 minibatch 峰值与该更新总体熵的差距。

第三次更新同样触发：`update_history[2].losses.entropy_guard` 的 `post_step_entropy=0.0243129767`、`attempted_optimizer_steps=55`；回滚后记录的聚合熵为 `0.0035884906`。更新日志累计 45,345 决策，rollout 14/16 胜。前三次更新均触发整次回滚；50k 评估仍待完成。

第四次更新在跨 50k 前触发：`update_history[3].losses.entropy_guard` 记录 `post_step_entropy=0.0205899328`、`attempted_optimizer_steps=26`，回滚后聚合熵 `0.0060883350`。累计更新日志为 49,500 决策，随后队列进入只收集 1 个 rollout episode 的 update 5，正在跨过 50k 评估门槛。

跨到 50k 的 update 5（1 个 episode，累计 50,000 决策）没有触发守卫：`update_history[4].losses.entropy_guard` 为 `triggered=false`、熵 `0.0075069414`、8 个 optimizer steps。随后 `training_state.json.phase=pending_evaluation`，50k 验证开始；结果尚未写入。

### G1 50k 补充配对比较（非 A2 预设终点）

从同一 256 validation seed 逐 seed 重算，G1 50k（`artifacts/research/bc_seed1_guard002_v1/output/evaluations/update_000005.json.gz`）对 C2 50k（`artifacts/research/bc_seed1_v1/output/evaluations/update_000004.json.gz`），双侧精确 McNemar，并在 greedy/sampled 两项间 Holm：greedy 218/256 vs 186/256，+12.50pp，C2赢/G1输 6、C2输/G1赢 38，p=9.4304e-7，Holm p=1.8861e-6；sampled 214/256 vs 183/256，+12.11pp，discordant 9/40，p=9.2635e-6，Holm p=9.2635e-6。两模式在 50k 均显著更好。计算明细 `/tmp/n6_A2_50k_supplement.json`。这是中点补充结果；预设 A2 的主要因果判定仍以 100k 节点和统一 Holm 家族为准。

50k 后的 update 6（累计 65,187 决策，rollout 15/16 胜）再次触发：`update_history[5].losses.entropy_guard` 的越阈熵为 `0.0289715044`、尝试 17 步后回滚；回滚后聚合熵 `0.0050275881`。G1 继续向 100k 训练。

update 7 累计 81,579 决策、rollout 胜 12/16；守卫再次触发，越阈熵 `0.0246719085`、尝试 27 步，回滚后聚合熵 `0.0051227837`。

update 8 累计 96,005 决策、rollout 胜 10/16；守卫触发于 `post_step_entropy=0.0202242099`、尝试 25 步，回滚后聚合熵 `0.0059291334`。下一次 update 9 从 96k 附近收集到 100k 节点。

update 9 在 99,779 决策、rollout 0/5 胜后触发守卫（越阈熵 `0.0240467135`，6 步后回滚；回滚聚合熵 `0.0060185324`）。update 10 收集 1 个 episode 到整 100,000 决策，未触发（熵 `0.0071170982`，4 步；rollout 0/1 胜），随后 `training_state.json.phase=pending_evaluation`，100k 验证正在运行。


### 队列执行记录：bc_seed1_guard002_v1

时间：2026-10-13T22:22:10.384589+08:00；状态：complete；exit=0；耗时=8107 秒。详细进度见 /tmp/n6_progress.txt，训练日志为 /tmp/n6_<run>.log 或对应干预日志。

### G1 100k 原始评估率（配对统计待统一 Holm）

100k 验证已完成 512/512。G1 `update_000010.json.gz`：greedy 216/256=84.38%，sampled 216/256=84.38%；C2 无守卫 `update_000009.json.gz`：greedy 130/256=50.78%，sampled 100/256=39.06%。原始差值分别 +33.59pp、+45.31pp。精确 McNemar 的 discordant counts 与 A2 统一 Holm 校正将在 G2 完成后由 `/tmp/n6_compare.py` 一次生成，避免把临时单臂校正值当成最终家族结果。

G2 update 3 累计 42,956 决策、rollout 胜 16/16；仍触发守卫，`post_step_entropy=0.0348555669`、尝试 60 步后回滚，回滚后聚合熵 `0.0039101405`。update 4 开始收集至 50k。

G2 的 update 4 未触发守卫（累计 49,993 决策、rollout 1/8 胜，批次熵 `0.0105859861`、56 步）；update 5 收集 1 个 episode 到 50,000 决策也未触发（熵 `0.0008319043`、2 步）。当前 phase=`pending_evaluation`，等待 50k 配对验证。

G2 50k 验证完成 512/512：greedy 215/256=83.98%，sampled 211/256=82.42%（文件 `artifacts/research/bc_seed0_guard002_v1/output/evaluations/update_000005.json.gz`）。训练已恢复至 update 6，最终 seed0 守卫影响仍待 100k 对照 v3。

G2 的 update 6（50k 后继续训练）触发守卫：`training_state.json:update_history[5].losses.entropy_guard` 记录越阈熵 `0.0212427974`、尝试 14 步后整次更新回滚；回滚后的聚合熵待该更新结束状态稳定后记录。当前已进入 update 7，G2 仍在训练至 100k。

G2 update 7 完成并进入 update 8。update 7 rollout 胜 12/16、累计 78,438 决策；守卫在 `post_step_entropy=0.0211521052`、9 个优化步后触发整次回滚，回滚后聚合熵 `0.0062026184`（`training_state.json:update_history[6].losses`）。

G2 update 8 完成：rollout 胜 14/16、累计 93,968 决策。守卫在熵 `0.0205773022`、18 个优化步后触发整次回滚，更新聚合熵 `0.0044351344`；随即 update 9 收集 7 个 episode 补到 100k（字段见 `bc_seed0_guard002_v1/output/training_state.json:update_history[7].losses`）。

G2 update 9 已到 100k 节点：本次补训 7 个 episode 为 0/7 胜，`training_state.json.phase=pending_evaluation`，100k 验证正在运行；守卫字段与最终 100k 指标待评估结束后一并记录。

补充核对 G2 update 9：熵守卫未触发，`post_step_entropy=0.0010335406`、43 个优化步，更新聚合熵 `0.0037607518`（`training_state.json:update_history[8].losses`）。

### 守卫代码位置（供审阅与回滚）

截至当前工作树行号：`python/train_pvz_ppo.py:223-250` 是更新后当前 minibatch 熵的重算 helper；`:253-278` 增加可选参数及校验；`:325-334` 保存更新前模型和优化器状态；`:421-444` 在 `optimizer.step()` 后测熵，越阈即恢复并结束更新；`:471-478` 写出触发元数据。`python/pvz_research.py:180-194` 接受并验证可选配置，`:692-696` 将配置传入训练函数。备份及恢复命令见上文。

工作树的 `python/train_pvz_ppo_task_family.py` 另有一处预先存在的差异，文件 mtime 为 2026-10-10；本任务未改这个文件，且它不属于熵守卫实现。没有暂存或提交任何文件。


### 队列执行记录：bc_seed0_guard002_v1

时间：2026-10-14T00:36:41.407139+08:00；状态：complete；exit=0；耗时=8047 秒。详细进度见 /tmp/n6_progress.txt，训练日志为 /tmp/n6_<run>.log 或对应干预日志。

## §A2 熵守卫因果判定：G1 / G2

精确 paired McNemar 比较共 4 项（run×mode），同一 256 个 validation seed；Holm 家族为本节所有有精确 100k 节点的 G1/G2 指定比较。证据 JSON：/tmp/n6_A2_results.json。

### G1：bc_seed1_guard002_v1 对 bc_seed1_v1

评估文件：artifacts/research/bc_seed1_guard002_v1/output/evaluations/update_000010.json.gz vs artifacts/research/bc_seed1_v1/output/evaluations/update_000009.json.gz。

| mode | baseline→run 通关 | 差值 | 不一致对（基线胜/运行败；基线败/运行胜） | exact p | Holm p |
|---|---:|---:|---:|---:|---:|
| greedy | 50.78% (130/256)→84.38% (216/256) | +33.59 pp | 19/105 | 1.30814e-15 | 3.92443e-15 |
| sampled | 39.06% (100/256)→84.38% (216/256) | +45.31 pp | 15/131 | 2.67977e-24 | 1.07191e-23 |

### G2：bc_seed0_guard002_v1 对 bc_finetune_level7_v3

评估文件：artifacts/research/bc_seed0_guard002_v1/output/evaluations/update_000009.json.gz vs artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz。

| mode | baseline→run 通关 | 差值 | 不一致对（基线胜/运行败；基线败/运行胜） | exact p | Holm p |
|---|---:|---:|---:|---:|---:|
| greedy | 84.77% (217/256)→83.98% (215/256) | -0.78 pp | 16/14 | 0.855536 | 0.855536 |
| sampled | 81.64% (209/256)→85.16% (218/256) | +3.52 pp | 16/25 | 0.211024 | 0.422047 |

### 四种预设情形判定
1. **G1 对 C2 greedy 显著更好**（+33.59 pp，Holm p=3.92443e-15），按预设判据支持 0.02 越阈回滚干预对 seed1 崩溃有因果保护作用。sampled 显著更好（差 +45.31 pp，Holm p=1.07191e-23）。严格边界：被操纵的是“阈值触发后整次 PPO 更新回滚”，不只是 entropy 数值本身。
2. **G2：G2 seed0 greedy 不显著**（greedy -0.78 pp，Holm p=0.855536；sampled +3.52 pp，Holm p=0.422047）。
3. G1 不显著的情形不证明熵只是症状，也不排除其他阈值；按预案不启动 A3。
4. 若 G2 显著更差，按预案将其解释为当前阈值可能限制好种子更新；是否调整只看已预设的 A3 扫描，不临时扩充参数。

### 四种预设结果情形逐项核对

1. **G1 显著优于 C2：发生。** greedy 与 sampled 均有正差且在 A2 四检验 Holm 家族内显著；支持 0.02 阈值回滚对 seed1 训练结果有因果保护作用（边界见上）。
2. **G1 与 C2 约等/不显著：未发生。** 当前数据不支持“守卫无效”；若出现也只能说 0.02 未显示效果，不能排除阈值或因果中介问题。
3. **G2 显著优于 v3：未发生。** greedy 与 sampled 虽方向相反但两者均不显著；没有证据称守卫提升好种子。
4. **G2 显著差于 v3：未发生。** greedy 的点估计为 -0.78pp，但 Holm p=0.855536；不能称为妨碍正常学习。

A3 阈值 0.01 的 T001 初始 512 样本验证于 01:01:23 完成，greedy=216/256、sampled=218/256（`bc_seed1_guard001_v1/output/evaluations/update_000000.json.gz`；与起点预期一致）。已进入第 1 次训练更新。

T001 update 1（16,958 决策、rollout 15/16 胜）触发 0.01 守卫：越阈熵 `0.0103002442`，11 步后整次回滚；该 update 的回滚后聚合熵以 `training_state.json:update_history[0].losses.entropy` 为准。已开始 update 2。
update 1 回滚后聚合熵为 `0.0028020675`。

T001 update 2：累计 31,037 决策、rollout 13/16 胜；守卫在熵 `0.0108463857`、9 个优化步后触发整次回滚，聚合熵 `0.0016031392`。已开始 update 3（`training_state.json:update_history[1].losses`）。

T001 update 3 完成：累计 45,345 决策、rollout 14/16 胜；0.01 守卫在熵 `0.0192639399`、13 个优化步后回滚整次更新，聚合熵 `0.0037043356`。update 4 收集 5 个 episode 补到 50k（`training_state.json:update_history[2].losses`）。

T001 update 4 在 49,500 决策时完成，rollout 2/5；熵 `0.0104945078`、5 个优化步后触发回滚，聚合熵 `0.0050355813`。update 5 收集 1 个 episode 跨过 50k 门槛。

T001 到达 50k：update 5 只收集 1 个 episode，0/1 胜；熵 `0.0119010601`、4 步后触发 0.01 守卫并回滚，50k 评估正在运行。
update 5 回滚后聚合熵为 `0.0069231689`。

T001 50k 验证完成 512/512：greedy 216/256=84.38%，sampled 218/256=85.16%（`evaluations/update_000005.json.gz`）；与起点持平。训练已恢复至 50k 后更新。

T001 update 6（50k 后）完成：累计 65,098 决策、rollout 14/16 胜；熵 `0.0170976818`、12 步后守卫回滚整次更新。聚合熵见 `training_state.json:update_history[5].losses.entropy`；已开始 update 7。
update 6 回滚后聚合熵为 `0.0036346188`。

T001 update 7 完成：累计 81,411 决策、rollout 12/16 胜；0.01 守卫在熵 `0.0111668026`、19 个优化步后整次回滚，聚合熵 `0.0031984798`（`training_state.json:update_history[6].losses`）。update 8 已开始。

T001 update 8 完成：累计 96,257 决策、rollout 11/16 胜；0.01 守卫在熵 `0.0102960095`、16 步后回滚整次更新，聚合熵 `0.0032462541`。update 9 收集 4 个 episode 补到 100k。

T001 已到 100k 训练预算，当前进入 512 样本验证。update 9（99,623 决策）与 update 10（100,000 决策）均触发 0.01 守卫，具体越阈熵与聚合熵记录在 `training_state.json:update_history[8:10].losses`；100k 评估结果待完成。

T001 100k 验证完成 512/512：greedy 216/256=84.38%，sampled 218/256=85.16%（`evaluations/update_000010.json.gz`），与初始验证持平。A3 配对显著性与 0.02、0.05 两档的统一 Holm 结果待 T005 完成后一起计算。


### 队列执行记录：bc_seed1_guard001_v1

时间：2026-10-14T02:08:26.360425+08:00；状态：complete；exit=0；耗时=5476 秒。详细进度见 /tmp/n6_progress.txt，训练日志为 /tmp/n6_<run>.log 或对应干预日志。

A3 阈值 0.05 的 T005 初始验证完成 512/512：greedy=216/256，sampled=218/256（`bc_seed1_guard005_v1/output/evaluations/update_000000.json.gz`）；与起点一致，已进入训练。

T005 update 1 完成：16,958 决策、rollout 15/16 胜；熵 `0.0934735015`、61 步后触发 0.05 守卫并回滚，聚合熵 `0.0051940799`。该首更尖峰与 G1 0.02 首更相同；update 2 已开始。

T005 update 2：累计 31,037 决策、rollout 13/16 胜；31 个优化步后未触发 0.05 守卫，聚合熵 `0.0051088292`（`training_state.json:update_history[1].losses`）。已开始 update 3。

T005 update 3 完成：累计 45,140 决策、rollout 15/16 胜；120 个优化步后未触发 0.05 守卫（测得 `post_step_entropy=0.0023702390`，聚合熵 `0.0024147501`）。update 4 收集 6 个 episode 到 50k。

T005 update 4 到 50k：补训 6 个 episode，0/6 胜；23 个优化步后未触发 0.05 守卫（测得 `post_step_entropy=0.0080474680`，聚合熵 `0.0040583042`）。50k 验证正在运行。

T005 50k 验证已完成 512/512：`evaluations/update_000004.json.gz`（文件序号对应 50k state update 4）greedy 208/256=81.25%，sampled 197/256=76.95%。训练已恢复到下一次 rollout。

T005 第 5 次更新：累计 64,564 决策、rollout 12/16 胜；3 个优化步后未触发 0.05 守卫（熵 `0.0077780015`，聚合熵 `0.0077061607`）。update 6 已开始。

T005 第 6 次更新完成：累计 79,120 决策、rollout 11/16 胜；124 个优化步后未触发 0.05 守卫（熵 `0.0139888655`，聚合熵 `0.0074937568`）。update 7 已开始。

T005 update 7 完成：累计 93,549 决策、rollout 12/16 胜；0.05 守卫在熵 `0.0688716471`、28 步后触发整次回滚，聚合熵 `0.0087085354`。update 8 收集 8 个 episode 补到 100k。

T005 已到 100k 训练预算并进入验证。update 8 累计 99,569 决策、rollout 0/8；0.05 守卫未触发，测得熵 `0.0361830629`、聚合熵 `0.0092615879`。update 9 收集 1 个 episode 补足 100k（`training_state.json:update_history[7:9].losses`）；最终验证待完成。
update 9 未触发守卫（熵 `0.0026594999`、8 步，聚合熵 `0.0017153892`）。


### 队列执行记录：bc_seed1_guard005_v1

时间：2026-10-14T05:22:21.640211+08:00；状态：complete；exit=0；耗时=11622 秒。详细进度见 /tmp/n6_progress.txt，训练日志为 /tmp/n6_<run>.log 或对应干预日志。

## §A3 阈值扫描最终结果

G1 greedy 达到 A2 预设显著性门槛后启动扫描。下表是各守卫阈值对无守卫 C2（seed=1，100k）的同 seed 配对精确 McNemar；六个“阈值×mode”检验统一 Holm 校正。主 JSON：/tmp/n6_A3_results.json。T005 的 100k 评估为 artifacts/research/bc_seed1_guard005_v1/output/evaluations/update_000009.json.gz。

| 熵阈值 | mode | C2→守卫通关 | 差值 | 不一致对（基线胜/运行败；基线败/运行胜） | exact p | Holm p |
|---:|---|---:|---:|---:|---:|---:|
| 0.01 | greedy | 130/256→216/256 | +33.59 pp | 17/103 | 3.40739e-16 | 6.81478e-16 |
| 0.01 | sampled | 100/256→218/256 | +46.09 pp | 15/133 | 8.28664e-25 | 4.97198e-24 |
| 0.02 | greedy | 130/256→216/256 | +33.59 pp | 19/105 | 1.30814e-15 | 1.30814e-15 |
| 0.02 | sampled | 100/256→216/256 | +45.31 pp | 15/131 | 2.67977e-24 | 1.33988e-23 |
| 0.05 | greedy | 130/256→221/256 | +35.55 pp | 19/110 | 9.10624e-17 | 2.73187e-16 |
| 0.05 | sampled | 100/256→208/256 | +42.19 pp | 19/127 | 8.42667e-21 | 3.37067e-20 |

三档的 greedy 与 sampled 均显著高于无守卫 C2（六项 Holm 后 p<0.05）。守卫之间另做三组阈值两两比较、两种 mode，共六项统一 Holm；结果见 /tmp/n6_A3_pairwise_results.json：全部不显著（最小 Holm p=0.728697）。0.05 的 greedy 点估计最高（221/256），0.01 的 sampled 点估计最高（218/256），现有样本不足以统计上选出唯一最优阈值。**操作推荐仍用 0.02**：它是预先指定并已在 G1 证明对 seed1 有效的阈值；不声称它显著优于另外两档。


### §B 阶段记录：bc_seed2_v1 的 50k 节点

50,000 决策配对评估已完成：greedy 172/256 (67.19%)，sampled 154/256 (60.16%)；相对起点 216/256、218/256 均回落。评估源为 `artifacts/research/bc_seed2_v1/output/evaluations/update_000005.json.gz`，字段 `seed_results[mode].validation_full_level7_v1_d1[*].won`；训练节点摘要在 `artifacts/research/bc_seed2_v1/output/training_state.json` 的 `learning_curve` 项（`counters.decisions=50000`）。截至此节点训练熵 first/last/max/median=0.00387/0.02791/0.02791/0.00975（`update_history[*].losses.entropy`，仅当前 5 个更新）；100k 尚未完成，不能据此分类崩溃。

更新 2 曾记录一次 CUDA allocator 203,423,744-byte 分配告警（free=0）；训练随后继续并提交到 30,640 决策，没有 traceback。此为可恢复资源告警，报告在 `/tmp/n6_progress.txt` 留有时间戳。


### 队列执行记录：bc_seed2_v1

时间：2026-10-14T07:25:17.554982+08:00；状态：complete；exit=0；耗时=7366 秒。详细进度见 /tmp/n6_progress.txt，训练日志为 /tmp/n6_<run>.log 或对应干预日志。


### §B 阶段记录：bc_seed2_v1 完成 100k

状态 `ready/budget_complete`，100,000 决策，10 个训练更新；队列记录 exit=0、耗时 7,366 秒。100k greedy 163/256 (63.67%)、sampled 128/256 (50.00%)；50k greedy 172/256 (67.19%)、sampled 154/256 (60.16%)。100k 评估源 `artifacts/research/bc_seed2_v1/output/evaluations/update_000010.json.gz`，50k 为 `artifacts/research/bc_seed2_v1/output/evaluations/update_000005.json.gz`；pass 字段为 `seed_results[mode].validation_full_level7_v1_d1[*].won`。

100k 熵轨迹首/末/最大/中位 = 0.00387/0.03240/0.07304/0.03015，来自 `artifacts/research/bc_seed2_v1/output/training_state.json` 的 `update_history[*].losses.entropy`。按要求，是否崩溃及与 v3 的显著性待 B 全部完成后，以统一配对精确 McNemar + Holm 判定。

该 run 的 update 2 曾出现 203,423,744-byte CUDA allocator 申请失败告警，但随后 update 2 成功提交，训练与 100k 验证均 exit=0；属于可恢复资源告警，不计作假设失败。


### §B 阶段记录：bc_seed3_v1 的 50k 节点

50,000 决策配对评估已完成：greedy 194/256 (75.78%)、sampled 158/256 (61.72%)。评估源 `artifacts/research/bc_seed3_v1/output/evaluations/update_000005.json.gz`；训练节点为 `artifacts/research/bc_seed3_v1/output/training_state.json` 的 `learning_curve` 项（`counters.decisions=50000`）。截至此节点训练熵首/末/最大/中位=0.00543/0.04179/0.04179/0.01052，来源 `update_history[*].losses.entropy`。100k 未完成，不在此节点判断崩溃。
