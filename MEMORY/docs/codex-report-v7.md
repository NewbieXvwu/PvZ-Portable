# 接续任务 v7：守卫保护下的训练机制与效果

开始：2026-10-14（Asia/Shanghai）。按 v7 任务书继续；v6 既有结果直接采信。原始证据文件、字段、配对统计 JSON 与命令日志逐阶段记录。

## §0 守卫的代价复核

### 身份与影响范围

`python/pvz_research.py:285` 的 `experiment_identity_digest()` 将完整语义配置与源码指纹一起哈希；`:458-483` 对每个实验计算包含 `python/pvz_research.py` 和 `python/train_pvz_ppo.py` 在内的来源指纹；`:492-506` 的 `--resume` 将重新计算的 identity 与检查点记录值比较，不同即拒绝续跑。v6 熵守卫改动了上述两个文件，所以旧 identity 下的检查点不能用当前源码直接续跑。

盘点 `artifacts/research/bc_*/output/provenance.json:fingerprints`（25 个 run）时，16 个 run 仍记录守卫实现之前的文件指纹 `pvz_research.py=0c7439f0…`、`train_pvz_ppo.py=6fb67bcd…`：`bc_clip05_v1`、`bc_criticwarm_v1`、`bc_ent_v1`、`bc_ep1_v1`、`bc_finetune_level7_v3`、`bc_gae95_v1`、`bc_kl001_night2`、`bc_kl_ab_v1`、`bc_lr10_night2`、`bc_lr10_v1`、`bc_mb16_v1`、`bc_mb8_v1`、`bc_roll64_v1`、`bc_seed1_v1`、`bc_seq32_v1`、`bc_val2_v1`。这包括 v3 的 1M 与 C1 `bc_lr10_v1/output_300k` 的 256,226 决策前缀。v6 后用守卫版本生成的结果有新 identity；未启用守卫的 seed2/3/4 虽然配置不含守卫字段，源码指纹仍是 guard-aware 版本。

受影响动作是**直接续跑**：当前源码重算的 identity 与这些旧检查点不符，resume 会因源码/配置 identity 校验失败。读取旧 `training_state.json`、评估 gzip、学习曲线和 provenance 不受影响；`episode_query.py --checkpoint` 直接载入权重做推理也不走训练 identity 校验。v7 不需要续跑上述旧 run：A1/B/A2 都是新实验，E1(a) 是推理；E2 从指定检查点做新的权重初始化（新 optimizer、新计数），不是 resume。因此接受该影响，不为旧 run 回滚或改写 identity。

回滚 v6 守卫源码的原命令仍为：`cp -p /tmp/n6_train_pvz_ppo.py.bak python/train_pvz_ppo.py` 与 `cp -p /tmp/n6_pvz_research.py.bak python/pvz_research.py`。这会同时移除熵守卫与配置接线，不执行。E1(b) 另有“仅熵分组日志”改动；其专用前置备份会在修改时记录，恢复它可单独撤销日志而保留 v6 守卫。源码指纹变化只影响续跑身份，实验数据读取和报告仍有效。

E1(b) 在前置备份 `/tmp/n7_train_pvz_ppo_pre_entropy_type.py.bak`（SHA256 `55f2518b5277c971b3b11f5055365da822b42d88b4bc8b6522f2ce691d60cabb`）后，仅在 `train_update()` 增加按 transition 的 `action.type` 汇总熵的日志，输出 `losses.entropy_on_plant` / `losses.entropy_on_wait`；没有改 loss、梯度、优化器、守卫或随机数调用。修改后文件 SHA256 为 `8b90cb10930b50c4fc1dfc9b99041c3f7ab92274e525d13f34120a411ea520e9`。这会再次更改 identity：对仓库现有 127 份 `artifacts/research/**/provenance.json`，当前 `train_pvz_ppo.py` 指纹均不匹配；其中 106 份有 `training_state.json`、65 份状态文件指向的 checkpoint 仍存在。受影响仍是从这些状态直接 resume，模型推理、capture、评估 JSON、状态/曲线读取不受影响。新建 v7 实验以改后源码生成 identity。若只撤销熵分组日志而保留 v6 守卫，恢复上述前置备份即可；不会回滚 `pvz_research.py` 中的守卫接线。

### 工作树与可用资源基线

开始时可用磁盘 861 GB（`df -h . /tmp "$HOME"`）；GPU 查询为空闲。当前工作树已有 v6 的守卫修改及其他未提交改动；本任务保留它们，不提交、不强制添加忽略文件。开始时 `HEAD=3c5117b13c1ff041b3f2ce70fd3efc64a8a0021e`，`git cat-file -t` 返回 `commit`。v7 新增的实验配置和报告均不代表新的 Git 提交。

## §E1(a) C2@100k 与 v3@100k 的落点错配

两模型都用 `episode_query.py capture --policy ppo --task-manifest experiments/t7/bc_finetune_level7_v1/train.json --task-id train_full_level7_v1_d1`，以相同 held-out seeds `30000..30019`、greedy 推理，各 20 局。C2 检查点为 `artifacts/research/bc_seed1_v1/output/runs/run_1/update_000009_evaluated_1791864630600560027.pt`；v3 为 `artifacts/research/bc_finetune_level7_v3/output/runs/run_1/update_000008_evaluated_1791479369559475624.pt`。两 checkpoint SHA256 分别为 C2 `8ef3db917aa4ccbf88a3dafd2e4b54665878c971435ff84c3f7ea69dc1d5b415`、v3 `a12ee9fb32774c82d8958a52f27cb4c3818637e3af56b38cd5bbfd6c194326e8`。episode `meta.json` 记录 checkpoint 路径和短 SHA。完整轨迹保存在 `/tmp/n7_E1a/{C2,v3}/seed_<seed>/{meta.json,frames.jsonl}`；指标使用 `scripts/render_episode.py::_reachable_dps`，统计与逐 seed 首次动作分叉在 `/tmp/n7_E1a_metrics.json`。

| 指标（每局先算比例/均值，再对 20 个配对 seed 求均值） | C2 | v3 | C2−v3 | 精确配对符号翻转 p | 三项 Holm p |
|---|---:|---:|---:|---:|---:|
| A：同路有僵尸时射手种在最前僵尸身后比例 | 0.7376（502/713 次） | 0.5019（280/540 次） | +0.2357 | 1.1444e−5 | 3.4332e−5 |
| B：其他路有承压裸路时，射手种在身后比例 | 0.7465（212/283 次机会） | 0.4458（92/164 次机会） | +0.3006 | 0.0034409 | 0.0034409 |
| C：第 5–15 波每帧承压裸路数 | 1.2268 | 0.6500 | +0.5769 路 | 0.0004787 | 0.0009575 |

20 个样本局的通关数是 C2 12/20、v3 18/20（仅描述，不在这组三项指标检验中）。三项错配指标在 Holm 后均显著更差，按预设判据，支持“C2 熵崩溃之后决策/落点质量下降”这一中间环节；这是一组检查点间配对机制证据，不单独识别熵作为唯一原因。

### §E1(c) 逐决策首次分叉

20/20 对局第一次动作差异都在第 4 次决策（`first_divergence.decision_histogram`）。当步双方都选择等待 60 tick，但 C2 的条件是 `left_zone_occupied`，v3 的条件是 `timeout`。这是集中且先于种植的等待决策分叉；不能把本批首次分叉描述成“种植落点分叉”。逐 seed 动作对象见 `/tmp/n7_E1a_metrics.json:first_divergence.by_seed`。

## §E1(b) 按动作类型拆分熵

实现已完成，100k run 已完整结束（exit=0，耗时 7,491 秒）。`train_update()` 对 `flat_transitions[*].action.type` 对齐的 `entropies_for_chunk` 分别累加 plant/wait 的熵并写入 `losses.entropy_on_plant`、`losses.entropy_on_wait`；这是 PPO minibatch 更新前的策略熵，聚合所有 optimizer minibatch/epoch。

前六次更新均与原 C2 轨迹完全对齐：新旧 `episode_digests` 逐项相同（16/16、16/16、16/16、3/3、16/16、16/16），`policy_loss`、`value_loss`、总 `entropy`、`advantage_mean/std`、`gradient_norm`、`approx_kl`、`clip_fraction`、`max_log_prob_change` 各字段差值均为 0，optimizer steps 也逐项一致（127、2、1、8、24、14）。结果字段在旧 C2 与新 run 的 `training_state.json:update_history[0:5]`。

| 更新 | 总 entropy | entropy_on_plant | entropy_on_wait | plant / wait / shovel 动作数 | rollout won |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.006788 | 0.044607 | 0.003543 | 879 / 16,079 / 0 | 15/16 |
| 2 | 0.057423 | 0.018987 | 0.057784 | 1,154 / 14,308 / 9 | 13/16 |
| 3 | 0.023776 | 0.051777 | 0.021012 | 1,184 / 14,234 / 1 | 14/16 |
| 4（补足 50k） | 0.004710 | 0.046466 | 0.001190 | 177 / 1,975 / 0 | 0/3 |
| 5 | 0.006103 | 0.046364 | 0.002160 | 1,238 / 14,482 / 0 | 14/16 |
| 6 | 0.021840 | 0.090674 | 0.014742 | 1,152 / 12,866 / 0 | 7/16 |
| 7 | 0.019886 | 0.165543 | 0.006068 | 1,142 / 13,027 / 0 | 6/16 |
| 8 | 0.031302 | 0.174811 | 0.015425 | 405 / 4,512 / 1 | 0/7 |
| 9（100k） | 0.113177 | 0.262384 | 0.106342 | 110 / 1,064 / 1 | 0/2 |

更新 4 的 50k 评估为 greedy 0.726562、sampled 0.714844，与无日志 C2 的 50k `learning_curve` 完全相同。最初一次明显的总熵跳升发生在更新 2：wait 条件熵从 0.00354 升到 0.05778，而 plant 条件熵降至 0.01899。按预设判据，plant-only 熵机制不成立，落点错配不能由“种植时犹豫”单独解释；更新 3 又转为 plant 熵较高，显示通道可能随阶段变化。E1(c) 的首个分叉也在 wait（两侧 wait 条件不同），因此“wait 熵只是旁观指标”仍需谨慎：wait 类动作确实可能有策略含义，这批数据不能把 wait 熵与决策质量割裂。截至 100,000 决策，9/9 个更新的 episode digests、所有原有 PPO loss/统计、optimizer steps 均与无日志 C2 完全一致（各差值 0）。最高总熵的 update 9 同时出现 plant 0.26238、wait 0.10634，plant 条件熵约为 wait 的 2.47 倍；该 update 仅 110 个 plant 样本、1,064 个 wait 样本，需按这一较小 plant 计数解释。update 8 已有 plant 0.17481、wait 0.01542。按预设判据，后段 plant 类决策熵显著升高，支持它参与落点错配中间环节；熵轨迹并非从一开始只涨在 plant：首次跃升 update 2 是 wait 主导（0.05778 vs 0.01899），update 3–9 再出现 plant 熵持续偏高。结合 E1(c) 的 wait 条件分叉，较合适的机制描述是先出现等待策略分化，随后不确定性扩散到种植选择；100k 最终评估已完成并逐 seed 复现 C2：`evaluation update=9` greedy 130/256=50.78%、sampled 100/256=39.06%；`seed_results[mode].validation_full_level7_v1_d1[*]` 的 256 个 `won` 序列在 C2 与日志 run 中逐项完全相同。50k 节点同样逐项相同（greedy 186/256=72.66%、sampled 183/256=71.48%）。对应评估文件为两 run 的 `evaluations/update_000004.json.gz` 与 `update_000009.json.gz`。最终 trained checkpoint 的所有 `state_dict` 张量逐键 `torch.equal`：C2 `runs/run_1/update_000009_trained_1791863343036599018.pt` 与 E1(b) `.../update_000009_trained_1791975915645413032.pt`。因此日志实验完全重现了崩溃轨迹。

判定：若只看首次总熵跃升（update 2），主通道是 wait，plant-only 解释不足；到更新 6–9，plant 条件熵升为 0.09067、0.16554、0.17481、0.26238；最终 update 9 的 wait 熵也升到 0.10634，但 plant 仍约 2.47 倍。按预设判据，后段 plant 熵明确支持“种植决策变犹豫/选错”参与错配机制；时间顺序则是先有 wait 熵跳升，后扩散到 plant。结合 E1(c) 的首次 wait 条件分叉，不能把 wait 熵简单说成旁观指标。

## §E2 单步脆弱性扫描

状态：起点复原成功，40 个单步重复待执行。`bc_seed1_v1/output/training_state.json:update_history[0]` 的 update 1 使用 16 个 saved rollout shards，原 checkpoint 为 `runs/run_1/update_000000_evaluated_1791562128028880895.pt`（SHA256 `6da20070143c24430137effeb36b2a8e43a168e40c1c301ab656106ed9b9af17`）。因该 update 结束 checkpoint 已按保留策略裁剪，我从该起点 checkpoint + 原 shards 精确重放至 update 边界，并在 `/tmp/n7_E2/checkpoints/update0_reconstructed.pt` 重建模型；输出 SHA256 `c2a7dcde3f28567420cf134bd6bd458aee2b58678a32f4c43bcffdea59f1d2ea`。16 个 episode digest 通过，127 个 optimizer steps 一致；`policy_loss`、`value_loss`、总体 `entropy`、`advantage_mean/std`、`gradient_norm`、`approx_kl`、`clip_fraction`、`max_log_prob_change` 与 `training_state.json:update_history[0].losses` 的差值全为 0，详见 `/tmp/n7_E2_reconstruction.json`。因此复原起点与原 C2 update 1 完全一致。

每个时点都从同一 `initialization.source_checkpoint` 做权重转移（新 AdamW、新 stage RNG/计数），不是恢复原 optimizer/RNG 的严格 resume。统一配置为 20,000 decisions、`max_episodes=16`（恰好一个 16 局 batch），评估只用单 seed；时点内唯一实验变量是 `sampling_seed`。两个时点选 update 0 结束与 update 3 结束（文件 `update_000004_trained_*.pt`），因为原 seed1 update 1 已出现熵爆。熵跳预定义为 post-update replay entropy `>0.015`；分析分层列出跳/不跳的 rollout `won`，以及两时点爆炸率。配置在 `experiments/t7/n7_e2/{update0,update3}/sampling_*.json`，逐 run 结果写到 `/tmp/n7_E2_replay/`，汇总 `/tmp/n7_E2_summary.json`。

执行校正：首个 arm `n7_E2_update0_s41000` 在尚未设置 episode cap 时，20,000 decisions 实际形成 3 次更新（16、6、1 局；总计 20,000），而非一更。其 update 1 的 16 局从固定起点直接采样，单独保留的 `update_000001_trained_*.pt` 与该 update 的 rollout shard 可独立复算；本扫描只纳入 update 1，后两更排除。后续 39 个 arm 已增加 `max_episodes=16`，确保每个只做第一个 16 局 rollout batch。该上限只影响何时停止、不改变 update 1 的 rollout 配额/初始权重/采样种子；分析脚本读取明确的 update 1 checkpoint，不读多更 run 的最终 `resume.json`。另，seed 41001 的首轮自动回放因每次更新有多个 checkpoint stage 文件而报歧义（训练完整、单更新有效）；回放脚本已改为优先选 update-1 `boundary`，再 `evaluated`，再最新 `trained`。该次及 seed 41000 的熵回放待 E2 队列停点补算，不能视为未发生熵跳。

## §A1 seed0 + 守卫 0.02 @1M

状态：待执行。

## §B 守卫下复跑四个受污染实验

状态：待执行。

## §A2 seed1 + 守卫 0.02 @1M

状态：待执行。

## §A3 多种子对照

v6 已有数据汇总（字段来源：各 `training_state.json:update_history[*].losses.entropy`，以及同文件 `learning_curve` 中 `counters.decisions=100000` 的 greedy/sampled `summary.*.overall.pass_rate`）。完整逐更新 JSON 为 `/tmp/n7_v6_a3_c_summary.json`。

| seed | 无守卫 greedy/sampled @100k | 无守卫熵峰值→末值 | 有守卫 @100k（greedy/sampled；阈值） | 守卫触发/更新数；最高 post-step 熵 |
|---:|---:|---:|---|---|
| 0 | 84.77% / 81.64% | 0.00470→0.00248 | 83.98% / 85.16%（0.02） | 6/9；0.03486 |
| 1 | 50.78% / 39.06% | 0.11318→0.11318 | 84.38% / 85.16%（0.01）；84.38% / 84.38%（0.02）；86.33% / 81.25%（0.05） | 10/10；0.01926（0.01）；8/10；0.09347（0.02）；2/9；0.09347（0.05）|
| 2 | 63.67% / 50.00% | 0.07304→0.03240 | v6 无守卫扫描中没有 seed2 guard arm | — |
| 3 | 82.42% / 82.81% | 0.04179→0.02381 | v6 无守卫扫描中没有 seed3 guard arm | — |
| 4 | 87.89% / 85.94% | 0.00914→0.00342 | v6 无守卫扫描中没有 seed4 guard arm | — |

已配对的 seed0、seed1 因果检验沿用 v6 同一 JSON：`/tmp/n6_A2_results.json` 与 `/tmp/n6_A3_results.json`。seed0 guard 0.02 对 v3 两模式均不显著（Holm p=0.855536 / 0.422047）；seed1 guard 0.02 对无守卫 C2 两模式显著更好；A3 六项阈值×模式 Holm 家族 p=1.30814e−15 / 1.33988e−23（`/tmp/n6_A3_results.json`）；G1 四项比较家族为 3.92443e−15 / 1.07191e−23（`/tmp/n6_A2_results.json`）。v6 覆盖了 seed0–4 的无守卫表现，但守卫 arm 只覆盖 seed0/1，不能把后三个 seed 的守卫收益估成多种子均值；v7 新增长跑 seed0/1 完成后再纳入。


## §C 守卫机制

v6 仓库内可识别的守卫 run 为 4 个（报告任务书称“六个”，但 `/home/newbiexvwu/codex-report-v6.md` 与 artifacts 只对应 G1/G2/T001/T005 四条；未找到另外两条，最终保留这一审计差异）。字段：`artifacts/research/<run>/output/training_state.json:update_history[*].losses.entropy_guard.{triggered,post_step_entropy,attempted_optimizer_steps}`、`losses.stopped_for_kl`、同项 `trajectory_stats.won`。v6 @100k：seed0/.02 为 6/9 触发（66.7%），seed1/.01 为 10/10（100%），seed1/.02 为 8/10（80%），seed1/.05 为 2/9（22.2%）。最高 post-step entropy 依次为 0.03486、0.01926、0.09347、0.09347。四条共 5 次 KL stop，且 KL stop 与熵守卫触发没有同一更新重合；这表明两个停止条件在当前实现记录中分工可辨。

seed1/.01 已出现逐次全触发，符合“锁住 PPO 更新”的风险信号；但它仍达到了已报告的 84.38%/85.16%，不能据此断言整段训练没有有效学习（守卫触发只说明整次 train_update 被还原，非所有训练来源都归零）。默认 .02 在 v6 100k 为 66.7–80% 触发，尚未全锁；是否长跑触发率上升，留给 A1/A2 逐节点轨迹。被回滚更新后的 `won` 恢复不一致：有的下一更新持平/回升，有的明显下降至 0；这些单次 rollout 胜场不支持一个稳定“回滚后恢复”效应。逐更新明细在 `/tmp/n7_v6_a3_c_summary.json`。

A1/A2 与 B1–B4 尚未完成；完成后本节合并检查每次触发、KL stop、post-step entropy、attempted steps、后续 wins，并计算长跑触发率随预算节点的变化。


## §D v6 收尾项

`bc_seed4_v1` 的 100k 数据已直接纳入 §A3：greedy 225/256 (87.89%)、sampled 220/256 (85.94%)；熵首/末/峰值/中位数 0.00443/0.00342/0.00914/0.00663，见 `artifacts/research/bc_seed4_v1/output/training_state.json` 的 `learning_curve` 与 `update_history[*].losses.entropy`。v6 对 v3 的两模式 Holm p=0.6460 / 0.3955，均不显著，数字出自 `/tmp/n6_B_results.json`。

I7/I7b/I7c 使用修正后触发条件（承压路有僵尸、reachable shooter count=0、front_col 两侧各有合法 pea 落点、sun≥100，且以 `observation.legal_actions` 实查）。每臂 15/15 局均触发，但送达分别 15/150、15/150、14/150（10.0%、10.0%、9.3%），因此都属于剂量不足的意向处理。I7/I7b/I7c 通关数为 9/15、7/15、11/15；相对 I0 的 Holm p 分别 0.0195312、0.0625、0.00585938。I7c 波数对 I0 显著改善（Holm p=0.0109863）。I7 对 I7b 不显著（Holm p=1），不能确认 ahead/behind 落点效应；I7c 对 I7/I7b 也不显著，不能归因植物特异性。逐局与重算统计在 `/tmp/n6_C_results.json`、`/tmp/pvz-v6/interventions/C_summary.json`；送达字段为各 episode `delivered_placements`。

`bc_mb8_retry_v1` 使用 `ppo.minibatch_chunks=4` 的重试，在 4 小时上限 exit=137 timeout，最后提交 94,785 决策、8 次更新；50k 为 greedy 203/256 (79.30%)、sampled 210/256 (82.03%)，100k 节点不存在。证据为 `artifacts/research/bc_mb8_retry_v1/output/training_state.json`、`evaluations/update_000005.json.gz` 和 `/tmp/n6_bc_mb8_retry_v1.log`。这是基础设施/超时导致的未完成 run，不当作假设被否；熵末/峰/均见 §A1 补表。

## §E 一句话总结

待 E1/E2 证据完成后填写。


### 队列记录：bc_seed1_entropy_split_v1

时间：2026-10-14T19:28:41.978089+08:00；状态：complete；exit=0；耗时=7491 秒。配置：`experiments/t7/bc_seed1_entropy_split_v1/seed1.json`；日志：`/tmp/n7_bc_seed1_entropy_split_v1.log`。


### 队列记录：n7_E2_update0_s41000

时间：2026-10-14T19:36:12.148393+08:00；状态：complete；exit=0；耗时=245 秒。配置：`experiments/t7/n7_e2/update0/sampling_41000.json`；日志：`/tmp/n7_n7_E2_update0_s41000.log`。


### 队列记录：n7_E2_update0_s41001

时间：2026-10-14T19:38:54.211029+08:00；状态：complete；exit=0；耗时=155 秒。配置：`experiments/t7/n7_e2/update0/sampling_41001.json`；日志：`/tmp/n7_n7_E2_update0_s41001.log`。


### 队列记录：n7_E2_update0_s41002

时间：2026-10-14T19:41:46.848583+08:00；状态：complete；exit=0；耗时=155 秒。配置：`experiments/t7/n7_e2/update0/sampling_41002.json`；日志：`/tmp/n7_n7_E2_update0_s41002.log`。


### 队列记录：n7_E2_update0_s41003

时间：2026-10-14T19:44:54.915545+08:00；状态：complete；exit=0；耗时=155 秒。配置：`experiments/t7/n7_e2/update0/sampling_41003.json`；日志：`/tmp/n7_n7_E2_update0_s41003.log`。


### 队列记录：n7_E2_update0_s41004

时间：2026-10-14T19:47:55.444191+08:00；状态：complete；exit=0；耗时=155 秒。配置：`experiments/t7/n7_e2/update0/sampling_41004.json`；日志：`/tmp/n7_n7_E2_update0_s41004.log`。


### 队列记录：n7_E2_update0_s41005

时间：2026-10-14T19:50:59.226901+08:00；状态：complete；exit=0；耗时=155 秒。配置：`experiments/t7/n7_e2/update0/sampling_41005.json`；日志：`/tmp/n7_n7_E2_update0_s41005.log`。


### 队列记录：n7_E2_update0_s41006

时间：2026-10-14T19:53:30.244116+08:00；状态：complete；exit=0；耗时=125 秒。配置：`experiments/t7/n7_e2/update0/sampling_41006.json`；日志：`/tmp/n7_n7_E2_update0_s41006.log`。


### 队列记录：n7_E2_update0_s41007

时间：2026-10-14T19:56:30.526203+08:00；状态：complete；exit=0；耗时=155 秒。配置：`experiments/t7/n7_e2/update0/sampling_41007.json`；日志：`/tmp/n7_n7_E2_update0_s41007.log`。


### 队列记录：n7_E2_update0_s41008

时间：2026-10-14T19:59:01.360444+08:00；状态：complete；exit=0；耗时=125 秒。配置：`experiments/t7/n7_e2/update0/sampling_41008.json`；日志：`/tmp/n7_n7_E2_update0_s41008.log`。


### 队列记录：n7_E2_update0_s41009

时间：2026-10-14T20:02:02.911953+08:00；状态：complete；exit=0；耗时=155 秒。配置：`experiments/t7/n7_e2/update0/sampling_41009.json`；日志：`/tmp/n7_n7_E2_update0_s41009.log`。


### 队列记录：n7_E2_update0_s41010

时间：2026-10-14T20:04:40.381748+08:00；状态：complete；exit=0；耗时=133 秒。配置：`experiments/t7/n7_e2/update0/sampling_41010.json`；日志：`/tmp/n7_n7_E2_update0_s41010.log`。


### 队列记录：n7_E2_update0_s41011

时间：2026-10-14T20:07:11.345659+08:00；状态：complete；exit=0；耗时=125 秒。配置：`experiments/t7/n7_e2/update0/sampling_41011.json`；日志：`/tmp/n7_n7_E2_update0_s41011.log`。


### 队列记录：n7_E2_update0_s41012

时间：2026-10-14T20:10:14.566307+08:00；状态：complete；exit=0；耗时=155 秒。配置：`experiments/t7/n7_e2/update0/sampling_41012.json`；日志：`/tmp/n7_n7_E2_update0_s41012.log`。


### 队列记录：n7_E2_update0_s41013

时间：2026-10-14T20:12:46.090172+08:00；状态：complete；exit=0；耗时=125 秒。配置：`experiments/t7/n7_e2/update0/sampling_41013.json`；日志：`/tmp/n7_n7_E2_update0_s41013.log`。


### 队列记录：n7_E2_update0_s41014

时间：2026-10-14T20:15:46.806793+08:00；状态：complete；exit=0；耗时=155 秒。配置：`experiments/t7/n7_e2/update0/sampling_41014.json`；日志：`/tmp/n7_n7_E2_update0_s41014.log`。


### 队列记录：n7_E2_update0_s41015

时间：2026-10-14T20:18:16.856157+08:00；状态：complete；exit=0；耗时=125 秒。配置：`experiments/t7/n7_e2/update0/sampling_41015.json`；日志：`/tmp/n7_n7_E2_update0_s41015.log`。


### 队列记录：n7_E2_update0_s41016

时间：2026-10-14T20:21:17.180710+08:00；状态：complete；exit=0；耗时=155 秒。配置：`experiments/t7/n7_e2/update0/sampling_41016.json`；日志：`/tmp/n7_n7_E2_update0_s41016.log`。


### 队列记录：n7_E2_update0_s41017

时间：2026-10-14T20:24:20.286343+08:00；状态：complete；exit=0；耗时=155 秒。配置：`experiments/t7/n7_e2/update0/sampling_41017.json`；日志：`/tmp/n7_n7_E2_update0_s41017.log`。


### 队列记录：n7_E2_update0_s41018

时间：2026-10-14T20:27:22.194767+08:00；状态：complete；exit=0；耗时=155 秒。配置：`experiments/t7/n7_e2/update0/sampling_41018.json`；日志：`/tmp/n7_n7_E2_update0_s41018.log`。


### 队列记录：n7_E2_update0_s41019

时间：2026-10-14T20:29:52.652064+08:00；状态：complete；exit=0；耗时=125 秒。配置：`experiments/t7/n7_e2/update0/sampling_41019.json`；日志：`/tmp/n7_n7_E2_update0_s41019.log`。


### 队列记录：n7_E2_update3_s41000

时间：2026-10-14T20:45:02.552977+08:00；状态：complete；exit=0；耗时=882 秒。配置：`experiments/t7/n7_e2/update3/sampling_41000.json`；日志：`/tmp/n7_n7_E2_update3_s41000.log`。


### 队列记录：n7_E2_update3_s41001

时间：2026-10-14T20:49:06.084994+08:00；状态：complete；exit=0；耗时=215 秒。配置：`experiments/t7/n7_e2/update3/sampling_41001.json`；日志：`/tmp/n7_n7_E2_update3_s41001.log`。
