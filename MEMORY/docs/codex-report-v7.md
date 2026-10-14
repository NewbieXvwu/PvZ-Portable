# 接续任务 v7：守卫保护下的训练机制与效果

开始：2026-10-14（Asia/Shanghai）。按 v7 任务书继续；v6 既有结果直接采信。原始证据文件、字段、配对统计 JSON 与命令日志逐阶段记录。

## §0 守卫的代价复核

### 身份与影响范围

`python/pvz_research.py:285` 的 `experiment_identity_digest()` 将完整语义配置与源码指纹一起哈希；`:458-483` 对每个实验计算包含 `python/pvz_research.py` 和 `python/train_pvz_ppo.py` 在内的来源指纹；`:492-506` 的 `--resume` 将重新计算的 identity 与检查点记录值比较，不同即拒绝续跑。v6 熵守卫改动了上述两个文件，所以旧 identity 下的检查点不能用当前源码直接续跑。

盘点 `artifacts/research/bc_*/output/provenance.json:fingerprints`（25 个 run）时，16 个 run 仍记录守卫实现之前的文件指纹 `pvz_research.py=0c7439f0…`、`train_pvz_ppo.py=6fb67bcd…`：`bc_clip05_v1`、`bc_criticwarm_v1`、`bc_ent_v1`、`bc_ep1_v1`、`bc_finetune_level7_v3`、`bc_gae95_v1`、`bc_kl001_night2`、`bc_kl_ab_v1`、`bc_lr10_night2`、`bc_lr10_v1`、`bc_mb16_v1`、`bc_mb8_v1`、`bc_roll64_v1`、`bc_seed1_v1`、`bc_seq32_v1`、`bc_val2_v1`。这包括 v3 的 1M 与 C1 `bc_lr10_v1/output_300k` 的 256,226 决策前缀。v6 后用守卫版本生成的结果有新 identity；未启用守卫的 seed2/3/4 虽然配置不含守卫字段，源码指纹仍是 guard-aware 版本。

受影响动作是**直接续跑**：当前源码重算的 identity 与这些旧检查点不符，resume 会因源码/配置 identity 校验失败。读取旧 `training_state.json`、评估 gzip、学习曲线和 provenance 不受影响；`episode_query.py --checkpoint` 直接载入权重做推理也不走训练 identity 校验。v7 不需要续跑上述旧 run：A1/B/A2 都是新实验，E1(a) 是推理；E2 从指定检查点做新的权重初始化（新 optimizer、新计数），不是 resume。因此接受该影响，不为旧 run 回滚或改写 identity。

回滚 v6 守卫源码的原命令仍为：`cp -p /tmp/n6_train_pvz_ppo.py.bak python/train_pvz_ppo.py` 与 `cp -p /tmp/n6_pvz_research.py.bak python/pvz_research.py`。这会同时移除熵守卫与配置接线，不执行。E1(b) 另有“仅熵分组日志”改动；其专用前置备份会在修改时记录，恢复它可单独撤销日志而保留 v6 守卫。源码指纹变化只影响续跑身份，实验数据读取和报告仍有效。

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

状态：待执行；源文件只增加日志字段，先保存 `/tmp/n7_train_pvz_ppo_pre_entropy_type.py.bak`，不改变 loss、梯度、更新、守卫条件或随机数路径。

## §E2 单步脆弱性扫描

状态：待执行。

## §A1 seed0 + 守卫 0.02 @1M

状态：待执行。

## §B 守卫下复跑四个受污染实验

状态：待执行。

## §A2 seed1 + 守卫 0.02 @1M

状态：待执行。

## §A3 多种子对照

状态：待执行；纳入 v6 已完成 seed0–4 的 100k 数据。

## §C 守卫机制

状态：待执行。

## §D v6 收尾项

`bc_seed4_v1` 100k 与 I7/I7b/I7c 触发点修正重跑均已在 v6 完成，结果直接采信；`bc_mb8_retry_v1` 以 v6 设定的 4 小时上限结束于 94,785 决策，没有 100k 评估，不视为策略假设被否。详细来源与字段在后续收尾核验中补录。

## §E 一句话总结

待 E1/E2 证据完成后填写。
