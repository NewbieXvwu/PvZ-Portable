# 通宵任务 v5：教师诊断上线 + 干预实验

## 状态与复现环境

- 2026-10-13 开始。`df -h ~` 可用 874 GB；GPU 起步时无计算进程。教师前置检查通过：`dsh` 位于 Node 24 路径，profile bundles 含 `@deepseek-ai/dsh-headless`，Torch 为 `2.14.0+cu132`，本机 8317 网关返回 HTTP 200。未触碰 `~/.pvz-keeper.sh` 或 keeper 计划任务。
- 起始仓库状态已有用户改动：`python/train_pvz_ppo_task_family.py` 修改 5 行，以及 `experiments/t7/bc_finetune_level7_v1/progress_evaluation_1024.json` 和四个新训练配置目录未跟踪。本任务保留这些改动；未改 `python/`、`scripts/` 源码。
- 用户明确要求不重做 A0、原始 A1、A3 关联分析；沿用 v4 报告与 `/tmp/n4_A1_summary.json`，本轮新增模拟只从已存档策略轨迹分叉。

## §A4 纯数据补充

来源：116 个 `artifacts/episode_samples/seed_*/v3_1m/{meta.json,frames.jsonl}`；分析脚本 `/tmp/n5_a4.py`，完整可复算输出 `/tmp/n5_A4_results.json`。这里的“突破”沿用脚本口径：首次有 c0–c1 植物消失的相邻帧；用该动作前帧阳光判定是否买不起 100 阳光豌豆射手。

- 107/116 局出现该前排植物损失；其中 78/107 的阳光低于 100。按结局分层：失败局 59/76（77.6%）低于 100，胜局 19/31（61.3%）低于 100。这是描述性比例，未作显著性声明。
- “有钱却未补火力”按 v4 的可行动机会规则，仅统计至少两行在 c2–c5 射手数少于 2、阳光≥100、豌豆卡可用且弱行有空格；漏失表示该步没有在弱行 c2–c5 种射手。按动作前波数分箱：

| 波段 | 有机会局数 | 机会数 | 漏失数 | 汇总漏失率 | 局均漏失率 |
|---|---:|---:|---:|---:|---:|
| w5–10 | 116 | 1,440 | 844 | 58.6% | 44.9% |
| w11–15 | 21 | 58 | 23 | 39.7% | 28.6% |
| w16–20 | 0 | 0 | 0 | 不适用 | 不适用 |
| w21+ | 2 | 7 | 2 | 28.6% | 25.0% |

  这是存活到各波并仍满足条件的机会分布；决策不是独立样本，以上仅描述，不据此声称波段差异显著。
- “末波死”操作定义为失败且 `final_wave >= 25`，共 17 局；对每局最后 3 个出现过的正波数中的种植动作汇总，得到 200 株、24,325 阳光。按植物编号/成本字段 `frames[i-1].packets[packet][5]` 聚合：豌豆射手(0) 9,300；双发射手(7) 9,400；寒冰射手(5) 3,675；向日葵(1) 1,450；坚果墙(3) 500。这里是末段总花费构成的描述性统计；逐局、逐植物计数在 `/tmp/n5_A4_results.json::late_death_spend.per_episode`。

## §A1 扰动对照 I4、§A2 保证剂量、§A3 wave 5 提前干预

全组都是 v4 冻结的 15 个失败 seed。I0 由 v4 原始配对 JSON 给出，15/15 与源档案全帧一致；新 arms 在触发点之前也都是 15/15 前缀匹配。新数据与精确检验汇总在 `/tmp/n5_A1_A3_summary.json`；新 arm 的逐局结果在 `/tmp/pvz-v5/interventions/<arm>_seed_<seed>.json`，完整干预帧档在同目录的 `episodes/<arm>_seed_<seed>/`。原 I1/I2/I3 结果仍来自 `/tmp/n4_A1_results/A1_seed_<seed>_<arm>.json`。

### A1：I4 随机合法动作控制

I4 在首次 c0–c1 植物损失的同一决策点开始，连续 10 个决策从当时所有合法 `plant` 动作与一个 `wait` 动作中均匀随机抽取；每个 seed 用固定 PRNG 种子 `seed*1009+4`。A1 的 Holm 家族含 I1/I2/I3/I4 对 I0、I3 对 I4、I2 对 I3 共 6 项精确配对 McNemar 检验。

| arm | 通关 | 终局波均值 | 实际种植 / 强制决策 | 对 I0 精确 p / Holm p |
|---|---:|---:|---:|---:|
| I0 原策略 | 0/15 | 13.67 | — | — |
| I1 首破行种 1 豌豆 | 1/15 | 14.60 | 2/15 | 1.000 / 1.000 |
| I2 旧版 10 步豌豆 | 6/15 | 19.87 | 9/150 (6.0%) | 0.03125 / 0.15625 |
| I3 旧版 10 步向日葵 | 8/15 | 21.87 | 14/150 (9.3%) | 0.0078125 / 0.046875 |
| I4 随机合法动作 | 5/15 | 19.87 | 34/150 (22.7%) | 0.0625 / 0.25 |

I3 对 I4 的精确配对 McNemar 为 `p=0.453125`、Holm `p=1.0`；I2 对 I3 为 `p=0.6875`、Holm `p=1.0`。I3 对 I0 在该六项家族中达到校正后 0.05 阈值，但 I3 与随机合法动作控制没有显著差异；I4 自己也救回 5/15。**判定：这组数据不能把 I3 的提升归因于向日葵的特异效果。轨迹扰动解释与结果相容，但 I3/I4 的差异不显著，不能据此证明全部提升都是伪影。**此外 I3 实际只放置 14 株/150 个干预决策，机制结论仍受剂量限制。终局波的配对精确符号检验经 Holm 后均不显著，精确值见同一 summary JSON 的 `paired_final_wave_sign_by_family.A1`。

### A2：从实时合法落点中挑选 I2g/I3g

I2g 从 `legal_actions.plants` 中挑豌豆动作，只在 c2–c5 且该行射手少于 4 的合法位置里优先选择火力最薄行；I3g 在合法向日葵位置中优先选择生产者较少的行，优先 c0–c1，同样限制所在行射手少于 4。模拟器合法动作集合已经排除了阳光不足、卡片冷却、占格和地形非法的动作。两个 arm 都从首次前排损失开始强制 10 个决策。

| arm | 通关 | 终局波均值 | 实际种植 / 强制决策 | 触发时 I-noop | 其中阳光不足 / 卡冷却 |
|---|---:|---:|---:|---:|---:|
| I2g 合法豌豆 | 6/15 | 19.87 | 9/150 (6.0%) | 13/15 | 10 / 3 |
| I3g 合法向日葵 | 8/15 | 21.87 | 14/150 (9.3%) | 10/15 | 8 / 2 |

这里的 I-noop 指**触发当下**没有合法种植动作；后续 10 个决策仍继续记录，若之后出现合法动作会计入实际剂量。15 局中 I2g 只有 2 局触发时可种豌豆，I3g 有 5 局触发时可种向日葵。触发时无钱的比例另有独立基线结论：在 76 个失败且出现首次前排植物损失的局中，59 局 (77.6%) 的动作前阳光低于豌豆成本 100。

I2g/I3g 的种植送达率都低于 50%，按预定规则标记为**意向处理，剂量不足**。A2 的 Holm 家族含两个 arm 对 I0、两 arm 彼此比较，以及两个 arm 各自对 I4 的扰动控制比较：I2g 对 I0 `p=0.03125`、Holm `p=0.125`；I3g 对 I0 `p=0.0078125`、Holm `p=0.0390625`；I2g 对 I3g Holm `p=1.0`；I2g/I3g 对 I4 的 Holm p 均为 `1.0`。所以 I3g 的通关差异相对 I0 显著，但它相对随机扰动控制仍不显著；低剂量也不允许把它解释成足量补种的疗效。

### A3：wave 5 起始的 I5/I6

wave 5 首个可见决策时，选 c2–c5 射手数最少的两行；I5 逐步补到每行 2 个射手，I6 在同样两行、按 I5 每局计划补种总数补向日葵。每个 arm 的总计划剂量都是 60 株，尝试窗口最多 10 个决策。

| arm | 通关 | 终局波均值 | 实际 / 计划种植 | 触发时 I-noop | 其中阳光不足 / 卡冷却 |
|---|---:|---:|---:|---:|---:|
| I5 补火力 | 4/15 | 18.13 | 10/60 (16.7%) | 14/15 | 10 / 4 |
| I6 补向日葵 | 6/15 | 21.80 | 13/60 (21.7%) | 5/15 | 3 / 2 |

I5/I6 都是意向处理、剂量不足。I5 对 I0 精确 McNemar `p=0.125`、Holm `p=0.25`；I6 对 I0 `p=0.03125`、Holm `p=0.09375`；I5 对 I6 Holm `p=0.6875`。**判定：I5 没有显著优于 I0；I5 与 I6 也没有显著差异。**波数的配对精确符号检验经 Holm 后同样都不显著，详见 `paired_final_wave_sign_by_family.A3`。

### 执行与统计复核记录

队列首轮曾在模拟前遇到种子列表转义问题；修正后又发现缺少末尾换行会漏掉最后一个 seed。各 arm 已补齐 seed 30953，最终分母均为 15。I2g 首次启动的植物观测结构读取错误发生在第一局未完成前，修正后重跑；该失败尝试不纳入统计。详情见 `/tmp/n5_progress.txt`。新臂每局串行执行，启动前检查 GPU；未并行运行模拟。

## §B 教师诊断（10 局，独立会话）

按任务书固定的样本结构，每局一份独立会话，统一使用 `--task-manifest experiments/t7/bc_finetune_level7_v1/train.json --task-id train_full_level7_v1_d1`：早/中死 30029、30059、30061；末波死 30329、30974；稳赢 30003、30014；I3g 救回 30718、30238；B9 `bc_seq32_100k` 崩局 30381（完整 899 动作，不是 32 步截断）。档案与提示词位于 `/tmp/pvz-v5/teacher/`，每次会话的 stdout、工具轨迹、卡点报告与 skill 输出分别放在每局 `run/`、`skills_out/`。

### 已完成 1/10：seed 30029（早死，wave 11）

- 独立 DSH session `session-0ec80a26-8548-4c86-82b8-23273572e967`，退出码 0，用时 1,111 秒；47 次工具调用。调用轨迹：`/tmp/pvz-v5/teacher/early_mid_30029/run/trace_summary.txt`；卡点统计：同目录 `trace_confusion.txt`；逐步回答：`session.log`。
- 会话核实后回放决策 38、90、95、130、152、184、396 等点。`whatif` 在决策 38/95/130/152/184/396 均有记录。代表性结果：决策 95 替换土豆雷落点 `plant:4:4:3` 使该条重放通关；决策 396 的候选最多多推 1 波，其他候选没有改善。单点通关不能单独作为机制证据。
- 该局可能的土豆雷安全距离判断只在决策 38 有干净证据；决策 95 同时改变路与落点，属混杂，未满足“两个不同决策点验证”。其他候选分别卡在反例问或证据问。**本局无新增 skill。**`lint_skills` 已调用，返回“没找到任何 skill”。
- 工具卡点：第 28 次 `whatif(enumerate=true)` 请求超时；后续改为手工候选并完成诊断。重复调用分组为 0。该单次超时不作为诊断假设被否的证据。
- 目前尚不能把“火力不足/向日葵易被吃”归为本局根因；此会话明确把相关现象保留为局部假设，没有升级成可迁移规则。

### 已完成 2/10：seed 30059（早死，wave 14）

- 独立 DSH session `session-dad68eb9-ac65-41c3-94ab-b38d740c611e`，退出码 0，用时 810 秒；44 次工具调用。轨迹、卡点与完整 session log 位于 `/tmp/pvz-v5/teacher/early_mid_30059/run/`。
- what-if 的主要命中：决策 170（从 `plant:0:2:3` 改到 `plant:0:3:2`）、224（`plant:0:3:3` 改到 `plant:0:4:2`）和 301（`plant:0:1:3` 改到同路 c1 的 `plant:0:1:1`），三种替换的策略续跑都通关。决策 119、145 是两个反例点；决策 220 的多个候选没有改善。第 224 步自动枚举超时，手工候选成功完成；不会把该工具超时记作机制被否。
- 候选机制为“最前僵尸已到内侧时，不要在僵尸右侧外列补直射射手”。三问在本局的检查记录：falsifier 列出全向/后向攻击、已有前排阻挡等不适用状态；3 个不同决策点有 what-if 结果；probe 仅依赖 `action_type`、`plant_role`、`action_col_min`、`lane_front_zombie_x_max`。`lint_skills` 返回 1/1 通过，随后 `probe` 验收 3/3 证据命中、2/2 反例未命中。文件：临时原件 `/tmp/pvz-v5/teacher/early_mid_30059/skills_out/avoid-rear-shooters-blind-spot/SKILL.md`，共享库副本 [avoid-rear-shooters-blind-spot](</home/newbiexvwu/PvZ-Portable/dsh/pvz-teacher/skills/avoid-rear-shooters-blind-spot/SKILL.md>)；两份 SHA256 都是 `df7be57109d7b44e2652e0861286808e7fba1db126cd5b72c2e160f3fb2be15f`。`dsh/pvz-teacher/README.md` 的技能清单已登记。
- 限制：probe 的 `shooter` 是较宽角色类，含裂荚射手、杨桃等特殊射手；skill 的 falsifier 排除了部分例外，因此 probe 是待人工核验的候选筛子，不是自动动作规则。本 skill 仅在一个 seed 的三个决策点验证，尚无跨 seed 复核；B 的汇总会把它列为单局过闸产物，不当作群体因果结论。
- 两次 lint 调用中第一次发生在文件写入前，返回目录为空；文件写入后第二次 1/1 通过。一次 `write` 调用带了不支持的 `justification` 参数而失败，随后无该参数重试成功。细节均保留在 trace。

### 已完成 3/10：seed 30061（早死/中死，wave 15）

- 独立 DSH session `session-3091a1e4-700c-4d4c-ba73-45de8995a323`，退出码 0，用时 900 秒；56 次工具调用。session log、完整调用摘要与卡点报告在 `/tmp/pvz-v5/teacher/early_mid_30061/run/`。
- What-if 点涵盖决策 114、121、135、149、157、219、432、446。决策 114/432/446 以射手或坚果墙补入薄弱路线后通关；决策 121/135 在平静路种向日葵改为向受压路前置坚果墙后通关；决策 149 的候选只推到 wave 27。自动枚举决策 75 超时，后续显式候选继续。
- 两条 skill 均调用最终 `lint_skills` 并通过（2/2）：
  - `threatened-lane-priority`：证据点 114、432、446。与 README 历史记载的 `match-lane-threat` 有语义重叠，本轮将其作为更广的受威胁路线分配规则记录；尚未做跨 seed 复核。
  - `avoid-economic-expansion-under-threat`：证据点 121、135，特指存在裸路承压时仍向平静路种向日葵；单局两点证据，不代表总体显著。
- 完全相同的文件已放入共享技能目录，SHA256 分别为 `9448814f66eb21d48b9a7abd3f11f49150e078a9dd3b31f3fee1112f8799fde6` 与 `2fba0e0b36e93d1403365d5cd4b01cde38b4d00364d92d4d3c4febe08a572c46`。本次只新增技能内容和更新既有技能索引，没有改工具、提示词或 MCP bundle。
- **统计边界**：这些 what-if 是同一随机 seed 下的策略续跑探针，不能替代 A1–A3 的 15 局配对实验；A3 的 I5/I6 在全组 Holm 校正后均未显著。全 10 个教师诊断完成后的跨样本归纳见本报告“B 轨合并结论”。

### 已完成 4/10：seed 30329（末波死，wave 29）

- 独立 DSH session `session-8f64e139-f159-4cf4-93a2-558796e9bf7b`，退出码 0，用时 1020 秒；60 次工具调用。归档在 `/tmp/pvz-v5/teacher/late_death_30329/run/`；完整诊断在 `run/session.log`，摘要和卡点在 `run/trace_summary.txt`、`run/trace_confusion.txt`。
- 诊断把可观测错误分成三类：承压裸路仍无火力、普通右射手种在已突破僵尸身后、未使用樱桃炸弹范围清场。对预先给定的群体假设，本局有 4 路在第 3 波承压裸路且射手为 0；第 105 步在阳光 50 时把向日葵种到安全路，第 113 步阳光 100 时又在僵尸已清空的安全路补射手。教师据当时阳光与合法卡片状态判断这里不是无钱被迫等待。
- What-if 重放在第 105、113、241、660、675、690、778 步均报告通关；前 3 点验证裸路威胁优先，后 4 点把后方/身侧普通射手替换为樱桃炸弹。第 113 步原始枚举超时一次，改成显式候选后完成；该超时是工具问题。以上是同一 seed 下的单步反事实重放，不是随机化的群体因果估计。
- 两条产物都回答反例问、列出至少两个 what-if 证据点并给出状态字段迁移判据，最终 `lint_skills` 为 2/2。`bare-lane-defense-priority` 与前面 `threatened-lane-priority` 语义重叠，本轮不再加一份重复共享规则；新机制 `emergency-cherry-bomb-clearance` 已复制到共享 skill 库，原件与副本 SHA256 均为 `30e6d30aea3f25304b447ff34cc1cb03e1f03aec391c4a72edfbf4480912babf`。文件：[emergency-cherry-bomb-clearance](</home/newbiexvwu/PvZ-Portable/dsh/pvz-teacher/skills/emergency-cherry-bomb-clearance/SKILL.md>)。这里只经单局验证，未做跨 seed 验证，也未用 `probe` 自动验收这条新 skill。
- `lint_skills` 对同一目录重复调用 4 次（最终仍为 2/2），what-if 自动枚举一次超时；这些是过程卡点，不当作策略假设被否的证据。

### 已完成 5/10：seed 30974（末波死，wave 29）

- 独立 DSH session `session-07f5ba55-e224-447c-b36f-2f09f924def1`，退出码 0，用时 1020 秒；61 次工具调用。完整日志与调用轨迹在 `/tmp/pvz-v5/teacher/late_death_30974/run/`。
- 本局没有复现“同波火力偏少”与“向日葵损失偏高”两条已知群体现象；教师将前期未补火力解释为阳光不足，报告称 88.1% 的相关决策低于 100 阳光。数字来源为 `run/session.log` 中的局内动作/阳光统计，属于单局描述，不能外推为群体比例。
- What-if 在 273、287、745、766 步把原射手替换为坚果墙后均通关；技能将机制限定为僵尸已进内侧、原射手很快死亡的紧急拦截。第 422 步的射手实际存活 12,086 tick，不符合 `plant_life_ticks_max:1200`，被排除；修订后 probe 证据 4/4 命中、反例 2/2 未命中，最终 `lint_skills` 1/1 通过。
- `emergency-wallnut-containment` 已复制到共享库，session 原件与副本 SHA256 均为 `fe135bd67658c1c50094e6e69c18c77f6d8b21507ea8809becbc9fef5ec5062c`。文件：[emergency-wallnut-containment](</home/newbiexvwu/PvZ-Portable/dsh/pvz-teacher/skills/emergency-wallnut-containment/SKILL.md>)。仅此一局四点验证，不能当作群体因果结论。
- 教师也在 744、745、766 步尝试樱桃炸弹候选并得到通关；它在本轮被识别为已存在的 `emergency-cherry-bomb-clearance`，所以不新建重复条目。注意本会话能看到上一局已加入共享库的该 skill；这些樱桃炸弹重放是有先前机制提示的跨局复核，不是盲诊断。
- 卡点：自动枚举 745 超时后改用显式候选；第 422 步导致首轮 lint 0/1，排除该误命中后最终 1/1；另有一次 write 参数错误后重试成功。trace confusion 共记录 3 次工具错误、1 组 lint 重复调用。均不当作策略假设被否。

### 已完成 6/10：seed 30003（稳赢对照，wave 30）

- 独立 DSH session `session-7690ab26-605d-40b3-8413-f963c2598869`，退出码 0，用时 630 秒；52 次工具调用。完整日志与轨迹在 `/tmp/pvz-v5/teacher/win_control_30003/run/`。
- 该胜局没有向日葵死亡，植物损失在 c4–c6；教师把它作为已知失败模式的反例控制，而不是要求从胜局硬找错。教师核查后明确“本局无新增 skill”。两个候选规则分别因只在决策 821 出现一次贴脸短命射手、以及撑杆越位仅有决策 695/696 一次事件，缺少第二个独立 what-if 点而未过证据问。
- 反事实：决策 38 跳过土豆雷后仍通关；决策 821 把短命豌豆替换为樱桃炸弹、坚果墙或等待都仍通关。没有结果显示可归因于候选动作的改善。
- trace 记录两次 `whatif(enumerate=true)` 在决策 38 超时，显式等待/候选重放成功；`lint_skills` 两次调用，结果均为空 skill 目录。超时与重复调用不记作假设被否。

### 已完成 7/10：seed 30014（稳赢对照，wave 30）

- 独立 DSH session `session-3fe69c67-ec35-4e66-b29e-1c5b0e31a165`，退出码 0，用时 720 秒；51 次工具调用。完整日志与轨迹在 `/tmp/pvz-v5/teacher/win_control_30014/run/`。
- 本局最终通关，但前 9 波植物损失中向日葵占 64.3%；全局共损失 9/31 株向日葵（29.0%），之后没有向日葵损失。教师把决策 134（有 50 阳光时在安全路种向日葵）与决策 146（有 100 阳光时在安全路补射手）列为优先级错配候选；读当帧阳光、卡片冷却和路况后，认为部分决策不是纯粹的资源不足。数字与判断见 `run/session.log` 对应的“局内假设核对”和阳光/动作明细。
- What-if 检查决策 146、195、224、254、393 等多个点；替换动作后仍通关且末波无提升。即便局内存在风险动作，胜局终局没有可区分的反事实结果；三个候选均未过证据问/反例问。教师明确“本局无新增 skill”，`lint_skills` 检查结果为空目录。
- trace confusion 记录 51 次调用、1 次错误：对未生成的 `save_best` 目录调用 `index`。这是工具路径错误；不影响原始存档，也不作为策略假设被否。

### 已完成 8/10：seed 30718（I3g 救回局）

- 独立 DSH session `session-763b7868-d5de-4451-bc88-01d7a4826664`，退出码 0，用时 781 秒；51 次工具调用。日志与轨迹在 `/tmp/pvz-v5/teacher/rescued_I3g_30718/run/`。
- 原 I0 轨迹第 19 波进屋，I3g 轨迹第 30 波通关、消耗 4 台割草机。教师认为 I3g 在第 4 波把原路线上的坚果墙换成向日葵，较早触发第 6 波割草机，清出 0 路后让策略重建火力；同一局另一个 what-if 在第 199 步换向日葵却只到 wave 13。**这支持“轨迹分叉/割草机清路可能带来救回”这一解释，但单局不能证明 I3g 的总体因果机制。**
- 更直接的替代策略证据：在 I0 第 115 步（tick 6240，0 路 0 火力、僵尸在 c7、阳光 100；原动作在已清空的 4 路补射手），改在 0 路 c2/c3/c4 种豌豆均到 wave 30；最佳存档 `run/out/best_whatif_115` 为 0 台割草机通关（原 I0 第 19 波失败，I3g 通关消耗 4 台）。决策 183（僵尸已过 0 路射手、3 路有来敌）把火力改投 3 路 c3 也通关。What-if 原件与结果见 session log、`trace_summary.txt` 及 `run/out/`。
- `prioritize-effective-firepower-placement` 通过三问和 `lint_skills` 1/1，两个证据点均有 +11 波变化。原件与共享库副本 SHA256 均为 `a8f5d1ffbd2cfe83b270d1a34ff23e7d51722333d6aeee4863d38d54c838ba82`。文件：[prioritize-effective-firepower-placement](</home/newbiexvwu/PvZ-Portable/dsh/pvz-teacher/skills/prioritize-effective-firepower-placement/SKILL.md>)。该综合规则与既有 `avoid-rear-shooters-blind-spot`、`threatened-lane-priority` 两条机制重叠；保留其合并表达和本局证据，不把它当成第三个独立根因。
- 自动枚举决策 115 超时，随后改为 115/183 显式动作；trace confusion 记录 1 次工具错误。自动枚举超时不作为假设被否。

### 已完成 9/10：seed 30238（I3g 救回局）

- 独立 DSH session `session-1d67efb5-d49d-420f-ad01-b0144fed9712`，退出码 0，用时 1081 秒；65 次工具调用。完整日志与轨迹在 `/tmp/pvz-v5/teacher/rescued_I3g_30238/run/`。
- I0 在第 17 波进屋，I3g 到 wave 30 通关。失败轨迹在第 6–17 波的总射手更多，但第 4 路直到 wave 13 仍空防；I3g 的射手分布更均衡。两轨在同一 tick 都损失一株向日葵，教师未把“向日葵被吃”判为这局的区分根因。
- 决策 296 是 I3g 干预点；替换向日葵/坚果墙/土豆雷会产生不同清路结果，向日葵延迟僵尸并改变后续割草机与重建防线的轨迹。教师明确说“种向日葵本身不是 skill”。单局轨迹差异不证明 I3g 总体疗效。
- 在 I0 上，决策 221、278、293 将已被越过僵尸的身后豌豆改成对准未防守路线的有效火力，三点均把结局从 wave 17 进屋改到 wave 30 通关。生成的 `no-shooter-behind-zombies` 回答三问、probe 证据 3/3、反例 2/2，`lint_skills` 最终 1/1 通过；它与已入库的 `avoid-rear-shooters-blind-spot` 在规则和 probe 上实质重复，所以不复制第二份。原件：[no-shooter-behind-zombies](</tmp/pvz-v5/teacher/rescued_I3g_30238/skills_out/no-shooter-behind-zombies/SKILL.md>)，SHA256 `2f5d1c3a4f90cc157d70b88d3b09a2608a92c32f79cb95c15b41c338a96cc0bf`。
- trace confusion 记 3 次工具错误（两次 what-if 请求超时、一次 write 参数错误）和 3 次 lint 调用；后续显式候选完成验证。初次超时不算假设被否。

### 已完成 10/10：seed 30381（`bc_seq32_100k` 失败局）

- 独立 DSH session `session-b2da9d48-a290-4521-918e-646cd478674b`，退出码 0，用时 780 秒；51 次工具调用。日志与轨迹在 `/tmp/pvz-v5/teacher/b9_seq32_30381/run/`。
- **口径校正**：`seq32` 是模型/训练运行标识，不是 32 决策截断。归档 `artifacts/episode_samples/seed_30381/bc_seq32_100k/meta.json` 记录 `outcome.actions=899`、`frame_count=900`、`won=false`、`final_wave=30`、`final_tick=49286`、`mower_triggered=2`；模型在第 30 波僵尸进屋失败。
- 决策 211 时有 125 阳光，豌豆/坚果/向日葵/土豆雷均就绪；策略仍在贴脸僵尸格种豌豆，300 tick 后被吃。决策 239 又在已越过僵尸身后补射手。What-if 中决策 211 改坚果或转投威胁路、决策 239 改坚果或补裸路射手均通关。
- 后段决策 751、771 在阳光 475–1050 且樱桃炸弹就绪时，基线仍补种短命寒冰/双发；分别改樱桃炸弹均通关。教师生成 `cherry-bomb-emergency-sweep`，回答三问、probe 与 `lint_skills` 1/1 通过。原件：[cherry-bomb-emergency-sweep](</tmp/pvz-v5/teacher/b9_seq32_30381/skills_out/cherry-bomb-emergency-sweep/SKILL.md>)，SHA256 `4c7ad6cd9985dd3ca39f9950ee7c9be5c4f0da221e2d493f1a3ba759b2848f04`。它与已共享的 `emergency-cherry-bomb-clearance` 重叠，因此作为跨模型/档案复核，不复制同义 skill。
- trace confusion 记录 2 次 What-if 超时（决策 211 枚举、决策 239 多候选）；改成单项显式候选后完成重放。超时不是假设被否。

### B 轨合并结论

- **重复出现的机制候选**：直射射手被放在已越过僵尸之后，或资源被投到安全路而承压裸路没有有效火力。seed 30059、30718、30238、30381 的局内 What-if 都有同类证据；这比“火力总数不足”更能解释部分局部崩溃。它是跨样本的机制复现，不是 348 局的新显著性检验。
- **“该种却没钱”与“有钱却投错路”并存**：A 轨 76 个首次前排损失失败局中 59 局阳光低于豌豆成本；但 seed 30381 决策 211（125 阳光、关键卡就绪）、seed 30014 决策 134/146 也有资源可用却优先安全路的例子。不能用单一根因概括全部失败。
- **I3g 救回不证明向日葵有特异疗效**：seed 30718 的早期割草机触发改变了轨迹，且原失败局直接补火力的 What-if 也可通关；seed 30238 则显示失败局后段总火力更多、但弱路分布更差。它们支持“轨迹与路线分配值得检验”，不能替代 A1/A2/A3 的配对检验。
- **技能边界**：通过闸门的 session 技能是单档案内的机制规则，不能当群体效应量。复用已登记的几何盲区、樱桃炸弹清场规则时，本报告计为跨局复核，不重复建同义文件；两个赢局都无新增 skill，符合反例控制用途。

复现入口：各 case 的 `run/session.log`（诊断结论与 What-if 前后动作）、`run/trace_summary.txt`（工具参数和结果）、`run/trace_confusion.txt`（超时/工具错误）；archive 身份与结果见对应 `meta.json` 的 `task` / `outcome` / `totals` 字段；过闸技能原件在各 case 的 `skills_out/*/SKILL.md`。B 轨至此完成；C 轨训练与配对统计见 §C。

## §C 补跑训练实验

对照为 v3 相同决策数的验证节点，按 256 个相同 validation seed 配对。主指标是通关/失败的精确 McNemar 检验；次指标是 terminal_wave 的精确配对符号检验。两类检验各自对预定 12 项 run×node×mode 家族做 Holm 校正；未形成完整节点的计划检验以 p=1 进入校正排序，但标记为不可估计，不报告推断。

| 实验 | 唯一变量 / 预算 | 状态 | 节点结果 |
|---|---|---|---|
| `bc_lr10_v1` | learning_rate=1e-5，300k | timeout，exit 137，28,846 秒 | 100k、200k 完整；300k 未完成。见下方逐节点检验。 |
| `bc_seed1_v1` | initialization_seed=1、sampling_seed=1，100k | 完成，exit 0，3,809 秒 | 50k：greedy 186/256、sampled 183/256（仅描述）；100k：130/256、100/256。 |
| `bc_gae95_v1` | gae_lambda=0.95，100k | 完成；3 次串行进程累计 5,878 秒，最终验证续跑 exit 0 | 50k：greedy 184/256、sampled 182/256（仅描述）；100k：192/256、183/256。 |
| `bc_mb8_v1` | minibatch_chunks=8，100k | OOM，exit 1，2,057 秒 | 初始 0 决策评估 greedy 216/256、sampled 218/256；update 1 的 epoch 1 部分执行 8 个 optimizer step 后 OOM，未提交完整训练更新，也没有 50k/100k 评估。 |

### 对 v3 的配对统计

| Run / 节点 / 模式 | v3 → run 通关 | 差值 | McNemar p → Holm p | terminal_wave 均值差；上升/下降/平局 | 波数符号检验 p → Holm p | 结论 |
|---|---:|---:|---:|---|---:|---|
| `bc_lr10_v1` / 100k / greedy | 217/256 → 216/256 | -0.39 pp | 1 → 1 | -0.297；13/26/217 | 0.053252 → 0.488739 | 不显著 |
| `bc_lr10_v1` / 100k / sampled | 209/256 → 216/256 | +2.73 pp | 0.281042 → 1 | +0.043；21/22/213 | 1 → 1 | 不显著 |
| `bc_lr10_v1` / 200k / greedy | 208/256 → 213/256 | +1.95 pp | 0.532709 → 1 | -0.125；23/24/209 | 1 → 1 | 不显著 |
| `bc_lr10_v1` / 200k / sampled | 215/256 → 212/256 | -1.17 pp | 0.735879 → 1 | -0.363；17/28/211 | 0.135156 → 0.989852 | 不显著 |
| `bc_seed1_v1` / 100k / greedy | 217/256 → 130/256 | -33.98 pp | 1.97862e-16 → 2.17648e-15 | -2.320；22/87/147 | 2.48928e-10 → 2.73821e-9 | 通关率与波数均显著更差 |
| `bc_seed1_v1` / 100k / sampled | 209/256 → 100/256 | -42.58 pp | 4.83238e-21 → 5.79886e-20 | -4.250；28/111/117 | 6.81331e-13 → 8.17597e-12 | 通关率与波数均显著更差 |
| `bc_gae95_v1` / 100k / greedy | 217/256 → 192/256 | -9.77 pp | 0.000621043 → 0.00621043 | -0.191；18/33/205 | 0.048874 → 0.488739 | 通关率显著更差；波数不显著 |
| `bc_gae95_v1` / 100k / sampled | 209/256 → 183/256 | -10.16 pp | 0.000685564 → 0.00621043 | -0.227；24/37/195 | 0.123731 → 0.989852 | 通关率显著更差；波数不显著 |

C1 的 100k/200k 通关与 terminal_wave 结果都不显著，不能据方向性差值称 learning_rate=1e-5 更好或更差；300k 因超时没有可比节点。C2 的这个单独改种子实验相对 v3 显著更差，说明该训练结果对这组初始化/采样种子敏感；单个替代种子不足以估计跨种子的总体噪声分布。C3 在 100k 的两个通关模式均经 Holm 校正显著更差，terminal_wave 检验均不显著。C4 的 OOM 是运行限制，不是效果检验，也不能据此支持或否定 minibatch_chunks=8。

### 训练续跑与复现字段

- C1 运行日志：`/tmp/n5_bc_lr10_v1.log`；100k/200k 评估日志分别见 `artifacts/research/bc_lr10_v1/output_300k/evaluations/update_000008.json.gz` 与 `update_000016.json.gz`。300k 的 28,846 秒超时记录在 `/tmp/n5_progress.txt`。
- C2 50k/100k 评估分别为 `artifacts/research/bc_seed1_v1/output/evaluations/update_000004.json.gz`、`update_000009.json.gz`；C3 对应 50k/100k 为 `artifacts/research/bc_gae95_v1/output/evaluations/update_000005.json.gz`、`update_000009.json.gz`。v3 的 100k/200k 基线文件分别为 `artifacts/research/bc_finetune_level7_v3/output/evaluations/update_000008.json.gz`、`update_000016.json.gz`。
- 对每个评估文件，核对 `counters.decisions`；通关数从 `seed_results[mode].validation_full_level7_v1_d1[*].won` 计数，波数取同一记录的 `terminal_wave`。精确配对数、discordant 数、原始 p、Holm p、基线与 run 文件路径均在 `/tmp/n5_training_comparisons.json` 的 `comparisons[*]` 字段中；该文件覆盖上述 12 个预定检验。
- C3 第一次在 50k 评估完成后遇到 CUDA worker 错误，第二次续跑到 100k 后 100k 验证 worker 再次遇到同类 CUDA/Dynamo 错误；训练检查点保留并完成预算。最终只在待完成的验证续跑中设 `PVZ_RELATION_BIAS_FUSION=0`，使用本地模型源码注明的 eager、bit-identical 关系偏置路径，因此通过 512/512 验证并落盘。未修改配置或源码。原始失败日志依次为 `/tmp/n5_bc_gae95_v1_attempt1.log`、`/tmp/n5_bc_gae95_v1_attempt2.log`；最后成功日志为 `/tmp/n5_bc_gae95_v1.log`。
- C4 初始评估在 `artifacts/research/bc_mb8_v1/output/evaluations/update_000000.json.gz`；OOM traceback 与 8 个部分 optimizer step 见 `/tmp/n5_bc_mb8_v1.log`。其 `training_state.json` 仍为 0 个已提交训练更新，不能把部分 step 当作训练结果。

配置来源：`experiments/t7/bc_lr10_v1/seed0.json`、`experiments/t7/bc_seed1_v1/seed1.json`、`experiments/t7/bc_gae95_v1/seed0.json`、`experiments/t7/bc_mb8_v1/seed0.json`。

## §D 最终结论

### 根因判断（按证据强度）

1. **有资源时仍错失补火力，是目前最强的群体级策略信号。** 在至少两行火力低于 2 时，失败局“有钱却没补火力”的错失率为 41.5%，赢局为 26.1%，差 +15.4 pp，`p=0.00006`；这是显著关联，不是单独的因果证明。口径与检验见 `~/codex-report-v4.md` §A3（表中 `p=0.000060`、95% CI `[+8.0,+22.5] pp`）；逐局来源为 v3_1m 帧记录。
2. **第一次前排植物损失时现金不足很常见，限制了补救。** 76 个失败档案中 59 个（77.6%）在该点阳光低于豌豆成本；I2g/I3g/I5/I6 都有大量 `I-noop`，实际送达率最高也低于 50%。这是早期资源约束的描述性证据，不能把之后的缺钱直接等同于最初根因。
3. **安全路花费与承压路防守不足是可复现的局内机制。** 教师与 What-if 在多个失败档案里找到“射手种在僵尸身后”或把资源投到安全路、承压路留空的反事实通关方案；这支持路线错配机制，但十局诊断是挑选样本，不能当群体发生率。
4. **没有通过干预证明应该补哪种植物。** I3g 对 I0 的意向处理比较显著（Holm `p=0.0391`），但实际种下仅 14/150（9.3%）；I3g 对随机扰动 I4 的差异不显著（Holm `p=1`）。I5/I6 在第 5 波的送达率分别 10/60（16.7%）和 13/60（21.7%），I5 对 I0 Holm `p=0.25`、I6 对 I0 Holm `p=0.09375`、I5 对 I6 Holm `p=0.6875`，均不显著。轨迹扰动解释与数据相容，但未证实。
5. **训练配置结果有明显不确定性。** C2 的单次替代种子与 C3 的 `gae_lambda=0.95` 在 100k 都比 v3 显著更差；C1 的 `learning_rate=1e-5` 在已完成的 100k/200k 节点均不显著；C4 OOM 后没有可作效果判断的训练节点。它们不直接证明游戏失败的根因。

### 下一步优先级

1. 在阳光耗尽前、尽量靠近首次前排植物损失之前做干预；预先区分“当时没有任何合法可执行落点”的 I-noop 与有合法落点的处理组，并报告实际送达率。
2. 在有合法落点的局内，比较补薄弱承压路火力、补经济与同时间随机合法动作；继续用配对种子和 Holm 校正。送达率未达到预设门槛时只报告意向处理。
3. C1 的小差值按不显著处理；不据此调整 learning_rate。不要把单个 C2 种子推广成一般噪声估计，也不要采用本次表现显著更差的 C3 参数。C4 先解决可复现的显存占用问题，再决定是否重跑。
4. 教师技能作为候选局部规则进入后续验证集，保持反例问、证据问、迁移问闸门；单局 What-if 成功仍不等于群体疗效。

**一句话总结：** 证据最支持“经济缓冲不足与已有资源的路线错配并存”，其中有钱时漏补火力有显著群体关联；下一步优先在首次损失前做合法落点约束、足量送达、带随机扰动对照的配对干预。
