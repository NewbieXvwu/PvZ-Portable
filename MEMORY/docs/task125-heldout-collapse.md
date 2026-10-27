# task125：第 21 轮学生 heldout −33pp 的根因（紧急）

主控 2026-10-27 09:15 派给台式机 codexb-task93（Business）。只读生产文件，不动 tmux qc；分支 `heldout21`，基于 `origin/deploy14f`。

## 现象

门禁 v4（`~/pvz-logs/curriculum_state.pipeline/round_000021/complete.json` 的 `promotion`），候选 = 第 21 轮种子 1（现行配方 dw4i2，model 6ee39ee7…）vs r11（1cf82f5e…）：

| 区 | 第 20 轮 Δ | 第 21 轮 Δ（gain/harm） |
|---|---:|---|
| day | −3.6 | 0.0（20/20） |
| **heldout**（`day_l7_eval_d7p_holdout1`、`day_l9_eval_d7p_holdout2`，没见过的卡组） | −2.6 | **−33.3（30/158）** |
| night | −0.3 | −2.1 |
| pool | +5.5 | −3.4 |
| roof | +6.2 | +15.1 |
| 总体 | +0.2 | −6.2 |

第 21 轮相对第 20 轮的变化：
- 代码：deploy14e → **deploy14f**（task99 起点格种子续期 **+ plane packing**、引擎 build4 de952714）——第 21 轮是第一个用 14f 打包 / 训练的轮次；deploy14g（task119）只多了 B 配方与开局格，A 配方不变。
- 数据：新增约 100 个标签（c21 本轮 + task108 回填的 c13/c19/c6 组），各约 0.1%。
- 训练时内存抖动（种子 1 第 1 个 epoch 4.3 小时），数据本身应相同。

## 要回答（按顺序，先便宜后贵）

1. task99 / deploy14f 改了打包的哪些东西（plane packing）？对卡组 / 牌包特征（尤其 heldout 卡组里出现、训练卡组里少见的卡）是否改变了训练行的数值？用少量行对比 r20 与 r21 的打包结果（同一档案、两版代码），找到第一处差异。
2. 若打包无差异：看 heldout 两个任务上 r21 学生的输局在哪里分叉（与 r11 同种子），判断是卡槽 / 牌包识别问题还是别的。
3. 若确认是打包 bug：给出最小修复，并说明第 22 轮（已排队、用同一代码）训练出的学生是否也受影响、需不需要重新打包。

## 约束

- AGENTS §6.5：用最便宜的手段定位（小样本、单档案），不全量重放、不全量重新打包。
- 先在本文件末尾写结论，再推送。紧急：第 22 轮训练即将开始。

## 结果

CodeBuddy 2026-10-27 接手（heldout21 @ `ad651176` = `origin/deploy14f`）。Codex 已写好探针并跑出一份结果，本轮修正了探针里 `changed_rows` 的归约轴、复跑核对，再往下做。

1. **打包确实变了，但变的不是卡组 / 牌包特征。** 用同一份真实生产段
   `frames/packs/segments/ce3ce4006691a0e1.npz`（label `c18demoroof48doomm200s1100_fast_search`，24 个档案 / 84 行），
   分别用 deploy14e 之前的 `CompactBoards` 与 `ad651176` 的 `CompactBoards` 重打
   （`scripts/task125_pack_probe.py`，输出 `artifacts/task125/pack_probe.json`）：
   - `board_plane_of`：旧 72 个 id / 84 行 → 新 84 个 id（前 12 行 `-1`）。
   - 42/84 行、4326 个格、8 个通道变化：植物血量 1/33/39/44、冰冻 89、车道 91、格内位置 93、护具 94。
     第一处差异 = 行 12、通道 1、格 (0,0)：`0.0 → 1.0`。
   - **逐位相同**：`scalar` / `plant` / `packet` / `cell` / `weight` / `seed` / `packet_columns` /
     `source_rows` / `csr_indices` / `csr_values` / `plane_values`。
   → scalars 里的 54 张卡就绪位与冷却、packet 头全部不变。**塌陷不是"认不出卡组"。**

2. **变化的方向是修复，不是回归。** `constant_plane` 把"整个档案里恒定的格子"提成每档案一个平面。
   旧 `append()` 只在带平面的行写 id，短档案（<4 行）不写，于是 72 个 id 对应 84 行；旧读端见到
   `len(plane_of) != rows` 就**整段按无平面读**（`scripts/train_rollout_policy.py:619` 的注释写明了这次事故），
   那 8 个通道在 r20 及以前是 0。deploy14f 修的正是这个：`append()` 补 `-1`（含中间出现的无平面档案，
   `scripts/train_rollout_policy.py:124`）、平面值按 board dtype 转 float16、`finish()` 断言 id 覆盖每行，
   再加 `segment_plane_shapes_valid`（`scripts/pack_training_rows.py:352`）强制重打。
   → r21 是第一个"训练行 == 评估时 featurizer 输出"的轮次；修复前是 r20 存在训练 / 评估输入不一致。
   `python/test_incremental_pack.py` + `python/test_compact_loader.py` 现测 17 passed / 1 skipped。

3. **修复的体量极小，不足以解释 −33pp。** 段头审计（`scripts/task125_metadata_probe.py`，
   回执 `artifacts/task125/metadata_probe.json`）：r20 的 513 个段里，**只有 1 个段**（`ce3ce4006691a0e1.npz`，
   label `c18demoroof48doomm200s1100_fast_search`，84 行 = 75 训练 / 9 验证）在 r21 pack 之后被改过，
   另外 **512 个逐字节未动**。r21 `built 101 / reused 512`，而 r20→r21 标签数 513→613（+100），
   所以 101 个 built = 100 个新标签的新段 + 1 个重打。84 / 19,649,496 = **0.0004%** 的行。
   r22 里这个标签还在（84 行，81 训练 / 3 验证），但段已合格。

4. **r20→r21 真正变大的是数据构成，不是打包。** `source_rows.json`：18,811,791 → 19,649,496
   （+837,705，+4.45%）。分区占比：

   | 区 | r20 | r21 | 变化 |
   |---|---:|---:|---:|
   | other-z（z13/z15/z16 dag） | 52.72% | 50.47% | −2.25pp |
   | night | 13.98% | 15.07% | +1.09pp |
   | pool | 6.04% | 6.64% | +0.60pp |
   | day | 6.14% | 6.67% | +0.53pp |
   | demo / 起点格 | 0.57% | 1.09% | **+0.52pp（行数翻倍 106,960 → 214,969）** |
   | fog | 0.52% | 0.80% | +0.28pp |
   | roof | 0.17% | 0.24% | +0.07pp |

   新增的 100 个标签按来源轮次：c21 59 个 / 473,266 行、c13 16 / 207,300、c20 17 / 101,397、
   c6 2 / 23,134、c19 2 / 18,953、c14 2 / 12,594、c18 2 / 1,061。
   配方除 `out` 与 `labels`（+100）外逐字段相同；两轮都是训练 2 个种子再挑选（r20 选 seed 2，r21 选 seed 1）。

5. **heldout 是已知的高方差区。** task84 主控追加已实测：配方与种子都不变、重训一次，heldout 从
   166/192 掉到 124/192（−21.9pp）。r21 的 −33.3pp（206/384 vs 334/384）比这更大，但 r21 同时还多了
   +4.45% 的行与翻倍的 demo 族。heldout 占 384/1920 对，−33.3pp × 20% ≈ −6.7pp，与总体 −6.2pp 基本相当；
   其余区几乎持平（day 0.0、roof +15.1、night −2.1、pool −3.4）。

6. **结论：塌陷不是打包 bug 造成的。** 打包那处是已经修好的正确性问题，只动了 0.0004% 的行、
   8 个非卡组通道；卡组 / 牌包特征逐位未变；C++ `observation == 6` 的 boss 通道 14→13 只影响 boss，
   而 day / pool / night / roof 都基本持平，可排除全局特征错位。r21 的 heldout 退步更像
   **数据构成变化 + 已知训练方差**，不是输入被写坏。

7. **第 22 轮：不需要重新打包。** 22 轮 pack 已跑完（09:11，`rows 19,799,497, built 28, reused 613`），
   641 个段全部通过 `segment_plane_shapes_valid`（实测 `board_plane_of` 长度全部 == `rows`，0 个不合格）。
   它与 21 轮同代码，数据 = 21 轮 + 28 个新段。**不要回退 deploy14f 的打包修复** —— 回退等于把那 84 行
   重新写成 0，并把闸门一起关掉。

8. **没做的事**：任务书第 2 项（heldout 两任务上的逐局分叉）没跑。按任务书自己的顺序，它是"若打包无差异"
   才做；实测差异只有 84 行且方向是修复，所以先给上面的构成 / 方差结论。逐局分叉要起游戏进程，
   本机此刻已有 6 个 `pvz-portable`（cq / llmimport 两个工作树）在跑、swap 已用 5.9/8G，
   按"最多 2 个游戏进程"的约束没有开。

## 问题

1. 要坐实 r21 的 heldout 塌陷，最省的下一步不是逐局分叉，而是按 task84 已预登记的口径补一次
   "同配方换种子"重训（heldout 192 对），把 −33pp 与 22pp 的方差带直接比。这要 GPU 与一次完整训练，
   不在本任务（只读诊断）范围内，交主控定。
2. 门禁 heldout 区仍按点估计 + Holm 判定，而 task84 已证明该区单次重训就有 ~22pp 摆动；
   r21 这次 −33pp 被显著否决。要不要把 heldout 改成多种子口径，是 task84 的遗留未决项。
3. 两份回执（`pack_probe.json`、`metadata_probe.json`）在 `artifacts/task125/` 下，按现行
   `artifacts/.gitignore` 的 `*` 不入库（没有用 `git add -f`）。要换机器复现就重跑
   `scripts/task125_pack_probe.py` / `scripts/task125_metadata_probe.py`（都只读生产文件，不起游戏进程）。
   若主控希望这两份结论层 json 入库，需要按 §7 在 `artifacts/.gitignore` 加白名单，由主控决定。
