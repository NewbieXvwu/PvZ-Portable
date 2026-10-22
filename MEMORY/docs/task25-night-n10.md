# task25（主控 2026-10-22 下发给台式机 CodeBuddy 第二条线）：夜间卡组 n10（+墓碑吞噬者）任务族

先读 `AGENTS.md`（§8）、DESIGN §14.9、`scripts/make_night_panel.py`、`MEMORY/docs/student-bottleneck-plan.md` 末尾 10-22 的几条。

## 背景
夜间 30 波关（2-2/2-4/2-7/2-9）是难关；最后一波开始约 200 tick 后每个坟头各冒一只僵尸（Board.cpp:4958–4962 → SpawnZombiesFromGraves），
2-7..2-9 有 11 个坟头。老师（dw4i_s1 推演，beta=1，16 局）n9 → n10（+墓碑吞噬者 11）：2-7 4 → 11、2-9 1 → 8。**夜间卡组定为 n10 `[0,1,2,3,4,5,7,8,9,11]`**（10 张 = SEEDBANK_MAX）。

## 要做的
工作区：`~/PvZ-search/.claude/worktrees/nightn10`（分支 `nightn10`，主控已建好）。
1. `make_night_panel.py --combined` 加 `night_lN_eval_n10` 与 `night_lN_train_n10`（N = 11–14、16–19；L20 同时出原版开局版 `night_l20_eval_n10_orig` / `night_l20_train_n10_orig`，
   用 task20 的 `conveyor_start` 字段，仿 task23 的 `night_l20_eval_n9_orig`），训练种子 `2200000 + (N−11)·10000` 起每关 8192 个，生成时自查与所有已有段不重叠。
   已有任务逐字节不变（生成前后对比）。写到 `~/PvZ-search/artifacts/stage4/night_panel.json`（先备份 `.bak_task25`）。
2. DESIGN §14.9 登记（含上面的老师数据与坟头机制）。冒烟：每个新训练任务 1 局（学生 `~/PvZ-search/artifacts/student/dw4i_s1/model.pt`，`--workers 1`），确认 reset 后卡槽是 10 张。
3. 提交、`git push origin nightn10`，在“结果”记一行，然后停下。

## 规则
不动任何 tmux 队列与别的代理会话；不改别的工作树；同时最多 2 个游戏进程；不改默认值与 `deck_for_level`；不写手工规则；不写字面密钥；不 `git add -f`。只用绝对路径。

## 结果

- 2026-10-22 第一步（脚本 + 清单）：`scripts/make_night_panel.py` 加 n10 族 —— `night_lN_eval_n10` / `night_lN_train_n10`（N = 11–14、16–19，卡组 `[0,1,2,3,4,5,7,8,9,11]` 共 10 张 = SEEDBANK_MAX，训练种子 `2200000+(N−11)·10000` 起每关 8192：2200000 / 2210000 / 2220000 / 2230000 / 2250000 / 2260000 / 2270000 / 2280000），L20 只出原版开局版 `night_l20_eval_n10_orig` / `night_l20_train_n10_orig`（`deck: []` + `conveyor_start: "original"`，训练 2290000–2298191）。自查：卡组长度 10 / 无重复 / 无升级植物；训练种子落在各自段位内、与 30000–31023 / 1500000 / 1600000 / 1800000–1878191 / 1880000–1888191 / 1900000–1998191 / 2000000–2088191 / 2100000 / 2120000 / 2130000 / 2150000 各段不重叠、训练任务之间不重叠。重新生成 `~/PvZ-search/artifacts/stage4/night_panel.json`（旧文件备份 `night_panel.json.bak_task25`）：**36 → 54 任务，原有 36 个任务逐任务 JSON 逐字节相同**；工作树内另存一份（cmp 相同）。

- 2026-10-22 第二步（冒烟）：本工作树单独编译（Release + BLAS，`build/pvz-portable`）。① reset 探针（`PvZEnv.reset`，逐任务核对 `obs["packets"]`）：18 个新任务全部 PASS —— 8 个 `night_lN_{eval,train}_n10` 的 `packets` 正好 10 项且类型逐位等于 `[0,1,2,3,4,5,7,8,9,11]`、`loadout_context.seed_slot_count = 10`；`night_l20_{eval,train}_n10_orig` 是 10 个空槽（`type = -1`）+ `conveyor_start = original`，继续等 120 tick 后原生发卡（15 → 10 → 11 → 14），证明传送带开局正常。② 9 个新训练任务各 1 局（学生 `dw4i_s1`，`--arms fast`，`--workers 1`，种子 2200000 / 2210000 / … / 2280000 / 2290000）：8 个 n10 任务全胜（L11/L13/L16/L18 20 波、L12/L14/L17/L19 30 波；L13 一局用满 0–9 十个槽），`night_l20_train_n10_orig` 在第 7 波丢完 5 辆车告负 —— 与 task23 统一面板里 `night_l20_eval_n9_orig` 所有模型 0/128 一致，是学生在原版传送带上的既有短板，不是 reset/清单问题。

## 问题

- **L20 的 n10 族按 task23 的口径只出原版开局版**：任务书写的是“`night_lN_eval_n10` 与 `night_lN_train_n10`（N = 11–14、16–19；L20 同时出原版开局版 …）”，我按 task23 统一面板的做法（`night_l20_eval_n9_orig` 取代而非并列 L20 的旧语义任务）只生成 `night_l20_eval_n10_orig` / `night_l20_train_n10_orig`，不再出“卡组写进初始手牌”的 `night_l20_eval_n10`。n9 的 L20 旧任务保留在清单里（历史档案照旧重放）。如果主控要的是两版并存，加一个循环即可。
- **主清单是在 q101 运行中被覆盖的**：`~/PvZ-search/artifacts/stage4/night_panel.json` 是 q101 每轮读的任务清单。本次只**追加**任务、原有 36 个任务逐任务 JSON 逐字节不变（已对比），所以正在跑/后续轮次的 `task_id` 查询不受影响；没有动任何队列或进程。
- **`scripts/check_task_manifests.py` 在本工作树跑不起来**：它要 `artifacts/task_family/heldout.json`，这份工作树没有（与本次改动无关）。清单合法性改用生成器内置自查 + reset 探针覆盖。
- **L20 原版开局对学生仍是 0 分**：冒烟 1 局在第 7 波丢完车，与 task23 统一面板里所有模型 0/128 一致。本任务只确认 reset/发卡正常，不评估它的难度。
