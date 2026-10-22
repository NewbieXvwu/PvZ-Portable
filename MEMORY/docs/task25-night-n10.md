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

## 问题
