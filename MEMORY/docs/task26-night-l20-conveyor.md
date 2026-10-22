# task26（主控 2026-10-22 下发给台式机 CodeBuddy 第二条线）：2-10（夜间传送带关）为什么第 7 波就输

先读 `AGENTS.md`、DESIGN §14.8–§14.9、`MEMORY/docs/task20-conveyor-fidelity.md`、`MEMORY/docs/night-panel.md`、`MEMORY/docs/task25-night-n10.md`。

## 背景
原版开局的 2-10（`night_l20_eval_n9_orig` / `night_l20_*_n10_orig`，开局空、卡由引擎 `Challenge::UpdateConveyorBelt` 发）在统一面板上所有模型 0/128，冒烟在第 7 波丢完车。
白天原版 1-10 学生 256/256。只做分析，不训模型、不改环境。

## 要做的
工作区：`~/PvZ-search/.claude/worktrees/l20diag`（分支 `l20diag`，主控已建好），只在这里写脚本与报告。
1. **引擎事实**（读源码，写清行号）：2-10 的传送带发卡权重表与间隔（对照 1-10 的表）、波数、僵尸名单、坟头数与最后一波冒僵尸、有无别的特殊规则（mini-boss 关）。
2. **对局事实**：学生 `~/PvZ-search/artifacts/student/dw4i_s1/model.pt` 跑 `night_l20_eval_n10_orig` 16 局（`--workers 2`，录帧 `--frames` 到本工作树），统计：每局发到的卡（种类与时刻）、学生实际种了什么 / 留着没种什么、
   车在第几波因哪路什么僵尸丢掉、阳光与坟头。再用老师（同学生 + 搜索，参数照 `~/pvz-logs/queue103.sh` 的 `run()`，但 `--dagger-beta 1`）跑 8 局对比。
3. **结论**：输在哪（发卡结构 / 学生不会用这些卡 / 老师也不会 / 别的），给出下一步建议（采集、卡组无关，因为传送带不用卡组）。
4. 报告写 `MEMORY/docs/night-l20.md`；提交、`git push origin l20diag`，在本文“结果”记一行，然后停下。

## 规则
不动任何 tmux 队列与别的代理会话；不改别的工作树；同时最多 2 个游戏进程；不改默认值；不写手工规则；不写字面密钥；不 `git add -f`；`.pt/.npz/.log` 不进 git。只用绝对路径。

## 结果

- 2026-10-22 第一步（引擎事实 + 学生 16 局）：读源码核对 —— 2-10 是 mini-boss 传送带关（`IsMiniBossLevel`，10/20/30），`gZombieWaves[19] = 20` 波（mini-boss 不 +10），僵尸点数 `aWave*2/5+1` 后 ×3、旗帜波 ×2.5 再补 `min(points,8)` 只普通僵尸 + 1 只旗帜僵尸，僵尸名单 5 种（普通/路障/铁门 ≥5 波/橄榄球 ≥5 波/舞王 ≥5 波），13 个坟头、最后一波开始 200 tick 后每个坟头冒一只（只普通/路障），传送带按剩余卡数 400/425/500/1000 帧发一张、容量 10 满了丢弃（权重表逐项对照 1-10）。学生 `dw4i_s1` 在 `night_l20_eval_n10_orig` 上 16 局（种子 30000–30015，`--workers 2`，录帧）：**0/16 胜、平均 6.88 波、每局 5/5 丢车、种植 0 株**；4415 个决策全是 `wait` 且 `action_rejected` 全 false。直接读 plant 门：L20 传送带状态 −0.32…−2.3（越到后面越负），对照 1-10 原版传送带 +4.03…+8.48、L20 固定卡组 `night_l20_eval_n9` +1.86 —— 不是环境挡住动作（tick 240 就有 13 个合法种植格），也不是不会用传送带。脚本 `scripts/analyze_l20_conveyor.py`，数据 `artifacts/task26/student_16/`（不入 git）。
- 2026-10-22 第二步（老师 8 局 + 结论 + 报告）：同任务同种子 30000–30007、同学生 + 锚定搜索（`--verify-samples 9 --student-candidates 2 --static-value value_branch1 --value value_net/v3 --dagger-beta 1`，`--workers 2`）：**6/8 胜（75%）**，平均 19.88 波、平均丢车 2.5、平均每局种 62.5 株、搜索 166 次、覆盖学生 61.5 次（**492 次覆盖里 490 次是 `plant_over_wait`**）；两局败局在 19、20 波。结论写进 `MEMORY/docs/night-l20.md` §2.3–§4：输因不是发卡结构（老师同发卡 75% 胜）也不是不会用传送带（L10 门正），而是 7 种卡里 5 种（10/12/13/14/15）从未进过任何任务卡组、`packet_head: identity` 在这些身份上把 plant 门压成负值 → 全程 wait。建议：先按 `night_l20_train_n10_orig`（2290000 起）采 32 局 DAgger（723 s/局、2 并发约 3.2 h，判据只看“会不会种”），另开新任务族把 10/12/13/14/15 放进固定卡组补覆盖；不动 L20 难度。

## 问题
