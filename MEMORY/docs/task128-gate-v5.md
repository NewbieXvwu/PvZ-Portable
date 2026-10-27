# task128：门禁 v5（冒险模式实际关卡与卡组）

分支 `gatev5`（基于 origin/selheldout 03620a84）。工作树
`/home/newbiexvwu/PvZ-search/.claude/worktrees/gatev5`（主检出停在 `search-wt`，
未动）。

## 为什么换面板

第 12–22 轮 11 次否决里 9 次含 heldout 否决，heldout 每轮退步 −3 到 −42pp，
总体不过关也主要被它拉下。崩塌集中在一个合成卡组任务
`day_l7_eval_d7p_holdout1`（`[0,1,2,27,5,32,3]`，白天三叶草 + 卷心菜输出）。
冒险模式每关用自定卡组，这种组合不会出现；目标顺序是"先通关、后泛化"。

## 面板推导（可复现）

`scripts/make_gate_v5_panel.py`（`--check` 逐字节校验）读三份冻结来源：

| 来源 | SHA256 | 用途 |
|---|---|---|
| `configs/adventure_scoreboard.json` | `5213bf9f…46cc` | task103 冻结的 45 关定义（1.0 倍、记分板卡组） |
| `MEMORY/docs/adventure-scoreboard-r11-32.json` | `de0c68c1…b288` | r11 学生逐关 32 种子胜局 |
| `MEMORY/docs/task115-teacher-arm-weak-levels.md` | `2db7763c…5bb1` | 老师臂逐关 16 种子胜局 |

入选规则：学生 ≥1 胜 **或** 老师 ≥1 胜。结果 36 关：

| 区 | 关 | 每关种子 |
|---|---:|---:|
| day | 1,2,3,4,6,7,8,9,10 | 48 |
| night | 11,12,13,14,16,17,18,19,20 | 48 |
| pool | 21,22,23,24,26,27,28,29,30 | 48 |
| fog | 31,32,33,34,36,37,38,39,40 | 48 |
| heldout（只监测） | `day_l7_eval_d7p_holdout1`、`day_l9_eval_d7p_holdout2` | 96 |

合计 36×48 + 2×96 = 1920 对 / 3840 局（v4 同量）。L30/L34/L39 学生 0/32、
老师 1–3/16，按老师胜局入选；L30 与 L40 是原生传送带（空卡组）。

## 屋顶区的缺口（已知，需要主控裁定）

L41–44、46–49 学生 0/32；task115 对屋顶只给聚合值"0–1/16"（没有逐关数字），
L50 Boss 0/16。按"全输的关不进门禁"，这 9 关都不入选，因此 **v5 的否决族只有
4 个区，没有屋顶区**。v4 的屋顶否决来自合成格 `roof_l41_train_r7`（第 20 轮
harm=6、gain=18），它不是冒险记分板的屋顶关。恢复屋顶否决需要 task115 的逐关
老师胜局，或另立屋顶探针任务。

## 决策与执行

- 总体：冒险对上的 Newcombe 配对 95% 下界 > −3pp（1728 对）。
- 区否决：4 个冒险区的单侧 exact McNemar + Holm–Bonferroni（族错误率 0.05）。
- heldout：照常跑 96 对，进 `decision.json.monitor` 与轮日志，不否决、不进总体。
- 种子：门禁账本每轮一个 1920 连续种子块（有序池，用完不回头），与记分板
  103M 段、挑选 130M 池、训练/采集清单不相交。
- 旧版本：v1–v4 的 policy 文本、冻结面板与证书全部保持可验（`test_promotion` 锁定）。

## 挑选面板对齐

主体换成同一批 36 冒险关（独立种子，41 任务 × 9 局 = 369 局/候选）；三个随机族
保留；task126 的两个换卡组任务标 `score_only`，只记分不进总分；30%–80% 采集
窗口筛选退役（`current_cells` 删除），挑选不再读采集窗口。

## 部署

`scripts/qc_deploy14i.sh`（只交主控执行，复制到台式机 `~/pvz-logs/`，未执行）。
切换条件同 v4：没有已排队/运行的门禁任务即可。未激活/回滚分支按
`~/pvz-logs/qc_deploy14h.mode` 判断当时在役是 deploy14h 还是 deploy14g，参数与
代码 baseline 都分两套校验；激活写 `qc_deploy14i.mode=gatev5`，回滚删掉它即回到
上一套，state/offset/账本保留。
