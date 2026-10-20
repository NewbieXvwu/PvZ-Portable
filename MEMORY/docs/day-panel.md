# task17：白天 L1–L10 原版难度评估面板

面板任务书：`MEMORY/docs/task17-day-panel.md`。分支 `daypanel`（工作树 `.claude/worktrees/daypanel`）。
清单与脚本：`scripts/make_day_panel.py`、`scripts/eval_panel.py`；
产物 `artifacts/stage4/day_panel.json`（6 卡）与 `artifacts/stage4/day_panel_d7.json`（7 卡），都不进 git。

## 1. 面板怎么定义的

每关一个任务 `day_lN_eval`（7 卡版是 `day_lN_d7_eval`），字段照 `artifacts/stage4/day_l7_hard.json`
里 `day_l7_*_eval` 的写法：

| 字段 | 值 | 含义 |
|---|---|---|
| `zombie_count_multiplier` | 1.0 | 原版僵尸数量，不做加难 |
| `wave_cap` | null | 不限波，打到关卡自己的最后一波 |
| `deck` | `deck_for_level(level)` 或 `--deck` | 6 卡版直接调 `scripts/pvz_tasks.py`，白天草地 = `[0,1,2,3,4,5]`（豌豆、向日葵、樱桃、坚果、土豆雷、寒冰射手）；7 卡版 `--deck 0,1,2,3,4,5,7`（加双发射手），**`deck_for_level` 本身没改** |
| `sun_start` | 50 | Adventure-II 的 reset 阳光值；`RESET_V2` 不覆盖它（`scripts/check_task_manifests.py` 的 `sun_start_semantics`） |
| `preplanted` | `[]` | 不预置植物 |
| `playthrough` | 2 | replay 语义（唯一支持的值）。reset 里翻成 `mFinishedAdventure = 1`，所以所有基础植物都可选；不是“第一次冒险”的教程流程 |
| `seeds` | 30000..31023 | 与 `day_l7_*_eval` 同一段评估种子，和训练段（1600000+ / 1500000+）不重叠 |

学生每关 256 局（种子段前 256 个），老师每关 32 局。老师的参数 = 学生 v32 + 搜索，
flags 照 `artifacts/hard_mac/round15_flags`：`--verify-samples 9 --value artifacts/hard_mac/value_v3/model.pt
--static-value artifacts/hard_mac/value_branch1_desk/model.pt`，其余搜索参数用 `search_play.py` 默认值。

**波的形状**（`Board::PickZombieWaves`，playthrough=2 非 mini-boss 走 `gZombieWaves` 再 `+10`）：
1–4、6、8 关 20 波，7、9 关 30 波，10 关是 mini-boss（`LawnApp::IsMiniBossLevel`）20 波。

## 2. 面板本身先说清楚的三件事

1. **1-5 不收**：`LawnApp::IsWallnutBowlingLevel` 在 Adventure 模式把 `mLevel == 5` 当坚果保龄球，
   没有卡组、没有阳光经济。任务书明确排除。
2. **1-10 能跑，但不是原版传送带**：`IsMiniBossLevel` 含 `mLevel == 10`，所以
   `Board::HasConveyorBeltSeedBank()` 为真；但 `RESET_V2` 仍然把卡组写进 seed bank
   （`LawnApp::EnvironmentReset` 末尾的 `SetPacketType(deck[i])`）。实测开局 `packets` 就是
   `[0..5]`、600 tick 后仍是这几张、阳光 50 不动。也就是说这一关是“L10 的波次脚本 + 固定卡组”，
   **不是**原版传送带的发牌规则。结果保留，读的时候要知道这一点。
3. **1-1 与 1-2 在环境里是同一局**：两关都是 20 波、只允许普通僵尸，同种子逐局结果 256/256 完全一致。
   面板实际只有 8 个不同场景。

## 3. 卡组核对（只报告，没改 `deck_for_level`）

- 地形：1–10 全是 `BACKGROUND_1_DAY`（`Board::PickBackground`），没有泳池/屋顶/夜晚混进来。
- 解锁：原版解锁顺序由 `LawnApp::GetAwardSeedForLevel` 给出（第 L 关奖励的种子编号就是已解锁上界）。
  按它算，1-1 只到向日葵，1-2 到樱桃，1-3 到坚果，1-4 到土豆雷，1-8 到双发射手。
  所以 **6 卡版在 1-1..1-4 带了解锁前的植物**（1-1 缺 2/3/4/5，1-2 缺 3/4/5，1-3 缺 4/5，1-4 缺 5）；
  **7 卡版在 1-1..1-7 都带双发射手**。模拟器不拦这个：`playthrough=2` 时 `GetSeedsAvailable()`
  直接返回 49（`HasFinishedAdventure()` 为真），`HasSeedType` 对基础植物一律放行，reset 不会失败。
  这是任务设定与原版解锁的偏差，不是运行错误。
- 训练配方里的白天任务（`day_l7_*_train/eval`、task family）都是 7 张卡、含双发射手，
  所以 6 卡版本身就是一个**卡组泛化**测试，不是“同一任务换难度”。

## 4. 卡组泛化：6 卡（`deck_for_level`，每关 256 局）

学生 v32 在这副没训练过的 6 卡卡组上：

| 关卡 | 卡组 | 胜局/256 | 胜率 | 平均丢车 | 平均到达波数 |
|---|---|---|---|---|---|
| 1-1 | [0,1,2,3,4,5] | 238 | 0.930 | 0.48 | 19.98 |
| 1-2 | [0,1,2,3,4,5] | 238 | 0.930 | 0.48 | 19.98 |
| 1-3 | [0,1,2,3,4,5] | 214 | 0.836 | 1.05 | 19.90 |
| 1-4 | [0,1,2,3,4,5] | 211 | 0.824 | 1.05 | 19.90 |
| 1-6 | [0,1,2,3,4,5] | 194 | 0.758 | 1.18 | 19.88 |
| 1-7 | [0,1,2,3,4,5] | 34 | 0.133 | 2.35 | 26.75 |
| 1-8 | [0,1,2,3,4,5] | 86 | 0.336 | 2.17 | 19.34 |
| 1-9 | [0,1,2,3,4,5] | 6 | 0.023 | 2.64 | 23.75 |
| 1-10 | [0,1,2,3,4,5] | 28 | 0.109 | 3.41 | 13.58 |

读法：1-1..1-6 的损失主要来自“少一张双发射手 + 后期波次输出不够”，不是关卡机制；
1-7..1-10 掉到 2%–34%，其中 1-7、1-9 是 30 波（比其它关多 10 波），1-9 只有 6/256。

老师（v32 + 搜索，`round15_flags`）在这副 6 卡卡组上只跑了 1-1、1-2 就按主控要求停掉：
两关都是 **32/32**、平均丢车 0.53、平均 20 波。也就是说 6 卡下这两关学生 238/256 与老师 32/32
的差距不是能力上限问题（学生本来就接近满分），真正的差距要看后面的关。

## 5. 7 卡（`day_panel_d7.json`，正式跑数在台式机）

7 卡版清单已生成：`python3 scripts/make_day_panel.py --deck 0,1,2,3,4,5,7 --suffix _d7
--out artifacts/stage4/day_panel_d7.json`，任务名 `day_lN_d7_eval`，种子段与 6 卡版相同（30000..31023）。
正式跑数由主控在台式机做（20 线程），本机不再跑大评估。结果路径回来后补这张表：

| 关卡 | 卡组 | v32 胜局/256 | d35rows2 胜局/256 | 老师 胜局/32 | 丢车 | 到达波数 |
|---|---|---|---|---|---|---|
| 1-1 … 1-10 | [0,1,2,3,4,5,7] | 待台式机 | 待台式机 | 待台式机 | | |

命令（台式机，清单路径换成它自己的工作树）：

```
python3 scripts/eval_panel.py --rollout-policy artifacts/student/v32/model.pt \
    --value artifacts/hard_mac/value_v3/model.pt --manifest artifacts/stage4/day_panel_d7.json \
    --out artifacts/day_panel/v32_d7 --seeds 256 --workers 20

python3 scripts/eval_panel.py --rollout-policy artifacts/student/v32/model.pt \
    --value artifacts/hard_mac/value_v3/model.pt --static-value artifacts/hard_mac/value_branch1_desk/model.pt \
    --verify-samples 9 --arms fast_search --manifest artifacts/stage4/day_panel_d7.json \
    --out artifacts/day_panel/teacher_d7 --seeds 32 --workers 20
```

## 6. 哪几关学生明显不如老师

等 7 卡版台式机结果（6 卡版只跑到 1-1、1-2，老师都是 32/32，还看不出差距）。按 6 卡版的学生曲线，
预期重点在 1-7、1-8、1-9、1-10：这几关 6 卡下已经掉到 2%–34%，是最值得先采集的关卡。

## 7. 问题

- **d35rows2 取不到**：HF `transfer/` 下没有 `d35rows2*`（2026-10-20 10:5x 复查，1278 个文件里
  `rows`/`d35` 均无命中），本机 `artifacts/student/` 也没有。它是在台式机 rowpool2 工作树训的，
  还没同步。它一旦上传，用同一份清单重跑即可（`games.jsonl` 会续跑，不会重复已完成的种子）。
- 1-10 的传送带语义见 §2.2：环境能跑，但跑的不是原版传送带。
- 老师每关 32 局只够看上限，不足以给出胜负的置信区间（32 局的 ±1 局 ≈ ±3%）。
