# task22：白天泳池 L21–L30 原版难度评估面板

面板任务书：`MEMORY/docs/task22-pool-panel.md`。分支 `pool`（工作树 `.claude/worktrees/pool`）。
清单与脚本：`scripts/make_pool_panel.py`（薄包装）、`scripts/zone_panel.py`（按区生成的公共核心，
task21 的 `make_night_panel.py` 也是它的薄包装）；产物 `artifacts/stage4/pool_panel.json`（不进 git）。
数据在 `artifacts/pool_panel/{identity,teacher}_{control,p9,p9k}/l{level}/`（不进 git）。

## 1. 面板怎么定义的

每关一个评估任务 `pool_lN_eval`（L21–24、26–30，共 9 个；3-5 排除，见 §2），字段与白天/夜间面板逐项对齐：

| 字段 | 值 | 含义 |
|---|---|---|
| `terrain` | `pool` | 白天泳池（`pvz_constants.background_for_level` 现解析，21–30 全是 `BACKGROUND_3_POOL`） |
| `zombie_count_multiplier` | 1.0 | 原版僵尸数量 |
| `wave_cap` | null | 不限波，打到关卡自己的最后一波 |
| `deck` | `deck_for_level(level)` = `[0,1,2,3,4,5,16,43]` | 白天 6 卡 + 睡莲 16 + 香蒲 43，**没有改 `deck_for_level`** |
| `sun_start` | 50 | Adventure-II 的 reset 阳光值 |
| `preplanted` | `[]` | 不预置 |
| `playthrough` | 2 | replay 语义（唯一支持值）；reset 翻成 `mFinishedAdventure = 1` |
| `seeds` | 30000–31023 | 与白天/夜间面板同一评估段，与所有训练段不重叠 |

**波数**（`Board::PickZombieWaves`，playthrough=2 非 mini-boss 走 `gZombieWaves` 再 `+10`）：
L21 = 20 波（基础 10 → 20）；L22 / L23 / L26 / L28 = 30 波；L24 / L27 / L29 = 40 波；
L30 是 mini-boss，不 +10 → 30 波。

**地形**：6 行，row 2/3 是水路。非水生植物必须种在睡莲 16 上（`Board::CanPlantAt` 的
`PLANTING_NOT_ON_WATER`）；香蒲 43 只能种在睡莲上（`PLANTING_NEEDS_UPGRADE`）；
缠绕水草 19 是水生植物，可以直接种在水里（`SEED_TANGLEKELP` 与睡莲同一条分支）。
泳池没有坟头（`Board::AddGraveStones` 只对夜间背景 / 挑战模式生效）。

**老师 = identity 学生 + 搜索**（`~/PvZ-search/artifacts/student/identity_dw3_s1/model.pt`，
PVZRN006 走 C++ CPU 前向，不开 CUDA），参数照 `~/pvz-logs/queue101.sh` 的 `run()` 但
`--dagger-beta 1`（搜索实际执行）、`--arms fast_search`、`--workers 2`。
学生是同一份 identity 模型 `--arms fast`。每关老师 16 局、学生 64 局。

## 2. 特殊关：3-5 排除，3-10 收

1. **3-5（L25）是 Little Trouble**，不是普通关：`LawnApp::IsLittleTroubleLevel` 在 Adventure 模式
   对 `mLevel == 25` 为真；`Board::HasConveyorBeltSeedBank()` 因此为真 —— 没有卡组与阳光经济，
   僵尸点数 ×4（`Board::PickZombieWaves`），僵尸半尺寸、血量 1/4（`Zombie` 构造里的
   `IsLittleTroubleLevel` 分支）。和 2-5 打地鼠同类：卡组面板对它没有意义，**默认不收**
   （`make_pool_panel.py --include 25` 可强行收）。
2. **3-10（L30）是 mini-boss**（`IsMiniBossLevel` 含 `mLevel == 30`），
   `HasConveyorBeltSeedBank()` 为真。语义同白天 1-10 / 夜间 2-10：`RESET_V2` 把卡组写进
   初始 seed bank，之后是原生传送带（免费种植、消耗移位、补卡）。收进来，报告里标注，
   跑数时避开（§4 的 5 关不含 L30）。
3. 21–24、26–29 是普通白天泳池，没有小游戏。

## 3. 卡组与地形核对（AGENTS §8，生成时强制）

三套卡组只换卡组，其余字段逐项相同（同关卡、同种子段 30000–31023、倍率 1.0、不限波、
playthrough 2、阳光 50）。任务族互不覆盖：`pool_lN_eval`（对照）/ `pool_lN_eval_p9` /
`pool_lN_eval_p9k`，登记在 DESIGN §14.10。

| 卡组 | 任务族 | 内容 | 地形覆盖 | 升级植物 |
|---|---|---|---|---|
| 对照 | `pool_lN_eval` | `[0,1,2,3,4,5,16,43]`（`deck_for_level`） | 睡莲 16 + 香蒲 43 | 香蒲 43 → `profile_for_deck` 声明所有权 |
| p9 | `pool_lN_eval_p9` | `[0,1,2,3,4,5,7,16,43]`（学生训练卡组 + 睡莲 + 香蒲） | 同上 | 同上 |
| p9k | `pool_lN_eval_p9k` | `[0,1,2,3,4,5,7,16,19]`（学生训练卡组 + 睡莲 + 缠绕水草 19） | 睡莲 16（19 直接种水里） | 无升级植物 |

- 三套长度 8 / 9 / 9，均在 6..10；无重复；`terrain_covered` 全通过；香蒲 43 有睡莲底座。
  生成脚本在写清单前会跑这组检查，不合法直接报错（`zone_panel.check_decks`）。
- **解锁列只是“原版首轮奖励”的记录，不是运行约束**：`playthrough = 2` 时引擎
  `GetSeedsAvailable()` 返回 49，模拟器不拦。按 `LawnApp::GetAwardSeedForLevel`，
  L21 解锁到 17 号：香蒲 43 **不在奖励列表里**（它是商店升级植物，靠所有权而不是关卡解锁），
  缠绕水草 19 从 L23 起才在奖励列表内 —— 这两条都不影响本面板的运行合法性，
  因为 43 走 `owned_upgrade_plants`、19 在 replay 语义下可选。
- 3 张卡的对照/p9 差别是双发射手 7；p9/p9k 的差别是香蒲 43 对缠绕水草 19
  （持续远程输出 vs 一次性秒杀 + 占水路格）。

## 4. 结果

三套卡组 × 5 关（21、22、23、24、26，避开 L25 小游戏与 L30 传送带）。
学生每关 64 局、老师每关 16 局，同一批种子（offset 0，即 30000 起）。

### 4.1 identity 学生（`--arms fast`），64 局/关

| 关卡 | 波数 | 对照 | p9 | p9k |
|---|---|---|---|---|
| 3-1 (L21) | 20 | 0/64 | 10/64 | 8/64 |
| 3-2 (L22) | 30 | 0/64 | 4/64 | 4/64 |
| 3-3 (L23) | 30 | 0/64 | 3/64 | 4/64 |
| 3-4 (L24) | 40 | 0/64 | 0/64 | 2/64 |
| 3-6 (L26) | 30 | 0/64 | 1/64 | 1/64 |
| **合计** | | **0/320 (0%)** | **18/320 (5.6%)** | **19/320 (5.9%)** |

平均到达波数（对照 / p9 / p9k）：L21 11.44/13.91/13.78、L22 11.12/14.61/14.67、
L23 11.53/15.22/16.59、L24 11.02/13.08/15.58、L26 11.38/13.55/15.38。
平均丢车 1.17–1.55（对照 1.22–1.42）。

### 4.2 老师（identity 推演 + 搜索，`--dagger-beta 1`），16 局/关

（跑数中，见文末“结果”一节的最新一行。）

## 5. 交付状态（对照任务书）

| 任务书条目 | 状态 |
|---|---|
| 清单 `artifacts/stage4/pool_panel.json`（不进 git）+ 生成脚本 | 完成（`scripts/make_pool_panel.py`，逻辑在 `zone_panel.py`，3-5 排除） |
| 泛化而不是复制 `make_night_panel.py` | 完成（公共核心 `zone_panel.py`；夜间清单 `--combined` 输出与 task21 逐字节相同） |
| 三套卡组 × 5 关 × identity 学生 64 局 | 完成（§4.1） |
| 三套卡组 × 5 关 × 老师 16 局 | 跑数中 |
| DESIGN §14.10 登记 | 待写 |
| 分支 `pool` 提交 | 进行中（每小步一提交） |

## 6. 问题

1. **泳池关的解锁列会误报香蒲 43**：`GetAwardSeedForLevel` 不含升级植物，
   所以对照/p9 的 43 在 L21–L30 一律显示“未解锁”。这不是运行问题（所有权由
   `profile_for_deck` 声明、replay 语义允许），但报告里必须解释，否则会被读成非法卡组。
2. **L25（3-5）没有卡组语义**：Little Trouble 是传送带 + 全僵尸点数 ×4 + 半尺寸 1/4 血，
   与 2-5 打地鼠同类，本面板不收；要评估它需要另开“传送带 / 小游戏”任务族。
