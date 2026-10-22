# 2-10（夜间传送带关）为什么第 7 波就输

task26 的诊断报告。只分析，不训模型、不改环境。
工作树 `~/PvZ-search/.claude/worktrees/l20diag`（分支 `l20diag`）。
任务 `night_l20_eval_n10_orig`（2-10 原版传送带开局：`deck: []` + `conveyor_start: "original"`，开局 10 个空槽）。
学生 `~/PvZ-search/artifacts/student/dw4i_s1/model.pt`；老师 = 同一学生 + 搜索（`--dagger-beta 1`）。
原始数据（不进 git）：`artifacts/task26/student_16/`、`artifacts/task26/frames/`、`artifacts/task26/teacher_8/`。

## 1. 引擎事实（源码逐条核对）

### 1.1 关卡身份：mini-boss 传送带关

- `LawnApp::IsMiniBossLevel()`（`src/LawnApp.cpp:3011-3018`）：Adventure 且 `mLevel` ∈ {10, 20, 30} → 2-10 是 mini-boss 关。
- `Board::HasConveyorBeltSeedBank()` 因此为真：没有卡组与阳光经济，卡由 `Challenge::UpdateConveyorBelt` 发、种植免费。
- `Challenge::StartLevel()`（`src/Lawn/Challenge.cpp:540-544`）：mini-boss 关开局 `mZombieCountDown = 100`（第一只僵尸 tick 100）、`mConveyorBeltCounter = 200`（第一张卡 tick 200）。

### 1.2 波数：20 波（mini-boss 不 +10）

- `Board::PickZombieWaves()`（`src/Lawn/Board.cpp:590-595`）：`mNumWaves = gZombieWaves[mLevel-1]`；只有“非首轮且非 mini-boss”才做 `<10 ? 20 : +10`。
- `gZombieWaves[19] = 20` → 2-10 是 **20 波**（1-10 也是 20 波；普通夜间 30 波关的 +10 在这里不适用）。

### 1.3 僵尸点数：playthrough 2 + mini-boss ×3

- `Board::PickZombieWaves()`（`src/Lawn/Board.cpp:656-659`）：`IsAdventureMode() && HasFinishedAdventure()`（我们的 `playthrough = 2`）→ `aZombiePoints = aWave * 2 / 5 + 1`。
- 同一函数（`src/Lawn/Board.cpp:689-692`）：`IsMiniBossLevel()` → `aZombiePoints *= 3`。
- 旗波另有 `aZombiePoints *= 2.5` 且先塞 `min(points, 8)` 只普通僵尸 + 旗僵尸（`Board.cpp:661-680`）。
  按这个公式，第 w 波（0 起）的裸点数是 `(w*2//5+1)*3`：第 1 波 3 点、第 6 波 9 点、第 10 波（旗波）`(4*2.5)*3 = 30` 点、最后一波（第 20 波）`(8*2.5)*3 = 60` 点 + 13 只坟头僵尸。
  **2-10 的难度不是线性上升，而是旗波/末波陡增**。

### 1.4 僵尸名单：5 种

- `gZombieAllowedLevels`（`src/Lawn/Challenge.cpp:76` 起）按关卡查表；`Board::CanZombieSpawnOnLevel`（`src/Lawn/Board.cpp:2353-2368`）= 已到起始关 且 `mPickWeight != 0` 且该关允许。
- 第 20 关允许的：**普通(1)**、**路障(2，起始 L3)**、**铁门(4，起始 L13，第 5 波起)**、**橄榄球(7，起始 L16，第 5 波起)**、**舞王(5，起始 L18，第 5 波起)**；旗波直接塞旗僵尸（`mPickWeight = 0`，只由代码放）。伴舞由舞王召唤。
- 数值来自 `ZombieDefinition`（`src/Lawn/Zombie.cpp:81-113`）；L20 没有巨人、没有蹦极、不是 final boss。

### 1.5 坟头：13 个，最后一波开始 200 tick 后每个坟头各冒一只

- `Board::AddGraveStones`（`src/Lawn/Board.cpp:1134-1142`）：Adventure `mLevel >= 20` → 第 3 列 1 个、4 列 2 个、5 列 2 个、6 列 2 个、7 列 3 个、8 列 3 个 = **13 个坟头**（实测帧里 `graves = 13`）。
- `Board::NextWaveComing`（`src/Lawn/Board.cpp:4960-4963`）：进入最后一波时 `mRiseFromGraveCounter = 200`；`Board::UpdateGameObjects`（`5279-5284`）倒数到 0 调 `SpawnZombiesFromGraves()`。
- `SpawnZombiesFromGraves`（`src/Lawn/Board.cpp:4850-4886`）：遍历未清掉的坟头，每个各 `AddZombie(PickGraveRisingZombieType(), mCurrentWave)` 并 `RiseFromGrave`。
- `PickGraveRisingZombieType`（`src/Lawn/Board.cpp:2388-2413`）：只在普通(权重 4000)与路障(4000)之间按权重抽；有坟头的关卡不加铁桶。

### 1.6 传送带：发卡权重表与间隔

- 权重表（`src/Lawn/Challenge.cpp:1683-1699`），**2-10 的 7 种卡**：

| 卡 | 权重 | 说明 |
|---|---:|---|
| 墓碑吞噬者 11 | 20 | 只在坟头数 > 已种+在带上时才保留权重（`Challenge.cpp:1869-1878`），最多 13 张 |
| 寒冰菇 14 | 15 | 一次性冻结 |
| 毁灭菇 15 | 15 | 一次性清场 |
| 催眠菇 12 | 10 | 一次性策反 |
| 胆小菇 13 | 15 | 远程，但僵尸靠近就缩 |
| 大喷菇 10 | 15 | 4 格穿透 |
| 小喷菇 8 | 10 | 0 费、3 格 |

  对照 **1-10** 的表（`Challenge.cpp:1665-1682`）：豌豆 20、樱桃 20、坚果 15、双发 20、寒冰射手 10、大嘴花 5、土豆雷 10 —— 有豌豆/双发/坚果这种“站桩持续输出 + 挡路”的骨架。

- 发卡间隔（`src/Lawn/Challenge.cpp:1659-1661`）：`mConveyorBeltCounter = 1 * (on_belt > 8 ? 1000 : > 6 ? 500 : > 4 ? 425 : 400)` 帧。
  即：带上 ≤4 张 400 帧、5–6 张 425、7–8 张 500、>8 张 1000。传送带容量 10，满了 `SeedBank::AddSeed` 直接丢弃新卡（`src/Lawn/SeedPacket.cpp:1018-1029`）。
- 传送带卡免费：`SeedPacket::CanPickUp` 在传送带关跳过阳光检查（task20 已核对），所以阳光经济在这里不存在（夜间本来也不掉阳光）。

### 1.7 小结（引擎层）

2-10 是一个 **20 波、僵尸点数 ×3、13 个坟头（末波 +13 只）、只能用 7 种蘑菇/灰烬卡且没有稳定远程火力** 的关。
它不是“卡组没带对”（传送带不用卡组），也不是“环境把传送带覆盖了”（原版开局 10 空槽、原生发卡已实测）。

## 2. 对局事实

### 2.1 学生 16 局：0 胜，一张卡都没种

`night_l20_eval_n10_orig`，种子 30000–30015，`--arms fast`（学生 `dw4i_s1`），`--workers 2`，逐决策录帧。

| 指标 | 结果 |
|---|---|
| 胜负 | **0/16** |
| 结束波数 | 6–8（均值 6.88） |
| 每局丢车 | **5/5**（每行各一次） |
| 种植数 | **0**（16 局合计 0 株） |
| 阳光 | 全程 50（没有阳光菇/向日葵，也不掉阳光） |

**发卡**（16 局合计 / 每局平均）：墓碑吞噬者 28 / 1.75、毁灭菇 29 / 1.81、寒冰菇 28 / 1.75、
胆小菇 25 / 1.56、催眠菇 21 / 1.31、大喷菇 19 / 1.19、小喷菇 10 / 0.63。
第一张卡固定 tick 240（计数器 200 + 60 tick 帧间隔）；**tick ≈4085（16 局 4058–4105）传送带就满 10 张**，之后新卡被丢弃。
也就是说整局只发了约 10 张卡，而学生一张都没用。

**丢车**：80 次全部发生在第 2–6 波（2 波 15 次、3 波 26、4 波 23、5 波 10、6 波 6），
触发者按类型：普通 62、路障 47、伴舞 1 —— **没有一次是橄榄球/铁门走到房子**，因为游戏在它们到达前就结束了。
每行各丢一次，5 行全丢 → 结束。

**为什么一张都没种**：帧里每个决策都带 `action`，16 局 4415 个决策**全部是 `wait`**，且 `action_rejected = false` ——
不是“选了非法动作被拒”，是策略自己一直选择等。直接读学生网络的 `plant` 门（`pvz_rollout_policy.act` 的第一道判断，
`python/pvz_rollout_policy.py:212`：`if float(net.plant(g)[0,0]) <= 0: return WAIT`）：

| 状态 | plant 门 | 结果 |
|---|---:|---|
| L20 原版传送带，tick 240（带上 1 张墓碑吞噬者） | **−0.32** | wait |
| 同上，tick 720 / 1560 / 2520 | −0.72 / −0.43 / −1.19 | wait |
| 同上，tick 3000–4440（带上 7–10 张） | −1.7 … −2.3 | wait |
| **对照：1-10 原版传送带**（同一模型 `dw4i_s1`），tick 240/720/1560/2520 | **+4.03 / +6.74 / +8.65 / +8.48** | 会种 |
| **对照：L20 固定卡组**（`night_l20_eval_n9`，9 张写进卡槽） | **+1.86** | 种了 |

所以：
1. 合法动作是有的 —— reset 后 tick 240 就有 13 个合法种植格（探针实测 `legal_actions.plants = 13`），
   所以不是环境/观测把动作挡住了。
2. 学生不是“不会用传送带”这么笼统 —— 同一个模型在 **1-10 原版传送带**上 plant 门是正的、会种（task20 里 identity 学生 256/256）。
3. 它在 **2-10 这个“空槽 + 蘑菇池”的状态**上把“种不种”判成不种，并且一直不种到输。

### 2.2 卡池与训练数据的覆盖差

本分支所有任务清单（night / day / pool）里出现过的卡：`0,1,2,3,4,5,7,8,9,11,16,19,43`。
2-10 传送带池的 7 种卡里，**大喷菇 10、催眠菇 12、胆小菇 13、寒冰菇 14、毁灭菇 15 从未作为任务卡组出现过**
（0 个任务），只有小喷菇 8（夜间 n8/n9/n10）和墓碑吞噬者 11（n10）出现过。
identity 头按“植物身份”出卡，这 5 种卡在学生的训练分布里几乎是空白；1-10 的池（豌豆/双发/坚果/寒冰射手/大嘴花/土豆雷/樱桃）
反而大多是它熟悉的卡。这与 2.1 的对照（L10 门正、L20 门负）方向一致。
