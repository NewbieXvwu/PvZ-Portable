# CUDA 阶段 1 语料：真实根局面 + CPU 参照续走

产物由 `scripts/cuda_root_corpus.py` 生成，**全部落在 `artifacts/` 下，按 AGENTS.md §1 不入库**（大文件走
`scripts/hf_sync.py`，见 §2）。本文只记录：怎么复现、落在哪里、校验数字、以及阶段 1 开跑前还缺什么。

基线：`git HEAD = b526e82`，工作区另有**未提交**改动 `cuda/CudaRollout.h`、`cuda/SimHost.cpp`、
`cuda/spec/host.md`（用户同时在改，本文不依赖也未修改它们）。导出器 `src/CudaExport.cpp` 与记录格式
**一个字节没动**。

---

## 1. 任务范围与合规性

| 项 | 值 |
|---|---|
| 关卡 | 冒险模式第 7 关（白天草地，`terrain=day`） |
| 卡组 | `[0, 1, 2, 3, 4, 5, 7]` = 豌豆 / 向日葵 / 樱桃 / 坚果 / 土豆雷 / 雪花豌豆 / 双发射手 |
| `zombie_count_multiplier` | 1.3 / 1.4 / 1.5 各一份 |
| 任务清单 | `~/PvZ-search/artifacts/stage4/day_l7_hard.json`（只读） |
| 任务 id | `day_l7_z15_train` / `day_l7_z13_train` / `day_l7_z14_train` |
| 学生 | `~/PvZ-search/artifacts/student/v14/model.pt` |
| 值网 | `~/PvZ-search/artifacts/value_net/v2/model.pt` |

卡组从任务清单解析，不手抄（`render_episode.load_training_task`）。第 7 关是白天草地，无水路、无屋顶格，
所以 AGENTS.md §8 的地形覆盖约束天然满足：不需要睡莲(16)/香蒲(43)/花盆(33)，卡组里也就没有升级植物，
`owned_upgrade_plants` 与槽位数（= 卡组长度 7，`LawnApp.cpp:1329` 的校验）由 `scripted_baseline.profile_for_deck`
自动给出。

---

## 2. 复现命令行

三份语料**串行**跑（每台 1 个模拟器子进程 + 1 个 Python 进程，脚本内部已把 torch 与 BLAS 钉成单线程）。
本机同一时间有 `~/pvz-logs/queue35.sh` / `queue36.sh` 两条生产 `search_play.py --workers 18` 在跑，
所以额外加 `taskset -c 2-5`（≤4 核）与 `nice -n 10`，并置 `CUDA_VISIBLE_DEVICES=""` 强制 CPU。

```bash
cd ~/pvz-cuda
PY=/home/newbiexvwu/.venvs/ml/bin/python3      # 系统 python3 没有 torch，必须用这个 venv
RUN="day_l7_z15_train 24 corpus_l7_z15"        # <task-id> <seeds> <输出目录名>

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
nice -n 10 taskset -c 2-5 $PY scripts/cuda_root_corpus.py \
    --manifest ~/PvZ-search/artifacts/stage4/day_l7_hard.json \
    --task-id day_l7_z15_train --seeds 24 --roots-per-game 18 --rollouts-per-root 8 \
    --rollout-policy ~/PvZ-search/artifacts/student/v14/model.pt \
    --value ~/PvZ-search/artifacts/value_net/v2/model.pt \
    --out-dir ~/pvz-cuda/artifacts/cuda/corpus_l7_z15
```

另外两份只换三个参数（其余同上）：

| `--task-id` | `--seeds` | `--out-dir` |
|---|---|---|
| `day_l7_z13_train` | 8 | `~/pvz-cuda/artifacts/cuda/corpus_l7_z13` |
| `day_l7_z14_train` | 8 | `~/pvz-cuda/artifacts/cuda/corpus_l7_z14` |

`--seeds N` 取该任务自己种子表的**前 N 个**（`--seed-offset` 可往后挪）。一个种子（=一局真棋）平均出
16~18 个根，因为循环还有 `--roots-per-game 18` 与 `--max-root-tick 24000` 两道闸。
加根就加大 `--seeds`，**不要**加大 `--rollouts-per-root`（参照值是给每个根配 8 个不同的随机未来，
分布才有意义）。

实测资源（`/usr/bin/time -v`）：z15 那份 wall 73.6 s / 峰值 RSS 674 MB，z13、z14 各 wall 25.7 s / RSS 674 MB。
GPU 全程未使用。

---

## 3. 落盘位置与体积

| 目录 | `roots.cudaroot` | `corpus.jsonl` | `roots.json` |
|---|---|---|---|
| `artifacts/cuda/corpus_l7_z15/` | 7,509,632 B | 1,571,979 B | 93,826 B |
| `artifacts/cuda/corpus_l7_z13/` | 2,546,048 B | 535,144 B | 32,120 B |
| `artifacts/cuda/corpus_l7_z14/` | 2,546,048 B | 535,125 B | 32,120 B |

三者合计 15 MB（`du -ch`），全部在 `artifacts/` 内（`artifacts/.gitignore:2` 的 `*` 命中，
`git check-ignore --no-index` 已验证）。旧的冒烟语料 `artifacts/cuda/corpus_smoke/`（12 根 / 48 条）保留未动。

一份 `.cudaroot` = 1 个 `LevelInfo` + N 个 `RootRecord`（`magic=PVC4`、`version=4`）。因为 `LevelInfo` 只有一个，
**一个文件只能装一个任务**的根：关卡、卡组、`zombie_count_multiplier` 都写在那一条里。想要 1.3/1.4/1.5
全覆盖就必须是三个目录，不能合并成一个文件。

---

## 4. 校验数字（`python/pvz_cuda_root.py` 解析，纯 CPU）

先跑 `scripts/cuda_layout_check.py`：`layout OK: root 17652 bytes, level 2420 bytes, 6 entity record kinds,
magic 50564334 v4` —— 手工镜像的 Python 布局与编译器 `offsetof` 一致，所以下面的解析值可信。

| 语料 | 记录数 | `supported=1` | `supported=0` | `unsupportedReason` 直方图 | 根 tick 区间 |
|---|---|---|---|---|---|
| `corpus_l7_z15` | 425 | **425** | 0 | `{}`（空） | [1200, 24300] |
| `corpus_l7_z13` | 144 | 144 | 0 | `{}` | [3000, 18600] |
| `corpus_l7_z14` | 144 | 144 | 0 | `{}` | [3000, 18600] |
| 合计 | **713** | **713** | **0** | —— | [1200, 24300] |

713 ≥ 200，且**没有一个根被判为不支持**，所以 `unsupportedReason` 直方图是空的 —— 这不是"没测到"，
而是本任务范围本来就在内核实现的那个子集里（`KernelSupportsPlant` 恰好覆盖这张 7 张卡组，
`KernelSupportsZombie` 覆盖第 7 关实际会出场的 4 种僵尸，见下表）。

其它校验：

- **连接完整性**：`roots.json` 的 tick 序列与 `.cudaroot` 解析出的 tick 序列逐位相同；`supported` 标记同样逐位相同。
  `corpus.jsonl` 的 `root` 下标覆盖 0..N-1 无缺无重，**每个根恰好 8 条**参照续走 → 共 **5704 条 (root, future) 对**
  （z15 3400 / z13 1152 / z14 1152）。
- **状态多样性**：z15 里根 tick 的直方图（3000 宽）= `{0:8, 3000:48, 6000:73, 9000:69, 12000:83, 15000:90, 18000:35, 21000:17, 24000:2}`；
  `wave` 取到 0..10；不同的 (植物数, 僵尸数) 组合 114 种。根里出现过的植物类型 `{0,1,2,3,4,7}`，
  僵尸类型 `{0,1,2,3}` = 普通 / 旗 / 路障 / 撑杆 —— 全在 `KernelSupportsZombie` 白名单内。
- **参照续走的分布**（z15）：`ticks` 中位 2520、最大 2559；`decisions` 中位 47、**最大 52**；
  `net_calls` 中位 33；终局 8 条（全是输，`result=-1`），其余 3392 条未到终局；`leaf_value` 0.0000–0.9991（中位 0.3617）。
  z13/z14 全部未到终局（根 tick 较低、2500 tick 打不完一关）。
- **槽位余量**（为什么没有 `supported=0`）：植物 25/60、僵尸 12/64、子弹 9/96、金币 1/24、
  收入事件 34/160、修剪机 5/6、种子槽 7/10。**离每个上限都很远**，所以 1000/2000/3000/4000/5000 这一组
  "容量类" 不支持原因在本任务范围内不可能触发。
- **第 6 行**：所有 20400 个 `row_min_zombie_x` 元素里第 5 行永远是 `INT32_MIN`（草地只用水陆 5 行，
  与 `LevelInfo.playable_rows=5` 一致），这是 `_row_min_zombie_x` 写对了的旁证。

---

## 5. `corpus.jsonl` 字段 ↔ `cuda/CudaRollout.h` 的 `RolloutResult`

按 GPU 侧那一列的名字对齐，免得比较时错列：

| `RolloutResult` | `corpus.jsonl` | 来源 |
|---|---|---|
| `rootIndex` | `root` | 本文件内的根下标 |
| —— | `seed`, `future` | 连接键；`future = seed * 1000 + 序号`，就是 `RESEED` 的参数 |
| `ticks` / `endTick` | `ticks` / `end_tick` | ROLLOUT 响应的 `ticks`；观测里的 `tick` |
| `decisions` / `netCalls` | `decisions` / `net_calls` | ROLLOUT 响应同名字段；语料里 `net_calls` 恒 ≤ `decisions`（z15 中位 33 / 47），说明 CPU 路径上有些决策没跑前向 |
| `result` | `result` | **归一化成 0/1/-1**：观测直接吐 `mBoardResult`，输是 2（`src/ConstEnums.h:128`），而 GPU 报 -1 |
| `sun` / `wave` | `sun` / `wave` | 叶子绝对值 |
| `plantsAlive` / `zombiesAlive` | `plants` / `zombies` | 叶子观测实体数 |
| `mowersLeft` | `mowers_left` | ROLLOUT 响应 `mowers_left`（**仅 READY**） |
| —— | `mowers_alive` | 观测 `defenses` 条数（含已触发未死亡的），和上一个是两个数，语料里 155/5704 条记录两者不等 |
| `plantsPlanted` / `plantedByType` | `plants_planted` / `planted_by_type` | ROLLOUT 响应 `planted[10]`，槽位与 `kPacketSlots=10` 对齐 |
| `rowMinZombieX` | `row_min_zombie_x` | 定点 x4，空行 `INT32_MIN`（口径抄 `StoreResult`） |
| —— | `income`, `leaf_value` | 6000 tick 阳光收入率；`--value` 的 sigmoid V(叶子)（终局取 1/0） |
| `zombiesKilled` / `plantsEaten` / `mowersTriggered` / `sunProduced` | **缺** | 见 §6.1 |
| `errorFlags` | 不适用 | CPU 侧没有对应错误位 |

`leaf_value` 的 context（`zombie_count_multiplier` + 僵尸名册）是按任务取常量、在 `reset` 的**完整**观测上
算一次复用的（`observation_context` 需要非 OBS_LEAN 观测里的 `zombie_count_multiplier`）——
这正是 z13 与 z14 的局面完全相同、`leaf_value` 仍能区分两者的原因。

---

## 6. 阶段 1 开跑前还要做的事

1. **CPU 参照缺 4 个增量字段**（`zombiesKilled` / `plantsEaten` / `mowersTriggered` / `sunProduced`）。
   `cuda/SimHost.cpp` 的统计表已经有这三列（`eaten` / `kill` / `sunProduced`，`cuda/SimHost.cpp:280-306`），
   但 CPU 参照里只有空对象 `events`。**没有参照值的列不能算一致率**，否则等于拿 0 去比。
2. **GPU 侧还没有把 `.cudaroot` 读进 lane 的批量入口**：`cuda/SimHost.cpp` 目前默认只读
   `artifacts/cuda/corpus_smoke/roots.cudaroot`，且 `--seed-base` 是单个常数（1600000000），
   而语料的 `future` 是**每个种子一组**（`seed*1000+f`，192 个不同的 future）。必须改成按
   `corpus.jsonl` 的 (root, future) 配对喂，否则两边跑的不是同一批随机未来。
3. **一致率与噪声地板的测量脚本还不存在**。需要：同一 (root, future) 两边各跑一次 → 逐字段比对 →
   以及"CPU 自己换个 future 再跑"当噪声参照。`python/pvz_cuda_root.py` 已经能把 `.cudaroot` 读成字典，
   比较脚本可以直接复用它。
4. **`--static-value` 不在语料脚本的参数里**（原 docstring 误写）。教师搜索用的静态值网
   `artifacts/hard/value_branch1/model.pt` 目前**一个字段都没进语料**。如果阶段 1 要评"搜索选择一致率"
   的分支打分，得决定是否把静态值网也写成语料的一列；现在只有学生决策 + `--value` 的叶子分数。
5. 每根 8 个 future 是**能测分布的下限**，不是够用的量。713 根 × 8 条已经够算分布，但要给一致率一个
   可信区间，加根只需 `--seeds` 提高（1 核 25 s / 144 根，很便宜），加 future 数会线性拉长 CPU 参照时间。

---

## 7. 看到但**没有**修的问题

按任务约束，`src/CudaExport.cpp` 与 `cuda/` 一个字没动；下面都是读代码 + 读语料得到的实测结论。

1. **`ROLLOUT` 命令永远不回 `events` 块**（`src/main.cpp`）。分支在 `src/main.cpp:403-405` 专门为了算增量
   把计数器重读一遍（注释写着 "The events block must describe the rollout"），但响应在
   `src/main.cpp:488-496` 自己打印并以 `<< "]}" << std::endl; continue;`（496-497）收尾，
   **走不到** `src/main.cpp:908-918` 那段负责 `"events":{zombies_killed, plants_eaten, sun_produced, sun_spent,
   mower_triggered, waves_started, ...}` 的公共尾段。于是 `before` 被赋值却从不使用，`env.rollout()` 的响应里
   根本没有 `events` 键，`python/pvz_env.py:352` 存的 `_last_response["events"]` 永远是 `None`。
   这就是 §6.1 那 4 个字段缺失的唯一原因。**修法**：让 ROLLOUT 分支别自己 `continue`，把尾段接上；
   代价是重建 `build/pvz-portable` —— 那个二进制（10-17 21:19）是阶段 0 吞吐实测用过的，
   重建会让"同一个二进制"这条可比性断掉，所以本次没动。
2. **`FillLevelInfo` 把卡组写死在代码里**（`src/CudaExport.cpp:109-111` 的 `static const SeedType deck[]`），
   不读 `board->mSeedBank`；同一函数 `src/CudaExport.cpp:86` 把 `playableRows` 硬编码为 5（草地）。
   第 7 关这张卡组恰好与白名单逐张相等（已实测：`LevelInfo.plants` 的 `seed_type` 序列 = 各根的
   `packets[].type` 序列 = `[0,1,2,3,4,5,7]`），所以本语料没错。但**换一个任务族（泳池/屋顶/别的卡组）
   就会静默错位**：`LevelInfo` 描述一套卡，`RootRecord` 装另一套卡，而 `KernelSupports*` 的
   `supported` 判定不受影响，语料看起来完全正常。建议：导出前把两者相等的断言加进 `scripts/cuda_root_corpus.py`
   （本次没加，因为它需要在解析完 `.cudaroot` 之后才能做，等于给语料脚本再挂一遍读取器）。
3. **`RootPacket::refresh_time` 与 `PlantDef::refresh_time` 是两个不同语义的数，别取错列。**
   实测（713 个根全量直方图，`PlantDef.refresh_time` 一律是定义值）：

   | 卡 | `PlantDef` | 包上出现过的 `refresh_time`（出现次数） |
   |---|---|---|
   | 豌豆 0 | 750 | **0**×76, 750×637 |
   | 向日葵 1 | 750 | 750×713 |
   | 樱桃 2 | 5000 | **3500**×608, 5000×105 |
   | 坚果 3 | 3000 | **2000**×553, 3000×160 |
   | 土豆雷 4 | 3000 | 3000×710, **2000**×3 |
   | 雪花豌豆 5 | 750 | **0**×713 |
   | 双发 7 | 750 | **0**×688, 750×25 |

   机制：`SeedPacket::SetPacketType`（`src/Lawn/SeedPacket.cpp:1106-1143`）开局把 5000/3000 冷却的卡
   降成 3500/2000，其余卡留 `mRefreshTime = 0`；第一次种下去之后 `StartRefresh`
   （`src/Lawn/SeedPacket.cpp:893`）才把它覆盖成定义值。**所以包上的 `refresh_time` 是"当前这一次冷却"
   的实时状态，不是这张卡的冷却常数**：雪 pea 在 713 个根里 `refresh_time` 全是 0。
   内核若拿 `RootPacket::refresh_time` 当"下次种植要等多久"，雪豌豆/双发/豌豆在没种过时就会**零冷却**，
   而樱桃/坚果的开局冷却（3500/2000）又和定义值（5000/3000）不一致 —— 两种错法都能发生。
   常数只能从 `LevelInfo.plants[].refresh_time` 取，开局降冷那条规则得在核**外**单独建模。
4. **第 7 关的 `numWaves` 正好等于 `kMaxWaves = 30`**（`LevelInfo.num_waves=30`；每根的波表基数合计
   149~164，随对局种子抽、共 11 种取值，单波基数 1~23，`kWaveZombieSlots=32`）。也就是说
   `src/CudaExport.cpp:348` 的
   `board->mNumWaves > kMaxWaves → unsupportedReason=6100` 这条闸门在第 7 关是**零余量贴边**：
   任何波数 > 30 的关卡会整关被弃（不是局部弃根）。本语料 713 个根的 `wave_count` 全是 30，
   单波基数离 32 只差 9，属于"目前安全、以后未必"。
5. **倍率 1.3 与 1.4 在第 7 关几乎不产生不同的局面**：`corpus_l7_z13/roots.cudaroot` 与
   `corpus_l7_z14/roots.cudaroot` **只有 `LevelInfo` 里那个 float 不同**（第 2429 字节起），
   713 条根记录逐字节相同；1152 条参照续走里只有 33 条（2.9%）在 `sun/wave/zombies/income/net_calls/
   row_min_zombie_x/leaf_value` 上分岔。原因在 `Board::SpawnZombieWave`
   （`src/Lawn/Board.cpp:4917`）：出场数是 `lround(基数 × 倍率)`，第 7 关每波基数小，1.3 与 1.4 的取整
   结果绝大多数相同。**结论：想覆盖 1.3–1.5 这一档，z14 不带来新局面**，真正的多样性在 z15
   （1.5 的取整差异更大，425 个根、根 tick 下界也更低）。
   反过来这也是个免费的检验点：如果 GPU 实现对倍率不敏感（或把 `lround` 写成截断），
   z13/z14/z15 三份语料的结果会互相重合，一测就露。

---

## 8. 与并行语料 `corpus_phase1` 的交叉核对

同一工作区另一路会话在 10-18 00:01 用**同一脚本**产出过 `artifacts/cuda/corpus_phase1/`：
276 根（全部 `supported=1`，根 tick 1020..23426）、2208 条参照对、`roots.cudaroot` 4,877,696 B，
参数是 `--seed-offset 11000 --root-stride 400 --roots-per-game 30`，报告在
`MEMORY/docs/cuda-corpus-phase1.md`（未入库的一次性报告）。
两份合计 **989 个受支持根 / 7912 条参照 (root, future) 对**，结论形态一致（全 `supported=1`、容量都不触顶）。
**`corpus.jsonl` 的 schema 相同** —— 那一批是在我改名之后跑的，已经是
`end_tick / mowers_left / mowers_alive / net_calls / planted_by_type / plants_planted` 这套键；
唯一差别是它跑的时候还没有 `--rollout-max-actions`（续走用 `pvz_env.rollout` 的默认上限 1000000），
而两份语料的实测最大决策数都只有 52，上限远未触达，所以语义等价、可以合池用。

我在自己这 713 根 / 997 抛射物 / 2508 僵尸上独立复核了那份报告列的弱点，**四条都成立**：

| 那份的结论 | 我这 713 根的复核值 |
|---|---|
| `RootProjectile::velX/velY` 是死字段 | 997 个抛射物两个速度**全为 0**；type 全 0（豌豆），`chill` 全 0，age 1..160 |
| `RootZombie::targetCell` 恒 -1 | 2508 个僵尸**全为 -1**，其中 216 个带 `ZB_EATING`（位值是 2，不是 1）正在啃食 |
| `LevelInfo::zombies[]` 只填前 33 项 | 非零槽 = 0..32 共 33 项，**第 33 槽整项为 0 且 `type` 读成 `ZOMBIE_NORMAL=0`**。循环上界是 `NUM_ZOMBIE_TYPES`（= 33，`src/ConstEnums.h:1340-1376`），数组长度是 `kZombieTypes`（= 34，`cuda/CudaRollout.h:55,119`）。不看 `allowed[]` 就遍历 `zombies[]` 的内核代码会多出一种"血 0 的普通僵尸" |
| 阳光币 0/276 命中 | **3/713 命中**（都是 `value=25`）。coin 分支不是完全测不到，但 0.4% 的覆盖率实质等于没覆盖。原因在 CPU 侧不在导出器：`LawnApp::EnvironmentWait` 每 tick 开头先 `ScoreCoin()`，根快照正好取在"已收完"那一刻（`src/CudaExport.cpp:295-297` 把 `ticksToCollect` 钉成 1 就是这个语义） |

再加一条两边都只点到没写透的盲区 —— **学生策略决定的**，加大 `--seeds` 也补不上：

- 根上活着的植物按**卡类型**统计：`{0 豌豆 3635, 1 向日葵 6278, 2 樱桃 1, 3 坚果 206, 4 土豆雷 243,
  7 双发 25}` —— 类型 5（雪豌豆）**一个根都没有**。续走里的种植次数按**种子槽位** 0..6 计数
  （`planted_by_type`，槽 5 = 雪豌豆、槽 6 = 双发）：
  `{0:13869, 1:6135, 2:463, 3:613, 4:1866, 5:28, 6:190}`，合计 23164。
- 于是 `PB_CHERRY`（樱桃，1 个根）与雪豌豆的减速链在这批语料里几乎没有实例：2508 个僵尸的
  `chilled` **全为 0**，997 个抛射物的 `chill` 也全为 0（没有一枚雪豌豆的子弹）。
  **阶段 1 的一致率实际只在考**：豌豆 / 双发 / 向日葵 / 坚果 / 土豆雷
  这几条路径。要考樱桃与雪豌豆，得往语料里灌**人为种下这两张卡**的根（合成根或改采样策略），
  光加真局不够。

