# task267：修 GPT 报告的四个接口问题，打 238j 包

本任务只在 Mac 上做，未连接台式机，未跑整局对局、未开窗口。分支 `pkg238j`（基于 `pkg238h` 8b8d546f），代码提交 `0a0bf257bc813b22ef58088afee7d92ff17b0549`。

## 1. `zombies()` 血量全是 0

**根因**：`scripts/drill_start.py` 的 `on_board()` 读的是 `z["health"]`，而完整观测里的僵尸字段是 `body_health` / `body_max_health` / `helm_health`；`health` 是植物字段。读不到就落到 `or 0`，同一批 id 的血量因此全报 0。

**改动**：

- `on_board()` 改读 `body_health`，`hp` 与 `body_health` 同值；同时保留 `body_max_health`、`helm_health` 和 `phase`。精简观测的定长数组路径仍读原来的第 5 列（学生观测里的本体血量）。
- `Game.zombies()` 的说明同步为“hp 是本体血量”，并列出新增字段。
- 核查了其余读血量处：`scene.py` 的僵尸行原本就用 `body_health`；`render_episode.py` 也用 `body_health + helm_health + shield_health`；没有第二处把 `health` 当僵尸血量。

## 2. `plant('repeater')` 报重名

**根因**：卡片名表来自 `gPlantDefs.mPlantName`。`SEED_REPEATER`(7) 与 `SEED_LEFTPEATER`(52) 在表里都写 `"REPEATER"`，于是 `repeater` 同时命中 7 和 52。

**改动**：

- `scripts/pvz_constants.py` 新增 `seed_types()`，从 `src/ConstEnums.h` 的 `enum SeedType` 现解析名字与数值，不手抄表。
- `scripts/llmplay/cards.py` 用枚举名作为唯一性来源：`SEED_LEFTPEATER` → `leftpeater`。`gPlantDefs.mPlantName` 只有在全局唯一时才继续作为别名保留，所以旧的 `doom_shroom` 等写法不受影响；52 的旧显示名 `repeater` 因为重名不再作为别名。初始化时机械检查所有别名，任何一个名字指向两张卡就抛错。
- `reference.rows()` 改用同一套 `cards.slug()`，`PLANTS.md` 不再出现两个 `repeater`。

改完后 53 个植物名全部唯一：

```text
   0 peashooter         1 sunflower          2 cherry_bomb        3 wall_nut
   4 potato_mine        5 snow_pea           6 chomper            7 repeater
   8 puff_shroom        9 sun_shroom        10 fume_shroom       11 grave_buster
  12 hypno_shroom      13 scaredy_shroom    14 ice_shroom        15 doom_shroom
  16 lily_pad          17 squash            18 threepeater       19 tangle_kelp
  20 jalapeno          21 spikeweed         22 torchwood         23 tall_nut
  24 sea_shroom        25 plantern          26 cactus            27 blover
  28 split_pea         29 starfruit         30 pumpkin           31 magnet_shroom
  32 cabbage_pult      33 flower_pot        34 kernel_pult       35 coffee_bean
  36 garlic            37 umbrella_leaf     38 marigold          39 melon_pult
  40 gatling_pea       41 twin_sunflower    42 gloom_shroom      43 cattail
  44 winter_melon      45 gold_magnet       46 spikerock         47 cob_cannon
  48 imitater          49 explode_o_nut     50 giant_wallnut     51 sprout
  52 leftpeater
```

`repeater` 只解析为 7，`leftpeater` 解析为 52；中文名与去下划线写法继续可用。

## 3. 冷却显示反了

**根因**：引擎观测的 `packets[].cooldown` 是 `SeedPacket::mRefreshCounter`，即**已过**的更新数；`refresh_time` 才是总冷却。`scene.cards()` 直接把它标成“冷却剩 N tick”，所以数值随时间变大。

**改动**：

- `scene.cards()` 显示 `remaining = max(0, refresh_time − cooldown)`：`就绪` / `冷却剩 N tick（总 M）` / `冷却到点，下一帧可种`。
- 卡槽区顶部固定说明：冷却显示的是剩余 tick；到 0 的那一帧还没进 `legal()`，下一帧才可种，始终以 `legal()` 为准。没有改引擎。
- `cards.packets()` 的字段改为 `elapsed`（已过）、`refresh_time`（总冷却）、`remaining`（剩余），名字和含义一致；`slot_for()` 用 `active` 判断就绪。
- `PROMPT.md` 第 5 节同步了这段说明。

## 4. 死亡动画期间仍算在场

**根因**：`drill_start.judge()` 只用 `on_board()` 的 id 集合判断目标是否还在场。僵尸进入 `PHASE_ZOMBIE_DYING` 后仍会 `on_board=True` 几百 tick，视界刚好结束在这段动画里就会误判失败。

**改动**：

- `scripts/pvz_constants.py` 新增 `zombie_phases()` 和 `zombie_death_phases()`：后者从 `Zombie.cpp` 的 `Zombie::IsDeadOrDying()` 函数体抓 `PHASE_*` 名字，再经 `enum ZombiePhase` 解析成数值。没有写死 1/2/3，也没有另抄阶段表。
- `judge()` 判据改为：目标已不在场，**或**其 `phase` 属于死亡类阶段，都算解决；结果新增 `dying` 列表。推车判据 `mower_unchanged`（`mower_used == 0`）不变。
- 判据只有一份：沙箱 `Game.drill_result()` → `Game._judge()` → `drill_start.judge()`；台式机导入器 `scripts/import_llm_run.py::verify_short()` 对 drill 调 `game.drill_result()`，对 scenario 直接调 `drill_start.judge()`，两条都走同一个函数。
- `PROMPT.md` 第 2 节同步：目标离场或进入 DYING/BURNED/MOWERED 都算解决，不要为等消失拖过视界。

## 测试

每个问题一条定向用例，均加 `timeout 300` 与 `ulimit -f 2000000`，只跑 `python/test_llmplay.py` 的相关 `-k` 用例：

| 问题 | 命令（`-k`） | 结果 |
|---|---|---|
| 1 血量字段 | `test_zombies_hp_reads_body_health` | 1 passed |
| 2 卡名重名 | `test_repeater_and_leftpeater_are_distinct_cards` | 1 passed |
| 3 冷却方向 | `test_cooldown_shows_remaining_ticks` | 1 passed |
| 4 死亡动画判据 | `test_dying_target_counts_as_resolved` | 1 passed |

回归只跑了相邻的纯逻辑用例：`-k 'ReferenceTable or Task267 or VisibleTimer or Grid'` 共 22 passed；`-k 'card or cooldown or zombie or Task267'` 共 10 passed。没有跑整套测试，没有启动引擎对局。

## 打包

命令按要求复用 238i 的静态引擎、资源、resume、drill/scenario 资产；`--partial` 用 `~/pvz-logs/task267/returned-out.zip` 软链指向 `LLMPlay_2026-11-11_005323_HKT_out.zip.001`（该文件本身就是完整 zip，只读其中的进度）。

- `check_engine` 通过：x86-64 Linux 静态 ELF，无 `PT_INTERP`、无动态桌面库入口。
- 静态引擎未变，SHA256 仍为 `1c3680f36aa5bd5fef97d035a97a1cdd9ca89fce2493a87f125c7bd507d29c07`。
- carry 与 238i 相同：15,687 条 archives，`run_id=430470527acb485983f7ea261a0584d4`；`PROMPT.md` 的 carry 时间仍为 `2026-11-11 00:53:23 HKT`。
- 包内 resume 8,457 项、drill 927 项、scenario 300 项；PROMPT/PLANTS/接口代码与工作区一致。

目录：`/Users/newbiexvwu/Downloads/pvz-llmplay-238j/`

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `llmplay_pkg.7z.001` | 20000000 | `ff82d1550008d4bece46e070969e262eabfbb20d19bbfc28b72e69dd25c58b7d` |
| `llmplay_pkg.7z.002` | 340160 | `d0f4fe9ec2a1b18df6ef75d252f8dfdbf8cf34c057c350499dbfc2e02c3414bd` |
| `llmplay_res.zip` | 34258402 | `0005bb8e31ad2e2ca7649bc0b6128211ff13c8f1c333c2163984302d7d65d94e` |

## 说明

`PROMPT.md` 当前没有“已知问题”小节（238i 包与仓库里都搜不到），所以没有可删除的旧条目；四个问题的旧描述已从正文规则中替换，版本号改为 task267 / 238j。台式机验收按要求留给主控，本任务未执行任何 `ssh ... wsl`。
