# task82：状态哈希跨进程确定性

## 结论与范围

根因是快照读取了三个位置的未初始化字段。已测冒险关卡中，哈希会变，但相同动作的游戏状态和终局不变；它们不是指针，也没有观察到时间或随机种子漂移。静态链接改变堆残值出现的频率，没有产生另一套根因。

修复为在创建对象时显式初始化，保留所有哈希字段、算法和 Save4 格式，没有新增引擎开关，没有改正式二进制或 qc。分支 `hashdet` 的部署交主控。证据以本报告、最终的 `state-hash-determinism-results.json` 和本机原始轨迹为准；不把中间只有两处修复的构建当成最终验收。

## 复现与逐字段定位

在任务指定的 deploy4、task81 全静态引擎上各启动 8 个独立进程，reset 同一份 L31 人类示范，再逐操作重放前 2000 操作。deploy4 开局出现 3 种哈希，静态版出现 5 种；同引擎首次哈希分歧都是操作 0。各引擎内部前 2000 操作的观察和事件完全相同。

`STATE_BYTES` 导出的是 `StableEnvironmentHash` 的实际输入：已清零绘制统计的 Save4 板面、RNG 和九个计数器。诊断脚本按 Save4 chunk 及可疑成员分段 SHA256，并比较原始字节。CRC 是整段载荷的派生值，不是第二个独立问题。

| 分歧位置 | 字段与证据 | 是否进入游戏逻辑 | 修复 |
|---|---|---|---|
| reset，chunk 19，字节 328–351 | `Challenge::mCloudsCounter[6]`；只有此数组及 CRC 变，其余 19 chunk、RNG 和额外计数器全同；残值有 `0xc61c4000` 等 | 冒险模式不读；智慧树使用前 `TreeOfWisdomInit` 已逐项赋值 | 构造时六项清零 |
| L31 操作 21375，chunk 4，某个弹丸 tail 偏移 98 | `Projectile::mLastPortalX`；一次对照只有该 int32 及 CRC 变，值 99 / 88 | 冒险模式不读；传送门小游戏在 `Challenge::UpdatePortal` 会读，因此该模式有实际逻辑风险 | `ProjectileInitialize` 设 -1，与僵尸/小推车哨兵一致 |
| L32 操作 41732，chunk 5，某个硬币 tail 偏移 25 | `Coin::mCollectX`（相邻 `mCollectY` 同样漏初始化）；差异只在未收集硬币的收集坐标及 CRC | `Collect()` 先覆盖两个坐标，再设置已收集；普通游戏不会读未赋值坐标 | `CoinInitialize` 两项设 0 |

原始字节例子分别在本机 `deploy4/reset_*.bin`、`midgame_dumps/operation_21375_*.bin`、`l32_dumps/operation_41732_*.bin`。目录根为 `/home/newbiexvwu/pvz-logs/task82/`。这些二进制与滚动输出不进 git。

Valgrind 3.26 的 `--track-origins=yes --num-callers=30` 对同源码、仅移除 mimalloc 的诊断重链接版本，确认 reset 时 `crc32_little → LawnSaveGameToMemory → StableEnvironmentHash` 使用未初始化值，来源是 `Board` 创建 `Challenge` 的堆分配；修复云数组后 reset 的 SaveGame 报告消失。保留相同编译对象，仅去 allocator 链接，用于让 Memcheck 跟踪真实分配，不是生产构建开关。另有资源缓存编译、闲置对象销毁的既有 Memcheck 报告，本任务不宣称整个引擎 Memcheck 零错误。

重置代码会覆盖启动时由 `time(0)` / `SDL_GetTicks` 初始化的 app seed、rand seed，重新播种两个 RNG 并清零 app counter；实际分段也确认它们一致。观察到的变化都是上述数字成员的堆残留，不能据哈希不同推断是 ASLR 或时钟进入了模拟。

两种指定旧二进制本身还不是相同版本：在操作 1806 检查到静态版的僵尸观察多了 `summon_counter: 0` 字段。于是本实验比较各引擎内部的可复现性，并用同源码修复前后构建做行为配对，不用两种旧二进制观察摘要不同来证明模拟漂移。

## 整局行为与旧存档

两种指定引擎各完成 8 次 L31 整局重放。每次覆盖全部 827 个存档边界点（含初态、全部决策点及非等待动作），比较完整 `PRIV` JSON 的 SHA256，包含阳光、己方植物、全部僵尸、隐藏对象和游戏 RNG，以及事件。各引擎内部全部一致，终局均为 tick 39479、wave 20、result 1。

含三处修复的最终构建（`build/pvz-portable`，`5e890b3a…`）另做 8 个独立进程的 L31 逐操作整局重放（39590 操作，privileged）：8 份 trace 的 SHA256 完全相同，reset 分段差异为空，终局同为 39479 / 20 / 1。这条不使用合并等待，覆盖每个操作而不只是 827 个边界点。

每种旧引擎另有两次不合并等待的完整逐操作基准（39590 操作）。扩样时只把相邻成功普通 wait 合并，停在每个存档 frame 和每次非等待动作处；与对应逐操作基准在全部 827 点核对 privileged 状态与累计事件，均完全相同。合并等待不用于前 2000 操作的逐操作哈希复现。

四份旧 Mac 示范的同源码修复前后配对：

| 存档 | 原动作数 | 全部边界点 | 原终局 tick / wave / result |
|---|---:|---:|---|
| `demo_fog_31_snow_m200` | 39590 | 827 | 39479 / 20 / 1 |
| `demo_fog_32_snow_m200` | 57596 | 1310 | 57363 / 30 / 1 |
| `demo_fog_33_cactus_snow_m200` | 41107 | 893 | 40969 / 20 / 1 |
| `demo_fog_33_cattail_snow_m200` | 41831 | 923 | 41682 / 20 / 1 |

修复前每份存档的两个独立进程得到两份不同 trace；修复后每份四个独立进程的 trace 字节完全相同（哈希、观测、事件全同）。跨修复版本逐边界比较，827 / 1310 / 893 / 923 个边界点的 privileged 观测与事件无一不同，终局 tick / wave / result 全同；首个哈希差分别在 3995 / 4191 / 3599 / 3874，都来自上述未初始化字段被写进快照。**行为会变的已测存档：无。** 每个操作的原 tick / end_tick / ok 由脚本逐条断言，源存档 12 个文件的 SHA256 前后一致。不以 `human_demo.replay` 的跨引擎宽松模式单独充当模拟一致证据。

## 兼容性

本次选择初始化修复，不排除字段，所以不另开 hash_schema：`portable_snapshot_v1_without_draw_statistics` 的字节口径、哈希算法和存档解析保持原样。没有添加跳过哈希或降级校验的开关，也没有改 `human_demo.py` 的既有严格/跨二进制校验逻辑。

旧存档中的垃圾字段哈希本来就没有跨进程可重现保证；初始化后，在场弹丸的 portal 字段会从旧的 0/堆残留变成 -1，未收集硬币的垃圾坐标也会清零，故旧哈希不一定相同。这与游戏重放结果变化是两件事。原始存档不重写、不偷偷 rebaseline；行为会变的已测存档由最终配对逐局报告（若为空就明确列为空），不承诺未测的小游戏或全部 50 关。

## 工具、验证与 Mac 交接

`scripts/state_hash_determinism.py` 默认串行启动 8 个独立进程，并逐操作检查前 2000 操作；`--operations -1` 只 reset，`--operations 0` 跑完整动作带，`--privileged --decisions` 做整局逐决策对照。所有输入输出路径必须绝对；脚本强制 `PVZ_NO_POPUP=1`，检测二进制运行中改变并拒绝混合证据。

`scripts/state_hash_compare.py` 汇总进程内/跨进程首次分歧和分段差异；`--full-reference` 将合并等待结果对齐到逐操作基准，并比较累计事件。它输出分歧而不把预期的旧哈希差异当成异常退出；门禁判断必须检查 JSON 中的具体 diff 字段。

新增 `python/test_state_hash_determinism.py` 在 8 个独立进程检查云数组、在场弹丸 portal 哨兵、未收集硬币坐标、固定序列哈希与 snapshot/restore。旧构建会在弹丸字段上失败（实测 `401cc600`），修复版通过；Python 协议测试 35 项通过。

Mac 上的正式构建按任务要求交主控运行，台式机没有 Mac，不声称在那里执行过。主控把本分支合入 Mac 检出后，直接运行 `scripts/state_hash_mac_probe.sh`（路径可用 `PVZ_MAC_ROOT` / `PVZ_MAC_EXECUTABLE` / `PVZ_RESOURCE_DIR` / `PVZ_ARCHIVE` / `PVZ_OUT` 覆盖），它等价于下面两条命令，对**当前正式 build 引擎**运行，不需要先部署本修复：

```bash
PVZ_NO_POPUP=1 python3 /Users/newbiexvwu/PvZAgent/scripts/state_hash_determinism.py \
  --executable /Users/newbiexvwu/PvZAgent/build/pvz-portable \
  --resource-dir /Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN \
  --archive /Users/newbiexvwu/PvZAgent/artifacts/human_demo/raw/seed_20261031/demo_fog_31_snow_m200 \
  --out /Users/newbiexvwu/pvz-logs/task82/mac --repeats 8 --operations 2000

python3 /Users/newbiexvwu/PvZAgent/scripts/state_hash_compare.py \
  /Users/newbiexvwu/pvz-logs/task82/mac \
  --out /Users/newbiexvwu/pvz-logs/task82/mac-comparison.json
```

交回 `$OUT/summary.json` 和 `$OUT/comparison.json` 即可（脚本把比较结果写在输出目录内；上面两条手工命令的写法是 `mac-comparison.json`，两者内容同构）；必要时再在本机比对 STATE_BYTES，不上传游戏资源或画面。脚本已在台式机用覆盖路径对本机修复版完整跑通：8 个独立进程、1 种 reset 哈希、2001 个逐操作点全同，比较 JSON 正常生成——交接的是执行过的路径，不是未验证的文本。正式 build 若是另一个 Mac 检出的目录，只替换上述绝对 executable 路径。该脚本同时最多一个游戏进程；台式机所有阶段最多两个，没有操作 tmux `qc` 或其他工作树。
