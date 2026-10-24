# task63：人类示范重放不一致——录制（有窗口）与校验重放（无头）在第 14754 步分叉

> 主控 2026-10-24 写。执行者：Mac CodeBuddy（`cb-demo` 会话）。工作树 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo`（分支 `demo`）。游戏进程最多 2 个；不碰 Mac 上在跑的 p5d 采集（`artifacts/hard_mac/`、`~/.pvz-dist/`）。

## 现象
- 用户用 `scripts/human_demo.py record --task demo_fog_31_fume` 打完第 31 关，**通关**：20 波，终 tick 40355，874 个动作，其中种植 110 次、铲子 22 次、手动收集 1 次。
- 关窗口后自动校验报错：

  ```
  RuntimeError: demonstration state hash differs at operation 14754, tick 14712
  ```

  （`human_demo.py` 第 185 行，`replay()`）
- 存档：`artifacts/human_demo/raw/seed_20261031/demo_fog_31_fume/`。**这局很珍贵，不要覆盖或删除**；需要试验时复制一份到别处。
- `ticks.jsonl` 里：
  - 第 14580 步（tick 14540）是**第一次铲子**（col 4, row 0）；
  - 第 14753 步（tick 14711）种植小喷菇（packet 1）到 (col 6, row 0)，哈希一致；
  - 第 14754 步是紧接着的 wait 1 tick，哈希不一致。
  - 前面 109 次种植和 1 次手动收集都一致。
- 疑点：第一次铲子之后约 170 tick 出现分叉，铲子在有窗口时可能有额外效果（例如高亮、光标对象、动画或粒子消耗随机数、Plant::Die 的分支不同等）。但这只是猜测，要用证据定位。

## 要做的
1. **复现**：
   - 用同一份存档，以**无头**方式重放，确认在第 14754 步分叉；
   - 再以**有窗口但自动**的方式（`headless=False`，按 ticks.jsonl 自动喂动作，不需要人操作）重放，看是否走到通关、与录制哈希一致。
   - 由此判断差异是"有窗口 vs 无头"，还是录制时的人为输入（光标、暂停、拖拽中的状态）造成的。
2. **定位**：
   - 在分叉前后的 tick 上，导出两边的完整引擎快照，或哈希覆盖的各字段：植物、僵尸、子弹、阳光、随机数状态、雾、光标对象等；逐字段比较，找出第一个不同的字段与原因。
   - 查 `portable_snapshot_v1_without_draw_statistics` 哈希包含哪些字段，以及有窗口时哪些代码路径会改动这些字段（Draw、Update 里依赖 mApp 窗口状态的分支、鼠标悬停 / 光标预览、铲子高亮等）。
3. **修复**：
   - 让录制与无头重放在引擎层面一致：有窗口时不该改变游戏状态的代码就不改；确实属于游戏状态的输入，就把它录进 ticks.jsonl，重放时照样喂入。
   - 修复后，用户这局存档要能 `verified: true`。如果修复改变了录制格式，旧存档不一定能直接通过：先试，不通过就写明原因，并提供一个转换或补录办法。不能为了让它通过而放宽校验。
   - 回归：
     - 自己用自动有窗口方式录一局含铲子的短局（可用 `--task demo_fog_31_fume`，随便种几株、铲两下、几百 tick 后 Esc），确认 `verified: true`；
     - 相关测试全过。
4. 引擎改动要按惯例构建，Mac 用 task47 的构建方式。若改动影响 Linux 生产引擎（deploy3），**只提交到 demo 分支，不重建生产引擎**，在结果里写明，由主控决定何时部署。
5. 结果写在本文件"结果"一节；每完成一小步就提交、`git push origin demo`。

## 规则
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则。
- 不 `git stash`；只用绝对路径；不碰 tmux 里的其他会话。
- 疑问写"问题"一节，按最佳判断继续。

## 结果

### 结论（一句话）

分叉不在"有窗口 vs 无头"，而在引擎本身：**粒子 `FIELD_SHAKE` 每帧用
`srand(age * reinterpret_cast<uintptr_t>(particle))` 播种 C 运行时的 `rand()`**，
粒子位置（属于存档状态、进哈希）因此依赖堆地址。用户那局的 40488 个哈希里有 **20366 个
（50.3%）编码了录制进程的堆地址**，任何进程都复现不了 —— 包括旧引擎自己。

### 1. 复现（先证伪"窗口"假设）

| 试验 | 结果 |
|---|---|
| 无头重放同一存档 | 第 14754 步分叉（与报告一致） |
| **有窗口**（`headless=False`，自动喂 ticks.jsonl） | 第 14754 步分叉，同一个位置 |
| 同一份存档、同一引擎，无头重放 **7 次** | 前 14753 步每次都对上；**14754 步得到 7 个不同哈希** |

7 次在第 14754 步的哈希：
`467c906ce7c336d7` / `9999f53af973df51` / `ebd8ffd837a3e5aa` / `3c06ae1bb4f115da` /
`845885cf3f2f97bb` / `12621765921cabd0` / `bba236e9d546e57c`。

**同一份存档、同一个二进制，重放自己都不自洽** —— 这一条就排除了"窗口""人的输入没录进去"
这类解释：分叉是引擎里的非确定性。也说明录制时那个哈希值永远不可能再被任何进程算出来。

### 2. 定位（字段级）

新增引擎调试命令 `STATE_BYTES <path>`（`src/main.cpp`）：按 `StableEnvironmentHash` 的口径
导出它折叠进去的**全部**内容 —— SaveGame V4 的整块 board 镜像（header + 20 个 TLV chunk）、
MTRand 状态（2500 B）、9 个标量。用 Python 重写 `EnvironmentSnapshotHash` 验证过逐位相同
（op14753 → `bdf680774be6d116`，op14754 → 引擎值）。

同一 tick、两次重放的完整快照逐 chunk 对比：

```
board size 90825 / 90825   rand 状态 2500 B 相同   scalars 相同
PARTICLE_SYSTEMS   size 2010->2010  first_diff@1836 ndiff=4     ← 只有这一处
```

解析该 chunk：粒子系统 id `89849857`（`PARTICLE_FUMECLOUD`，大喷菇开火喷出的雾），
emitter[0] 的 `p[0:1].position`：

| 字段 | 运行 A | 运行 B |
|---|---|---|
| `p[0:1].position.x` | 473.4210205078125 | 472.5303955078125 |
| `p[0:1].position.y` | 112.34538269042969 | 112.8812484741211 |

同一颗粒子的 age/duration/timeValue/16 个 interp/4×2 个 fieldInterp/imageFrame/spin
**全部相同**，只有位置不同。

### 3. 根因（代码）

`src/PvzpLib/PvzpParticle.cpp` 的 `ParticleFieldType::FIELD_SHAKE`（原第 534–550 行）：

```cpp
srand(aLastRandSeed * reinterpret_cast<uintptr_t>(theParticle));   // ← 种子是堆地址
theParticle->mPosition.x -= aLastX * (rand() / RAND_MAX * 2 - 1);
...
srand(theParticle->mParticleAge * reinterpret_cast<uintptr_t>(theParticle));
theParticle->mPosition.x += x * (rand() / RAND_MAX * 2 - 1);
```

三个后果：

1. 位置进了存档（`SyncParticlePortable` → `mPosition`），所以**哈希依赖堆布局**；
2. `rand()` 的实现由 libc 决定（macOS 与 glibc 不同），Mac 录的局在 Linux 上不可能逐位重放；
3. `srand()` 是进程全局的，顺带把别人的 `rand()` 流也搅了。

**时间线正好对上**：这颗雾粒子 age=0 那帧的种子是 `0 * 地址 = 0`，确定性，所以录制与重放
一直对到 op14753；age=1 那帧（tick 14712）种子变成 `1 * 地址`，op14754 立刻分叉。这也解释了
为什么第一次铲子（op14580）之后的 170 tick 才炸 —— 与铲子无关，是**关卡里第一发大喷菇雾**
恰好在那时出现（`FumeCloud` 是唯一带 shake 场的粒子定义）。

### 4. 修复

`FIELD_SHAKE` 不再碰 libc 随机数，改用粒子**已经序列化的 DataArray id**（存档里有）与帧号、
坐标轴的 32 位雪崩混合，映射到 [-1,1)：

```cpp
static float ShakeUnit(uint32_t theParticleId, int theAge, uint32_t theAxis);
...
uint32_t aParticleId = ...mParticles.DataArrayGetID(theParticle);
theParticle->mPosition.x += x * ShakeUnit(aParticleId, theParticle->mParticleAge, 0U);
```

- 不新增存档字段、不改存档格式（id 本来就在 emitter 的 particle list 里），没有额外的状态；
- 抖动幅度仍由每颗粒子自己的 `mParticleFieldInterp`（来自 `Sexy::Rand`）决定，观感不变；
- 跨平台一致（不依赖 libc）。

修复后：

| 检查 | 结果 |
|---|---|
| 无头重放 3 次，op14754/14755/14760 | 三次哈希完全相同 |
| **有窗口**重放用户整局 40488 步 vs 无头重算 | **0 处不一致**，终局 40355 tick / 20 波 / result=1 |

### 5. 用户那局存档

**原存档无法 `verified: true`**，原因是可证的、不是配置问题：它 40488 个哈希里有 20366 个
把录制进程的堆地址折了进去。旧引擎自己都算不出来（上面 7 次 7 个值），新引擎也不可能。
不放宽校验 —— 把哈希检查改成"跳过 shake 相关的字节"就等于让所有存档都能过，那正是这条链
存在的意义。

**补救**：新增 `scripts/human_demo.py rebaseline`。它用当前引擎把存档的**动作**重放一遍，
逐条核对 `end_tick` / `ok`（动作时间线不许变）、核对终局（tick / 波次 / result），
然后把哈希写进**新目录**（源目录只读，不覆盖、不删除），最后对新目录跑一次 `verify`。

对用户这局的执行结果（原始目录 sha256 前后一致，未被触碰）：

```
python3 scripts/human_demo.py rebaseline \
  --archive artifacts/human_demo/raw/seed_20261031/demo_fog_31_fume \
  --out     artifacts/human_demo/raw/seed_20261031/demo_fog_31_fume_rebaselined
{"verified": true, "operations": 40488, "tick": 40355, "wave": 20, "result": 1}
```

- 重写 20366 / 40488 条哈希，**动作一条没动**，终局仍是 20 波、40355 tick、通关；
- 新目录 `verified: true`，连跑两次 verify 都过（同一台机器、跨进程、跨窗口/无头都一致）；
- meta 里留 `rebaseline: {source, rewritten_operations, reason}` 作为来源标记。

这是**一次性转换**，不是把校验放水：动作时间线与终局仍然逐条对照原存档，只是把"被堆地址
污染的那把尺子"换成确定性的那把，换完以后这个存档在任何机器上都能自查。

### 6. 回归

- 自动**有窗口**短局：`scripts/human_demo_autoplay.py`（4 株小喷菇、**2 次铲子**、270 tick，
  `demo_fog_31_fume`）→ `{"verified": true, "operations": 276, "tick": 270}`；再跑一次 verify
  仍是 true。存档在 `artifacts/human_demo/autoplay/`（不入 git）。
- 用户整局的**有窗口**重放 == 无头重算（0 处不一致），见上表。
- 测试：`python3 -m pytest python -q -p no:cacheprovider` →
  **326 passed / 1 failed / 26 skipped / 133 subtests passed**（34.5 s）。
  唯一的失败是 `test_dagger_experiment_runner.py::...test_low_memory_pauses_and_resumes_only_own_parent_and_child`，
  这是**既有的 Mac 环境性失败**（task48 记录为 323 passed / 26 skipped / 1 failed，task18/task32/task41 都记过同一条：
  Mac 上拿不到 Linux 的进程内存统计），与本任务无关 —— 本次改动只碰粒子抖动、`main.cpp` 的调试命令和
  `scripts/human_demo*.py`，该测试连引擎都不加载。
  `python3 -m pytest python/test_human_demo.py` → 3 passed。

### 7. 部署

改动落在 `src/PvzpLib/PvzpParticle.cpp` 与 `src/main.cpp`，**影响 Linux 生产引擎（deploy3）**：
任何跑过 shake 场的 tick 哈希会变。按要求只提交到 `demo` 分支，**没有重建生产引擎**，
何时部署由主控决定。部署前要知道：旧引擎录的示范（含雾/雾霾类粒子）在部署后无法直接
`verify`，需要 `rebaseline` 或重录。

提交（`demo` 分支，起点 `3780877c`，已 `push origin demo`）。下表是三个实质提交；它们之后的提交只改本文件与 `MEMORY/2026-10-24.md`，不复述哈希以免自我指涉：

| commit | 内容 |
|---|---|
| `6bf977e8` | 粒子抖动不再以堆地址做随机种子（根因修复） |
| `23731d42` | `human_demo.py rebaseline` + `human_demo_autoplay.py` 有窗口回归 |
| `3ddf7468` | 本文件的结果与测试数字、`MEMORY/2026-10-24.md` |

推送后复核：源档 `demo_fog_31_fume` 的 `ticks.jsonl` / `meta.json` sha256 仍是
`d582232d…` / `a7d9dd90…`（未被改动）；`demo_fog_31_fume_rebaselined` 再跑一次 `verify` 仍
`{"verified": true, "operations": 40488, "tick": 40355, "wave": 20, "result": 1}`；
`artifacts/human_demo/autoplay/seed_20261031/demo_fog_31_fume` 再跑一次仍
`{"verified": true, "operations": 276, "tick": 270, "wave": 0, "result": 0}`。

## 问题

1. **rebaseline 出来的存档能不能当训练数据**：它 `verified: true`，动作与终局都逐条对过原档，
   但它的哈希链是转换时重建的，不是录制时独立的第二遍。我的判断是可以用（`export` 只要求
   `verified`），但这属于"证据强度"的口径问题，请主控定；如果不接受，这局就只能重录。
2. **修复让分叉点提前了一个 tick**（op14752 而非 op14754）：旧代码 age=0 那帧种子是 `0*地址=0`，
   恰好是确定性的，新公式换掉了那个值。它不影响任何判断，但意味着"旧存档在新引擎上从第一发雾
   开始就对不上"，比报告的 14754 更早。
3. **别的 `rand()` 用法没动**：`LawnApp.cpp:3863`（戴夫气泡 `{SHAKE}`，只在绘制里）与
   `SexyAppBase.cpp:3471`（`srand(SDL_GetTicks())` 启动播种）都碰不到存档状态；env 的 reset 会
   `std::srand(seed)`，所以状态路径上是干净的。真要彻底，启动那句也该去掉，但那会改变
   非 env 模式的观感，不在本任务范围。
