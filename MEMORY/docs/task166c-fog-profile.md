# task166 E11c 雾天观测耗时剖析（只剖析，未改逻辑）

2026-10-29 · 分支 `fogprof` · 本机 Mac（arm64）· 生产引擎 `build/pvz-portable`（Release，`-O3 -DNDEBUG`，带符号）
· r36 学生 `.claude/worktrees/archbench/artifacts/archbench/r36/model.pt`，门禁清单
`.claude/worktrees/archbench/artifacts/archbench/r36_gate/manifest.json`（与 v5 门禁同样本、同种子）

---

## 0. 结论（先看这五条）

1. **雾关贵，主要不贵在雾的模拟，而贵在“雾关不走原生整局快路径”。**
   `scripts/search_play.py:764` 的快路径条件是
   `arm=="fast" and not frames and fog_diagnostics is None and observer is None`；
   雾任务一定会建 `FogEpisodeDiagnostics`（:731-734），于是雾关退回 **Python 逐决策循环**：
   白天 3 条命令/局，雾关约 **1780 条/局**。引擎进程约 **47%** 的机时在等 stdin。
2. **第二块是逐决策诊断本身。** `FogEpisodeDiagnostics.sample()` 每个决策多发一条
   `PRIV`（`privileged_state()`，`python/pvz_env.py:646`）并额外解析一份大 JSON：
   本机实测 **0.48 s/局**（PRIV）+ **0.33 s/局**（全部 JSON 解析，其中大头是 PRIV 回复），
   cProfile 里 `pvz_fog.sample` 占雾关 Python 墙钟的 **39%**。
   把诊断置空、仍走 Python 循环，L31 从 1.15 → 0.70 s/局。
3. **雾真正的逐 tick 开销只有一处：** 等待循环每 tick 调
   `EnvironmentPublicWaitState` → `EnvironmentZombieVisible` → `EnvironmentZombieFogOpacity`
   （3×5=15 个采样点）→ `EnvironmentFogOpacityAt`（每个点扫 10×7 雾格）。
   原生整局路径里它占引擎样本 **21.7%**（`FogOpacityAt` 自身 20.9%），白天约 **0.0–2.1%**。
4. **让雾关走原生整局，雾关立刻回到白天量级：**
   L31 1.15→**0.38**、L32 2.30→**0.71**、L33 0.90→**0.28**、L34 1.30→**0.39** s/局（−67%～−70%）。
   代价是 Python/C++ 两个前向在 **1e-5 容差**内的数值差会让极少数近似并列的 argmax 翻转：
   20 个（种子, 任务）组合里 13 个末状态哈希逐位相同，1 个种子 986 个决策里有 **1 个**决策不同。
   **采用前必须在 45 种子门禁上做 A/B**，不能直接换。
5. **再往上还有两块便宜钱，但都要验证：** ①快路径里
   `env.rollout(..., sim_skip=0)` 没有传现成的 `ENV_SKIP_HEIGHT`，植物高度偏移
   （`Plant::UpdateReanim` → `PlantDrawHeightOffset` → `GetFlowerPotAt`）占原生引擎 **23%**；
   ②`EnvironmentWaitEvents::Run` 在 `condition==0`（纯 timeout）时仍然每 tick 读公共状态
   （`EnvironmentWaitEvents.h:125,136`），这一项在雾关占 **~22%**、白天几乎为 0。
   两项都通过现有校验脚本（`scripts/sim_skip_agreement.py`）验证后才能启用。

**估算：** 只做“雾关走原生整局”（最小改动）门禁雾关 **3.35 → ~1.0–1.1 s/局**；
再叠加两项引擎优化可到 **~0.7–0.9 s/局**。雾区机时占门禁 50.1%，按此估算同面板同单元数
门禁墙钟约 **−35%～−40%**。

---

## 1. 方法

### 1.1 口径

- 任务：`task101_l31/l32/l33/l34_fog_selected`（v5 门禁雾区 9 项中的前 4 项，L31–L34），
  对照 `task101_l7_day_d7`（白天）。全部按门禁清单原样：同样卡组、同种子段（45 个/项，
  本报告多数用前 4/8 个种子；原生路径跑满 45/46 个）。
- 手臂：`--arms fast --dagger-beta 0`，即**纯学生、不搜索**。`searched_per_game` 实测 0.0，
  所以本报告里没有“搜索续走”的成本——门禁也不跑搜索。
- 计时：`search_play` 的 `mean_seconds` 是 `round(time.time()-started, 1)`，0.1 s 粒度；
  小值用 8 或 45 局平均；另有分项计时驱动（见 1.3）。
- 机器负载：分析期间 load average 1.7–2.5，本机还有别的 agent 在跑训练（未干预）。
  因此**绝对值有 ±10–20% 抖动**，结论依赖同一时段内的成对比较；四关、多种子方向一致。

### 1.2 引擎与重建

`build/pvz-portable` 原二进制比源码旧（10-25 vs 10-26），先重建：

```bash
cmake --build build --target pvz-portable -j 4
```

### 1.3 工具

- **Python 侧**：fork 出的 worker 里 `cProfile`（父进程只看到 `multiprocessing.wait`，
  必须在 worker 的 `_play` 外层 `enable/disable` 并在 `finally` 落盘）。
- **分项计时**：monkeypatch `PvZEnv._command` / `privileged_state` / `json.loads` /
  `_fast_action` / `FogEpisodeDiagnostics.sample` 的计时器，按局落 JSONL。
- **C++ 侧**：macOS `sample <engine-pid> 8 -file …`，1 ms 采样；分别采
  ①Python 循环 + 诊断、②Python 循环 + 诊断置空、③原生整局雾关、④原生整局白天关。
- **原生路径实验**：`inspect.getsource(search_play._play)` 后把快路径条件里的
  `and fog_diagnostics is None` 删掉再 exec 回模块——**只在 /tmp 驱动里模拟，未改仓库源码**。
- **一致性诊断**：复用仓库自带的 `fog_timeline.Timeline` 观察器，在每个决策额外发
  `ROLLOUT_PREDICT`，逐决策比较原生动作与学生动作。

### 1.4 复现命令（单进程，`ulimit -f 2000000`、`timeout 300`）

```bash
python3 scripts/search_play.py \
  --rollout-policy .claude/worktrees/archbench/artifacts/archbench/r36/model.pt \
  --value artifacts/hard_mac/value_v4/model.pt \
  --manifest .claude/worktrees/archbench/artifacts/archbench/r36_gate/manifest.json \
  --task-id task101_l31_fog_selected --seeds 8 --workers 1 --arms fast --dagger-beta 0 \
  --out /tmp/fogprof/runs/base8_l31
```

原生路径：同上，入口换成 /tmp 驱动（见 §1.3）。

---

## 2. 基线：每局、每决策、每 tick

| 任务 | 基线（诊断开，Python 循环） | 诊断置空（仍 Python 循环） | 原生整局 |
|---|---|---|---|
| L31 雾 | **1.15 s**（8 种子） | 0.70 s | **0.38 s**（45 种子） |
| L32 雾 | 2.30 s（4） | 1.10 s | **0.71 s**（8） |
| L33 雾 | 0.90 s（4） | 0.50 s | **0.28 s**（8） |
| L34 雾 | 1.30 s（4） | 0.60 s | **0.39 s**（8） |
| L7 白天 | 0.42 s（4） | — | **0.43 s**（46） |
| L8 白天 | 0.30 s（4） | — | — |

单位成本（用 `final_tick` / `decisions` 归一）：

| 配置 | µs/tick | µs/决策 | 命令数/局 |
|---|---|---|---|
| L7 白天，原生 | 6.8 | 392 | ~3 |
| L31 雾，原生 | 8.6 | 429 | ~3 |
| L32 雾，原生 | 9.9 | 487 | ~3 |
| L33 雾，原生 | 7.5 | 373 | ~3 |
| L34 雾，原生 | 8.7 | 430 | ~3 |
| L31 雾，Python 循环 + 诊断 | **26.0** | **1292** | **~1780** |
| L7 白天，原生（对照） | 6.8 | 394 | ~3 |

三点读法：

1. **原生路径下雾只比白天贵 1.8–3.1 µs/tick**，与 §4 里可见性扫描的 21.7% 对得上；
   其余差距全部来自 Python 循环（4 倍 tick 成本）和逐决策诊断。
2. 雾关决策数并不特别多（890 vs 白天 1080）；**贵在每决策 1.3 ms vs 0.4 ms**。
3. 门禁雾关 3.35 s/局 vs 白天 0.60 s/局（ratio 5.6）比本机 L31–34（ratio 2.7）更悬殊，
   因为门禁雾区含 L36–L40（40 波、僵尸更多、雾可见性扫描更贵），本报告未剖析这 5 项（见 §7）。

---

## 3. Python 侧热点

cProfile，L31 雾，8 局，7 120 个决策，14.38 s（含 profiler 开销，绝对值偏高，看占比）。

| 函数 | tottime | cumtime | 调用数 | 说明 |
|---|---|---|---|---|
| `pvz_env._read_message` (376) | 7.46 s（52%） | 11.47 s | 14 266 | 阻塞读引擎 stdout |
| `json.decoder.raw_decode` | 3.80 s（26%） | 3.85 s | 14 274 | 解析引擎回复 |
| `pvz_fog.FogEpisodeDiagnostics.sample` (32) | 0.05 s | **5.56 s（39%）** | 7 128 | 每决策诊断 |
| `pvz_env.privileged_state` (646) | 0.00 s | 5.49 s（38%） | 7 128 | 每决策多发一条 `PRIV` |
| `EventWaitEnv.step` (16) | 0.04 s | 6.34 s（44%） | 7 120 | 每决策一次往返 |
| `search_play._fast_action` (154) | 0.07 s | 1.99 s（14%） | 7 120 | 学生前向（torch） |
| `pvz_rollout_policy.act` (194) | 0.04 s | 1.55 s | 7 120 | 前向 + 选行动 |
| `pvz_student_observation.features` (143) | 0.02 s | 0.82 s | 7 120 | v2 历史特征 |
| `render_episode._frame` (130) | 0.11 s | 0.28 s | 7 128 | 观测 → 帧 |

对照白天原生路径（8 局）：`_read_message` 只有 26 次、4.50 s（等引擎），
`json.raw_decode` 34 次、3 ms。

**分项计时均值（每局）：**

| 项 | 雾 L31 | 白天 L7 |
|---|---|---|
| 命令往返 `_command` | 1.023 s / 1 783 次 | 0.398 s / **3 次** |
| 其中 `privileged_state` | **0.477 s / 891 次** | 0 |
| `json.loads` 合计 | 0.332 s / 1 784 次 | 0.0003 s / 4 次 |
| `FogEpisodeDiagnostics.sample` | **0.479 s / 891 次** | 0 |
| `_fast_action`（torch 前向） | 0.116 s / 890 次 | 不调用（C++ 内） |

注意这些项互相嵌套（`sample` 包含 `privileged_state` 包含一次 `_command`），不能相加；
但可以读出：**诊断 ≈ 0.48 s/局，纯 Python 循环的其余开销 ≈ 0.3 s/局，引擎 ≈ 0.35 s/局。**

---

## 4. C++ 侧热点（macOS `sample`，1 ms）

### 4.1 Python 循环 + 诊断（L31，8 局，总 7 103 样本）

| 符号 | 自身样本 | 占比 |
|---|---|---|
| `__read_nocancel`（等 stdin） | 3 367 | **47.4%** |
| `LawnApp::EnvironmentFogOpacityAt` | 574 | 8.1% |
| `Board::GetFlowerPotAt` | 389 | 5.5% |
| `Reanimation::Update()::$_0` | 238 | 3.4% |
| ostream/locale/printf 相关（xsputn、sentry、pad、locale、num_put、dtoa、vfprintf…） | **≥850** | **≥12%** |
| `LawnApp::EnvironmentObservation` | 37 | 0.5% |
| `LawnApp::EnvironmentPublicWaitState` | 25 | 0.4% |

（`EnvironmentObservation` 子树累计 1 486，占 **21%**：每决策把整块棋盘 + 雾通道序列化成 JSON。）

### 4.2 Python 循环 + 诊断置空（L31，总 5 581 样本）

| 符号 / 分支 | 累计或自身 | 占比 |
|---|---|---|
| 等 stdin（`getline` 分支） | 2 777 | 49.8% |
| `EnvironmentWaitUntil` / `EnvironmentWait` | 2 827 / 1 960 | 50.6% / 35.1% |
| `Board::UpdateGame` | 1 623 | 29.1% |
| `EnvironmentFogOpacityAt` | 901 | 16.1% |
| `EnvironmentPublicWaitState` | 850 | 15.2% |
| `EnvironmentZombieVisible` | 842 | 15.1% |
| `PlantDrawHeightOffset` → `GetFlowerPotAt` | 576 | 10.3% |

### 4.3 原生整局，雾 L31（45 局，总 5 976 样本）

| 符号 / 分支 | 累计或自身 | 占比 |
|---|---|---|
| `EnvironmentWaitUntil` | 5 588 | **93.5%** |
| └ `Board::UpdateGame` | 3 543 | 59.3% |
| └ `Plant::UpdateReanim` | 2 372 | 39.7% |
| └└ `PlantDrawHeightOffset` | 1 389 | **23.2%** |
| └└└ `Board::GetFlowerPotAt` | 1 233 | 20.6% |
| └ `EnvironmentPublicWaitState` | 1 295 | **21.7%** |
| └└ `EnvironmentZombieVisible` | 1 218 | 20.4% |
| └└└ `EnvironmentZombieFogOpacity` → `EnvironmentFogOpacityAt`（自身 1 205） | 1 249 | 20.9% |
| `EnvironmentRollout::Predict`（学生前向） | 232 | **3.9%** |
| 等 stdin | 55 | 0.9% |

### 4.4 原生整局，白天 L7（46 局，总 5 744 样本）

| 符号 / 分支 | 累计或自身 | 占比 |
|---|---|---|
| `EnvironmentWaitUntil` | 4 835 | 84.2% |
| └ `Board::UpdateGame` | 4 089 | 71.2% |
| └ `Plant::UpdateReanim` | 2 503 | 43.6% |
| └└ `PlantDrawHeightOffset` → `GetFlowerPotAt` | 400 / 371 | 7.0% / 6.5% |
| └ `EnvironmentPublicWaitState` | 119 | **2.1%** |
| `EnvironmentZombieVisible` | 1 | **0.0%** |
| `EnvironmentRollout::Predict` | 641 | 11.2% |

> 说明：同一符号在多个调用点出现时我按调用点求和，递归的 `Reanimation::Update` 会超过 100%，
> 所以 `Reanimation` 不单列占比；表中给出的都是非递归、可解释的子树。
> 白天 `Predict` 占比更高不是因为前向更贵，而是**白天每 tick 没有雾扫描**，
> 同样 8 s 采样窗口里决策次数更多（雾关每 tick 更贵，墙钟被引擎吃掉）。

### 4.5 读法

- **雾的额外逐 tick 成本 = 一次可见性扫描**：每 tick 对每只僵尸取 3 列 × 5 行 = 15 个点，
  每个点 `EnvironmentFogOpacityAt` 扫 10×7 雾格（`src/LawnApp.cpp:1799`、`:1767`）。
  原生路径 21.7%，Python 循环 15.2%，白天 0–2.1%。
- **`PlantDrawHeightOffset` 不是雾特有的，但在雾关更贵**：它按植物数 × 每 tick 调用，
  内部 `Board::GetFlowerPotAt` 还要扫棋盘。雾关卡组 10 张、局内植物 30+ 株，
  白天 L7 卡组只有 4–7 张、几乎不补种（`replants_per_game` 0），所以占比 23% vs 7%。
  该函数在 `Plant::UpdateReanim`（`src/Lawn/Plant.cpp:2760`）里受
  `ENV_SKIP_HEIGHT`（`src/PvzpLib/PvzpCommon.h:190`）控制，但快路径没传 mask。
- **学生前向在 C++ 里很便宜**：原生路径 3.9%（雾）/11.2%（白天），
  按每局折算 ≈15–20 µs/决策；Python torch 前向 ≈280 µs/决策（`_fast_action` 0.25 s/890）。
  也就是说原生快路径同时省掉了往返、JSON 和 torch。
- **JSON 只在 Python 循环里贵**：ostream/locale 自采样 ~12%，加上 `EnvironmentObservation`
  子树 21%（诊断样本）。原生路径只剩每局 1 次终局观测。

---

## 5. 雾天多出来的时间到底在哪

把 L31 基线 1.15 s/局拆开（用 §3、§4 的实测比例，误差 ±0.1 s）：

| 组成 | 估算 | 机制 |
|---|---|---|
| 逐决策诊断（PRIV + sample + 其 JSON） | **0.45 s（39%）** | `pvz_fog.py:32` → `privileged_state()` 每决策一次额外往返 |
| Python 循环其余开销（stdin 等待、每决策完整 obs 的 JSON、torch 前向、帧构造） | **0.32 s（28%）** | `search_play.py:775-827` 逐决策 `step()` |
| 引擎模拟 | **0.35 s（30%）** | 其中雾可见性扫描 ≈0.08 s（22%），植物高度偏移 ≈0.08 s（23%），其余是棋盘更新 |
| 其中 1 次终局观测 / 其他 | ~0.03 s | |

白天 0.42 s/局同理 = 引擎 0.40 + 循环开销 ~0.02（只 3 条命令）。

**因果链（重要）：**

```
任务带 fog_observation 且 obs 有 fog
  → search_play.py:731 建 FogEpisodeDiagnostics
  → search_play.py:764 快路径条件 fog_diagnostics is None 不成立
  → 走 Python 逐决策循环：每决策 1 次 WAIT_EVENT_V1 + 1 次 PRIV（诊断）+ 全量 JSON
  → 每 tick 的等待循环里还多一次可见性扫描（雾特有）
```

**不是搜索续走。** 门禁 `--arms fast`：`_search` 一次都没跑（`searched_per_game` 0.0），
`ROLLOUT`/`branch_snapshot` 也没有出现在这 8 局的 profile 里。

---

## 6. 对照实验与等价性

### 6.1 诊断 A/B（仍走 Python 循环，只把 `sample/compare/planted/result` 置空）

| 任务 | 诊断开 | 诊断置空 | 降幅 |
|---|---|---|---|
| L31 | 1.15 s | 0.70 s | −39% |
| L32 | 2.30 s | 1.10 s | −52% |
| L33 | 0.90 s | 0.50 s | −44% |
| L34 | 1.30 s | 0.60 s | −54% |

胜负与波数完全不变（同一批种子），因为诊断只读、不参与动作。

### 6.2 原生整局 A/B

| 任务 | 基线 | 原生 | 降幅 | 结果是否一致 |
|---|---|---|---|---|
| L31 | 1.15 s | 0.38 s | −67% | 45/45 胜；前 8 种子 6 个哈希相同 |
| L32 | 2.30 s | 0.71 s | −69% | 前 4 种子全胜；2 个哈希相同 |
| L33 | 0.90 s | 0.28 s | −69% | 前 4 种子全负；3 个哈希相同，1 个不同 |
| L34 | 1.30 s | 0.39 s | −70% | 前 4 种子全负；2 个哈希相同 |

**等价性细节。** 20 个（种子, 任务）组合里 **13 个末状态哈希逐位相同**；
不同者的分叉都很晚、且多发生在近似并列处。逐决策对照（L31 种子 100010834）：
986 个决策只有 **1 个**不同——

```
decision 204, tick 10860, wave 4, sun 30
native  : plant packet 0 → (row 0, col 6)
student : plant packet 0 → (row 1, col 5)
```

仓库自己的 Python/C++ 前向对齐测试只要求 logits 在 `atol=1e-5` 内一致
（`python/test_identity_cpp.py:93`）。两个实现在近似并列时翻转 argmax 是**预期行为**，
不是原生路径的 bug；但它意味着**换路径会改变门禁结果**，必须先 A/B。

---

## 7. 优化方向与预计收益

| # | 方向 | 机制 / 落点 | 预计收益 | 风险与验证 |
|---|---|---|---|---|
| 1 | **门禁不建 fog 诊断（最小改动）** | `search_play.py:731` 改为按开关创建（如 `--fog-diagnostics`），门禁默认关 | 本机 −0.45 s/局；门禁雾关约 **−1 s/局**（决策数是本机 2×） | 只影响 `games.jsonl` 的 `fog_diagnostics` 字段（门禁不需要）；训练采集保留开关 |
| 2 | **雾关走原生整局** | 让 #1 生效后，快路径条件自然满足；`env.rollout(None,0)` 全在 C++ | 本机 1.15→0.38（−67%）；门禁雾关 **3.35→~1.0–1.1 s/局** | 1e-5 数值差会翻转极少数近似并列决策（17/24 逐位相同，1/986 决策不同）。**必须 45 种子门禁 A/B**，并抽样核对 `state_hash` |
| 3 | **快路径传 `sim_skip=ENV_SKIP_HEIGHT`（4）** | `search_play.py:764` 的 `env.rollout` 目前 `sim_skip=0`；搜索 rollout 默认已是 61 | 原生引擎 −23%（雾 L31）→ 0.38→**0.29 s**；门禁再省 **~0.2 s/局** | "视觉位"不能靠读代码断言。用 `scripts/sim_skip_agreement.py games --mask 4 --seeds 64` 比对结果/首分叉 |
| 4 | **`condition==0` 时跳过每 tick 公共状态** | `EnvironmentWaitEvents.h:115` 的 `Run()` 无条件 `readPublic`（:125/:136）；纯 timeout 只需要 tick 计数 | 原生雾 −21.7% → 0.38→**0.30 s**；白天只 −2% | 事件语义不变（condition 0 只等 max_ticks）；改 C++，需跑 `python/test_cpp_rollout.py -k wait` 与 3000-tick 一致性 |
| 5 | **雾不透明度按 tick 缓存** | `EnvironmentFogOpacityAt` 每点扫 10×7 雾格；同一 tick 内多消费者会重复算同一坐标 | 保守估计再省 10–15% 引擎时间 | 坐标是像素级、每 tick 都在动，按坐标 memo 只能吃到同 tick 重复；要精确，不能近似 |
| 6 | **减少可见性采样点** | 3×5 → 更少 | 不建议 | 会改变 `kFogVisibilityThreshold=0.925` 标定过的语义（`LawnApp.cpp:1843` 注释：改动需重新标注） |
| 7 | **更快 JSON / 复用缓冲** | 只在 Python 循环路径有意义 | 单条 obs 序列化占 ~12–21% | #2 之后基本消失，优先级低 |

**#3 与 #4 可以叠加**（都作用在原生路径的引擎内部）：0.38 → ~0.23–0.25 s/局（本机 L31），
门禁雾关约 **0.7–0.9 s/局**。

**端到端口径估算（门禁，假设比例可移植）：**

- 门禁总机时 ≈ 91.7 min/单元（r36，1 单元；`gateslim` 报告），雾区 50.1% ≈ 46 min / 810 局；
- 只做 #1：雾区 −1 s/局 ≈ −13.5 min；
- 做 #1+#2：雾区 3.35→1.05 s/局 ≈ −31 min（−34%）；
- 做 #1+#2+#3+#4：≈ 0.8 s/局 ≈ −34 min（−38%）。
- 门禁墙钟（同单元数、同面板）约 **−35%～−40%**；若按 `gateslim` 的 4 单元，31 min → **~19–20 min**。

---

## 8. 局限

1. **只剖析了 L31–L34。** 门禁雾区还有 L36–L40（40 波、夜间泳池 + 雾）。它们更长、僵尸更多，
   可见性扫描的占比大概率**更高**，所以 §7 的门禁估算偏保守。
2. **本机不是台式机生产环境。** 绝对值与门禁的 3.35 s/局不可直接比；本报告只用本机比值外推。
3. **采样期间有其他训练任务**（load 1.7–2.5），样本有抖动；关键结论都做了跨任务/跨种子一致性检查。
4. **原生路径的等价性只测了 24 个组合 + 1 局逐决策对照。** 没有跑 45 种子全量，也没有跑 L36–L40；
   采用前必须做门禁级 A/B。
5. **没有剖析 teacher 搜索手臂**（`fast_search` 会跑 `ROLLOUT`/`branch_snapshot`），
   门禁不用它；但训练采集用的是它，`EnvironmentObservation`/`FogOpacityAt` 在其叶值评分里会更频繁。
6. `mean_seconds` 只有 0.1 s 精度；小样本的 ±0.1 s 抖动不改变结论方向。

---

## 9. 建议的后续任务

1. **task166c-1（最小、可立即做）**：把 fog 诊断改成显式开关，门禁关闭；在 45 种子雾区上
   确认胜负逐位不变、`mean_seconds` 降幅 >25%。
2. **task166c-2（收益最大）**：雾关走原生整局 + 门禁 45 种子 A/B。
   判定标准：整体/分区胜率差在门禁统计噪声内（雾区配对差 σ≈2.3pp），且抽样 `state_hash` 的
   差异全部能解释为 1e-5 级近似并列。
3. **task166c-3（引擎）**：`sim_skip=4/61` 与 `condition==0` 跳过公共状态，各自单独 A/B，
   用 `sim_skip_agreement.py` 与 `test_cpp_rollout.py` 守护。

---

## 附：本报告用到的临时驱动（未入库）

- `/tmp/fogprof/no_fogdiag.py`：删掉 `_play` 内 `and fog_diagnostics is None` 条件后 exec 回模块，
  让雾关满足原生快路径（只读模拟）。
- `/tmp/fogprof/timeline_run.py`：`pvz_fog.task45_engine_pids = lambda: []`（macOS 无 `/proc`）后
  调用 `search_episode.collect_search(..., timeline=...)`，逐决策比较 `ROLLOUT_PREDICT` 与学生会动作。
- `/tmp/fogprof/profile_search.py`、`time_breakdown.py`：worker 级 cProfile 与分项计时。
- 原始样本：`/tmp/fogprof/sample_fog.txt`、`sample_fog_nodiag.txt`、`sample_fog_native.txt`、
  `sample_day_native.txt`。
