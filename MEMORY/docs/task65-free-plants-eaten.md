# task65：被吃植物里把零费植物单独计一列

> 主控 2026-10-24 写。执行者：台式机 CodeBuddy（`cb-qcalloc` 会话）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/eaten`（分支 `eaten`，从 origin/pvz-env 拉出，主控已建好）。游戏进程最多 2 个；**不碰 tmux `qc`、不重建生产引擎 deploy3**。

## 为什么
- 用户 10-24 指出：夜间关后期，零费的小喷菇严重溢出，可以毫无顾忌地往前线种，当垫材挡僵尸。被吃掉是有意的打法，不是损失。
- 现在的"被吃植物"统计（引擎 `Board::mPlantsEaten`，env 事件 `plants_eaten`）不分费用。它出现在下面这些地方：
  - 示范 meta 与各类诊断报告的 outcome；
  - `episode_query.py whatif` 排序，在丢车之后作为第二判据；
  - `render_episode.py` 的叙述；
  - `search_play.py` 帧里的 `eaten`。
- 不分费用的结果：拿小喷菇当垫材的局会被算成"损失惨重"，`whatif` 也会把这种好打法排在后面。

## 要做的
1. **引擎**：在 env 事件里新增 `plants_eaten_free`，与 `plants_eaten` 并列。`plants_eaten` 的含义不变，是全部被吃的植物；新字段只计其中费用为 0 的。
   - 费用用游戏自己的函数判断，例如 `Plant::GetCost` / `GetCostForSeedType`，按种植时实际的费用算，传送带关卡另行处理；不写死植物编号。
   - 在 `mPlantsEaten++` 的各处（Board.cpp、Zombie.cpp，共约 5 处）旁边计数。
   - **不改存档格式**：新计数不进 SaveGame，存档哈希不变。参照 `branchAfter/branchBefore` 的差分做法，在每步、每个分支前后取差。要证明快照恢复、分支推演下差分仍然正确，并写测试。
2. **Python 侧**：
   - 事件字典、outcome、`totals` 里都带上 `plants_eaten_free`；
   - `human_demo.py`、`search_episode.py`、`roof_phase2.py`、`render_episode.py` 的报告里，被吃数显示为"被吃 N 株（其中零费 M）"；
   - `episode_query.py whatif` 的排序改用**付费植物被吃数**（`plants_eaten − plants_eaten_free`）作第二判据；
   - 旧存档没有新字段时，按旧口径处理，写明。
3. **验证**：用用户两局示范的动作带重放，报出 `plants_eaten` / `plants_eaten_free` 两个数：
   - `demo_fog_31_fume_rebaselined`，原 10 株；
   - `demo_fog_31_snow_m200`，原 10 株。

   两局存档在 Mac：`/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/artifacts/human_demo/raw/seed_20261031/`。需要时从 Mac 拷过来（只读，几 MB），或者在台式机上用同样的 task 设置重放；拷不过来就在"问题"里写明，由主控在 Mac 上跑。
4. 测试全过，push origin eaten，结果写本文件。**主控合并；引擎随下次部署生效。**

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则（零费由游戏自己的费用函数判定，这是统计口径，不是打法规则）。
- 每完成一小步就提交、push，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

## 问题

## 结果（台式机 CodeBuddy，2026-10-24）

### 引擎（src/）
- `Board` 新增 `mPlantsEatenFree`（Board.h，紧挨 mPlantsEaten）：被吃植物里**种植时实际费用为 0** 的株数。
- 计数收敛到一个入口 `Board::CountPlantEaten(Plant*)`（Board.cpp）：mPlantsEaten++ 语义不变，
  零费判定走 `Board::PlantingWasFree(seedType, imitaterType)`——**游戏自己的费用函数**
  `GetCurrentPlantCost`（含 Beghouled 特价、生存 endless 加价），不写死植物编号；
  传送带关卡（`HasConveyorBeltSeedBank()`）与 Easy Planting 种下不扣阳光（AddPlant 跳过
  TakeSunMoney），按"实际支付"口径一律算零费——这就是"传送带另行处理"。
- 6 处 `mPlantsEaten++`（Board.cpp 1 处 KillAllPlantsInRadius + Zombie.cpp 5 处：火豆烧行 ×2、
  碾压 Squish、进食 Gulp、Bungee 搬运）全部改走 CountPlantEaten，行为等价。
- **存档格式不变**：SaveGame.cpp 未动，`mPlantsEatenFree` 不进 SaveGame，存档哈希不变；
  也不进 `EnvironmentSnapshotHash`/`STATE_BYTES` dump（state_hash 输出与旧引擎逐位一致）。
  差分正确性靠两条：(a) 每命令/每分支本来就按 branchAfter−branchBefore 取差（差分对
  绝对值漂移天然免疫）；(b) 内存快照 `EnvironmentSnapshot` 显式携带 plantsEatenFree
  （Save/Restore），恢复后绝对值也正确——LawnApp 结算 JSON（play record）读的是绝对值。
- env 事件：每命令与 BRANCH 的 events JSON 都新增 `plants_eaten_free`（main.cpp），
  play record JSON 新增同名字段（LawnApp.cpp）。

### Python（scripts/ python/）
- `pvz_env.EVENT_KEYS` 收编新键（interned，事件 dict 键共享不受影响）。
- outcome 带 `plants_eaten_free`：human_demo（录制与反事实两条路径）、search_episode、
  roof_phase2；totals 随事件字典自动带上。search_play 帧新增 `eaten_free`。
- 报告：episode_query 总述与 render_episode 事件合计显示「损失 N 株植物（其中零费 M）」，
  render_episode 叙述行补零费标注；旧存档没有该字段时保持旧口径不显示（写明于代码注释）。
- **whatif 排序第二判据改为付费被吃**：`episode_query.paid_eaten(o) =
  plants_eaten − plants_eaten_free`（旧存档缺省 0 → 与旧排序完全一致）；
  「最好的一步」行显示付费被吃对比。

### 验证
- 引擎集成测试 `python/test_plants_eaten_free.py`（真实引擎，level 7 预置三株：付费在前、
  零费垫后，3× 僵尸量）：事件两列正确（free ≤ eaten、两列都动）；**同一父快照下分支推演
  与直放的逐键事件差分完全一致**（含新列）。零费/付费植物由 pvz_constants.plant_defs()
  的游戏费用列现解析，测试不写死编号。
- 纯 Python 单测 `python/test_plants_eaten_free_report.py` 5 项：paid_eaten 减法与旧存档口径、
  零费垫材排序优于付费、eaten_free_note 新旧显示、EVENT_KEYS 声明。
- 台式机同设置替身烟测（demo_fog_31_snow_m200 的任务设置：L31、seed 20261031、2.0×、同 deck，
  零费小喷菇往前线垫）：**plants_eaten 3 / plants_eaten_free 3 / 付费 0**，口径符合
  「垫材被吃是有意的打法」。这不是用户的两局示范。
- 测试全过：82 个测试文件逐一跑（PVZ_RESOURCE_DIR 指向台式机资源），除 2 类**既有失败**
  （与本次改动无关，已在无改动的 HEAD 引擎上复现同样错误）：test_human_demo 2 例
  （initial state hash differs / KeyError value_files——台式机 main.pak 与 Mac 资源有差异，
  演示确定性测试在这台机器跑不过）；5 个 ModuleNotFoundError（缺的模块在 HEAD 也缺）。
- 台式机构建：`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja && cmake --build build -j8`
  通过（build/ 不进 git）。SaveGame.cpp 零改动（git diff 可查证）。

### 问题
- **用户两局示范的存档只在 Mac**（`/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/
  artifacts/human_demo/raw/seed_20261031/`），台式机没有 SSH 配置可达，拷不过来——
  需主控在 Mac 上跑。合并本分支、重编引擎后，对两个存档各跑一次：
  ```bash
  python3 - <<'PY'
  import sys
  from collections import Counter
  from pathlib import Path
  sys.path.insert(0, 'scripts')
  from human_demo import replay
  from pvz_event_env import EventWaitEnv
  RESOURCE = '/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN'
  for name in ('demo_fog_31_fume_rebaselined', 'demo_fog_31_snow_m200'):
      archive = Path('artifacts/human_demo/raw/seed_20261031') / name
      class CountingEnv(EventWaitEnv):
          def __init__(self, *a, **k):
              super().__init__(*a, **k); self.totals = Counter()
          def step(self, action):
              obs, r, term, trunc, info = super().step(action)
              self.totals.update(info.get('events') or {})
              return obs, r, term, trunc, info
      with CountingEnv(RESOURCE) as env:
          replay(env, archive)
      print(name, 'plants_eaten', env.totals['plants_eaten'],
            'plants_eaten_free', env.totals['plants_eaten_free'])
  PY
  ```
  （纯重放计数，不写存档。）注意：demo 存档 meta.json 里的 totals 是录制时的旧口径，
  新列要靠重放现算。
- test_human_demo 在台式机跑不过（HEAD 也如此），Mac 上应当是过的——若 Mac 合并后同样
  出现 hash differs，才是真回归，需要另查。
