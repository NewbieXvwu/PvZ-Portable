# task66：雾的可见边界要和人眼一致——按实际渲染的雾浓度判定僵尸可见

> 主控 2026-10-24 写。执行者：Mac CodeBuddy（`cb-fog` 会话）。工作树 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/fogvis`（分支 `fogvis`，从 origin/pvz-env 拉出，主控已建好）。游戏进程最多 2 个；不碰 Mac 上的 p5d 采集与 `demo` 工作树里在跑的 task64。

## 为什么
- 用户在 L33 示范时截图指出：雾的左边缘是**渐变淡出**的。最左一列雾里的僵尸，人眼能清楚看到轮廓，也能认出类型。
- 现行规则在 `src/LawnApp.cpp` 的 `EnvironmentZombieVisible`：
  - 先由僵尸 `mPosX` 换算出所在格；
  - 该格 `EnvironmentFogAmount`（即 `mGridCelFog`，0–255）**大于 0 就隐藏**。
  - 子弹、`wave_health` 也按这个口径屏蔽。
- 这比人眼严格。
- 原因在 `Board::DrawFog`：雾按格画，图像在 `x*80 + mFogOffset - 15` 处，带纹理自身的软边透明。格值 `mGridCelFog` 只是整张雾图的整体 alpha。所以：
  - 最左一列雾格值可以是 255，左半部却几乎透明；
  - 路灯花、三叶草让格值渐变下降时，中间值的雾实际上已经很淡，但仍算"隐藏"。
- 后果：
  - 人类示范是在人眼可见的条件下做的决定，模型看到的却更少。示范标签对模型来说像"透视"，这就是特权老师问题的一个小号版本。
  - 模型在雾边缘也比人吃亏。
- 盲种可以接受（主控 10-24 已表态），但观测必须和人的信息对齐：不多给，也不少给。

## 要做的
1. **算出渲染出来的雾浓度**：
   - 在引擎里按 `DrawFog` 的同一套几何（格位置、`mFogOffset`、-15 偏移、最后一列多画一张、`mGridCelLook` 选的图格），加上雾图的 alpha 通道，算出**僵尸所在位置**被雾覆盖后的合成不透明度 `fog_opacity`（0–1）。
   - 取点：用僵尸身体的代表点，例如碰撞框中心，或头部的几个采样点取最小值，写清选择理由。
   - 多张雾图叠加按 alpha 合成。只算 alpha，不需要颜色。
   - 要求：
     - 无头环境下也能算；资源里雾图的 alpha 要读得到；`IMAGE_FOG` 与 `IMAGE_FOG_SOFTWARE` 的区别要处理，按 3D 加速版本走，写明原因。
     - 随机动画（`aMotion`）只影响颜色、不影响 alpha，要核实这一点。
     - 确定性：同一状态算出同一值，不依赖绘制。
2. **量差距**：
   - 用用户三局示范的动作带重放：
     - `demo_fog_31_fume_rebaselined`；
     - `demo_fog_31_snow_m200`；
     - `demo_fog_33_cactus_snow_m200`。

     路径都在 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/artifacts/human_demo/raw/`，只读。
   - 统计所有"现行规则隐藏"的僵尸-tick，其 `fog_opacity` 的分布直方图。尤其是 opacity < 0.5 的比例，以及它们在哪些列、哪些时刻（雾边缘、路灯花或三叶草刚生效时）。
3. **请用户校准阈值**（不要自己拍脑袋定）：
   - 从上面的样本里挑约 16 个僵尸，覆盖 opacity 0.2–0.95，再加 4 个对照。
   - 用有窗口的引擎在对应时刻截图，裁出僵尸附近区域，存成一页图：网格排列，每张标编号，不标 opacity。
   - 写一份给用户的说明：每张图请回答"能看到僵尸吗 / 能认出是什么僵尸吗"。
   - 把图和说明放到 `artifacts/task66/calibration/`（不入 git），**主控转交用户**。拿到回答后，用"能认出类型"对应的 opacity 区间定阈值 τ。
4. **实现**：
   - 等主控转回用户的回答后，`EnvironmentZombieVisible` 改为 `fog_opacity < τ`。子弹按同一口径；`fog_amount` 观测保持原样，它是格值，供记忆等用途。
   - 观测语义变了，要按惯例升版本：新的 fog 观测族名或版本号，旧存档和旧任务的重放不受影响。雾区族目前不在 qc 训练集合里，影响范围小，但仍须登记 DESIGN。
   - task58（记忆探针）以"现行规则隐藏"作标签；改口径后要通知主控，由主控决定 task58 是否换标签。
5. 测试：确定性、无头可算、旧族逐位不变、新族边界行为。结果写本文件，提交并 `git push origin fogvis`。**主控合并、决定部署。**

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`（截图时除外）。
- 禁止 `git add -f`；图片、`.pt`、`.npz`、`.log` 不进 git；不写字面密钥；不手写打法规则。阈值来自用户标注，是观测口径，不是打法规则。
- 每完成一小步就提交、push，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

### 1. 渲染不透明度（提交 54c073ea）

- `LawnApp::EnvironmentFogOpacityAt(x, y)` 复刻 `Board::DrawFog` 的每一笔：
  `IMAGE_FOG` 是 1680×190、8 格（`resources.xml` `cols="8"`），每格 210×190；
  摆放在 `x*80 + mFogOffset - 15`、`y*85 + 20`（`DrawImageCel` 收 int，浮点截断照做）；
  最后一列多画一张（x 与 x+80）；图格由 `mGridCelLook[x][y%6] % 8` 选。
  每笔的覆盖 = 图格 alpha × 格值/255，多笔按 `1-Π(1-a)` 合成。
- **为什么按 3D 加速版（`IMAGE_FOG`）走**：软件版 `IMAGE_FOG_SOFTWARE` 只在
  `DrawFog` 的 `!aIs3DAccelerated` 分支用，本移植的窗口/无头运行都不走那条路；
  两者几何也不同（3 列、+10/+3 偏移、尺寸不同）。用户截图里的雾就是 3D 版。
- **aMotion 已核实不影响 alpha**：它只进 `aColorVariant` / `aLightnessVariant`，
  `SetColor(..., aFadeAmount)` 的第 4 个参数才是 alpha，与 `mMainCounter` 无关。
- **无头可算**：`-env` 下所有图只读尺寸（task32），所以新增
  `ResourceManager::LoadAnalysisImage`：临时关掉 `gSizeOnly` 把这**一张**图的像素读回来
  （alpha 走 `ImageLib::GetImage` 的自动找伴生图，即 `images/fog_.jpg`，与渲染同一份），
  抽走 alpha 后释放，不动资源缓存、资源组和任何观察。
- **取点**：采样框 = 碰撞框（`GetZombieRect()`，普通僵尸 42×115）左右各外扩 h/4、
  上方外扩 h/8 —— 覆盖头、头饰、手持物（精灵 120×120，碰撞框只占中间一条）。
  框内 3 列 × 5 行取 15 个点，**取最小值**：人只要能看清身上任何一块就能认出僵尸。
  上 2 行单独算「头带」，下 3 行算「身体」，于是"只有头露出"可以和"什么都看不见"分开。
- **交叉核对**：`scripts/task66_fog_reference.py` 是独立第二实现 —— 自己解析
  `main.pak`（XOR 0xF7）、自己解 `images/fog_.jpg` 的蓝通道（引擎
  `LoadAlphaImage` 取的就是蓝通道）、自己按几何叠加。对 12000 tick 的 633 次
  比较（全体/头带/身体），**最大误差 0.0005**，就是千分位取整的差。
- **公开观察逐位不变**：对拍未修改引擎（`/tmp/t66base` 从 HEAD 的 `git archive` 构建），
  `fog_observation` 为 human / xray / 不启用三种模式各 36 个观察，**全部相同**。

### 2. 差距统计（三局示范重放，动作带逐 tick 校验 state_hash）

`scripts/task66_fog_opacity.py replay` 重放三局示范（`demo_fog_31_fume_rebaselined`、
`demo_fog_31_snow_m200`、`demo_fog_33_cactus_snow_m200`），每个 tick 的
state_hash 与存档一致（否则直接报错）。汇总在 `artifacts/task66/summary.json`。

| 指标 | 数值 |
|---|---|
| 僵尸-tick 总数 | 764,207 |
| 现行规则隐藏的僵尸-tick | 704,423（92.2%） |
| 其中 opacity < 0.5 | 199,397（28.3% of hidden） |
| 其中 opacity < 0.3 | 172,149（24.4%） |
| 其中 opacity < 0.2 | 107,352（15.2%） |
| 其中 opacity < 0.05（几乎全透） | 63,664（9.0%） |
| **"只有头露出"**（头带 < 0.5 ≤ 身体） | 112,365（15.9%） |
| 隐藏发生的列 | col 5: 27,374 / col 6: 102,771 / col 7: 202,668 / col 8: 371,610 |
| 隐藏时的格值 | 200: 92,451 / 255: 611,972 |

按 x 的平均不透明度（只在 x≤780 的僵尸上算，20 px 一档）：

```
x    400  420  440  460  480  500  520  540  560  580  600  620  640  660  680  700  720  740  760  780
op   .25  .35  .44  .41  .64  .78  .80  .82  .83  .78  .83  .81  .83  .82  .83  .82  .80  .72  .42  .32
```

- 雾的"墙"在 x≈480–500 之间陡起（0.44 → 0.78），不是一条硬边；x≈740 之后又开始下降，
  是最后一列图格右半边的软边。**这正是用户说的"左边缘渐变淡出"**。
- x=400–460 的平均值不为 0，是开局 200 tick 的雾涌入动画（`mFogBlownCountDown`）——
  那段时间雾确实铺满全屏，模型也算得出来。
- 结论：现行"格值 > 0 就隐藏"在这三局里隐藏了 70 万个僵尸-tick，其中近三成的人眼覆盖
  不到一半，另有 11 万个是"只有头露出"。

### 3. 校准图（等待用户标注）

工具：`scripts/task66_calibration.py`（`select` 选样 → `capture` 截图 → `panel` 拼页）。

- **池子用全部僵尸-tick**，不只是现行规则隐藏的那些。要校准的正是"规则和人眼哪里不一致"，
  先按规则过滤会把边界样本剔掉。
- **23 格**：16 格按 0.05 一档覆盖 opacity 0.20–0.95；3 格是"只有头露出"（头带 < 0.40、
  身体 > 0.80，也就是用户截图里那种）；4 格对照（2 格 ≥ 0.98 浓雾、2 格 0 无雾）。
- **唯一性**：同一个僵尸最多出现一次；同一局两格至少隔 90 tick，避免"同一只僵尸的下一帧"。
- **优先左缘**（x≈460–560，僵尸完整在板内、雾墙横跨身体）；右缘（x>740）只在没有左缘候选时
  才用 —— 那类僵尸有三分之二在板外，人眼判不了，拿来做校准是坏样本。
  （实测 0.45–0.70 这一段只在 x>750 的右缘和 x≈475–525 的左缘出现，左缘样本够用。）
- **截图走引擎自己的帧缓冲**：窗口模式（不调 `begin_demo()`，它会把 app 强制成无头）、
  `SCREENSHOT` 命令在 `Flush()` 的 swap **之前** `glReadPixels`，所以桌面休眠/锁屏也拍不到黑屏。
  每格是 240×240 棋盘像素的原图裁剪。
- **面板不写 opacity**，编号用固定种子（66）打乱 —— 按浓度排序等于把答案写在纸上。
- 产物（不入 git）：`artifacts/task66/calibration/panel.png`、`answer-sheet.md`、
  `selection.json`、`shots.json`、`01.png`–`23.png`。

**一个坑（记下来，别处也会踩）**：窗口模式的回放**不能**用存档里的 `state_hash` 校验。
同一盘棋，无头运行（开不开 demo 都一样）和有窗口运行在 tick 1800 的状态字节只差 3 个 u32
（`SAVE4_CHUNK_REANIMATIONS` 里相邻的三个字段：无头是 `-1`、有窗口是 `0`），
其余 52,044 字节完全相同 —— 无头不绘制，某个 reanim 的定义没被初始化，属于表现层差异，
不是棋局分歧。所以 `capture` 改成拿该 tick 的**僵尸表**（id / row / type / x / fog_opacity）
与存档逐条比对来确认轨迹，23 个目标 tick 全部一致（不一致直接报错，不会悄悄拍错帧）。

### 4. 等待用户标注（停在这里）

第 4 步（`EnvironmentZombieVisible` 改成 `fog_opacity < τ`、升观测族版本、登记 DESIGN、
通知主控 task58）要等主控把用户对 23 格的回答转回来。τ 取"能认出类型"的最低 opacity 区间
的下沿；"勉强"那一档单独记，看是否落在阈值附近。


### 3b. 用户标注（主控转达，2026-10-24）

- **看不到**：09、13、14；其余 20 张都看得到，包括只露头的 01 / 08 / 23。
- 按 `shots.json` 的不透明度排开，切分很干净：看得到的最高 0.900（06），看不到的最低 0.950（09）。
  **主控定 τ = 0.925**（两者中点），记作 `fog_opacity < 0.925 → 可见`。
- 现行格值规则会把这 20 张里的一半藏起来（01、16、08、02、23、21、19、04、10、06）。
- 用户另提截图偏橙；不影响标注，本次不修。

### 4. 实现（提交 2ec6eb5d）

- `EnvironmentZombieVisible`：`human_v2` 族下 `EnvironmentZombieFogOpacity(zombie) < 0.925`
  （`kFogVisibilityThreshold`，全身 15 点最小值）；`human` 族仍走格值规则，旧任务与旧存档逐位不变。
- **子弹同口径**：新增 `EnvironmentProjectileVisible`，在 `GetProjectileRect()` 上 3×3 取最小值比同一个阈值；
  完整观测与 v2 观测两个过滤点都改成调它，`human` 族的格值分支原样保留。
- **观测族**：`fog_observation` 增加 `human_v2`（reset 尾串 `fog_human_v2`），`human` / `xray` 语义不变；
  引擎里 `humanFogObservation` 这个 bool 换成 `EnvironmentFogObservation{None,Cell,Opacity}`。
  走族名而不是抬 `OBSERVATION_VERSION`，因为要保证旧存档/旧任务重放逐位不变。
- **诊断字段挪位置**：`fog_opacity` 等五项原先直接挂在 privileged 的僵尸对象上，这破坏了
  `test_fog_observation.py` 的两个不变式（公开帧 == privileged 减去隐藏项）。改为 privileged 帧里
  独立的 `fog_zombies` 数组（按 id 对应），Python 侧用 `pvz_env.fog_diagnostics(state)` 取。
- DESIGN.md §14.25 登记（新族名、阈值来源、影响面）。

### 5. 测试（提交 2ec6eb5d）

| 检查 | 结果 |
|---|---|
| `python/test_fog_opacity.py` | 7 passed（新增 4 个） |
| `test_fog_observation` / `test_fog_memory` / `test_replay_task_spec` / `test_search_episode` / `test_shared_helpers` | 44 passed, 9 subtests passed |
| 旧族逐位不变（`/tmp/t66base` = 改动前引擎） | human / xray / None 各 36 个公开观察，全部相同 |
| 参考实现交叉核对（独立解析 pak） | 0 处不符，最大误差 0.00049 |

新增的 4 个测试：

1. `test_human_v2_hides_exactly_what_the_threshold_says` —— 20000 tick 里每个僵尸-tick 的
   "在不在公开帧"都必须等于 `opacity < 925`，且两侧都要被踩到（否则断言是空的）。
2. `test_human_v2_shows_what_the_cell_rule_hides` —— 同一 tick、同一批僵尸，格值规则藏起来的
   那个 `human_v2` 必须露出来，且它的 opacity 确实小于阈值。
3. `test_human_v2_applies_the_same_rule_to_projectiles` —— 预置豌豆射手，找出格值规则藏掉子弹的
   tick，`human_v2` 必须多露出来（实测 tick 2500 那颗 x=598.5 的豌豆）。
4. `test_fog_family_only_changes_what_is_reported` —— 三个族跑同一串动作，state_hash 必须一致：
   可见性是呈现层，不改棋局。

## 问题

## 主控追加（2026-10-24，用户第二张截图）
用户在 L33 的截图里，雾很深的右上角，**第 0 行僵尸的头部露在雾的上沿之外**，人眼看得很清楚。因此：
- 可见性不能只看身体中心。要在僵尸整个绘制区域上多点采样：头、头饰、手持物都要覆盖，区域按 reanim 或碰撞框估计。
- 判定规则：任一部位的不透明度低于 τ，就算可见。
- 统计里单列"只有头部露出"这一类。
- 校准图里要放 2–3 张这类样本。

## 用户标注（2026-10-24，主控转达）
用户直接在图上打勾 / 叉，只回答了"能否看到"，没有区分类型。
- **看不到**：09、13、14。
- **看得到**：其余 20 张。包括 01、08、23 这几张只露头的；也包括 cell=255 或 200、按现行规则会被隐藏的 01、16、08、02、23、21、19、04、10、06。

按 shots.json 的不透明度（0–1000）排开，可以干净地切开：
- 看得到的最高值是 900（06）；
- 看不到的最低值是 950（09）。

**主控定 τ = 925**，取两者中点，记作 `fog_opacity < 925 → 可见`。
- 现行规则（格值 > 0 即隐藏）会把用户看得到的 20 张里的 10 张藏起来。
- 新规则下，只有深雾里身体和头都被盖住的僵尸才隐藏。

用户另提：截图颜色有偏差（偏橙）。现在不修，不影响标注。

请按任务书第 4、5 步实现，结果写本文件，提交并 push origin fogvis。
- **task58（记忆探针）需要主控决定**：它以"现行规则隐藏"作标签。`human` 族语义未变，所以既有标签
  仍然自洽；但若要把探针换到 `human_v2`，标签必须重打（新口径下同一批 tick 里约有一半不再隐藏）。
  本任务只提供口径，不擅自改 task58 的标签。
- **截图偏橙**：用户指出校准图颜色偏橙。已核对不是截图链路的问题（帧缓冲原样输出），更像是
  夜景/雾色调色与用户显示器配置的差异；不影响"能否看到"的标注，未修。若以后要用截图做颜色相关的
  判断，需要先查这条。
- **τ 的适用面**：23 张样本只覆盖 L31/L33 的雾区（草地夜间），没有覆盖屋顶/泳池与"路灯花开雾后"的
  瞬态。若以后在别的场景发现口径不对，按同样流程重新标定，不要就地调数字。
