# CUDA 机制规格 03：波次调度与出怪（Wave scheduling & spawning）

## 2026-10-18 实现勘误：未来组成必须重抽

交接实现直到 f198afa 都直接读取 `waves.type[wave]`，**没有**执行下文 §5 对应的
`ReseedEnvironment → PickZombieWaves`。原先关于“已洗牌/已重抽”的描述不能作为代码证据。
现在 `DrawFutureWave` 在波次生成时按 `(futureSeed,wave)` 的独立流惰性重抽。
点数、旗帜波强制普通/旗帜、最后一波补齐 allowed 类型、首次可出波与权重均对应原版。
第 7 关无新增介绍类型、无生存模式跨波配额，所以不需要生成未用到的 30 波表。
同一候选未来的组成不受该候选在模拟中消耗的随机数影响，也不读取存档的未来组成。
`SimMechanicsTest.cpp` 检查改变存档未来表不改变生成结果，并在 30 波×256 种子上验证
点数、旗帜、首次出波与末波类型覆盖。随机流与 CPU 不同，分布仍须阶段 1 验收。

导出器还曾把 `allowed == false` 的定义整项留零。旗帜的随机权重是零，因此不在 allowed 中，
但旗帜波仍会强制生成它。现在单独填充旗帜定义：270 HP、速度 0.45、点数 1，
且保持 allowed=false。`.cudaroot` 升 v6，拒收缺该定义的旧语料。
波表重抽只实现第二周目第 7 关，导出器明确拒收其他关卡/第一周目（reason=6200）。

适用场景：冒险模式第 7 关（`mLevel == 7`，`BACKGROUND_1_DAY` 草地白天），
`EnvironmentTaskSpec.playthrough == 2`（⇒ `mFinishedAdventure == 1` ⇒ `IsFirstTimeAdventureMode() == false`），
`zombieCountMultiplier ∈ [1.3, 1.5]`，`waveCap == 0`（⇒ `mNumWaves == 30`），
本关自然出怪池恰好 = `{NORMAL(0), TRAFFIC_CONE(2), POLEVAULTER(3)}` + 由 flag 波硬插的 `FLAG(1)`。

所有引用格式为 `文件:行`。未找到的一律标注 **NOT FOUND**。

---

## 0. 常量表（全部字面值）

| 名字 | 值 | 定义处 |
|---|---|---|
| `ZOMBIE_COUNTDOWN_FIRST_WAVE` | `1800` | `src/Lawn/Board.cpp:60` |
| `ZOMBIE_COUNTDOWN` | `2500` | `src/Lawn/Board.cpp:61` |
| `ZOMBIE_COUNTDOWN_RANGE` | `600` | `src/Lawn/Board.cpp:62` |
| `ZOMBIE_COUNTDOWN_BEFORE_FLAG` | `4500` | `src/Lawn/Board.cpp:63` |
| `ZOMBIE_COUNTDOWN_BEFORE_REPICK` | `5499` | `src/Lawn/Board.cpp:64` |
| `FLAG_RAISE_TIME` | `100` | `src/Lawn/Board.cpp:69` |
| `MAX_GRID_SIZE_Y` | `6` | `src/Lawn/Board.h:43` |
| `MAX_ZOMBIES_IN_WAVE` | `50` | `src/Lawn/Board.h:44` |
| `MAX_ZOMBIE_WAVES` | `100` | `src/Lawn/Board.h:45` |
| `MAX_ZOMBIE_FOLLOWERS` | `4` | `src/Lawn/Zombie.h:29` |
| `ZOMBIE_START_RANDOM_OFFSET` | `40` | `src/Lawn/Zombie.cpp:48` |
| `WIDE_BOARD_WIDTH` | `800` | `src/GameConstants.h:30` |
| `gZombieWaves[6]`（第 7 关原始波数） | `20` | `src/Lawn/Challenge.cpp:68`（表首行第 7 列） |
| zombie 对象池上限 | `1024` | `src/Lawn/Board.cpp:77`（`DataArrayInitialize(1024U,"zombies")`） |
| `ZOMBIE_INVALID` | `-1`；`NORMAL=0, FLAG=1, TRAFFIC_CONE=2, POLEVAULTER=3` | `src/ConstEnums.h:1340-1376` |
| `ZOMBIE_WAVE_CUTSCENE=-2, ZOMBIE_WAVE_UI=-3, ZOMBIE_WAVE_WINNER=-4` | | `src/Lawn/Zombie.h:80-82` |

行选择权重字面值：`0.01f` / `0.5f` / `1.0f`（`src/Lawn/Board.cpp:2599,2603,2607`）。
平滑数组权重常数：`1e-6f`、`0.75f`、`0.25f`、clamp 下/上界 `0.01f`/`100.0f`
（`src/PvzpLib/PvzpCommon.cpp:177,188`）。
催波字面常数：`200`、`400`（`src/Lawn/Board.cpp:5356,5358`）；
催波阈值随机数：`RandRangeFloat(0.5f, 0.65f)`（`src/Lawn/Board.cpp:5388`）。
巨型波横幅：`750`、音乐/sound 刻度 `725`/`700`/`400`、`mZombieCountDown = 5`、`mZombieCountDown = 1`
（`src/Lawn/Board.cpp:5295,5299,5310,5317,5360,5366`）。

---

## 1. 波次倒计时状态机

### 1.1 字段与声明

`src/Lawn/Board.h`：`mWaveRowGotLawnMowered[6]` :158、`mRowPickingArray[6]` :163、
`mZombiesInWave[100][50]` :164、`mZombieAllowed[100]` :165、`mNumWaves` :177、
`mCurrentWave` :184、`mTotalSpawnedWaves` :185、`mZombieHealthToNextWave` :190、
`mZombieHealthWaveStart` :191、`mZombieCountDown` :192、`mZombieCountDownStart` :193、
`mHugeWaveCountDown` :194、`mFlagRaiseCounter` :207、`mLevelAwardSpawned` :206、
`mRiseFromGraveCounter` :182、`mFinalWaveSoundCounter` :213。全部 `int32_t`（`mZombiesInWave` 为 `ZombieType`，`mRowPickingArray` 为 `PvzpSmoothArray`，含 `float mWeight / mLastPicked / mSecondLastPicked`，`src/PvzpLib/PvzpCommon.h:62-69`）。

> **`mWaveCountDown` NOT FOUND in `src/Lawn/Board.h` / `src/Lawn/Board.cpp`。** 代码里只有
> `mZombieCountDown`（波与波之间的间隔计时器）与 `mHugeWaveCountDown`（flag 波横幅计时器）。
> 全仓库对 `mWaveCountDown` 的搜索（`src/**`）零命中。

### 1.2 初始化（每关一次，在 `Board::InitZombieWaves`）

`Board::InitZombieWaves` `src/Lawn/Board.cpp:1198`，由 `Board::InitLevel`（:1338，调用点 :1351）
和 `Board::InitSurvivalStage`（:1264，调用点 :1269）调用。顺序：

```
memset(mZombieAllowed, false, sizeof(mZombieAllowed));                 // Board.cpp:1200
InitZombieWavesForLevel(mLevel);                                       // :1202 → 逐类型填 mZombieAllowed
PickZombieWaves();                                                     // :1205  画整张 mZombiesInWave
PVZP_ASSERT(IsZombieWaveDistributionOk());                             // :1206
mCurrentWave      = 0;                                                 // :1212
mTotalSpawnedWaves = 0;                                                // :1213
if (IsFirstTimeAdventureMode() && mLevel == 2) mZombieCountDown = 2500*2;  // :1215-1217  （= 5000）
else if (IsSurvivalMode() && mChallenge->mSurvivalStage > 0) mZombieCountDown = 600;  // :1219-1221
else mZombieCountDown = ZOMBIE_COUNTDOWN_FIRST_WAVE;                   // :1225  ← 第 7 关走这里，= 1800
mZombieHealthWaveStart = 0;                                            // :1228
mLastBungeeWave        = 0;                                            // :1229
mProgressMeterWidth     = 0;                                           // :1230
mHugeWaveCountDown      = 0;                                           // :1231
mLevelAwardSpawned      = false;                                       // :1232
mZombieCountDownStart   = mZombieCountDown;                            // :1233  ← 与初值相同
mZombieHealthToNextWave = -1;                                          // :1234  ← 哨兵值，见 §4
```

**第 7 关的开局标量：`mZombieCountDown = mZombieCountDownStart = 1800`，`mCurrentWave = 0`，
`mZombieHealthWaveStart = 0`，`mZombieHealthToNextWave = -1`，`mHugeWaveCountDown = 0`。**
首波在第 1800 个 tick 出（推导见 §1.5）。

### 1.3 tick 位置（每 tick 恰好一次，且只在 `SCENE_PLAYING`）

`Board::Update` `src/Lawn/Board.cpp:5722` → `UpdateGame` `:5660`（调用点 `:5815`）：

```
Board::Update():
    if (mPaused) { mChallenge->Update(); ...; return; }   // :5741-5748  ← 暂停时整个 UpdateGame 被跳过
    ...
    if (mTimeStopCounter > 0) return;                       // :5791-5793
    UpdateGame();                                           // :5815
    UpdateLevelEndSequence();                               // :5818

Board::UpdateGame():
    UpdateGameObjects();                                    // :5662  plant/zombie/projectile/coin/mower 先更新
    ... fog ...
    if (mApp->mGameScene != SCENE_PLAYING && !mCutScene->ShouldRunUpsellBoard()) return;  // :5679-5681
    mMainCounter++;                                         // :5683
    UpdateSunSpawning();                                    // :5684
    UpdateZombieSpawning();                                 // :5685  ← 波次状态机
    UpdateIce();                                            // :5686
    ...
    UpdateProgressMeter();                                  // :5719
```

即：**波次计时器在“本 tick 所有实体更新之后”才递减**，且 `mMainCounter` 的自增在它之前一行。
headless 环境每 tick 走一次 `LawnApp::AdvanceLogicTick`（`src/LawnApp.cpp:1581`）。

### 1.4 `Board::UpdateZombieSpawning` 全文逐行（`src/Lawn/Board.cpp:5256-5401`）

对本 scope 相关的分支保留，其余标注不可达：

```
5258  if (mGameMode == UPSELL || mGameMode == INTRO) return;              // 不可达（环境固定 GAMEMODE_ADVENTURE）
5261  if (mFinalWaveSoundCounter > 0) { mFinalWaveSoundCounter--; if (== 0) PlaySample(SOUND_FINALWAVE); }
      // mFinalWaveSoundCounter 由 NextWaveComing() 设为 60（:5242），只播声音，不影响出怪
5270  if (mTutorialState == TUTORIAL_LEVEL_1_PICK_UP_PEASHOOTER || ... || TUTORIAL_SLOT_MACHINE_PULL) return;
      // 环境把 mTutorialState 置为 TUTORIAL_OFF（src/LawnApp.cpp:1411）→ 不可达
5276  if (HasLevelAwardDropped()) return;                // ★ 奖励已掉落 ⇒ 波次状态机彻底停摆
      // HasLevelAwardDropped() = mLevelAwardSpawned || mNextSurvivalStageCounter > 0 || mBoardFadeOutCounter >= 0
      //   (src/Lawn/Board.cpp:5191-5194)
5279  if (mRiseFromGraveCounter > 0) { mRiseFromGraveCounter--; if (== 0) SpawnZombiesFromGraves(); }
      // 白天草地第 7 关无墓碑石 ⇒ mRiseFromGraveCounter 只在最终波被设为 200（:4962），
      //   到 0 时 SpawnZombiesFromGraves() 在无坟场地上不产生任何僵尸。
5288  if (mHugeWaveCountDown > 0) {                      // ★ 巨型波（flag）横幅
5290      mHugeWaveCountDown--;
5291      if (mHugeWaveCountDown == 0) {
5293          ClearAdvice(ADVICE_HUGE_WAVE);
5294          NextWaveComing();
5295          mZombieCountDown = 1;                      // ← 然后**不 return**，继续往下走 → 本 tick 就出波
5296      } else {
5299          if (mHugeWaveCountDown == 725) { PlaySample(SOUND_HUGE_WAVE); }   // ← 这一 tick **不 return**
5303          else { ...音乐刻度 400 / 700...; return; }                        // ← 其余 tick 全部提前返回
5322      }
5325  }
5327  if (mChallenge->UpdateZombieSpawning()) return;    // 冒险模式返回 false（src/Lawn/Challenge.cpp:2906-2924）
5330  if (mCurrentWave == mNumWaves) {                   // ★ 最后一波已出 ⇒ 白天草地/非生存 ⇒ 直接 return
5332      if (IsFinalSurvivalStage()) return;                                       // 冒险: false (:5158-5161)
5336      if (mGameMode == CHALLENGE_LAST_STAND) return;                             // 不可达
5340      if (!IsSurvivalMode() && !IsContinuousChallenge()) return;                 // ← 本 scope 命中
5343  }
5346  mZombieCountDown--;                                // ★ 唯一的每 tick 递减
5347  if (mCurrentWave == mNumWaves && IsSurvivalMode()) { ... return; }             // 不可达
5356  if (mZombieCountDown > 200 && mZombieCountDownStart - mZombieCountDown > 400 &&
          TotalZombiesHealthInWave(mCurrentWave - 1) <= mZombieHealthToNextWave)
5358      mZombieCountDown = 200;                       // ★ 催波（见 §4）
5360  if (mZombieCountDown == 5) {
5362      if (IsFlagWave(mCurrentWave)) {               // 下一波是 flag 波
5364          ClearAdviceImmediately();
5365          DisplayAdviceAgain("[ADVICE_HUGE_WAVE]", MESSAGE_STYLE_HUGE_WAVE, ADVICE_HUGE_WAVE);
5366          mHugeWaveCountDown = 750;
5367          return;                                   // ← 本 tick 不调用 NextWaveComing()
5368      }
5369      NextWaveComing();                             // 普通波：仅音效/reanim/进度提示，无状态改变
5370  }
5371  if (mZombieCountDown == 0) {                      // ★ 出波条件
5373      SpawnZombieWave();                            //        内部会 mCurrentWave++ / mTotalSpawnedWaves++
5374      mZombieHealthWaveStart = TotalZombiesHealthInWave(mCurrentWave - 1);   // 注意：用的是**自增后**的索引
5376      if (mCurrentWave == mNumWaves && IsSurvivalMode()) {          // 不可达
5378          mZombieHealthToNextWave = 0;
5379          mZombieCountDown = ZOMBIE_COUNTDOWN_BEFORE_REPICK + 1;    // = 5500
5381      } else if (IsFlagWave(mCurrentWave) &&
                     !(IsWallnutBowlingLevel() || mGameMode == CHALLENGE_LAST_STAND)) {
5383          mZombieHealthToNextWave = 0;
5384          mZombieCountDown = ZOMBIE_COUNTDOWN_BEFORE_FLAG;          // = 4500
5386      } else {
5388          mZombieHealthToNextWave = RandRangeFloat(0.5f, 0.65f) * mZombieHealthWaveStart;   // float→int32 截断
5390          if (IsLittleTroubleLevel() || mGameMode == CHALLENGE_COLUMN || mGameMode == CHALLENGE_LAST_STAND)
5391              mZombieCountDown = 750;                                // 不可达
5393          else
5395              mZombieCountDown = ZOMBIE_COUNTDOWN + Rand(ZOMBIE_COUNTDOWN_RANGE);   // = 2500 + [0,599] → [2500,3099]
5397      }
5398      mZombieCountDownStart = mZombieCountDown;     // ★ 注意：在 :5374 之后，催波判定用的是**新**的 Start
5399  }
```

**开始一波的确切条件（字面）**：`mHugeWaveCountDown == 0`（或本 tick 刚归零）且
`mChallenge->UpdateZombieSpawning() == false` 且
`!(mCurrentWave == mNumWaves && !IsSurvivalMode() && !IsContinuousChallenge())` 且
`!HasLevelAwardDropped()` 且 递减后 `mZombieCountDown == 0`。
比较运算符是 `==`，不是 `<=`；`mZombieCountDown` 一旦被催波钳到 200 就只会向下穿过 0 一次。

`IsFlagWave` 的**索引语义**（`src/Lawn/Board.cpp:532-539`）：

```
bool Board::IsFlagWave(int theWaveNumber, int theNumWaves = -1)
{
    if (mApp->IsFirstTimeAdventureMode() && mLevel == 1) return false;               // :534-535  本 scope 不成立
    int aWavesPerFlag = GetNumWavesPerFlag(theNumWaves);                             // :537
    return theWaveNumber % aWavesPerFlag == aWavesPerFlag - 1;                        // :538
}
int Board::GetNumWavesPerFlag(int theNumWaves)                                       // :524
{
    if (theNumWaves < 0)
        theNumWaves = mApp->mEnvironmentMode && originalWaveCount > 0 ? originalWaveCount : mNumWaves;  // :526-529
    return (IsFirstTimeAdventureMode() && theNumWaves < 10) ? theNumWaves : 10;      // :530
}
```

⇒ 本 scope `aWavesPerFlag == 10`，flag 波索引为 `w ∈ {9, 19, 29}`（`w % 10 == 9`）。
`SpawnZombieWave`/`UpdateZombieSpawning` 里调用 `IsFlagWave(mCurrentWave)` 时都**不带第二参数**，
所以 `theNumWaves` 取默认 `-1` ⇒ 走 `originalWaveCount`/`mNumWaves` 分支。
`IsFlagWave(30)` = `30 % 10 == 9` = **false**，所以第 30 波（index 29）出完之后不会再设 4500。

### 1.5 首波与相邻波的时间轴（可直接照抄的算术）

以第 7 关为例，`T` 表示 `mMainCounter`（`mMainCounter` 在 `UpdateZombieSpawning` 之前自增，`:5683`）：

* 开局 `mZombieCountDown = mZombieCountDownStart = 1800`（`:1225,1233`）。
* 每个 `SCENE_PLAYING` 的 tick 递减 1（`:5346`），无催波（`mZombieHealthToNextWave == -1` ⇒ `0 <= -1` 为假，见 §4）。
* 第 `T` 满足 `1800 - T == 5`，即 `T == 1795`：`IsFlagWave(0)` = `0 % 10 == 9` = false ⇒ `NextWaveComing()`。
* `T == 1800`：`mZombieCountDown == 0` ⇒ 出 index 0 的波。
* 出波后：`mZombieHealthToNextWave = trunc(RandRangeFloat(0.5f,0.65f) * mZombieHealthWaveStart)`，
  `mZombieCountDown = mZombieCountDownStart = 2500 + Rand(600)`（`:5395,5398`）。
* flag 波（index 9/19/29）**提前 5 tick** 触发横幅：当 `mZombieCountDown` 递减到 `5` 时
  （`:5360,5362`），`mHugeWaveCountDown = 750` 并 `return`（`:5366-5367`）。此后：
  * `mHugeWaveCountDown` 每 tick 减 1（`:5290`），`mZombieCountDown` **不减**（`:5322` 提前返回），
    唯一例外是 `mHugeWaveCountDown` 减到 `725` 的那一个 tick（`:5299-5302` 不 return），
    那一 tick `mZombieCountDown` 从 5 变成 4。
  * 减到 `0` 的那个 tick（横幅开始后第 750 个 tick）：`mZombieCountDown = 1`（`:5295`）→ 继续往下 →
    `mZombieCountDown--` ⇒ `0`（`:5346`）⇒ **同一 tick 出波**（`:5371`）。
  * 结论：**flag 波相对“原本该出波的那一 tick”被推迟了 745 tick**
    （原本再等 5 tick；实际等 750 tick，且中间 `mZombieCountDown` 只走掉 1 格）。
* 出 flag 波之后（`mCurrentWave` 变成 10/20/30；对 10 与 20：`IsFlagWave(10)==IsFlagWave(20)==false`）
  走 `:5386` 的 else 分支，`mZombieCountDown = mZombieCountDownStart = 2500 + Rand(600)`。
  注意 `:5381` 判的是**下一**波是否 flag 波，而不是刚出的这波；`w==9` 出完后 `mCurrentWave==10`，
  `IsFlagWave(10) == false` ⇒ **不会**出现 4500。4500 只可能出现在"刚出完的那波的下一波是 flag 波"，
  即 `mCurrentWave ∈ {9, 19, 29}` 时（也就是 index 8/18/28 出完之后）。
* 出完最后一波（`mCurrentWave == mNumWaves == 30`）后：`mZombieCountDown` 停在
  `2500 + Rand(600)` 那个值上**再也不递减**（`:5330-5343` 每 tick 直接 return）。

---

## 2. 出波：`Board::SpawnZombieWave`（**不存在 `Board::DoWave`**）

> **`Board::DoWave` NOT FOUND in `src/Lawn/Board.cpp` / `src/Lawn/Board.h`。**
> 等价函数是 `Board::SpawnZombieWave`（`src/Lawn/Board.cpp:4913-4970`），由
> `UpdateZombieSpawning` 的 `mZombieCountDown == 0` 分支调用（`:5373`）。

### 2.1 数量：一波一次全部生成，无间隔

```
void Board::SpawnZombieWave()                                             // :4913
{
    mChallenge->SpawnZombieWave();                                        // :4915 冒险模式是空操作
                                                                            //   (src/Lawn/Challenge.cpp:2973-3016 全是 challenge/continuous 分支)
    const int aBaseCount = NumberZombiesInWave(mCurrentWave);              // :4916
    const double aMultiplier = mApp->mEnvironmentMode
        ? mApp->mEnvironmentTaskSpec.zombieCountMultiplier                 // :4917 ← 本 scope 用这个
        : mApp->mZombieMultiplier;
    const int aSpawnCount = aBaseCount > 0 ? static_cast<int>(std::lround(aBaseCount * aMultiplier)) : 0;   // :4918
    ...
    else {                                                                  // :4937 非 BungeeBlitz
        PVZP_ASSERT(mCurrentWave >= 0 && mCurrentWave < MAX_ZOMBIE_WAVES && mCurrentWave < mNumWaves);  // :4939
        for (int i = 0; i < aSpawnCount; i++) {                             // :4941  **同一个 tick 内全出**
            ZombieType aZombieType = mZombiesInWave[mCurrentWave][i % aBaseCount];   // :4942  表长不足则整表循环复用
            if (aZombieType == ZOMBIE_BOBSLED && !CanAddBobSled()) { ... }   // :4944-4950 本 scope 不可能
            else AddZombie(aZombieType, mCurrentWave);                       // :4952
        }
    }
    const int aOriginalWaves = mApp->mEnvironmentMode && originalWaveCount > 0 ? originalWaveCount : mNumWaves;  // :4958-4959
    if (mCurrentWave == aOriginalWaves - 1 && !IsContinuousChallenge())      // :4960
        mRiseFromGraveCounter = 200;                                          // :4962
    if (IsFlagWave(mCurrentWave))                                            // :4964  （自增**前**判定 ⇒ 刚出的这波）
        mFlagRaiseCounter = FLAG_RAISE_TIME;                                 // :4966  = 100
    mCurrentWave++;                                                           // :4968
    mTotalSpawnedWaves++;                                                     // :4969
}
```

**必须逐字复现的算术：**

* `aBaseCount = NumberZombiesInWave(mCurrentWave)`；`NumberZombiesInWave`
  （`src/Lawn/Board.cpp:9510-9524`）= 从头扫 `mZombiesInWave[w][0..49]`，
  返回**第一个 `ZOMBIE_INVALID(-1)` 的下标**；全 50 格都不是 -1 则 `PVZP_ASSERT(false)`。
* `aMultiplier` 是 **`double`**（`EnvironmentTaskSpec::zombieCountMultiplier`，`src/LawnApp.h:87`，
  校验范围 `[1.0, 10.0]` 且必须 finite，`src/LawnApp.cpp:1330-1331`）。
* `aSpawnCount = static_cast<int>(std::lround(aBaseCount * aMultiplier))`（字面即此表达式）：
  `aBaseCount` 是 `int`，与 `double` 相乘提升为 `double`；`std::lround(double)` **四舍五入、.5 一律远离零**，
  **不是** `static_cast<int>()` 的向零截断（`static_cast<int>(6.5)` = 6，`lround(6.5)` = 7 —— 必须用 `lround`）。
  `aBaseCount <= 0` 时 `aSpawnCount = 0`（三目表达式 `:4918`）。
  参考值（用真实 `double` 乘算）：`base=5` ⇒ `×1.3 → 7`、`×1.4 → 7`、`×1.5 → 8`；
  `base=8` ⇒ `×1.3 → 10`、`×1.4 → 11`、`×1.5 → 12`；`base=11` ⇒ `×1.3 → 14`、`×1.4 → 15`、`×1.5 → 17`。
  kernel 若用 float32 复现这个乘算会引入额外舍入（`zombieCountMultiplier` 声明是 `double`，
  `src/LawnApp.h:87`；从命令行按 `double` 读入，`src/main.cpp:516`），必须保持 `double` 运算。
* 类型取用：`mZombiesInWave[mCurrentWave][i % aBaseCount]` ⇒ 当 `aSpawnCount > aBaseCount` 时
  **整张波表被重复遍历**（`aBaseCount*1.3` 时前 `aSpawnCount % aBaseCount` 项会多出一份）。
* **无 inter-zombie delay**：`aSpawnCount` 个僵尸在同一次 `SpawnZombieWave` 调用里全部 `AddZombie`。
  位置分散靠每个僵尸各自的 `mPosX` 随机抖动（§2.2）实现，不靠时间。
* 波表的长度上限是 50（`MAX_ZOMBIES_IN_WAVE`）；导出侧上限是 32（`kWaveZombieSlots`），
  超出即 `supported = 0, unsupportedReason = 6000 + wave`
  （`src/CudaExport.cpp:306-318`），`mNumWaves > kMaxWaves(30)` ⇒ `unsupportedReason = 6100`（:322-325）。

### 2.2 每个僵尸的生成（x、行、以及 RNG 消耗顺序）

`Board::AddZombie` `src/Lawn/Board.cpp:2652-2655`：
`return AddZombieInRow(theZombieType, PickRowForNewZombie(theZombieType), theFromWave);`
⇒ 行随机数**先于** `AddZombieInRow` 体内的一切随机数消耗（实参在函数体前求值；
C++ 未规定实参求值顺序，但只此一处调用形式，实现里 `PickRowForNewZombie` 的结果作为第 2 个实参传入）。

`Board::AddZombieInRow` `src/Lawn/Board.cpp:2626-2649`：

```
if (mZombies.mSize >= mZombies.mMaxSize - 1) { PvzpLogLn("Too many zombies!!"); return nullptr; }   // :2628-2632
bool aVariant = !Rand(5);                                    // :2640  ← 1 次整数 RNG（1/5 概率为 true）
Zombie* aZombie = mZombies.DataArrayAlloc();
aZombie->ZombieInitialize(theRow, theZombieType, aVariant, nullptr, theFromWave);   // :2642
// BOBSLED 追加 3 个 follower（:2643-2648），本 scope 无关
```

`Zombie::ZombieInitialize` `src/Lawn/Zombie.cpp:143` 中与出怪相关的字面顺序：

```
if (theType == ZOMBIE_FLAG && mBoard) {                                        // :148-157
    int aCount = mPlayerInfo->mZombatarData.size() / ZOMBATAR_RECORD_SIZE;
    if (aCount > 0) aZombatarRecordIndex = Rand(aCount);                       // ← 仅当存档里有 Zombatar 数据才消耗 RNG
}
mFromWave = theFromWave;                                                       // :159
mRow       = theRow;                                                           // :160
mPosX      = 780 + Rand(ZOMBIE_START_RANDOM_OFFSET);                            // :161  ← 1 次整数 RNG，780..819（int→float）
mPosY      = GetPosYBasedOnRow(theRow);                                        // :162
mZombieType= theType; mVariant = theVariant;                                   // :169-170
mHasHead = true; mMindControlled = false; mIsEating = false;                    // :178-180
mGroanCounter = RandRangeInt(300, 400);                                        // :190  ← 1 次整数 RNG，Rand(101)+300 → 300..400
mZombiePhase = PHASE_ZOMBIE_NORMAL;                                             // :195
mRelatedZombieID = ZOMBIEID_NULL;                                               // :200
if (mBoard && mBoard->IsFlagWave(mFromWave)) mPosX += 40.0f;                     // :233-236  ← flag 波整体右移 40
PickRandomSpeed();                                                              // :237
mBodyHealth = 270;                                                              // :238
switch (theType) { ... }                                                        // :248
```

本 scope 四种类型的 x 与速度（`switch` 内的覆盖）：

| 类型 | `mPosX` 表达式（字面） | 取值范围（非 flag 波） | 取值范围（flag 波，`w%10==9`） | `mVelX` |
|---|---|---|---|---|
| `ZOMBIE_NORMAL(0)` | `780 + Rand(40)` `Zombie.cpp:161` | `780..819` | `820..859` | `RandRangeFloat(0.23f,0.37f)`（`Zombie.cpp:1213`，1 次 float RNG） |
| `ZOMBIE_TRAFFIC_CONE(2)` | 同上 `:161` | `780..819` | `820..859` | 同上；额外 `mHelmType = HELMTYPE_TRAFFIC_CONE; mHelmHealth = 370;` `:258-263` |
| `ZOMBIE_FLAG(1)` | 同上 `:161` | `780..819` | `820..859` | `0.45f` **固定值，不消耗 RNG**（`Zombie.cpp:1194-1197`，`mZombieType == ZOMBIE_FLAG` 分支） |
| `ZOMBIE_POLEVAULTER(3)` | `WIDE_BOARD_WIDTH + 70 + Rand(10)` `:383` ⇒ `870 + Rand(10)` | `870..879` | **`870..879`**（switch 在 `:233` 的 `+40` **之后**执行，直接**覆盖** `mPosX`，所以旗波的 +40 对跳格僵尸无效） | 两次 `PickRandomSpeed()`：`:237` 先取 `0.23..0.37`，随后 `:380` 设 `PHASE_POLEVAULTER_PRE_VAULT`、`:387`（仅 `IsOnBoard()`）再取 `RandRangeFloat(0.66f,0.68f)` ⇒ 最终速度 0.66..0.68，**共 2 次 float RNG**；`mBodyHealth = 500`（`:378`） |

* `bool Zombie::IsOnBoard()` `src/Lawn/Zombie.cpp:4284-4293`：
  `return !(mFromWave == ZOMBIE_WAVE_CUTSCENE || mFromWave == ZOMBIE_WAVE_UI)`，
  且对 `mFromWave == ZOMBIE_WAVE_WINNER` 返回 **true**。由 `AddZombie(type, mCurrentWave)` 生成的僵尸
  `mFromWave ∈ [0, 29]` ⇒ 恒为 `IsOnBoard() == true`。
* `IsFlagWave(mFromWave)` 在 `:233` 的判定用**刚出波的索引**（`SpawnZombieWave` 里 `AddZombie` 发生在 `mCurrentWave++` 之前）。

**每个被生成僵尸消耗的 RNG 次数（本 scope，`mZombatarData` 为空时）**：

| 类型 | 行选择 | `!Rand(5)` | x | groan | 速度 | 合计 |
|---|---|---|---|---|---|---|
| NORMAL | 1 (float) | 1 | 1 | 1 | 1 | **5** |
| TRAFFIC_CONE | 1 | 1 | 1 | 1 | 1 | **5** |
| FLAG | 1 | 1 | 1 | 1 | 0 | **4** |
| POLEVAULTER | 1 | 1 | 1 | 1 | 2 | **6** |

（RNG 具体是 MT19937：`MTRand::NextNoAssert` `src/SexyAppFramework/misc/MTRand.cpp:141-176`；
`Next(unsigned long range)` = `NextNoAssert() % range` :178-182；
`Next(float range)` = `(float)((double)NextNoAssert() / (double)0x7FFFFFFF * range)` :191-195；
`Sexy::Rand(int)`/`Rand(float)` 分别转发于 `src/SexyAppFramework/Common.cpp:113-123`，
默认种子 `SRand(4357)` `src/SexyAppFramework/misc/MTRand.cpp:68`。`RandJitter()`（:99-105）默认关闭。）

### 2.3 flag / 巨型波路径与普通波的差异汇总

| 项 | 普通波 | flag 波（`w % 10 == 9`） |
|---|---|---|
| 提前 5 tick 的预告 | `NextWaveComing()`（`:5369`；`mCurrentWave==0` 播 `SOUND_AWOOGA`，`IsFlagWave(mCurrentWave)` 播 `SOUND_SIREN`，`:5246-5253`） | 改走横幅分支：`DisplayAdviceAgain("[ADVICE_HUGE_WAVE]"...)`、`mHugeWaveCountDown = 750`、`return`（`:5364-5367`） |
| 出波时刻 | `mZombieCountDown` 归 0 的 tick | 横幅归 0 的 tick（= 横幅开始 + 750 tick，见 §1.5） |
| 出波后 `mFlagRaiseCounter` | 不动（保持 0） | `= FLAG_RAISE_TIME = 100`（`:4964-4966`）；在 `UpdateProgressMeter` 里每 tick 递减（`:5464-5465`），仅影响进度条/旗杆绘制 |
| 出波后 `mZombieHealthToNextWave` | `trunc(RandRangeFloat(0.5f,0.65f) * mZombieHealthWaveStart)`（`:5388`） | 若**下一**波也是 flag 波：`= 0` 且 `mZombieCountDown = 4500`（`:5381-5384`）；否则同普通波。第 7 关不会出现"相邻两波都是 flag 波" |
| 僵尸 `mPosX` | 基线 | `+40.0f`（`Zombie.cpp:233-236`），跳格僵尸除外（被 switch 覆盖） |
| 波表内容 | 随机点数抽取 | 先硬插 `min(points,8)` 个 NORMAL + 1 个 FLAG，再随机抽取（§5） |

---

## 3. 行选择

### 3.1 状态

`PvzpSmoothArray mRowPickingArray[6]`（`src/Lawn/Board.h:163`），元素为
`{ int32_t mItem; float mWeight; float mLastPicked; float mSecondLastPicked; }`（`src/PvzpLib/PvzpCommon.h:62-69`）。

初始化：`memset(mRowPickingArray, 0, sizeof(mRowPickingArray))`（`src/Lawn/Board.cpp:1376`），
随后 `mRowPickingArray[aRow].mItem = aRow`（`:1383`）。所以开局
`mWeight = mLastPicked = mSecondLastPicked = 0.0f`，`mItem = 0..5`。
存档同步见 `src/Lawn/System/SaveGame.cpp:1621`。

`mWaveRowGotLawnMowered[6]`（`src/Lawn/Board.h:158`）：
`InitLevel` 里全部置 `-100`（`src/Lawn/Board.cpp:1379`），
`InitSurvivalStage` 里同样置 `-100`（`:1289`）；
割草机触发时 `mBoard->mWaveRowGotLawnMowered[mRow] = mBoard->mCurrentWave`
（`src/Lawn/LawnMower.cpp:401`）——注意写入的是**下一波索引** `mCurrentWave`（此时该波还没出）。

### 3.2 `Board::PickRowForNewZombie`（`src/Lawn/Board.cpp:2563-2612`）

```
GridItem* aRake = GetRake();                                            // :2565  本 scope 卡组无耙子时不进入
if (aRake && aRake->mGridItemState == GRIDITEM_STATE_RAKE_ATTRACTING && RowCanHaveZombieType(aRake->mGridY, type)) {
    aRake->mGridItemState = GRIDITEM_STATE_RAKE_WAITING;                 // :2568
    PvzpUpdateSmoothArrayPick(mRowPickingArray, MAX_GRID_SIZE_Y, aRake->mGridY);   // :2570  ← 仍会更新 last/secondLast
    return aRake->mGridY;                                                // :2571  ← **不消耗 RNG**
}
for (int aRow = 0; aRow < 6; aRow++) {                                   // :2573
    if (!RowCanHaveZombieType(aRow, type))                       mRowPickingArray[aRow].mWeight = 0;        // :2575-2579
    else if (mGameMode == CHALLENGE_PORTAL_COMBAT)               mRowPickingArray[aRow].mWeight = ...;      // :2580-2583 不可达
    else if (mGameMode == CHALLENGE_INVISIGHOUL && mCurrentWave <= 3 && aRow == 5) mRowPickingArray[aRow].mWeight = 0;  // :2584-2587 不可达
    else {
        int aWavesMowered = mCurrentWave - mWaveRowGotLawnMowered[aRow];   // :2591   ← "催波割草机保护"
        if (IsContinuousChallenge() && mCurrentWave == mNumWaves - 1) aWavesMowered = 100;   // :2592-2595 不可达
        if      (aWavesMowered <= 1) mRowPickingArray[aRow].mWeight = 0.01f;   // :2597-2599
        else if (aWavesMowered <= 2) mRowPickingArray[aRow].mWeight = 0.5f;    // :2601-2603
        else                       mRowPickingArray[aRow].mWeight = 1.0f;      // :2605-2607
    }
}
return PvzpPickFromSmoothArray(mRowPickingArray, MAX_GRID_SIZE_Y);        // :2611
```

**注意**：`mWeight` 是每次选行前**重新写一遍**的临时量，不跨 tick 累积；
跨 tick 累积的只有 `mLastPicked` / `mSecondLastPicked`。
`aWavesMowered = mCurrentWave - mWaveRowGotLawnMowered[aRow]`（`int32_t` 减法）。
未割过的行 `mWaveRowGotLawnMowered == -100` ⇒ `aWavesMowered == mCurrentWave + 100 >= 100 > 2` ⇒ 权重 `1.0f`。
第 7 关草地：`mPlantRow[0..4] = PLANTROW_NORMAL`、`mPlantRow[5] = PLANTROW_DIRT`
（`src/Lawn/Board.cpp:1004-1009`；`IsFirstTimeAdventureMode()` 为 false 或 `mLevel` 不属于 1/2/3，
所以 `:1012-1024` 的 DIR 覆盖**不生效**）⇒ 行 5 权重恒为 0。
`RowCanHaveZombies(r)` `src/Lawn/Board.cpp:5838-5844`：`r ∈ [0,6)` 且
`mPlantRow[r] != PLANTROW_DIRT`（RESONDED 特例不适用）。

`RowCanHaveZombieType(r, t)` `src/Lawn/Board.cpp:2508-2560`（对本 scope 四种类型逐条）：

```
if (!RowCanHaveZombies(theRow)) return false;                                 // :2510-2513   行 5 ⇒ false
if (mGameMode == CHALLENGE_RESODDED && mPlantRow[theRow] == PLANTROW_DIRT && mCurrentWave < 5) return false;  // :2515-2518 不可达
if (mPlantRow[theRow] == PLANTROW_POOL && !ZombieTypeCanGoInPool(t) && t != ZOMBIE_BALLOON) return false;     // :2519-2522 草地无水路
int aCurrentWave = mCurrentWave;                                              // :2527
if (mPlantRow[theRow] == PLANTROW_POOL) { if (aCurrentWave < 5 && !IsZombieTypePoolOnly(t)) return false; }
else if (IsZombieTypePoolOnly(t)) return false;                               // :2542-2546   NORMAL/CONE/FLAG/POLEVAULTER 都不是 pool-only
if (t == ZOMBIE_BOBSLED && !mIceTimer[theRow]) return false;                   // :2547-2550 不适用
if (theRow == 0 && !IsSurvivalMode() && (t == ZOMBIE_GARGANTUAR || t == ZOMBIE_REDEYE_GARGANTUAR)) return false;  // :2551-2556 不适用
if (t != ZOMBIE_DANCER || StageHasPool()) return true;                          // :2557-2559  ← 本 scope 四种类型都在此返回 true
// 之后是 dancer 专属的行邻接要求，本 scope 不可达
```

⇒ **第 7 关白天草地：四种类型可选行集合恒为 `{0,1,2,3,4}`，各自权重初始化为 `1.0f`。**

### 3.3 权重公式与 RNG（`src/PvzpLib/PvzpCommon.cpp:175-238`）

```
float PvzpCalcSmoothWeight(float w, float lastPicked, float secondLastPicked)     // :175
{
    if (w < 1E-6f) return 0.0f;                                                    // :177-180
    float expectedLength1 = 1.0f / w;                                              // :182
    float expectedLength2 = expectedLength1 * 2.0f;                                // :183
    float advancedLength1 = lastPicked + 1.0f - expectedLength1;                   // :184
    float advancedLength2 = secondLastPicked + 1.0f - expectedLength2;             // :185
    float factor1 = 1.0f + advancedLength1 / expectedLength1 * 2.0f;                // :186  = 2*w*lastPicked + 2*w - 1
    float factor2 = 1.0f + advancedLength2 / expectedLength2 * 2.0f;                // :187  = secondLastPicked*w + w - 1
    float factorFinal = std::clamp(factor1 * 0.75f + factor2 * 0.25f, 0.01f, 100.0f);  // :188
    return w * factorFinal;                                                        // :189
}
```

`factor1 / factor2` 的**运算顺序是字面**的：先 `advancedLength / expectedLength`（float 除法），
再 `* 2.0f`，再 `1.0f +`。等价闭式已写在源码注释里（`:186-187`），但为逐位复现浮点结果请按上面的顺序写。

```
int PvzpPickFromSmoothArray(PvzpSmoothArray* a, int n)          // :192, n = 6
{
    float totalWeight = 0.0f;  for (i=0;i<n;i++) totalWeight += a[i].mWeight;     // :194-198
    PVZP_ASSERT(totalWeight > 0.0f);                                                // :199
    float normalizeFactor = 1.0f / totalWeight;                                     // :201
    float totalAdjusted = 0.0f;
    for (j=0;j<n;j++) totalAdjusted += PvzpCalcSmoothWeight(a[j].mWeight * normalizeFactor,
                                                           a[j].mLastPicked, a[j].mSecondLastPicked);   // :203-206
    PVZP_ASSERT(totalAdjusted > 0.0f);                                              // :207
    float randWeight = Rand(totalAdjusted);                                          // :209  ← **float 重载**，[0, totalAdjusted)
    float acc = 0.0f; int k;
    for (k = 0; k < n - 1; k++) {                                                   // :212  注意上界是 n-1（不含最后一个元素）
        acc += PvzpCalcSmoothWeight(a[k].mWeight * normalizeFactor, a[k].mLastPicked, a[k].mSecondLastPicked);  // :214
        if (randWeight <= acc) break;                                               // :215  ← <=
    }
    PvzpUpdateSmoothArrayPick(a, n, k);                                             // :221
    return a[k].mItem;                                                              // :222
}
void PvzpUpdateSmoothArrayPick(PvzpSmoothArray* a, int n, int pick)                 // :225
{
    for (i = 0; i < n; i++)
        if (a[i].mWeight > 0.0f) { a[i].mLastPicked += 1.0f; a[i].mSecondLastPicked += 1.0f; }   // :227-234
    a[pick].mSecondLastPicked = a[pick].mLastPicked;                                 // :236
    a[pick].mLastPicked = 0.0f;                                                     // :237
}
```

关键语义（都是字面行为，务必照抄）：

1. **归一化是每次调用现场做的**（`mWeight * normalizeFactor`），所以 5 个可行行 ⇒ 每个 `1.0/5.0 = 0.2f`。
   `PvzpCalcSmoothWeight` 的输入永远是"归一化后的概率"，不是原始 `1.0f`。
   对 `w = 0.2f`：`expectedLength1 = 5.0f`、`expectedLength2 = 10.0f`、
   `factor1 = 1 + (lastPicked - 4)/5*2`、`factor2 = 1 + (secondLastPicked - 9)/10*2`。
2. **`mLastPicked` / `mSecondLastPicked` 的自增只针对 `mWeight > 0.0f` 的行**（`:229`）。
   ⇒ 行 5（DIR，权重 0）的 `mLastPicked`/`mSecondLastPicked` **永远保持 0**，不随波次增长。
   若某行被割草机保护压到 `0.01f`（`> 0.0f`）它**仍会计数**。
3. `aRandWeight <= aAccumulatedWeight` 用的是 `<=`；循环上界是 `n-1`，
   所以 `k` 可以等于 `n-1`（此时不 break，直接落到 `k == n-1`）——**最后一个元素是兜底项**。
4. 一次选行消耗 **恰好 1 次** float RNG（`:209`）。`PvzpCalcSmoothWeight` 内部无 RNG。
5. 权重衰减（"割草机保护"）只改 `mWeight`，不改 `mLastPicked/mSecondLastPicked`。
   `mWaveRowGotLawnMowered[row] = mCurrentWave` 后，接下来两波该行的权重分别是 `0.01f`（`aWavesMowered <= 1`）
   和 `0.5f`（`aWavesMowered <= 2`），第三波及以后回到 `1.0f`。
   归一化会把这 0.01/0.5 摊到 5 行之上（`totalWeight = 0.01+1+1+1+1 = 4.01` 等），
   所以**实际概率不是字面的 0.01/0.5**，而是 `weight * normalizeFactor` 再进 `PvzpCalcSmoothWeight`。

> 对比：`PvzpPickFromWeightedArray`（整数权重，用于**画波表**，`src/PvzpLib/PvzpCommon.cpp:116-146`）
> 与上面的平滑数组是**两套不同的抽样**：整数值求和、`aTotalWeight = Sexy::Rand(aTotalWeight)`（整数）、
> 然后 `aTotalWeight -= theArray[i].mWeight; if (aTotalWeight < 0) return i;`（`< 0`，不是 `<=`）。

---

## 4. 催波（advance wave early）

### 4.1 波血量口径

```
int Board::TotalZombiesHealthInWave(int theWaveIndex)                                // src/Lawn/Board.cpp:4897
{
    int aTotalHealth = 0;                                                             // :4899  int
    for (Zombie* z : mZombies) {
        if (z->mDead) continue;                                                       // :4902-4904
        if (z->mFromWave == theWaveIndex && !z->mMindControlled && !z->IsDeadOrDying()
            && z->mZombieType != ZOMBIE_BUNGEE
            && z->mRelatedZombieID == ZombieID::ZOMBIEID_NULL) {                      // :4905-4907
            aTotalHealth += z->mBodyHealth + z->mHelmHealth + z->mShieldHealth * 0.2f + z->mFlyingHealth;   // :4908
        }
    }
    return aTotalHealth;                                                              // :4911
}
```

* 表达式 `mBodyHealth + mHelmHealth + mShieldHealth * 0.2f + mFlyingHealth` 是 **float**
  （`mShieldHealth * 0.2f`），`aTotalHealth += (float)` 每次加完**截断回 int32_t**（向零取整），
  逐只累加、逐只截断 ⇒ 不能先算浮点总和再截断。
* 排除：`mDead`、`mMindControlled`、`IsDeadOrDying()`、bungee、以及任何**有父僵尸**的单位
  （`mRelatedZombieID != ZOMBIEID_NULL`；`enum ZombieID : uint32_t { ZOMBIEID_NULL }` `src/ConstEnums.h:1389`，
  `mRelatedZombieID` 在 `Zombie.cpp:200` 初始化为 `ZOMBIEID_NULL`）。
* 第 7 关四种类型的相关字段：`NORMAL` body 270（`Zombie.cpp:238`）；`TRAFFIC_CONE` body 270 + helm 370
  （`Zombie.cpp:263`）；`POLEVAULTER` body 500（`Zombie.cpp:378`）；`FLAG` body 270。
  shield/flying 均为 0（`Zombie.cpp:208-211`）。⇒ 单只满血值 = NORMAL 270 / CONE 640 / VAULT 500 / FLAG 270。
* `theWaveIndex == -1`（即 `mCurrentWave == 0`）时，`mFromWave == -1` 不匹配任何由 `AddZombie(..., mCurrentWave>=0)`
  生成的僵尸 ⇒ 返回 0。（关卡开场街道预览僵尸用 `ZOMBIE_WAVE_CUTSCENE = -2`，也不匹配。）

### 4.2 两个字段怎么算出来的

* 出生：`mZombieHealthWaveStart = 0`（`src/Lawn/Board.cpp:1228`），`mZombieHealthToNextWave = -1`（`:1234`）。
* 每次出波后（**在 `SpawnZombieWave()` 已经把 `mCurrentWave` 自增之后**）：

```
mZombieHealthWaveStart  = TotalZombiesHealthInWave(mCurrentWave - 1);                  // :5374  ⇒ 刚出这一波（此刻的当前血量）
...
mZombieHealthToNextWave = RandRangeFloat(0.5f, 0.65f) * mZombieHealthWaveStart;         // :5388  float 乘 int ⇒ float ⇒ 赋值给 int32_t 向零截断
mZombieCountDownStart   = mZombieCountDown;                                             // :5398
```

  出 flag 波后（下一波也是 flag 波时）或生存 repick 后：`mZombieHealthToNextWave = 0`（`:5378,5383`）。
* `RandRangeFloat(lo, hi) = Rand(hi - lo) + lo`（`src/PvzpLib/PvzpCommon.cpp:390-394`），
  即 `Rand(0.15f) + 0.5f` ∈ `[0.5, 0.65)`，1 次 float RNG。

### 4.3 每 tick 的判定与算术（`src/Lawn/Board.cpp:5356-5358`）

```
mZombieCountDown--;                                                     // :5346  **先减**
if (mZombieCountDown > 200
    && mZombieCountDownStart - mZombieCountDown > 400
    && TotalZombiesHealthInWave(mCurrentWave - 1) <= mZombieHealthToNextWave)
    mZombieCountDown = 200;                                             // :5358
```

复现要点（全部是字面 `int32_t` 比较，`<=` 不是 `<`）：

1. 判定用的是**递减之后**的 `mZombieCountDown`。
2. `mZombieCountDownStart` 在钳位时**不更新** ⇒ 第二个条件 `Start - CD > 400`
   意思是"本波计时已经走过至少 **401** tick"，此后恒为真。
3. 一旦钳到 `200`，第一个条件 `> 200` 恒假 ⇒ **每段波间隔最多钳一次**（不会反复钳）。
4. 钳位后剩余等待恰好 `200` tick（200 → 0），并且**钳位那一 tick 不会出波**：
   后续的 `== 5`、`== 0` 判定都拿 `200` 去比（`:5360,5371`）。
5. 首波永不催：`mZombieHealthToNextWave == -1`，`TotalZombiesHealthInWave(-1) == 0`，
   `0 <= -1` 为假。
6. 催波检查在 `mHugeWaveCountDown` 检查**之后**、`== 5` 检查**之前**。
   所以横幅期间（除 `== 725` 那一 tick）根本不会做催波判定。
7. 阈值语义：`mZombieHealthToNextWave` 是"**剩余**血量下限"——当上一波（其实是"当前波"，
   索引 `mCurrentWave-1`）的现存总血量降到 `[0.5, 0.65] × 波起始血量` 以下时立刻开始下一波的计时收尾。
   `UpdateProgressMeter`（`:5485-5495`）用同一个量做进度条血量分数：
   `aDamageTarget = mZombieHealthWaveStart - mZombieHealthToNextWave; if (aDamageTarget < 1) aDamageTarget = 1;`
   `aHealthFraction = (aDamageTarget - aHealthCurrent + mZombieHealthToNextWave) / (float)aDamageTarget;`
   —— 这是纯 UI，不反馈到状态机，kernel 可以不实现。

---

## 5. `Board::PickZombieWaves`：抽表算法（`src/Lawn/Board.cpp:578-783`）

（kernel 只**导入**已提交的表、不重抽；本节用于核对导入分布的形状。）

```
mNumWaves = gZombieWaves[clamp(mLevel-1, 0, 49)];                       // :589  第 7 关 = 20
if (!IsFirstTimeAdventureMode() && !IsMiniBossLevel())
    mNumWaves = mNumWaves < 10 ? 20 : mNumWaves + 10;                    // :590-593  ⇒ 20 + 10 = 30   ← 本 scope
const int aFullWaves = mNumWaves;                                        // :618
if (mApp->mEnvironmentMode) originalWaveCount = aFullWaves;               // :619
（waveCap 在 InitLevel 之后才裁剪：mNumWaves = min(mNumWaves, task.waveCap)  src/LawnApp.cpp:1446-1447，
 originalWaveCount 保持 30；waveCap == 0 时不裁。）
PVZP_ASSERT(mNumWaves <= MAX_ZOMBIE_WAVES);                              // :623
全表填 ZOMBIE_INVALID                                                     // :624-625
for (aWave = 0; aWave < aFullWaves; aWave++) {                            // :630  **按 aFullWaves 画满原表**
    ZombiePickerInitForWave(&picker);                                    // :632  mZombieCount=0, mZombiePoints=0, 计数数组清零
    mZombiesInWave[aWave][0] = ZOMBIE_INVALID;                            // :633
    bool aIsFlagWave = IsFlagWave(aWave, aFullWaves);                     // :635
    bool aIsFinalWave = aWave == aFullWaves - 1;                          // :636

    int& aZombiePoints = picker.mZombiePoints;                             // :647  **int 引用**
    // 点数预算（互斥 if-else 链 :648-663）
    if      (LAST_STAND)      aZombiePoints = (stage*perStage + aWave + 10) * 2 / 5 + 1;   // :650
    else if (生存 stage>0)     aZombiePoints = (stage*perStage + aWave) * 2 / 5 + 1;        // :654
    else if (冒险 && HasFinishedAdventure() && mLevel != 5)
                              aZombiePoints = aWave * 2 / 5 + 1;          // :658  ← 本 scope（playthrough=2）
    else                      aZombiePoints = aWave / 3 + 1;              // :662

    if (aIsFlagWave) {                                                    // :665
        int aPlainZombiesNum = std::min(aZombiePoints, 8);                 // :667
        aZombiePoints *= 2.5f;                                            // :668  **int = (int)(int*2.5f)：float 乘法后向零截断**
        for (i = 0; i < aPlainZombiesNum; i++) PutZombieInWave(ZOMBIE_NORMAL, aWave, &picker);   // :672-675
        PutZombieInWave(ZOMBIE_FLAG, aWave, &picker);                      // :676
    }
    // 关卡点数倍乘 :681-700：COLUMN *6 / LittleTrouble|WallnutBowling *4 / MiniBoss *3 /
    //   StormyNight&&冒险 *3 / Shovel|BungeeBlitz|PortalCombat|Invisighoul *2 —— 第 7 关**全部不命中**

    // 新引入僵尸的固定出场 :703-729：aIntroZombieType = GetIntroducedZombieType()（:622）
    //   = 满足 mStartingLevel == mLevel 的第一个类型（src/Lawn/Board.cpp:2370-2386）
    //   第 7 关：gZombieDefs 里没有任何类型 mStartingLevel == 7 ⇒ aIntroZombieType == ZOMBIE_INVALID ⇒ 本段不做事
    if (mApp->IsAdventureMode() && aIsFinalWave) PutInMissingZombies(aWave, &picker);   // :737-740
    //   PutInMissingZombies（:567-576）：对每个类型 t，若 picker.mZombieTypeCount[t] <= 0 && t != YETI
    //   && CanZombieSpawnOnLevel(t, mLevel)  ⇒ PutZombieInWave(t, aWave, &picker)

    while (aZombiePoints > 0 && picker.mZombieCount < MAX_ZOMBIES_IN_WAVE) {   // :776  整数循环，上限 50 只/波
        ZombieType t = PickZombieType(aZombiePoints, aWave, &picker);            // :778
        PutZombieInWave(t, aWave, &picker);                                       // :779
    }
}
```

`PutZombieInWave`（`src/Lawn/Board.cpp:554-565`）：写入 `mZombiesInWave[w][mZombieCount++] = t`，
然后在新空位写 `ZOMBIE_INVALID` 作哨兵，并且

```
picker.mZombiePoints -= GetZombieDefinition(t).mZombieValue;     // :562   ← 点数是**扣减**的，可被扣成负数
picker.mZombieTypeCount[t]++;  picker.mAllWavesZombieTypeCount[t]++;
```

`CanZombieSpawnOnLevel`（`src/Lawn/Board.cpp:2353-2367`）：

```
if (t == ZOMBIE_YETI) return CanSpawnYetis();
if (theLevel < def.mStartingLevel || def.mPickWeight == 0) return false;     // :2361  ← FLAG 的 mPickWeight == 0 ⇒ 永不在 mZombieAllowed 里
return gZombieAllowedLevels[t].mAllowedOnLevel[clamp(theLevel-1, 0, 49)];    // :2365
```

`PickZombieType`（`src/Lawn/Board.cpp:2416-2501`）——**整数加权抽样**：

```
对 aZombieType = 0 .. NUM_ZOMBIE_TYPES-1：
    if (!mZombieAllowed[aZombieType]) continue;                                 // :2422
    if (aZombieType == ZOMBIE_BUNGEE && 生存无尽) { if (!IsFlagWave(w)) continue; }   // :2430-2437
    else if (mode != POGO_PARTY && mode != BOBSLED_BONANZA && mode != AIR_RAID) {    // :2438
        int aFirstAllowedWave = def.mFirstAllowedWave;                            // :2439
        （无尽会提前，:2441-2446，本 scope 不适用）
        if (w + 1 < aFirstAllowedWave || theZombiePoints < def.mZombieValue) continue;   // :2447  ← 1-based 波序 + 点数门槛
    }
    int aPickWeight = def.mPickWeight;                                          // :2454
    （生存模式会重算权重/上限 :2455-2487，本 scope 不进入）
    aZombieWeightArray[aPickCount++] = { aZombieType, aPickWeight };            // :2489-2491
return (ZombieType)PvzpPickFromWeightedArray(aZombieWeightArray, aPickCount);    // :2501
```

`gZombieDefs` 里本 scope 四种类型的字面参数（`src/Lawn/Zombie.cpp:81-84`）：

| 类型 | `mZombieValue` | `mStartingLevel` | `mFirstAllowedWave` | `mPickWeight` | 第 7 关 `mAllowedOnLevel[6]`（`src/Lawn/Challenge.cpp:76-108`） | 进 `mZombieAllowed`？ |
|---|---|---|---|---|---|---|
| `ZOMBIE_NORMAL` | `1` | `1` | `1` | `4000` | `1` | ✅ |
| `ZOMBIE_FLAG` | `1` | `1` | `1` | `0` | `1` | ❌（`mPickWeight == 0`） |
| `ZOMBIE_TRAFFIC_CONE` | `2` | `3` | `1` | `4000` | `1` | ✅ |
| `ZOMBIE_POLEVAULTER` | `2` | `6` | `5` | `2000` | `1` | ✅ |
| `ZOMBIE_PAIL`（下一档，用于确认边界） | `4` | `8` | `1` | `3000` | `0` | ❌（`mStartingLevel 8 > 7`） |

⇒ 第 7 关 `mZombieAllowed` 只有 `{NORMAL, TRAFFIC_CONE, POLEVAULTER}` 三项为 true，
随机抽样权重 `4000 : 4000 : 2000`（`aWave + 1 >= 5`，即 `aWave >= 4` 时三项齐全 ⇒ 概率 `0.4 / 0.4 / 0.2`；
`aWave ∈ {0,1,2,3}` 时 POLEVAULTER 被 `mFirstAllowedWave = 5` 拦掉 ⇒ 只剩 `0.5 / 0.5`）。
`FLAG` 只能由 flag 波硬插（`:676`）或最终波的 `PutInMissingZombies`（不插，因为它不在 `mZombieAllowed`/`CanZombieSpawnOnLevel` 里）。

点数预算的具体数值（`aWave * 2 / 5 + 1`，**整数除法**，`aWave = 0..29`）：
`1,1,1,2,2,3,3,3,4,4, 5,5,5,6,6,7,7,7,8,8, 9,9,9,10,10,11,11,11,12,12`。
flag 波（w=9,19,29）先做 `min(points,8)`（= 4, 8, 8）个 NORMAL + 1 FLAG，
`points` 变为 `static_cast<int>(points * 2.5f)`（= 10, 20, 30），再由 `PutZombieInWave` 逐只扣掉
`mZombieValue`（NORMAL 扣 1、FLAG 扣 1）后进入 while 抽取循环。
最后一波（w=29）额外保证 `{NORMAL, CONE, VAULT}` 各至少一只（`PutInMissingZombies`）。

抽样与状态机之间**没有**耦合：`PickZombieWaves` 只在 `InitZombieWaves` 里被调用一次
（`src/Lawn/Board.cpp:1205`）；`ReseedEnvironment` 会重画整表但保留已出波并复原 `mNumWaves`
（`src/main.cpp:269-280`）。

---

## 6. 胜利（关卡完成）判定

判定函数只有一个：**`Zombie::TrySpawnLevelAward()`**（`src/Lawn/Zombie.cpp:7138-7276`）。
它是**事件驱动**的，不是每 tick 扫描；调用点只有三处：

| 调用点 | 时机 |
|---|---|
| `src/Lawn/Zombie.cpp:7432`（`DieNoLoot`，紧跟 `mDead = true;` `:7431`） | 任何僵尸死亡 |
| `src/Lawn/Zombie.cpp:7289`（`DropLoot`，开头先 `if (!IsOnBoard()) return;` `:7280-7281`） | 僵尸掉落战利品 |
| `src/Lawn/Zombie.cpp:4838`（踩中 Hypnoshroom → `StartMindControlled()`） | 心智控制 |

（另有 `Board.cpp:1867`、`Challenge.cpp:2268` 直接写 `mLevelAwardSpawned = true`，都不属于冒险模式第 7 关。）

字面判定链：

```
bool Zombie::TrySpawnLevelAward()                                                     // Zombie.cpp:7138
{
    if (!IsOnBoard() || mBoard->HasLevelAwardDropped() || mBoard->mLevelComplete || mDroppedLoot)
        return false;                                                                 // :7140-7143
    if (mApp->IsFinalBossLevel())   { ... }                                          // :7145-7151  不适用
    else if (mApp->IsScaryPotterLevel()) { ... }                                     // :7152-7158  不适用
    else if (mApp->IsContinuousChallenge()
             || mBoard->mCurrentWave < mBoard->mNumWaves                             // :7159  ★ 所有波都已**开始**
             || mBoard->AreEnemyZombiesOnScreen())                                   // :7159  ★ 场上没有敌人
        return false;
    if (mApp->IsWhackAZombieLevel() && mBoard->mZombieCountDown > 0) return false;   // :7164-7167  不适用
    mBoard->mLevelAwardSpawned = true;                                                // :7169
    mApp->mBoardResult = BoardResult::BOARDRESULT_WON;                                // :7170
    if (!mBoard->IsSurvivalStageWithRepick()) mBoard->RemoveAllZombies();              // :7176-7179
    ... 掉落奖励硬币 ...
    mDroppedLoot = true;  return true;                                                // :7274-7275
}
```

**胜利条件（字面比较）：`mCurrentWave >= mNumWaves && !AreEnemyZombiesOnScreen()`**
（源码写成 `<` 就 `return false` 的反面）。

* 读的两个计数器：`mCurrentWave`（**已开始**的波数，在 `SpawnZombieWave` 末尾 `++`，
  `src/Lawn/Board.cpp:4968`）与 `mNumWaves`（第 7 关 = 30；若 `waveCap>0` 则是被
  `src/LawnApp.cpp:1447` 裁剪后的值——**胜利门禁用的是裁剪后的 `mNumWaves`**，
  而 `mRiseFromGraveCounter`/`NextWaveComing` 用的是 `originalWaveCount`）。
  `mZombieCountDown` 只在 Whack-A-Zombie 里参与判定。
* 敌方扫描 `Board::AreEnemyZombiesOnScreen()`（`src/Lawn/Board.cpp:262-274`）：

```
for (Zombie* z : mZombies) {
    if (z->mDead) continue;                                                          // :266-267
    if (z->mHasHead && !z->IsDeadOrDying() && !z->mMindControlled && z->IsOnBoard())  // :268
        return true;
}
return false;
```

  逐条回答规格里问的三类僵尸：
  * **旗子僵尸算不算？** **算。** 没有任何类型豁免（`ZOMBIE_FLAG` 只要活着就在屏幕上 ⇒ 不能通关）。
  * **濒死僵尸算不算？** **不算。** `IsDeadOrDying()`
    （`src/Lawn/Zombie.cpp:9643-9650`）= `mDead || mZombiePhase ∈ {PHASE_ZOMBIE_DYING, PHASE_ZOMBIE_BURNED,
    PHASE_ZOMBIE_MOWERED}`；另外 `mHasHead == false` 也不算（掉头的僵尸要等死亡相位被设置才彻底豁免）。
  * **不在棋盘上的僵尸算不算？** **算。** `IsOnBoard()`（`src/Lawn/Zombie.cpp:4284-4293`）
    **与坐标无关**，只看 `mFromWave ∉ {ZOMBIE_WAVE_CUTSCENE(-2), ZOMBIE_WAVE_UI(-3)}`。
    一个刚生成、`mPosX = 780..879`（草坪 800 边界之外）的僵尸 `mFromWave ∈ [0,29]` ⇒ `IsOnBoard() == true`
    ⇒ 计入"场上有敌人"。`ZOMBIE_WAVE_WINNER(-4)`（走进房子）也**不算**离场的豁免。
    ⇒ 通关要求**每一个**由波次生成的僵尸都进入 `mDead` / `IsDeadOrDying()` / `!mHasHead` / `mMindControlled` 之一。
* 判定发生的**时刻**：`TrySpawnLevelAward` 在僵尸死亡流程里被调用，`DieNoLoot` 里 `mDead = true` **先于**调用
  （`Zombie.cpp:7431-7432`），所以"最后一只僵尸"不会把自己算进 `AreEnemyZombiesOnScreen`。
* 胜利之后：`mLevelAwardSpawned = true` ⇒ `HasLevelAwardDropped()` 为真（`Board.cpp:5191-5194`）
  ⇒ `UpdateZombieSpawning` 在 `:5276-5277` 直接 return，波次状态机停摆。
* 终局读取：`LawnApp::EnvironmentTerminal()`（`src/LawnApp.cpp:1591-1595`）
  `= !mBoard || mBoard->mLevelComplete || mBoardResult == BOARDRESULT_WON || mBoardResult == BOARDRESULT_LOST
     || mGameScene == SCENE_ZOMBIES_WON`。`mLevelComplete` 由 `UpdateLevelEndSequence`
  （`src/Lawn/Board.cpp:1692`，置位点 `:1737,1750,1829`）与 `FadeOutLevel`（`:1829`）设置；
  失败路径是 `Board::ZombiesWon`（`:5068-5074` 写 `BOARDRESULT_LOST`，`:5119` `WalkIntoHouse`）。

---

## 7. kernel 必须携带的状态

### 7.1 不可变（每个 rollout 只读，来自导入表 / LevelInfo）

| 项 | 字面来源 | 备注 |
|---|---|---|
| 波表本身 | `RootRecord::waveCount / waveZombieCount[30] / waveTable[30][32]`（`src/CudaRollout.h:218-224`），导出逻辑 `src/CudaExport.cpp:300-326` | 第 7 关 = `aFullWaves = 30`，`mNumWaves == 30`，索引 9/19/29 是 flag 波 |
| `numWaves` | `RootRecord::numWaves`（`CudaRollout.h:190`）= `board->mNumWaves` | `mNumWaves > 30` ⇒ 导出直接判 `supported = 0, reason 6100` |
| `zombieCountMultiplier` | `LevelInfo::zombieCountMultiplier`（`src/CudaExport.cpp:87`）；生成点在 `SpawnZombieWave` 用 `std::lround(base * mult)` | `double`；`[1.0,10.0]` 校验 `src/LawnApp.cpp:1330-1331` |
| `originalWaveCount` | `EnvironmentTaskSpec::originalWaveCount`（`src/LawnApp.h:92`） | 只有 `waveCap > 0` 时才与 `numWaves` 不同；本 scope 相同 ⇒ kernel 可以只带 `numWaves` |
| 波池定义 | `LevelInfo::zombies[]` 的 `value / firstAllowedWave / pickWeight / velMin / velMax`（`src/CudaExport.cpp:89-105`） | 只用于校验导入分布（§5），出怪运行时不需要 |
| 每波是否 flag 波 | 可由 `w % 10 == 9` 现场算（`aWavesPerFlag == 10`，`src/Lawn/Board.cpp:530,538`） | 或预计算成 30 bit 掩码 |
| 行数与可行行集合 | 第 7 关草地：`{0,1,2,3,4}`；行 5 权重恒 0（`src/Lawn/Board.cpp:1004-1009, 5838-5844`） | 不变量 |
| 常量 | §0 全部 | `MAX_ZOMBIES_IN_WAVE = 50`、`kWaveZombieSlots = 32`、`kMaxWaves = 30` |

### 7.2 每 rollout 可变（per-lane 标量）

对应现有标量槽位名（`src/CudaRollout.h:323-328` 的 `S_*` 枚举；从 root 装载见 `src/SimImpl.inc:113-119,134`）：

| 标量 | 原始字段 | 类型 | 更新点（字面） |
|---|---|---|---|
| `S_WAVE` | `mCurrentWave` | `int32_t` | `= 0` `Board.cpp:1212`；`++` `Board.cpp:4968`。**语义 = 下一波的索引 / 已开始的波数** |
| `S_NUM_WAVES` | `mNumWaves` | `int32_t` | 任务常量，rollout 内不变（`Board.cpp:589-593`；`LawnApp.cpp:1447`） |
| `S_ZOMBIE_COUNT_DOWN` | `mZombieCountDown` | `int32_t` | 递减 `Board.cpp:5346`；钳位 `:5358`；横幅结束置 1 `:5295`；出波后 `:5379/5384/5391/5395` |
| `S_ZOMBIE_COUNT_DOWN_START` | `mZombieCountDownStart` | `int32_t` | `:1233`、`:5398`（出波后）。**催波钳位时不改**（§4.3 第 2 点） |
| `S_HEALTH_WAVE_START` | `mZombieHealthWaveStart` | `int32_t` | `:1228` = 0；`:5374` = `TotalZombiesHealthInWave(mCurrentWave-1)` |
| `S_HEALTH_TO_NEXT_WAVE` | `mZombieHealthToNextWave` | `int32_t` | `:1234` = **-1（哨兵）**；`:5378/5383` = 0；`:5388` = `trunc(RandRangeFloat(0.5f,0.65f) * healthWaveStart)` |
| `S_HUGE_WAVE_COUNT_DOWN` | `mHugeWaveCountDown` | `int32_t` | `:1231` = 0；`:5366` = 750；`:5290` 每 tick `--`；判定 `== 0` / `== 725`（`:5291/5299`） |
| `S_WAVE_SPAWN_CURSOR` | （原始代码**不存在**此概念） | `int32_t` | 原始实现把一波在**一个 tick 内**全部生成（§2.1），所以 kernel 的 cursor 只是"把一波摊开在若干 tick 生成"的自研机制；若要保持等价，cursor 必须在一 tick 内跑完 `aSpawnCount` 次并只让 `S_WAVE` 在末尾 +1 |
| 每行割草机波次 | `mWaveRowGotLawnMowered[6]` | `int32_t[6]` | 初值 `-100` `:1379`；触发时 `= mCurrentWave` `LawnMower.cpp:401`。只被 §3.2 的 `aWavesMowered = mCurrentWave - mWaveRowGotLawnMowered[aRow]` 读 |
| 每行平滑状态 | `mRowPickingArray[r].mLastPicked` / `.mSecondLastPicked` | `float[6] × 2` | 只由 `PvzpUpdateSmoothArrayPick`（`PvzpCommon.cpp:225-238`）推进；**自增只对 `mWeight > 0.0f` 的行生效**。`mWeight` 是每次选行前重算的临时量，**不需要携带** |
| `mTotalSpawnedWaves` | `mTotalSpawnedWaves` | `int32_t` | `:1213` = 0；`:4969` `++`。除存档/统计（`main.cpp:160`）与 `Board.cpp:9108` 的一个奖杯判定外**不参与任何出怪逻辑** ⇒ kernel 可省，除非要复算成就 |
| `mFlagRaiseCounter` | `mFlagRaiseCounter` | `int32_t` | `:4966` = 100；`:5464-5465` 在 `UpdateProgressMeter` 里 `--`。纯 UI（进度条 `:6615`）⇒ kernel 可省 |
| `mRiseFromGraveCounter` | `mRiseFromGraveCounter` | `int32_t` | `:4962` = 200（仅最终波）；`:5281` `--`；第 7 关白天草地不产生坟墓 ⇒ 归 0 时 `SpawnZombiesFromGraves()` 无效果 ⇒ **可省** |
| `mLevelAwardSpawned` | `mLevelAwardSpawned` | `bool` | `:7169`（`Zombie.cpp`）；决定 `HasLevelAwardDropped()` ⇒ **必须携带**（它会冻结波次状态机，§1.4 `:5276`） |
| 每只僵尸的 `mFromWave` | `RootZombie::fromWave`（`CudaRollout.h:148`） | `int32_t` | 催波血量求和（§4.1）与胜利判定的 `IsOnBoard()`（§6）都依赖它 ⇒ **必须携带** |

### 7.3 与 kernel 现有导出的一致性检查点（对照 §1–§6）

1. `src/CudaExport.cpp:166-168` 导出 `mZombieCountDown / mZombieCountDownStart /
   mZombieHealthWaveStart / mZombieHealthToNextWave`，`:169` 导出 `mHugeWaveCountDown`，
   `:182-186` 导出 `mWaveRowGotLawnMowered[r]` 与 `mRowPickingArray[r].mLastPicked / mSecondLastPicked`
   ——**没有** `mWaveRowGotLawnMowered` 的 `mWeight`（正确，见上表），也**没有** `mTotalSpawnedWaves` /
   `mFlagRaiseCounter`（若 kernel 不实现 UI/成就可以不补）。
2. `PvzpPickFromSmoothArray` 只用 `mLastPicked/mSecondLastPicked` 两个 float；
   `mWeight` 每次调用前由 `PickRowForNewZombie` 重写 ⇒ 导入的 root 只需两行 float 数组。
3. 导出对 `mNumWaves > kMaxWaves(30)` 与 `NumberZombiesInWave(w) > kWaveZombieSlots(32)`
   会直接判 `supported = 0`（`src/CudaExport.cpp:306-325`）；而原始数组容量是
   `MAX_ZOMBIE_WAVES = 100` × `MAX_ZOMBIES_IN_WAVE = 50`（`src/Lawn/Board.h:44-45`）
   ⇒ 第 7 关若某波抽到 > 32 只（flag 波点数放大后是有可能的），kernel 会拿到不支持的 root，
   这是**导入侧**的容量约束，不是 §2.1 的原始约束。
