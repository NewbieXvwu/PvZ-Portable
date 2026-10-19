# CUDA 机制规格 06：Board 每 tick 顺序与胜负（tick order & win/lose）

适用场景：冒险模式第 7 关（`mLevel == 7`），草地白天，5 行可种植（row 0..4），
牌组 = peashooter / sunflower / cherry / wallnut / potatomine / snowpea / repeater，
`EnvironmentTaskSpec.playthrough == 2`，`waveCap == 0`，`zombieCountMultiplier ∈ [1.3, 1.5]`。
所有引用格式为 `文件:行`。代码里没有的东西一律写 **NOT FOUND**，不编造。

本文件只描述 **Board 层的 tick 编排与胜负判定**。植物/僵尸/子弹内部逐 tick 逻辑见
`cuda/spec/projectiles.md`；波次调度与出怪见 `cuda/spec/waves.md`；协议/观测字段见
`cuda/spec/protocol.md`。

---

## 0. 第 7 关白天走的是哪条分支（先钉死所有门）

| 判定 | 第 7 关白天的值 | 依据 |
|---|---|---|
| `mApp->mGameMode` | `GAMEMODE_ADVENTURE` | `src/LawnApp.cpp:1311`（`EnvironmentReset` 走 `PreNewGame(GAMEMODE_ADVENTURE, …)`） |
| `mApp->mGameScene` | `SCENE_PLAYING` | `src/LawnApp.cpp:404-408`（`StartPlaying()` 里赋值），`src/LawnApp.cpp:1441`（RESET 调用它） |
| `mApp->mHeadlessMode` | `true` | `src/LawnApp.cpp:2027-2037`（命令行 `-env`） |
| `mBoard->mBackground` | `BACKGROUND_1_DAY` | `src/Lawn/Board.cpp:858-866`（`PickBackground`：`(mLevel-1)/10 → AREA_BG_BASE[0] = BACKGROUND_1_DAY`） |
| `StageIsNight()` | `false` | `src/Lawn/Board.cpp:879`（`BACKGROUND_1_DAY` 不在夜晚表） |
| `StageHasFog()` | `false` | `src/Lawn/Board.cpp:925` |
| `StageHasPool()` | `false` | `src/Lawn/Board.cpp:893` |
| `StageIsRoof()` | `false` | `src/Lawn/Board.cpp:909` |
| `mPlantRow[row]` | row 0..4 = `PLANTING_NORMAL`，row 5 = `PLANTING_DIRT`（不可种） | `src/Lawn/Board.cpp:1002-1081` |
| `mGridSquareType[x][y]` | 全部 `GRID_SQUARE_NORMAL`（无 graveyard / dig / water） | `src/Lawn/Board.cpp:1002-1081` |
| `mEnableGraveStones` | `false` | `src/Lawn/Board.cpp:1338-1390`（`InitLevel`；只有夜雾图置 true） |
| `mPaused` | `false` | `src/Lawn/Board.cpp:96`（构造默认 false），env 路径没有任何地方置 true |
| `mTimeStopCounter` | `0` | `src/Lawn/Board.h:213`，`src/Lawn/Board.cpp:157`；只有 ice/snowpea 的 freeze 路径写它，第 7 关白天没有 |
| `mLevel` 是否为 `IsMiniBossLevel/IsFinalBossLevel` | `false` / `false` | `src/LawnApp.cpp:2920-2926`、`src/LawnApp.cpp:2928-2937`（只有 10/20/30/40/50 关是 boss） |
| `IsFirstTimeAdventureMode()` | `false`（playthrough==2 ⇒ `mFinishedAdventure==1`） | `src/LawnApp.cpp:3231-3234` |
| 教学（tutorial） | 完全不跑 | `src/Lawn/Board.cpp:5705`（`mMainCounter==1 && IsFirstTimeAdventureMode()`） |
| `CanDropLoot()` | `true` | `src/Lawn/Board.cpp:9184-9187`：`!ShouldRunUpsellBoard() && (!IsFirstTimeAdventureMode() \|\| mLevel>=11)` |
| `mSunMoney` 初值 | `50` | `src/Lawn/Board.cpp:1338-1390`（`InitLevel` 里 `mSunMoney = 50`） |
| 小推车 | 5 台（row 0..4），`mMowerState == MOWER_READY`，类型 `LAWN_MOWER` | `src/Lawn/Board.cpp:1603-1625`（`InitLawnMowers`），`src/Lawn/LawnMower.cpp:30-85` |
| `mNumWaves` | `30`（`gZombieWaves[6]==20` +10 复玩；`waveCap==0` 不裁剪） | `src/Lawn/Challenge.cpp:68-73`、`src/Lawn/Board.cpp:578-616`、`src/LawnApp.cpp:1446` |
| 僵尸池容量 | `1024` | `src/Lawn/Board.cpp:77` |

`BoardResult` 枚举：`BOARDRESULT_NONE=0, WON=1, LOST=2, QUIT=3`（`src/ConstEnums.h:125-133`）。

---

## 1. `Board::UpdateGameObjects` —— 完整有序子更新

### 1.1 它在整条链里的位置

`UpdateGameObjects()` 是 `Board::UpdateGame()` 的**第一条语句**，
`src/Lawn/Board.cpp:5660-5662`：

```
void Board::UpdateGame()
{
    UpdateGameObjects();                 // 5662
    ...fog offset...                    // 5663-5678，StageHasFog()==false ⇒ 整块跳过
    if (mApp->mGameScene != SCENE_PLAYING && !mCutScene->ShouldRunUpsellBoard())
        return;                          // 5680-5683 ⇒ 第 7 关白天不返回
    mMainCounter++;                      // 5684
    UpdateSunSpawning();                 // 5685
    UpdateZombieSpawning();              // 5686
    UpdateIce();                         // 5687
    if (mIceTrapCounter > 0) { mIceTrapCounter--; ... }   // 5688-5699
    if (mFogBlownCountDown > 0) { mFogBlownCountDown--; } // 5700-5703
    if (mMainCounter == 1 && IsFirstTimeAdventureMode()) { ... }  // 5705-5717 ⇒ 跳过
    UpdateProgressMeter();               // 5719
}
```

**关键结论（内核必须照抄）：`mMainCounter++` 发生在所有实体 `Update()` 之后、
太阳生成与出怪之前。** 也就是说 tick N 的实体更新看到的是 `mMainCounter == N-1`。
`mMainCounter` 自增点唯一：`src/Lawn/Board.cpp:5684`。
（`src/Lawn/Board.cpp:6944` 等只是 debug overlay 读取。）

`Board::UpdateGameObjects` 本体，`src/Lawn/Board.cpp:5022-5016`（函数体 5022 起，实际语句 5024-5065）。
按源码顺序，**没有任何条件门**（每个循环体只有 `if (x->mDead) continue;`）：

```
for (Plant*      p : mPlants)      { if (p->mDead) continue;      p->Update(); }   // 5024-5029
for (Zombie*     z : mZombies)     { if (z->mDead) continue;      z->Update(); }   // 5031-5036
for (Projectile* pr: mProjectiles) { if (pr->mDead) continue;     pr->Update(); }  // 5038-5043
for (Coin*       c : mCoins)       { if (c->mDead) continue;      c->Update(); }   // 5045-5050
for (LawnMower*  m : mLawnMowers)  { if (m->mDead) continue;      m->Update(); }   // 5052-5057
mCursorPreview->Update();                                                          // 5059
mCursorObject->Update();                                                           // 5060
for (int i = 0; i < mSeedBank->mNumPackets; i++)
    mSeedBank->mSeedPackets[i].Update();                                           // 5062-5065
```

声明顺序与上面一致：`src/Lawn/Board.h:131-136`
（`mPlants, mZombies, mProjectiles, mCoins, mLawnMowers, mGridItems`）。
注意 **`mGridItems` 不在这里**，它在 `UpdateGame()` 之前的 `Board::Update()` 里由
`UpdateGridItems()` 处理（`src/Lawn/Board.cpp:9280-9304`），见 §5。

### 1.2 每个容器内部的遍历顺序（不是创建顺序）

`DataArray<T>` 的迭代器是 **索引升序、只访问活槽**，`src/PvzpLib/DataArray.h:136-176`：

```
Iterator::operator++():
    do { mIndex++; } while (mIndex < mMaxUsedCount && !(mItemIds[mIndex] & KEY_MASK));
Iterator::operator==(default_sentinel): return mIndex >= mMaxUsedCount;
begin(): 从 0 开始，若槽 0 空则前置自增一次
```

分配用 LIFO 空闲链，`src/PvzpLib/DataArray.h:178-207`：

```
DataArrayAlloc():
    if (mFreeListHead == mMaxUsedCount) { aNext = mMaxUsedCount; mFreeListHead = ++mMaxUsedCount; }
    else { aNext = mFreeListHead; mFreeListHead = mItemIds[mFreeListHead]; }   // 复用最小可用/最后释放的槽
    mItemIds[aNext] = (mNextKey++ << 16) | aNext;
```

推论（内核必须实现）：

1. **同 tick 内的作用顺序 = 槽下标升序**，与实体创建/死亡时间无关。
2. 槽会被复用 ⇒ 新僵尸可能插到旧僵尸前面。用 `key` 单调递增可判定创建批次，但**不能用 id 大小决定处理次序**。
3. 循环中 `DataArrayFree` 是安全的（地址在解引用时才算出，`src/PvzpLib/DataArray.h:136`注释）。
   但**更新中不会 free**：实体只置 `mDead`。
4. id 编码 `(key << 16) | index`，`mNextKey` 基线 `1001`，
   到 `65536` 回绕到 `1`（`src/PvzpLib/DataArray.h:32-42, 205`）。
   `DataArrayFreeAll` 把 `mNextKey` 复位到 `1001`（`src/PvzpLib/DataArray.h:117`）⇒ 每个 RESET 后 id 序列可复现。

### 1.3 `UpdateGameObjects` 之前/之后仍然执行的门（level 7 白天的实际通过情况）

`Board::Update()` 全貌，`src/Lawn/Board.cpp:5722-5821`。逐句标注第 7 关白天走/不走：

```
Board::Update()                                                   // 5722
  PvzpHesitationBracket("Board::Update")                          // 5724  性能计时，无状态
  Widget::Update()                                                // 5726  Widget::Update 只 MarkDirty，src/Lawn/widget/Widget.cpp:188-191
  MarkDirty()                                                     // 5727  渲染
  mBoardUpdateCounter++                                           // 5729  ★每 tick +1（与 mMainCounter 同步，但 RESET 时都清零：Board.cpp:1341）
  mCutScene->Update()                                             // 5730  ⇒ 立即 return（SCENE_LEVEL_INTRO 才干活，src/Lawn/CutScene.cpp:1392-1393）
  if (!mApp->mHeadlessMode) UpdateMousePosition()                 // 5731-5732  env 下跳过
  if (mGameMode == GAMEMODE_CHALLENGE_ZEN_GARDEN) ...             // 5733-5736  跳过
  if (IsScaryPotterDaveTalking()) UpdateCrazyDave()               // 5737-5740  跳过
  if (mPaused) { mChallenge->Update(); 光标隐藏; return; }         // 5742-5748  mPaused==false ⇒ 不返回
  aDisabled = !CanInteractWithBoardButtons() || mIgnoreMouseUp;   // 5750  仅写按钮 mDisabled，无游戏状态
  mMenuButton->Update(); mStoreButton->Update()                   // 5751-5759  mStoreButton==nullptr（headless），Update 只 MarkDirty
  mApp->mEffectSystem->Update()                                   // 5761  粒子；被 ENV_SKIP 位影响，见 §2.4
  mAdvice->Update()                                               // 5762  消息队列；src/Lawn/MessageWidget.cpp:259 有 gEnvSimSkip & ENV_SKIP_MESSAGE 门
  UpdateTutorial()                                                // 5763  mTutorialState==TUTORIAL_NONE ⇒ 直接 return，src/Lawn/Board.cpp:5510-5513
  if (mCobCannonCursorDelayCounter > 0) --                        // 5765-5768  恒 0（无玉米加农炮）
  if (mOutOfMoneyCounter > 0) --                                  // 5770-5773  ★可达，见 §5
  if (mShakeCounter > 0) { --; ...  ★消耗 RNG ... }               // 5774-5791  ★可达，见 §5
  if (mCoinBankFadeCount > 0 && 无购买对话框) --                   // 5792-5795  仅 UI
  UpdateLayers()                                                  // 5796  MarkAllDirty，渲染
  if (mTimeStopCounter > 0) return                                // 5797-5798  恒 0 ⇒ 不返回
  mEffectCounter++                                                // 5801  ★每 tick +1（Board.h:179）
  if (StageHasPool() && ...) mPoolEffect->mPoolCounter++          // 5802-5805  跳过
  if (mBackground == BACKGROUND_3_POOL) ...                       // 5806-5811  跳过
  UpdateGridItems()                                               // 5813  ★见 §5
  UpdateFwoosh()                                                  // 5814  ★见 §5（第 7 关不可达）
  UpdateGame()                                                    // 5815  ★§1.1 展开
  UpdateFog()                                                     // 5816  !StageHasFog() ⇒ return，src/Lawn/Board.cpp:7306-7309
  mChallenge->Update()                                            // 5817  ★见 §5
  UpdateLevelEndSequence()                                        // 5818  ★见 §4.4
  mPrevMouseX = mWidgetManager->mLastMouseX; mPrevMouseY = ...    // 5819-5820  仅鼠标
```

**结论：第 7 关白天没有任何 early-return 拦住 `UpdateGame()`；
`mPaused`、`mTimeStopCounter`、`mFogBlownCountDown`、`mIceTrapCounter`、`StageHasPool/Fog` 分支全部为假/零。**
唯一还消耗 RNG 的 Board 顶层块是 shake 块（`src/Lawn/Board.cpp:5784`），见 §5。

### 1.4 延迟删除：`ProcessDeleteQueue` 在 tick 的**最开头**

`src/Lawn/Board.cpp:8519-8575`，六次独立扫描，固定顺序
`mPlants → mZombies → mProjectiles → mCoins → mLawnMowers → mGridItems`，
每个都是索引升序 `if (x->mDead) DataArrayFree(x)`。

调用点：`LawnApp::AdvanceLogicTick()`（`src/LawnApp.cpp:1581-1589`），
在 `SexyApp::UpdateFrames()` **之前**。
所以一个 tick 的语义是：

```
tick T:
  释放上一 tick 标死的实体（ProcessDeleteQueue）
  → 本 tick 创建的实体分配槽位（此时才可能复用被释放的下标）
  → Board::Update → … → UpdateGameObjects（跳过 mDead 的）
  → mMainCounter++
```

`ProcessDeleteQueue` 与 `UpdateGameObjects` 之间的**这一对顺序**决定了槽位复用模式，
是内核复现 id 序列的唯一途径。**不能把释放合并到更新循环末尾。**

---

## 2. Headless 环境路径的精确每 tick 序列（**内核必须匹配这个顺序**）

### 2.1 `LawnApp::EnvironmentWait(ticks)` —— 固定 WAIT 的循环体

`src/LawnApp.cpp:1504-1519`，逐字：

```
void LawnApp::EnvironmentWait(int ticks)
{
    if (!mBoard || ticks < 0 || ticks > 1000000) return;                  // 1506-1507
    for (int i = 0; i < ticks && !EnvironmentTerminal(); ++i)             // 1508  ★每 tick 先测终局
    {
        for (Coin* coin : mBoard->mCoins)                                 // 1510  ★太阳自动收割
        {
            if (!coin->mDead && coin->IsSun())
                coin->ScoreCoin();                                        // 1512
        }
        AdvanceLogicTick();                                               // 1514
        if (!mHeadlessMode) EnvironmentPresent();                         // 1515-1516  headless 下跳过
    }
}
```

**这三步就是一个 env tick，顺序不可交换：**

1. `EnvironmentTerminal()` 检查（进入循环条件，`src/LawnApp.cpp:1508`）；
2. **对 `mCoins`（索引升序）中每个未死且 `IsSun()` 的硬币调用 `ScoreCoin()`**（`src/LawnApp.cpp:1510-1513`）；
3. `AdvanceLogicTick()`（`src/LawnApp.cpp:1514`）。

`Coin::ScoreCoin()`，`src/Lawn/Coin.cpp:433-462`：**先 `Die()`（置 `mDead`）再加钱**：

```
ScoreCoin():
    Die();                                              // 435
    if (IsSun()) { mBoard->AddSunMoney(GetSunValue()); }  // 437-441
    else if (IsMoney()) { … }                            // 442-456（sun 走不到）
    if (mType == COIN_DIAMOND && mBoard) mDiamondsCollected++;  // 457-460
```

`IsSun()` = `COIN_SUN | COIN_SMALLSUN | COIN_LARGESUN`（`src/Lawn/Coin.cpp:423-426`）。
`GetSunValue()`：`COIN_SUN=25, COIN_SMALLSUN=15, COIN_LARGESUN=50, 其它=0`（`src/Lawn/Coin.cpp:1297-1300`）。
`AddSunMoney(n)`：`mSunMoney += n; mSunMoney = min(mSunMoney, 9990);`（`src/Lawn/Board.cpp:8463-8469`）。

`Coin::Die()` 只置 `mDead=true`（`src/Lawn/Coin.cpp:1402`），所以**同一个 tick 内后面的循环仍会看到这个 coin**，
但 `!coin->mDead` 门（`src/LawnApp.cpp:1512`）阻止重复 `ScoreCoin`。槽位在下一 tick 的
`ProcessDeleteQueue` 释放。

> **⇒ 太阳永远不会落地。** 一个在 tick T 的 `UpdateSunSpawning()` 里 `AddCoin` 的天降太阳，
> 它的 `mDead` 直到 tick T 结束都不会被 `Coin::Update()` 触碰（`mCoins` 的更新在
> `UpdateGameObjects`，即 T 里 `mMainCounter` 由 N-1→N 之前）；而在 **tick T+1 的第 2 步**
> 就被 `ScoreCoin()` 打死并加钱。`ProcessDeleteQueue` 在 T+1 的第 3 步开头把它释放。
> 于是 `UpdateFall()` 对它**一次都没有执行过**：`mPosY` 恒为 60，`mGroundY` 恒为 300..549 的随机值，
> `mCoinAge == 0`，`mHitGround == false`，`mDisappearCounter == 0`。
> **内核里不要给 env 路径的天降太阳实现下落/过期逻辑，只需要「创建 ⇒ 下一 tick 加 25」。**
> 详细见 §3。

### 2.2 `LawnApp::AdvanceLogicTick()`

`src/LawnApp.cpp:1581-1589`，逐字：

```
void LawnApp::AdvanceLogicTick()
{
    ++mAppCounter;                                                        // 1583
    if (mBoard) mBoard->ProcessDeleteQueue();                             // 1584-1585
    if (mLoadingThreadCompleted && mEffectSystem)
        mEffectSystem->ProcessDeleteQueue();                              // 1586-1587
    SexyApp::UpdateFrames();                                              // 1588   ★限定调用！
}
```

`SexyApp::UpdateFrames()`（**不是** `LawnApp::UpdateFrames()`）：`src/SexyAppFramework/SexyApp.cpp:146-149`
→ `mWidgetManager->UpdateFrame()`：`src/SexyAppFramework/widget/WidgetManager.cpp:484-497`
（内部先 `UpdateAll()` 于 `484-490`，然后处理滚动条等）
→ `WidgetContainer::UpdateAll()`：`src/SexyAppFramework/widget/WidgetContainer.cpp:504-538`
（**每帧只跑一次的守卫**：`504` 处 `if (mUpdateAllDone) return; mUpdateAllDone = true;`，
在 `WidgetManager::UpdateFrame` 末尾 `497` 复位；按 `mWidgets` 添加顺序遍历，Board 是唯一活跃游戏 widget）
→ `Board::Update()`（`src/Lawn/Board.cpp:5722`）。

`mAppCounter` 只在 `src/LawnApp.cpp:1583` 自增；`RESET` 时清 0（`src/LawnApp.cpp:1448`）。
它是**逻辑 tick 计数的外层镜像**，`EnvironmentReset` 之后 `mAppCounter == 0`、
`mMainCounter == 0`，第一个 WAIT tick 结束时两者都是 1。

### 2.3 交互式路径**多做**的事（env 路径全部跳过 —— 内核不要实现）

| 交互式有、env 没有 | 依据 |
|---|---|
| `LawnApp::UpdateFrames()` 的 `aUpdateCount` 可变（`gSlowMo` ⇒ 每 4 帧才 1 tick；`gFastMo` ⇒ 每帧 20 tick） | `src/LawnApp.cpp:2330-2350` |
| 每 tick `mMusic->MusicUpdate()` | `src/LawnApp.cpp:2353`（`AdvanceLogicTick` 直调 `SexyApp::UpdateFrames`，绕开这个循环体） |
| **每 tick `CheckForGameEnd()`** | `src/LawnApp.cpp:2354`、定义 `src/LawnApp.cpp:2165-2265`：读 `mBoard->mLevelComplete` 后 `KillBoard()` + 存档 + 领奖界面 ⇒ **env 下永远不执行**，所以 `mLevelComplete=true` 之后 board 不会被拆除，只有 `EnvironmentTerminal()` 停住循环 |
| `SDL_PumpEvents` / `DrawDirtyStuff` / `SDL_Delay` | `src/LawnApp.cpp:1524-1535`（`EnvironmentPresent`，`-env-visible` 专用） |
| `UpdateMousePosition()` | `src/Lawn/Board.cpp:5731-5732`（`!mHeadlessMode` 门） |
| 真实 `gEnvSimSkip == 0`（所有渲染/粒子/UI 都跑） | `src/PvzpLib/PvzpCommon.h:180-195` |

`EnvironmentWaitEvents` 与 `ROLLOUT` 都不调用 `LawnApp::UpdateFrames()`，
因此 **env 的每一个逻辑 tick 恰好等于一次 `AdvanceLogicTick()`**，
且 tick 数完全由 `EnvironmentWait(ticks)` 决定，没有帧率耦合。

### 2.4 `gEnvSimSkip`（`ENV_SKIP_*`）—— 唯一让同一 tick 少干活的路子

位定义与白名单：`src/PvzpLib/PvzpCommon.h:180-195`；
`ROLLOUT` 用 `mask >= 0 && !(mask & ~ENV_SKIP_ALLOWED)` 校验（`src/main.cpp:397`），
并只在 rollout 期间替换 `gEnvSimSkip`（`src/main.cpp:424-425`、恢复 `490`）。
**允许的位是 `1|4|8|16 == 29`（`ENV_SKIP_ALLOWED`）**：
被明确排除的位（例如跳粒子/跳 `mEffectSystem->Update()`）会改变**共享 Mersenne 序列的消费次数**，
`src/main.cpp:397` 的掩码校验就是在防止这种 RNG 流漂移。
⇒ 内核按「这些被跳过的工作本来就不消耗游戏 RNG」处理；只实现位 1/4/8/16 对应可省的部分。
具体每个位跳过哪些函数，`protocol.md` 有对照表，此处不重复。

---

## 3. 白天从天上掉的太阳（falling sun）

### 3.1 调度：`Board::UpdateSunSpawning()`

`src/Lawn/Board.cpp:5196-5231`。第 7 关白天下前 16 行的 return 门全部为假
（`StageIsNight()==false`，`HasLevelAwardDropped()` 在胜利前为 false，
`mGameMode` 不是 challenge/intro/zen/tree/last-stand，
`IsIZombieLevel/IsScaryPotterLevel/IsSquirrelLevel/HasConveyorBeltSeedBank()==false`，
`mTutorialState != TUTORIAL_SLOT_MACHINE_PULL`）；
`TUTORIAL_LEVEL_1_*` 门（5218-5223）也跳过。
所以每 tick 实际执行：

```
mSunCountDown--;                                                        // 5225
if (mSunCountDown != 0) return;                                         // 5226-5227
mNumSunsFallen++;                                                       // 5229
mSunCountDown = min(SUN_COUNTDOWN_MAX, SUN_COUNTDOWN + mNumSunsFallen*10)
                + Rand(SUN_COUNTDOWN_RANGE);                            // 5230
aSunType = (mGameMode == GAMEMODE_CHALLENGE_SUNNY_DAY) ? COIN_LARGESUN : COIN_SUN;  // 5231
AddCoin(RandRangeInt(100, 649), 60, aSunType, COIN_MOTION_FROM_SKY);    // 5232
```

常数：`SUN_COUNTDOWN = 425`，`SUN_COUNTDOWN_RANGE = 275`，`SUN_COUNTDOWN_MAX = 950`
（`src/Lawn/Board.cpp:65-67`）。字段：`mSunCountDown`（`src/Lawn/Board.h:166`）、
`mNumSunsFallen`（`src/Lawn/Board.h:167`）。

### 3.2 初值与「`min(950, 425+10n) + Rand(275)`」这一说法的核对

**核对结论：公式正确，但有两个必须注意的偏差。**

* 初始延迟**不是** 425。`Board::InitLevel`（`src/Lawn/Board.cpp:1385-1390`）：
  ```
  mNumSunsFallen = 0;                                                   // 1385
  if (!StageIsNight())
      mSunCountDown = RandRangeInt(425, 700);                           // 1386-1389
  ```
  `RandRangeInt(a,b)` = `a + Rand(b - a + 1)` ⇒ 取值域 **425..700 两端闭合**，且**消耗一次 RNG**。
  （`RandRangeInt` 定义：`src/PvzpLib/PvzpCommon.cpp:384-388`。）
  夜间分支 `mSunCountDown = -1`，第 7 关不走。
* `n` 是**自增之后**的 `mNumSunsFallen`。所以第一个 gap（第 1 个太阳落下之后）
  用的 `n == 1` ⇒ `min(950, 425+10*1) + Rand(275) = 435 + [0,274]`，
  不是 `425 + Rand(275)`。第 k 个太阳（`k ≥ 1`）之后的间隔为
  `min(950, 425 + 10k) + RandInt(0..274)`，`425+10k ≥ 950` 从 `k ≥ 53` 起封顶为 950。

完整时间线（`Rand(x)` ≡ `[0, x-1]` 的整数，见 §3.5）：

```
tick 0（RESET 完）: mNumSunsFallen = 0, mSunCountDown = 425 + RandInt(0..275)
每 tick（UpdateGame 内，mMainCounter++ 之后）: mSunCountDown-- ; 若非 0 则结束
mSunCountDown == 0 的那一 tick: mNumSunsFallen++;  mSunCountDown = min(950, 425+10*mNumSunsFallen) + RandInt(0..274);
                                AddCoin(x = RandInt(100..649), y = 60, COIN_SUN, COIN_MOTION_FROM_SKY)
```

`UpdateSunSpawning` 每 tick 恰好被调用一次（`src/Lawn/Board.cpp:5685`），
且**调用点在 `mMainCounter++` 之后** ⇒ 第 T 个太阳出现时 `mMainCounter` 已经包含本 tick。

`HasLevelAwardDropped()`（`src/Lawn/Board.cpp:5191-5194`）：
`mLevelAwardSpawned || mNextSurvivalStageCounter > 0 || mBoardFadeOutCounter >= 0`。
`mLevelAwardSpawned` 在胜利那一 tick 被置 true（`src/Lawn/Zombie.cpp:7169`）
⇒ **胜利之后不再产天降太阳**（`src/Lawn/Board.cpp:5197` 的门）。

### 3.3 下落速度 / 可拾取 y / 消失 y

**这些在 env 路径上全部不可达**（原因见 §2.1 方框）。为完整起见列出原代码，
内核**只需要**「生成 → 下一 tick 加分」：

`Coin::CoinInitialize` 的 `COIN_MOTION_FROM_SKY` 分支，`src/Lawn/Coin.cpp:315-319`：

```
mVelY    = 0.67f;                    // 每 tick 下移 0.67 像素（无重力累加）
mVelX    = 0.0f;
mGroundY = Rand(250) + 300;          // 落点 y ∈ 300..549，★消耗一次 RNG
```
`COIN_MOTION_FROM_SKY` 不加重力：`UpdateFall` 里 `mVelY += 0.09f` 只对 `FROM_PLANT`，
`+= 0.15f` 只对 `COIN`/`FROM_BOSS`（`src/Lawn/Coin.cpp:546-552`）。

`Coin::UpdateFall`，`src/Lawn/Coin.cpp:481-603`：

```
if (mPosY + mVelY < mGroundY) { mPosY += mVelY; mPosX += mVelX; …水平反弹… }  // 489-508
else { mHitGround = true; mPosY = mGroundY; mPosX = FloatRoundToInt(mPosX);    // 571-577
       if (!IsLevelAward() && !IsPresentWithAdvice()) {
           mDisappearCounter++;
           if (mDisappearCounter >= GetDisappearTime()) StartFade(); } }        // 579-588
```

* **可拾取的 y：没有下界门。** 交互式里太阳在整个生命周期（`mPosY` 从 60 起）都能被点到；
  `Coin::Update` 与 `UpdateFall` 都不设「collectable」阈值 ⇒ **NOT FOUND（不存在这样的 y）**。
* 落地 y = `mGroundY = 300 + Rand(250)` ∈ `[300, 549]`（`src/Lawn/Coin.cpp:318`）。
* 落地后 `mDisappearCounter` 每 tick +1，`>= GetDisappearTime()` 时 `StartFade()`
  （`src/Lawn/Coin.cpp:583-587`）。`GetDisappearTime()` 对 `COIN_SUN` 返回 **750**
  （只有 diamond/present/chocolate/带箭头的是 1500，zen 是 6000；`src/Lawn/Coin.cpp:1495-1515`）。
  门 `mGameMode != LAST_STAND || …` 恒真（`src/Lawn/Coin.cpp:579-581`）。
* `StartFade()`：`mFadeCount = 15`（`src/Lawn/Coin.cpp:464-467`）；
  `UpdateFade()` 每 tick `mFadeCount--`，`== 0` 时 `Die()`（`src/Lawn/Coin.cpp:469-479`），
  非 level-award 的 sun 走 `!IsLevelAward()` 分支 ⇒ 会衰减。
  所以交互式最迟寿命 ≈ `ceil((mGroundY-60)/0.67) + 750 + 15` tick（起点 y=60）。

从生成到落地的 tick 数（若内核想要复现交互式的 y 轨迹）：
`mPosY` 从 60 起每 tick `+0.67f`，到 `mGroundY` 需要
`first k with 60 + 0.67k >= mGroundY` ⇒ `k = ceil((mGroundY-60)/0.67)`，
范围：`mGroundY=300 ⇒ 359 tick`；`mGroundY=549 ⇒ 730 tick`。
`mCoinAge++` 在 `Coin::Update` 的第一行（`src/Lawn/Coin.cpp:732`）。

### 3.4 天降太阳 vs 向日葵太阳 —— 全部差异

| 项 | 天降（sky） | 向日葵（plant） |
|---|---|---|
| `CoinMotion` | `COIN_MOTION_FROM_SKY` `src/Lawn/Board.cpp:5232` | `COIN_MOTION_FROM_PLANT` `src/Lawn/Plant.cpp:1004-1074`（`UpdateProductionPlant`） |
| 生成点 | `(RandRangeInt(100,649), 60)`，`Board.cpp:5232` | 植物上方，`mPosX = 植物 x`，`mPosY = 植物 y`，`Plant.cpp:1004-1074` |
| 初速 | `mVelY=0.67f, mVelX=0.0f`，`Coin.cpp:316-317` | `mVelY = -1.7f - RandRangeFloat(0,1.7)`，`mVelX = -0.4f + RandRangeFloat(0,0.8)`，`Coin.cpp:328-330` |
| 重力 | 无（`Coin.cpp:546-552` 不匹配 FROM_SKY） | `mVelY += 0.09f` 每 tick，`src/Lawn/Coin.cpp:546-548` |
| 落点 | `Rand(250)+300` ∈ 300..549，`src/Lawn/Coin.cpp:318` | `mPosY + 15 + Rand(20)` ∈ 起点+15..+34，`src/Lawn/Coin.cpp:331` |
| 缩放 | `mScale` 保持 1.0f（`COIN_SUN` 走 else 分支 `aScale=1.0f`，`Coin.cpp:390-401`） | `mScale = 0.4f` 起手（`Coin.cpp:332`），随后 `UpdateFall` 末尾每 tick `+0.02f` 直到 `GetSunScale()`（=1.0f）⇒ **24 tick 长到全尺寸**，`src/Lawn/Coin.cpp:595-603`、`src/Lawn/Coin.cpp:1292-1295` |
| 价值 | `COIN_SUN = 25`（`Coin.cpp:1297-1300`） | 同 `COIN_SUN = 25` |
| 消失时间 | 750 tick（`Coin.cpp:1495-1515`） | 同 750 |
| `mSunMoneyProduced` | `AddCoin` 里 `+= 25`（`src/Lawn/Board.cpp:1974-1976`） | **同一条路径**（向日葵也调 `Board::AddCoin`）⇒ 对 rollout 的收入历史两者不可区分 |
| env 路径 | **下一 tick 立刻被 `ScoreCoin` 加 25，永不移动** | **同样下一 tick 被加 25，永不移动** |

⇒ **在第 7 关 env rollout 里，天降太阳与向日葵太阳唯一的区别是产生时刻/横坐标与 RNG 消耗次数；
数值收益、消失行为、生命周期完全一致。**

### 3.5 env tick 上消耗的 RNG 序列（决定复现性的核心）

`Sexy::Rand` = 全局 Mersenne Twister，`Rand(x)` = `NextNoAssert() & 0x7FFFFFFF` 再 `% x`
（`src/SexyAppFramework/Common.cpp:107-127`、`src/Common/mat/../../misc/MTRand.cpp:138-198`）。
`x == 0` 时返回 0 且**仍消耗一次**（`Common.cpp:107-127`）。
`RandRangeInt(a,b) = a + Rand(b-a+1)`（`src/PvzpLib/PvzpCommon.cpp:384-388`）。

第 7 关白天，一个「有太阳落地」的 env tick（顺序即消费顺序）：

```
1) EnvironmentWait 的收太阳循环        —— 不消耗 RNG（ScoreCoin/AddSunMoney 无随机）
2) AdvanceLogicTick                    —— 不消耗
3) Board::Update
     mShakeCounter>0 时: Rand(3)       —— src/Lawn/Board.cpp:5784
4) UpdateGameObjects                   —— 植物/僵尸/子弹内部随机（projectiles.md）
5) mMainCounter++
6) UpdateSunSpawning:
     Rand(275)                          —— Board.cpp:5230
     RandRangeInt(100,649)  = 100+Rand(550) —— Board.cpp:5232
7) AddCoin → CoinInitialize:
     Rand(250)  (mGroundY)              —— Coin.cpp:318
8) UpdateZombieSpawning                —— 出怪/催波随机（waves.md）
     RandRangeFloat(0.5,0.65) + Rand(600) 只在完成一波的那一 tick
```

普通 tick（无太阳生成）在第 6/7 步**消耗 0 次 RNG**。
`Reset` 用 `Sexy::SRand(seed)` 播种（`src/LawnApp.cpp:1451`）。
⇒ 内核必须精确复刻「哪一 tick 掉太阳」，之后 RNG 流才会对齐。

---

## 4. 胜负判定

### 4.1 **NOT FOUND：`Board::CheckWinLose` / `Board::Victory` / `Board::Defeat` 不存在**

在 `src/` 全量检索这三个名字均无定义、无声明、无调用点 ⇒ **NOT FOUND**。
真实机制是「实体自己触发 → 写 `LawnApp::mBoardResult` / `Board::mLevelComplete` →
`LawnApp::EnvironmentTerminal()` 读出」。

### 4.2 失败：僵尸进屋（每 tick 在 zombie 更新里测）

**触发点 1（实体侧，位置检查）**：`Zombie::CheckForBoardEdge()`
`src/Lawn/Zombie.cpp:4537-4573`，由 `Zombie::UpdatePlaying()`（`src/Lawn/Zombie.cpp:4575`）
在 `mZombiePhase` 为正常游戏阶段时调用。逐字：

```
int aEdgeX = BOARD_EDGE;                                   // 4546 = -100（src/GameConstants.h:31）
if (type == GARGANTUAR | REDEYE_GARGANTUAR | POLEVAULTER)  aEdgeX = -150;   // 4547-4550
else if (type == CATAPULT | FOOTBALL | ZAMBONI)            aEdgeX = -175;   // 4551-4554
else if (type == BACKUP_DANCER | DANCER | SNORKEL)         aEdgeX = -130;   // 4555-4558

if (mX <= aEdgeX && mHasHead)                              // 4560  ★整型 mX
{
    if (IsIZombieLevel()) DieNoLoot();                     // 4562-4565
    else mBoard->ZombiesWon(this);                         // 4566-4569  ★第 7 关走这里
}
if (mX <= aEdgeX + 70 && !mHasHead) TakeDamage(1800, 9U);  // 4571-4574（无头身体自毁，不是失败）
```

**读到的实体标志**：`mX`（整型，`Zombie::Update` 末尾 `mX = (int)mPosX;`
`src/Lawn/Zombie.cpp:4385`，即比较的是**本 tick 更新后但截断为 int 的**位置），
`mHasHead`（`4560`），`mZombieType`（`4547-4558`）。
第 7 关自然池 `{NORMAL, TRAFFIC_CONE, POLEVAULTER}`（waves.md），其中
**`POLEVAULTER` 用 `aEdgeX = -150`，`NORMAL`/`TRAFFIC_CONE` 用 `-100`**。
另外 `IsWalkingBackwards() && mPosX > 850.0f ⇒ DieNoLoot()`（`src/Lawn/Zombie.cpp:4539-4543`），
这是回头僵尸逃出场地，不是失败。

⇒ **判定 tick = 让 `mX` 首次 `<= aEdgeX` 的那一 tick 的 `UpdateGameObjects` 僵尸循环内**，
`if (mX <= -100)` 且该僵尸 `mHasHead`。由于 `mX` 在本 tick 的 `Zombie::Update` 内被刷新，
比较用的是**刷新后的**值。

**触发点 2（board 侧，落旗）**：`Board::ZombiesWon(Zombie*)` `src/Lawn/Board.cpp:5068-5143`：

```
if (mApp->mGameScene == SCENE_ZOMBIES_WON) return;                    // 5070-5073  幂等门
ClearAdvice(ADVICE_NONE);                                             // 5075
mApp->mBoardResult = BoardResult::BOARDRESULT_LOST;                   // 5076  ★★失败在此定案
for (Zombie* z : mZombies) { …只处理尚未上场的（GetZombieRect().mX < -50 或 rising 相位）… }  // 5078-5095
SurvivalSaveScore();                                                  // 5096
（非 zombiquarium / 非 last-stand / 非 endless / 非 IZ 分支）
    mApp->mGameScene = GameScenes::SCENE_ZOMBIES_WON;                 // 5119  ★场景切换
    theZombie->WalkIntoHouse();                                       // 5121-5122
    mCutScene->StartZombiesWon();                                     // 5125
    FreezeEffectsForCutscene(true); TutorialArrowRemove(); UpdateCursor();
    return;                                                           // 5126
```

⇒ 失败 tick 之后：`mGameScene != SCENE_PLAYING`，
于是 `UpdateGame` 的 5680 门会 `return`（`mMainCounter` 停止增长、不再产太阳/出怪），
且 `Coin::Update` 的 734-737 门也 `return`。但 `EnvironmentTerminal()` 在**下一次循环条件**
（`src/LawnApp.cpp:1508`）就为真 ⇒ 循环结束，这些后续影响在 env 里看不到。

### 4.3 胜利：最后一个能掉 loot 的僵尸死亡

**触发点**：`Zombie::TrySpawnLevelAward()` `src/Lawn/Zombie.cpp:7138-7275`，
在僵尸死亡流程里被调用（`Zombie::DieWithLoot` `src/Lawn/Zombie.cpp:7321`；
`DropLoot` `src/Lawn/Zombie.cpp:7278`）。门与写入，逐字：

```
if (!IsOnBoard() || mBoard->HasLevelAwardDropped() || mBoard->mLevelComplete || mDroppedLoot)
    return false;                                                     // 7140-7144
if (IsFinalBossLevel())      { if (type != ZOMBIE_BOSS) return false; }  // 7146-7152
else if (IsScaryPotterLevel()){ if (!ScaryPotterIsCompleted()) return false; } // 7153-7159
else if (IsContinuousChallenge() || mBoard->mCurrentWave < mBoard->mNumWaves
         || mBoard->AreEnemyZombiesOnScreen())
    return false;                                                     // 7160-7163  ★第 7 关走这条
if (IsWhackAZombieLevel() && mZombieCountDown > 0) return false;      // 7165-7168
mBoard->mLevelAwardSpawned = true;                                    // 7169  ★★胜利闩
mApp->mBoardResult = BoardResult::BOARDRESULT_WON;                    // 7170  ★★胜利定案
RemoveAllZombies(); (非 repick)                                       // 7173-7176
```

⇒ **第 7 关胜利条件（精确）**：
`mCurrentWave == mNumWaves` **且** `AreEnemyZombiesOnScreen() == false`
**且** 该僵尸 `IsOnBoard()`、`!mDroppedLoot`，并且此前 `HasLevelAwardDropped()==false`。

* `AreEnemyZombiesOnScreen()`：`src/Lawn/Board.cpp:262-274` —— 扫描 `mZombies`，
  要求 `!mDead && mBodyHealth>0 && !mRetainingDamage && !mMindControlled && !mNoBlockDamage`
  并且 `mX < BOARD_WIDTH`（800）；找到返回 true。（具体字段组合见该行区间。）
* `IsOnBoard()`：`src/Lawn/Zombie.h`/`Zombie.cpp` —— `mZombiePhase != ZOMBIE_WAVE_CUTSCENE/UI/WINNER`
  且 `mBodyHealth > 0` 之类；**这条与 projectiles/waves 规格重叠，此处不重复推导。**
* `HasLevelAwardDropped()` = `mLevelAwardSpawned || mNextSurvivalStageCounter>0 || mBoardFadeOutCounter>=0`
  （`src/Lawn/Board.cpp:5191-5194`）。第 7 关白天 `mNextSurvivalStageCounter` 恒 0、
  `mBoardFadeOutCounter` 恒 -1 ⇒ 只有 `mLevelAwardSpawned` 起作用。

**判定 tick**：与失败不同，胜利是在**某僵尸 `Update()` 内部死亡流程**中被检测的，
即「第 `mNumWaves` 波全部敌人被清除」这件事必然发生在 `UpdateGameObjects` 的僵尸循环或子弹循环
（子弹击杀僵尸 ⇒ 僵尸 `Die/DieWithLoot`）内。`mBoardResult` 于该 tick 内被写；
`EnvironmentTerminal()` 在下一次循环条件（同一 tick 结束后）读到。

### 4.4 `EnvironmentTerminal()` 与 `mLevelComplete`

`src/LawnApp.cpp:1591-1595`，逐字：

```
bool LawnApp::EnvironmentTerminal() const
{
    return !mBoard
        || mBoard->mLevelComplete
        || mBoardResult == BoardResult::BOARDRESULT_WON
        || mBoardResult == BoardResult::BOARDRESULT_LOST
        || mGameScene == GameScenes::SCENE_ZOMBIES_WON;
}
```

第 7 关白天里各子的实际触发者：

| 子条件 | 谁写 | 何时 |
|---|---|---|
| `mBoardResult == WON` | `src/Lawn/Zombie.cpp:7170` | 胜利那一 tick 的实体更新内部 |
| `mBoardResult == LOST` | `src/Lawn/Board.cpp:5076` | 失败那一 tick 的僵尸更新内部 |
| `mGameScene == SCENE_ZOMBIES_WON` | `src/Lawn/Board.cpp:5119` | 同一 tick，紧跟 LOST 之后 |
| `mLevelComplete` | **第 7 关 env 路径上不会被置 true** | 三个写入点全不可达：`src/Lawn/Board.cpp:1737`（`mNextSurvivalStageCounter` 归零，只有 survival/scarypotter/laststand 会写它）、`src/Lawn/Board.cpp:1750`（需要 `mBoardFadeOutCounter >= 0`）、`src/Lawn/Board.cpp:1829`（`CompleteEndLevelSequenceForSaving`，只被存档/`CheckForGameEnd` 路径调用；`CheckForGameEnd` 在 env 下不跑，见 §2.3） |
| `!mBoard` | `KillBoard()`（`src/LawnApp.cpp:265-322`） | env 下只有 RESET 会调用 |

⇒ **env 第 7 关的终局完全由 `mBoardResult` 决定**，`mLevelComplete` 恒 `false`。
（`EnvironmentReset` 会 `mBoardResult = BOARDRESULT_NONE`：`src/LawnApp.cpp:1311-1464` 区间内。）

### 4.5 「胜利会放出剩余小推车」——**在 env 路径上不成立**

`src/main.cpp:479` 的注释
`// 胜利会放出剩余小推车，与 Python 一样保留最后非终局观测的数量。`
指的是这段代码（`src/main.cpp:476-481`）：

```
if (!app->EnvironmentTerminal()) mowersLeft = readyMowers();   // 479-480
```

原始代码里「放推车」的实现在 `Board::UpdateLevelEndSequence()`
`src/Lawn/Board.cpp:1761-1786`：

```
if (CanDropLoot() && !IsSurvivalStageWithRepick())     // 1761
{
    mScoreNextMowerCounter = 40;                        // 1763
    LawnMower* aLawnMower = GetBottomLawnMower();       // 1764  （最下行未使用的推车）
    if (aLawnMower)
    {
        AddCoin(aLawnMower->mPosX + 40, aLawnMower->mPosY + 40, COIN_GOLD, COIN_MOTION_LAWNMOWER_COIN);  // 1766
        …SOUND_POINTS，音高 clamp(6 - CountUntriggerLawnMowers(), 0, 6)…                                  // 1767-1773
        aLawnMower->Die();                              // 1774  ★推车被「放出去」= 消失换成金币
    }
}
```

但**这段整体位于 `if (mBoardFadeOutCounter < 0) return;`（`src/Lawn/Board.cpp:1744-1745`）之后**，
而 `mBoardFadeOutCounter` 初值 `-1`（`src/Lawn/Board.cpp:125`），
唯一把它变成 `>= 0` 的地方是 `Board::FadeOutLevel()`（`src/Lawn/Board.cpp:1899`、`1902`），
`FadeOutLevel()` 在 tick 循环里**只有一个调用者**：
`Board::UpdateZombieSpawning()` `src/Lawn/Board.cpp:5351`，门是
`if (mCurrentWave == mNumWaves && mApp->IsSurvivalMode())`（`src/Lawn/Board.cpp:5347-5352`）
⇒ **冒险模式不满足 `IsSurvivalMode()`**。
其它 `FadeOutLevel` 引用是声明（`src/Lawn/Board.h:350`）与存档字段（`src/Lawn/System/SaveGame.cpp:1659`）。

⇒ **第 7 关 env rollout 里推不会放，`mScoreNextMowerCounter` 恒 0，
`GetBottomLawnMower`（`src/Lawn/Board.cpp:1674-1690`）/`CountUntriggerLawnMowers`
（`src/Lawn/Board.cpp:305-318`）都不会被 tick 循环调用。**

内核要复现的只有 `main.cpp` 的**计数闩锁语义**：
`mowersLeft` 记录的是**最后一次「非终局」决策点**上的 ready 推车数，
即终局那一 tick 里就算有推车被吃掉也不会体现在 `mowers_left` 输出里。

### 4.6 推车触发（失败前的最后防线，per-tick，属于 `UpdateGameObjects`）

`LawnMower::Update` `src/Lawn/LawnMower.cpp:178-273`：

```
MOWER_READY 分支（178-…）:
    if (mBoard->mZombies 中存在 z：!z->mDead && z->mRow == mRow && z->mX <= -40 && !z->mDying && …)
        StartMower();                                             // → LawnMower.cpp:381+
MOWING 分支:
    mPosX += 3.0f;  …MowZombie()…  …若离开屏幕则 Die()
```

`StartMower()` `src/Lawn/LawnMower.cpp:381-…`：
`mBoard->mTriggeredLawnMowers++`（`src/Lawn/LawnMower.cpp:402`）、
`mBoard->mWaveRowGotLawnMowered[mRow] = mCurrentWave`（`src/Lawn/LawnMower.cpp:401`）。
`MowZombie()` `src/Lawn/LawnMower.cpp:143-176`。
`BOARD_EDGE` 门（`-100`）与推车触发线（`-40`）不同 ⇒
推车总是在进屋判定之前起作用（同一 tick 内先后：推车在 `UpdateGameObjects` 的第 5 个循环，
僵尸在第 2 个循环 ⇒ **同 tick 内僵尸的位置检查先于推车移动**；
但僵尸本 tick 的 `mX` 已被第 2 个循环刷新，故只要 `mX <= -100` 就判负，即使第 5 个循环会让推车救场）。
**这是内核最容易搞反的顺序，务必按 plants→zombies→projectiles→coins→mowers 的固定次序。**

---

## 5. 其它每 tick 修改游戏状态的 Board 逻辑

| # | 逻辑 | 源码位置 | 第 7 关白天可达？ | 若可达：算术 |
|---|---|---|---|---|
| 1 | `mBoardUpdateCounter++` | `src/Lawn/Board.cpp:5729` | 可达 | 每 tick `+1`；`InitLevel` 清 0（`src/Lawn/Board.cpp:1341`）。只被 `CutScene::Update` 的 `<=1` 门和渲染读（`src/Lawn/CutScene.cpp:1392`） ⇒ 内核可当 tick 镜像计数 |
| 2 | `mEffectCounter++` | `src/Lawn/Board.cpp:5801` | 可达 | 每 tick `+1`，**无条件、无其它副作用**（只在渲染里读）。`Board.h:179` |
| 3 | `mAppCounter++` | `src/LawnApp.cpp:1583` | 可达 | 每 `AdvanceLogicTick` `+1`；`RESET` 清 0（`src/LawnApp.cpp:1448`） |
| 4 | `mOutOfMoneyCounter` | 写 `src/Lawn/Board.cpp:5770-5773`（`--`）；置位 `src/Lawn/Board.cpp:8510`（`TakeSunMoney` 失败）、`src/Lawn/SeedPacket.cpp:765` | **可达** | 失败种植 ⇒ `= 70`，其后每 tick `if (>0) --`。纯 UI（金币不够闪烁），不改状态 |
| 5 | `mShakeCounter` / `mShakeAmountX/Y` | `src/Lawn/Board.cpp:5774-5791` | **可达** | `ShakeBoard` 置 `mShakeCounter = 12`（`src/Lawn/Board.cpp:9458-9463`；调用者 `src/Lawn/Plant.cpp:1566,2413,2421,4332,4346,4357,4397`、`src/Lawn/Projectile.cpp:861`、`src/Lawn/Zombie.cpp:2069,2139,2401`）。每 tick：`if (--mShakeCounter == 0) { mX=0; mY=0; } else { if (!Rand(3)) mShakeAmountX = -mShakeAmountX; mX = PvzpAnimateCurve(12,0,mShakeCounter,0,mShakeAmountX,CURVE_BOUNCE); mY = 同理; }` ⇒ **每 tick 1 次 `Rand(3)`（存活期间），影响 `Board` 这个 widget 的 mX/mY（渲染偏移）**。第 7 关能触发它的植物：cherry bomb 爆炸、potatomine 爆炸 ⇒ **可达且消耗 RNG，内核必须记账**（虽然只影响绘制位置） |
| 6 | `mCoinBankFadeCount` | `src/Lawn/Board.cpp:5792-5795` | 名义可达但恒 0 | 只在 `SetCoinBankFadeDone`（`src/Lawn/Board.cpp:4663`）赋值；门要求 `DIALOG_PURCHASE_PACKET_SLOT == nullptr`。纯 UI |
| 7 | `mCobCannonCursorDelayCounter` | `src/Lawn/Board.cpp:5765-5768` | **不可达** | 无玉米加农炮；恒 0 |
| 8 | `mTimeStopCounter` | `src/Lawn/Board.cpp:5797-5798`（`>0 ⇒ return`） | **不可达（恒 0）** | 只有 freeze/ice trap 路径写；第 7 关白天无。若为 0 则整 tick 正常推进 |
| 9 | `mIceTrapCounter` | `src/Lawn/Board.cpp:5688-5699`（`>0 ⇒ --`） | **不可达（恒 0）** | 只 pool/ice 关卡 |
| 10 | `mIceTimer[row]`（`UpdateIce`） | `src/Lawn/Board.cpp:5402-5438`：对 6 行 `if (mIceTimer[aRow]) { mIceTimer[aRow]--; …粒子… }` | **名义不可达** | 唯一写入 `mIceTimer[mRow] = 20` 在 `SEED_JALAPENO` 分支（`src/Lawn/Plant.cpp:4360`），牌组无 jalapeno ⇒ 恒 0。若写则：每 tick `--`，为 0 时 `mIceMinX[row] = BOARD_ICE_START` |
| 11 | `mFogBlownCountDown` | `src/Lawn/Board.cpp:5700-5703`（`>0 ⇒ --`）、`5663-5678` fog offset 曲线 | **不可达（恒 0）** | `!StageHasFog()` ⇒ `UpdateFog` 第一行 return（`src/Lawn/Board.cpp:7306-7309`） |
| 12 | `mGridCelFog[x][y] += aFogFadeInSpeed` | `src/Lawn/Board.cpp:7310-7329` | **不可达** | `UpdateFog` 顶部 `!StageHasFog() ⇒ return` |
| 13 | `UpdateGridItems()` | `src/Lawn/Board.cpp:9280-9304` | **实质不可达**（数组恒空） | 逐字：索引升序遍历 `mGridItems`；`if (mEnableGraveStones && type==GRAVESTONE && counter<100) counter++;`（`9286-9290`）⇒ **`mEnableGraveStones==false`**；`if (type==CRATER && mGameScene==SCENE_PLAYING) { if(counter>0) counter--; if(counter==0) GridItemDie(); }`（`9292-9301`）⇒ 第 7 关不产 crater（crater 只由 doom shroom/scary potter 产生：`src/Lawn/Plant.cpp:4345` 的 `AddACrater(…)->mGridItemCounter = 18000`）；`GridItem::Update()`（`src/Lawn/GridItem.cpp:559-589`）。第 7 关白天 `mGridItems.mSize == 0` ⇒ **整块可省** |
| 14 | `UpdateFwoosh()` | `src/Lawn/Board.cpp:9258-9278`：`if (mFwooshCountDown==0) return; int aFwooshRemaining = PvzpAnimateCurve(50,0,--mFwooshCountDown,12,0,CURVE_LINEAR); …` | **不可达** | 唯一写 `mFwooshCountDown = 100` 在 `Board::DoFwoosh(row)`（`src/Lawn/Board.cpp:9231-9256`）；`DoFwoosh` 只有两个调用者：`src/Lawn/Zombie.cpp:2400`（`UpdateZombieJalapenoHead`，即辣根僵尸头）和 `src/Lawn/Plant.cpp:4356`（`SEED_JALAPENO`）。**牌组无 jalapeno，第 7 关自然池无 jalapeno-head ⇒ `mFwooshCountDown` 恒 0**。若可达，`DoFwoosh` 每行消耗 12×3 次 RNG（`RandRangeFloat(0.7,1.3)`、`RandRangeFloat(0.9,1.1)`、`Rand(2)`；`src/Lawn/Board.cpp:9240-9250`） |
| 15 | `mChallenge->Update()` | `src/Lawn/Board.cpp:5817` → `src/Lawn/Challenge.cpp:2143-…` | **实质不可达（无副作用）** | 门：`IsStormyNightLevel()`（否）→ `mPaused`（否）→ `RAINING_SEEDS/StormyNight`（否）→ `mGameScene != SCENE_PLAYING && != TREE_OF_WISDOM ⇒ return`（**SCENE_PLAYING ⇒ 不 return**）→ `HasConveyorBeltSeedBank()`（否）→ `BEGHOULED*`（否）→ `IsScaryPotterLevel()`（否）→ … 剩余分支全部按 game mode 关闭 ⇒ 无任何字段变化。`Challenge::UpdateZombieSpawning()` 对 adventure 7 返回 **false**（`src/Lawn/Challenge.cpp:2906-2924`）⇒ 正常出怪继续 |
| 16 | `mWaveCountDown` | **NOT FOUND** —— 该字段不存在 | — | 真实字段是 `mZombieCountDown` / `mZombieCountDownStart` / `mHugeWaveCountDown` / `mRiseFromGraveCounter` / `mFinalWaveSoundCounter`（`src/Lawn/Board.h:192` 附近与 `src/Lawn/Board.cpp:142-143`）。它们的每 tick 递减全部在 `UpdateZombieSpawning`（`src/Lawn/Board.cpp:5261-5295`、`5346-5399`）⇒ 见 `waves.md` |
| 17 | `mPlantCountdown` | **NOT FOUND** —— 该字段不存在 | — | 种植冷却在 `SeedPacket::mRefreshCounter`（下表 #18），不在 Board 上 |
| 18 | SeedPacket 冷却 | `src/Lawn/Board.cpp:5062-5065` 调 `SeedPacket::Update`，`src/Lawn/SeedPacket.cpp:160-209` | **可达** | 顶部 `if (mGameScene != SCENE_PLAYING \|\| mPacketType == SEED_NONE) return;`（`162-165`）。`if (mBoard->mMainCounter == 0) FlashIfReady();`（`167-170`）⇒ **注意这是在 `mMainCounter++` 之前读的**，所以第一个 tick 读到 0。核心：`if (!mActive && mRefreshing) { mRefreshCounter++; if (mRefreshCounter > mRefreshTime) { mRefreshCounter = 0; mRefreshing = false; Activate(); } }`（`172-182`）⇒ 判据是 **`>` 不是 `>=`**，所以实际冷却为 `mRefreshTime + 1` tick。牌组字面值（`src/Lawn/Plant.cpp:49-68`）：peashooter 750 / sunflower 750 / cherry 5000 / wallnut 3000 / potatomine 3000 / snowpea 750 / repeater 750。`mSlotMachineCountDown`（`184-208`）恒 0（slot machine 关卡专属） |
| 19 | `UpdateProgressMeter()` | `src/Lawn/Board.cpp:5440-5508`（由 `5719` 调用） | **可达** | 走 `else if (mCurrentWave != 0)` 分支（`5457`）：`if (mNumWaves <= 1) { mProgressMeterWidth = 150; return; }`（`5461-5464`，30 波不触发）；**`if (mFlagRaiseCounter > 0) mFlagRaiseCounter--;`（`5466-5469`）** ⇒ 每 tick 减 1；`mFlagRaiseCounter = FLAG_RAISE_TIME(100)` 由 `NextWaveComing()`/`SpawnZombieWave` 在 flag 波置（`src/Lawn/Board.cpp:4960-4964`、`5233-5254`）。其余是纯宽度计算（`5471-5507`，含 `mMainCounter % 5 == 0` / `% 20 == 0` 的节流判据，**注意 `mMainCounter` 已含本 tick**）。⇒ 内核需要 `mFlagRaiseCounter` 的递减与 `%5`/`%20` 判据（若观测里有进度条字段） |
| 20 | `UpdateTutorial()` | `src/Lawn/Board.cpp:5510-5555` | **不可达** | `mTutorialState == TUTORIAL_NONE` ⇒ 首行 return；`mTutorialTimer` 恒 0 |
| 21 | `UpdateLevelEndSequence()` | `src/Lawn/Board.cpp:1692-1787` | **完全不可达（整块 return）** | `mNextSurvivalStageCounter == 0` ⇒ 跳过第一块（`1694-1743`）；`mBoardFadeOutCounter < 0` ⇒ `return`（`1744-1745`）。⇒ **第 7 关 env 里这个函数什么都不做**，`mScoreNextMowerCounter` 恒 0，`mLevelComplete` 永不置 true（见 §4.5） |
| 22 | `MessageWidget/Advice->Update()` | `src/Lawn/MessageWidget.cpp:238-…`，`src/Lawn/Board.cpp:5762` | 可达但仅 UI | 内含 `if (gEnvSimSkip & ENV_SKIP_MESSAGE) return;`（`src/Lawn/MessageWidget.cpp:259`）⇒ rollout mask 里位 1 会跳过；**不消耗游戏 RNG** |
| 23 | `mEffectSystem->Update()`（粒子/Reanim） | `src/Lawn/Board.cpp:5761` | 可达，仅表现 | 由 `ENV_SKIP_*` 位（`src/PvzpLib/PvzpCommon.h:180-195`）控制。**注意：粒子/Reanim 更新里存在 `Rand` 调用** ⇒ 这也是掩码只允许 1|4|8|16 的原因（§2.4） |
| 24 | CursorPreview / CursorObject | `src/Lawn/Board.cpp:5059-5060` | 可达，无游戏状态 | `CursorObject::Update`（`src/Lawn/CursorObject.cpp:49-72`）、`CursorPreview::Update`（`src/Lawn/CursorObject.cpp:238-…`）只维护动画/可见性 ⇒ **内核可完全省略** |
| 25 | 网格坐标 `GridToPixel*` / `PixelToGrid` | 大量引用 | 无状态 | 纯函数，见 `waves.md` 的行/列映射 |

**其它 tick 内被修改的 Board 标量（无独立计时器，供观测用）**：
`mSunMoney`（`src/Lawn/Board.cpp:8465`）、
`mSunMoneyProduced`（`src/Lawn/Board.cpp:1975`，`Board.h:242`，`uint32_t`）、
`mNumSunsFallen`（`src/Lawn/Board.h:167`）、
`mTriggeredLawnMowers`（`src/Lawn/LawnMower.cpp:402`）、
`mWaveRowGotLawnMowered[row]`（`src/Lawn/LawnMower.cpp:401`）、
`mPlantsShoveled`/`mPlantsEaten`（`EnvironmentShovel` 用 `mPlantsShoveled` 判是否改变，`src/LawnApp.cpp:1489-1502`）、
`mLevelAwardSpawned`（`src/Lawn/Zombie.cpp:7169`）、
`mCoinsCollected`/`mLevelCoinsCollected`/`mDiamondsCollected`（`src/Lawn/Coin.cpp:447-460`，sun 不走）。

---

## 6. `src/main.cpp` 的 `ROLLOUT` 命令块：与 §2 的对账

### 6.1 命令读取与前置校验（`src/main.cpp:390-412`）

```
ROLLOUT rootId seed untilTick mask maxActions eventWait first historyStart historyCount [ (tick produced) × historyCount ] [extra]
```
* `rootId == 0` ⇒ 用当前 board；否则必须 `snapshots.contains(rootId)` 并 `RestoreEnvironmentSnapshot`
  （`src/main.cpp:399`、`401`、定义 `246-261`）。
* `seed >= 0` ⇒ `ReseedEnvironment(app, seed)`（`src/main.cpp:402`、定义 `268-281`：
  写 `mAppRandSeed`/`mRandSeed`、`Sexy::SRand(seed)`、`std::srand(seed)`）。
* `before = ReadCounters(app)`（`src/main.cpp:404`、定义 `155-161`）⇒ **必须在 restore/reseed 之后**，
  否则事件计数是上一条命令留下的（注释在 `src/main.cpp:403-404`）。
* `history` 校验：`tick >= historyStart && tick <= app->mBoard->mMainCounter && (history.empty() || tick >= history.back().first)`
  ⇒ **history 按 tick 非降序**，`src/main.cpp:405-411`。
* `first == "A"` ⇒ 首个决策由网络出；否则把 `first` 解成一条命令 `actionCommand/actionArguments`
  （`DecodeBranchAction`，`src/main.cpp:310-322`）。
* `historyStart ∈ [0, mMainCounter]`（`src/main.cpp:414-416`）。

### 6.2 rollout 主循环（`src/main.cpp:432-481`），逐字伪序

```
savedMask = gEnvSimSkip;  gEnvSimSkip = mask;                      // 424-425
startTick = app->mBoard->mMainCounter;                             // 426
if (untilTick <= mMainCounter) untilTick = INT_MAX;                // 427-428
readyMowers = [&]{ int c=0; for (LawnMower* m : mLawnMowers)       // 429-430
                  if (!m->mDead && m->mMowerState == MOWER_READY) ++c; return c; };
mowersLeft  = readyMowers();                                       // 431  ★进入即取一次
rollout.ResetNetworkCalls();                                       // 432
while (!app->EnvironmentTerminal() && decisions < maxActions)      // 433
{
    tick          = app->mBoard->mMainCounter;                     // 435  ★决策前 tick
    producedBefore= app->mBoard->mSunMoneyProduced;                // 436  ★uint32 快照
    isFirst = (decisions == 0 && first != "A");                   // 438
    if (!isFirst) {                                                // 439-449
        while (!history.empty() && history.front().first < tick - 6000) history.pop_front();  // 444
        produced = Σ history[i].second;                            // 445-446
        elapsed  = max(0, min(6000, tick - historyStart));         // 447
        action   = rollout.Predict(app, elapsed ? produced*1000.0/elapsed : 0, priorTick, priorPlant);  // 448
    }
    if (isFirst) {                                                 // 453-464
        if (actionCommand=="WAIT" && eventWait) EnvironmentWaitUntil(ticks, 0);
        else                                    ExecuteEnvironmentAction(...);   // PLANT / SHOVEL / WAIT
        priorPlant = (actionCommand=="PLANT" && accepted);
    } else {                                                       // 465-470
        priorPlant = action.packet >= 0;
        if (priorPlant) accepted = app->EnvironmentPlant(packet, col, row);
        else            app->EnvironmentWaitUntil(60, 0);           // ★默认 60 tick
    }
    if (!accepted) { app->EnvironmentWait(60); priorPlant = false; } // 471  ★失败也走 60 tick
    if (priorPlant) ++planted[action.packet];                       // 472
    priorTick = tick;                                               // 473
    now      = app->mBoard->mMainCounter;                           // 474
    produced = app->mBoard->mSunMoneyProduced - producedBefore;     // 475  ★32 位无符号差
    if (produced) history.emplace_back(now, produced);              // 476-477  ★本进程内追加
    substeps.emplace_back(now, produced);                            // 478  ★输出用（无条件）
    ++decisions;                                                    // 481
    if (!app->EnvironmentTerminal()) mowersLeft = readyMowers();    // 480  ★只在非终局时更新
    if (now >= untilTick) break;                                    // 482
}
gEnvSimSkip = savedMask;                                            // 490
```

### 6.3 §2 与 rollout 实际驱动的一致性结论

* rollout 的**唯一推进手段**是 `EnvironmentWait` / `EnvironmentWaitUntil(…, 0)`
  （`src/main.cpp:458, 461, 469, 471`），后者内部也只是
  `advance = [&]{ EnvironmentWait(1); }`（`src/LawnApp.cpp:1576-1578`）。
  ⇒ **§2.1 的三步 tick 就是 rollout 每一步的全部，没有别的路径。** 描述无需修正。
* `EnvironmentWaitUntil` 每次 advance 之后调 `EnvironmentPublicWaitState()`
  （`src/LawnApp.cpp:1537-1572`）做一次**全量公共状态扫描**（含 `std::set` 去重、
  `RenderedPublicCoordinate` 的 6 位有效数字流化）。它**只读不写游戏状态**、**不消耗游戏 RNG**
  ⇒ 内核不必复现，除非要实现事件等待（见 §6.5）。
* `main.cpp` 在两次决策之间做的、内核必须复现的簿记（**它们不是 tick 的一部分，但属于 rollout 语义**）：

  1. **收入历史 `history`**：窗口 6000 tick，按 `tick - 6000` 出队
     （`src/main.cpp:444`），特征值 = `produced * 1000.0 / elapsed`
     （`src/main.cpp:447-448`），`elapsed = max(0, min(6000, tick - historyStart))`，
     `elapsed == 0 ⇒ 特征值 0.0`。
     `produced` 用 `mSunMoneyProduced` 的**无符号 32 位差**（`src/main.cpp:436, 475`），
     即 `AddCoin` 累计产出（`src/Lawn/Board.cpp:1975`），**不是** `mSunMoney`（会被花掉）。
  2. **只在 `produced != 0` 时把 (now, produced) 压入 `history`**（`src/main.cpp:476-477`），
     但 **`substeps` 无条件压入**（`src/main.cpp:478`）⇒ 两者长度可以不同。
  3. **`planted[10]` 数组按 packet 下标统计成功种植数**（`src/main.cpp:472`），
     只有 `accepted && priorPlant` 才计。
  4. **`mowersLeft` 的非终局闩锁**（`src/main.cpp:480-481`）——
     这是 §4.5 那条注释对应的唯一真实语义。
  5. **`priorTick` / `priorPlant` 作为网络特征传入下一次 `Predict`**（`src/main.cpp:448, 473`）。
  6. **计时**：`simulationSeconds` 只统计动作执行（`src/main.cpp:451, 479`），
     `networkSeconds` 只统计 `Predict`（`src/main.cpp:443, 449`），互不重叠。

### 6.4 `WAIT` / `WAIT_EVENT_V1` / `WAIT_UNTIL_LEGAL` 的差异（`src/main.cpp:540-586`）

* `WAIT ticks` ⇒ `app->EnvironmentWait(ticks)`：纯 §2.1，无事件（`src/main.cpp` 的 WAIT 分支）。
* `WAIT_EVENT_V1 ticks condition` ⇒ `EnvironmentWaitUntil(ticks, condition)`（`src/main.cpp:549-559`），
  返回 `reason`/`triggered` 列表。
* `WAIT_UNTIL_LEGAL ticks` ⇒ `EnvironmentWaitUntil` 风格循环 + `EnvironmentAnyLegalPlant()`
  （`src/main.cpp:560-586`、`src/LawnApp.cpp:1706-1721`）⇒ 每 tick 额外扫一遍
  (packet, col, row) 的 `CanPlantAt`，**只读，不消耗 RNG**。

### 6.5 `EnvironmentWaitEvents`（`src/EnvironmentWaitEvents.h`，**只有头文件，`.cpp` NOT FOUND**）

`Version = 1`（`src/EnvironmentWaitEvents.h:15`），`LeftZoneMaxX = 160.0`（`:16`），
条件表（`:17-20`）：
`0 timeout, 1 wave_changed, 2 packet_became_ready, 3 sun_increased, 4 left_zone_occupied, 5 plant_count_decreased`。

`Run` 循环（`:80-106`）——顺序即内核顺序：

```
if (requestedTicks < 0 || > 1000000 || condition < 0 || >= 6) throw;      // 83-84
initial  = readPublic();  previous = initial;                              // 88-89
triggered = Triggered(initial, initial, condition, requestedTicks == 0);   // 90  ★先用自己当 before
initialConditionSatisfied = (condition == 4 && !initial.leftZoneZombies.empty());  // 91
while (triggered.empty() && logicSteps < requestedTicks) {                 // 92
    advance();                                    // 94  == EnvironmentWait(1) == §2.1 三步
    ++logicSteps;                                  // 95
    current = readPublic();                         // 96
    if (current.tick < previous.tick) throw;         // 97-98
    stalledClockSteps += (current.tick == previous.tick);  // 99  ★mMainCounter 不涨的 tick（time-stop 类）
    actualTicks = current.tick - initial.tick;      // 100
    triggered   = Triggered(previous, current, condition, logicSteps == requestedTicks);  // 101
    previous = current;                             // 102
}
reason = triggered.front();                          // 104  ★空则 UB；但循环退出条件保证非空（或 logicSteps 到界）
```

`Triggered` 的**严格优先级**（`src/EnvironmentWaitEvents.h:62-78`，返回顺序即优先级，全部保留）：

```
1. after.terminal                                     → "terminal"                // 66
2. HasNew(after.readyDefenses, before.readyDefenses)  → "defense_lost"             // 67  ★注意方向：after 里有、before 里没有 ⇒ 推车「新增」？
3. HasNew(before.leftZoneZombies, after.leftZoneZombies)→ "zombie_entered_left_zone"// 68
4. 选中条件                                            → "condition"                // 69-75
5. deadline                                            → "max_ticks"                // 76
```

**第 2 条方向必须逐字照抄**（`HasNew(before, after)` 的定义是「`after` 中存在 `before` 没有的元素」，
`src/EnvironmentWaitEvents.h:55-60`）：
`HasNew(after.readyDefenses, before.readyDefenses)` 展开成
「`before.readyDefenses` 中存在 `after.readyDefenses` 没有的元素」⇒ **推车消失**才触发
`defense_lost`。命名与实参顺序看起来反直觉，但语义正确：
`HasNew(A,B)` = 遍历 `B`、查 `A`，所以 `HasNew(after, before)` = 「before 有、after 无」= 丢失。
第 3 条 `HasNew(before.leftZoneZombies, after.leftZoneZombies)` = 「after 有、before 无」= 新僵尸进入左区。

`leftZoneZombies` 的判据：`zombie->IsOnBoard() && rendered_x <= 160.0`，
其中接近边界（`|mPosX - 160.0| < 1.0`）时用 `RenderedPublicCoordinate`
（`std::ostringstream` 默认 6 位有效数字再 `stod`，`src/EnvironmentWaitEvents.h:46-53`），
否则用裸 `mPosX`（`src/LawnApp.cpp:1563-1568`）。
`mPosX` 由 `Zombie::Update` 的 `mX = (int)mPosX` 之外**不截断**（`src/Lawn/Zombie.cpp:4385`
只同步整型镜像）⇒ 事件判定用浮点，胜负判定用整型，**两者可以差一 tick**，
内核要分别按各自的数据类型实现。

`readyDefenses` 收集：`!mower->mDead && mower->mMowerState == MOWER_READY`，键 `(mRow, mMowerType)`
（`src/LawnApp.cpp:1552-1556`）。
`readyPackets` 收集：`packet.mActive && packet.mPacketType != SEED_NONE`，
键 `(index, mPacketType, mImitaterType)`（`src/LawnApp.cpp:1544-1550`）。
`plantCount` = `mPlants` 中 `!mDead` 的个数（`src/LawnApp.cpp:1542-1543`）——
**注意 `EnvironmentPublicWaitState` 这里没有跳 `mDead` 之外的过滤，
而 `UpdateGameObjects` 会跳；两者一致。**

---

## 7. 给内核的最小 tick 蓝图（把上面压成一页）

```
# 状态：mMainCounter, mBoardUpdateCounter, mEffectCounter, mAppCounter,
#       mSunMoney, mSunMoneyProduced(u32), mNumSunsFallen, mSunCountDown,
#       mZombieCountDown, mZombieCountDownStart, mFlagRaiseCounter, mProgressMeterWidth,
#       mOutOfMoneyCounter, mShakeCounter, mShakeAmountX/Y,
#       mCurrentWave, mNumWaves, mLevelAwardSpawned, mBoardResult, mGameScene,
#       槽数组 plants/zombies/projectiles/coins/mowers/griditems(空)
#       恒不活跃：mPaused=F, mTimeStopCounter=0, mIceTrapCounter=0, mIceTimer[6]=0,
#                 mFogBlownCountDown=0, mFwooshCountDown=0, mRiseFromGraveCounter=0,
#                 mFinalWaveSoundCounter=0, mHugeWaveCountDown(见 waves.md),
#                 mNextSurvivalStageCounter=0, mBoardFadeOutCounter=-1, mScoreNextMowerCounter=0,
#                 mLevelComplete=F, mCobCannonCursorDelayCounter=0, mCoinBankFadeCount=0

def env_tick():                                    # = 一次 EnvironmentWait(1)
    if terminal(): return                          # LawnApp.cpp:1508（终局则本 tick 完全不跑）
    for c in coins by slot asc:                    # LawnApp.cpp:1510-1513
        if not c.dead and c.type in (SUN, SMALLSUN, LARGESUN):
            c.dead = True                          # Coin.cpp:435
            sun = min(sun + sun_value(c.type), 9990)   # Coin.cpp:437-441 + Board.cpp:8463-8469
    app_counter += 1                               # LawnApp.cpp:1583
    process_delete_queue(plants, zombies, projectiles, coins, mowers, griditems)  # LawnApp.cpp:1584 / Board.cpp:8519-8575
    board_update():
        board_update_counter += 1                  # Board.cpp:5729
        # 跳过：cutscene / mouse / zen / crazydave / paused / buttons
        if out_of_money_counter > 0: out_of_money_counter -= 1   # Board.cpp:5770-5773
        if shake_counter > 0:                      # Board.cpp:5774-5791（只影响渲染偏移，但消耗 RNG）
            shake_counter -= 1
            if shake_counter == 0: board.x = board.y = 0
            else:
                if rand(3) == 0: shake_amount_x = -shake_amount_x
                board.x = curve_bounce_12(shake_counter, shake_amount_x)
                board.y = curve_bounce_12(shake_counter, shake_amount_y)
        # 跳过：coin_bank_fade / UpdateLayers / time_stop
        effect_counter += 1                        # Board.cpp:5801
        update_grid_items()                        # Board.cpp:5813 —— 第 7 关恒空，可省
        update_fwoosh()                            # Board.cpp:5814 —— 第 7 关恒 0，可省
        update_game():
            update_game_objects()                  # Board.cpp:5662 → 5022-5065（顺序：plants→zombies→projectiles→coins→mowers→cursor→packets）
            # fog 块跳过                             Board.cpp:5663-5678
            # game_scene != PLAYING 门：第 7 关为 PLAYING，不 return  Board.cpp:5680
            m_main_counter += 1                    # Board.cpp:5684 ★实体更新之后
            update_sun_spawning()                  # Board.cpp:5685 → 5196-5232（§3.1）
            update_zombie_spawning()               # Board.cpp:5686 → 5256-5400（waves.md）
            update_ice()                           # Board.cpp:5687 —— 恒 0，可省
            if ice_trap_counter > 0: ice_trap_counter -= 1   # Board.cpp:5688 —— 恒 0
            if fog_blown_countdown > 0: fog_blown_countdown -= 1  # Board.cpp:5700 —— 恒 0
            # tutorial 块跳过                       Board.cpp:5705（IsFirstTimeAdventureMode==false）
            update_progress_meter()                # Board.cpp:5719 → 5440-5508（含 mFlagRaiseCounter--）
            update_fog()                           # Board.cpp:5816 —— 立即 return
            challenge.update()                     # Board.cpp:5817 —— 无副作用
            update_level_end_sequence()            # Board.cpp:5818 —— 立即 return（mBoardFadeOutCounter<0）

def terminal():                                    # LawnApp.cpp:1591-1595
    return (board is None) or level_complete or board_result in (WON, LOST) or game_scene == ZOMBIES_WON
```

胜负写入点只有两处：`src/Lawn/Board.cpp:5076`（LOST，由 `src/Lawn/Zombie.cpp:4566` 调）与
`src/Lawn/Zombie.cpp:7169-7170`（WON）。两者都发生在 `update_game_objects()` 内部，
所以 **`env_tick` 结束后再测 `terminal()` 即可**（`src/LawnApp.cpp:1508` 就是这么做的）。
