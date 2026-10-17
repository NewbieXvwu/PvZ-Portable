# CUDA 机制规格：天降阳光、阳光生产、草坪修剪机、种子槽刷新

范围 = 任务书点定的四个主题，别的不动：
① 天降阳光（关卡定时器掉的太阳）；② 产阳光植物的产出物**仅就其与天降阳光不同之处**；
③ 草坪修剪机（小推车）；④ 种子槽刷新与种植合法性门控。

交叉引用（**这些文件已经写过的内容本文不重复**）：
- `cuda/spec/protocol.md` §1.4/§1.5（一步 tick 的确切推进、可以提前结束的等待）、§1.6（PLANT 的执行链）、
  §3.2（`scalars[0]`、`scalars[5+k]`、`scalars[59+k]`）、§4（`mowers_left` 口径）、§7.2（exporter 字段清单）。
- `cuda/spec/board.md`（`UpdateGameObjects` 的对象顺序、`InitLawnMowers` 在 5-行草地给 5 台车）。
- `cuda/spec/plants.md`（`mLaunchRate`、生产者/射手的重抽区间语义、`Plant::UpdateProduce` 的植物侧）。
- `cuda/spec/waves.md`（`mWaveRowGotLawnMowered` 参与的选行权重表；本文只给"推车写了什么"）。

判据：所有数字与谓词都要 `file:line`。凡"依赖关卡/场景表"的量，都按**冒险第 7 关（白天草地）**求值。
第 7 关的场景判定：`Board::PickBackground`（src/Lawn/Board.cpp:858-866）→ `mLevel <= 1 * LEVELS_PER_AREA`
→ `BACKGROUND_1_DAY`；`StageIsNight()`（src/Lawn/Board.cpp:8654-8662）因此为**假**，
`StageHasPool()`（8684-8687）为**假**。环境模式固定 `mGameMode = GAMEMODE_ADVENTURE`
（src/LawnApp.cpp:1394），`mApp->mGameMode == GAMEMODE_CHALLENGE_SUNNY_DAY` 的所有分支都不成立。

---

## 0. 常量表

### 0.1 天降阳光

| 名称 | 值 | 出处 |
|---|---|---|
| `SUN_COUNTDOWN`（间隔基数） | **425** | src/Lawn/Board.cpp:65 |
| `SUN_COUNTDOWN_RANGE`（抖动） | **275** | src/Lawn/Board.cpp:66 |
| `SUN_COUNTDOWN_MAX`（基数上限） | **950** | src/Lawn/Board.cpp:67 |
| 每掉一颗太阳的基数增量 | **+10 × mNumSunsFallen** | src/Lawn/Board.cpp:5228 |
| 开局首发相位（白天） | `mSunCountDown = RandRangeInt(425, 700)` | src/Lawn/Board.cpp:1386-1389 |
| 开局首发相位（夜晚） | `mSunCountDown = 0`（构造函数值，且夜晚整段早退） | src/Lawn/Board.cpp:116、1386（`if (!StageIsNight())`）、5198 |
| `mNumSunsFallen` 复位 | 0（`InitLevel`） | src/Lawn/Board.cpp:1385 |
| `mSunMoneyProduced` 复位 | 0（Board 构造） | src/Lawn/Board.cpp:159 |
| `mSunMoney` 第 7 关初值 | **50**（最后一个 `else`） | src/Lawn/Board.cpp:1354-1373 |
| 调度器调用点 | `mMainCounter++` **之后**、`UpdateZombieSpawning()` 之前 | src/Lawn/Board.cpp:5683-5684 |
| 递减与判零 | `mSunCountDown--; if (mSunCountDown != 0) return;` | src/Lawn/Board.cpp:5223-5226 |
| 计数与重排 | `mNumSunsFallen++` 先于重排 | src/Lawn/Board.cpp:5227-5228 |
| 产出 x | `RandRangeInt(100, 649)`（**闭区间**，550 个取值） | src/Lawn/Board.cpp:5230；实现 src/PvzpLib/PvzpCommon.cpp:384-388 |
| 产出 y | **60** | src/Lawn/Board.cpp:5230 |
| 太阳类型 | `COIN_SUN`（`SUNNY_DAY` 挑战才用 `COIN_LARGESUN`） | src/Lawn/Board.cpp:5229-5230 |
| 运动类型 | `COIN_MOTION_FROM_SKY` | src/Lawn/Board.cpp:5230 |
| 下落速度 | `mVelY = 0.67f` px/tick，`mVelX = 0` | src/Lawn/Coin.cpp:315-317 |
| 落地高度 | `mGroundY = Rand(250) + 300` ∈ **[300, 549]** | src/Lawn/Coin.cpp:318 |
| 太阳面值 | `COIN_SUN` **25**、`COIN_SMALLSUN` 15、`COIN_LARGESUN` 50 | src/Lawn/Coin.cpp:1297-1300 |
| 太阳判定 | `IsSun()` = 上述三类 | src/Lawn/Coin.cpp:423-426 |
| 落地后存活 | `GetDisappearTime()` = **750** tick（只有钻石/盆栽/巧克力/带弹跳箭头才 1500） | src/Lawn/Coin.cpp:1495-1515；计数 src/Lawn/Coin.cpp:577-588 |
| 淡出 | `StartFade()` 置 `mFadeCount = 15`；`UpdateFade()` 每 tick 减，减到 0 → `Die()` | src/Lawn/Coin.cpp:464-479 |
| `mCoinAge` | 每 tick +1（在场景检查之前） | src/Lawn/Coin.cpp:730-732 |
| **生产记账（唯一写入点）** | `if (aCoin->IsSun()) mSunMoneyProduced += aCoin->GetSunValue();` —— 在 **`AddCoin` 发射时**，不是收集时 | src/Lawn/Board.cpp:1971-1976 |
| 收集记账 | `Coin::ScoreCoin()` → `mBoard->AddSunMoney(GetSunValue())` | src/Lawn/Coin.cpp:433-441 |
| `AddSunMoney` 上限 | `mSunMoney = std::min(mSunMoney, 9990)` | src/Lawn/Board.cpp:8463-8466 |
| 无头自动收集 | `EnvironmentWait` 每轮循环**开头**把所有存活太阳 `ScoreCoin()`，然后才 `AdvanceLogicTick()` | src/LawnApp.cpp:1504-1519 |
| 收入窗统计的是**生产** | `producedBefore = mSunMoneyProduced` … `produced = mSunMoneyProduced - producedBefore` | src/main.cpp:435、473-475 |
| `mSunMoneyProduced` 类型 | `uint32_t` | src/Lawn/Board.h:242 |
| `Rand(n)` 语义 | `MTRand::Next(n) = Next() % n` → **[0, n-1]** | src/SexyAppFramework/Common.cpp:113-117；misc/MTRand.cpp:178-187 |

### 0.2 产阳光植物的产出物（只列与天降阳光不同处）

| 名称 | 值 | 出处 |
|---|---|---|
| 发射调用 | `mBoard->AddCoin(mX, mY, COIN_SUN, COIN_MOTION_FROM_PLANT)` —— **同一漏斗**，所以生产同样记在发射 | src/Lawn/Plant.cpp:1050（向日葵）、1054-1055（双胞向日葵两颗）、1045/1041（阳光菇小形态是 `COIN_SMALLSUN` = 15） |
| 发射位置 | 植物自身的 `mX, mY`（不是随机 x、不是 y=60） | src/Lawn/Plant.cpp:1050 |
| 运动 | `mVelY = -1.7 - RandRangeFloat(0,1.7)`（先上抛），`mVelX = -0.4 + U(0,0.8)`，`mGroundY = mPosY + 15 + Rand(20)`，`mScale = 0.4` | src/Lawn/Coin.cpp:327-332 |
| 重力 | 下落每 tick `mVelY += 0.09f`（`FROM_SKY` **没有**重力，匀速） | src/Lawn/Coin.cpp:497-500 |
| 放大 | `mScale += 0.02` 直到 `GetSunScale()`（`COIN_SUN` = 1.0） | src/Lawn/Coin.cpp:591-602、1292-1295 |
| 重抽时钟 | `mLaunchCounter = RandRangeInt(mLaunchRate - 150, mLaunchRate)`（闭区间 151 个取值），且在 `<= 0` 分支里**先重抽后发太阳** | src/Lawn/Plant.cpp:1026-1035 |
| 存活/淡出/面值 | 与天降阳光同表（`IsSun()` 为真即同一套 `GetDisappearTime`/`StartFade`/`ScoreCoin`） | src/Lawn/Coin.cpp:1495-1515、464-479、433-441 |

### 0.3 草坪修剪机

| 名称 | 值 | 出处 |
|---|---|---|
| 状态枚举 | `MOWER_ROLLING_IN=0, MOWER_READY=1, MOWER_TRIGGERED=2, MOWER_SQUISHED=3` | src/ConstEnums.h:569-575 |
| 第 7 关台数与行 | 5 台（row 0..4，`mPlantRow[aRow] != PLANTROW_DIRT` 的行） | src/Lawn/Board.cpp:1603-1623（另见 cuda/spec/board.md:36） |
| 构造位置/状态 | `mPosX = -160.0f`，`mMowerState = MOWER_READY`，`mChompCounter = mRollingInCounter = mSquishedCounter = 0` | src/Lawn/LawnMower.cpp:30-43 |
| **入场后停靠 x** | **-21.0f**（过场把 `mPosX` 从 -80 缓动到 -21，时长 250 tick） | src/Lawn/CutScene.cpp:96-97（`TimeLawnMowerDuration=250`、`TimeLawnMowerStart[6]={6300,6250,6200,6150,6100,6050}`）、1252-1262 |
| 环境模式确实走完入场 | `EnvironmentReset` 把 `mCutsceneTime = 1000000` 后 `CancelIntro()`，后者把过场快进到尾并调 `PlaceLawnItems()`（→ `InitLawnMowers()`），且 `mCutsceneTime > …` 使 1261 行的缓动取右端点 | src/LawnApp.cpp:1406-1410；src/Lawn/CutScene.cpp:959-1006、603-621、1250-1265 |
| 攻击矩形 | `Rect(mPosX, mPosY, 50, 80)` | src/Lawn/LawnMower.cpp:417-420 |
| **`GetRectOverlap` 只看 x 轴** | `min(x1+w1, x2+w2) - max(x1, x2)`（y 完全不参与） | src/Lawn/Board.cpp:8969-8973 |
| 僵尸矩形（普通/路障/旗帜的默认值） | `Rect(36, 0, 42, 115)`，按 `(mX, mY+bodyY)` 平移，`mX = (int)mPosX` | src/Lawn/Zombie.cpp:201、8301-8319、4389-4390 |
| 僵尸矩形（撑杆跳） | 未覆盖 → 沿用默认 `Rect(36,0,42,115)`（落地瞬间另有 `Rect(20,0,42,115)`） | src/Lawn/Zombie.cpp:1913 |
| 气球僵尸门槛 | 需要 `overlap > 20`，其他 `> 0` | src/Lawn/LawnMower.cpp:225 |
| 循环门控 | `!mDead`；跳过 `ZOMBIE_BOSS`；`mRow == 本车行`；相位 ≠ `PHASE_ZOMBIE_MOWERED`；非 `IsTangleKelpTarget()`；`EffectedByDamage(127U)` | src/Lawn/LawnMower.cpp:207-222 |
| `127U` 的位含义 | `127 = 0b1111111` = bit0..bit6 = GROUND \| FLYING \| SUBMERGED \| DOG \| OFF_GROUND \| **DYING** \| **UNDERGROUND**；只**没**有 bit7 `DAMAGES_ONLY_MINDCONTROLLED`。∴ `Zombie.cpp:8086` 的 `!TestBit(…, DAMAGES_DYING) && IsDeadOrDying()` 早退**对车永远不触发** —— 濒死僵尸照样能被车碾 | src/ConstEnums.h:257-266；src/Lawn/Zombie.cpp:8084-8089 |
| `EffectedByDamage(127U)` 为 **false** 的只有两种 | ① 雪橇队里非领头（`mZombieType == ZOMBIE_BOBSLED && GetBobsledPosition() > 0`）；② 非雪橇僵尸已经离场 `GetZombieRect().mX > WIDE_BOARD_WIDTH(800)` | src/Lawn/Zombie.cpp:8133-8135、8154-8157 |
| `EffectedByDamage(127U)` 为 **true** 的相位 | 腾空/特殊相位表（`PHASE_POLEVAULTER_IN_VAULT`、`PHASE_RISING_FROM_GRAVE`、`PHASE_DANCER_RISING`、`PHASE_SNORKEL_INTO_POOL`、`PHASE_DOLPHIN_IN_JUMP`、`PHASE_BALLOON_POPPING`…）→ `return TestBit(127, DAMAGES_OFF_GROUND)` = **true**；另外 SUBMERGED / UNDERGROUND（`PHASE_DIGGER_TUNNELING`）/ FLYING / GROUND 分支各自命中 | src/Lawn/Zombie.cpp:8138-8151、8159-8176 |
| READY 触发附加条件 | `mMowerState != MOWER_READY \|\| (type != ZOMBIE_BUNGEE && aZombie->mHasHead)` | src/Lawn/LawnMower.cpp:228 |
| 首次触发 | `MowZombie()`: READY → `StartMower(); mChompCounter = 25;` | src/Lawn/LawnMower.cpp:143-149 |
| 后续每杀 | 已是 TRIGGERED → `mChompCounter = 50`（**每次新杀都重置**） | src/Lawn/LawnMower.cpp:150-153 |
| `StartMower()` | 已 TRIGGERED 直接 return；`mWaveRowGotLawnMowered[mRow] = mCurrentWave`；`mTriggeredLawnMowers++`；`mMowerState = MOWER_TRIGGERED` | src/Lawn/LawnMower.cpp:382-404 |
| 击杀方式 | 草地车 → `theZombie->MowDown()`（泳池车 → `DieWithLoot()`），**不是**扣固定伤害 | src/Lawn/LawnMower.cpp:155-175；`MowDown` src/Lawn/Zombie.cpp:8584+ |
| `MowDown()` 早退 | `mDead \|\| mZombiePhase == PHASE_ZOMBIE_MOWERED \|\| mZombieType == ZOMBIE_BOSS` → return（所以同一 tick 车可以扫到多个僵尸，但已被本车碾过的不会重复计数） | src/Lawn/Zombie.cpp:8586-8587 |
| `MowDown()` 的两条死路 | ① 清单命中 → 冒烟 + （非泳池行）掉头/臂/盔/盾 → `DieWithLoot()`：相位 `PHASE_ZOMBIE_DYING`/`PHASE_ZOMBIE_BURNED`/`PHASE_POLEVAULTER_IN_VAULT`/`PHASE_RISING_FROM_GRAVE`/`PHASE_DANCER_RISING`/`PHASE_SNORKEL_INTO_POOL`，类型 `GARGANTUAR`/`REDEYE_GARGANTUAR`/`BUNGEE`/`DIGGER`/`IMP`/`YETI`/`DOLPHIN_RIDER`/`CATAPULT`/`ZAMBONI`，或 `IsBobsledTeamWithSled()`/`IsFlying()`/`mInPool`；② 其余（正常走路的僵尸）→ 掉防具后 `mZombiePhase = PHASE_ZOMBIE_MOWERED; DropLoot();`（**不立即 `mDead`**：`UpdateMowered()` 要等 `REANIM_LAWN_MOWERED_ZOMBIE` 播完一圈（`mLoopCount > 0`）才 `DieWithLoot()`） | src/Lawn/Zombie.cpp:8588-8603（catapult/zamboni）、8605-8637（清单 → `DieWithLoot` 8634）、8664-8670（`PHASE_ZOMBIE_MOWERED`）、4329-4332 + 9312-9321（`UpdateMowered`） |
| 滚动速度 | `aSpeed = 3.33f`（泳池车 2.5f），但 `mChompCounter > 0` 时改为曲线 `PvzpAnimateCurveFloat(50, 0, mChompCounter, aSpeed, 1.0f, CURVE_BOUNCE_SLOW_MIDDLE)` 并被减 1 | src/Lawn/LawnMower.cpp:241-251 |
| 位移 | `mPosX += aSpeed`（仅 TRIGGERED / SQUISHED 会走到这里） | src/Lawn/LawnMower.cpp:236-251 |
| 离场 | `mPosX > WIDE_BOARD_WIDTH` → `Die()`；`WIDE_BOARD_WIDTH = 800` | src/Lawn/LawnMower.cpp:267-270；src/GameConstants.h:28-30 |
| `Die()` | `mDead = true`；若 `mBonusLawnMowersRemaining > 0` 且未掉关卡奖励 → 追加一台 `MOWER_ROLLING_IN` | src/Lawn/LawnMower.cpp:369-380 |
| ROLLING_IN | `mRollingInCounter++`；`mPosX = 缓动(-160 → -21)`；`== 100` 时转 READY（**100 tick，不是 60**） | src/Lawn/LawnMower.cpp:190-199 |
| SQUISHED | `mSquishedCounter--`（`SquishMower()` 置 500），≤ 0 → `Die()` | src/Lawn/LawnMower.cpp:180-188、406-415 |
| `mTriggeredLawnMowers` 复位 | 0（Board 构造） | src/Lawn/Board.cpp:145 |
| `mWaveRowGotLawnMowered` 复位 | 每行 **-100** | src/Lawn/Board.cpp:1379、1289 |
| 就绪集合（等待事件用） | `!mower->mDead && mMowerState == MOWER_READY`，键 `(mRow, mMowerType)` | src/LawnApp.cpp:1554-1556 |
| `mowers_left` | 同上口径的**绝对计数**，终局那一步不刷新 | src/main.cpp:428-430、479（语义已在 cuda/spec/protocol.md §4 指出与 `SimImpl.inc` 现实现不同义） |
| **终局失败谓词** | `if (mX <= aEdgeX && mHasHead) mBoard->ZombiesWon(this);`，`aEdgeX = BOARD_EDGE = -100` | src/Lawn/Zombie.cpp:4537-4569；src/GameConstants.h:32 |
| 分类型边线 | 巨人/红眼/撑杆 **-150**；投手/橄榄/冰车 **-175**；伴舞/舞王/潜水 **-130**；其余 **-100** | src/Lawn/Zombie.cpp:4545-4557 |
| 无头僵尸 | `mX <= aEdgeX + 70 && !mHasHead` → `TakeDamage(1800, 9U)`（不是失败） | src/Lawn/Zombie.cpp:4570-4573 |
| `ZombiesWon` | `mApp->mBoardResult = BOARDRESULT_LOST`，并把 `GetZombieRect().mX < -50` 的其他僵尸清掉 | src/Lawn/Board.cpp:5068-5090 |
| 终局判定 | `EnvironmentTerminal()` 含 `BOARDRESULT_LOST` 与 `SCENE_ZOMBIES_WON` | src/LawnApp.cpp:1591-1595 |
| 击杀计数点 | `mBoard->mZombiesKilled++`（死亡尾部） | src/Lawn/Zombie.cpp:7423 |
| 更新次序 | 车在僵尸**之后**（`mPlants → mZombies → mProjectiles → mCoins → mLawnMowers → 种子槽`） | src/Lawn/Board.cpp:4972-5016 |

### 0.4 种子槽刷新与合法性

| 名称 | 值 | 出处 |
|---|---|---|
| 字段 | `mRefreshCounter`、`mRefreshTime`、`mActive`、**`mRefreshing`**、`mTimesUsed`（**本代码库没有 `mCost`，也没有 `mPacketState`**） | src/Lawn/SeedPacket.h:33-44 |
| 构造 | `mRefreshCounter = 0; mRefreshTime = 0; mActive = true;`（`mRefreshing` 同 false） | src/Lawn/SeedPacket.cpp:47-50 |
| 每 tick 唯一推进点 | `if (!mActive && mRefreshing) { mRefreshCounter++; if (mRefreshCounter > mRefreshTime) { mRefreshCounter = 0; mRefreshing = false; Activate(); FlashIfReady(); } }` | src/Lawn/SeedPacket.cpp:172-182 |
| ⇒ 冷却总长 | **`mRefreshTime + 1` tick**（严格大于） | 同上 |
| 拿起（进光标） | `Deactivate()`：`mActive=false, mRefreshCounter=0, mRefreshTime=0, mRefreshing=false` | src/Lawn/SeedPacket.cpp:140-146、868 |
| 放弃（收回光标） | 按 `mCursorObject->mSeedBankIndex` 重新 `Activate()` | src/Lawn/Board.cpp:2021 |
| 种下 | `WasPlanted()`：`mTimesUsed++; mRefreshing = true; mRefreshTime = Plant::GetRefreshTime(mPacketType, mImitaterType);`（**不在此处清 `mActive`**） | src/Lawn/SeedPacket.cpp:872-896 |
| 开局预置冷却 | `SetPacketType`：非生存模式且 `IsUpgrade` 或 `GetRefreshTime == 5000` → `mRefreshTime = 3500; mRefreshing = true; mActive = false;`；`GetRefreshTime == 3000` → `mRefreshTime = 2000; mRefreshing = true; mActive = false;` | src/Lawn/SeedPacket.cpp:1107-1145 |
| 环境模式会走这条路 | `EnvironmentReset` 对每个槽调 `packet.SetPacketType(deck[i].type, …)` | src/LawnApp.cpp:1411-1419 |
| 成本 | `Plant::GetCost` → `GetPlantDefinition(t).mSeedCost`（Beghouled 例外不影响本关） | src/Lawn/Plant.cpp:4989-5050、48-78 |
| 刷新表 | `Plant::GetRefreshTime` → `GetPlantDefinition(t).mRefreshTime`；僵尸种子返回 **0** | src/Lawn/Plant.cpp:5076-5093 |
| 拾取门控 | `CanPickUp()`：非暂停、`SCENE_PLAYING`、`mPacketType != SEED_NONE`、**`mActive`**、`CanTakeSunMoney(aCost)`、`PlantingRequirementsMet` | src/Lawn/SeedPacket.cpp:686-724 |
| 按下门控（同一套再判一次） | `!mActive` → 蜂鸣返回；`!CanTakeSunMoney(aCost)` → 蜂鸣 + `mOutOfMoneyCounter = 70` 返回 | src/Lawn/SeedPacket.cpp:749-771 |
| 钱的谓词 | `CanTakeSunMoney(n) = n <= mSunMoney + CountSunBeingCollected()`；`CountSunBeingCollected` 只算 `mIsBeingCollected && IsSun()` | src/Lawn/Board.cpp:8514-8517、8470-8484 |
| 扣款 | `TakeSunMoney(theAmount)` 成功则 `mSunMoney -= theAmount`，失败 `PlaySample(BUZZER); mOutOfMoneyCounter = 70` | src/Lawn/Board.cpp:8500-8512 |
| 种植链上的扣款/刷新 | `Board::Plant` 先 `TakeSunMoney(GetCurrentPlantCost(...))`（3793），成功后 `mSeedBank->mSeedPackets[mCursorObject->mSeedBankIndex].WasPlanted()`（3876） | src/Lawn/Board.cpp:3793、3876（执行链全貌见 cuda/spec/protocol.md §1.6） |
| 第 7 关牌组逐张（cost / refresh / 开局是否可用） | 见 §4.4 表 | src/Lawn/Plant.cpp:49-78 |

---

## 1. 天降阳光

### 1.1 调度（`Board::UpdateSunSpawning`，src/Lawn/Board.cpp:5196-5231）

```
UpdateSunSpawning():                                  // 5196
  if StageIsNight() || HasLevelAwardDropped()          // 5197-5198
     || gamemode in {RAINING_SEEDS, ICE, UPSELL, INTRO, ZOMBIQUARIUM,
                     ZEN_GARDEN, TREE_OF_WISDOM, LAST_STAND}
     || IsIZombieLevel() || IsScaryPotterLevel() || IsSquirrelLevel()
     || HasConveyorBeltSeedBank() || tutorial == SLOT_MACHINE_PULL:
        return                          // 早退在递减之前 → 倒计时冻结
  if tutorial in {LEVEL_1_PICK_UP_PEASHOOTER, LEVEL_1_PLANT_PEASHOOTER} and mPlants.size == 0:
        return                                          // 5216-5222（第 7 关 TUTORIAL_OFF，见 LawnApp.cpp:1420）
  mSunCountDown--                                       // 5223
  if mSunCountDown != 0: return                         // 5224-5226
  mNumSunsFallen++                                      // 5227
  mSunCountDown = min(950, 425 + 10*mNumSunsFallen) + Rand(275)   // 5228
  aSunType = (gamemode == SUNNY_DAY) ? COIN_LARGESUN : COIN_SUN   // 5229
  AddCoin(RandRangeInt(100, 649), 60, aSunType, COIN_MOTION_FROM_SKY)  // 5230
```

第 7 关求值：`StageIsNight() == false`（`BACKGROUND_1_DAY`），游戏模式是 `GAMEMODE_ADVENTURE`，
`mTutorialState = TUTORIAL_OFF` ⇒ **每段早退都不成立**，倒计时每 tick 减 1。

**间隔分布**（`n = mNumSunsFallen`，重排在发射之后、用**已加过 1** 的 n）：

```
interval(n) = min(950, 425 + 10n) + U{0..274}      期望 = min(950, 425+10n) + 137
n=1  → 基数 435, 期望 572
n=8  → 基数 505, 期望 642
n=43 → 基数 855, 期望 992
n=53 → 基数 950（封顶）, 期望 1087
```

`HasLevelAwardDropped()` = `mLevelAwardSpawned || mNextSurvivalStageCounter > 0 || mBoardFadeOutCounter >= 0`
（src/Lawn/Board.cpp:5190-5193）—— **过关瞬间起太阳就停了**，续走跨过胜利点时不能继续掉。

### 1.2 开局首发相位

```
InitLevel():                                          // 1338
  mNumSunsFallen = 0                                  // 1385
  if !StageIsNight(): mSunCountDown = RandRangeInt(425, 700)   // 1386-1389
  // 夜晚不写：沿用 Board() 构造里的 mSunCountDown = 0（Board.cpp:116），
  // 但因为 5197 行早退，夜晚永远不会掉太阳。
```

⇒ 第 7 关：第一颗天降阳光落在 **tick ∈ [425, 700]**（`mMainCounter` 从 0 起算，1388 在 1340 的 `mMainCounter = 0` 之后），
以后每颗间隔按 §1.1 的 `interval(n)`。这是"收入窗"里唯一与关卡编号相关的量：**没有关卡表**，只有
`425 + 10n` 这个自增；白天/泳池白天安稳同一套，夜晚整套关掉。

### 1.3 太阳对象本身（`Coin`）

```
CoinInitialize(x=100..649, y=60, COIN_SUN, FROM_SKY)   // Coin.cpp:50-83
   mWidth = mHeight = 60; mDisappearCounter = 0; mFadeCount = 0; mIsBeingCollected = false; mHitGround = false
   IsSun() → 挂 REANIM_SUN（渲染，无逻辑）
CoinInitialize 之后立刻（同一次 AddCoin 调用内）:        // Board.cpp:1971-1976
   mSunMoneyProduced += GetSunValue() == 25             // ← 生产记账点：发射，不是收集
Coin::Update() 每 tick                                  // Coin.cpp:730
   mCoinAge++
   if mFadeCount != 0: UpdateFade()                      // 464-479：--mFadeCount; ==0 → Die()
   elif !mIsBeingCollected: UpdateFall()
   else: UpdateCollected()
UpdateFall()                                            // Coin.cpp:481-603
   if mPosY + mVelY < mGroundY:  mPosY += mVelY          // FROM_SKY: 匀速 0.67，无重力
   else:  mHitGround = true; mPosY = mGroundY
         if !IsLevelAward() && !IsPresentWithAdvice():
             mDisappearCounter++
             if mDisappearCounter >= GetDisappearTime(): StartFade()   // 750 tick
```

落地前需要 `(mGroundY - 60) / 0.67` tick，`mGroundY ∈ [300, 549]` ⇒ **358..731 tick**；
再加落地后的 750 tick 与 15 tick 淡出。所以**一颗太阳从出现到自行消失 ≈ 1123..1500 tick**
（无头环境里永远等不到，见 §1.4）。

`Rand(275)`/`RandRangeInt(100,649)` 都是 `%`，所以 `Rand(275) ∈ [0,274]`、
`RandRangeInt(100,649) = Rand(550)+100 ∈ [100,649]`（src/SexyAppFramework/Common.cpp:113-117；
misc/MTRand.cpp:178-187；src/PvzpLib/PvzpCommon.cpp:384-388）。

### 1.4 无头环境怎么"收集"（rollout 唯一的收集路径）

```
LawnApp::EnvironmentWait(int ticks):                   // LawnApp.cpp:1504-1519
  for i in 0..ticks-1, while !EnvironmentTerminal():
      for coin in mBoard->mCoins:                      // 1510-1514
          if !coin->mDead && coin->IsSun(): coin->ScoreCoin()
      AdvanceLogicTick()                               // 1515 → 一次完整逻辑 tick
```

```
Coin::ScoreCoin()                                      // Coin.cpp:433-441
   Die()                                               //   mDead = true
   if IsSun(): mBoard->AddSunMoney(GetSunValue())      //   mSunMoney += 25; 夹到 9990（Board.cpp:8463-8466）
```

由此得到三条**必须照抄**的结论：

1. **一颗太阳永远只需要 1 tick**：某颗太阳在 tick T 的逻辑里被 `AddCoin` 创建，
   那么它在下一次 `EnvironmentWait` 迭代的**循环头**就被 `ScoreCoin` 掉，
   连一次 `Coin::Update` 都不会走（`UpdateGameObjects` 在 `AdvanceLogicTick` 里面）。
   坐标、下落速度、750 tick 存活期在 rollout 里全部**不可观测**。
2. **`mSunMoneyProduced` 与 `mSunMoney` 的时点不同**：前者在 tick T 的 `AddCoin` 里加（生产），
   后者在 tick T+1 的循环头加（入库）。`EnvironmentPlant`/`EnvironmentWaitUntil` 也都走 `EnvironmentWait(1)`
   这条链（src/LawnApp.cpp:1574-1579），所以任何一次"动作"都恰好推进一轮"先入库、后逻辑"。
3. **收入窗统计生产，不统计收集**：src/main.cpp:435 取 `producedBefore`、473 取 `mSunMoneyProduced` 的差、
   474 以 `now = mMainCounter`（动作之后的值）为键塞进 `history`。因此 CUDA 侧记账必须落在**发射那一 tick**。

---

## 2. 产阳光植物的产出物（只在与天降阳光不同的意义上）

```
Plant::UpdateProduce():                                // Plant.cpp:1026-1072
   mLaunchCounter--
   if mLaunchCounter <= 100: mEatenFlashCountdown = max(..., 曲线)     // 渲染，无逻辑
   if mLaunchCounter <= 0:
       mLaunchCounter = RandRangeInt(mLaunchRate - 150, mLaunchRate)   // 1034 —— 先重抽
       if SEED_SUNSHROOM and STATE_SUNSHROOM_SMALL: AddCoin(COIN_SMALLSUN, FROM_PLANT)   // 1041 → 值 15
       elif SEED_SUNSHROOM:                            AddCoin(COIN_SUN,     FROM_PLANT) // 1045 → 值 25
       elif SEED_SUNFLOWER:                            AddCoin(COIN_SUN,     FROM_PLANT) // 1050 → 值 25
       elif SEED_TWINSUNFLOWER:                        AddCoin(COIN_SUN ×2,  FROM_PLANT) // 1054-1055 → 值 50
       elif SEED_MARIGOLD:                              钱币，不是太阳
```

与天降阳光的差异，**只有这些会影响 rollout**：

| 维度 | 天降阳光 | 植物产出 |
|---|---|---|
| 生产记账时点 | 发射（`AddCoin`，Board.cpp:1976） | **同一个点、同一行代码**（Plant.cpp:1050 走的仍是 `Board::AddCoin`） |
| 值 | 25（`COIN_SUN`） | 25（向日葵/大阳光菇）；**15**（小形态阳光菇，COIN_SMALLSUN）；50（双胞一次发两颗） |
| 一个 tick 内发几颗 | 至多 1 | 双胞向日葵同一 tick 2 颗（1054-1055）→ 需要 2 个 coin 槽位 |
| 时钟 | 全局 `mSunCountDown`，`min(950,425+10n)+U[0,274]` | 每株独立 `mLaunchCounter`，重抽 `[rate-150, rate]` 闭区间；**首发用关卡表的短区间**（见 cuda/spec/plants.md） |
| 是否被"关卡奖励已掉"停掉 | 是（Board.cpp:5197） | 否（植物自己的 `Update` 不受该谓词影响） |
| 是否受 `StageIsNight()` 影响 | 是（夜晚整段不产） | 否 |
| 下落/存活 | 不可观测（1 tick 入库） | 同样不可观测；`FROM_PLANT` 的上抛/重力/放大只在有头模式有意义 |

**结论：不要在 `AddCoin` 之外另开生产账。** 全代码库只有 src/Lawn/Board.cpp:1976 一处写
`mSunMoneyProduced`（grep 确认），任何绕过 `Board::AddCoin` 自己造太阳的写法都会漏计生产。

---

## 3. 草坪修剪机

### 3.1 状态机（`LawnMower::Update`，src/Lawn/LawnMower.cpp:178-273）

```
Update():
  if state == MOWER_SQUISHED:                          // 180-188
      mSquishedCounter--; if <= 0: Die(); return
  if state == MOWER_ROLLING_IN:                        // 190-199
      mRollingInCounter++                              //   注意是 ++，从 0 起
      mPosX = 缓动(0,100, mRollingInCounter, -160, -21, CURVE_EASE_IN_OUT)
      if mRollingInCounter == 100: state = MOWER_READY
      return
  if mGameScene != SCENE_PLAYING && !ShouldRunUpsellBoard(): return    // 201-204
  # ---- 碰撞段：ROLLING_IN / SQUISHED 走不到，其余状态每 tick 都跑 ----
  attackRect = Rect(mPosX, mPosY, 50, 80)              // 206 / 417-420
  for zombie in mBoard->mZombies:                      // 207-234
      if zombie->mDead: continue
      if zombie->mZombieType == ZOMBIE_BOSS: continue
      if zombie->mRow != mRow: continue                // 只有行号相等
      if zombie->mZombiePhase != PHASE_ZOMBIE_MOWERED
         and !zombie->IsTangleKelpTarget()
         and zombie->EffectedByDamage(127U):        // 127 = bit0..6，含 DAMAGES_DYING
          #   ⇒ 8086 的"濒死免疫"对车**不生效**；只有雪橇非领头(8133)与
          #     离场 zombie rect.mX > 800(8154) 会返回 false；腾空相位表(8138-8151)返回 true
          overlap = GetRectOverlap(attackRect, zombie->GetZombieRect())   // 只算 x 轴！
          if overlap > (type == ZOMBIE_BALLOON ? 20 : 0):                 // 225
             if state != MOWER_READY or (type != ZOMBIE_BUNGEE and zombie->mHasHead):  // 228
                 MowZombie(zombie)                     // 143-176
  if state != MOWER_TRIGGERED and state != MOWER_SQUISHED: return   // 236-239（READY 永不动）
  aSpeed = 3.33f ; if type == POOL: aSpeed = 2.5f      // 241-245
  if mChompCounter > 0:                                // 246-250
      mChompCounter--
      aSpeed = PvzpAnimateCurveFloat(50, 0, mChompCounter, aSpeed, 1.0f, CURVE_BOUNCE_SLOW_MIDDLE)
  mPosX += aSpeed                                      // 251
  if type == LAWN and mPlantRow[mRow] == POOL and mPosX > 50: Die()   // 258-266（草地车落水）
  if mPosX > WIDE_BOARD_WIDTH(800): Die()              // 267-270
```

```
MowZombie(z):                                          // 143-176
   if state == MOWER_READY:      StartMower(); mChompCounter = 25
   elif state == MOWER_TRIGGERED:                       mChompCounter = 50
   if type == POOL: z->DieWithLoot()
   else:            z->MowDown()          // 普通草地车：进入 PHASE_ZOMBIE_MOWERED，不是扣血

StartMower():                                          // 382-404
   if state == MOWER_TRIGGERED: return                 // 每台车只会计一次
   mBoard->mWaveRowGotLawnMowered[mRow] = mBoard->mCurrentWave     // 401 —— 写的是**波号**
   mBoard->mTriggeredLawnMowers++                       // 402 —— S_TRIGGERED_MOWERS 的唯一增量点
   mMowerState = MOWER_TRIGGERED                        // 403

Die():                                                 // 369-380
   mDead = true; RemoveReanimation(...)
   if mBoard->mBonusLawnMowersRemaining > 0 and !HasLevelAwardDropped():
       新alloc 一台, LawnMowerInitialize(mRow), state = MOWER_ROLLING_IN, 扣一个 bonus
```

### 3.2 触发谓词（第 7 关，停靠 x = **-21**）

```
mowerRect.x  = [-21, 29]                              // LawnMower.cpp:419 + 停靠 x
zombieRect.x = [mX + 36, mX + 78]                     // Zombie.cpp:201（默认矩形）, 8301-8319, mX=(int)mPosX
GetRectOverlap = min(29, mX+78) - max(-21, mX+36)     // Board.cpp:8969-8973（只看 x）
解 overlap > 0  ⟺  -99 < mX < -7   （即 mX ∈ [-98, -8]，整数）
```

⇒ **一台停在 -21 的普通车，被向左走的普通/路障/旗帜僵尸触发的时刻是 `mX` 第一次 ≤ -8**；
带 `mHasHead` 是 READY 触发的额外条件（LawnMower.cpp:228）。
落地后的撑杆僵尸沿用默认矩形，所以触发点相同，但它的失败线是 -150（4546-4548）。

### 3.3 一条线上杀几个？

**不是"一次一个"。** 碰撞段（207-234）在 TRIGGERED 之后每 tick 都跑，
所以车会沿行**一路碾过去**，直到 `mPosX > 800` 才 `Die()`。
每杀一个新僵尸把 `mChompCounter` 重置成 50，速度被压到最低 1.0 px/tick（246-250），
因此**杀得越多、走得越慢、活得越久、能杀的也越多**（3.33 px/tick 时从 -8 走到 800 需要约 242 tick）。

### 3.4 `S_TRIGGERED_MOWERS` 与等待提前退出

```
事件源：readyDefenses = { (mower->mRow, mower->mMowerType) | !mDead && state == MOWER_READY }
                                                              // LawnApp.cpp:1554-1556
提前退出：defense_lost = HasNew(after.readyDefenses, before.readyDefenses)   // 实参反着传
                                                              // cuda/spec/protocol.md §1.5（199-201 行）
```

⇒ CUDA 侧"某行离开 READY 集合"必须是**集合差**语义（离开的人算，新来的人不算）。
`S_TRIGGERED_MOWERS` 是 `mTriggeredLawnMowers` 的差值口径（`mower_triggered`），
它**不等于** `mowers_left`；后者是 READY 的绝对计数、且终局那一步不刷新
（src/main.cpp:428-430、479；已有条目见 cuda/spec/protocol.md §4 表内 `mowers_left` 行）。
`mTriggeredLawnMowers` 只在 `StartMower()` 加，且 `StartMower()` 对已 TRIGGERED 早退 ⇒ **每台车最多 +1**。

### 3.5 车已用过之后的失败路径

```
Zombie::CheckForBoardEdge()  （Zombie::UpdatePlaying 内调用，Zombie.cpp:4649）
   aEdgeX = -100（普通/路障/旗帜）                       // 4545, GameConstants.h:32
   if mX <= aEdgeX and mHasHead: mBoard->ZombiesWon(this)          // 4559-4568
   if mX <= aEdgeX + 70 and !mHasHead: TakeDamage(1800, 9U)        // 4570-4573

Board::ZombiesWon(z):                                   // Board.cpp:5068-5090
   若已在 SCENE_ZOMBIES_WON 直接 return
   mApp->mBoardResult = BOARDRESULT_LOST                // 5074 ← 终局
   对所有 GetZombieRect().mX < -50 的其他僵尸收尾
   …（5123）mCutScene->StartZombiesWon()
LawnApp::EnvironmentTerminal():  mBoardResult == BOARDRESULT_LOST 即终局   // LawnApp.cpp:1591-1595
```

注意：一台 **READY** 的车挡在 `mX = -8`，失败线在 `mX = -100`，两者相距 92 px。
所以"这一行还有没有车"完全决定这条线会不会输；车用掉之后，同一条线上再来一个僵尸就直接终局
（车此时要么是 TRIGGERED 且已经 `mDead`，要么根本不在表里）。

---

## 4. 种子槽刷新与种植合法性

### 4.1 状态机（每 tick 一次，`SeedPacket::Update`，src/Lawn/SeedPacket.cpp:160-182）

```
Update():
  if mGameScene != SCENE_PLAYING or mPacketType == SEED_NONE: return     // 162-165
  if mBoard->mMainCounter == 0: FlashIfReady()                            // 167-170（纯渲染）
  if !mActive and mRefreshing:                     // ← 两个条件都要，缺一不可  // 172
      mRefreshCounter++                                                    // 174
      if mRefreshCounter > mRefreshTime:            // ← 严格大于           // 175
          mRefreshCounter = 0                       // ← 归零，不是夹到上限 // 177
          mRefreshing = false                                              // 178
          Activate()                    // mActive = true                 // 179
          FlashIfReady()                                                     // 180
```

⇒ 从"种下"到"可再种"实际耗 **`mRefreshTime + 1` tick**（`counter` 从 1 涨到 `refreshTime+1` 才放行）。
⇒ `mRefreshing == false` 的 `!mActive` 槽（刚被拿起、还没落地成植物）**永远不涨 counter**：
`Deactivate()`（140-146）把 `mRefreshTime` 也清 0；只有 `WasPlanted()`（894）才重新写入表值。

### 4.2 一次 PLANT 动作对槽的净效果（`EnvironmentPlant` 的原子语义）

```
EnvironmentPlant(packet, col, row):
   合法性（CanPickUp 全套）  →  TakeSunMoney(GetCurrentPlantCost(type))  →  Board::Plant(...)
   →  mSeedBank->mSeedPackets[index].WasPlanted()          // Board.cpp:3876
WasPlanted():                                               // SeedPacket.cpp:872-896
   mTimesUsed++
   mRefreshing  = true
   mRefreshTime = Plant::GetRefreshTime(mPacketType, mImitaterType)   // 表值，见 4.4
   （mActive 保持 false —— 它在"拿起"那步就被 Deactivate() 清了，SeedPacket.cpp:868/140-146）
   （mRefreshCounter 保持 0，因为 Deactivate() 已归零）
```

于是 CUDA 侧的"一步种植"应等价地写成：`active=0; counter=0; refreshing=1; refreshTime=表值; sun-=cost;`
**不要**用 `mLaunchRate`/`refreshTime` 猜，表值就是 4.4 那一列。

### 4.3 合法动作谓词

```
CanPickUp()                                            // SeedPacket.cpp:686-724
   if mPaused or scene != SCENE_PLAYING or type == SEED_NONE: false
   if IsSlotMachineLevel(): false
   if !mEasyPlantingCheat:                              // 环境模式不开作弊 → 全查
       if !mActive: false                                // 686-709
       aCost = GetCurrentPlantCost(mPacketType, mImitaterType)
       if !CanTakeSunMoney(aCost) and !HasConveyorBeltSeedBank(): false   // 711-715
       if !PlantingRequirementsMet(useType): false       // 717-720
   return true
CanTakeSunMoney(n) = n <= mSunMoney + CountSunBeingCollected()    // Board.cpp:8514-8517
```

`CountSunBeingCollected()` 只累 `mIsBeingCollected` 的太阳（Board.cpp:8470-8484），
无头模式下太阳永远进不了 `mIsBeingCollected`（它被 `ScoreCoin()` 直接吃掉，LawnApp.cpp:1513），
所以对环境 rollout 可以简化成 `cost <= mSunMoney` —— 但**要留着这一项的注释**，
因为它在 CPU 侧真实存在，且"这一步的差值"会影响 `sun_spent` 的口径（cuda/spec/protocol.md §4）。

格子合法性与地形覆盖（睡莲/花盆/墓碑）是 `Board::CanPlantAt`（src/Lawn/Board.cpp:2705+）的职责，
本文不重复；见 cuda/spec/board.md 与 AGENTS.md §8。

### 4.4 第 7 关牌组：cost / refresh / **开局可用性**

表值来自 `gPlantDefs`（src/Lawn/Plant.cpp:48-78）；开局冷却规则来自
`SeedPacket::SetPacketType`（src/Lawn/SeedPacket.cpp:1122-1144，`GAMEMODE_ADVENTURE` 且非生存 → 不早退）。

| 植物 | cost | `mRefreshTime`（表） | 开局 `mRefreshTime` | 开局 `mActive` |
|---|---|---|---|---|
| 豌豆射手 PEASHOOTER | 100 | 750 (Plant.cpp:49) | 0 | **true** |
| 向日葵 SUNFLOWER | 50 | 750 (Plant.cpp:50) | 0 | **true** |
| 坚果 WALLNUT | 50 | 3000 (Plant.cpp:52) | **2000** (1139-1143) | **false** |
| 樱桃炸弹 CHERRYBOMB | 150 | **5000** (Plant.cpp:51) | **3500** (1127-1131) | **false** |
| 土豆 mines POTATOMINE | 25 | 3000 (Plant.cpp:53) | **2000** (1139-1143) | **false** |
| 寒冰射手 SNOWPEA | 175 | 750 (Plant.cpp:54) | 0 | **true** |
| 双发射手 REPEATER | 200 | 750 (Plant.cpp:56) | 0 | **true** |

⇒ tick 0 上合法的牌只有 **豌豆射手 / 向日葵 / 寒冰射手 / 双发射手**（还要求钱够）。
坚果与土豆 mine 要等 2001 tick，樱桃要等 **3501** tick。这条对"早期动作分布"影响很大，
而 CUDA 侧现在完全没建模（`LoadRoot` 只在 `src.refreshTime` 为 0 时回落表值，见 §5）。

---

## 5. `cuda/SimTick.inc` / `cuda/SimImpl.inc` 现在错在哪

（先读原文件；PROVISIONAL 标记出现在 SimTick.inc 的 10、29、32、34、36、46、68、340、465、526、595 行。
其中与本文四个主题相关的是 **46（推车）**、**51-56（天降阳光，其实已经对）**、**32/34/36（不是本文范围）**。）

### 5.1 天降阳光

1. **`S_SUN_PRODUCED` 记错时点**（SimTick.inc:101-113 `BankCoins` vs 115-123 `AddCoin`）。
   引擎在 `Board::AddCoin` **发射时**加 `mSunMoneyProduced`（src/Lawn/Board.cpp:1976），
   `ScoreCoin/AddSunMoney` 只加 `mSunMoney`（src/Lawn/Coin.cpp:440；src/Lawn/Board.cpp:8463）。
   现在 CUDA 在入库那一 tick 同时加 `S_SUN` 和 `S_SUN_PRODUCED`（SimTick.inc:108-109），
   于是 `Rollout.inc:48/71` 的每步生产差值会把太阳归到**晚一个 tick 的那一步**；
   若某条续走在"发射与入库之间"结束（终局那一步不推进 `EnvironmentWait`，src/LawnApp.cpp:1508），
   这颗生产就被整颗丢掉。修法：`AddCoin()` 里 `lane.s(S_SUN_PRODUCED) += value`，
   `BankCoins()` 里只 `lane.s(S_SUN) += value`。
   顺带：`mSunMoneyProduced` 是 `uint32_t`（src/Lawn/Board.h:242），`S_SUN_PRODUCED` 用 int32 会在长程累计上溢出到负数。
2. **间隔多 1 tick**。引擎：`mSunCountDown--; if (!= 0) return;` ⇒ 从设值那天起**恰好 `interval` 个 tick** 后发射
   （src/Lawn/Board.cpp:5223-5228）。CUDA：`if (countdown > 0) { --countdown; return; }`，
   读到 0 之后**还要再等一个 tick**才发射（SimTick.inc:454-455）。
   结果每条续走的天空太阳数量按 `1/(interval+1)` 而非 `1/interval` 慢约 0.1%–0.2%（期望间隔 572–1087）。
   正确写法与 `UpdatePlant` 的 `--launch; if (launch > 0) break;` 不同，需要照 5223-5226 的顺序：
   `if (--countdown != 0) return;`（并保留 `countdown` 可能 ≤ 0 的防御）。
3. **`mSunMoney` 没夹 9990**（SimTick.inc:108 直接 `+=`；引擎 src/Lawn/Board.cpp:8466 有
   `mSunMoney = std::min(mSunMoney, 9990)`）。不影响 `scalars[0]`（那里再夹 2000），
   但影响 `sun` 的叶子统计与 `CanTakeSunMoney` 的可达上界。
4. **`HasLevelAwardDropped()` 的停发条件缺失**（SimTick.inc:452-462 无条件调度）。
   引擎在过关瞬间（`mLevelAwardSpawned` / `mBoardFadeOutCounter >= 0` / `mNextSurvivalStageCounter > 0`，
   src/Lawn/Board.cpp:5190-5198）就不再掉太阳，而 CUDA 的 `CheckVictory` 之后还会继续发。
5. `AddCoin` 的 `value` 参数被写死 25（`cSkySunValue`，SimTick.inc:56、457）。天降阳光在第 7 关确实是 25
   （`COIN_SUN`，src/Lawn/Board.cpp:5229），这条**没错**；但 `SUN_COUNTDOWN*` 三个常量（SimTick.inc:52-55）
   与 425/275/950/+10 完全对得上，**PROVISIONAL 之外的部分可以摘掉标记**。
   另外 `if (S_NUM_SUNS_FALLEN == 0 && countdown == 0 && S_TICK < 100) return;`（SimTick.inc:456）
   是为夜晚造的特例；第 7 关 `mSunCountDown` 恒 ∈ [425,700]（src/Lawn/Board.cpp:1388），
   引擎里根本没有这条判断 —— 夜晚是靠 `StageIsNight()` 早退实现的（5197）。
   留着它会让"root 恰好抓到 countdown==0 且 tick<100"的第 7 关根少掉一颗太阳。

### 5.2 修剪机

6. **触发谓词是编的**。`cMowerTriggerX = 30.0f`（SimTick.inc:49）配上 `zombie.x > cMowerTriggerX → continue`
   （SimTick.inc:402）：引擎没有 x 阈值，是 `GetRectOverlap(mowerAttackRect, zombieRect) > 0`
   的**整数矩形相交**（src/Lawn/LawnMower.cpp:206/224/417-420；src/Lawn/Board.cpp:8969-8973 只算 x 轴；
   src/Lawn/Zombie.cpp:201/8301-8319）。按停靠 x = **-21**（src/Lawn/CutScene.cpp:1261）解出
   **触发带 `mX ∈ [-98, -8]`，向左走时首次命中在 `mX = -8`**。
   现在 `30.0` 会让车**早触发 38 px**（约 127 tick @0.3 px/tick），
   直接改变"哪个 tick 离开 READY"这个等待提前退出事件，以及 `mowers_left` 的叶子值。
   另外 READY 触发还要求 `mHasHead`（LawnMower.cpp:228），气球需要 `overlap > 20`（225）。
7. **`F_ROW_GOT_MOWERED` 写 1，引擎写波号**。SimTick.inc:406 `lane.i(F_ROW_GOT_MOWERED, row) = 1;`；
   引擎是 `mWaveRowGotLawnMowered[mRow] = mBoard->mCurrentWave`（src/Lawn/LawnMower.cpp:401），
   复位值 **-100**（src/Lawn/Board.cpp:1379，exporter 也是 -100：src/CudaExport.cpp:206）。
   消费端算的是 `aWavesMowered = mCurrentWave - mWaveRowGotLawnMowered[aRow]`，然后
   `<=1 → 权重 0.01`、`<=2 → 0.5`、否则 1.0（src/Lawn/Board.cpp:2591-2608，见 cuda/spec/waves.md）。
   写成 1 会让"碾过车之后好几个波"里该行权重永远停在 `currentWave-1`，选行分布一路错到底。
8. **`kMowerRollingIn` 时长 60 应为 100，且终值应是 READY@-21**。
   `cMowerRollInTicks = 60`（SimTick.inc:50）对 SimTick.inc:387-392；引擎
   `mRollingInCounter++ … if (mRollingInCounter == 100) state = MOWER_READY`（src/Lawn/LawnMower.cpp:190-198）。
   第 7 关正常开局没有 ROLLING_IN 的车（`InitLawnMowers` 直接给 READY，
   入场动画是 CutScene 写的 x，不是状态机），所以这条只在 `mBonusLawnMowersRemaining > 0` 时生效
   （src/Lawn/LawnMower.cpp:373-379）。同时 SimTick.inc:391 转 READY 时把 `F_MOWER_X` 留在旧值，
   而引擎的停靠 x 明确是 **-21**。
9. **触发那一 tick 不杀人**。SimTick.inc:395-410：READY → 只把 state 改成 TRIGGERED、计数 +1、`break`，
   杀人分支 `if (state != kMowerTriggered) continue`（411）在同一次调用里被跳过。
   引擎 `MowZombie()` 是 `StartMower(); mChompCounter = 25; … z->MowDown()` —— **触发与首杀同一个 tick**
   （src/Lawn/LawnMower.cpp:145-149、174）。现在实现让触发僵尸多活 1 tick 并继续向左走。
10. **滚动速度不是常数**。SimTick.inc:47 `cMowerSpeed = 3.33f` + 413 `mx += cMowerSpeed`；
    引擎 `mChompCounter > 0` 时速度走 `PvzpAnimateCurveFloat(50,0,chomp, 3.33, 1.0, CURVE_BOUNCE_SLOW_MIDDLE)`
    并且**每次新杀把 chomp 重置为 50**（src/Lawn/LawnMower.cpp:241-251、150-153）。
    一辆正在吃僵尸的车最慢只有 1.0 px/tick，于是"一车杀几个"和"几个 tick 后 `mPosX > 800` 离场"都变了。
    `SimImpl.inc:240-242` 已经把 `mChompCounter` 导成 `F_MOWER_TIMER`/`F_MOWER_CHOMP`，字段够用，
    SimTick.inc:411-431 没读它。
11. **击杀窗口、门控、死法三处都不同**。
    (a) 几何：SimTick.inc:418-419 用 `|zx - mx| > cHitWindow(20)`；
    引擎是 §3.2 的矩形相交（车 `[mx, mx+50]` vs 僵尸 `[mX+36, mX+78]`，只算 x，且 `mX = (int)mPosX` 是**截断成 int 的**）。
    两者既不是同一个中心、也不是同一个宽度；正确写法：
    `overlap = min(mx+50, (int)zx+78) - max(mx, (int)zx+36) > 0`（草地车 `overlap > 0`，气球 `> 20`，LawnMower.cpp:222-225）。
    (b) 过滤条件：`ZombieLive()`（SimTick.inc:94-97）把 `ZB_DYING` 排除掉了，但引擎传的是
    `EffectedByDamage(127U)`，而 `127 = bit0..6` **包含** `DAMAGES_DYING`（bit5，src/ConstEnums.h:257-266），
    于是 src/Lawn/Zombie.cpp:8086-8089 的 `!TestBit(DYING) && IsDeadOrDying()` 早退**永远不成立**：
    一个已经被豌豆打死、正在 `PHASE_ZOMBIE_DYING` 的僵尸照样会被车碾。
    引擎真正会跳过的只有：相位已是 `PHASE_ZOMBIE_MOWERED`（LawnMower.cpp:219）、`IsTangleKelpTarget()`、
    雪橇队非领头（Zombie.cpp:8133-8135）、已离场的 `GetZombieRect().mX > 800`（8154-8157）。
    反过来，`PHASE_POLEVAULTER_IN_VAULT` 这类腾空相位经 8138-8151 返回 `TestBit(127, OFF_GROUND)` = **true**，
    是**可以**被车打中的（`MowDown()` 再把它送去 `DieWithLoot()`，8605-8634）。
    正确写法：CUDA 应把 `ZombieLive()` 换成"`!ZB_DEAD && phase != MOWERED && rect.x <= 800`（+ 雪橇领头）"。
    (c) 死法：SimTick.inc:421-425 无条件 `body/helm/shield = 0; flags |= ZB_DYING; phase_counter = cDeathFadeTicks`，
    对应引擎 `MowDown()` 的**冒烟直接 `DieWithLoot()` 分支**（Zombie.cpp:8605-8637）；
    而"正常走路的僵尸"走的是另一条：`mZombiePhase = PHASE_ZOMBIE_MOWERED; DropLoot();`（8664-8670），
    要等 `UpdateMowered()`（4329-4332 调用，实现 9312-9321，reanim `mAnimRate = 8.0f`，8665）播完一圈才 `DieWithLoot()`。
    也就是说：普通僵尸被碾后**还挂着若干 tick**（不计入 `mZombiesKilled`、但已不再啃食、也不再被车重复计数）。
12. **`state = kMowerSquished; flags = MB_GONE` 用错状态**。SimTick.inc:427-431 在 `mx > 800` 时置
    `MOWER_SQUISHED`；引擎是 `Die()`（`mDead = true`，src/Lawn/LawnMower.cpp:267-270 / 369-371），
    `MOWER_SQUISHED` 是**被巨人/保龄球压扁**那条路（`SquishMower()` 置 500 tick 才 `Die()`，406-415）。
    后果是 `DefenseMask`/`StoreResult` 里的 READY 计数不受影响，但 `F_MOWER_STATE` 与引擎导出的
    `state`（src/CudaExport.cpp:304 直接 `static_cast<int>(mMowerState)`）语义不再一致：
    续走一个 `mDead` 的车在引擎里是 `state=MOWER_TRIGGERED, mDead=1`，CUDA 却读成 SQUISHED。
13. **`S_TRIGGERED_MOWERS` 只加 1 台/车**（SimTick.inc:405 在 READY 分支里 `break`）—— 这条**是对的**，
    与 `StartMower()` 的 TRIGGERED 早退一致（src/Lawn/LawnMower.cpp:384-387、402），可以摘掉 PROVISIONAL。
    但 `SimImpl.inc:302-303` 把 `mowersLeft` 写成 READY 绝对计数、
    而 `out.mowersTriggered` 才是差值（SimImpl.inc:276）：`protocol.md` §4 已点名"`SimImpl.inc:262`
    现在把 `mowersLeft` 写成差值"这一条 —— 现文件里已经修好了，别退回。
    还差的语义是**"终局那一步不刷新 `mowersLeft`"**（src/main.cpp:478-479），
    `StoreResult` 目前无条件按叶子状态算 `mowersReady`。

### 5.3 种子槽

14. **少 1 tick：`counter >= refresh` 应为 `counter > refresh`**。SimTick.inc:444-448；
    引擎 src/Lawn/SeedPacket.cpp:175 是严格大于 ⇒ 冷却实际 `refreshTime + 1` tick。
    一张 750 的牌每周期多放行 1/751 的动作机会。
15. **激活时 `counter` 应归 0，而不是夹在 `refresh`**。SimTick.inc:446（`counter = refresh`）；
    引擎 src/Lawn/SeedPacket.cpp:177（`mRefreshCounter = 0`）。
    今天看不出差别只因为 `SimImpl.inc:395-396` 的 `scalars[59+k]` 在 `active` 时短路成 1.0；
    但**导出的 root 会带这个错值**（`CudaExport.cpp:318` 取的是引擎真值 0），
    续走回来的 `counter` 就与 CPU 存档不同，一旦将来 `active` 短路被去掉就是隐性偏差。
16. **没有 `mRefreshing` 这个量**。SimTick.inc:440 的条件是 `if (F_PACKET_ACTIVE) continue;` ——
    只看了 `mActive`，于是"被拿起但没种下"的槽（引擎 `!mActive && !mRefreshing` 且 `mRefreshTime == 0`，
    src/Lawn/SeedPacket.cpp:140-146/868）也会在 CUDA 里倒计时。
    `CudaRollout.h:311-315` 的字段里没有 refreshing 位（有 `F_MOWER_CHOMP` 那种空位可借，
    或把 `F_PACKET_REFRESH == 0` 当作 `!mRefreshing` 的严格代理 —— 但必须先把 §5.17 的回落去掉）。
17. **`LoadRoot` 的 `refreshTime == 0` 回落表值是错的**。SimImpl.inc:251
    `F_PACKET_REFRESH = src.refreshTime ? src.refreshTime : PlantRefresh(level, src.type)`。
    引擎里 `mRefreshTime == 0` 是**有含义的合法值**：僵尸种子 `GetRefreshTime` 恒 0
    （src/Lawn/Plant.cpp:5078-5080），`Deactivate()` 也置 0（src/Lawn/SeedPacket.cpp:144），
    `SetPacketType` 正常牌也是 0 + `mActive = true`（1111-1114）。
    把 0 当"缺失"会凭空造出一段冷却。
18. **开局预置冷却完全没建模**（见 §4.4）。第 7 关 tick 0 上，坚果/土豆 mine/樱桃在引擎里
    `mActive = false` 且 `mRefreshTime = 2000/2000/3500`（src/Lawn/SeedPacket.cpp:1122-1144，
    经 src/LawnApp.cpp:1418 的 `SetPacketType` 生效）。
    根状态如果是引擎导出的，`src.active`/`src.refreshTime` 带着真值，`LoadRoot` 能接上；
    但**任何"CUDA 自己从空板起一局"的路径（冒烟测试、单测根）会把这三张牌当成一开始就能种**，
    与 `AnyLegalPlant`（SimTick.inc:640-654）和 `scalars[5+k]`（SimImpl.inc:394）一起错。
    `SimImpl.inc:19-39` 的注释把回落解释成"老 exporter 可能写 0"，那条理由不成立（见 §5.17）。
19. `ActionPlant` 的扣款门（SimTick.inc:676-678）用的是 `lane.i(F_PACKET_COST, packet)`，
    引擎用的是 `GetCurrentPlantCost(mPacketType, mImitaterType)`（src/Lawn/SeedPacket.cpp:711、761；
    扣款在 Board.cpp:3793）。`CudaExport.cpp:320-321` 导出的正是 `Plant::GetCost(...)`，第 7 关没有折扣场景，
    两者现在相等 —— 但这是"碰巧"，不是同一条谓词；`mEasyPlantingCheat` 与
    `HasConveyorBeltSeedBank()` 两个短路（SeedPacket.cpp:704/712）也没抄。
20. `AnyLegalPlant`（SimTick.inc:640-654）与 `CanPlantAt`（658-663）只看空格与行号，
    而引擎在 `!mActive` 之后还查 `PlantingRequirementsMet`（SeedPacket.cpp:717-720）。
    第 7 关牌组里没有升级植物 ⇒ 目前不触发；换卡组（AGENTS.md §8 的睡莲/香蒲、花盆线）时这条会变成硬错。
