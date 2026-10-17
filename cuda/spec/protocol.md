# CUDA 机制规格：ROLLOUT 决策协议与特征

只描述**已有 CPU 路径**的确切行为，供 `cuda/` 的批量续走逐位复现。所有结论都来自代码，行号按工作区文件。
基线：`git HEAD = 12a8e83`，工作区另有**未提交**改动 `src/CudaExport.cpp`（波表导出）与 `cuda/CudaRollout.h`（`WaveTable`、SoA 字段增删）。
凡代码里没有的概念，写 `NOT FOUND`。伪代码里的比较运算符、除数、clamp 边界照抄源码，不做"等价改写"。

固定常量：`kEnvironmentProtocolVersion = 4`（src/main.cpp:52）、就绪 banner 里的 `wait_events_version = 1`（src/EnvironmentWaitEvents.h:15）与 `rollout_version:1`（src/main.cpp:331-332）、
`MAX_GRID_SIZE_X = 9`、`MAX_GRID_SIZE_Y = 6`、`MAX_ZOMBIES_IN_WAVE = 50`、`MAX_ZOMBIE_WAVES = 100`（src/Lawn/Board.h:42-45）、`SEEDBANK_MAX = 10`（src/GameConstants.h:39）、
`NUM_SEED_TYPES = 53`（`SEED_PEASHOOTER = 0 … SEED_LEFTPEATER = 52`，src/ConstEnums.h:1035-1090）、`SEED_NONE = -1`（src/ConstEnums.h:1113）、`NUM_ZOMBIE_TYPES = 34`（src/ConstEnums.h:1340-1376）、
`MOWER_ROLLING_IN=0 / MOWER_READY=1 / MOWER_TRIGGERED=2 / MOWER_SQUISHED=3`（src/ConstEnums.h:571-574）、`GRIDSQUARE_GRASS=1 / GRIDSQUARE_DIRT=2 / GRIDSQUARE_POOL=3`（src/ConstEnums.h:551-553）。

---

## 1. ROLLOUT 主循环，逐语句（src/main.cpp:390-498）

### 1.1 命令文法与准入校验

```
ROLLOUT <rootId> <seed> <untilTick> <mask> <maxActions> <eventWait> <first> \
        <historyStart> <historyCount> [<tick> <produced>] x historyCount [<actionArg>...]
```
`>>` 顺序 = src/main.cpp:397；Python 侧构造见 python/pvz_env.py:676-678（`seed` 缺省写 `-1`，`first` 缺省写 `"A"`）。

`ok` 的合取式（src/main.cpp:397-400），任何一条不成立则整个循环被跳过：

```
input >> rootId >> seed >> untilTick >> mask >> maxActions >> eventWait >> first
       >> historyStart >> historyCount    (读成功)
&& rollout.Loaded()                       // 已 LOAD_ROLLOUT_NET
&& app->mBoard
&& mask >= 0 && !(mask & ~ENV_SKIP_ALLOWED)          // ENV_SKIP_ALLOWED == 1|4|8|16 == 29
&& maxActions > 0 && maxActions <= 1000000
&& (eventWait == 0 || eventWait == 1)
&& historyCount >= 0 && historyCount <= 100000
&& rootId >= 0 && (rootId == 0 || snapshots.contains(rootId))
&& seed >= -1 && seed <= UINT32_MAX
```

副作用严格按此顺序，每步都可能把 `ok` 打回 false 并短路后续（src/main.cpp:401-416）：

1. `if (ok && rootId) ok = RestoreEnvironmentSnapshot(app, snapshots.at(rootId));`（401）。
   恢复内容 = board 存档字节 + `GetRandState()` + `mAppRandSeed/mRandSeed/mAppCounter` + `mZombiesKilled/mPlantsEaten/mSunMoneyProduced/mTriggeredLawnMowers` + `mGameScene/mBoardResult`（163-178、246-261）。
2. `if (ok && seed >= 0) ReseedEnvironment(app, (uint32_t)seed);`（402）—— 见 §5。
3. `if (ok) before = ReadCounters(app);`（405）—— **死写**：ROLLOUT 分支在 497 行 `continue`，从不进入带 `events` 的通用尾部（908-919），`before` 在本分支内不再被读。403-404 的注释描述的是 BRANCH/通用路径的语义。
4. 历史读入（406-413），逐条校验（用的是**恢复之后**的 `mMainCounter`）：
   ```
   tick >= historyStart
   && tick <= app->mBoard->mMainCounter
   && (history.empty() || tick >= history.back().first)      // 非递减
   ```
5. `ok &&= !(input >> extra) && historyStart >= 0 && historyStart <= mMainCounter
        && (first == "A" || DecodeBranchAction(first, actionCommand, actionArguments))`（415-416）。
   `DecodeBranchAction`（310-322）：要求 `spec.size() >= 3 && spec[1] == ':'`，`spec[0]` ∈ {`P`,`S`,`W`} → `PLANT`/`SHOVEL`/`WAIT`，`arguments = spec.substr(2)` 后把所有 `:` 替换为空格。
   `PLANT` 的参数顺序是 `packet col row`（286-290）；`SHOVEL` 是 `col row`；`WAIT` 是 `ticks`。

### 1.2 进入循环前的初始化（src/main.cpp:417-431）

```
decisions = 0; mowersLeft = 0; startTick = 0;                       // 417
std::array<int,10> planted{};                                        // 418   索引 = 卡槽号
simulationSeconds = 0; networkSeconds = 0;                           // 419
if (ok) {
    savedMask = gEnvSimSkip; gEnvSimSkip = mask;                     // 422-423   482 行复原
    startTick = app->mBoard->mMainCounter;                           // 424
    if (untilTick <= app->mBoard->mMainCounter)
        untilTick = std::numeric_limits<int>::max();                 // 425   「until 0 = 打到通关」
    priorTick = -1;                                                  // 426
    priorPlant = false;                                              // 427
    readyMowers = [&]{ int c=0; for (LawnMower* m : mBoard->mLawnMowers)
                            if (!m->mDead && m->mMowerState == MOWER_READY) ++c; return c; };  // 428-429
    mowersLeft = readyMowers();                                      // 430
    rollout.ResetNetworkCalls();                                     // 431
```

`ok == false` 时 `startTick` 保持 0、`ResetNetworkCalls()` 不被调用 → 响应的 `ticks` 变成**绝对时钟**、`net_calls` 变成**历次累计**（见 §4）。CUDA 侧不要把这条当噪声。

### 1.3 循环体（src/main.cpp:432-481），顺序即语义

```
while (!app->EnvironmentTerminal() && decisions < maxActions)        // 432
{
    const int      tick           = app->mBoard->mMainCounter;       // 434  本步"观测时钟"
    const uint32_t producedBefore = app->mBoard->mSunMoneyProduced;   // 435
    EnvironmentRollout::Action action;                               // 436  默认 {-1,0,0}
    bool isFirst = (decisions == 0 && first != "A");                  // 437  注入动作只作用一次

    if (!isFirst)                                                   // 438-447  学生前向
    {
        start = steady_clock::now();                                 // 440
        while (!history.empty() && history.front().first < tick - 6000)
            history.pop_front();                                     // 441  严格 <；用动作前的 tick
        double produced = 0;
        for (auto entry : history) produced += entry.second;         // 442-443  uint32→double 逐项累加
        const int elapsed = std::max(0, std::min(6000, tick - historyStart));   // 444  分母是 historyStart，不是 front()
        action = rollout.Predict(app,
                    elapsed ? produced * 1000.0 / elapsed : 0,       // 445  income
                    priorTick, priorPlant).action;                   // 445  diagnostics = false（默认）
        networkSeconds += now() - start;                             // 446
    }

    start = steady_clock::now();                                     // 448
    bool accepted = true;                                            // 449
    if (isFirst)                                                     // 450-462  执行注入动作
    {
        std::istringstream arguments(actionArguments);
        if (actionCommand == "WAIT" && eventWait) {
            int ticks = -1;
            accepted = (arguments >> ticks) && ticks >= 0 && ticks <= 1000000;
            if (accepted) app->EnvironmentWaitUntil(ticks, 0);       // 457  事件等待（condition 0 = timeout）
        } else {
            accepted = ExecuteEnvironmentAction(app, actionCommand, arguments);   // 459  283-308
        }
        priorPlant = (actionCommand == "PLANT" && accepted);         // 460
        if (priorPlant) { arguments.clear(); arguments.str(actionArguments);
                          arguments >> action.packet; }              // 461  仅为了给 470 行的 planted[] 取卡槽号
    }
    else                                                             // 463-468  执行学生动作
    {
        priorPlant = (action.packet >= 0);                            // 465
        if (priorPlant) accepted = app->EnvironmentPlant(action.packet, action.col, action.row);  // 466
        else            app->EnvironmentWaitUntil(60, 0);             // 467  注意：不是 EnvironmentWait(60)
    }

    if (!accepted) { app->EnvironmentWait(60); priorPlant = false; }  // 469  拒绝路径：固定 60 tick
    if (priorPlant) ++planted[action.packet];                         // 470  卡槽号 0..9
    priorTick = tick;                                                 // 471  记的是动作"前"的时钟
    const int      now      = app->mBoard->mMainCounter;              // 472
    const uint32_t produced = app->mBoard->mSunMoneyProduced - producedBefore;  // 473  uint32 差
    if (produced) history.emplace_back(now, produced);                // 474  只入非零
    substeps.emplace_back(now, produced);                              // 475  每步都入（含 0）
    simulationSeconds += now() - start;                                // 476
    ++decisions;                                                       // 477
    if (!app->EnvironmentTerminal()) mowersLeft = readyMowers();        // 479  终局那一步不更新
    if (now >= untilTick) break;                                        // 480
}
gEnvSimSkip = savedMask;                                               // 482
```

必须复现的九个要点：

1. **停止条件优先级**：`terminal` 与 `decisions < maxActions` 只在循环**入口**判；`untilTick` 在动作执行**之后**判（480），因此越过 horizon 的那一步的 tick、`produced`、`decisions`、`planted` 全部计入。同一步既终局又越界时，479 先执行 → `mowersLeft` 保留**最后一个非终局观测**的数量（478 行的中文注释就是这个意思），这正是"终局观测保留最后非终局小推车计数"的行为。
2. **`history` 与 `substeps` 长度可以不同**（474 vs 475）。Python 恢复历史时同样按 `produced > 0` 过滤（python/pvz_env.py:684-686）。
3. **pop 用动作前的 `tick`，push 用动作后的 `now`**，二者相差整步推进的 tick。
4. **`elapsed` 的分母**是命令传入的 `historyStart`（444），与 Python 的 `_sun_history_start_tick` 同义（python/pvz_env.py:251）。窗口过滤是 `< tick - 6000` 丢弃、其余**全量求和**，不再按 `>= cutoff` 二次筛（与 python/pvz_env.py:249-253 等价，因为那边筛的正是刚 pop 的判据）。CUDA 侧 `cuda/SimCore.h:191-200` 的 `IncomeRate` 已按同一式子实现（`>= tick - 6000` + `ClampInt(tick - windowStart, 0, 6000)`）。
5. **两条 WAIT 路径不等价**：467 的 `EnvironmentWaitUntil(60, 0)` 可提前退出（见 1.5），469 的 `EnvironmentWait(60)` 只在终局提前退出。`ROLLOUT_PREDICT` 回报给 Python 的动作语义却统一写成 `{"type":"wait","ticks":60,"until":"timeout"}`（383）——上报口径与实际执行**不同**，CUDA 若按上报口径实现会偏。
6. **被拒绝的动作仍然推进 60 tick**（469），且不进 `planted[]`、`priorPlant` 被抹成 `false`（→ 下一步 `scalars[113]=0`、`scalars[114]` 用被拒绝那一步的 `priorTick`）。`ExecuteEnvironmentAction` 对越界的 `WAIT ticks` 返回 false（302-303），于是"请求 200 万 tick 的 WAIT"实际只推进 60 tick。
7. **`planted[]` 计的是"请求被接受"**，不是"场上多了一株植物"：`EnvironmentPlant` 在调用 `MouseDownWithPlant` 之后无条件 `return true`（LawnApp.cpp:1485-1486），而后者是 `void` 且内部还会因 `TakeSunMoney` 失败等提前 `return`（Board.cpp:3791-3796）。
8. **`isFirst` 时不调用网络**（438），所以 `net_calls` 至多 `decisions`（注入那一步不计），且 `first == "A"` 时首步也计。
9. **`SHOVEL` 也可以当注入动作**（311-317），此时 `priorPlant = false`，同样占一个 `decision`。

### 1.4 一步 tick 的确切推进（src/LawnApp.cpp）

```
EnvironmentWait(ticks)                                    // 1504-1519
    if (!mBoard || ticks < 0 || ticks > 1000000) return;   // 1506-1507
    for (i = 0; i < ticks && !EnvironmentTerminal(); ++i) {
        for (Coin* coin : mBoard->mCoins)
            if (!coin->mDead && coin->IsSun()) coin->ScoreCoin();   // 1510-1514  自动收阳光
        AdvanceLogicTick();                                            // 1515
        if (!mHeadlessMode) EnvironmentPresent();                      // 1516-1517  headless 下为空
    }

AdvanceLogicTick()                                        // 1581-1589
    ++mAppCounter; mBoard->ProcessDeleteQueue();
    mEffectSystem->ProcessDeleteQueue(); SexyApp::UpdateFrames();

EnvironmentTerminal()                                     // 1591-1595
    return !mBoard || mBoard->mLevelComplete || mBoardResult == BOARDRESULT_WON
        || mBoardResult == BOARDRESULT_LOST || mGameScene == SCENE_ZOMBIES_WON;
```

- `mSunMoneyProduced` 在**硬币生成时**自增（`Board::AddCoin`：`if (aCoin->IsSun()) mSunMoneyProduced += aCoin->GetSunValue();`，src/Lawn/Board.cpp:1974-1976），不是收集时。所以 income 窗口量的是"产出"而非"入账"。
- 硬币在第 N tick 生成、在第 N+1 tick 的循环**开头**被 `ScoreCoin()` 记账 → 这就是 `CudaExport.cpp:262-269` 把 `ticksToCollect = 1` 写死的依据。
- `Board::mMainCounter++` 位于 `Board::Update()` 的 `if (mApp->mGameScene != SCENE_PLAYING && !mCutScene->ShouldRunUpsellBoard()) return;` 之后（src/Lawn/Board.cpp:5681-5683）→ 时钟**可能停住**，这就是 `Result.stalledClockSteps`（src/EnvironmentWaitEvents.h:39、99）。CUDA 侧不能假定"1 次 advance == 1 tick"。

### 1.5 `EnvironmentWaitUntil(60, 0)`：可以提前结束的等待

`EnvironmentWaitUntil(ticks, condition) = EnvironmentWaitEvents::Run(ticks, condition, EnvironmentPublicWaitState, []{ EnvironmentWait(1); })`（src/LawnApp.cpp:1574-1579）。`Run`（src/EnvironmentWaitEvents.h:80-106）：

```
入口先算一次：triggered = Triggered(initial, initial, condition, requestedTicks == 0)   // 90
while (triggered.empty() && logicSteps < requestedTicks) {
    advance();                                                      // = EnvironmentWait(1)
    ++logicSteps;
    current = readPublic();
    if (current.tick < previous.tick) throw;                         // 97-98
    stalledClockSteps += (current.tick == previous.tick);            // 99
    actualTicks = current.tick - initial.tick;                       // 100
    triggered = Triggered(previous, current, condition, logicSteps == requestedTicks);  // 101
    previous = move(current);
}
reason = triggered.front();                                          // 104（非空保证：初值或 deadline）
```

`Triggered`（src/EnvironmentWaitEvents.h:62-78）里能结束 condition 0 等待的只有三条：

| reason | 谓词 | 解读 |
|---|---|---|
| `terminal` | `after.terminal` | `EnvironmentTerminal()` 为真 |
| `defense_lost` | `HasNew(after.readyDefenses, before.readyDefenses)` | 注意**实参是反着传的**：`before` 有、`after` 没有的 `(mower->mRow, mower->mMowerType)`，即某台推车离开 READY 集 |
| `zombie_entered_left_zone` | `HasNew(before.leftZoneZombies, after.leftZoneZombies)` | `after` 新出现的 zombieID（`Board::ZombieGetID`），且 `IsOnBoard()` 且渲染坐标 `x <= 160.0`（`LeftZoneMaxX`，src/EnvironmentWaitEvents.h:16） |
| `condition` | condition ∈ 1..5 的选中判定（70-74） | **condition == 0 永远不为真** |
| `max_ticks` | `deadline`（`logicSteps == requestedTicks`，或入口的 `requestedTicks == 0`） | |

同一 tick 多个原因同时成立时，`result` 的**构造顺序**就是优先级（77 行注释），且 `reason` 取 `front()`。
左区判定读的是**渲染后**坐标：`std::abs(mPosX - 160.0) < 1.0 ? RenderedPublicCoordinate(mPosX) : (double)mPosX`（src/LawnApp.cpp:1566-1569）；live 僵尸 ID 非法或 `mPosX` 非有限会 `throw std::runtime_error`（1562-1563）。入口若已终局，**一步都不推进**。

`readyPackets`/`readyDefenses`/`leftZoneZombies` 都是 `std::set`，键分别是 `(index, mPacketType, mImitaterType)`、`(mRow, mMowerType)`、`zombieID`（src/LawnApp.cpp:1548-1570；`readyPackets` 额外要求 `mActive && mPacketType != SEED_NONE`，**不**要求 `CanPickUp`）。

### 1.6 PLANT 的实际执行链（CPU 侧必须被"逻辑化"复现的部分）

```
LawnApp::EnvironmentPlant(packetIndex, col, row)                    // LawnApp.cpp:1466-1487
    if (!mBoard || mGameScene != SCENE_PLAYING || col < 0 || col >= 9 || row < 0 || row >= 6
        || packetIndex < 0 || packetIndex >= mSeedBank->mNumPackets) return false;   // 1468-1470
    if (!packet.CanPickUp()) return false;                                           // 1473-1474
    seed = (mPacketType == SEED_IMITATER && mImitaterType != SEED_NONE) ? mImitaterType : mPacketType;  // 1475
    if (mBoard->CanPlantAt(col, row, seed) != PLANTING_OK) return false;             // 1476-1477
    packet.Deactivate();                          // mActive=false, mRefreshCounter=0, mRefreshTime=0, mRefreshing=false  (SeedPacket.cpp:141-147)
    cursor = { CURSOR_TYPE_PLANT_FROM_BANK, mPacketType, mImitaterType, packetIndex }; // 1480-1484
    mBoard->MouseDownWithPlant(GridToPixelX(col,row) + 40, GridToPixelY(col,row) + 40, 1);  // 1485
    return true;                                                                     // 1486
```

`MouseDownWithPlant` 再把像素换算回格（Board.cpp:8768-8771、8812-8828、8836-8868）：

```
GridToPixelX = col*80 + LAWN_XMIN(40)                       // 8890
PlantingPixelToGridX = PixelToGridX(x+?) = std::clamp((x - 40)/80, 0, 8)    // 8824-8827
   → (col*80 + 40 + 40 - 40)/80 == col                       // 整数除法，恒等
GridToPixelY：草地 row*100 + 80；泳池 row*85 + 80；屋顶 row*85 + (col<5 ? (5-col)*20 : 0) + 80 - 10
   另有 if (mGridSquareType[col][row] == GRIDSQUARE_HIGH_GROUND) aY -= HIGH_GROUND_HEIGHT;   // 8946-8948
PixelToGridY：先 OffsetYForPlanting（飞行/墓碑 +15，spikeweed/spikerock -15，8752-8766），再
   非屋顶非泳池 std::clamp((y - 80)/100, 0, MAX_GRID_SIZE_Y-2)   // 上界 4！
   泳池         std::clamp((y - 80)/85, 0, MAX_GRID_SIZE_Y-1)    // 上界 5
   屋顶         if (gridX < 5) y -= (4-gridX)*20; std::clamp((y-80)/85, 0, MAX_GRID_SIZE_Y-2)  // 上界 4
```

结论（CUDA 可以省掉像素往返，但必须把这三条后果一并实现）：
1. x 方向恒等；y 方向在草地/夜/池/雾上 row 0..4 恒等（`+40` 居中，`±15` 不跨行：`(40+15)<85<100`、`(40-15)>=0`）。
2. **草地与屋顶上 row 5 被 clamp 成 row 4**。`EnvironmentPlant` 的 `CanPlantAt(col,5,...)` 是按 row 5 判的，落地却按 row 4 → 二者可能不一致（草地 row 5 是 `GRIDSQUARE_DIRT`，通常已被 `CanPlantAt` 拒掉；屋顶 row 5 可种，则真会错位）。
3. `GRIDSQUARE_HIGH_GROUND`（仅由 `mPlantRow == PLANTROW_HIGH_GROUND && col ∈ 4..8` 产生，Board.cpp:1076-1078）会让 `GridToPixelY` 少掉 `HIGH_GROUND_HEIGHT`，换算回格时可能差一行。冒险 41-49/50 屋顶关需实测确认。
4. 扣阳光发生在 `MouseDownWithPlant` 内部：`if (!mEasyPlantingCheat && cursor == PLANT_FROM_BANK && !HasConveyorBeltSeedBank()) if (!TakeSunMoney(GetCurrentPlantCost(aPlantingSeedType, SEED_NONE))) return;`（Board.cpp:3791-3796、8500-8508、9406-9414）。注意传入的 imitater 参数是 `SEED_NONE`，而 `CanPickUp` 用的是 `GetCurrentPlantCost(mPacketType, mImitaterType)`（SeedPacket.cpp:708）；`PlantUsesAcceleratedPricing` 的类型每种额外 +50×已有株数。
5. 冷却在 `Board::AddPlant` 末尾武装：`mSeedBank->mSeedPackets[cursor.mSeedBankIndex].WasPlanted()`（Board.cpp:3876）→ `mTimesUsed++; mRefreshing = true; mRefreshTime = Plant::GetRefreshTime(mPacketType, mImitaterType);`（SeedPacket.cpp:888-893）。顺序是"先 `Deactivate()`（把 `mRefreshTime` 清 0）后 `WasPlanted()`（重设 `mRefreshTime`）"，`mRefreshCounter` 保持 0。
6. `SeedPacket::Update` 的推进式：`if (!mActive && mRefreshing) { ++mRefreshCounter; if (mRefreshCounter > mRefreshTime) { mRefreshCounter = 0; mRefreshing = false; Activate(); } }`（SeedPacket.cpp:172-182）——**严格 `>`**。

---

## 2. `EnvironmentRollout::Predict` 的决策语义（src/EnvironmentRollout.cpp:267-297）

字面转写（保留运算符与边界）：

```
Prediction result{};                              // 237/269  plant=0.0f, packets[10]=0, cells[54]=0,
                                                //          cellPacket=-1, action={packet=-1,col=0,row=0}
if (!diagnostics && !app->EnvironmentAnyLegalPlant()) return result;   // 270
++networkCalls;                                   // 271
Features(app, income, previousTick, previousPlant, board, scalars);   // 272-273
Encode(board, scalars);                           // 274
result = Heads(-1);                               // 275
if (result.plant <= 0 && !diagnostics) return result;                 // 276  「<= 0」，logit 域，无 sigmoid
float bestPacket = -std::numeric_limits<float>::infinity();           // 277
std::vector<Action> legal;                        // 278
for (int index = 0; index < app->mBoard->mSeedBank->mNumPackets; ++index) {   // 279
    SeedPacket& packet = app->mBoard->mSeedBank->mSeedPackets[index];
    if (!packet.CanPickUp() || result.packets[index] <= bestPacket) continue;  // 282
    const SeedType seed = packet.mPacketType == SeedType::SEED_IMITATER
                            ? packet.mImitaterType : packet.mPacketType;        // 283
    std::vector<Action> cells;
    for (int row = 0; row < 6; ++row)
        for (int col = 0; col < 9; ++col)
            if (app->mBoard->CanPlantAt(col, row, seed) == PlantingReason::PLANTING_OK)
                cells.push_back({index, col, row});                             // 285-288
    if (!cells.empty()) { bestPacket = result.packets[index]; legal = std::move(cells); }  // 289
}
if (legal.empty()) return result;                 // 291
result = Heads(legal[0].packet);                  // 292
if (result.plant > 0)                             // 293
    result.action = *std::max_element(legal.begin(), legal.end(),
        [&result](const Action& left, const Action& right) {
            return result.cells[left.row * 9 + left.col] < result.cells[right.row * 9 + right.col]; });  // 294-295
return result;
```

逐条对齐说明：

1. **`EnvironmentAnyLegalPlant` 短路真的跳过整张网络。** 它自己就是"卡 × 6×9 格"的 `CanPickUp` + `CanPlantAt` 扫描，第一个合法即 return true（src/LawnApp.cpp:1706-1721），谓词与 `EnvironmentLeanObservation` 的 `legal_actions.plants` 完全一致（1684-1700）。返回 false 时 Predict 在 `++networkCalls` **之前**退出（270 vs 271）→ `net_calls <= decisions`；返回的 `Prediction` 里 `plant == 0.0f`、`cellPacket == -1`、`action.packet == -1`，主循环据此走 467 行的 `EnvironmentWaitUntil(60,0)`。CUDA 必须同样"先判合法、再决定要不要跑前向"，否则 `net_calls` 对不上。
2. **两处 imitater 解析不一致**（必须逐处照抄，或明确声明偏差）：
   | 位置 | 表达式 |
   |---|---|
   | `Predict`（283） | `mPacketType == SEED_IMITATER ? mImitaterType : mPacketType` |
   | `EnvironmentAnyLegalPlant`（1714） | `mPacketType == SEED_IMITATER && mImitaterType != SEED_NONE ? mImitaterType : mPacketType` |
   | `EnvironmentPlant`（1475） | 同上（带 `!= SEED_NONE`） |
   | `SeedPacket::CanPickUp`（693-697） | 同上（带 `!= SEED_NONE`） |
   | `EnvironmentLeanObservation`（1690） | 同上（带 `!= SEED_NONE`） |
   `RESET_V2` 允许 `imitater` 槽位写 `SEED_NONE`（`ParseDeck`，src/main.cpp:112-122）。真出现这种槽位时，`Predict` 会拿 `SEED_NONE` 去扫格（`CanPlantAt` 必然不 OK → 该卡被跳过），而合法动作表里用的是 `SEED_IMITATER` → 两处的合法集不同。
3. **平票取卡槽号小者**：跳过条件是 `result.packets[index] <= bestPacket`（282），即只有**严格大于**才可能胜出；`index` 从 0 递增 → 相同 logit 时先出现的槽位赢。
4. **`bestPacket` 只在"该卡至少有一个合法格"时更新**（289），且 `legal` 是**整批替换**而非累加。后果：一个分数更高但无处可种的卡既不占 `bestPacket`、也不进 `legal`，后面分数更低的卡照样能赢。
5. **格序是行主序**：`row` 外层 0..5、`col` 内层 0..8（285-286），所以 `legal` 的插入序 = `(0,0),(0,1)…(0,8),(1,0)…(5,8)`。`std::max_element` 返回**第一个**最大值（293-295 的比较器只在严格小于时换）→ 格 logit 平票时取 `(row, col)` 字典序最小者。
6. **第二次 `Heads` 会重算 `plant`**：293 行的 `result.plant > 0` 是新的值。若第二次算出 `plant <= 0`，`action` 保持默认 `{packet=-1,…}`（`Heads` 只在 `plant > 0` 时才写 `result.action`）→ 主循环 465 行 `priorPlant = false`，这一步变成 60-tick 事件等待，`planted[]` 不加。所以"选了卡但没种"是合法轨迹。
7. **两套 cell 索引并存，别混**：
   - 特征网格是 `Cells = 60` = 6 行 × **10** 列（`EnvironmentRollout.h:13`），列 9 = "草坪外"；`Heads` 累加时用 `scratch[channel*Cells + cell/9*10 + cell%9]`（263）。
   - `Prediction::cells` 是 **54** 个逻辑格（`std::array<float,54>`），索引式 `cells[row*9 + col]`（295、`ROLLOUT_PREDICT` 打印 54 个：src/main.cpp:381）。
   - 转换：`padded = logical/9*10 + logical%9`；反向 `logical = padded/10*9 + padded%10`（仅对 `padded%10 < 9` 成立）。列 9 的值**永远不参与动作选择**。
   - `Heads` 的 cells 基线不是 0，而是 `result.cells.fill(weights[16][0])`（244），随后 `+= max(0.0f, scratch[…]) * weights[15][channel]`（263）；`packet < 0` 时 `Heads` 在 243 行提前 return，cells 保持 `Prediction{}` 的 0。
8. **`diagnostics` 只改变两处早退**（270、276），不改变动作选择逻辑；调用者只有 `ROLLOUT_PREDICT`（src/main.cpp:377）。
9. `Predict` 对 board 只读（`SeedPacket&` 是非 const 引用，但只调 `CanPickUp()`；`CanPlantAt` 也视为只读，但内部会走 `mChallenge->CanPlantAt`，src/Lawn/Board.cpp:2712-2716）。CUDA 侧可当作纯函数。
10. `CanPlantAt(col, row, seed)` 的合法集是深函数（边界检查、`Challenge::CanPlantAt`、花盆/睡莲/grave/crater/scary pot/ice/屋顶线等，src/Lawn/Board.cpp:2705-2900+），谓词只有一个：`== PlantingReason::PLANTING_OK`。CUDA 核心必须逐条覆盖，另见 §7 的支持性判定。

---

## 3. `Features()`：学生网络读到的每一个输入（src/EnvironmentRollout.cpp:106-167）

`EnvironmentRollout.h:13`：`Channels = 92, Cells = 60, Scalars = 115, ZombieTypes = 34, TaskContext = 1 + ZombieTypes = 35`。
`board.assign(Channels*Cells, 0.0f)`（110，5520 个 float）、`scalars.assign(Scalars + (taskContext ? TaskContext : 0), 0.0f)`（111）。`taskContext` / `meanmax` / `width` 来自网络文件头（`PVZRN002`/`PVZRN003` + size + pool + withContext，106 行的 `Load`：68-104）。**`taskContext == false` 时 3.3 节那 35 个标量根本不存在**（数组只有 115 个），CUDA 必须按导出文件的头字段分支。

### 3.1 92 个 board 通道（cell 索引一律 `row*10 + col`）

| 通道 | 写入者 | 字面公式 |
|---|---|---|
| `0..53` | `mPlants` 中 `!mDead && !mSquished && mRow >= 0 && mRow < 6 && mPlantCol >= 0 && mPlantCol < 9` 的每株（112-115） | `board[min((int)mSeedType, 53)*Cells + mRow*10 + mPlantCol] += (double)mPlantHealth / std::max(1, mPlantMaxHealth)`（两个字段都是 `int32_t`：Plant.h:174-175） |
| `54..87` | `mZombies` 中 `!mDead && IsOnBoard() && mRow >= 0 && mRow < 6` 的每只（116-125） | `board[(54 + min((int)mZombieType, 33))*Cells + cell] += (mBodyHealth + mHelmHealth + mShieldHealth) / 200.0`（三个 `int32_t` 先整型相加，再 `/200.0` → double：Zombie.h:128/130/133） |
| `88` | 同一批僵尸 | `board[88*Cells + cell] += mIsEating;`（`bool`，Zombie.h:98 → `float += bool`） |
| `89` | 同一批僵尸 | `board[89*Cells + cell] += (mChilledCounter > 0);`（`int32_t`，Zombie.h:115） |
| `90` | `mLawnMowers` 中 `!mDead && mRow >= 0 && mRow < 6` 的每台（129-131） | `board[90*Cells + mRow*10] = (mMowerState == MOWER_READY);` **赋值，不是累加**，且固定写在 col 0；同 row 出现第二台时后者覆盖前者 |
| `91` | 地形（132-139） | 对每个 `row` 0..5：`terrain = mGridSquareType[0][row]`；`uniform = (∀col∈1..8: mGridSquareType[col][row] == terrain)`；若 `uniform && (terrain == GRIDSQUARE_GRASS || terrain == GRIDSQUARE_POOL)` → `std::fill_n(board + 91*Cells + row*10, 10, 1.0f)`（**一次填满 10 列，含列 9**） |

`mGridSquareType` 的第一维是 **col**（`mGridSquareType[MAX_GRID_SIZE_X][MAX_GRID_SIZE_Y]`，src/Lawn/Board.h:148）→ 上面的 `[0][row]` 是"第 0 列第 row 行"。

僵尸落格（119-123，最容易走偏的一段）：

```
double position = zombie->mPosX;                                  // mPosX 是 float（Zombie.h:88），展宽为 double 是精确的
const double boundary = 40.0 + 80.0 * std::round((position - 40.0) / 80.0);   // std::round = 半值远离 0，不是 half-to-even
if (std::abs(position - boundary) < 0.001)                        // 严格 <，0.001 px 容差
    position = EnvironmentWaitEvents::RenderedPublicCoordinate(position);   // 十进制 6 位有效数字往返
const int cell = zombie->mRow * 10 + std::clamp((int)std::floor((position - 40.0) / 80.0), 0, 9);
```

- `RenderedPublicCoordinate(v) = std::stod(ostringstream{} << v)`（src/EnvironmentWaitEvents.h:46-53），即按 `ostream` 默认的 **6 位有效数字**格式化再读回。CUDA 等价实现见 `cuda/SimCore.h:142-155`（只在边界 ±0.001 px 内付出代价）。
- `clamp(..., 0, 9)`：`mPosX < 40` 折到列 0，`mPosX >= 840` 折到列 9（草坪外那一列）。
- 注意"只在恰好踩格线时才做十进制往返"这个条件：不踩线时用**未取整的原始 `mPosX`** 去 `floor`。这是渲染口径与私有坐标唯一被打通的地方，位精度必须一致。

### 3.2 115 个标量（141-160）

```
const int waves = std::max(1, state->mNumWaves);                                 // 140
scalars[0]   = std::min(state->mSunMoney, 2000) / 1000.0;                        // 141  int→double 除
scalars[1]   = (double)state->mCurrentWave / waves;                              // 142
scalars[2]   = state->mMainCounter / 60000.0;                                    // 143
scalars[3]   = income / 100.0;                                                   // 144  ← §1.3 的 income，不是 board 状态
scalars[4]   = std::log1p(waves) / std::log1p(50);                                // 145
for (int index = 0; index < mSeedBank->mNumPackets; ++index) {                    // 146-155
    const SeedPacket& packet = mSeedPackets[index];
    const int kind = std::min((int)packet.mPacketType, 53);
    const int cost = packet.mPacketType == SeedType::SEED_NONE ? 0
                     : Plant::GetCost(packet.mPacketType, packet.mImitaterType);
    if (kind < 0) continue;                                                      // SEED_NONE(-1) 的槽位整块跳过
    scalars[5 + kind]  = packet.mActive && (state->mSunMoney >= cost);           // 152  bool→float
    scalars[59 + kind] = std::min(1.0, (packet.mActive || !packet.mRefreshTime) ? 1.0
                                           : (double)packet.mRefreshCounter / packet.mRefreshTime);  // 153-154
}
if (previousTick >= 0) {                                                          // 156-160
    scalars[113] = previousPlant;                                                 // bool→float
    scalars[114] = (state->mMainCounter - previousTick) / 300.0;                  // 用的是**当前** mMainCounter
}
```

占用区间：`0..4`(5) + `5..58`(54 个植物类型通道) + `59..112`(54 个冷却) + `113,114` = **115**，无空洞。

三个偏差源：

1. `scalars[5+kind]` 用**基准价** `Plant::GetCost`，而能否拔卡用 `Board::GetCurrentPlantCost`（含 `PlantUsesAcceleratedPricing` 的 `+50 × CountPlantByType`，src/Lawn/Board.cpp:9406-9414）以及 `mActive`、`mPaused`、`mGameScene != SCENE_PLAYING`、`IsSlotMachineLevel()`、`PlantingRequirementsMet`（`SeedPacket::CanPickUp`，src/Lawn/SeedPacket.cpp:686-723，`mEasyPlantingCheat` 时全部豁免）。→ 特征位与真实合法位可以不一致，CUDA 不要把两者合并成一个。
2. `153-154` 的三目条件是 `(mActive || !mRefreshTime)`（`||` 优先于 `?:`），`mRefreshTime == 0` 视为"已满"。
3. `113/114` 只在 `previousTick >= 0` 时**写**；否则保持 111 行 `assign(…, 0)` 的 0。ROLLOUT 首步传 `priorTick = -1`（src/main.cpp:426）→ 首步这两位恒 0。CUDA 的 `previousValid` 分支等价（`cuda/SimImpl.inc:388-397`）。

### 3.3 35 个任务上下文标量（`taskContext` 为真时，161-166）

```
scalars[Scalars]     = app->mEnvironmentTaskSpec.zombieCountMultiplier / 3.0;      // 163  double
for (int type = 0; type < NUM_ZOMBIE_TYPES && type < 100; ++type)                   // 164
    if (state->mZombieAllowed[type])
        scalars[Scalars + 1 + std::min(type, ZombieTypes - 1)] = 1.0f;              // 165
```

`NUM_ZOMBIE_TYPES == 34 == ZombieTypes` → `min(type,33)` 实际不折叠。`mZombieAllowed[100]`（src/Lawn/Board.h:165）由 `CanZombieSpawnOnLevel(t, theForLevel)` 在关卡初始化写入（src/Lawn/Board.cpp:1162），**RESEED 不改它**（§5）。`zombieCountMultiplier` 来自 `EnvironmentTaskSpec`（`RESET_V2` 第 13 个参数，src/main.cpp:512-517；校验 `1.0 ≤ m ≤ 10.0` 且有限，LawnApp.cpp:1330-1331）。

### 3.4 续走必须自带、board 里没有的数据

| 特征位 | 数据 | CPU 侧来源 | CUDA 侧承载 |
|---|---|---|---|
| `scalars[3]` | income | `history` deque + `historyStart`，纯调用方状态（src/main.cpp:396、441-445） | 每 lane 一个精确事件环 + 窗口起点；`RootRecord.incomeTick/incomeValue/incomeWindowStart`（`CudaExport.cpp:139-149`）+ `S_INCOME_*`。窗口是**只读时过滤**、写入时不裁剪，`PushIncomeEvent` 的满环丢 1/4 是 CUDA 侧的近似（`cuda/SimCore.h:166-189`），CPU 永不丢（history 只在 441 行按 tick 丢） |
| `scalars[113]` | 上一步是否真的种下 | `priorPlant`（460/465/469） | `S_PRIOR_PLANT` |
| `scalars[114]` | 上一步的起始时钟 | `priorTick`（426、471）——**动作前**的 `mMainCounter`，不是 `now` | `S_PRIOR_TICK` |
| 事件推送 | `mSunMoneyProduced`（绝对 uint32） | 435、473 | `S_SUN_PRODUCED`；跨 `RESTORE` 会被快照覆盖回绝对值（src/main.cpp:256），不能只带增量 |
| 其它 | `mSunMoney / mMainCounter / mCurrentWave / mNumWaves / mZombieAllowed / zombieCountMultiplier` | board / taskspec | `RootRecord` 162-172、`LevelInfo.zombieCountMultiplier`、`LevelInfo.allowed[]` |

---

## 4. ROLLOUT 响应回报的统计（src/main.cpp:488-496）

单行 `PVZENV {…}`，按打印顺序：

| 键 | 计算式（file:line） | 增量/绝对 |
|---|---|---|
| `protocol_version` | 常量 `4`（src/main.cpp:52） | — |
| `ok` | §1.1 的合取链与副作用 | — |
| `observation` | `app->mBoard ? app->EnvironmentLeanObservation() : "null"`（486）；**恒为 lean 形态**，与 `OBS_LEAN` 开关无关（`mEnvironmentLeanObservation` 只影响 `EnvironmentObservation(false)`，LawnApp.cpp:1727-1728）。lean 字段表见 LawnApp.cpp:1602-1702：`level, terrain, night, pool, fog, roof, tick, sun, wave, wave_count, wave_timer, terminal, result, enemy_zombies_on_screen, grid[6][9], packets[index,type,active,refreshCounter,refreshTime,cost], plants[row,col,type,health,max_health,squished], zombies[row,type,x,velX,body+helm+shield,on_board,is_eating,chilled,iceTrap,id], defenses[row,state], grid_items[row,col,type], legal_actions.plants[{packet,col,row}]` | 绝对 |
| `decisions` | `++decisions` 次数（477） | 本次续走增量 |
| `mowers_left` | `readyMowers()` = `count(!mDead && mMowerState == MOWER_READY)`（428-430、479） | **绝对**计数，但取"最后一个非终局观测"的值（终局那步不刷新） |
| `planted` | `std::array<int,10>`，`++planted[action.packet]`（418、470） | 本次增量；索引是**卡槽号 0..9**，不是植物类型；只统计 `EnvironmentPlant` 返回 true 的请求 |
| `simulationSeconds` | 动作执行段计时和（448、476）；`isFirst` 与 `!isFirst` 两条路径都计时 | 本次增量 |
| `networkSeconds` | Predict 段计时和（440、446）；注入动作那一步不计 | 本次增量 |
| `net_calls` | `rollout.NetworkCalls()`（493 ← EnvironmentRollout.h:30、.cpp:271）；`ResetNetworkCalls()` 只在 `ok` 分支调用（431） | **`ok == false` 时是历次累计值**，不是 0 |
| `ticks` | `app->mBoard->mMainCounter - startTick`（493；`startTick` 只在 `ok` 分支赋值，424） | `ok` 时是增量；`ok == false` 时 `startTick == 0` → 绝对时钟。此行**无条件解引用 `mBoard`**（`observation` 那里才做判空） |
| `substeps` | `[[now, produced], …]`，**每步一条**、含 `produced == 0`（475、494-495） | `now` 绝对 tick，`produced` 每步 `uint32` 增量 |

明确 `NOT FOUND` 的项（对齐 `RolloutResult` 时别按注释找）：

- **`events` 块：ROLLOUT 响应里没有。** `ROLLOUT` 分支在 497 行 `continue`，走不到通用尾部的 events 输出（src/main.cpp:908-919）；405 行 `before = ReadCounters(app)` 因此是死写。`events` 只出现在：
  - `BRANCH_SNAPSHOT_FAST` / `BRANCH_OBSERVE` 的每个分支（701-708）；
  - `PLANT`/`SHOVEL`/`WAIT`/`WAIT_EVENT_V1`/`WAIT_UNTIL_LEGAL`/`OBS`/`PRIV`/`RESET_V2`/`SNAPSHOT`/`RESTORE`/`RESEED`/`SIM_SKIP`(非 minimal 的那些) 等落入通用尾部的命令。
  两处口径相同：`ReadCounters`（155-161）的**差值**，键为 `zombies_killed = after.zombiesKilled - before.zombiesKilled`、`plants_eaten`、`sun_produced = after.sunProduced - before.sunProduced`、`sun_spent = std::max(before.sun - after.sun, 0)`（唯一"减法取正、且不记录赚多少"的项）、`mower_triggered = after.mowers - before.mowers`（`mowers` ← `mTriggeredLawnMowers`）、`waves_started = after.waves - before.waves`（`waves` ← **`mTotalSpawnedWaves`**，注意与 `scalars[1]` / lean 的 `wave`（← `mCurrentWave`）不是同一个计数器）、`level_won = (mBoard->mLevelComplete)`、`level_lost = (mGameScene == SCENE_ZOMBIES_WON)`。
- **`plants_planted`：ROLLOUT 响应里没有**，只有 `planted[10]`（按卡槽）。CPU 侧根本没有"总种植数"计数器（`CudaExport.cpp:176` 的注释明说 `record.plantsPlanted = 0; // the original does not count plantings`）。
- `cuda/CudaRollout.h:265-267` 与 `cuda/SimCore.h:226` 写的"与 ROLLOUT 响应的 events 块同口径"**不成立**（该响应没有 events 块）。`RolloutResult` 各字段的 CPU 对应物只能是：
  | `RolloutResult` | CPU 对应 | 备注 |
  |---|---|---|
  | `decisions` | `decisions`（477） | ✓ 直接可比 |
  | `ticks` / `endTick` | `ticks` = `mMainCounter - startTick`；`endTick` = `mMainCounter` | ✓ |
  | `sunProduced` | `events.sun_produced`（差分）/ `substeps` 求和 | 两者应相等（`producedBefore` 链覆盖无产出步） |
  | `zombiesKilled`/`plantsEaten` | `events.*` 差分 | ROLLOUT 不回报，需要另发命令比对 |
  | `mowersLeft` | `mowers_left` = READY 计数（428-430、479） | ⚠ `cuda/SimImpl.inc:262` 现在写的是 `S_TRIGGERED_MOWERS` 的**差值**（那是 `mower_triggered` 口径），与 CPU 不同义；同文件 288 行已算出 `mowersReady` 但 `RolloutResult` 里没有该字段。终局那步不刷新这条语义也不能丢 |
  | `plantsPlanted` / `plantedByType[]` | CPU 只有 `planted[]`（按**卡槽**） | 卡槽 → 类型的映射由卡组决定；若 CUDA 想按类型统计，比对时要先换算，且"被接受但没落地"的语义要一致（§1.3 第 7 点） |
  | `result` | `observation.terminal` + `observation.result`（lean，LawnApp.cpp:1615-1616） | ROLLOUT 不回报 `events.level_won/lost` |
  | `rowMinZombieX[]` | **NOT FOUND**（CPU 侧没有对应回报字段，只能从 `observation.zombies` 自行归约，注意渲染口径） |
| 一项 | `net_calls` | `NetworkCalls()` | ✓ 语义 = "真的跑了前向的次数"，合法短路不计 |

---

## 5. RESEED 语义（src/main.cpp:269-281；命令 768-776）

`RESEED <uint32>` 与 `ROLLOUT` 的 `seed` 参数（`seed >= 0` 时）走的是同一个 `ReseedEnvironment`（402、775）。逐语句：

```
Board* board = app->mBoard;
const int numWaves = board->mNumWaves;                                   // 272  先存（这是 waveCap 截断之后的值）
const int sent = std::clamp(board->mCurrentWave, 0, (int)MAX_ZOMBIE_WAVES);   // 273
std::vector<ZombieType> kept(&mZombiesInWave[0][0],
                             &mZombiesInWave[0][0] + sent * MAX_ZOMBIES_IN_WAVE);  // 274-275  扁平前缀拷贝
SRand(seed);                                                             // 276  重播全局 Sexy rand
board->PickZombieWaves();                                                // 277  整表重抽
board->mNumWaves = numWaves;                                             // 278  把 waveCap 塞回去
std::copy(kept.begin(), kept.end(), &mZombiesInWave[0][0]);              // 279  已开打的波原样覆盖回来
SRand(seed ^ 0x9E3779B9u);                                               // 280  对局随机流换另一个种子
```

`Board::PickZombieWaves`（src/Lawn/Board.cpp:578-782）的确切副作用：
- 重算并写 `mNumWaves`（581-616；冒险模式 `mNumWaves = gZombieWaves[clamp(level-1,0,49)]`，非首周目且非小 boss 关再 `mNumWaves < 10 ? 20 : mNumWaves+10`，589-593）。
- `if (mApp->mEnvironmentMode) mApp->mEnvironmentTaskSpec.originalWaveCount = aFullWaves;`（619）——**会改 taskspec**（`aFullWaves` 是未截断的原值，与 `mNumWaves` 在 278 行被复原不同）。
- **`for (auto& aWave : mZombiesInWave) std::fill(…, ZOMBIE_INVALID);`**（624-625）——整张 100×50 表被清空。
- 逐波 `ZombiePickerInitForWave` / `PutZombieInWave` / `PickZombieType` / `PutInMissingZombies` 重抽（630-781），每次抽都从全局 `SRand` 流取值；注意 627-629 的注释：先生成**完整**原表再考虑 `waveCap`，因为这段 RNG 消耗会影响后续初始计数器（含天空阳光）。

回答"续走能不能重抽波"：

1. **能，而且 `mZombiesInWave` 确实会被重抽** —— 但只有 `wave >= mCurrentWave`（尚未开打）的波；`wave < mCurrentWave` 的波被 274-275 的 `kept` 缓冲原样保留（279 行覆盖回来）。代码证据：src/main.cpp:272-279 + src/Lawn/Board.cpp:624-625。设计意图见 src/main.cpp:263-268 的注释（搜索不许看真实未来）。
2. **不传 `seed`（`seed == -1`）时完全不碰波表** —— 续走必须复现导入的表，这正是 `RootRecord.waveTable` 存在的理由（`CudaExport.cpp:303-319` 导出；HEAD 版本还没导，见 §7）。
3. RESEED **不改**：`mNumWaves`（被 278 行复原）、`mZombieAllowed[]`、地形、卡组/`SeedPacket` 状态、`mMainCounter`、`mAppCounter`、`mSunMoney*`、植物/僵尸/投射物/硬币/推车数组、`mBoardRandSeed`、`std::srand`。它**改**：全局 `SRand` 状态两次（最终是 `seed ^ 0x9E3779B9`）、`mZombiesInWave[wave >= mCurrentWave]`、`mEnvironmentTaskSpec.originalWaveCount`。
4. **顺序敏感**：RESEED 不写快照，也不受快照影响；但其后的 `RESTORE`/带 `rootId` 的 `ROLLOUT` 会用快照里的 `randState` 覆盖全局流（src/main.cpp:250）→ `ROLLOUT` 永远是"先恢复、后重播"（401→402），CUDA 侧 `RolloutRequest.futureSeed` 只能对应"恢复之后"的那次重播。
5. ⚠ **三处注释与实现冲突**，别一起抄：`cuda/CudaRollout.h:218-220`（工作区）与 `src/CudaExport.cpp:300-302`（未提交）都写"RESEED 从不重抽这张表"，`cuda/SimCore.h:239-241` 写"spawn cursor 只走不写"。前两句被 src/main.cpp:277 → Board.cpp:624-625 直接否证。成立的最强命题只是"**不重抽已开打的波**"。

---

## 6. `SIM_SKIP` / `EnvSimSkip` mask 29（src/PvzpLib/PvzpCommon.h:177-195；命令 src/main.cpp:744-755）

- 进程级 `int gEnvSimSkip = 0`（src/PvzpLib/PvzpCommon.cpp:381）。`SIM_SKIP <mask>` 是 **sticky** 的：`mask >= 0 && (mask & ~ENV_SKIP_ALLOWED) == 0` 才接受，接受就写 `gEnvSimSkip`，直到下次改动（748-753）。`ROLLOUT` 的第 4 个参数带同一个 mask，进入循环前保存/退出时复原（422-423、482），所以只作用于循环内部。允许位 `ENV_SKIP_ALLOWED = ENV_SKIP_COLOR(1) | ENV_SKIP_HEIGHT(4) | ENV_SKIP_ATTACHMENT_MATRIX(8) | ENV_SKIP_MESSAGE(16) = 29`（PvzpCommon.h:193），Python 侧同校验（python/pvz_env.py:649-650）。
- 六个站点，各自跳掉的**只是呈现层工作**：

| 位 | 站点 | 跳掉的确切内容 | 仍照常执行的部分 |
|---|---|---|---|
| 1 `ENV_SKIP_COLOR` | src/Lawn/Plant.cpp:2745（`Plant::UpdateReanim`） | `UpdateReanimColor()` 整段（reanim 颜色） | `UpdateReanim` 其余全部（transform、track、shake） |
| 1 | src/Lawn/Zombie.cpp:5383 | `aBodyReanim->PropogateColorToAttachments()` | `aBodyReanim->Update()`（上一行）、overlay 矩阵合并 |
| 4 `ENV_SKIP_HEIGHT` | src/Lawn/Plant.cpp:2749 | `PlantDrawHeightOffset(...)`（花盆/水面浮动）→ `aOffsetY = 0.0f` | reanim 定位、缩放、旋转 |
| 8 `ENV_SKIP_ATTACHMENT_MATRIX` | src/PvzpLib/Reanimator.cpp:520 | `GetAttachmentOverlayMatrix(trackIndex, aOverlayMatrix)` 的**矩阵计算** | `UpdateAttacherTrack`、`AttachmentUpdateAndSetMatrix` 调用本身（写入未初始化的 `aOverlayMatrix`，见 517-522 的声明位置） |
| 8 | src/PvzpLib/Attachment.cpp:721 | `aAttachment->SetMatrix(theMatrix)` | `aAttachment->Update()`、`DataArrayTryToGet`、ID 失效分支 |
| 16 `ENV_SKIP_MESSAGE` | src/Lawn/MessageWidget.cpp:259 | 逐字形文本 reanim 的更新循环（`mTextReanimCount` 次 `ReanimationTryToGet` + track 推进） | 消息 widget 状态机、`mDuration`/倒计时、`SetLabel` |

- 为什么动作逐条相同：这六处都在 reanim / attachment / 文字的**绘制数据**分支上，不触碰 `mSunMoney*`、`mMainCounter`、`mZombieCountDown`、种子卡冷却、僵尸坐标与血量，也**不消耗共享 RNG**。PvzpCommon.h:185-186 记录了反例：**bit 2（自由粒子/拖尾）被否决**，因为跳过它会改变共享 RNG 流，实测 224 局无一局保持动作一致、夜关 16/64 结果翻转 → 所以 bit 2 不在 `ENV_SKIP_ALLOWED` 内。bit 8 也只是"跳过矩阵计算"，挂点更新本身照常跑（183-184：整块跳过会崩）。
- 实测口径（PvzpCommon.h:181-185，`scripts/sim_skip_agreement.py`，2026-10-15）：mask **1 / 4 / 16 / 21** 在日、夜、池、雾、屋顶五类地形上 **224/224 局动作逐条相同**；mask **29** 同样保留全部动作**以及最终 RNG 状态**。
- 对 CUDA 的含义：`CudaExport.cpp:6-9` 用这条实测依据论证"核函数只带 logic、不给 reanim/attachment/particle 留任何字段"。写规格时要把上面那张六行表当证据保留，不要笼统写"渲染无关"。

---

## 7. `CUDA_DUMP` 与 CudaExport（src/main.cpp:783-830；src/CudaExport.cpp）

### 7.1 命令与文件格式

```
CUDA_DUMP "<path>" <writeLevel:0|1> <windowStart> <eventCount> [<tick> <produced>] x eventCount
```
（src/main.cpp:786；Python 侧 python/pvz_env.py:691-713，`windowStart = _sun_history_start_tick or 0`。）
- `0 <= eventCount <= 100000`、`app->mBoard` 非空、`!input.fail()` 才 `ok`（792-803）。**没有** 尾参校验（不读 `extra`），多余 token 被忽略。
- `fopen(path, writeLevel ? "wb" : "ab")`（809）：`writeLevel` 截断并先写 `LevelInfo`（kind 1，811），否则追加；两种都写一条 `RootRecord`（kind 2，812）。
- 记录头 = 三个 `int32` `{kMagic, kVersion, kind}`（`CudaExport.cpp:329-334`）；`kMagic = 0x50564334`("PVC4")、`kVersion = 4`（`cuda/CudaRollout.h:21-23`）。
- 响应键（815-822）：`protocol_version, ok, tick, wave, supported, reason, plants, zombies, projectiles, coins, bytes`。`bytes = std::ftell(out)`，即**整个文件**长度而非本条记录；`ok == false` 时只回 `{protocol_version, ok:false}`（826-827）。
- `pvzc::BuildCudaRoot` 被调用**两次**（一次填响应用的 `record`，804-806；一次在 `WriteCudaRoot` 内部，316/347）：board 未变故结果相同，但成本翻倍，且响应字段与落盘记录是两次独立构建。

### 7.2 exporter 当前写了哪些字段（`BuildCudaRoot`，CudaExport.cpp:153-327）

- 标量：`magic, version`（157-158）、`tick, sun, wave(=mCurrentWave), numWaves, zombieCountDown, zombieCountDownStart, zombieHealthWaveStart, zombieHealthToNextWave, hugeWaveCountDown, sunCountDown, numSunsFallen, sunMoneyProduced, zombiesKilled, plantsEaten, plantsPlanted(恒 0), triggeredLawnMowers, result`（162-180）、`rowGotMowered[6], rowLastPicked[6], rowSecondLastPicked[6]`（181-186）、`supported=1, unsupportedReason=0`（187-188）、`incomeWindowStart, incomeHead(恒 0), incomeCount, incomeTick[], incomeValue[]`（`FillIncome`，136-151）、`waveCount, waveZombieCount[], waveTable[]`（303-319，**仅工作区未提交改动**）。
  - `result = EnvironmentTerminal() ? (BOARDRESULT_WON ? 1 : (BOARDRESULT_LOST ? -1 : 0)) : 0`（178-180）：终局但不是 WON/LOST（如只有 `mGameScene == SCENE_ZOMBIES_WON`）会被写成 `result = 0`，与 `EnvironmentTerminal()` 不同义。
  - `FillIncome` 丢 `event.first < tick - 6000` 与 `event.second == 0`，超过 `kIncomeEvents = 160` 判 `supported = 0, reason = 5000`。
- 实体：`RootPlant{cell=row*10+col, type, health, maxHealth, state, stateCountdown, launchCounter, launchRate, special, extraA(=mShootingCounter), extraB(=mAnimCounter), onBoard(=mIsOnBoard), posX, posY}`（196-209，后五项是工作区未提交改动）；`RootZombie{type,row,body,bodyMax,helm,helmMax,shield,shieldMax,flying,phase,phaseCounter,chilled,age,fromWave,targetCell(row*10+col|-1),posX,posY,velX,stateBits}`（222-242）；`RootProjectile{type,row,damage(gProjectileDefinition),age,posX,posY,velX,velY}`（253-259）；`RootCoin{value=GetSunValue(), ticksToCollect=1}`（266-269）；`RootMower{row,state,timer=(mChompCounter>0?mChompCounter:mRollingInCounter),flags,chompCounter,posX}`（275-281）；`RootPacket{type,active,refreshCounter,refreshTime,cost}`（288-293）。
- 支持性/原因码：植物非内核 7 种 / `mSquished` / `mImitaterType != SEED_NONE` → `reason = out.type`（205-214 区间，判据在 210-214）；僵尸非 4 种 / mind-controlled / `ZOMBIE_INVALID` → `10000 + type`（242-245）；容量 `1000 / 2000 / 3000 / 4000 / 5000`；卡槽类型非 7 种 → `20000 + type`（294-297）；波内僵尸数 > `kWaveZombieSlots(32)` → `6000 + wave`；`mNumWaves > kMaxWaves(30)` → `6100`（310-323）。
- `LevelInfo`（`FillLevelInfo`，80-129）：`level, numWaves, playableRows = 5`（**硬编码**，86）、`zombieCountMultiplier`、`allowed[34]` + `zombies[34]{type,value,firstAllowedWave,pickWeight,variantCount=5(硬编码),body/helm/shield 健康表,velX,velMin,velMax}`、`plantCount=7` + `plants[7]{seedType,cost,health(表),refreshTime,launchRate,damage}`（固定 7 种：PEASHOOTER/SUNFLOWER/CHERRYBOMB/WALLNUT/POTATOMINE/SNOWPEA/REPEATER）。

### 7.3 需要点名的缺口（事实清单）

1. `RootRecord::waveCount / waveZombieCount / waveTable` 只在**未提交的工作区改动**里被写（CudaExport.cpp:303-319）；`HEAD` 版本完全不写，`cuda/SimImpl.inc:111-146`（`LoadRoot`）也不读 → HEAD 导出的 `.cudaroot` 里没有未来波表。
2. `struct WaveTable`（`cuda/CudaRollout.h:235-241`）声明了但**没有生产者也没有消费者**：没有 `kind 3` 记录，`LoadRoot` 不接收，而 `cuda/SimCore.h:241` 的 `UpdateTick(Lane, const LevelInfo&, const WaveTable&, Rand&)` 目前无处取得该对象。
3. `RootRecord::plantsPlanted` 恒 0（176），导入时被硬写 `lane.s(S_PLANTS_PLANTED) = 0`（SimImpl.inc:125）→ 写了没人用，语义上归内核自计。`RootRecord::incomeHead` 恒 0（140），导入时同样硬写 0（SimImpl.inc:133）→ 同上。
4. `magic/version` 由 exporter 写（157-158），设备侧 `LoadRoot` 不校验（只看 `supported`）；校验只存在于 Python 读取器（python/pvz_cuda_root.py）。
5. `LevelInfo`/`ZombieDef` 的全部字段（`value, firstAllowedWave, pickWeight, variantCount, velX, velMin, velMax, bodyHealth/helmHealth/shieldHealth`，CudaExport.cpp:88-108）当前**没有任何 CUDA 代码消费**——`LoadRoot` 收尾就是 `(void)FindZombieDef;`（SimImpl.inc:241）。`PlantDef` 里只有 `refreshTime/cost` 被当 fallback 读（SimImpl.inc:238-239），`health/launchRate/damage` 未被读（`RootPlant.launchRate` 才是被用的那个）。
6. `RootCoin` 只有 `{value, ticksToCollect}`，但 `cuda/SimImpl.inc:207-218` 读 `src.type/posX/posY/velX/velY`（`IntField` 里也没有 `F_COIN_TYPE`，只有 `F_COIN_ACTIVE`）；工作区的 `cuda/CudaRollout.h` 还删掉了 `F_COIN_X/F_COIN_Y`。`StoreResult` 的签名也不一致（SimImpl.inc:249 三参数 vs SimCore.h:227 两参数），`SimImpl.inc:288` 写的 `out.mowersReady` 在 `RolloutResult` 里不存在 → 现状下 CUDA 侧编译不过，字段清单要以 `CudaExport.cpp` 为准。
7. 僵尸位 `ZB_SHIELD_DIED / ZB_VAULTED / ZB_BURNED / ZB_ENTERING`（cuda/CudaRollout.h:104-108）exporter 从不置位（`stateBits` 只组合 6 个，CudaExport.cpp:237-241）→ 内核若依赖需自行推导。`MB_FALLING` 同理（flags 只可能写 `MB_GONE/MB_STATIONARY/MB_MOWING`，278-279）。
8. `RootRecord::mowerCount` 无容量检查（直接 `record.mowers[record.mowerCount++]`，274），依赖 `kMowerSlots == MAX_GRID_SIZE_Y == 6` 恒等；`rowGotMowered[row] = (row < MAX_GRID_SIZE_Y ? … : -100)` 的 else 分支是死代码（183，`kRows == MAX_GRID_SIZE_Y == 6`）。
9. `RootPacket` 不导出 `mImitaterType`（288-293），而 `KernelSupportsPlant(packet.mImitaterType)` 也没被检查——imitater 槽位的 `mPacketType == SEED_IMITATER(48)` 会直接触发 `unsupportedReason = 20000 + 48`（294-297）。带模仿卡的任务族因此整条不可导出。
10. `FillLevelInfo` 的 `playableRows = 5`（86）与 `variantCount = 5`（98）是硬编码假设（草地/夜关 row 5 为 `GRIDSQUARE_DIRT`），与 §3.1 通道 91 的"整行统一 grass/pool"判定不是同一来源：池/雾关 row 5 可种，导出值仍是 5。
