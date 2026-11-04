# task236 马尔可夫完整性审计（只列不改，2026-11-04）

范围：`src/EnvironmentTokens.cpp`（学生 obs-v3 token）、`src/EnvironmentValueNet.cpp`（老师叶值 /
静态网 v5 特征）、`src/LawnApp.cpp` 的 lean/v2 观测。结论先说：**学生 obs-v3 已经带了不少植物内部
计时器，但价值网 v5 几乎只看“种类 + 血量比例”；铲子决策真正需要的“这株还能做什么、还差多久”
在价值网里基本是零信息。** 这不是记忆问题，是特征缺失。

记号：`obs-v3` = `PublicObservationTokens`；`v2` = lean 观测里的 `v2.plants/zombies`（v2timer 学生
读它）；`VN` = 价值网 `EnvironmentValueNet::Features`（v5 = encoder 2：92 通道 × 60 格 + 119 标量
+ 35 任务上下文）。

## 1. 植物内部状态

| 状态 | 影响未来 | obs-v3 | v2 | VN | 说明 |
|---|---|---|---|---|---|
| `mState` | 强 | ✓ /128 | ✓ | — | 土豆雷埋/出土、大嘴花咀嚼、窝瓜起跳、玉米炮装填 |
| `mStateCountdown` | 强 | ✓ /1200 | ✓ | — | 距下一次状态变化；土豆雷剩余准备时间、窝瓜落地 |
| `mDoSpecialCountdown` | 强 | ✓ /1200 | ✓ | — | 特殊动作倒计时（如玉米炮、冰冻菇） |
| `mLaunchCounter` | 强 | **—** | ✓ | — | 下一次射击 / 产阳光倒计时；向日葵“还有多久产 25” |
| `mLaunchRate` | 中 | ✓ /1200 | ✓ | — | 该类植物的射击 / 产阳光周期 |
| `mShootingCounter` | 中 | ✓ /1200 | — | — | 连发 / 多发的相位 |
| `mWakeUpCounter` `mIsAsleep` | 中 | ✓ | ✓ | — | 夜间蘑菇醒来的剩余时间 |
| `mPlantHealth` / `mPlantMaxHealth` | 中 | ✓ | ✓ | 仅 health/max 一个通道 | 价值网只有一个血量比例，没有“当前值”和“上限”分开 |
| `mSquished` | 弱 | ✓ | — | — | 被压扁（占用/不占用格子的边界） |
| `mOnBungeeState` | 中 | ✓ /3 | — | — | 被蹦极抓住/正在被偷 |
| `mX` `mY` | 中 | ✓（投掷类） | — | — | 窝瓜/玉米炮在空中的位置；普通植物恒等于格子中心 |
| `mTargetX` `mTargetY` | 强（少数） | **—** | — | — | 窝瓜落点、玉米炮目标；只看 state+countdown 不能完全恢复 |
| `mDisappearCountdown` | 中 | **—** | — | — | 墓碑吞噬者/一次性植物还剩多久消失 |
| `mSubclass` | 弱 | **—** | — | — | 玉米炮左右/上下、部分变体 |
| `mAnimCounter`（年龄） | 弱 | **—** | — | — | **不需要单独补**：影响未来的不是年龄本身，而是上面的计时器；年龄是这些计时器的派生量 |
| `mRecentlyEatenCountdown` | 无 | — | — | — | 受击闪白，纯表现 |
| `mStartRow` / `mPottedPlantIndex` | 特殊模式 | — | — | — | 保龄球 / 禅境花园，冒险模式无影响 |

**结论**：植物侧真正缺的是 `mLaunchCounter`（obs-v3 与 VN）、`mTargetX/Y`、`mDisappearCountdown`、
`mSubclass`。其中 `mLaunchCounter` 是向日葵/射手“还剩多少产出”的直接状态，也是铲子经济账的核心；
价值网则连 `mState/mStateCountdown/mLaunchRate` 都没有，只剩“类型 + 血量比例”。

## 2. 僵尸内部状态

| 状态 | 影响未来 | obs-v3 | v2 | VN | 说明 |
|---|---|---|---|---|---|
| `mZombiePhase` | 强 | ✓ /128 | — | — | 撑杆跳有杆/无杆、跳跳弹跳、梯子、气球、舞王 |
| `mPhaseCounter` | 强 | **—** | — | — | 当前 phase 内还剩多少 tick；下一次起跳/召唤/落地的时机 |
| `mVelX` | 强 | ✓ /5 | ✓ | — | 价值网只按 80px 分格，速度完全丢失 |
| `mVelZ` / `mAltitude` | 强 | **—** / posY 间接 | — | — | 气球高度、跳跳抛物线；posY 与 altitude 不完全等价 |
| `mFlyingHealth` / `mFlyingMaxHealth` | 强 | **—** | — | — | 气球的本体血量与飞行血量分开；现在看起来永远满血 |
| `mMindControlled` | 强 | **—** | — | — | 被催眠的僵尸是友军；观测里仍是敌方，方向完全反了 |
| `mUseLadderCol` / `mTargetCol` | 强（屋顶） | **—** | — | — | 梯子会架在哪一列、蹦极/投石目标列 |
| `mSummonCounter` | 强（舞王） | **—** | — | — | 下一次召唤伴舞的倒计时 |
| `mFlatTires` | 中（冰车） | **—** | — | — | 冰车轮胎被打掉后的速度变化 |
| `mHasHead` `mHasArm` `mHasObject` | 中 | ✓ | 仅 hasObject | — | 价值网只有总血量，没有“头还在/手还在/梯子还在” |
| `mBodyHealth` / `mHelmHealth` / `mShieldHealth` | 中 | ✓ 分开 | 只有 helm/shield | 仅总血量/200 | 价值网把三者加总，头盔/盾的剩余量丢失 |
| `mChilledCounter` `mButteredCounter` `mIceTrapCounter` | 中 | ✓ | 仅 chilled/ice | 仅 chilled>0 一个通道 | 黄油/冰冻剩余时间在 VN 里是布尔 |
| `mIsEating` | 中 | ✓ | ✓ | ✓ | 唯一进入 VN 的僵尸行为位 |
| `mZombieAge` | 弱 | — | — | — | 不是独立原因；由 phase/counter 决定 |
| `mPlayingSong` / `mDroppedLoot` / `mJustGotShotCounter` | 无 | — | — | — | 表现/掉落/受击反馈 |

**结论**：僵尸侧 obs-v3 缺 `mPhaseCounter`、`mVelZ/mAltitude`、`mFlyingHealth`、`mMindControlled`、
`mUseLadderCol/mTargetCol`、`mSummonCounter`、`mFlatTires`。价值网缺几乎所有行为状态，只有
“种类 + 80px 格 + 总血量 + 是否在吃 + 是否被冻”。

## 3. 卡牌 / 波次 / 棋盘

| 状态 | 影响未来 | obs-v3 | v2 | VN | 说明 |
|---|---|---|---|---|---|
| 卡牌冷却 `mRefreshCounter/mRefreshTime` | 强 | ✓ /3000 | ✓ | ✓ 比例 + ready | 已覆盖 |
| 卡牌是否买得起 | 强 | ✓ | ✓ | ✓ | 已覆盖 |
| 卡牌槽位身份（同种两张） | 弱 | 按类型分通道 | ✓ | 按类型合并 | 冒险模式卡组无重复，影响小 |
| 下一波倒计时 `mZombieCountDown` | 强 | **—** | ✓ | **—** | 学生 obs-v3 看不到；价值网也看不到 |
| 大波倒计时 `mHugeWaveCountDown` | 强 | **—** | ✓ | — | 终局压力 |
| 本波剩余血量 / 起始血量 | 强 | **—** | ✓ | — | 判断这一波是否快清完 |
| 天空阳光倒计时 `mSunCountDown` | 中 | **—** | ✓ | — | 下一次天降阳光 |
| 地上未收集阳光 `Coin` | 中 | **—** | ✓ 总 pending | — | 环境自动收集，实际影响小 |
| 格子物品计时 `GridItem::mGridItemCounter` | 强 | **—** | — | — | 墓碑升起、花瓶内容、弹坑恢复、传送门 |
| 弹坑 / 冰面 | 中 | 仅类型 | — | — | 能否种植 |
| 小推车位置/状态 | 中 | ✓ | 仅状态 | 仅 ready | 价值网只有 ready 位 |
| 子弹 | 中 | ✓ | ✓ | **—** | 价值网完全没有子弹通道 |
| 蹦极 | 强 | ✓ 专用 | 专用 | **—**（encoder 2） | v5 用 encoder 2，没有 bungee 扩展 |

**结论**：波次计时（下一波/大波/波血量）是 obs-v3 和价值网共同的最大盲区；价值网还完全看不到
子弹和格子物品计时。

## 4. 补充方案（不在本任务实施）

1. **价值网先补观测里已有的东西**（不改学生、不改存档）：植物 `state/stateCountdown/doSpecial/
   launchCounter/launchRate`、僵尸 `phase/phaseCounter/velX/velZ/altitude/flyingHealth/
   mindControlled/ladderCol`、波次 `zombieCountDown/hugeWaveCountDown/waveHealth`、子弹、格子物品
   计时。做法：在 `EnvironmentValueNet::Features` 里加通道/标量；`Channels` 与 `Scalars` 升版本
   （PVZVN002），旧 v5 保留只读。代价：价值网重训（叶值与静态网都要），搜索成本增加约 1.2–1.5×。
2. **学生 obs-v3 补 token 字段**：优先用 token 里现成的空槽（植物 slot 9 目前恒 -1，可放
   `mLaunchCounter/mLaunchRate`；僵尸有若干空槽）避免加宽 token 数，但仍会改变特征语义，学生必须
   重训（旧帧新槽为 0，可直接混训但分布有偏移）。
3. **不建议补“年龄 / 已投入阳光”**：它们是沉没成本，且可由 `mLaunchCounter/mStateCountdown`
   恢复“还剩多少产出/还要多久”。补它们只会把 10-18 的“铲掉重种看起来没损失”再喂回去。
4. **顺序**：先补价值网（老师叶值 + 静态网），因为铲子候选的采纳门槛直接读叶值；学生头最后做。
   否则学生会先学会“铲”这个动作，但标签仍来自看不见后果的叶值。

## 5. 影响面

- 改 `EnvironmentValueNet` 特征 = 价值网权重形状变化 → v5/v5roof/static 全部重训，旧 checkpoint 不能
  直接加载（可以按 encoder 版本分支保留旧版）。
- 改 obs-v3 token = 学生输入形状/语义变化 → 学生重训；旧存档的 token 数组仍可读，但新槽为 0。
- 生产 teacher 的 `SEARCH` 候选和叶值共用价值网；重训前不要打开铲子候选（本任务原型默认关）。
- 若只补价值网、不动 obs，学生仍然看不到 `mLaunchCounter`，但老师搜索的 rollout 是真实引擎，能
  推演出来；这条路线可以先验证“铲子后果数据 + 重训叶值”是否足够，再决定学生头。
