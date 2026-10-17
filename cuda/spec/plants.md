# CUDA 机制规格 07：植物 —— 行为、射击时钟、生产与死亡 (PLANT mechanics)

Citations are repo-relative to `/home/newbiexvwu/pvz-cuda`. Line numbers are the current working tree
(branch `cuda-rollout`, commit `cecc87d`). Every constant is copied literally from the source; nothing is
inferred from commercial PvZ knowledge. Where the source is silent the line reads `NOT FOUND`.

Path shorthand used inside sections once a file has been named in full: `Plant.cpp` = `src/Lawn/Plant.cpp`,
`Plant.h` = `src/Lawn/Plant.h`, `Board.cpp/.h` = `src/Lawn/Board.cpp` / `.h`, `Zombie.cpp` = `src/Lawn/Zombie.cpp`,
`SeedPacket.cpp` = `src/Lawn/SeedPacket.cpp`, `Coin.cpp` = `src/Lawn/Coin.cpp`, `ConstEnums.h` = `src/ConstEnums.h`,
`GameConstants.h` = `src/GameConstants.h`, `LawnApp.cpp` = `src/LawnApp.cpp`, `Reanimator.cpp` /
`PvzpParticle.cpp` / `PvzpCommon.cpp` / `PvzpFoley.cpp` / `Attachment.cpp` = `src/PvzpLib/…`,
`SimTick.inc` / `CudaRollout.h` = `cuda/…`. A bare `:1234` continues the file cited most recently in the same
paragraph.

Companion specs (do not re-derive what they already pin down):
* `cuda/spec/board.md` — `Board::Update` / `UpdateGameObjects` / `ProcessDeleteQueue` tick position, RNG stream, sun-from-sky.
* `cuda/spec/projectiles.md` — pea flight, impact, damage table, the `mShootingCounter` deferral (lines 314-317 there).
* `cuda/spec/waves.md` — zombie spawning.
* `cuda/spec/protocol.md` — the decision protocol.

Scope is the supported deck only: `SeedType` 0,1,2,3,4,5,7. Target stage is adventure level 7,
`BACKGROUND_1_DAY` (daytime lawn: no pool, no fog, no roof, no graves, no crater, no lilypad, no flowerpot).
Consequences of that are listed in §0.1 and are what let whole branches be marked dead.

---

## 0. The one table that matters: `gPlantDefs`

`src/Lawn/Plant.cpp:48-102`, type `PlantDefinition` (`src/Lawn/Plant.h:324-336`), accessed only through
`GetPlantDefinition(SeedType)` (`src/Lawn/Plant.cpp:4981-4987`), which asserts `gPlantDefs[t].mSeedType == t`
(`:4983`) — so **the array index IS the SeedType value**. Rows for the deck:

| SeedType | name | `mSubClass` | `mLaunchRate` | `mSeedCost` | `mRefreshTime` | `mPacketIndex` | src line |
|---|---|---|---|---|---|---|---|
| 0 `SEED_PEASHOOTER` | PEASHOOTER | `SUBCLASS_SHOOTER` (1) | **150** | 100 | 750 | 0 | `Plant.cpp:49` |
| 1 `SEED_SUNFLOWER` | SUNFLOWER | `SUBCLASS_NORMAL` (0) | **2500** | 50 | 750 | 1 | `Plant.cpp:50` |
| 2 `SEED_CHERRYBOMB` | CHERRY_BOMB | `SUBCLASS_NORMAL` (0) | **0** | 150 | 5000 | 3 | `Plant.cpp:51` |
| 3 `SEED_WALLNUT` | WALL_NUT | `SUBCLASS_NORMAL` (0) | **0** | 50 | 3000 | 2 | `Plant.cpp:52` |
| 4 `SEED_POTATOMINE` | POTATO_MINE | `SUBCLASS_NORMAL` (0) | **0** | 25 | 3000 | 37 | `Plant.cpp:53` |
| 5 `SEED_SNOWPEA` | SNOW_PEA | `SUBCLASS_SHOOTER` (1) | **150** | 175 | 750 | 4 | `Plant.cpp:54` |
| 7 `SEED_REPEATER` | REPEATER | `SUBCLASS_SHOOTER` (1) | **150** | 200 | 750 | 5 | `Plant.cpp:56` |

Note `SeedType` 6 (`SEED_CHOMPER`) is skipped by the deck. `mPacketIndex` is **not** the SeedType: cherry bomb is
packet 3 and wall-nut is packet 2 (`Plant.cpp:51-52`). `mPacketIndex` is only seed-selection UI data and is read
by no gameplay logic in this deck.

Consumers of the table:
* `mLaunchRate` → `Plant::mLaunchRate` at `Plant.cpp:157` (only place it is read into the plant).
* `mSubClass` → `Plant::mSubClass` at `Plant.cpp:158`; gates `UpdateShooter` at `Plant.cpp:2578-2581`.
* `mSeedCost` → `Plant::GetCost` default branch `Plant.cpp:5045-5046` (the switch above it at `:5015-5035` only
  covers zombie/slot-machine seed types, none of which are in this deck).
* `mRefreshTime` → `Plant::GetRefreshTime` `Plant.cpp:5088-5092` (no override in this deck).
* `mReanimationType` → creates the body reanimation (`Plant.cpp:162-184`) — see §2.4, it changes firing timing.

Health is **not** in the table. It is a hard-coded switch in `PlantInitialize`:

| field | value | source |
|---|---|---|
| `mPlantHealth` default | **300** | `Plant.cpp:133` |
| `SEED_WALLNUT` override | **4000** | `Plant.cpp:288-289` |
| `mPlantMaxHealth` | `= mPlantHealth` after the switch | `Plant.cpp:492` |

So the whole deck except wall-nut is 300/300 and wall-nut is 4000/4000. There is no `mMaxHealth` field;
the name is `mPlantMaxHealth` (`Plant.h:175`).

### 0.1 Dead branches for this deck+stage (verified, so they can be omitted from the kernel)

| gate | expression | value here | source |
|---|---|---|---|
| `Plant::IsInPlay()` | `IsOnBoard() && mode != ZEN_GARDEN && mode != TREE_OF_WISDOM` | **true** | `Plant.cpp:2734-2737` |
| `Plant::IsNocturnal()` | puff/scaredy/fume/seashroom/ice/doom/hypno only | **false for all 7** | `Plant.cpp:5095-5109` |
| `Plant::IsFungus()` | none of the deck | **false** | `Plant.cpp:5110-5123` |
| `Plant::IsFlying()` | none of the deck | **false** | `Plant.cpp:5134-5137` |
| `Plant::IsUpgrade()` | 40-47 only | **false for all 7** | `Plant.cpp:5139-5150` |
| `Plant::MakesSun()` | sunflower / twinsunflower / sunshroom | **sunflower only** | `Plant.cpp:999-1002` |
| `mApp->IsWallnutBowlingLevel()` | not adventure 7 | **false** | `Plant.cpp:2549-2553`, `:2518` skip |
| `mApp->IsIZombieLevel()` / `GAMEMODE_UPSELL` / `GAMEMODE_INTRO` | false | sun producer not silenced | `Plant.cpp:1006-1007` |
| `mBoard->HasLevelAwardDropped()` | false until level end | sun producer runs | `Plant.cpp:1009-1010` |
| `GetFlowerPotAt(col,row)` | roof only | **nullptr** → no `-5` on projectile Y | `Plant.cpp:4672-4675` |
| `StageHasRoof()` / `StageHasPool()` | false | `CanPlantAt` pool/pot branches dead | `Board.cpp:8679-8687` |
| `mIsAsleep`, `mWakeUpCounter`, `mOnBungeeState` | never set for this deck on a day stage | §1 step 2/3 are no-ops | `Plant.cpp:186-187`, `:153-155` |
| `mSquished` | set only by gargantuar/imp/zamboni drive-over | reachable only vs. those zombies | `Plant.cpp:2320` |

---

## 1. A. Update order inside one tick — the literal sequence

### 1.1 Where a plant is driven from

`Board::UpdateGameObjects` iterates `mPlants` **first**, before zombies, projectiles, coins, mowers, cursor and
seed packets (`Board.cpp:4972-5016`). It skips entries with `mDead` (`:4976-4977`), then calls `aPlant->Update()`
(`:4978`). Iteration order is the `DataArray` slot order, not creation order — see `board.md` §1.2.

### 1.2 `Plant::Update()` — 4 steps (`Plant.cpp:2856-2878`)

```cpp
1  if (doUpdate)                 // :2868
2      UpdateAbilities();        // :2870
3      Animate();                // :2871
4      if (mPlantHealth < 0)     // :2873
5          Die();
6      UpdateReanim();           // :2876
```

`doUpdate` is true when `!IsOnBoard() || mApp->mGameScene == SCENE_PLAYING` (`:2865-2866`); the three earlier
`IsOnBoard() && ...` clauses (`:2859-2864`) are wallnut-bowling / zen-garden / upsell and are all false here.
**In the env path the scene is `SCENE_PLAYING`, so `doUpdate` is always true.**

`UpdateReanim()` (`:2739-2854`) is presentation only: reanim position/scale plus `UpdateReanimColor`
(`:2745-2746`). It mutates no counter in this deck. It does, however, indirectly affect one gameplay number —
see §2.4 (pea spawn offset).

### 1.3 `Plant::UpdateAbilities()` — the counter order (`Plant.cpp:2513-2590`)

Literal order. Steps 1-2 are skipped for a plant that is neither in `STATE_DOINGSPECIAL` nor squished, and step 3
for a plant that is not asleep — but **the tests themselves always run**, and so does the RNG.

| # | line | what happens | deck relevance |
|---|---|---|---|
| 1 | `:2515-2516` | `if (!IsInPlay()) return;` | always passes |
| 2 | `:2518-2526` | `if (mState == STATE_DOINGSPECIAL \|\| mSquished)` → `mDisappearCountdown--`; if `< 0` → `Die(); return;` | only via `Squish()`; no deck plant ever sets `STATE_DOINGSPECIAL` (cherry/potato call `Die()` directly, `:4334`, `:4399`) |
| 3 | `:2528-2539` | `mWakeUpCounter--`, wake foley at 60, `SetSleeping(false)` at 0 | dead (no nocturnal plant in deck) |
| 4 | `:2541-2542` | `if (mIsAsleep \|\| mSquished \|\| mOnBungeeState != NOT_ON_BUNGEE) return;` | only `mSquished` can trigger |
| 5 | `:2544` | **`UpdateShooting()`** — owns `mShootingCounter` | shooters only |
| 6 | `:2546-2547` | **`if (mStateCountdown > 0) mStateCountdown--;`** | potato mine only (in deck) |
| 7 | `:2549-2553` | wallnut-bowling dispatch → `return` | dead |
| 8 | `:2555-2576` | the seed-type dispatch chain. For this deck exactly two entries matter: `MakesSun()` → `UpdateProductionPlant()` (`:2570`) and `SEED_POTATOMINE` → `UpdatePotato()` (`:2573`) | sunflower, potato mine |
| 9 | `:2578-2581` | **`if (mSubclass == SUBCLASS_SHOOTER) UpdateShooter();`** — owns `mLaunchCounter` for shooters | peashooter, snowpea, repeater |
| 10 | `:2582-2589` | **`if (mDoSpecialCountdown > 0) { mDoSpecialCountdown--; if (mDoSpecialCountdown == 0) DoSpecial(); }`** | cherry bomb fuse |

Ordering facts that a kernel must not re-arrange:

* **step 5 runs before step 9.** So in the tick where `UpdateShooter` sets `mShootingCounter = 35`, that counter is
  *not* decremented until the following tick. The first pea therefore leaves `35 - 1 = 34` ticks later (§2.3).
* **step 6 runs before step 8.** `UpdatePotato` sees `mStateCountdown` already decremented this tick. A potato mine
  planted with `mStateCountdown = 1500` reaches the `== 0` test inside `UpdatePotato` on the **1500th** tick
  after planting, not the 1501st (§5.4).
* **step 8 and step 9 are mutually exclusive for this deck**: `UpdateProductionPlant` is reached only when
  `MakesSun()` (sunflower, `mSubclass == SUBCLASS_NORMAL`), `UpdateShooter` only when `SUBCLASS_SHOOTER`.
  Both consume `mLaunchCounter`, and no plant in the deck consumes it twice in one tick. (Verified against the
  table: the only type with `MakesSun()` and `SUBCLASS_SHOOTER` would be a hypothetical one; none exists.)
* **step 10 is the last thing `UpdateAbilities` does**, and `UpdateAbilities` is the *first* thing `Update()` does
  (`:2870`); `Animate()` is the second (`:2871`). So in the cherry bomb's fuse tick the sequence is:
  fuse hits 0 → `DoSpecial()` → `Die()` sets `mDead=true` (`:4957`) → `Animate()` still runs on the corpse (and
  still draws its 2 shake floats) → `if (mPlantHealth < 0) Die()` re-runs `Die()` (harmless: the `mDead = true`
  write is idempotent, `RemoveEffects()` at `:4958` no-ops on already-null reanims) → `UpdateReanim()`. The zombie
  and projectile loops that follow inside `UpdateGameObjects` (`:4981`, `:4988`) already see the plant as dead,
  because `GetTopPlantAt` skips `mDead` (`Board.cpp:2269-2283`). See §6.
* `Plant::Update` is never called on a plant with `mDead` set (`Board.cpp:4976`), so the counters of a dead-but-not-yet-freed
  plant are frozen from the next tick on.

### 1.4 `Plant::Animate()` — 5 counter decrements + 2 RNG draws (`Plant.cpp:3420-3484`)

```
1  :3422-3426  if ((SEED_CHERRYBOMB || SEED_JALAPENO) && mode != ZEN_GARDEN)
                  mShakeOffsetX = RandRangeFloat(-1.0f, 1.0f);      // 1 float RNG draw
                  mShakeOffsetY = RandRangeFloat(-1.0f, 1.0f);      // 1 float RNG draw
2  :3428-3431  if (mRecentlyEatenCountdown > 0) mRecentlyEatenCountdown--;
3  :3432-3435  if (mEatenFlashCountdown      > 0) mEatenFlashCountdown--;
4  :3436-3439  if (mBeghouledFlashCountdown  > 0) mBeghouledFlashCountdown--;
5  :3441-3445  if (mSquished) { mFrame = 0; return; }
6  :3447-3450  if (SEED_WALLNUT || SEED_TALLNUT) AnimateNuts();
7  :3460       UpdateBlink();                    // owns mBlinkCountdown
8  :3462-3483  mAnimPing ping-pong on mAnimCounter;  mFrame = mAnimCounter / mFrameLength;
```

**Step 1 is an unconditional 2-float-per-tick RNG tax while a cherry bomb is alive.** A cherry bomb lives exactly
100 ticks (§5.1), so planting one consumes **200 extra float draws** before it explodes. In a shared global MT stream
(`board.md` §3.5) any kernel that omits these draws desynchronises every later random event — zombie waves, sun
drops, shooter phase — from the CPU reference. This is the single most load-bearing line in this file for
reproducibility.

`mPlantHealth` is **not** decremented anywhere in `Animate()`. Damage only arrives from `Zombie::EatPlant`
(`Zombie.cpp:7100`) and the explicit `Die()`/`Squish()` paths (§6).

Step 8 (the `mAnimCounter` ping-pong) is pure presentation: it only produces `mFrame`
(`Plant.h:170`) and is read by `Draw`/`DrawSeedType`. `mAnimCounter` is **not** read by any gameplay logic for
this deck — verified: reads of `mAnimCounter` are `Plant.cpp:3464`, `:3466`, `:3471`, `:3474`, `:3476`,
`:3481`, `:3483`, all inside `Animate()`, plus serialisation (`SaveGame.cpp`) and observation export.
`mFrameLength` (`:123`, `RandRangeInt(12,18)`) and `mNumFrames` (`:127`, fixed 5) feed only that ping-pong.

`mDoSpecialCountdown` / `mStateCountdown` / `mLaunchCounter` / `mShootingCounter` / `mDisappearCountdown`
are **all owned by `UpdateAbilities`**, not by `Animate`. There is no double-decrement anywhere.

### 1.5 RNG draw budget of one planted plant (exact sequence, global MT stream)

`Sexy::Rand` is a single global Mersenne Twister (`board.md` §3.5).
`RandRangeInt(a,b) = a + Rand(b-a+1)` (`src/PvzpLib/PvzpCommon.cpp:384-387`, both ends inclusive).
`Rand(x)` consumes one draw even for `x == 0`.

`PlantInitialize` (`Plant.cpp:108-501`) draws in this order. All 7 deck types take the `mReanimationType !=
REANIM_NONE` branch (`:162`), so the `:167` and `:183` draws always happen.

| order | line | draw | peashooter/snowpea/repeater | sunflower | cherry bomb | wall-nut | potato mine |
|---|---|---|---|---|---|---|---|
| 1 | `:123` | `int` `RandRangeInt(12,18)` → `mFrameLength` | yes | yes | yes | yes | yes |
| 2 | `:167` | `float` `RandRangeFloat(10,15)` → body reanim `mAnimRate` | yes | yes | yes | yes | yes |
| 3 | `:183` | `int` `400 + Rand(400)` → `mBlinkCountdown` | yes | yes | yes | yes | yes |
| 4 | `:192` or `:194` | `int` — see §2.1 / §3.1 | `RandRangeInt(0,150)` | `RandRangeInt(300,1250)` | none (`mLaunchRate==0`, `:197`) | none | none |
| 5 | `:226` | `float` `RandRangeFloat(15,20)` → shooter body+head `mAnimRate` | yes | — | — | — | — |
| 5' | `:290` | `int` `1000 + Rand(1000)` → `mBlinkCountdown` **overwritten** | — | — | — | yes | — |

Totals: **shooter = 5 draws** (3 int + 2 float), **sunflower = 4** (3 int + 1 float), **wall-nut = 4** (3 int + 1 float),
**cherry bomb = 3** (2 int + 1 float), **potato mine = 3** (2 int + 1 float).
Note the wall-nut's blink countdown is drawn twice in the source (`:183` then overwritten at `:290`) — both draws
are consumed from the stream even though only the second value survives.

After planting, per-tick ongoing draws while the plant lives:
* cherry bomb: **2 float/tick** for 100 ticks (`:3424-3425`).
* potato mine on reaching armed: **1 int** at `:1177` (`mBlinkCountdown = 400 + Rand(4000)`).
* **every plant is a recurring blink-RNG consumer.** `UpdateBlink` (`:3067-3089`) decrements `mBlinkCountdown`
  (`:3083`) and calls `DoBlink()` when it reaches 0 (`:3086`). The **first statement** of `DoBlink`
  (`:3015`) is `mBlinkCountdown = 400 + Rand(400);` (`:3017`) — it runs *before* every early-return guard
  (`NotOnGround() || mShootingCounter != 0` at `:3019-3020`, unarmed potato mine at `:3022-3023`, null reanim at
  `:3031`, tallnut/garlic image override at `:3034-3036`), so it is one draw per blink event for all 7 deck types.
  Wall-nut then draws **two more** ints on the same event: the nut redraw `1000 + Rand(1000)` at `:3041` and
  `Rand(10)` inside `AttachBlinkAnim` (`:2891`, draw at `:2902`, call site `:3044`). No other deck type draws in
  `AttachBlinkAnim` — `Rand(3)` at `:2914` is threepeater, `Rand(2)` at `:2939`/`:2954` are splitpea/twinsunflower.
  Blink cadence therefore is: non-nut 1 int per `400..799` ticks; wall-nut 3 ints per `1000..1999` ticks.
* reanimations with a non-zero `mShakeOverride` track draw 2 floats per track per tick inside
  `Reanimation::Update` (`src/PvzpLib/Reanimator.cpp:508-511`). Whether any deck reanim has shake data is unknowable from
  the repo (the `.reanim` files are not checked in) — see `UNKNOWN` §9.

### 1.6 Particle emission consumes RNG, and it is NOT env-skipped

This is easy to get wrong, so it is stated explicitly. `LawnApp::AddPvzpParticle`
(`src/LawnApp.cpp:3088-3091`) → `PvzpParticleHolder::AllocParticleSystem`
(`src/PvzpLib/PvzpParticle.cpp:1271-1274`) → `AllocParticleSystemFromDef` (`:1252-1268`) →
`PvzpParticleInitializeFromDef` (`:251-274`), which loops the definition's emitters and calls
`PvzpEmitterInitialize` (`:276-311`) for each. That function draws, **per emitter**:

```
297   1 float  if the emitter's SystemDuration track is set (0 if not)
304-305  2 × mSystemFields.count floats
307-308  16 floats  (ParticleTracks::NUM_PARTICLE_TRACKS == 16, src/PvzpLib/PvzpParticle.h:218-237)
310   Update() — which can immediately call SpawnParticle()
```

and `SpawnParticle` (`src/PvzpLib/PvzpParticle.cpp:325-440`) draws **per particle**:

```
338-339  2 × mParticleFields.count floats
341-342  16 floats
344-347  4 floats (duration, launch speed, emitter offset X/Y)
355      1 float  if PARTICLE_RANDOM_START_TIME
357      1 float  (launch angle interp)
371 / 382 / 391-392   1 / 1 / 2 more floats depending on emitter type (random angle, circle radius, box X/Y)
435-436  2 floats (emitter skew X/Y)
```

i.e. **roughly 24-28 float draws per spawned particle plus ~17-19 per emitter instantiated**, all from the same
global MT stream as gameplay. No `gEnvSimSkip` flag and no `mEnvironmentMode` / `mHeadlessMode` test guards any of it
(the only env skip in this layer is `ENV_SKIP_ATTACHMENT_MATRIX` at `src/PvzpLib/Attachment.cpp:721`). The
exact per-emitter counts live in the `.particle` definition data (`gParticleDefArray`, loaded at
`PvzpParticle.cpp:206-220`), which is not in the repo — see `UNKNOWN` §9.

Plant-side emission sites that a rollout therefore has to reproduce (or reproduce the *absence* of):

| trigger | effect | site |
|---|---|---|
| any planting (day lawn) | `PARTICLE_PLANTING` at `GridToPixelX+41, GridToPixelY+74` | `Board.cpp:2084` inside `DoPlantingEffects` `:2044-2086`; pool variant `:2079` is roof/pool only |
| wall-nut crack stage change | `PARTICLE_WALLNUT_EAT_LARGE` | `Plant.cpp:3127`, `:3135` |
| potato mine reveal (`NOTREADY → RISING`) | `PARTICLE_POTATO_MINE_RISE` | `Plant.cpp:1153` |
| snow pea fire | `PARTICLE_SNOWPEA_PUFF` at `aOriginX+8, aOriginY+13` | `Plant.cpp:4680` |
| **cherry bomb detonation** | `PARTICLE_POWIE` | `Plant.cpp:4331` (between the kill at `:4328` and `Die()` at `:4334`) |
| **potato mine detonation** | `PARTICLE_POTATO_MINE` at `aPosX+20, aPosY` | `Plant.cpp:4396` (between the kill at `:4392` and `Die()` at `:4399`) |
| peashooter / repeater fire | **no particle** — only `FOLEY_THROW`, which is suppressed | `Plant.cpp:4557`, `:4680-4690` (puff branches are not this deck) |

**Foley is the opposite case and is fully suppressed:** `LawnApp::PlayFoley` (`src/LawnApp.cpp:2734-2740`) and
`PlayFoleyPitch` (`:2742-2748`) both early-out when `mEnvironmentMode || mHeadlessMode`, so the pitch draw at
`src/PvzpLib/PvzpFoley.cpp:305-306` never happens in a rollout. `FOLEY_SPAWN_SUN` (`Plant.cpp:1035`),
`FOLEY_THROW` (`:4557`) and `FOLEY_SNOW_PEA_SPARKLES` (`:4560`) therefore cost **0 RNG** here, even though they cost
1 draw each in the interactive game.

---

## 2. B. Shooters: peashooter / snow pea / repeater

### 2.1 `Plant::UpdateShooter` — the shot clock (`Plant.cpp:945-997`)

```cpp
947   mLaunchCounter--;
948   if (mLaunchCounter <= 0)
949   {
950       mLaunchCounter = mLaunchRate - Sexy::Rand(15);
...           // dispatch, see below
980   }
986   if (mLaunchCounter == 25)
987   {
988       if (mSeedType == SEED_REPEATER || mSeedType == SEED_LEFTPEATER)
990           FindTargetAndFire(mRow, WEAPON_PRIMARY);
...
996   }
```

Initial value (`Plant.cpp:189-197`):
```cpp
189  if (mLaunchRate > 0) {
191      if (MakesSun())  mLaunchCounter = RandRangeInt(300, mLaunchRate / 2);   // sunflower path
194      else             mLaunchCounter = RandRangeInt(0, mLaunchRate);          // shooters: 0..150
     }
197  else mLaunchCounter = 0;
```

**Decisive facts:**

1. **The clock free-runs.** `mLaunchCounter` is decremented every tick (`:947`) and reset unconditionally the
   moment it is `<= 0` (`:950`) — *before* any target test. A row that was empty for one whole cycle
   **consumes** that shot. A kernel that holds the clock at 0 until a target appears fires measurably more peas.
   (The draft kernel gets this right — `cuda/SimTick.inc:197-200` comments it explicitly.)
2. **The reset value is `150 - Rand(15)`** ⇒ integer in **[136, 150]** inclusive, one `int` draw per reset.
   `Rand(15)` yields 0..14, so 150-0=150 down to 150-14=136. Never 151, never 135.
3. **The initial phase is one `int` draw from [0, 150] inclusive** (`:194`). Two shooters planted on the same tick
   are out of phase by up to 150 ticks, and this is the *only* thing that decorrelates them. Reproducing aggregate
   DPS requires reproducing these draws in stream order.
4. **Firing requires a target, but the decision does not gate the clock.** Inside the `<= 0` branch the `else`
   clause (`:976-979`) calls `FindTargetAndFire(mRow, WEAPON_PRIMARY)`; that returns `false` with no side effect
   when no zombie is in range (`:734-736`).
5. The `mLaunchCounter == 25` test at `:986` is evaluated in the **same tick** as the reset at `:950`. Since the
   reset value is `>= 136`, it can never be true on the reset tick — it becomes true `reset - 25` ticks later.

### 2.2 `Plant::FindTargetAndFire` — it does NOT fire (`Plant.cpp:732-818`)

```cpp
734   Zombie* aZombie = FindTargetZombie(theRow, thePlantWeapon);
735   if (aZombie == nullptr) return false;
738   EndBlink();
751   else if (aHeadReanim && aHeadReanim->TrackExists("anim_shooting"))
752   {
...       // head anim set to "anim_shooting", PLAY_ONCE_AND_HOLD, rate 35 (45 for repeater)
758       mShootingCounter = 35;
759       if (mSeedType == SEED_REPEATER || mSeedType == SEED_SPLITPEA || mSeedType == SEED_LEFTPEATER)
761       {   aHeadReanim->mAnimRate = 45.0f;
762           mShootingCounter = 26;   }
...
814   else
815       Fire(aZombie, theRow, thePlantWeapon);     // only if NO reanim has an shooting track
817   return true;
```

`mShootingCounter` values for the deck: **35** (peashooter, snowpea) `:758`; **26** (repeater) `:762`.
The `Fire(aZombie, ...)` at `:815` is only reached when both the head reanim and the body reanim lack an
`anim_shooting` track — for peashooter/snowpea/repeater the head reanim **is created** at `:227-231` and the
fallback at `:785-813` (`mShootingCounter = 29`, `:811`) applies to the non-head plants. See §2.4 for why this
still has to be modelled, and `UNKNOWN` §9.

### 2.3 `Plant::UpdateShooting` — the deferred fire (`Plant.cpp:3205-3418`)

```cpp
3207  if (NotOnGround() || mShootingCounter == 0) return;
3210  mShootingCounter--;
...
3248  else if (mShootingCounter == 1)
3249  {   ...
3303      else
3305          Fire(nullptr, mRow, WEAPON_PRIMARY);     // peashooter / snowpea / repeater land here
3308      return;
3311  if (mShootingCounter != 0) return;
...     // reanim reset; if the head anim has not finished looping:
3417  mShootingCounter = 1;
```

**The fire happens exactly 34 ticks after the decision tick for peashooter/snowpea, and 25 ticks after for the
repeater.** (`UpdateShooting` runs at `:2544`, i.e. before `UpdateShooter` at `:2580`, so the counter set on tick
`T` is first decremented on tick `T+1`; it reaches `1` on `T+34` / `T+25`.)
This matches `projectiles.md` lines 314-317.

`Fire()` at `:3305` is called with **`theTargetZombie == nullptr`** — the pea is not target-seeking. For PEA and
SNOWPEA `mMotionType` stays `MOTION_STRAIGHT` (`projectiles.md` §0). The target test is therefore purely a
**gate**, not an aim.

The reanim-dependent tail (`:3311-3417`) cannot change firing timing:
* after `Fire`, the counter is 1; next tick it becomes 0 and either (a) `mLoopCount > 0` → head reset to idle and
  the counter **stays 0**, or (b) still looping → `mShootingCounter = 1` (`:3417`) which pins it at 1.
* In case (b) the pin never re-triggers `:3248` on a later tick, because `:3248` is only reached by *decrementing
  into* 1, and the pin re-sets it to 1 without decrementing past it — each subsequent tick goes `1 → 0 → 1`.
* Either way the next `FindTargetAndFire` overwrites it with 35/26 and the fire lands 34/25 ticks later.
  **So the `mLoopCount` value is provably irrelevant to shot timing** — the kernel does not need reanimation data
  here, only the fixed offset.

### 2.4 Pea spawn position (`Plant::Fire`, `Plant.cpp:4491-4784`)

```cpp
4603  else if (mSeedType == SEED_PEASHOOTER || mSeedType == SEED_SNOWPEA || mSeedType == SEED_REPEATER)
4604  {
4605      int aOffsetX, aOffsetY;
4606      GetPeaHeadOffset(aOffsetX, aOffsetY);
4607      aOriginX = mX + aOffsetX + 24;
4608      aOriginY = mY + aOffsetY - 33;
4609  }
4672  if (mBoard->GetFlowerPotAt(mPlantCol, mRow))  aOriginY -= 5;      // never on a day stage
4693  Projectile* aProjectile = mBoard->AddProjectile(aOriginX, aOriginY, mRenderOrder - 1, theRow, aProjectileType);
4694  aProjectile->mDamageRangeFlags = GetDamageRangeFlags(thePlantWeapon);
```

`GetPeaHeadOffset` (`:3673-3705`) reads the **current transform** of the body reanim's `anim_stem` (or fallback
`anim_idle`) track: `theOffsetX = aTransform.mTransX; theOffsetY = aTransform.mTransY;`. Because the body reanim
animates, **`aOffsetX/aOffsetY are not constant** — they are a function of the reanim's animation time.
If the body reanim cannot be resolved it returns `0,0` (`:3686-3688`) and the origin collapses to `mX+24, mY-33`.

Projectile type is chosen by seed at `:4513-4523`: peashooter and repeater → `PROJECTILE_PEA`;
snowpea → `PROJECTILE_SNOWPEA`. Sound/RNG-free presentation side effects: `FOLEY_THROW` (`:4557`) for all three,
plus `FOLEY_SNOW_PEA_SPARKLES` (`:4560`) and a `PARTICLE_SNOWPEA_PUFF` (`:4680`) for snow pea only.
Damage: `PROJECTILE_PEA` = 20, `PROJECTILE_SNOWPEA` = 20 (`projectiles.md`; table row `Projectile.cpp:39`).
`mDamageRangeFlags` = **1** (`GetDamageRangeFlags` default, `Plant.cpp:636-637`) — ground-only, non-flying.

### 2.5 Snow pea chill — applied on impact, not by the plant

There is **no** chill application inside `Plant::Fire` or `Plant::UpdateShooter`. It is carried as a damage flag:

```
Projectile.cpp:418-421   if (mProjectileType == PROJECTILE_SNOWPEA || == PROJECTILE_WINTERMELON)
                             SetBit(aDamageFlags, DAMAGE_FREEZE);      // bit index 2, ConstEnums.h:252
Zombie.cpp:7522-7539     Zombie::ApplyChill(bool theIsIceTrap)
7532-7536                    int aChillTime = 1000;   (2000 if theIsIceTrap)
7537                         mChilledCounter = std::max(aChillTime, mChilledCounter);
```

So a snow-pea hit sets `mChilledCounter = 1000` ticks (`max`, never additive). The chill then halves the zombie's
animation rate (`Zombie.cpp:6639` `IsMovingAtChilledSpeed() ? rate*0.5f`) and, relevant to plants, **halves eating**:
`Zombie.cpp:7087` `if (mChilledCounter > 0 && mZombieAge % 2 == 1) return;` inside `Zombie::EatPlant` — a chilled
zombie chews only on even `mZombieAge` ticks, i.e. `DAMAGE_PER_EAT` lands every 8 ticks instead of every 4.
Decrement is at `Zombie.cpp:4619-4622`. `NOT FOUND` in this file's scope: the exact `IsMovingAtChilledSpeed()`
speed factor — see `projectiles.md` / zombie spec.

### 2.6 Repeater second pea — exact mechanism

Two separate triggers on the *same* counter, both routed through the same deferral:

| pea | decision | line | requires target? | `mShootingCounter` set | tick the pea spawns |
|---|---|---|---|---|---|
| 1st of the cycle | `mLaunchCounter <= 0`, then reset to `R` | `:948-979` | yes (`:735`) | 26 (`:762`) | decision tick **+25** |
| 2nd of the cycle | `mLaunchCounter == 25` | `:986-991` | yes (`:735`) | 26 (`:762`) | the `== 25` tick **+25** |

Let `T` be the tick on which the decrement at `:947` drives the counter to `<= 0`, and let
`R = 150 - Rand(15) ∈ [136, 150]` be the value written at `:950` on that same tick. Everything follows from that:

| event | tick | note |
|---|---|---|
| decision #1 | `T` | `:986` is also evaluated on `T`, with `mLaunchCounter == R >= 136`, so it cannot double-trigger |
| pea 1 fires | `T + 25` | 26 is set on `T` (step 9) and first decremented on `T+1` (step 5); reaches 1 on `T+25` → `:3248`/`:3305` |
| decision #2 | `T + (R - 25)` | the counter is `R - n` on tick `T+n`, so it equals 25 at `n = R-25` |
| pea 2 fires | `T + R` | decision #2 + 25 |
| decision #1 of the next cycle | `T + R` | the counter hits `<= 0` on the **same tick** pea 2 appears |
| pea 3 (= next cycle's pea 1) | `T + R + 25` | |

Derived numbers the kernel needs, with `R ∈ [136,150]`:
* gap **within** a cycle (pea1 → pea2) = `R - 25` ∈ **[111, 125]** ticks — *not* 25.
* gap **across** the cycle boundary (pea2 → next pea1) = **exactly 25 ticks**. This is the visually paired
  double shot; the pair straddles the cycle boundary, so a "burst" as perceived is (last pea of cycle N,
  first pea of cycle N+1).
* cycle period = `R` ticks for 2 peas ⇒ mean 71.0–75.0 ticks per pea. A peashooter with the same `R` gives
  `R` ticks per pea, so repeater DPS is ≈2x peashooter DPS, as expected, but with a 25/118-ish duty cycle
  rather than two close-together peas.
* `projectiles.md` line 316 calls the 25-tick deferral "the repeater's burst"; read together with the table above
  so the 25 is not mistaken for the intra-burst spacing.

No counter overwrite can occur between the two decisions: by tick `T+(R-25)` the first shot's `mShootingCounter` has
long since been decremented to 0 (it would need `R < 51` to still be live, and `R >= 136`). Once at 0,
`UpdateShooting` returns at `:3207` and never goes negative; the tail at `:3311-3417` may pin it to 1 (`:3417`) while
the head reanim is still looping, and `FindTargetAndFire` then simply overwrites it with 26 at `:762`.

The `if (mLaunchCounter == 25)` second pea exists **only** for a plant whose clock is running. A repeater that never
sees a target still marches through 25 every cycle and no-ops at `:735` each time; a target that appears only on the
`<= 0` tick yields **one** pea, not two.

---

## 3. C. Sun producers: sunflower (`Plant::UpdateProductionPlant`, `Plant.cpp:1004-1074`)

### 3.1 The production clock

Gates, in order (`:1006-1024`), all false/neutral on level 7 daytime:
```
1006  if (!IsInPlay() || IsIZombieLevel() || GAMEMODE_UPSELL || GAMEMODE_INTRO) return;
1009  if (mBoard->HasLevelAwardDropped()) return;
1012  (marigold-only block) — dead for sunflower
1023  (Last Stand gate) — dead
```

The clock:
```cpp
1026  mLaunchCounter--;
1027  if (mLaunchCounter <= 100)
1029      int aFlashCountdown = PvzpAnimateCurve(100, 0, mLaunchCounter, 0, 100, CURVE_LINEAR);
1030      mEatenFlashCountdown = std::max(mEatenFlashCountdown, aFlashCountdown);   // presentation only
1032  if (mLaunchCounter <= 0)
1033  {
1034      mLaunchCounter = RandRangeInt(mLaunchRate - 150, mLaunchRate);
1035      mApp->PlayFoley(FoleyType::FOLEY_SPAWN_SUN);
1050      mBoard->AddCoin(mX, mY, CoinType::COIN_SUN, CoinMotion::COIN_MOTION_FROM_PLANT);
1062      (BIG_TIME extra coin) — dead
1073  }
```

| quantity | value | source |
|---|---|---|
| first production delay | `RandRangeInt(300, mLaunchRate/2)` = `300 + Rand(951)` ⇒ **[300, 1250]** inclusive | `:191-192` (`MakesSun()` branch), `mLaunchRate = 2500` from `:50` |
| every later delay | `RandRangeInt(mLaunchRate-150, mLaunchRate)` = `2350 + Rand(151)` ⇒ **[2350, 2500]** inclusive | `:1034` |
| RNG cost per production | 1 `int` draw | `:1034` |
| `PlayFoley(FOLEY_SPAWN_SUN)` | unconditional on production, before the coin | `:1035` |
| coin | `AddCoin(mX, mY, COIN_SUN, COIN_MOTION_FROM_PLANT)` — one coin | `:1050` |

`RandRangeInt(a,b)` is **inclusive on both ends** (`Rand(b-a+1)+a`, `src/PvzpLib/PvzpCommon.cpp:384-387`) — the draft
kernel's `rand.Range(def->launchRate - cProduceRedrawJitter, def->launchRate)` must match that convention
(see §8 row 9).

**There is no `mProductCounter` field.** The sunflower uses `mLaunchCounter`, the same field the shooters use;
the two are never active on the same plant (§1.3). `NOT FOUND in Plant.h`: `mProductCounter`.

**There is no light reanim gating production.** `mLightReanimID` (`Plant.h:194`) is assigned only in
`UpdatePotato` (`Plant.cpp:1166-1173`) for the potato-mine glow and read for drawing. Nothing in
`UpdateProductionPlant` touches any reanim. The only "light" side effect is `mEatenFlashCountdown` at `:1030`,
which is cosmetic and decays in `Animate` (`:3432-3435`). Production does **not** wait for the flash.

### 3.2 Where the sun value comes from

```
Board.cpp:1971-1981  Coin* Board::AddCoin(int theX, int theY, CoinType, CoinMotion)
1974                     Coin* aCoin = mCoins.DataArrayAlloc();
1975                     aCoin->CoinInitialize(theX, theY, theCoinType, theCoinMotion);
1976-1978              if (aCoin->IsSun())  mSunMoneyProduced += aCoin->GetSunValue();
Coin.cpp:1297-1300     int Coin::GetSunValue() { COIN_SUN ? 25 : COIN_SMALLSUN ? 15 : COIN_LARGESUN ? 50 : 0 }
```

So a sunflower sun is **worth 25**, credited to the cumulative `Board::mSunMoneyProduced` **at creation time**,
not at collection time. `mSunMoney` (the spendable wallet) only changes when the coin is collected — the collection
path is in `board.md` §3.4/§2.1, do not double-count here.

Spawn position is the plant's **top-left cell corner** `mX, mY` (not the centre — contrast cherry bomb's
`mX + mWidth/2`, `:4308`). `mX/mY` come from `GridToPixelX/Y` (`:114-115`). `COIN_MOTION_FROM_PLANT` then applies
its own jitter inside `CoinInitialize`; `board.md` §3.4 records it as `mPosY + 15 + Rand(20)` for the plant-spawned
case — that is **1 extra `int` draw per produced sun** and it belongs to the coin, not the plant, but it is on the
same global stream.

### 3.3 Sun-count bookkeeping at planting

`Board::AddPlant` recomputes `mMaxSunPlants` as `CountPlantByType(SEED_SUNSHROOM) + CountPlantByType(SEED_SUNFLOWER)`
and raises the high-water mark (`Board.cpp:2094-2098`). It counts **sunflower + sunshroom only**, and it is a
cumulative maximum, never decremented on death.

---

## 4. D. Wall-nut and defensive plants

**Nothing about a wall-nut ticks except the generic housekeeping.** `mSubclass == SUBCLASS_NORMAL`
(`Plant.cpp:52`/`:158`) so step 9 (`UpdateShooter`) is skipped; the seed dispatch at `:2555-2576` has no entry for
`SEED_WALLNUT`; `mLaunchRate == 0` so `mLaunchCounter` stays 0 forever (`:197`); `mDoSpecialCountdown` stays 0
(`:134`) so step 10 no-ops. The only per-tick mutation is `Animate()`'s:
`mRecentlyEatenCountdown--`, `mEatenFlashCountdown--`, `mBeghouledFlashCountdown--`, `AnimateNuts()`,
`UpdateBlink()`, and the `mAnimCounter` ping-pong.

`AnimateNuts` (`Plant.cpp:3091-3157`) is where health thresholds surface, and it is **write-only presentation**:

```
3122  if (mPlantHealth < mPlantMaxHealth / 3)          → SetImageOverride(cracked2) + PARTICLE_WALLNUT_EAT_LARGE
3130  else if (mPlantHealth < mPlantMaxHealth * 2/3)   → SetImageOverride(cracked1) + PARTICLE_WALLNUT_EAT_LARGE
```
For a 4000/4000 wall-nut the thresholds are `health < 1333` (stage 2) and `health < 2666` (stage 1). The particle
is emitted only on the tick the stage *changes* (guarded by `aImageOverride != aCrackedN`, `:3124`, `:3132`).

Blink housekeeping is the wall-nut's only RNG cost while idle. `UpdateBlink` (`:3067-3089`) decrements
`mBlinkCountdown` (`:3083`) and calls `DoBlink()` (`:3086`). `DoBlink` (`:3015`) draws **3** ints for a nut on each
blink event: `400 + Rand(400)` at `:3017` (first statement, before any guard), then the nut redraw `1000 + Rand(1000)`
at `:3041`, then `Rand(10)` at `AttachBlinkAnim` `:2902` (called `:3044`) which only picks
`anim_blink_twitch` / `anim_blink_twice` / `anim_blink_thrice`. Non-nut deck types draw only the `:3017` one.
Initial `mBlinkCountdown` for wall-nut: `1000 + Rand(1000)` (`:290`, overwriting `:183`).

**Fields, and what the exporter should read:**

| field | declared | wall-nut value | note |
|---|---|---|---|
| `mPlantHealth` | `Plant.h:174` | 4000 → decreasing | **this is "health"** |
| `mPlantMaxHealth` | `Plant.h:175` | 4000 | assigned once at `:492`, never updated after |

The existing exporter already reads the right ones: `out.health = plant->mPlantHealth;`
(`src/CudaExport.cpp:221`) and `out.maxHealth = plant->mPlantMaxHealth;` (`:222`), with the plant-time base health
mirrored by `PlantBaseHealth` (`src/CudaExport.cpp:71-78`: wallnut 4000, default 300) — consistent with
`Plant.cpp:133` / `:288-289`.

Damage arrives only from `Zombie::EatPlant` (`Zombie.cpp:7100`, `DAMAGE_PER_EAT = 4`, `Zombie.cpp:67-68`) and from
`Board::KillAllPlantsInRadius` / `Squish()` (§6). `Plant::SpikeRockTakeDamage` (`:646-666`) and `NotOnGround`
are not in this deck. There is **no** health regeneration, no armour sub-field, and no separate "eaten" health
bucket on plants.

The other defensive type in the deck is the potato mine's *pre-arm* state, which is eatable (§5.4) — that is the
only defensive nuance here besides the wall-nut.

---

## 5. E. Cherry bomb, and F. Potato mine

### 5.1 Cherry bomb fuse (`Plant::PlantInitialize` + `UpdateAbilities` step 10)

| item | value | source |
|---|---|---|
| fuse field | `mDoSpecialCountdown` | `Plant.h:178` |
| initial value at planting | **100** | `Plant.cpp:327` (inside `case SEED_CHERRYBOMB:`, guarded by `IsInPlay()` at `:325`) |
| default otherwise | 0 | `:134` |
| decrement site | `UpdateAbilities` last step | `:2582-2584` |
| trigger | `if (mDoSpecialCountdown == 0) DoSpecial();` | `:2585-2588` |

So the explosion happens on the **100th tick after planting** (`100 → 99` on tick 1 … `1 → 0` on tick 100).
There is no target test at all — a cherry bomb detonates on a fixed schedule whatever the board looks like.

`Animate` adds the 2 float draws/tick (`:3422-3426`) for exactly those 100 ticks (§1.4).

### 5.2 Cherry bomb AoE (`Plant::DoSpecial`, `Plant.cpp:4306-4336`)

```cpp
4308  int aPosX = mX + mWidth / 2;                 // mWidth == 80  (:149)  ⇒ mX + 40
4309  int aPosY = mY + mHeight / 2;                // mHeight == 80 (:150)  ⇒ mY + 40
4310  int aDamageRangeFlags = GetDamageRangeFlags(WEAPON_PRIMARY);
4323  case SeedType::SEED_CHERRYBOMB:
4328      if (mBoard->KillAllZombiesInRadius(mRow, aPosX, aPosY, 115, 1, true, aDamageRangeFlags) >= 10)
4329          ReportAchievement::GiveAchievement(mApp, Explodonator, true);
4334      Die();
```

| parameter | value | meaning |
|---|---|---|
| radius | **115** | circle radius, px |
| rowRange | **1** | kills in `mRow-1 .. mRow+1` — **yes, adjacent rows** |
| theBurn | **true** | `ApplyBurn()`, not flat damage |
| `aDamageRangeFlags` | **127** | `GetDamageRangeFlags` for cherry bomb, `Plant.cpp:610-614` |

`127 = 0b1111111` ⇒ every `DamageRangeFlags` bit except `DAMAGES_ONLY_MINDCONTROLLED` (index 7) is set
(`ConstEnums.h:257-267`), i.e. ground + flying + submerged + dog + off-ground + **dying** + underground.
`Zombie::EffectedByDamage` (`Zombie.cpp:8084-8086`, whose very first test is the `DAMAGES_DYING` bit at `:8086`)
therefore does not filter anything for a cherry bomb. Contrast potato mine's 77 and shooters' default 1
(`Plant.cpp:636-637`) which is ground-only, so a normal pea cannot hit a flying/submerged zombie.

`Board::KillAllZombiesInRadius` (`Board.cpp:9322-9370`):
```
9327    if (aZombie->mDead) continue;
9329    if (aZombie->EffectedByDamage(theDamageRangeFlags))
9331        Rect aZombieRect = aZombie->GetZombieRect();
9332        int  aRowDist = aZombie->mRow - theRow;            // BOSS forced to 0, :9333-9336
9338        if (aRowDist <= theRowRange && aRowDist >= -theRowRange && GetCircleRectOverlap(theX,theY,theRadius,aZombieRect))
9340-9342       if (theBurn) aZombie->ApplyBurn();            // cherry bomb, jalapeno, doom shroom
9344-9346       else         aZombie->TakeDamage(1800, 18U);  // potato mine, cob cannon, others
9349            aKilledZombies++;
```
`GetCircleRectOverlap` (`Board.cpp:8975-8983`) is a **clamped nearest-point test**, not an x-distance test:
```
aNearX = clamp(circleX, rect.mX, rect.mX + rect.mWidth)
aNearY = clamp(circleY, rect.mY, rect.mY + rect.mHeight)
return (circleX-aNearX)^2 + (circleY-aNearY)^2 <= radius^2
```
It is `<=` and integer arithmetic (`int`), so a zombie exactly on the boundary **is** hit. Note that the row test
(`:9338`) is applied on top of the circle test — an adjacent-row zombie must satisfy **both**. The circle is
centred on the plant, and the rows are 100 px apart (`GridToPixelY = row*100 + LAWN_YMIN`, `board.md` §0), so
r = 115 comfortably reaches both neighbours' rect bands.

`Zombie::ApplyBurn()` (`Zombie.cpp:8687-8829`), literal control flow:
```
8689-8690  if (mDead || mZombiePhase == PHASE_ZOMBIE_BURNED) return;                 // idempotent
8692-8695  if (mBodyHealth >= 1800 || mZombieType == ZOMBIE_BOSS) { TakeDamage(1800, 18U); return; }
              → armoured/heavy zombies take a plain 1800 hit instead of burning
8704-8707  mIceTrapCounter > 0 → RemoveIceTrap();  8708 mButteredCounter = min(mButteredCounter, 0)
8710-8711  detach zamboni-smoke particle; BungeeDropPlant()
8713-8729  if phase ∈ {DYING, POLEVAULTER_IN_VAULT, IMP_GETTING_THROWN, RISING_FROM_GRAVE, DANCER_RISING,
                      DOLPHIN_INTO_POOL/JUMP/RIDING, SNORKEL_INTO_POOL, DIGGER_TUNNELING*/RISING*,
                      MOWERED} || mInPool   →   DieWithLoot()      // :8729
8731-8748  else if BUNGEE || YETI || zombotany || bobsled-on-sled || IsFlying() || !mHasHead
                      → anim rate 0, mZombiePhase = PHASE_ZOMBIE_BURNED (:8740), mPhaseCounter = 300 (:8741),
                        mJustGotShotCounter = 0 (:8742), DropLoot() (:8743)
8750-8828  else (a normal walking ground zombie)
                      → add REANIM_ZOMBIE_CHARRED at (mPosX+22, mPosY-10) (:8752-8754, type swap :8762-8788),
                        aCharredReanim->mAnimRate *= RandRangeFloat(0.9f, 1.1f)   // :8792  ← 1 FLOAT RNG DRAW
                        then DieWithLoot() (:8828)  [= DieNoLoot() + DropLoot(), :7321-7325]
```
So "cherry bomb damage = 1800" is a **simplification that is only exact for `mBodyHealth >= 1800`** (and the boss).
For everything else the result is an immediate kill (`DieWithLoot()`) plus a charred corpse reanim, and the
`mPhaseCounter = 300` burned phase belongs only to the special-case branch at `:8731`. **Each charred kill also
consumes 1 float from the global stream** (`:8792`), on top of the bomb's own 200 shake draws (§1.4). `DropLoot`
draws nothing here — its only `Rand` is gated on `IsLittleTroubleLevel()` (`Zombie.cpp:7295`).
`theBurn == false` (potato mine) is the pure `TakeDamage(1800, 18U)` path (`:9346`) — flag 18 =
`DAMAGE_BYPASSES_SHIELD(0) | DAMAGE_DOESNT_LEAVE_BODY(4)`, i.e. it ignores buckets/screens entirely
(`ConstEnums.h:249-256`).

### 5.3 Cherry bomb self-removal

`DoSpecial` calls `Die()` (`Plant.cpp:4334`), and `Plant::Die()` (`:4946-4979`) is:
```
4957  mDead = true;
4958  RemoveEffects();          // drops reanims/particles
4960-4978  ladder cleanup, flowerpot anim — inert on this deck
```
**There is no fading phase for plants.** `Plant::Die()` is immediate: the plant is marked `mDead` in the same tick
as the explosion. `Plant.h` has **no `mFading` field** (`NOT FOUND`; `grep mFading src/` is empty) — the nearest
thing, `mDisappearCountdown` (`Plant.h:177`, initial **200** at `:135`), is only counted down while
`mState == STATE_DOINGSPECIAL || mSquished` (`:2518-2520`), and a cherry bomb never enters `STATE_DOINGSPECIAL`
(only blover `:4318` and coffee bean `:4410` do). Cherry bomb self-removal therefore costs **0 fading ticks**;
the slot frees on the next `ProcessDeleteQueue` (§6.2).

### 5.4 Potato mine arming (`Plant::PlantInitialize` + `UpdatePotato`, `Plant.cpp:1145-1195`)

| item | value | source |
|---|---|---|
| arming field | `mStateCountdown` | `Plant.h:179` |
| initial value | **1500** | `Plant.cpp:346` (`case SEED_POTATOMINE`, guarded by `IsInPlay()` at `:343`) |
| initial state | `STATE_NOTREADY` | `Plant.cpp:128` (generic init; the enum is `Plant.h:51`) |
| decrement site | `UpdateAbilities` **step 6**, before the seed dispatch | `Plant.cpp:2546-2547` |

State machine, one step max per tick:

1. `:1149-1158` `STATE_NOTREADY` and `mStateCountdown == 0` → particle `PARTICLE_POTATO_MINE_RISE`,
   `PlayBodyReanim("anim_rise", PLAY_ONCE_AND_HOLD, 20, 18.0f)`, **`mState = STATE_POTATO_RISING`**.
   Because step 6 of `UpdateAbilities` already decremented this tick, this happens on **tick 1500**.
   The "spudup" reveal is this transition (the plant becomes visible); there is no separate reveal counter.
2. `:1159-1178` `STATE_POTATO_RISING` and **`aBodyReanim->mLoopCount > 0`** → play `anim_armed`, create the
   `anim_glow` light reanim, `mState = STATE_POTATO_ARMED`, `mBlinkCountdown = 400 + Rand(4000)` (`:1177`, 1 `int` draw).
   **This transition is gated by reanimation data, not a counter** — see `UNKNOWN` §9. It is the only place in the
   whole deck where a gameplay state change depends on animation progress.
3. `:1180-1194` `STATE_POTATO_ARMED`: each tick calls `FindTargetZombie(mRow, WEAPON_PRIMARY)`; if non-null →
   **`DoSpecial()` immediately** (`:1184`). Otherwise it only modulates the glow reanim's `mFrameCount` (`:1191`).

So the mine is triggered by **the plant's own update, not by the zombie**. `Zombie::CheckEatingPlant` does not
exist (`NOT FOUND`; the eat entry point is `Zombie::EatPlant`, `Zombie.cpp:7026`). The mine never appears in a
zombie-side trigger path: `Zombie.cpp:6357` is a *negative* test — a zombie may eat a potato mine **only while
`mState == STATE_NOTREADY`**, and `Zombie::EatPlant` `:7067-7070` returns early (no damage) for any mine in a
later state. Consequence: an armed mine is unkillable by eating, and a buried mine during those first 1500 ticks
is fully eatable (300 HP / 4 per chew).

### 5.5 Potato mine explosion (`Plant::DoSpecial`, `Plant.cpp:4386-4401`)

```cpp
4386  case SeedType::SEED_POTATOMINE:
4388      aPosX = mX + mWidth / 2 - 20;         // ⇒ mX + 20
4389      aPosY = mY + mHeight / 2;             // ⇒ mY + 40
4392      if (mBoard->KillAllZombiesInRadius(mRow, aPosX, aPosY, 60, 0, false, aDamageRangeFlags) >= 1)
4393          ReportAchievement::GiveAchievement(mApp, Spudow, true);
4399      Die();
```

| parameter | value | meaning |
|---|---|---|
| centre | `mX + 20, mY + 40` | **20 px left of the cell centre**, unlike cherry bomb's exact centre |
| radius | **60** | |
| rowRange | **0** | **same row only** — does not damage adjacent rows |
| theBurn | **false** | plain `TakeDamage(1800, 18U)` — no charred phase, no `DropLoot` special-casing |
| `aDamageRangeFlags` | **77** = `0b1001101` | bits `DAMAGES_GROUND(0)`, `DAMAGES_SUBMERGED(2)`, `DAMAGES_DOG(3)`, `DAMAGES_DYING(5)`, `DAMAGES_UNDERGROUND(6)`; **not** `DAMAGES_FLYING(1)`, **not** `DAMAGES_OFF_GROUND(4)` (`ConstEnums.h:257-267`, `Plant.cpp:620-621`) |

The **trigger** geometry and the **kill** geometry are different:
* trigger test (`FindTargetZombie`, `Plant.cpp:4864-4896`): `aAttackRect` is `GetPlantAttackRect(WEAPON_PRIMARY)`
  = `Rect(mX, mY, mWidth - 25, mHeight)` = **`mX .. mX+55`** (`Plant.cpp:5193`); a zombie qualifies when
  `GetRectOverlap(attackRect, zombieRect) >= -aExtraRange`, where `aExtraRange = 30` if `aZombie->mIsEating`
  (`:4883-4886`) else `0` (`:4843`). `GetRectOverlap` (`Board.cpp:8969-8972`) is signed:
  `min(r1.right, r2.right) - max(r1.left, r2.left)`, so a gap of up to 30 px still triggers while the zombie eats.
  Additional exclusions at `:4866-4881`: pogo zombie with `mHasObject`, poltervauler in/pre vault, bungee not
  targeting this column — all `continue` (no trigger). Pole-vaulters specifically get the rect narrowed by
  `mX += 40; mWidth -= 40` (`:4874-4875`, the documented "four pole-vaulters" quirk).
  `:4804-4807` also excludes headless and tangle-kelp-target zombies for mines.
* kill test: the r=60 circle above, applied to *every* zombie in the row that passes the filter — so a zombie that
  satisfied the trigger rect but is beyond 60 px of `(mX+20, mY+40)` is **not** damaged, and a second zombie within
  60 px **is** damaged even though it did not trigger.

Self-death: `Die()` at `:4399`, i.e. immediate `mDead = true` (§5.3). No fading ticks. The mine is consumed even if
`KillAllZombiesInRadius` returned 0 — `DoSpecial` is unconditional after the trigger, but note that the trigger
itself requires a target, so a 0-kill result can only come from the circle/rect mismatch or from
`EffectedByDamage` rejecting the only candidate between the two tests (it cannot, same flags — so in practice
`>= 1` always holds).

---

## 6. G. Death and removal — how fast the slot is reusable

### 6.1 Two independent death paths

| path | test | action | line |
|---|---|---|---|
| plant self-check | `if (mPlantHealth < 0) Die();` | strictly `< 0` | `Plant.cpp:2873-2874` |
| zombie chew | `if (thePlant->mPlantHealth <= 0)` | `<= 0`, and does `mPlantsEaten++; Die();` itself | `Zombie.cpp:7110-7115` |

A plant eaten to exactly 0 dies from the **zombie** path, in the zombie's own update, which runs *after* all plant
updates in the same tick (`Board.cpp:4974` then `:4981`). So on the kill tick: plant updates with health still > 0
(no self-Die), then the zombie chews it to `<= 0` and marks `mDead`. It will **not** shoot/produce again that tick
even if its clock expired, because the plant loop already ran. This ordering is observable and must be reproduced.

### 6.2 Slot reuse = **1 tick**, no fading

```
Plant.cpp:4957        mDead = true;                       (inside Die())
Board.cpp:2269-2283   GetTopPlantAt(...) skips mDead      → cell is legal to replant THIS tick
Board.cpp:8519-8529   ProcessDeleteQueue: if (aPlant->mDead) mPlants.DataArrayFree(aPlant)
LawnApp.cpp:1581-1585 AdvanceLogicTick calls ProcessDeleteQueue BEFORE the board update
```
`ProcessDeleteQueue` sits at the **top of the logic tick** (see `board.md` §1.4). So: `mDead` set on tick `T`
⇒ object freed at the start of tick `T+1` ⇒ the memory slot is recyclable from `T+1`. But the *cell* is reusable
from `T` itself, because every occupancy query (`GetTopPlantAt`, `GetPumpkinAt`, `GetFlowerPotAt`,
`GetPlantsOnLawn`) skips `mDead` entries (`Board.cpp:2131-2132` `GetPumpkinAt`, `:2144-2145` `GetFlowerPotAt`, `:2269-2283` `GetTopPlantAt`).
**Zero fading ticks for plants.** (Contrast zombies, which fade for `mPhaseCounter` ticks; the draft kernel's
`cDeathFadeTicks = 100` at `cuda/SimTick.inc:33` has no plant analogue.)

Does the zombie that ate it notice? Only via `mBoard->mChallenge->ZombieAtePlant(thePlant)`
(`Zombie.cpp:7116`), which is a challenge-mode hook, and `StartEating()` at `:7049` was already called before the
damage. The zombie keeps its `mEatingPlant` target reference until the next `Zombie::Update`; there is no
"target vanished" repath triggered by `Plant::Die()` itself. For the deck this means a zombie chewing a dead
wall-nut wastes the remainder of that tick's chew but is re-evaluated by the normal eat-target search next tick.

### 6.3 `mPlantsEaten` — every increment site, and `mPlantsPlanted` does not exist

`Board::mPlantsEaten` is `uint32_t` (`Board.h:239`), initialised 0 (`Board.cpp:156`). Complete list of writers:

| site | condition | semantics |
|---|---|---|
| `Zombie.cpp:7114` | inside `Zombie::EatPlant`, guarded by `mPlantHealth <= 0` (`:7110`) | **the normal death-by-eating counter.** Also plays `SOUND_GULP`, calls `Die()`, `ZombieAtePlant` |
| `Board.cpp:8992` | `Board::KillAllPlantsInRadius` — every not-dead plant whose `GetPlantRect()` overlaps an r=`theRadius` circle | Gargantuar smash / imp throw. Uses `GetPlantRect` = `Rect(mX+10, mY, mWidth-20, mHeight)` (`Plant.cpp:5169`) |
| `Zombie.cpp:2417` / `:2430` | zombie death (`#ifdef DO_FIX_BUGS` row-scoped vs. global variant): every plant with `mRow == mRow && !NotOnGround()` | `mPlantsEaten++; aPlant->Die();` — the dying-zombie sweep |
| `Zombie.cpp:6492` | `ATTACKTYPE_DRIVE_OVER` skips spiky plants (`:6485`), and `SEED_SPIKEROCK` is excluded (`:6490`) | Zamboni/catapult: `mPlantsEaten++; aPlant->Squish();` — **note `Squish()`, not `Die()`** |
| `Zombie.cpp:7408` | (fourth zombie-side site) | same family |

`Squish()` (`Plant.cpp:2320-2360`) sets `mSquished = true` rather than `mDead`, which is why `mDisappearCountdown`
exists at all: a squished plant keeps ticking (with step 2 of `UpdateAbilities` active, `:2518-2526`) for up to
**200 ticks** before `Die()`. That is the plant analogue of fading, and it is reachable in this deck only against a
gargantuar/imp/zamboni.

**`mPlantsPlanted` is `NOT FOUND` anywhere in `src/`.** The planted-side counters that do exist are
`Board::mPlantsShoveled` (`Board.h`, init `Board.cpp:157`, incremented `Board.cpp:4027` on shovel),
`Board::mMaxSunPlants` (high-water mark, `Board.cpp:2094-2098`), and `SeedPacket::mTimesUsed` (per-packet,
incremented in `WasPlanted`, `SeedPacket.cpp:886`, `:892`). If the exporter wants "plants planted", it must be
derived (e.g. from the action log or as a delta on `mPlants.mSize`); there is no engine field to read.
`src/CudaExport.cpp` does not currently export a planted count.

---

## 7. H. Planting — the path the simulator actually takes

### 7.1 `LawnApp::EnvironmentPlant` is the entry point, not `Board::PlantSeed`

**`Board::PlantSeed` does not exist** (`NOT FOUND`; grep for `PlantSeed` across `src/` returns nothing). The
headless/ROLLOUT path is:

```cpp
LawnApp.cpp:1466  bool LawnApp::EnvironmentPlant(int packetIndex, int col, int row)
1468-1470            bounds: SCENE_PLAYING, 0<=col<MAX_GRID_SIZE_X(9), 0<=row<MAX_GRID_SIZE_Y(6),
                       0<=packetIndex<mSeedBank->mNumPackets      → else false
1472-1474            if (!packet.CanPickUp()) return false;        // active && affordable && requirements met
1475                 seed = (mPacketType==SEED_IMITATER && mImitaterType!=SEED_NONE) ? mImitaterType : mPacketType
1476-1477            if (mBoard->CanPlantAt(col,row,seed) != PLANTING_OK) return false;
1479                 packet.Deactivate();                           // ← sets mRefreshCounter=0, mRefreshing=false
1480-1484            cursor.mCursorType = CURSOR_TYPE_PLANT_FROM_BANK; mType; mImitaterType; mSeedBankIndex
1485                 mBoard->MouseDownWithPlant(GridToPixelX(col,row)+40, GridToPixelY(col,row)+40, 1);
```
`called from main.cpp:290` (`ROLLOUT`/`PLANT` command) and `main.cpp:466`. The `+40` lands mid-cell so
`PlantingPixelToGridX/Y` (`Board.cpp` → `PixelToGridX/Y`) recovers `col,row` exactly:
`GridToPixelX = gridX*80 + LAWN_XMIN(40)` (`Board.cpp:8876`, return at `:8890`; `LAWN_XMIN` `src/GameConstants.h:35`),
`GridToPixelY = gridY*100 + LAWN_YMIN(80)` on a non-pool non-roof stage (`Board.cpp:8909`, return at `:8943`;
`LAWN_YMIN` `src/GameConstants.h:36`). Roof uses `row*85 + slope - 10` (`:8935`) and pool `row*85` (`:8939`).
Note `OffsetYForPlanting` (`PlantingPixelToGridY`) is a no-op for this deck (it only shifts for
`SEED_INSTANT_COFFEE` / cob-cannon-style 2x2 plants).

### 7.2 `Board::CanPlantAt` (`Board.cpp:2705-2917`) — the reachable `PlantingReason`s

`PlantingReason` values (`src/ConstEnums.h:748-762`, in order): `PLANTING_OK`=0, `PLANTING_NOT_HERE`=1,
`PLANTING_ONLY_ON_GRAVES`=2, `PLANTING_ONLY_IN_POOL`=3, `PLANTING_ONLY_ON_GROUND`=4, `PLANTING_NEEDS_POT`=5,
`PLANTING_NOT_ON_ART`=6, `PLANTING_NOT_PASSED_LINE`=7, `PLANTING_NEEDS_UPGRADE`=8, `PLANTING_NOT_ON_GRAVE`=9,
`PLANTING_NOT_ON_CRATER`=10, `PLANTING_NOT_ON_WATER`=11, `PLANTING_NEEDS_GROUND`=12,
`PLANTING_NEEDS_SLEEPING`=13.

Order of tests, with deck/stage reachability:

| # | line | test | result on level 7 day, deck {0,1,2,3,4,5,7} |
|---|---|---|---|
| 1 | `:2707-2710` | `col<0 \|\| col>=9 \|\| row<0 \|\| row>=6` | `PLANTING_NOT_HERE`(1) |
| 2 | `:2712-2716` | `mChallenge->CanPlantAt` (adventure: `PLANTING_OK`) | pass |
| 3 | `:2720-2732` | zen garden | dead |
| 4 | `:2735-2743` | gravebuster | dead |
| 5 | `:2744-2758` | instant coffee | dead |
| 6 | `:2759-2762` | `aHasGrave` → `PLANTING_NOT_ON_GRAVE`(9) unless flying | night-grave stages only; **reachable on rows 11-20/31-40 boards with graves** |
| 7 | `:2776-2779` | crater → `PLANTING_NOT_ON_CRATER`(10) | reachable (doom shroom / jackbox) |
| 8 | `:2780-2783` | scary pot / ice → `PLANTING_NOT_HERE`(1) | reachable (ScaryPotter, freeze levels) |
| 9 | `:2784-2788` | `GRIDSQUARE_DIRT`/`NONE` → `PLANTING_NOT_HERE`(1) | **row 5 is dirt on grass** → this is why `playableRows = 5` in `CudaExport.cpp:86` |
| 10 | `:2790-2798` | lilypad/tanglekelp/seashroom pool rule | dead for this deck |
| 11 | `:2799-2802` | flying | dead |
| 12 | `:2803-2809` | spikeweed/spikerock ground rule | dead |
| 13 | `:2812-2818` | pool without lilypad → `PLANTING_NOT_ON_WATER`(11) | reachable on pool stages |
| 14 | `:2819-2822` | flowerpot | dead |
| 15 | `:2823-2826` | `StageHasRoof() && !aHasFlowerPot` → `PLANTING_NEEDS_POT`(5) | reachable on roof stages |
| 16 | `:2828-2846` | pumpkin shell | dead |
| 17 | `:2847-2850` | **`aHasLilypad && theSeedType == SEED_POTATOMINE` → `PLANTING_ONLY_ON_GROUND`(4)** | **the only deck-specific rule besides occupancy** — reachable on pool stages |
| 18 | `:2852-2876` | upgrade / under-plant handling | dead (no upgrade plants in deck) |
| 19 | `:2878-2900` | `aNormalPlant` occupied: upgrade check, `IsUpgrade`, wall-nut first-aid, else → `PLANTING_NOT_HERE`(1) | **the occupancy rule the kernel needs.** First-aid (`:2890-2897`) requires the STORE_ITEM_FIRSTAID purchase and re-planting the *same* wallnut/tallnut with health `< max*2/3` — it `Die()`s the old one and plants fresh (`:3810-3818`) |
| 20 | `:2903-2914` | `IsUpgrade` without cheat; cob/cattail spot | dead |
| 21 | `:2916` | otherwise | **`PLANTING_OK`(0)** |

So for the deck on a day-lawn the *only* non-OK reasons a rollout can observe are `PLANTING_NOT_HERE` (out of
bounds, occupied cell, dirt row 5) plus `PLANTING_NOT_ON_GRAVE`/`PLANTING_NOT_ON_CRATER` if the task family's
levels contain them. The kernel's legality mask must be `col in 0..8 && row in 0..4 && cell empty` for level 7, and
must use the row-band from `CanPlantAt`'s bounds test (`row < 6`, **not** `row < 5`) — row 5 is rejected only by
grid-square type (`:2784-2788`), which is stage data, not a constant.

### 7.3 Cost, refresh, and the fields set on the fresh `Plant`

Cost is deducted inside `Board::MouseDownWithPlant` (`Board.cpp:3602`), before the plant exists:
```
3791-3795  if (!mApp->mEasyPlantingCheat && cursor==PLANT_FROM_BANK && !HasConveyorBeltSeedBank())
               if (!TakeSunMoney(GetCurrentPlantCost(aPlantingSeedType, SEED_NONE))) return;
```
`Board::TakeSunMoney` (`Board.cpp:8501-8512`) returns false and sets `mOutOfMoneyCounter = 70` (`:8510`) when
unaffordable. `GetCurrentPlantCost` (`Board.cpp:9406`) wraps `Plant::GetCost` and applies
`PlantUsesAcceleratedPricing` (`Board.cpp:9401`). **`CanPickUp()` already verified affordability**
(`SeedPacket.cpp:712`), and `EnvironmentPlant` checked it at `:1473`, so the `TakeSunMoney` failure path is
unreachable from the env entry — but note `EnvironmentPlant` **skips `SeedPacket::MouseDown`'s pick-up entirely**
and calls `packet.Deactivate()` + `MouseDownWithPlant` directly.
Deduction is `mSunMoney -= cost` with `cost` = `gPlantDefs[].mSeedCost` (§0) for this deck.

Refresh side effects — `SeedPacket::WasPlanted()` (`SeedPacket.cpp:872-896`), called from `Board.cpp:3876`
only for `CURSOR_TYPE_PLANT_FROM_BANK`:
```
892   mTimesUsed++;
893   mRefreshing = true;
894   mRefreshTime = Plant::GetRefreshTime(mPacketType, mImitaterType);   // 750 / 3000 / 5000 for this deck
```
`mRefreshCounter` is **not** reset here — it was already zeroed by `Deactivate()` (`SeedPacket.cpp:142-143`,
called from `EnvironmentPlant` `LawnApp.cpp:1479`). The countdown then advances only in
`SeedPacket::Update` (`:160`) under `if (!mActive && mRefreshing)` (`:172`):
```
174   mRefreshCounter++;
175   if (mRefreshCounter > mRefreshTime)  { 177 mRefreshCounter=0; 178 mRefreshing=false; 179 Activate(); }
```
⇒ **the packet is ready again exactly `mRefreshTime + 1` ticks after planting** (strict `>`), and the packet
update runs at the **end** of `UpdateGameObjects` (`Board.cpp:5012-5015`), i.e. after plants/zombies/projectiles.
`SeedPacket::Update` early-returns unless `mGameScene == SCENE_PLAYING` (`:162`), and does nothing on
`SEED_NONE` packets. Note `SetPacketType` (`SeedPacket.cpp:1107-1145`) applies a *different* initial cooldown for
`mRefreshTime == 5000` → `3500`, and `3000` → `2000`, at deal time (`:1127-1144`), but the guards exclude
adventure levels 1-10 patterns… actually those special cases only fire for the listed game modes at `:1122-1125`;
for plain adventure the packet is left fully active at start. **Do not confuse the deal-time 3500/2000 with the
post-plant refresh, which always uses the raw `mRefreshTime` (750/3000/5000).**

Fresh `Plant` field values at the end of `Board::AddPlant` (`Plant.cpp:108-501` via `Board::NewPlant`,
`Board.cpp:2036-2042`), for this deck:

| field | value | line |
|---|---|---|
| `mIsOnBoard` | **`true`, set BEFORE `PlantInitialize`** | `Board.cpp:2039` |
| `mPlantCol` / `mRow` | the grid coords | `Plant.cpp:110-111` |
| `mX` / `mY` | `GridToPixelX/Y(col,row)` | `:114-115` |
| `mWidth` / `mHeight` | 80 / 80 | `:149-150` |
| `mAnimCounter` | 0 | `:117` |
| `mAnimPing` | true | `:118` |
| `mFrame` | 0 | `:119` |
| `mShootingCounter` | **0** (idle; `UpdateShooting` returns at `:3207`) | `:120` |
| `mShakeOffsetX/Y` | 0.0f | `:121-122` |
| `mFrameLength` | `RandRangeInt(12,18)` = 12..18 | `:123` |
| `mTargetX` / `mTargetY` | -1 / -1 | `:124-125` |
| `mStartRow` | `= mRow` | `:126` |
| `mNumFrames` | 5 | `:127` |
| `mState` | **`STATE_NOTREADY` (0)** | `:128` |
| `mDead` | false | `:129` |
| `mSquished` | false | `:130` |
| `mPlantHealth` | 300, or **4000 for wall-nut** | `:133`, `:288-289` |
| `mDoSpecialCountdown` | 0, or **100 for cherry bomb** | `:134`, `:327` |
| `mDisappearCountdown` | **200** | `:135` |
| `mStateCountdown` | 0, or **1500 for potato mine** | `:136`, `:346` |
| `mLaunchCounter` | shooters `RandRangeInt(0,150)`; sunflower `RandRangeInt(300,1250)`; **0 for cherry/wall/potato** | `:189-197` |
| `mLaunchRate` | `gPlantDefs[].mLaunchRate` (150 / 2500 / 0) | `:157` |
| `mSubclass` | `gPlantDefs[].mSubClass` (1 for the three shooters, else 0) | `:158` |
| `mPlantMaxHealth` | `= mPlantHealth`, after the switch | `:492` |
| `mBlinkCountdown` | `400 + Rand(400)`; wall-nut **re-set** to `1000 + Rand(1000)` | `:183`, `:290` |
| `mRecentlyEatenCountdown`, `mEatenFlashCountdown`, `mBeghouledFlashCountdown` | 0 | `:146-148` |
| `mWakeUpCounter`, `mIsAsleep`, `mOnBungeeState`, `mPottedPlantIndex` | 0, false, `NOT_ON_BUNGEE`, -1 | `:153-156` |
| `mTargetZombieID` | **not initialised here** — see `UNKNOWN` §9 | — |
| `mRenderOrder` | `CalcRenderOrder()` | `:159`, `:503` |

`Board::AddPlant` additionally, after `NewPlant`: `DoPlantingEffects` (`Board.cpp:2044-2086` — a suppressed foley call plus a
`PARTICLE_PLANTING` particle at `GridToPixelX+41, GridToPixelY+74`, and **that particle does consume RNG** —
see §1.6), `mChallenge->PlantAdded` (`:2092`), `mMaxSunPlants` update (`:2094-2098`), and
`mPeaShooterUsed = true` for peashooter/snowpea/repeater (`:2100-2108`) — the latter gates an achievement only.
Planting **never** runs a plant's `Update()` in the same tick: `EnvironmentPlant` is called from the ROLLOUT
command handler between logic ticks (`main.cpp:466`), and the plant loop already ran or has not yet started.

---

## 8. Divergences between this spec and the current kernel draft

The draft in `cuda/SimTick.inc` is right about the clock free-running (§2.1 fact 1) and wrong about these.
All of them are aggregate-visible.

| # | draft | original engine | cite |
|---|---|---|---|
| 1 | `cPotatoArmTicks = 500` | **1500** | `SimTick.inc:43` vs `Plant.cpp:346` |
| 2 | cherry bomb: `abs(zombieX - plantX) > cCherryRadius` (+ `cCherryRowGap` band hack) | clamped **circle-vs-rect** test `GetCircleRectOverlap(x, y, 115, rect)`, `<=` inclusive, in integer arithmetic, applied *on top of* the `rowDist in [-1,1]` test | `SimTick.inc:236-243` vs `Board.cpp:9338`, `:8975-8983` |
| 3 | cherry bomb: flat `cCherryDamage = 1800` + `ZB_BURNED` | `theBurn == true` → `Zombie::ApplyBurn()`, which is `TakeDamage(1800,18U)` **only when `mBodyHealth >= 1800` or boss**, else: a walking ground zombie gets a charred-reanim kill via `DieWithLoot()` (`:8791-8828`, **1 float draw at `:8792`**), and only the flying/headless/bungee/yeti/zombotany/bobsled set reaches `PHASE_ZOMBIE_BURNED` + `mPhaseCounter = 300` (`:8740-8743`) | `SimTick.inc:244-245` vs `Zombie.cpp:8687-8695`, `:8731`, `:8740-8743`, `:8792`, `:8828` |
| 4 | cherry bomb centre = plant x | `mX + 40, mY + 40` (cell centre) | `Plant.cpp:4308-4309` |
| 5 | potato mine: triggered by an x-window `cPotatoRadius = 60` from `plantX`, and it damages the first zombie found | trigger = signed `GetPlantAttackRect` overlap `Rect(mX, mY, 55, 80)` with `+30` slack while eating, plus pogo/vaulter/bungee/headless exclusions; kill = circle r=60 about **`mX+20, mY+40`** hitting **all** zombies in row; the `>= 1` achievement test is separate | `Plant.cpp:5193`, `:4864-4896`, `:4388-4392` |
| 6 | potato mine arms and triggers in the same branch, no `STATE_POTATO_ARMED` | three states; **armed only after the `anim_rise` loop completes**; a `STATE_NOTREADY` mine is eatable, an armed one is not | `Plant.cpp:1149-1194`, `Zombie.cpp:6357`, `:7067-7070` |
| 7 | shooters: `AddProjectile` on the **same tick** the clock hits 0, and the repeater's 2nd pea likewise | the shot is deferred by `mShootingCounter`: **+34** ticks (peashooter/snowpea from 35, `:758`) / **+25** ticks (repeater from 26, `:762`). Both repeater decisions set 26, so pea1 lands at `T+25` and pea2 at `T+R` (`R = 150-Rand(15)`), and the *visible* pair is pea2-of-N with pea1-of-N+1, 25 ticks apart (§2.6) | `SimTick.inc:201-213` vs `Plant.cpp:3205-3308`, §2.3, §2.6 |
| 8 | shooters: no initial phase draw modelled as `RandRangeInt(0, mLaunchRate)` — verify the inclusive upper bound | `Rand(151) + 0` ⇒ 0..**150** inclusive; the reset is `150 - Rand(15)` ⇒ 136..150 | `Plant.cpp:194`, `:950` |
| 9 | sunflower redraw `rand.Range(launchRate - cProduceRedrawJitter, launchRate)` | `RandRangeInt(2350, 2500)` = **inclusive both ends** (`Rand(151)+2350`) | `SimTick.inc:224` vs `Plant.cpp:1034`, `src/PvzpLib/PvzpCommon.cpp:384-387` |
| 10 | no cherry-bomb shake draws | **2 float RNG draws per tick for 100 ticks**, from the shared global stream | `SimTick.inc` (absent) vs `Plant.cpp:3422-3426` |
| 11 | `lane.i(F_PLANT_TYPE, slot) = -1;` immediately on detonation | `mDead = true` this tick; object freed at the start of the next tick; the **cell** is reusable this tick already | `SimTick.inc:246` vs `Plant.cpp:4957`, `Board.cpp:8519-8529`, `LawnApp.cpp:1581-1585` |
| 12 | pea origin `plantX` (`F_PLANT_X`) | `mX + aOffsetX + 24, mY + aOffsetY - 33` with `aOffset*` from the animated reanim transform | `SimTick.inc:204` vs `Plant.cpp:4603-4608` |
| 13 | `cChilledTicks = 400` (zombie-side, listed for completeness) | `mChilledCounter = max(1000, existing)`; 2000 for the ice-shroom trap | `SimTick.inc:31` vs `Zombie.cpp:7532-7537` |
| 14 | plant death: health `<= 0` checked in the plant loop | plants self-Die only at `< 0`; `<= 0` is handled by the **zombie** update, which runs later in the same tick | `SimTick.inc` vs `Plant.cpp:2873`, `Zombie.cpp:7110` |
| 15 | no particle-emission RNG tax anywhere in the kernel | every emitted particle system/particle draws 17-19 + 24-28 floats from the global stream, and nothing in the particle layer is env-skipped | `SimTick.inc` (absent) vs `src/PvzpLib/PvzpParticle.cpp:296-310`, `:334-357` — see §1.6 |
| 16 | blink housekeeping absent (`mBlinkCountdown` not in the SoA) | blink is a per-plant recurring int draw (`:3017`, nuts `:3041` + `:2902`) even though the visual is presentation-only | `SimTick.inc:186-270` vs `Plant.cpp:3015-3044`, `:3081-3086` |

---

## 9. `UNKNOWN` — what could not be confirmed from the repo

* `UNKNOWN:` the exact tick count of `STATE_POTATO_RISING → STATE_POTATO_ARMED`. The gate is
  `aBodyReanim->mLoopCount > 0` (`Plant.cpp:1161`) on the `anim_rise` track played once at rate 18.0f
  (`:1154`). `mLoopCount` increments when a reanim passes its last frame
  (`src/PvzpLib/Reanimator.cpp:441`, `:475`), so the delay is `ceil(frameCount / (18/fps))` ticks — but **the frame count lives
  in the `.reanim` data file, which is not checked into this repo** (no `*.reanim`, no `main.pak` present).
  Everything else in the deck is counter-determined; this is the one animation-dependent gameplay edge.
* `UNKNOWN:` whether the peashooter/snowpea/repeater **head** reanimation actually has an `anim_shooting` track
  (`Plant.cpp:751`). The head reanim is definitely created (`:227-231`), the track name is used at `:756`
  and in `UpdateShooting`'s reset (`:3395`), but the track list itself is `.reanim` data (same reason as above).
  If the head lacked it, the body branch (`:785`, `mShootingCounter = 29` via the `default` at `:811`) would apply
  instead → a 28-tick deferral rather than 34. The three-way alternatives are all listed in §2.2 so the kernel can
  pick by measurement rather than by guess.
* `UNKNOWN:` the literal `aOffsetX / aOffsetY` from `GetPeaHeadOffset` (`Plant.cpp:3703-3704`). These are the
  `anim_stem` track's live transform, i.e. per-tick animation data from `.reanim`. Only the static part
  (`+24, -33` at `:4607-4608`) and the no-reanim fallback `0,0` (`:3686-3688`) are readable in source.
* `UNKNOWN:` whether any deck reanim has a non-zero `mShakeOverride` track, which would add
  2 float draws per such track per tick (`src/PvzpLib/Reanimator.cpp:508-511`). Same data-file reason. If the CPU reference and
  the kernel diverge only after a cherry bomb is planted, this is the second candidate after `Plant.cpp:3424-3425`.
* `UNKNOWN:` the exact **number** of RNG draws per particle emission. That particles consume RNG at all is
  confirmed (`src/PvzpLib/PvzpParticle.cpp:296-310` per emitter, `:334-357`/`:371`/`:382`/`:391-392`/`:435-436` per
  particle — see §1.6); what cannot be confirmed is `mSystemFields.count`, `mParticleFields.count`, the emitter count
  per `ParticleEffect`, `PARTICLE_RANDOM_START_TIME` and the emitter type, because those live in the parsed
  `.particle` definitions (`gParticleDefArray`, loaded at `PvzpParticle.cpp:206-220`) and the data files are not in
  the repo. Bounded estimate: 17-19 floats per emitter instantiated, 24-28 per spawned particle.
* `UNKNOWN:` `mTargetZombieID` is never assigned in `PlantInitialize`; the only writers found are
  `Plant.cpp:1229` (tangle-kelp) and the chomper paths in `Plant.cpp`. For this deck it is read by nothing, so it is
  safe to omit from the kernel — but its initial value is whatever `GameObject`'s constructor gives, which was not
  traced.
* `NOT FOUND:` `mFading` (any plant field of that name) — `grep mFading src/` returns nothing. The concept the
  question refers to is `mDisappearCountdown` (§5.3).
* `NOT FOUND:` `Board::PlantSeed`, `Zombie::CheckEatingPlant`, `mPlantsPlanted`, `Plant::mProductCounter`,
  `Board::CheckWinLose` (also recorded in `board.md` §4.1).

---

## 勘误确认（2026-10-18）

外部审计就第 7 关 + `[PEASHOOTER, SUNFLOWER, CHERRYBOMB, WALLNUT, POTATOMINE, SNOWPEA, REPEATER]` 卡组提出 7 条
疑似勘误。逐条回到 `src/` 复核，判决汇总如下（**CONFIRMED** = 断言成立；**WRONG** = 断言不成立，给出正确值）：

| # | 审计断言 | 判决 |
|---|---|---|
| 1 | 土豆雷 1500 tick 武装；转移写作 `STATE_NOTREADY → STATE_READY` | 数值 **CONFIRMED**（1500）；状态机 **WRONG**（三段，且终点不是 `STATE_READY`） |
| 2 | 触发矩形 `Rect(mX, mY, mWidth-25, mHeight)` | **CONFIRMED**；但"门控 = 矩形"不完整（另有 4 条 `continue` 与 ±30/−40 容差） |
| 3 | 樱桃：引信长度 / 半径 / 伤害 / 邻行取舍 | 引信 **CONFIRMED**（100）；半径 **CONFIRMED**（115）；邻行 **CONFIRMED 含入**（rowRange=1）；"伤害 1800" **WRONG**（`ApplyBurn()`，只有 `mBodyHealth >= 1800` 才是 1800） |
| 4 | 太阳花首次 `RandRangeInt(300,1250)`、其后 `RandRangeInt(2350,2500)`、值 25、产物是 Coin | **全部 CONFIRMED** |
| 5 | `mLaunchCounter--` / 重置 `mLaunchRate - Rand(15)` / `==25` 第二发；`mLaunchRate`；`gPlantDefs.mDamage`；chill 1000 + 行走减半 | 时钟 **CONFIRMED**；`mLaunchRate` **CONFIRMED**（150/150/150）；**`PlantDefinition` 没有 `mDamage` 字段**；chill 1000 **CONFIRMED**；**"行走减半" WRONG**（×0.4，减半的是动画速率） |
| 6 | 种植血量 / 延迟击发 +34 与 +25 / 死亡判定 / `mPlantsEaten` | **全部 CONFIRMED**；死亡判定是**两套**（植物侧 `< 0`，僵尸侧 `<= 0`） |
| 7 | 必须携带的 tick 状态 | **CONFIRMED**；但"纯装饰所以可跳"只对**位置与伤害**成立，对 **RNG 流不成立**（见 §7.2） |

顺带发现本文件已有的一处笔误（不是审计提出的，但同属伤害标志位，一并订正）：
**§5.2 末行** 把 `TakeDamage` 的 flag `18U` 写成 `DAMAGE_BYPASSES_SHIELD(0) | DAMAGE_DOESNT_LEAVE_BODY(4)`，
**正确解码是 `DAMAGE_HITS_SHIELD_AND_BODY(1) | DAMAGE_DOESNT_LEAVE_BODY(4)`**（`ConstEnums.h:248-256`，
`18 = 0b010010`，bit0 并**未**置位）。语义差别很大：`DAMAGE_BYPASSES_SHIELD` 才会在
`Zombie::TakeDamage` `Zombie.cpp:7984` 处跳过盾牌，而 18 没有它，于是盾牌先受击、再由
`DAMAGE_HITS_SHIELD_AND_BODY` 在 `:7987-7990` 把 `aDamageRemaining` 重新设为完整的 `theDamage`
—— 即"盾牌挨一遍、甲与身体再挨全额 1800"，**不是**"无视桶/门"。
同一段里 §5.5 把 77 解码成"bits 0,2,3,**5**,6"也是错的：`77 = 0b1001101` ⇒ bits **0,2,3,6**，
**不含 `DAMAGES_DYING(5)`**（含它的话应为 109）。这条对土豆雷是实质结论：`Zombie::EffectedByDamage`
的第一道测试就是 `DAMAGES_DYING`（`Zombie.cpp:8086-8089`），77 没有该位 ⇒ **土豆雷打不到濒死僵尸**，
而樱桃的 127 有该位 ⇒ 打得到。§5.2 正文"127 ⇒ 不过滤任何东西"仍然对。

### 1. 土豆雷武装时长与状态机

**数值 CONFIRMED，状态机 WRONG。**

字面量确实 1500，且它在 `PlantInitialize` 里、被 `IsInPlay()` 门控：

```cpp
// src/Lawn/Plant.cpp:337-347
	case SeedType::SEED_POTATOMINE:
	{
		PVZP_ASSERT(aBodyReanim);

		aBodyReanim->mAnimRate = 12.0f;

		if (IsInPlay())
		{
			aBodyReanim->AssignRenderGroupToTrack("anim_glow", RENDER_GROUP_HIDDEN);
			mStateCountdown = 1500;
		}
```

`Plant.cpp:346` 就是那个 1500。默认 `mState` 是 `STATE_NOTREADY`（`Plant.cpp:128`），
`mStateCountdown` 每 tick 在 `Plant::UpdateAbilities` **step 6**（种子分派之前）减一
（`Plant.cpp:2546-2547`）。审计写的 `500` 是 `cuda/SimTick.inc:43` 的 `cPotatoArmTicks`（已登记在 §8 行 1）。

**状态机不是 `NOTREADY → READY`。** 完整转移在 `Plant::UpdatePotato`：

```cpp
// src/Lawn/Plant.cpp:1145-1195（逐字全文）
void Plant::UpdatePotato()
{
	Reanimation* aBodyReanim = mApp->ReanimationGet(mBodyReanimID);

	if (mState == PlantState::STATE_NOTREADY)
	{
		if (mStateCountdown == 0)
		{
			mApp->AddPvzpParticle(mX + mWidth / 2, mY + mHeight / 2, mRenderOrder, ParticleEffect::PARTICLE_POTATO_MINE_RISE);
			PlayBodyReanim("anim_rise", ReanimLoopType::REANIM_PLAY_ONCE_AND_HOLD, 20, 18.0f);
			mState = PlantState::STATE_POTATO_RISING;
			mApp->PlayFoley(FoleyType::FOLEY_DIRT_RISE);
		}
	}
	else if (mState == PlantState::STATE_POTATO_RISING)
	{
		if (aBodyReanim->mLoopCount > 0)
		{
			float aRate = RandRangeFloat(12.0f, 15.0f);
			PlayBodyReanim("anim_armed", ReanimLoopType::REANIM_LOOP, 0, aRate);

			Reanimation* aLightReanim = mApp->AddReanimation(0.0f, 0.0f, mRenderOrder + 2, GetPlantDefinition(mSeedType).mReanimationType);
			aLightReanim->mLoopType = ReanimLoopType::REANIM_LOOP;
			aLightReanim->mAnimRate = aRate - 2.0f;
			aLightReanim->SetFramesForLayer("anim_glow");
			aLightReanim->mFrameCount = 10;
			aLightReanim->ShowOnlyTrack("anim_glow");
			aLightReanim->SetTruncateDisappearingFrames("anim_glow", false);
			mLightReanimID = mApp->ReanimationGetID(aLightReanim);
			aLightReanim->AttachToAnotherReanimation(aBodyReanim, "anim_light");

			mState = PlantState::STATE_POTATO_ARMED;
			mBlinkCountdown = 400 + Sexy::Rand(4000);
		}
	}
	else if (mState == PlantState::STATE_POTATO_ARMED)
	{
		if (FindTargetZombie(mRow, PlantWeapon::WEAPON_PRIMARY))
		{
			DoSpecial();
		}
		else
		{
			Reanimation* aLightReanim = mApp->ReanimationTryToGet(mLightReanimID);
			if (aLightReanim)
			{
				aLightReanim->mFrameCount = PvzpAnimateCurve(200, 50, DistanceToClosestZombie(), 10, 3, PvzpCurves::CURVE_LINEAR);
			}
		}
	}
}
```

三个必须记住的点：

1. `STATE_READY` 对本卡组的**任何**植物都不会被写入。全 `src/Lawn/Plant.cpp` 里
   `mState = PlantState::STATE_READY` 共 **6 处**：`:410`（chomper 初值）、`:440`（scaredy shroom 初值）、
   `:1364`（scaredy）、`:1835`（chomper）、`:2027`（magnet）、`:2282`（gold magnet）。
   土豆雷的终点是 `STATE_POTATO_ARMED`（枚举 `Plant.h:66-67`，排在 `STATE_READY` `Plant.h:52` 之后）。
2. **1500 tick 只是"出土"，不是"武装"**。`NOTREADY → RISING` 在 `:1149-1158`（`Plant.cpp:1155` 写状态），
   `RISING → ARMED` 的闸门是 `aBodyReanim->mLoopCount > 0`（`Plant.cpp:1161`）——
   这是动画数据（`anim_rise` @ rate 18.0f，`:1154`），不是计数器，时长无法从源码读出（见 §9 `UNKNOWN` 第 1 条）。
   内核里"1500 tick 后可爆"是**偏早**的，真实值 = `1500 + anim_rise 一整轮`。
3. 只有 `STATE_POTATO_ARMED` 才每 tick 调 `FindTargetZombie` 并 `DoSpecial()`（`Plant.cpp:1182-1184`）。
   `STATE_NOTREADY` 期间的雷**可以被僵尸吃掉**，一旦离开 `NOTREADY` 就吃不动：
   ```cpp
   // src/Lawn/Zombie.cpp:7067-7070（Zombie::EatPlant 内）
   if (thePlant->mSeedType == SeedType::SEED_POTATOMINE && thePlant->mState != PlantState::STATE_NOTREADY)
   {
       return;
   }
   ```
   另有 `Zombie.cpp:6357` 的负面测试（掘土地雷只吃 `NOTREADY` 的雷）。

### 2. 土豆雷的触发门控、伤害与自毁

**矩形 CONFIRMED（`Plant.cpp:5193`），但"唯一矩形测试"这个描述 WRONG。**

触发矩形字面量：

```cpp
// src/Lawn/Plant.cpp:5175-5205（GetPlantAttackRect 的相关分支）
Rect Plant::GetPlantAttackRect(PlantWeapon thePlantWeapon)
{
	Rect aRect;
	if (mApp->IsWallnutBowlingLevel())            // 第 7 关死分支
	{
		aRect = Rect(mX, mY, mWidth - 20, mHeight);
	}
	else if (thePlantWeapon == PlantWeapon::WEAPON_SECONDARY && mSeedType == SeedType::SEED_SPLITPEA)
	{
		aRect = Rect(0, mY, mX + 16, mHeight);   // 死分支
	}
	else switch (mSeedType)
	{
	...
	case SeedType::SEED_POTATOMINE:     aRect = Rect(mX,            mY,             mWidth - 25,        mHeight);               break;
```

`mWidth == mHeight == 80`（`Plant.cpp:149-150`），所以是 `Rect(mX, mY, 55, 80)` ⇒
**世界 x 区间 `[mX, mX+55]`**，比格子（80 px）窄 25 px、且**不**像 `GetPlantRect()` 那样右移 10
（`Plant.cpp:5169` 的通用格是 `Rect(mX+10, mY, mWidth-20, mHeight)`）。审计写的 `mWidth-25` 是对的。

完整门控在 `Plant::FindTargetZombie`（`Plant.cpp:4785-4920`），土豆雷相关的判定顺序如下（逐字）：

```cpp
// src/Lawn/Plant.cpp:4787-4788   先取 flags 与矩形
	int aDamageRangeFlags = GetDamageRangeFlags(thePlantWeapon);   // ⇒ 77，Plant.cpp:620-621
	Rect aAttackRect = GetPlantAttackRect(thePlantWeapon);         // ⇒ Rect(mX, mY, 55, 80)
// :4794-4795   if (aZombie->mDead) continue;
// :4802-4808   if (!aZombie->mHasHead || aZombie->IsTangleKelpTarget())
//                  if (mSeedType == SEED_POTATOMINE || CHOMPER || TANGLEKELP) continue;   // 无头/海带目标不触雷
// :4835-4838   else if (aRowDeviation) continue;                                          // 严格同行
// :4841        if (aZombie->EffectedByDamage(aDamageRangeFlags))                          // 77 的解码见本节末
// :4864-4887   土豆雷专属：
			if (mSeedType == SeedType::SEED_POTATOMINE)
			{
				if ((aZombie->mZombieType == ZombieType::ZOMBIE_POGO && aZombie->mHasObject) ||
					aZombie->mZombiePhase == ZombiePhase::PHASE_POLEVAULTER_IN_VAULT || aZombie->mZombiePhase == ZombiePhase::PHASE_POLEVAULTER_PRE_VAULT)
				{
					continue;
				}

				if (aZombie->mZombieType == ZombieType::ZOMBIE_POLEVAULTER)
				{
					aAttackRect.mX += 40;
					aAttackRect.mWidth -= 40;  // classic potato mine quirk; lets pole-vaulters set off the mine ("four pole-vaulters" trick)
				}

				if (aZombie->mZombieType == ZombieType::ZOMBIE_BUNGEE && aZombie->mTargetCol != mPlantCol)
				{
					continue;
				}

				if (aZombie->mIsEating)
				{
					aExtraRange = 30;
				}
			}
// :4895-4899   几何比较（注意是「小于 -aExtraRange 才排除」）：
			Rect aZombieRect = aZombie->GetZombieRect();
			if (!needPortalCheck && GetRectOverlap(aAttackRect, aZombieRect) < -aExtraRange)
			{
				continue;
			}
```

`GetRectOverlap` 是**带符号**的一维重叠量（`Board.cpp:8969-8972`）：
`min(r1.right, r2.right) - max(r1.left, r2.left)`。所以触发条件是

* 一般僵尸：`overlap >= 0` —— 矩形必须真的重叠（差 1 px 都不炸）。
* **正在啃食的僵尸**（`mIsEating`）：`overlap >= -30` —— 允许 30 px 的间隙。
* **撑杆僵尸**：矩形被就地收窄成 `Rect(mX+40, mY, 15, 80)`（`:4874-4875`），再套上面的规则。

默认碰撞盒 `Rect(36, 0, 42, 115)`（`Zombie.cpp:201`，NORMAL / FLAG / CONE / POLEVAULTER 都用它），
经 `GetZombieRect()` 平移 `+mX, +mY`（`Zombie.cpp:8311`），所以这是纯 x 区间算术，没有圆。

关于审计列的三个谓词：

* `EffectedByDamage` **有**，flag = **77**（`Plant.cpp:620-621`）。
* `IsDeadOrDying` **不是显式调用**，而是经由 77 缺少 `DAMAGES_DYING(5)` 位间接生效
  （`Zombie.cpp:8086-8089`：没有该位就先 `IsDeadOrDying()` 早退）。另外 `:4802` 的 `!mHasHead`
  显式排除掉无头目标，`Zombie.cpp:8138-8152` 还要求非离地相位（撑杆在空中、蹦床、跳跃等）除非有
  `DAMAGES_OFF_GROUND(4)` —— 77 也没有该位。
* `IsOnBoard` **不被检查**（`FindTargetZombie` 与 `KillAllZombiesInRadius` 都不调它）。
  越界过滤来自 `Zombie::EffectedByDamage` 的另一条：`Zombie.cpp:8154-8157`
  `if (mZombieType != ZOMBIE_BOBSLED && GetZombieRect().mX > WIDE_BOARD_WIDTH) return false;`
  （`WIDE_BOARD_WIDTH = 800`，`GameConstants.h:30`）。

伤害与半径（`Plant::DoSpecial` 逐字）：

```cpp
// src/Lawn/Plant.cpp:4386-4401
	case SeedType::SEED_POTATOMINE:
	{
		aPosX = mX + mWidth / 2 - 20;
		aPosY = mY + mHeight / 2;

		mApp->PlaySample(SOUND_POTATO_MINE);
		if (mBoard->KillAllZombiesInRadius(mRow, aPosX, aPosY, 60, 0, false, aDamageRangeFlags) >= 1)
			ReportAchievement::GiveAchievement(mApp, Spudow, true);

		int aRenderPosition = Board::MakeRenderOrder(RenderLayer::RENDER_LAYER_PARTICLE, mRow, 0);
		mApp->AddPvzpParticle(aPosX + 20.0f, aPosY, aRenderPosition, ParticleEffect::PARTICLE_POTATO_MINE);
		mBoard->ShakeBoard(3, -4);

		Die();
		break;
	}
```

⇒ 圆心 `(mX+20, mY+40)`（**比格子中心偏左 20 px**）、半径 **60**、`theRowRange = 0`（**只本行**）、
`theBurn = false`，于是走 `Board.cpp:9344-9347` 的 `aZombie->TakeDamage(1800, 18U)`。
**是的，它一定自毁**：`Die()` 在 `Plant.cpp:4399`，无条件、在同 tick（`Plant::Die` 只写 `mDead = true`，
`:4957`，没有淡出，§5.3 / §6.2）。

⚠ **触发几何 ≠ 杀伤几何**：触发用 x 区间 `[mX, mX+55]`（撑杆时 `[mX+40, mX+55]`），
杀伤用 `(mX+20, mY+40)` 半径 60 的**圆-矩形**测试（`Board.cpp:9338` + `:8975-8982`）。
两者都各自遍历整行的僵尸，**引爆的那只可能超出 60 px 而不受伤，另一只没引爆但进了圆也会受 1800**。

### 3. 樱桃炸弹：引信、伤害、爆炸几何

**引信 CONFIRMED = 100 tick；半径 CONFIRMED = 115 px；邻行 CONFIRMED = 含入；伤害 WRONG。**

字面量（`PlantInitialize`，与土豆雷同一个 `switch`）：

```cpp
// src/Lawn/Plant.cpp:320-336
	case SeedType::SEED_CHERRYBOMB:
	case SeedType::SEED_JALAPENO:
	{
		PVZP_ASSERT(aBodyReanim);

		if (IsInPlay())
		{
			mDoSpecialCountdown = 100;

			aBodyReanim->SetFramesForLayer("anim_explode");
			aBodyReanim->mLoopType = ReanimLoopType::REANIM_PLAY_ONCE_AND_HOLD;

			mApp->PlayFoley(FoleyType::FOLEY_REVERSE_EXPLOSION);
		}

		break;
	}
```

引信**没有** `Plant::UpdateCherryBomb` 这样的函数（`NOT FOUND`；`grep "UpdateCherryBomb" src/` 为空）。
它由 `Plant::UpdateAbilities` 的最后一步驱动（逐字）：

```cpp
// src/Lawn/Plant.cpp:2582-2589
	if (mDoSpecialCountdown > 0)
	{
		mDoSpecialCountdown--;
		if (mDoSpecialCountdown == 0)
		{
			DoSpecial();
		}
	}
```

⇒ 种植后第 **100** tick 起爆（tick1 `100→99` … tick100 `1→0` 并立刻 `DoSpecial`）。
`Plant::DoSpecial`（`Plant.cpp:4306-4336`，樱桃分支逐字）：

```cpp
// src/Lawn/Plant.cpp:4308-4310（函数头，对樱桃生效）
	int aPosX = mX + mWidth / 2;      // ⇒ mX + 40
	int aPosY = mY + mHeight / 2;     // ⇒ mY + 40
	int aDamageRangeFlags = GetDamageRangeFlags(PlantWeapon::WEAPON_PRIMARY);
// :4323-4335
	case SeedType::SEED_CHERRYBOMB:
	{
		mApp->PlayFoley(FoleyType::FOLEY_CHERRYBOMB);
		mApp->PlayFoley(FoleyType::FOLEY_JUICY);

		if (mBoard->KillAllZombiesInRadius(mRow, aPosX, aPosY, 115, 1, true, aDamageRangeFlags) >= 10)
			ReportAchievement::GiveAchievement(mApp, Explodonator, true);

		mApp->AddPvzpParticle(aPosX, aPosY, static_cast<int>(RenderLayer::RENDER_LAYER_TOP), ParticleEffect::PARTICLE_POWIE);
		mBoard->ShakeBoard(3, -4);

		Die();
		break;
	}
```

四个参数：半径 **115**、`theRowRange = 1`、`theBurn = true`、`aDamageRangeFlags = 127`
（`Plant.cpp:610-614`：CHERRYBOMB / JALAPENO / COBCANNON / DOOMSHROOM 共用 127）。

**邻行规则（这是审计最关心的一条）**：几何测试在 `Board::KillAllZombiesInRadius`，逐字：

```cpp
// src/Lawn/Board.cpp:9325-9352
	for (Zombie* aZombie : mZombies)
	{
		if (aZombie->mDead)
			continue;
		if (aZombie->EffectedByDamage(theDamageRangeFlags))
		{
			Rect aZombieRect = aZombie->GetZombieRect();
			int aRowDist = aZombie->mRow - theRow;
			if (aZombie->mZombieType == ZombieType::ZOMBIE_BOSS)
			{
				aRowDist = 0;
			}

			if (aRowDist <= theRowRange && aRowDist >= -theRowRange && GetCircleRectOverlap(theX, theY, theRadius, aZombieRect))
			{
				if (theBurn)
				{
					aZombie->ApplyBurn();
				}
				else
				{
					aZombie->TakeDamage(1800, 18U);
				}

				aKilledZombies++;
			}
		}
	}
```

所以**邻行是"含入"而不是"排除"**：`rowDist ∈ [-1, +1]`（`mRow-1 / mRow / mRow+1` 三行都算），
并且这个行号条件与圆测试是 **AND** 关系。圆测试是 clamp 最近点法（逐字）：

```cpp
// src/Lawn/Board.cpp:8975-8982
bool GetCircleRectOverlap(int theCircleX, int theCircleY, int theRadius, const Rect& theRect)
{
	int aNearX = std::clamp(theCircleX, theRect.mX, theRect.mX + theRect.mWidth);
	int aNearY = std::clamp(theCircleY, theRect.mY, theRect.mY + theRect.mHeight);
	int dx = theCircleX - aNearX;
	int dy = theCircleY - aNearY;
	return dx * dx + dy * dy <= theRadius * theRadius;
}
```

`theRect` 就是 `Zombie::GetZombieRect()`（**不是** `GetZombieAttackRect()`、**不是**植物的矩形），
整数算术、`<=` 闭区间 ⇒ 恰在边界上**算命中**。行间距 100 px（`GridToPixelY`），
圆心 `(mX+40, mY+40)` 在上/下邻行的 rect 带上至少留 `115-60=55` px 的纵向余量，
所以"邻行能不能炸到"实际取决于邻行僵尸 **x** 是否落进圆 —— 不是无条件命中邻行整行。
被命中的**清单** = `EffectedByDamage(127)` ∧ 行差 ∈ [-1,1] ∧ 圆内；`127 = 0b01111111`
（bit0..bit6，只缺 `DAMAGES_ONLY_MINDCONTROLLED(7)`，`ConstEnums.h:257-267`）⇒ 地面 / 飞行 /
潜水 / 狗 / 离地 / **濒死** / 地下全都在内， practically 只剩"被洗脑的僵尸"和"不在场上的"被排除
（`Zombie.cpp:8091-8101` 与 `:8154-8157`）。

**WRONG 的一条：伤害不是平摊 1800。** `theBurn = true` ⇒ 走 `Zombie::ApplyBurn()`（`Zombie.cpp:8687-8695`）：

```cpp
void Zombie::ApplyBurn()
{
	if (mDead || mZombiePhase == ZombiePhase::PHASE_ZOMBIE_BURNED)
		return;

	if (mBodyHealth >= 1800 || mZombieType == ZombieType::ZOMBIE_BOSS)
	{
		TakeDamage(1800, 18U);
		return;
	}
```

只有 `mBodyHealth >= 1800` 或 Boss 才是那一下 1800（第 7 关卡组的四只都没到：默认 `mBodyHealth = 270`
`Zombie.cpp:238`，POLEVAULTER 500 `:378`）。其余全部落到"烧死"路径：普通地面僵尸直接
`DieWithLoot()` + `REANIM_ZOMBIE_CHARRED`，并且**每次这样杀掉一只会从全局 MT 流抽 1 个 float**
（`Zombie.cpp:8792`，已在 §5.2 / §8 行 3 记录）；飞行 / 无头 / bungee / yeti / 僵尸草 / 雪橇队
才吃 `PHASE_ZOMBIE_BURNED` + `mPhaseCounter = 300`（`:8740-8741`）。
`flag 18` 的正确解码见本节引言的更正框。

自毁：`Die()`（`Plant.cpp:4334`），同 tick `mDead = true`，**没有淡出**（§5.3）。

### 4. 产阳光：首/后续延迟、数值、Coin 与 headless 收取

**全部 CONFIRMED**，含审计给的三个数字与"是 Coin 对象"。

构造函数里的**首次**延迟（`Plant.cpp:189-197`，逐字）：

```cpp
	if (mLaunchRate > 0)
	{
		if (MakesSun())
			mLaunchCounter = RandRangeInt(300, mLaunchRate / 2);
		else
			mLaunchCounter = RandRangeInt(0, mLaunchRate);
	}
	else
		mLaunchCounter = 0;
```

`SEED_SUNFLOWER` 行 `mLaunchRate = 2500`（`Plant.cpp:50`），`MakesSun()` 只对 SUNFLOWER /
TWINSUNFLOWER / SUNSHROOM 为真（`Plant.cpp:999-1002`）⇒ 首次 = `RandRangeInt(300, 1250)`，
两端闭（`RandRangeInt(a,b) = a + Rand(b-a+1)`）。审计写的 `[300, 1250]` **CONFIRMED**。

**后续**延迟在 `Plant::UpdateProductionPlant`（不是 `Plant::UpdateProduce` ——
`NOT FOUND`，正确函数名是 `UpdateProductionPlant`，`Plant.cpp:1004-1074`）：

```cpp
// src/Lawn/Plant.cpp:1026-1034 与 :1048-1051（逐字节选）
	mLaunchCounter--;
	if (mLaunchCounter <= 100)
	{
		int aFlashCountdown = PvzpAnimateCurve(100, 0, mLaunchCounter, 0, 100, PvzpCurves::CURVE_LINEAR);
		mEatenFlashCountdown = std::max(mEatenFlashCountdown, aFlashCountdown);
	}
	if (mLaunchCounter <= 0)
	{
		mLaunchCounter = RandRangeInt(mLaunchRate - 150, mLaunchRate);
		mApp->PlayFoley(FoleyType::FOLEY_SPAWN_SUN);
	...
	else if (mSeedType == SeedType::SEED_SUNFLOWER)
	{
		mBoard->AddCoin(mX, mY, CoinType::COIN_SUN, CoinMotion::COIN_MOTION_FROM_PLANT);
	}
```

`RandRangeInt(2500-150, 2500)` ⇒ **[2350, 2500]** 闭区间，**CONFIRMED**。
`mLaunchCounter <= 100` 那段只写 `mEatenFlashCountdown`（§7 归为装饰），**不**延迟产出。

* **值 25 CONFIRMED**：`Coin::GetSunValue()` `Coin.cpp:1299`
  `return mType == CoinType::COIN_SUN ? 25 : mType == COIN_SMALLSUN ? 15 : mType == COIN_LARGESUN ? 50 : 0;`
* **确实是 Coin 对象 CONFIRMED**：`Board::AddCoin`（`Board.cpp:1971-1982`）从 `mCoins` 数组分配一个
  `Coin` 并 `CoinInitialize(...)`；同处 `:1975-1976` 在**创建那一刻**把
  `mSunMoneyProduced += aCoin->GetSunValue()`（所以"产量"统计与"入账"统计是两个不同的量）。
* **headless 自动收取**：`LawnApp::EnvironmentWait`（`src/LawnApp.cpp:1504-1519`，逐字）：

  ```cpp
  void LawnApp::EnvironmentWait(int ticks)
  {
      if (!mBoard || ticks < 0 || ticks > 1000000)
          return;
      for (int i = 0; i < ticks && !EnvironmentTerminal(); ++i)
      {
          for (Coin* coin : mBoard->mCoins)
          {
              if (!coin->mDead && coin->IsSun())
                  coin->ScoreCoin();
          }
          AdvanceLogicTick();
          if (!mHeadlessMode)
              EnvironmentPresent();
      }
  }
  ```

  即**每 tick 在推进逻辑之前**把场上所有还活着的阳光 Coin 结算掉；`Coin::ScoreCoin()`
  （`Coin.cpp:433-448`）第一句就是 `Die()`，然后 `mBoard->AddSunMoney(GetSunValue())`
  ⇒ `Board::AddSunMoney`（`Board.cpp:8463-8466`）`mSunMoney += 25; mSunMoney = std::min(mSunMoney, 9990);`。
  **可观测后果**：产出 tick `T` 的太阳花，其阳光在 tick `T+1` 才进 `mSunMoney`（收取循环在 tick 头部），
  但 `mSunMoneyProduced` 在 `T` 就已经加了 —— 两个量的口径差 1 tick，`src/CudaExport.cpp:289-297`
  给导入的根局面 Sun 一律写 `ticksToCollect = 1` 就是在对齐这条。
  交互式游戏的正常收取路径（鼠标点击 → `mIsBeingCollected` → `Coin.cpp:712-723` 的
  `mCollectionDistance < 8.0f` 才 `ScoreCoin()`）在 rollout 里被上面这个循环短路，内核**不需要**建模飞行轨迹。

### 5. 射击者：时钟、`mLaunchRate`、伤害来源、雪豌豆 chill

**时钟 CONFIRMED，`mLaunchRate` CONFIRMED；两处 WRONG：`gPlantDefs` 里没有 `mDamage`；chill 的"行走减半"。**

`Plant::UpdateShooter`（`Plant.cpp:945-997`）——审计描述的三件事全在，逐字（去掉与本卡组无关的分支）：

```cpp
void Plant::UpdateShooter()
{
	mLaunchCounter--;                                              // :947
	if (mLaunchCounter <= 0)
	{
		mLaunchCounter = mLaunchRate - Sexy::Rand(15);             // :950  ⇒ [136, 150]
		...
		else
		{
			FindTargetAndFire(mRow, PlantWeapon::WEAPON_PRIMARY);   // :978  PEASHOOTER/SNOWPEA/REPEATER 都走这里
		}
	}
	...
	if (mLaunchCounter == 25)                                       // :986
	{
		if (mSeedType == SeedType::SEED_REPEATER || mSeedType == SeedType::SEED_LEFTPEATER)
		{
			FindTargetAndFire(mRow, PlantWeapon::WEAPON_PRIMARY);   // :990
		}
		else if (mSeedType == SeedType::SEED_SPLITPEA) { ... }      // 死分支
	}
}
```

`mLaunchCounter--` 在**函数第一句**、重置写在**同一 tick**、第二发用 `== 25` 精确相等 ⇒ 三条 **CONFIRMED**。
`mLaunchRate - Sexy::Rand(15)` ⇒ `Rand(15) ∈ [0,14]` ⇒ 重置值 `R ∈ [136, 150]`。
它由 `Plant::UpdateAbilities` 的 `if (mSubclass == SUBCLASS_SHOOTER)` 调用（`Plant.cpp:2578-2581`），
`SUBCLASS_SHOOTER` 来自 `gPlantDefs` 行本身。

`gPlantDefs` 的 `mLaunchRate`（逐行）：

| 行 | file:line | `mLaunchRate` | `mSubClass` |
|---|---|---|---|
| PEASHOOTER | `src/Lawn/Plant.cpp:49` | **150** | `SUBCLASS_SHOOTER` |
| SUNFLOWER | `src/Lawn/Plant.cpp:50` | 2500 | `SUBCLASS_NORMAL` |
| SNOWPEA | `src/Lawn/Plant.cpp:54` | **150** | `SUBCLASS_SHOOTER` |
| REPEATER | `src/Lawn/Plant.cpp:56` | **150** | `SUBCLASS_SHOOTER` |

**WRONG（前提不成立）：`PlantDefinition` 没有 `mDamage` 成员。** 全类只有 9 个字段
（`src/Lawn/Plant.h:324-336`）：`mSeedType, mPlantImage, mReanimationType, mPacketIndex, mSeedCost,
mRefreshTime, mSubClass, mLaunchRate, mPlantName`。伤害在弹道上，与发弹的植物无关：

```cpp
// src/Lawn/Projectile.cpp:38-39
	{ .mProjectileType = ProjectileType::PROJECTILE_PEA,     .mImageRow = 0, .mDamage = 20 },
	{ .mProjectileType = ProjectileType::PROJECTILE_SNOWPEA, .mImageRow = 0, .mDamage = 20 },
```

⇒ **PEASHOOTER / REPEATER / SNOWPEA 的单发伤害都是 20**（`Plant::Fire` 只决定弹型：
`SEED_SNOWPEA → PROJECTILE_SNOWPEA` `Plant.cpp:4521-4523`，其余 peashooter 家族 `→ PROJECTILE_PEA`
`:4513-4520`）。REPEATER 的"DPS 翻倍"完全来自每周期发两发（§2.6），不是来自更大的 `mDamage`。

雪豌豆的 chill 与"谁写 flag"：

```cpp
// src/Lawn/Projectile.cpp:418-421（Projectile::GetDamageFlags）
	if (mProjectileType == ProjectileType::PROJECTILE_SNOWPEA || mProjectileType == ProjectileType::PROJECTILE_WINTERMELON)
	{
		SetBit(aDamageFlags, static_cast<int>(DamageFlags::DAMAGE_FREEZE), true);   // bit 2，ConstEnums.h:252
	}
```

落地时 `GetDamageFlags` 的结果随伤害一起传下去（`Projectile.cpp:496`、`:838`），chill 在**受击方**结算，
`Zombie::TakeHelmDamage` 与 `Zombie::TakeBodyDamage` 各有一份同样的检查：

```cpp
// src/Lawn/Zombie.cpp:7731-7734（护甲路径）与 :7831-7834（身体路径）—— 两处逐字相同
	if (TestBit(theDamageFlags, static_cast<int>(DamageFlags::DAMAGE_FREEZE)))
	{
		ApplyChill(false);
	}
```

```cpp
// src/Lawn/Zombie.cpp:7522-7540
void Zombie::ApplyChill(bool theIsIceTrap)
{
	if (!CanBeChilled())
		return;

	if (mChilledCounter == 0)
	{
		mApp->PlayFoley(FoleyType::FOLEY_FROZEN);
	}

	int aChillTime = 1000;
	if (theIsIceTrap)
	{
		aChillTime = 2000;
	}
	mChilledCounter = std::max(aChillTime, mChilledCounter);

	UpdateAnimSpeed();
}
```

**chill = 1000 tick CONFIRMED**（`Zombie.cpp:7532`；冰蘑菇陷阱是 2000 `:7535`；写入是 `max`
`zombie.cpp:7537` ⇒ **不叠加、只刷新**）。`CanBeChilled()`（`:8035-8059`）排除压路机 / 有雪橇的雪橇队 /
濒死 / 掘地隧道与出土相位 / 被洗脑的。每 tick 递减在 `:4619-4626`。

**WRONG：chill 下行走速度不是"减半"，是 ×0.4。**

```cpp
// src/Lawn/Zombie.cpp:70
constexpr const float CHILLED_SPEED_FACTOR = 0.4f;
// 应用于 Zombie::UpdateZombieWalking（:4104 起）的 :4117-4120 与 :4134-4137（以及 :4202-4205）
			if (IsMovingAtChilledSpeed())
			{
				aSpeed *= CHILLED_SPEED_FACTOR;
			}
```

被**减半**的是**动画速率**，不是位移速率：`Zombie::ApplyAnimRate` `:6639`
`aBodyReanim->mAnimRate = IsMovingAtChilledSpeed() ? theAnimRate * 0.5f : theAnimRate;`。
（`cuda/SimTick.inc` 里若按 0.5 建模位移，位移会比 CPU 参照**快 25%**。）
第三条 chill 后果别漏：啃食变慢有两道闸，`Zombie::CheckIfPreyCaught` `:6860-6864`
把 `aTicksBetweenEats = TICKS_BETWEEN_EATS(4)` **翻倍成 8**，`Zombie::EatPlant` `:7087-7088`
又额外只放行 `mZombieAge` 为偶数的 tick；前者是决定性的 ⇒ 被 chill 的僵尸每 **8** tick 才咬一口。

### 6. 种植血量、延迟击发 +34/+25、死亡判定、`mPlantsEaten`

**四项全部 CONFIRMED。**

种植血量：默认 300，本卡组只有 WALLNUT 改。

```cpp
// src/Lawn/Plant.cpp:133（PlantInitialize 通用段）
	mPlantHealth = 300;
// src/Lawn/Plant.cpp:288-291
	case SeedType::SEED_WALLNUT:
		mPlantHealth = 4000;
		mBlinkCountdown = 1000 + Sexy::Rand(1000);
		break;
// src/Lawn/Plant.cpp:487-492
	if ((mApp->mGameMode == GameMode::GAMEMODE_CHALLENGE_BIG_TIME) && ...)   // 第 7 关死分支
		mPlantHealth *= 2;
	mPlantMaxHealth = mPlantHealth;
```

⇒ PEASHOOTER / SUNFLOWER / CHERRYBOMB / POTATOMINE / SNOWPEA / REPEATER = **300**，WALLNUT = **4000**；
`mPlantMaxHealth` 在 `PlantInitialize` 末尾一次性抄写（`:492`），之后只有碎裂贴图分档读它
（`Plant.cpp:3122`、`:3130`），不改变伤害。`Plant::Die()`（`:4946-4958`）**不**碰血量，只 `mDead = true`。

**延迟击发 +34 / +25 CONFIRMED。** 机制是：`FindTargetAndFire` 在有 `anim_shooting` 头轨的射手上
**不发弹**，只写动画与 `mShootingCounter`；真正的 `Fire()` 在 `UpdateShooting` 里计数器读到 1 时发生。

```cpp
// src/Lawn/Plant.cpp:751-769（FindTargetAndFire 内）
	else if (aHeadReanim && aHeadReanim->TrackExists("anim_shooting"))
	{
		aHeadReanim->StartBlend(20);
		...
		mShootingCounter = 35;                                      // :758  PEASHOOTER / SNOWPEA
		if (mSeedType == SeedType::SEED_REPEATER || mSeedType == SeedType::SEED_SPLITPEA || mSeedType == SeedType::SEED_LEFTPEATER)
		{
			aHeadReanim->mAnimRate = 45.0f;
			mShootingCounter = 26;                                  // :762  REPEATER / LEFTPEATER
		}
		...
	}
...
	else
		Fire(aZombie, theRow, thePlantWeapon);                     // :815 只有「没有射击动画」的类型才当场发
```

```cpp
// src/Lawn/Plant.cpp:3205-3210 与 :3248-3309（UpdateShooting）
	if (NotOnGround() || mShootingCounter == 0)
		return;
	mShootingCounter--;
	...
	else if (mShootingCounter == 1)                                 // :3248
	{
		if (THREEPEATER) {...} else if (SPLITPEA) {...} ...
		else
		{
			Fire(nullptr, mRow, PlantWeapon::WEAPON_PRIMARY);       // :3305
		}
		return;
	}
```

⇒ 从 35 起跳的（PEASHOOTER / SNOWPEA）在**决策 tick 之后第 34 tick** 出现弹；
从 26 起跳的（REPEATER / LEFTPEATER）在第 **25** tick。审计的 `+34/+25` **CONFIRMED**
（§8 行 7 同一条）。两个必须一起带的细节：

* `Fire()` 收到的是 `theTargetZombie = nullptr`（`:3305`）—— 目标只在决策 tick 起作用，
  34/25 tick 后**不重新选靶**，弹道行号也固定为 `mRow`。
* `UpdateShooting` 尾部 `Plant.cpp:3417` 有 `mShootingCounter = 1;`，把头动画还没循环完的射手钉在 1，
  这**不会**多发一弹（`== 1` 的判定在递减**之后**，钉成 1 的下一 tick 递减到 0 就跳过了 `:3248`），
  只是让动画保持在射击姿态。

死亡判定：`Plant::Update()` 里是**严格 `< 0`**（逐字）：

```cpp
// src/Lawn/Plant.cpp:2868-2877
	if (doUpdate)
	{
		UpdateAbilities();
		Animate();

		if (mPlantHealth < 0)
			Die();

		UpdateReanim();
	}
```

⇒ `mPlantHealth == 0` 的植物在**植物循环里不自杀**。把 0 或负数变成尸体的是僵尸那条路：

```cpp
// src/Lawn/Zombie.cpp:7100-7115（Zombie::EatPlant 尾部）
	thePlant->mPlantHealth -= DAMAGE_PER_EAT;                       // DAMAGE_PER_EAT = 4，Zombie.cpp:67-68
	thePlant->mRecentlyEatenCountdown = 50;
	...
	if (thePlant->mPlantHealth <= 0)                                // :7110  ← 这里是 <= 0
	{
		mApp->PlaySample(SOUND_GULP);

		mBoard->mPlantsEaten++;                                     // :7114
		thePlant->Die();                                            // :7115
		mBoard->mChallenge->ZombieAtePlant(thePlant);
		...
	}
```

`mPlantsEaten` 全部写入点（`uint32_t`，`Board.h:239`，init `Board.cpp:156`）与本关可达性：

| file:line | 所在函数 | 第 7 关 + 本卡组 |
|---|---|---|
| `src/Lawn/Zombie.cpp:7114` | `Zombie::EatPlant`，由 `mPlantHealth <= 0` 门控 | **唯一常态路径**（啃死植物） |
| `src/Lawn/Board.cpp:8992` | `Board::KillAllPlantsInRadius`（Gargantuar / imp 砸） | 不可达（无巨人） |
| `src/Lawn/Zombie.cpp:2417` / `:2430` | `Zombie::UpdateZombieJalapenoHead` 的死亡扫行（`#ifdef DO_FIX_BUGS` 两份） | 不可达 |
| `src/Lawn/Zombie.cpp:6492` | `Zombie::SquishAllInSquare`（压路机/投石车 `ATTACKTYPE_DRIVE_OVER`）—— 注意调 `Squish()` 不是 `Die()` | 不可达 |
| `src/Lawn/Zombie.cpp:7408` | `Zombie::BungeeDie` | 不可达（第 7 关无 bungee） |

⇒ 内核只需要实现 `:7114` 一处；`<= 0` 与 `< 0` 的差异在血量恰好被咬到 0 时可观测
（植物 tick 先跑、僵尸 tick 后跑，见 §6.1 的顺序结论）。

### 7. 内核必须携带的 tick 状态：逐项（伤害/位置 vs 装饰）

按 `Plant.h:164-213` 的字段清单逐个判定。**"装饰"只对位置与伤害成立**——见 §7.3 的 RNG 例外。

| 字段 | file:line（写入 / 递减） | PEASHOOTER / SNOWPEA / REPEATER | SUNFLOWER | WALLNUT | CHERRYBOMB | POTATOMINE | 判定 |
|---|---|---|---|---|---|---|---|
| `mLaunchCounter` | 初值 `Plant.cpp:192`/`:194`；递减 `:947`（射手）/ `:1026`（产阳）；重置 `:950` / `:1034` | 发弹时钟（`R=150-Rand(15)`），且 `==25` 是第二发决策 | 产阳时钟 | — | — | — | **改变伤害**：必带 |
| `mShootingCounter` | `:758`/`:762` 写入；`:3210` 递减；`:3248` 命中；`:3417` 钉 1 | 决定弹在哪一 tick 出现（+34/+25） | — | — | — | — | **改变伤害与弹的初始位置**：必带 |
| `mStateCountdown` | `:346` 写 1500；`:2546-2547` 递减 | — | — | — | — | 出土时刻 | **改变伤害**（土豆雷唯一计时闸门）：必带 |
| `mState`（土豆雷三段） | `:128`、`:1155`、`:1176` | — | — | — | — | `NOTREADY` 时可被啃（`Zombie.cpp:7067`），之后不能 | **改变存活/伤害**：必带；`RISING → ARMED` 依赖动画，见 §9 |
| `mDoSpecialCountdown` | `:327` 写 100；`:2582-2588` 递减并触发 | — | — | — | 引信 | — | **改变伤害**：必带 |
| `mPlantHealth` | `:133`/`:289`；`Zombie.cpp:7100`；`Projectile.cpp:315`/`:617`（本关只从啃食来） | — | — | 300→4000 决定还能被啃多久 | — | — | **改变存活**：必带（`mPlantMaxHealth` 只用于贴图分档，可只存初值常量） |
| `mDisappearCountdown` | `:135` 写 200；`:2518-2526` 递减 | 只在 `STATE_DOINGSPECIAL \|\| mSquished` 才动 | 同左 | 同左 | 樱桃不进 `DOINGSPECIAL` | 不进 | **本卡组全程恒为 200**：可不带（除非实现巨人砸） |
| `mBlinkCountdown` | `:183`、`:290`、`:1177`；`:3081-3088` 递减并调 `DoBlink()` | ✔ | ✔ | ✔ | ✔ | ✔（`:3022` 未武装时早退） | 视觉**装饰**，但 `DoBlink` `:3017` 每次抽 **1 int**（wallnut 再 +2：`:3041` 与 `AttachBlinkAnim` 的 `:2902`） ⇒ **RNG 上不可跳** |
| `mRecentlyEatenCountdown` | `Zombie.cpp:7101` 写 50；`Plant.cpp:3428-3431` 递减 | 无消费者（`FindStarFruitTarget` `:860` 是星星果） | 无 | `AnimateNuts` `:3145-3149` 把身体动画压到 0.1f | 无 | 无 | **装饰**：不改变伤害/位置，且**不抽 RNG** ⇒ 可跳 |
| `mEatenFlashCountdown` | `Plant.cpp:1030`（产阳前 100 tick 的灰色渐显）；`:3432-3435` 递减 | — | 只喂 `UpdateReanimColor` `:2700-2702` | — | — | — | **装饰**，不抽 RNG ⇒ 可跳 |
| `mAnimCounter` / `mFrame` / `mFrameLength` / `mAnimPing` / `mNumFrames` | `:117-127`、`Animate` 步骤 8 `:3462-3483` | 只产出 `mFrame` | 同左 | 同左 | 同左 | 同左 | **纯装饰**：`mFrame` 唯一读者是 `Plant::Draw` `:3961` 的旧图集索引；本卡组全用 reanim。**但 `mFrameLength = RandRangeInt(12,18)`（`:123`）种植时抽 1 int**，该 draw 不可跳 |
| `mShakeOffsetX/Y` | `Animate` 步骤 1 `:3422-3426`（只 CHERRYBOMB/JALAPENO）；`UpdateReanim` `:2747-2749` 用作 reanim 偏移 | — | — | — | **每 tick 2 个 float RNG**，共 100 tick ⇒ 200 draw | — | 位置是**装饰**（不改 `mX/mY`），**RNG 不可跳**（§1.4 / §8 行 10） |
| `mIsAsleep` / `mWakeUpCounter` | 初值 `Plant.cpp:153-154`；入睡 `Plant.cpp:186-187`（`IsNocturnal && !StageIsNight()`）；递减 `Plant.cpp:2528-2539` | — | — | — | — | — | **第 7 关白天 + 全日行卡组：恒 false / 恒 0** ⇒ 整块可跳 |
| `mSquished` / `mOnBungeeState` / `mTargetZombieID` / `mPottedPlantIndex` / `mImitaterType` / `mTargetX/Y` / `mStartRow` | 声明 `Plant.h:184-186`、`:203-210`；初值 `Plant.cpp:124-131`、`:155-156`；写入者只出现在 squash `:1532-1544` / gravebuster / 巨人砸 `Squish()` `:2320-2360` / bungee / 屋顶花盆 / imitater 路径 | — | — | — | — | — | **本卡组本关全程不变** ⇒ 可不带（`mSquished` 一旦为真会改变行为，但没有写入者） |
| `mLightReanimID` / `mBodyReanimID` / `mHeadReanimID(2/3)` / `mBlinkReanimID` / `mSleepingReanimID` / `mParticleID` | 初值 `Plant.cpp:137-144`；土豆雷 `anim_glow` 建在 `:1166-1173` | 头 reanim 的 transform 会喂 `GetPeaHeadOffset` ⇒ **弹的初始位置**取自动画数据 | 无 | 无 | 无 | `anim_glow` 只改 `mFrameCount` | 位置类：源码读不出数值（§9 `UNKNOWN` 第 3 条）；`mParticleID` 本身装饰，但**发射粒子抽 RNG**（§1.6） |

#### 7.1 明确可以跳过的装饰（前提：不碰 RNG）

* 旧 sprite 帧动画的**递推本身**：`mAnimCounter` 的 ping-pong 与 `mFrame`（`Plant.cpp:3462-3483`）
  —— 种植时的 `RandRangeInt(12,18)` 那一次 draw 仍要保留。
* `mEatenFlashCountdown`（`Plant.cpp:1029-1030`、`:2700-2702`、`:3432-3435`）。
* `mRecentlyEatenCountdown` 对 wallnut 的 `mAnimRate = 0.1f`（`Plant.cpp:3145-3149`）。
  坚果没有任何依赖自身动画速率的游戏逻辑（`GetPlantRect` 用 `mX/mY`，恒定）。
* wallnut 三段裂纹贴图（`AnimateNuts` `Plant.cpp:3122-3142`）与 `Plant::DrawShadow`、`Draw`。
* `aLightReanim->mFrameCount` 随最近僵尸距离变化（`Plant.cpp:1188-1193`）——土豆雷指示器，纯视觉。

#### 7.2 装饰但**不能**跳的（因为消耗全局 MT 流）

审计若按"装饰"把这些删掉，CPU 参照与内核会在第一次 blink / 樱桃种植之后错开，与 §1.5、§1.6 的账目一致：

1. 每次 blink 1 int（`DoBlink` `Plant.cpp:3017` 在**所有**早退 return 之前），wallnut 同一事件再 2 int（`:3041`、`:2902`）。
2. 樱桃炸弹存活期间每 tick 2 float（`Animate` `Plant.cpp:3424-3425`），共 200。
3. 土豆雷 `RISING → ARMED` 那一刻 1 int + 1 float：`Plant.cpp:1163` 的 `RandRangeFloat(12.0f, 15.0f)`
   （**1 次** draw —— `RandRangeFloat(a,b) = Rand(b-a) + a`，`PvzpCommon.cpp:390-394`，与 int 版
   `Rand(max-min+1)` 的闭区间约定**不同**）与 `:1177` 的 `400 + Rand(4000)`。
   两者都只决定动画速率 ⇒ 数值上装饰，但 **draw 本身不可跳**。
4. 每一次粒子发射（§1.6）：种植 `PARTICLE_PLANTING`（`Board.cpp:2084`）、土豆出土
   `PARTICLE_POTATO_MINE_RISE`（`Plant.cpp:1153`）、雪豌豆 `PARTICLE_SNOWPEA_PUFF`（`Plant.cpp:4680`）、
   樱桃 `PARTICLE_POWIE`（`:4331`）、土豆雷 `PARTICLE_POTATO_MINE`（`:4396`）、
   wallnut 裂纹换档 `PARTICLE_WALLNUT_EAT_LARGE`（`:3127`/`:3135`）。
   Foley 相反，**在 rollout 里 0 draw**（`LawnApp.cpp:2734-2748` 的 `mEnvironmentMode || mHeadlessMode` 早退）。
5. 樱桃烧死一只 `mBodyHealth < 1800` 的普通僵尸时，`Zombie.cpp:8792` 抽 1 float（charred 动画速率抖动）。

#### 7.3 一句话清单（内核 SoA 需要的植物 tick 状态）

`mSeedType, mPlantCol/mRow(→mX/mY 常量), mPlantHealth, mState(土豆雷三段), mStateCountdown,
mDoSpecialCountdown, mLaunchCounter, mShootingCounter, mBlinkCountdown, mFrameLength(为复现 draw),
mShakeOffset 的 2 draw/tick 计数`。其余字段在"第 7 关 + 这副卡组 + NORMAL/FLAG/CONE/POLEVAULTER"
的乘积下要么恒为初值，要么只走视觉。
