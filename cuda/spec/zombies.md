# cuda/spec/zombies.md — Zombie mechanics as they actually are in the engine

Scope: the zombie-side facts `cuda/SimCore.h` / `SimImpl.inc` / `SimTick.inc` need for
**adventure level 7 (`BACKGROUND_1_DAY`, grass, 5 playable rows)**.  Every number is a
`file:line` from this repo.  Where the CUDA core disagrees with the engine, §5 says so.

Files read: `src/Lawn/Zombie.cpp` (10732 l), `src/Lawn/Zombie.h`, `src/Lawn/Board.cpp`,
`src/Lawn/LawnMower.cpp`, `src/Lawn/Plant.cpp`, `src/PvzpLib/Reanimator.cpp`,
`src/GameConstants.h`, `src/LawnApp.cpp`, `src/EnvironmentWaitEvents.h`.

Cross-references (do not duplicate): `waves.md` (spawn cadence / flag waves),
`plants.md` (plant health & the eaten-side bookkeeping), `projectiles.md`, `board.md`,
`protocol.md`.

---

## 1. Constants the CUDA core needs

### 1.1 Level-7 type table

Which types exist at level 7 is decided by `ZombieDefinition::mStartingLevel` in the
`gZombieDefs` table (`src/Lawn/Zombie.cpp:80-113`), evaluated at level 7:

| type (enum id) | starting level | appears at 7? | body | helm | shield | flying | `mScaleZombie` | `mVelX` (px/update) |
|---|---|---|---|---|---|---|---|---|
| `ZOMBIE_NORMAL` (0) | 1 (`:81`) | yes | 270 (`:241`) | 0 | 0 | 0 | 1.0 (`:206`) | `RandRangeFloat(0.23,0.37)` (`:1215`) |
| `ZOMBIE_FLAG` (1) | 1 (`:82`) | yes, but `mPickWeight = 0` (`:82`) → only spawned by flag waves | 270 (`:241`) | 0 | 0 | 0 | 1.0 (`:206`) | **0.45 fixed** (`:1193-1196`) |
| `ZOMBIE_TRAFFIC_CONE` (2) | 3 (`:83`) | yes | 270 (`:241`) | **370** (`:264`) `HELMTYPE_TRAFFIC_CONE` (`:263`) | 0 | 0 | 1.0 | `0.23-0.37` |
| `ZOMBIE_POLEVAULTER` (3) | 6 (`:84`) | yes | **500** (`:378`) | 0 | 0 | 0 | 1.0 | `RandRangeFloat(0.66,0.68)` (`:1202-1204`) |
| `ZOMBIE_PAIL` (4) | **8** (`:85`) | **no** — first appears at level 8 | (1100 helm, `:272`) | — | — | — | — | — |

Notes on how the table was evaluated:

* `mBodyHealth = 270` is the **default for every zombie**, set before the type switch at
  `Zombie.cpp:241` (`mBodyHealth = 270;`).  Types override it in their `case`; NORMAL and
  TRAFFIC_CONE do not, so their body is 270 and their `mBodyMaxHealth` is 270
  (`Zombie.cpp:886`: `mBodyMaxHealth = mBodyHealth;`).
* `mHelmHealth` defaults to 0 (`:210`); `mHelmType` defaults to `HELMTYPE_NONE` (`:187`).
  Only the `case` sets them.  Cone = 370 (`:264`), pail = 1100 (`:272`), football = 1400 (`:365`),
  digger = 100 (`:346`).  Nothing at level 7 other than the cone has a helmet.
* `mShieldHealth` defaults to 0 (`:215`).  **No level-7 type has a shield** (door is level 13,
  ladder 43).  `mShieldType` defaults to `SHIELDTYPE_NONE` (`:188`).
* `mFlyingHealth` defaults to 0 (`:217`); only the balloon sets it (`:641`, 20).  Irrelevant at 7.
* `mScaleZombie = 1.0f` for all four level-7 types (`:206`).  Only dancer/backup-dancer (0.8,
  `:649`/`:666`) and the "Little Trouble" challenge flag (0.5, `:873`) change it, and neither
  applies to adventure 7.  **So the CUDA core may treat `mScaleZombie` as the literal 1.0.**
* `mVariant` (the tongue anim, `:881-884`) is cosmetic for the state machine but is drawn from
  the spawner; `ZOMBIE_POLEVAULTER` forces `mVariant = false` (`:382`).
* `mAnimTicksPerFrame = 12` (`:192`) and `mAnimFrames = 12` (`:193`) for normal/cone/flag;
  the pole vaulter overrides only `mAnimTicksPerFrame = 6` (`:379`).  These drive the *legacy*
  frame counter, not the reanim, and are not used by the position update.
* There is **no separate "brain" health**.  `Zombie.h:128-136` declares exactly
  `mBodyHealth/mBodyMaxHealth`, `mHelmHealth/mHelmMaxHealth`, `mShieldHealth/mShieldMaxHealth`,
  `mFlyingHealth/mFlyingMaxHealth`.  "Head" is not a hit-point pool: it is the boolean
  `mHasHead` (`Zombie.h:120`), which falls off at a body-health *fraction* (see §1.3).

### 1.2 Spawn, x, and the column geometry

| name | value | file:line |
|---|---|---|
| `mPosX` at spawn | `780 + Rand(40)` → `[780, 819]` | `Zombie.cpp:24-25,161` (`ZOMBIE_START_RANDOM_OFFSET = 40`) |
| flag-zombie extra | `mPosX += 40` if `mBoard->IsFlagWave(mFromWave)` | `Zombie.cpp:236-239` |
| flag-zombie overwrite | `mPosX = WIDE_BOARD_WIDTH` (= 800) | `Zombie.cpp:595` |
| pole-vaulter overwrite | `mPosX = WIDE_BOARD_WIDTH + 70 + Rand(10)` → `[870, 879]` | `Zombie.cpp:383` |
| `WIDE_BOARD_WIDTH` / `BOARD_WIDTH` | 800 | `GameConstants.h:28,30` |
| house line (`BOARD_EDGE`) | **-100** | `GameConstants.h:32` |
| per-type house-line override | gargantuar/redeye/pole-vaulter `-150`; catapult/football/zamboni `-175`; dancer/backup/snorkel `-130` | `Zombie.cpp:4545-4554` |
| the loss test uses the **truncated int** x | `mX = static_cast<int>(mPosX)`; test is `mX <= aEdgeX` | `Zombie.cpp:4384`, `:4556` |
| headless auto-kill line | `mX <= aEdgeX + 70 && !mHasHead` → `TakeDamage(1800, 9U)` | `Zombie.cpp:4566-4569` |
| walk-off-board edge | `IsWalkingBackwards() && mPosX > 850.0f` → `DieNoLoot()` | `Zombie.cpp:4539-4542` |
| left-zone threshold (event) | `LeftZoneMaxX = 160.0` | `EnvironmentWaitEvents.h:16` |
| LAWN_XMIN (col 0 left pixel) | 40 | `GameConstants.h:35` |
| `GridToPixelX(col)` | `col * 80 + 40` | `Board.cpp:8874` |
| grid columns | `MAX_GRID_SIZE_X = 9` | `Board.h:42` |
| `PixelToGridX(x)` | `-1` if `x < 40`, else `clamp((x-40)/80, 0, 8)` | `Board.cpp:8812-8827` |
| plant pixel x | `mX = GridToPixelX(col, row)` = `80*col + 40` | `Plant.cpp:114` |
| plant `GetPlantRect()` (generic) | `Rect(mX + 10, mY, mWidth - 20, mHeight)`, `mWidth = mHeight = 80` | `Plant.cpp:149-150`, `Plant.cpp` `GetPlantRect` → `[80c+50, 80c+110]` |
| `mZombieRect` (hurt box) | `Rect(36, 0, 42, 115)` → x span `[mX+36, mX+78]` | `Zombie.cpp:199` |
| `mZombieAttackRect` (mouth) normal/cone/flag | `Rect(50, 0, 20, 115)` → x span `[mX+50, mX+70]` | `Zombie.cpp:200` |
| `mZombieAttackRect` pole vaulter | `Rect(-29, 0, 70, 115)` → x span `[mX-29, mX+41]` | `Zombie.cpp:390` |
| zombie `mWidth`/`mHeight` | 120 / 120 (`:166-167`); used by the backwards mirror `mX = mWidth - mX - mWidth` | `Zombie.cpp:8332-8335` |
| `GetRectOverlap` | **1-D, x only**: `min(r1.x+r1.w, r2.x+r2.w) - max(r1.x, r2.x)` | `Board.cpp:8969-8973` |

Unit convention: `mPosX` is **pixels** in board/screen space (0 = left of the lawn strip,
`LAWN_XMIN = 40` is the left edge of column 0, 9 columns × 80 px = 720 px, so columns 0..8
are `[40,760]`).  `mX` is `int(mPosX)`.  `mRow` is a logical row 0..`MAX_GRID_SIZE_Y-1` = 0..5;
on grass only rows 0..4 are plantable (row 5 is dirt) — see `board.md`.  **There is no separate
"logical x" space for zombies**: all thresholds above (`160`, `-100`, `800`, `850`) are pixels.

The backwards mirror is a no-op for the default mouth rect: `120 - 50 - 20 = 50`.  So a
mind-controlled zombie has the same mouth rect, and only the pole vaulter's mirror differs
(`120 - (-29) - 70 = 79`).

### 1.3 Health bookkeeping / damage order

`Zombie::TakeDamage` (`Zombie.cpp:7973-8000`), in order:

```
TakeDamage(dmg, flags):
    if mZombiePhase == PHASE_JACK_IN_THE_BOX_POPPING or IsDeadOrDying(): return    # :7975
    rem = dmg
    if IsFlying():                        rem = TakeFlyingDamage(rem, flags)       # :7980-7983
    if rem > 0 and mShieldType != NONE
       and not TestBit(flags, DAMAGE_BYPASSES_SHIELD):
        rem = TakeShieldDamage(rem, flags)                                          # :7984-7990
        if TestBit(flags, DAMAGE_HITS_SHIELD_AND_BODY): rem = dmg                   # :7991-7994
    if rem > 0 and mHelmType != NONE:     rem = TakeHelmDamage(rem, flags)          # :7995-7998
    if rem > 0:                           TakeBodyDamage(rem, flags)                # :7999-8002
```

* **Helm**: `aDamageActual = min(mHelmHealth, theDamage)`; `mHelmHealth -= aDamageActual`;
  return `theDamage - aDamageActual` (`Zombie.cpp:7730-7734`).  Overkill therefore never
  leaks into the body during the bite that breaks the helmet, and `mHelmHealth` lands
  **exactly on 0** and is compared with `== 0` (`:7736`), not `<= 0`.
  At 0 → `DropHelm(flags)` and the remainder is returned (`:7736-7740`).
  Level 7: the cone's 370 absorbs exactly 370 damage; the excess carries into the body.
* **Body**: `mBodyHealth -= theDamage` (`Zombie.cpp:7837`) — **unclamped subtraction**, then
  `UpdateDamageStates(flags)` for our types (`:7962`), then
  ```
  if (mBodyHealth <= 0) { mBodyHealth = 0; PlayDeathAnim(flags); DropLoot(); }   # :7965-7970
  ```
  So the death predicate is **`<= 0`**, and the negative value is clamped back to 0 after the
  fact: **overkill damage is discarded, never banked or propagated**.  Note the contrast with
  the plant side, `EatPlant`: `thePlant->mPlantHealth -= DAMAGE_PER_EAT;` then
  `if (thePlant->mPlantHealth <= 0)` → `Die()` (`Zombie.cpp:7103,7110-7113`) — also `<= 0`,
  and DAMAGE_PER_EAT is small enough that plants never overshoot by more than 3.
* There is no "brain" pool.  What the community calls "brain health" is `mBodyHealth` once the
  head is gone (see next point).
* **Head/arm fall off on body-health fractions**, `UpdateDamageStates` (`Zombie.cpp:3904-3930`),
  only when `CanLoseBodyParts()` (`:3488-3499` — true for all four level-7 types):
  ```
  if (mHasArm  and mBodyHealth < 2*mBodyMaxHealth/3 and mBodyHealth > 0): DropArm()    # :3910-3913
  if (mHasHead and mBodyHealth <     mBodyMaxHealth/3):                   DropHead(); DropLoot(); StopZombieSound()  # :3915-3920
        if (mBoard->HasLevelAwardDropped()): PlayDeathAnim(flags)                       # :3922-3925
        if (mZombiePhase == PHASE_SNORKEL_WALKING_IN_POOL): DieNoLoot()                 # :3927-3930
  ```
  Integer division.  For body/max 270 (normal, cone, flag): arm at `mBodyHealth < 180`,
  head at `mBodyHealth < 90`.  For the pole vaulter (500): arm at `< 333`
  (`1000/3 = 333`), head at `< 166` (`500/3 = 166`).
* **A headless zombie does not die immediately.**  `mHasHead == false` sets `isDying` in
  `UpdatePlaying`, which then applies bleed-out (`Zombie.cpp:4655-4676`):
  ```
  if (!IsDeadOrDying() and mFromWave != ZOMBIE_WAVE_WINNER):
      isDying = !mHasHead                                    # :4657
      if (isDying):
          aDamage = 1;  if (mBodyMaxHealth >= 500) aDamage = 3     # :4667-4673
          if (Rand(5) == 0) TakeDamage(aDamage, 9U)          # :4674-4677
  ```
  So a headless 270-body zombie loses 1 HP on ~1/5 of ticks (expected ~5 ticks per HP),
  and a headless pole vaulter loses 3 HP per event.  `Rand(5)` makes this a **random draw the
  rollout must reproduce** — a deterministic kernel that bleeds every tick runs ~5× too fast.
* `9U` is the flags literal used for the bleed and for `EatZombie` (`Zombie.cpp:4676`, `:7129`);
  bit meanings are in `DamageFlags` — see `projectiles.md` for the decoded set.

### 1.4 Speed and the walk-advance relation

`Zombie::PickRandomSpeed` (`Zombie.cpp:1165-1218`) sets `mVelX` (px per update):

| condition (first match wins) | `mVelX` | line |
|---|---|---|
| `PHASE_SNORKEL_WALKING_IN_POOL` | 0.3f | `:1167-1170` |
| `PHASE_DIGGER_WALKING` | 0.23f / 0.12f | `:1171-1180` |
| IMP in I, Zombie | 0.9f | `:1181-1184` |
| `PHASE_YETI_RUNNING` | 0.8f | `:1185-1188` |
| `ZOMBIE_YETI` | 0.4f | `:1189-1192` |
| **`ZOMBIE_DANCER`, `ZOMBIE_BACKUP_DANCER`, `ZOMBIE_POGO`, `ZOMBIE_FLAG`** | **0.45f (fixed)** | `:1193-1197` |
| `PHASE_DIGGER_TUNNELING`, **`PHASE_POLEVAULTER_PRE_VAULT`**, `ZOMBIE_FOOTBALL`, `ZOMBIE_SNORKEL`, `ZOMBIE_JACK_IN_THE_BOX` | **`RandRangeFloat(0.66f, 0.68f)`** | `:1198-1205` |
| `PHASE_LADDER_CARRYING`, `ZOMBIE_SQUASH_HEAD` | `RandRangeFloat(0.79f,0.81f)` | `:1206-1209` |
| `PHASE_NEWSPAPER_MAD`, `PHASE_DOLPHIN_WALKING*` | `RandRangeFloat(0.89f,0.91f)` | `:1210-1214` |
| **else (NORMAL, TRAFFIC_CONE, PAIL, …)** | **`RandRangeFloat(0.23f, 0.37f)`** | `:1215-1218` |

Level 7 therefore needs: normal/cone `0.23..0.37` (uniform draw), flag `0.45` exact,
pole vaulter `0.66..0.68` (drawn at spawn *and* re-drawn on `StartWalkAnim`).

**`mVelX` is re-randomised every time a zombie resumes walking.**  `StopEating()`
(`:6794-6811`) calls `StartWalkAnim(20)` (`:6806`), and `StartWalkAnim` calls
`PickRandomSpeed()` as its first statement (`:6739`).  So eating ends with a fresh speed draw.
Any kernel that treats `mVelX` as fixed for a zombie's lifetime will drift.

The actual per-update advance is *not* `mPosX -= mVelX` for a normally rendered zombie.
`Zombie::UpdateZombieWalking` (`:4104-4216`) picks `aSpeed` by first match:

```
if ZombieNotWalking(): return                                       # :4106-4109
if body reanim exists:                                              # :4111
    if IsBouncingPogo() or PHASE_BALLOON_FLYING or PHASE_DOLPHIN_RIDING
       or PHASE_SNORKEL_WALKING_IN_POOL or ZOMBIE_CATAPULT:
        aSpeed = mVelX; if IsMovingAtChilledSpeed(): aSpeed *= CHILLED_SPEED_FACTOR   # :4116-4124
    elif ZOMBIE_ZAMBONI or PHASE_DIGGER_TUNNELING or PHASE_DOLPHIN_IN_JUMP
       or IsBobsledTeamWithSled() or PHASE_POLEVAULTER_IN_VAULT or PHASE_SNORKEL_INTO_POOL:
        aSpeed = mVelX                                              # :4125-4130
    elif aBodyReanim->TrackExists("_ground"):
        aSpeed = aBodyReanim->GetTrackVelocity("_ground") * mScaleZombie     # :4131-4134
    else:
        aSpeed = mVelX; if IsMovingAtChilledSpeed(): aSpeed *= CHILLED_SPEED_FACTOR   # :4135-4142
    if IsWalkingBackwards() or PHASE_DANCER_DANCING_IN: mPosX += aSpeed
    else:                                              mPosX -= aSpeed     # :4144-4151
else (no reanim — fallback):                                       # :4162
    doWalk = <phase table> or (mFrame in [0,2]) or (mFrame in [6,8])  # :4163-4187
    if doWalk: aSpeed = mVelX * (CHILLED_SPEED_FACTOR if chilled); mPosX ∓= aSpeed  # :4189-4204
```

The `_ground` branch is the one that runs in play for normal/cone/flag/pole-vaulter.
`Reanimation::GetTrackVelocity` (`Reanimator.cpp:1315-1325`):

```
aDis   = TransX[animFrameAfterInt] - TransX[animFrameBeforeInt]     # px moved per anim frame
return aDis * SECONDS_PER_UPDATE * mAnimRate                        # SECONDS_PER_UPDATE = 0.01  (Reanimator.h:52)
```

and `Zombie::UpdateAnimSpeed` (`Zombie.cpp:6685-6694`) sets, for a walking zombie:

```
aDistance    = TransX[frameStart + frameCount - 1] - TransX[frameStart]   # total loop displacement
aOneOverSpeed = mFrameCount / aDistance
mAnimRate     = mVelX * aOneOverSpeed * 47.0f / mScaleZombie
```

Substituting (with `aDis ≈ aDistance/mFrameCount`, the mean over the loop, and
`mScaleZombie = 1` at level 7):

```
aSpeed ≈ (aDistance/mFrameCount) * 0.01 * (mVelX * mFrameCount/aDistance * 47)
       = 0.47 * mVelX          px per update
```

**So the mean walk advance is `0.47 * mVelX` px per update, i.e. ≈ 0.108–0.174 px/update for a
normal zombie (`mVelX = 0.23–0.37`) and ≈ 0.11 px/update for the flag zombie (`mVelX = 0.45`,
so `0.47*0.45 = 0.2115`).**  It is *not* constant across a stride: `aDis` is the displacement
between two integer animation frames of the `_ground` track and varies within the cycle, so the
true advance pulses around that mean; only the mean is derivable from source.
**The per-frame `_ground` deltas are in the `.reanim` data, not in this repo's C++ — not
determined here.**  If the acceptance test needs bit-exact x, that table has to be extracted.
`UpdateAnimSpeed` is called from `ZombieInitialize` (`:880`), `ApplyChill` (`:7539`),
`RemoveIceTrap` (`:6604`), and when `mChilledCounter` expires (`:4622-4626`).

Eating freezes x: `ZombieNotWalking()` returns true when `mIsEating || IsImmobilizied()`
(`:3956-3959`), and `UpdateZombieWalking` returns at `:4106-4109`.  So **the chew step is
exactly 0 px** (`cChewStepX = 0.0f` is right, but for this reason, not the one in the comment
at `SimTick.inc:72-74` — there is no step-forward/step-back pair in the position code).

Chill: `CHILLED_SPEED_FACTOR = 0.4f` (`Zombie.cpp:69`), applied either through
`IsMovingAtChilledSpeed()` (`:4121,4140,4191`) or by having been baked into `mAnimRate`
when `UpdateAnimSpeed` re-ran.  `ApplyChill` sets `mChilledCounter = max(1000, current)`,
or `2000` for an ice trap (`Zombie.cpp:7534-7539`).  `mChilledCounter` decrements once per
update inside `UpdatePlaying` (`:4620-4627`).

---

## 2. The per-update zombie state machine, in `Board`'s update order

`Board::UpdateGameObjects` (`Board.cpp:4972-5011`) runs, in this order, one pass per update:

```
for Plant     in mPlants     : if !mDead: Plant::Update()          # Board.cpp:4972-4977
for Zombie    in mZombies    : if !mDead: Zombie::Update()         # Board.cpp:4979-4984
for Projectile in mProjectiles: if !mDead: Projectile::Update()    # Board.cpp:4986-4991
for Coin      in mCoins      : if !mDead: Coin::Update()           # Board.cpp:4993-4998
for LawnMower in mLawnMowers : if !mDead: LawnMower::Update()      # Board.cpp:5000-5005
mCursorPreview->Update(); mCursorObject->Update()                  # Board.cpp:5007-5008
for i in mSeedBank->mNumPackets: SeedPackets[i].Update()           # Board.cpp:5010-5013
```

`Board::UpdateGame` (`Board.cpp:5659+`) calls `UpdateGameObjects()` **first**, then
`mMainCounter++` (`:5683`), then `UpdateSunSpawning()` (`:5684`), then
`UpdateZombieSpawning()` (`:5685`).  Zombies therefore see the plant list *as of the start of
the update*, and the mower runs after projectiles and after the zombie that it will kill.

### Step 0 — skip if dead
`Board.cpp:4981-4983`: `if (aZombie->mDead) continue;`.  `mDead` is set only by `DieNoLoot()`
(`Zombie.cpp:7427`) / `DieWithLoot()`.  A zombie in `PHASE_ZOMBIE_DYING` is **not** `mDead` and
**is** still updated, still occupies its row, and **still counts for the left-zone event** (§4).

### Step 1 — age
`Zombie::Update` (`:4304-4309`): `PVZP_ASSERT(!mDead); mZombieAge++;` — unconditional, every
update, including while eating and while dying.  This is the eat cadence clock (§2 step 5).

### Step 2 — phase dispatch (`:4325-4370`)
```
if mZombiePhase == PHASE_ZOMBIE_BURNED:  UpdateBurn()                 # :4327-4330
elif mZombiePhase == PHASE_ZOMBIE_MOWERED: UpdateMowered()            # :4331-4334
elif mZombiePhase == PHASE_ZOMBIE_DYING:  UpdateDeath(); UpdateZombieWalking()   # :4335-4339
else:
    if (mPhaseCounter > 0 and !IsImmobilizied()): mPhaseCounter--     # :4342-4345
    if gameScene == SCENE_ZOMBIES_WON: ...                            # :4347-4355
    elif IsOnBoard(): UpdatePlaying()                                 # :4356-4359
    if ZOMBIE_BUNGEE: UpdateZombieBungee()                             # :4361-4364
    if ZOMBIE_POGO:   UpdateZombiePogo()                               # :4365-4368
    Animate()                                                          # :4370
```
Then the tail, executed for **every** non-`mDead` zombie (`:4373-4390`):
```
mJustGotShotCounter--                                   # :4373  (unconditional, goes negative)
if mShieldJustGotShotCounter > 0: --                    # :4374-4377
if mShieldRecoilCounter      > 0: --                    # :4378-4381
if mZombieFade > 0: mZombieFade--; if mZombieFade == 0: DieNoLoot()   # :4382-4388
mX = int(mPosX); mY = int(mPosY)                        # :4390-4391
UpdateReanim()                                          # :4394
```

### Step 3 — `UpdatePlaying` (`:4576-4678`), the branch that matters
```
PVZP_ASSERT(mBodyHealth > 0 or PHASE_BOBSLED_CRASHING)   # :4578
mGroanCounter--                                          # :4580   (sound only)
if mIceTrapCounter > 0: --; on 0 → RemoveIceTrap()       # :4613-4620
if mChilledCounter > 0: --; on 0 → UpdateAnimSpeed()     # :4621-4626   <-- speed restored here
if mButteredCounter > 0: --; on 0 → RemoveButter()       # :4627-4632
if mZombiePhase == PHASE_RISING_FROM_GRAVE: UpdateZombieRiseFromGrave(); return   # :4638-4642
if !IsImmobilizied():                                    # :4644
    UpdateActions()          # type-specific phase machines; all no-ops for level-7 non-vaulter  # :4646
    UpdateZombiePosition()   # x advance + y settle      # :4647
    CheckIfPreyCaught()      # the eat decision          # :4648
    CheckForPool()           # no-op on grass (:6930-6935, ZombieTypeCanGoInPool)  # :4649
    CheckForHighGround()     # no-op on grass            # :4650
    CheckForBoardEdge()      # the loss test             # :4651
if !IsDeadOrDying() and mFromWave != ZOMBIE_WAVE_WINNER:  # headless bleed-out  # :4654-4677
```

`UpdateZombiePosition` (`:4251-4282`): skips BUNGEE/BOSS/RISING_FROM_GRAVE/ZOMBIQUARIUM; calls
`UpdateZombieWalking()` (§1.4), then `CheckForZombieStep()` (zamboni/catapult only, `:4219-4222`),
then the y-settle `mPosY → GetPosYBasedOnRow(mRow)` at ≤1.0 px per update (`:4268-4281`).

### Step 4 — the loss test, `CheckForBoardEdge` (`:4537-4571`)
```
if IsWalkingBackwards() and mPosX > 850.0f: DieNoLoot(); return        # :4539-4542
aEdgeX = BOARD_EDGE (-100)                                             # :4544
        ; -150 for GARGANTUAR/REDEYE/POLEVAULTER                       # :4545-4548
        ; -175 for CATAPULT/FOOTBALL/ZAMBONI                           # :4549-4552
        ; -130 for BACKUP_DANCER/DANCER/SNORKEL                        # :4553-4554
if mX <= aEdgeX and mHasHead:                                          # :4556
    if IsIZombieLevel(): DieNoLoot()                                   # :4558-4561
    else: mBoard->ZombiesWon(this)                                     # :4562-4565
if mX <= aEdgeX + 70 and !mHasHead: TakeDamage(1800, 9U)                # :4566-4569
```
**Level 7 uses `aEdgeX = -100` for NORMAL / TRAFFIC_CONE / FLAG but `-150` for POLEVAULTER.**
The comparison is on the **truncated int** `mX`, so the loss fires on the first update whose
`int(mPosX)` reaches `-100` (i.e. `mPosX ∈ (-101, -100]`).
`Board::ZombiesWon` (`Board.cpp:5068-5074`) sets `mApp->mBoardResult = BOARDRESULT_LOST` and
leaves the scoring zombie in place (`theZombie` is explicitly skipped, `:5075-5079`); other
zombies already past `GetZombieRect().mX < -50` are `DieNoLoot()`-ed (`:5082-5093`).
Note the headless clause at `:4566` compares against `aEdgeX + 70 = -30`, not against 160.

### Step 5 — eating (`CheckIfPreyCaught`, `:6818-6900`)
The guard list at `:6820-6861` returns immediately for BUNGEE/GARGANTUAR/ZAMBONI/CATAPULT/BOSS,
several phases (**including `PHASE_POLEVAULTER_PRE_VAULT` and `PHASE_POLEVAULTER_IN_VAULT`**),
`HEIGHT_UP_LADDER`, pool transitions, `!mHasHead`, `IsFlying()`.  For level 7 the effective body is:

```
aTicksBetweenEats = TICKS_BETWEEN_EATS                       # 4,  Zombie.cpp:57
if mChilledCounter > 0: aTicksBetweenEats *= 2               # :6863-6866  → 8 while chilled
if mZombieAge % aTicksBetweenEats != 0: return               # :6867-6870  <-- modulo on the AGE, not a countdown
aZombie = FindZombieTarget(); if aZombie: EatZombie(aZombie); return    # :6872-6877   (mind-control only)
if !mMindControlled:
    aPlant = FindPlantTarget(ATTACKTYPE_CHEW)                          # :6879-6881
    if aPlant: EatPlant(aPlant); return                                # :6882-6884
if mIsEating: StopEating()                                             # :6895-6898
```

Target selection, `FindPlantTarget` (`:6423-6441`):
```
aAttackRect = GetZombieAttackRect()                          # x span [mX+50, mX+70] for normal/cone/flag
for aPlant in mBoard->mPlants:
    if aPlant->mDead: continue
    if aPlant->mRow != mRow: continue
    if GetRectOverlap(aAttackRect, aPlant->GetPlantRect()) >= 20 and CanTargetPlant(aPlant, CHEW):
        return aPlant                                        # FIRST match in list order, not leftmost
return nullptr
```
`GetRectOverlap` is x-only (§1.2), so **row matching is by `mRow` equality and vertical overlap
is never tested.**  Because the mouth rect is exactly 20 px wide and the threshold is `>= 20`,
the requirement is *"the mouth rect is entirely inside the plant's 60 px rect"*, which solves to

```
zombie eats the plant in column c  <=>  mX ∈ [80c, 80c + 40]          (41 integer values)
```

with `mX = int(mPosX)`.  Between consecutive plants there is a **39-px gap with no target**
(`mX ∈ [80c+1, 80c+39]` measured from the *next* column's origin): the zombie walks past the
plant's left half without biting it, then bites when its mouth enters the cell, and stops
biting 40 px before the plant's left edge is fully behind it.  For the **pole vaulter**
(mouth `[mX-29, mX+41]`, width 70, `Zombie.cpp:390`) the window is instead
`mX ∈ [80c+29, 80c+119]` (91 values) — a mouth that reaches **10 px behind itself**,
which is why a post-vault vaulter starts eating earlier (higher x) than a normal zombie.

`CanTargetPlant` (`:6344-6421`) for `ATTACKTYPE_CHEW` rejects: `thePlant->NotOnGround()`,
`SEED_TANGLEKELP`, a plant standing in a pool square while `!mInPool`, and spiky plants
(`SEED_SPIKEWEED`/`SEED_SPIKEROCK`) unless the plant cell is a pool square / has a flower pot /
the zombie is a gargantuar/zamboni.  Spikeweed is handled elsewhere (`IsStandingOnSpikeweed`,
`:4225-4249`) — outside zombie scope, see `plants.md`.

The bite itself, `EatPlant` (`:7027-7125`), after the ladder check (`:7038-7050`, no ladders at 7):
```
StartEating()                                                # :7051 → mIsEating = true (:6722), plays anim_eat (:6749)
# potato mine already armed is NOT eaten:  if SEED_POTATOMINE and mState != STATE_NOTREADY: return   # :7070-7073
# jalapeno/cherry/doom/ice/hypno & blover & squash states return early                                  # :7052-7069, :7075-7089
if mChilledCounter > 0 and mZombieAge % 2 == 1: return       # :7092-7095  <-- chilled skips odd ages
thePlant->mPlantHealth -= DAMAGE_PER_EAT                     # :7103  (DAMAGE_PER_EAT = TICKS_BETWEEN_EATS = 4, :58)
thePlant->mRecentlyEatenCountdown = 50                       # :7104
if thePlant->mPlantHealth <= 0:                              # :7110
    mBoard->mPlantsEaten++                                   # :7113
    thePlant->Die()                                          # :7114
    mBoard->mChallenge->ZombieAtePlant(thePlant)             # :7115
```
**There is no `mTargetPlantID` latch in the chew path.**  The target is re-found by rect
overlap every 4 ticks (`:6879`); `mTargetPlantID` (`Zombie.h:149`) is only used by the
catapult/gargantuar paths.  When the plant dies it drops out of `mPlants` iteration
(`aPlant->mDead` skip at `:6430`) and the *next* `CheckIfPreyCaught` either finds a new
overlapping plant or falls through to `StopEating()` (`:6895-6898`).

`StartEating` (`:6716-6754`): sets `mIsEating = true`, plays `anim_eat` with blend 20 and
rate `0.0f`, and does nothing for `PHASE_DIGGER_TUNNELING`.  `StopEating` (`:6794-6816`):
clears `mIsEating` and **calls `StartWalkAnim(20)` which calls `PickRandomSpeed()`** (§1.4) —
a fresh `mVelX` draw on every resume, plus a fresh `anim_walk2`/`anim_walk` choice (`:6772-6792`).

Because eating is gated on `mZombieAge % 4`, and `mZombieAge` starts at 0 and is incremented in
`Update` *before* `CheckIfPreyCaught` (`:4309` vs `:4648`), a zombie born at age 0 eats on
updates where `mZombieAge ∈ {4, 8, 12, …}` — i.e. the first possible bite is on its
**4th update**, and only if `mZombieAge % 4 == 0` happens to coincide with the mouth being
inside the plant rect.  The cadence is **phase-locked to the zombie's own age**, not to the
moment it started eating: a zombie that walks 137 updates before its first bite is offset
`137 % 4 = 1` in its bite phase.  `SimTick.inc:352-353` reproduces the phase-lock but only
starts counting `F_ZOMBIE_AGE` once the target exists (see §5).

### Step 6 — death
`TakeBodyDamage` reaches `<= 0` → `PlayDeathAnim(flags)` (`:7965-7970`).
`PlayDeathAnim` (`:8926-9057`):
```
if already DYING / BURNED / MOWERED: return                       # :8928-8930
if no reanim or no "anim_death" track: DieNoLoot(); return        # :8932-8936
if mIceTrapCounter > 0: mIceTrapCounter = 0                       # :8950-8955
StopEating()                                                      # :8976   (mIsEating = false)
if mShieldType != NONE: DropShield(1U)                            # :8978-8981
mVelX = 0.0f                                                      # :9057
mZombiePhase = PHASE_ZOMBIE_DYING                                 # :9058
aDeathAnimRate = RandRangeFloat(24.0f, 30.0f)  # for normal/cone/flag/pail/pole-vaulter  # :9033
aDeathTrack    = "anim_death"                                     # :9037
aDeathAnimHit  = Rand(100)                                        # :9038
aCanDoSuperLongDeath = mApp->HasFinishedAdventure() or mBoard->mLevel > 5    # :9039
   # level 7 > 5  → TRUE (this is a level-7-relevant branch)
if aDeathAnimHit == 99 and has "anim_superlongdeath" and aCanDoSuperLongDeath
   and mChilledCounter == 0 and mBoard->CountZombiesOnScreen() <= 5:
        aDeathAnimRate = 14.0f; aDeathTrack = "anim_superlongdeath"   # :9046-9050
elif aDeathAnimHit > 50 and has "anim_death2": aDeathTrack = "anim_death2"   # :9051-9054
PlayZombieReanim(aDeathTrack, REANIM_PLAY_ONCE_AND_HOLD, 20, aDeathAnimRate)  # :9056
```
`UpdateDeath` (`:9095-9312`) then runs each update (step 2 DYING branch), and the only part that
affects the simulation is the tail (`:9308-9311`):
```
elif mZombieFade == -1 and aBodyReanim->mLoopCount > 0 and mZombieType != ZOMBIE_BOSS:
        mZombieFade = mInPool ? 10 : 100
```
`mZombieFade` is initialised to `-1` (`:203`).  On grass (`mInPool == false`) it becomes **100**,
and `Zombie::Update`'s tail (`:4382-4388`) then counts it down and calls `DieNoLoot()` at 0.

**Total dying lifetime = (one loop of the death animation) + 100 updates.**
The loop length is `mFrameCount / aDeathAnimRate * 100` updates (same formula the pole-vaulter
uses at `Zombie.cpp:1730`), with `aDeathAnimRate ∈ [24, 30)` — the anim frame count per death
track is `.reanim` data, **not determined here**.  `cDeathFadeTicks = 100` is therefore the
*second* half only; the CUDA core currently uses 100 as the whole dying duration.

While DYING the zombie: is still in `mZombies` and is still `!mDead`, so it **is updated**
(`Board.cpp:4981`), **is iterated by the mower** (`LawnMower.cpp:207`, only skips `mDead`),
**is skipped by `TakeDamage`** (`Zombie.cpp:7975` via `IsDeadOrDying()` at `:9643-9651`),
**counts for the left-zone event** (only `mDead` is skipped, `LawnApp.cpp:1561`), and
`UpdateZombieWalking()` is still called (`:4338`) — but `mVelX = 0` while the `_ground` branch
at `:4131-4134` reads the *death* anim's `mAnimRate` and the `_ground` track, so the corpse's
x drift is whatever the death track's ground displacement says.  **Not determined from source
alone**; the DYING zombie occupies its zone either way (the zone is `mRow`, never recomputed).

`IsDeadOrDying()` (`:9643-9651`) = `mDead || phase ∈ {DYING, BURNED, MOWERED}`.
`DieNoLoot()` (`:7420-7446`) is the only place `mZombiesKilled` increments
(`:7422-7423`, guarded by `if (!mDead)`), and it sets `mDead = true` (`:7427`).
`EffectedByDamage(DAMAGES_DYING)` (`:8084-8087`) lets an explosion hit a dying zombie; the
cherry bomb's flags decide — see `plants.md` / `projectiles.md`.

### Step 7 — mower interaction (`LawnMower::Update`, `:177-273`)
```
if state == MOWER_SQUISHED: --mSquishedCounter; if <= 0: Die(); return    # :179-187
if state == MOWER_ROLLING_IN:
    mRollingInCounter++
    mPosX = PvzpAnimateCurveFloat(0, 100, mRollingInCounter, -160.0f, -21.0f, CURVE_EASE_IN_OUT)  # :193
    if mRollingInCounter == 100: mMowerState = MOWER_READY                 # :194-197
    return
aAttackRect = GetLawnMowerAttackRect()   # = Rect(mPosX, mPosY, 50, 80)     # :419
for aZombie in mBoard->mZombies:
    if aZombie->mDead: continue                                            # :207
    if aZombie->mZombieType == ZOMBIE_BOSS: continue                       # :209
    if aZombie->mRow != mRow: continue                                     # :212
    if aZombie->mZombiePhase != PHASE_ZOMBIE_MOWERED
       and !aZombie->IsTangleKelpTarget()
       and aZombie->EffectedByDamage(127U):                                # :215-220
        aOverlap = GetRectOverlap(aAttackRect, aZombie->GetZombieRect())   # :222-223
        if aOverlap > 0:                                                   # :224   (balloon: > 20)
            if state != MOWER_READY or (type != BUNGEE and aZombie->mHasHead):
                MowZombie(aZombie)                                         # :228-231
if state not in {MOWER_TRIGGERED, MOWER_SQUISHED}: return                  # :237-240
aSpeed = 3.33f  (2.5f for LAWNMOWER_POOL)                                  # :242-246
mPosX += aSpeed                                                            # :251
if mPosX > WIDE_BOARD_WIDTH: Die()                                         # :267-270
```
**There is no `x <= 30` trigger.**  A READY mower sits at `mPosX = -21.0f` (end of the
rolling-in curve, `:193`), its attack rect is `[-21, +29]`, and a zombie's hurt rect is
`[mX+36, mX+78]` (`Zombie.cpp:199`).  With the x-only `GetRectOverlap`, `overlap > 0` requires

```
mX + 36 < 29   and   mX + 78 > -21      =>      mower triggers at mX ∈ [-99, -8]
```

i.e. the zombie must already be essentially at the house, not at x = 30.  The mower kills by
`Zombie::MowDown` (`:8584-8700`): for a normally walking zombie it drops shield/helm/flag/pole
and calls `DieWithLoot()` (which sets `mDead` immediately — a mowed zombie **skips** the
DYING/fade stage entirely, unlike a shot zombie).  Rolling speed `3.33f` px/update
(`:242`) and despawn at `mPosX > 800` (`:267`) **are** confirmed.

---

## 3. The `rendered x <= 160` left-zone predicate (`SimTick.inc:585-609`)

Authoritative build site, `LawnApp::EnvironmentPublicWaitState` (`LawnApp.cpp:1558-1571`):

```
std::set<int> identities;
for Zombie* zombie : mBoard->mZombies:
    if zombie->mDead: continue                                     # :1561
    identity = int(mBoard->ZombieGetID(zombie))                     # :1562
    if identity == 0 or identity == -1 or already-seen or !isfinite(zombie->mPosX):
        throw "invalid public zombie ID or coordinate for event wait"   # :1563-1565
    x = (abs(zombie->mPosX - 160.0) < 1.0)
          ? RenderedPublicCoordinate(zombie->mPosX)                 # :1566-1567
          : double(zombie->mPosX)
    if zombie->IsOnBoard() and x <= 160.0:                          # :1568
        result.leftZoneZombies.insert(identity)                      # :1569
```

* The membership filter is **`!mDead` only** (`LawnApp.cpp:1561`).  `IsDeadOrDying()`,
  `mHasHead`, `mZombiePhase` and `mIsEating` are **not** consulted.  A zombie in
  `PHASE_ZOMBIE_DYING` (fading, 100+ updates) at `x <= 160` **is a member of the set**, and it
  stays a member until `DieNoLoot()` sets `mDead`.  Resolves `SimTick.inc:595-596`.
* `Zombie::IsOnBoard()` (`Zombie.cpp:4284-4293`) is
  ```
  return !(mFromWave == ZOMBIE_WAVE_CUTSCENE or mFromWave == ZOMBIE_WAVE_UI)
  ```
  It is **not an x test at all** — it is a *provenance* test on which wave spawned the zombie.
  For adventure play every spawned wave is `>= 1`, so **every gameplay zombie has
  `IsOnBoard() == true` from the instant of construction**, at `mPosX = 800+`.
* The decimal substitute is applied only inside a **±1 px band** of 160 (`LawnApp.cpp:1566`),
  and `RenderedPublicCoordinate` round-trips through a `std::ostringstream`'s default
  six-significant-digit formatting (`EnvironmentWaitEvents.h:46-53`).  `SimTick.inc:585-590`
  matches this exactly.
* **Newly-entering** semantics, `Triggered()` (`EnvironmentWaitEvents.h:62-78`):
  ```
  if HasNew(before.leftZoneZombies, after.leftZoneZombies): "zombie_entered_left_zone"   # :68
  ```
  with `HasNew(a, b)` = "some element of `b` is not in `a`" (`:55-60`) — note the arguments are
  `(previous, current)`, so this is "the *current* set contains an identity the *previous* set
  lacked".  The comparison is by **public zombie identity**, not by slot or index.
  Also condition 4 (`left_zone_occupied`) is `!after.leftZoneZombies.empty()` (`:73`) and
  `initialConditionSatisfied` is `condition == 4 && !initial.leftZoneZombies.empty()` (`:91`)
  — the *initial* state is evaluated before any advance (`:90`, and the loop at `:92` does not
  run when `triggered` is already non-empty).
* Reason precedence when several fire in the same update (`:66-77`): `terminal` →
  `defense_lost` → `zombie_entered_left_zone` → `condition` → `max_ticks`; `reason` is
  `triggered.front()` (`:104`).
* Because identity is a monotone public id (`ZombieGetID`) and a freed slot's id is never
  reused inside one board, a **slot index is only a safe stand-in for identity if ids are
  allocated in slot order and a slot is never recycled while the old occupant is still in the
  set.**  `SimTick.inc:592-594` argues safety from the 780+ spawn x; that argument holds, but
  the real guarantee is the `identities.insert(...).second` throw at `LawnApp.cpp:1563`.

---

## 4. Level-7-specific consequences

* `mBoard->mLevel > 5` is true at 7 → `aCanDoSuperLongDeath == true` (`Zombie.cpp:9039`), so 1 %
  of deaths (`Rand(100) == 99`, `:9038,9046`) take the `anim_superlongdeath` track at rate 14.0
  **when** `CountZombiesOnScreen() <= 5` and `mChilledCounter == 0` (`:9046-9048`).  This is a
  2×-plus-longer dying lifetime on a random draw.  `UpdateDeath`'s `aFallTime` for
  NORMAL/FLAG/TRAFFIC_CONE is `0.77`, or `0.788` while `anim_superlongdeath` is playing, or
  `0.71` while `anim_death2` is (`Zombie.cpp:9145-9156`); pole vaulter `0.68` (`:9161-9164`).
  These only gate a foley sound (`:9217-9229`) — no state change — so they are safe to ignore.
* The pole vaulter is level 6+ (`gZombieDefs`, `Zombie.cpp:84`) and its `mFirstAllowedWave = 5`
  (`:84`), so it can only arrive in wave 5 or later; that is a `waves.md` fact.  Its **zombie-side**
  consequences are: `mPosX = WIDE_BOARD_WIDTH + 70 + Rand(10)` instead of `780 + Rand(40)`
  (`:383`), body 500 (`:378`), `mVelX = 0.66-0.68` (`:1202-1204`), house line `-150` (`:4545-4548`),
  mouth rect `Rect(-29, 0, 70, 115)` (`:390`), no chewing before the vault
  (`:6831-6832`), and the vault itself (`UpdateZombiePolevaulter`, `:1707-1735`) which sets
  `PHASE_POLEVAULTER_IN_VAULT`, `mHasObject = false`, `mVelX = aJumpDistance / aAnimDuration`
  with `aJumpDistance = mX - aPlant->mX - 80` and `aAnimDuration = mFrameCount / mAnimRate * 100`
  (`:1729-1733`), then on landing `mPosX -= 150.0f` (`:1768`) — a 150-px teleport.
* `mBoard->mDanceMode` swaps the walk anim for NORMAL/TRAFFIC_CONE/PAIL (`:6767-6770`) but is a
  mini-game flag, false in adventure 7.
* `IsLittleTroubleLevel()` (`:868-877`) scales to 0.5 and divides all four health pools by 4 —
  a challenge-mode flag, false at level 7, but the `mScaleZombie = 0.5f` at `:873` is the only
  place a level-7-relevant type would get a non-unit scale.
* Flag waves: `mPosX += 40` for every zombie whose `mFromWave` is a flag wave
  (`Zombie.cpp:236-239`), on top of the flag zombie's own `mPosX = WIDE_BOARD_WIDTH` (`:595`).
  The cadence itself is `waves.md`; **the consequence for the left-zone predicate is that a
  flag-wave zombie starts 40 px further right**, so it enters the zone 40/0.47·mVelX updates
  later than its siblings.
* `Zombie::ZombieTargetLeadX` (`:3932-3952`) — used by plant targeting for lead — is
  `speed = mVelX * CHILLED_SPEED_FACTOR if chilled`, negated if walking backwards, and **0 if
  `ZombieNotWalking()`**; then `aCurrentPosX = GetZombieRect().mX + GetZombieRect().mWidth/2`
  and `return aCurrentPosX - speed * theTime`.  Note it uses `mVelX` (not the 0.47-scaled
  `aSpeed`) and the **hurt-rect centre** (`mX+36+42 = mX+78`), not `mPosX`.  Relevant to
  `projectiles.md` aiming, listed here because the x it reports is not the x the zombie moves by.

---

## 5. What the current `cuda/SimTick.inc` gets wrong

Ordered by how much they move the leaf statistics.

**(1) `ZB_ON_BOARD` is being used as an x test; `IsOnBoard()` is a provenance test.**
`SimTick.inc:64-67` defines `cOnBoardX = 800.0f` and `SimTick.inc:325-329` sets the flag when
`x < 800`.  The engine's `Zombie::IsOnBoard()` (`Zombie.cpp:4284-4293`) is
`mFromWave != ZOMBIE_WAVE_CUTSCENE && mFromWave != ZOMBIE_WAVE_UI` — true from construction for
every gameplay zombie, at `mPosX = 780..819`.  Consequences: `TargetZombie` (`SimTick.inc:170-184`)
and the pea hit scan (`:287-295`) **cannot target a zombie still in `[800, 819]`**, i.e. every
zombie for its first ~0–40 px of life, and `LeftZoneMask` (`:604`) gates on it too.  The
original's shooters use `mBoard->mZombies` with no `IsOnBoard()` gate.  Fix: `ZB_ON_BOARD` should
be set unconditionally at spawn (from `F_ZOMBIE_FROM_WAVE >= 1`, which `SimTick.inc:496` already
stores) and the `x < 800` test removed.

**(2) Walk advance is missing the 0.47 factor.**
`SimTick.inc:369` does `x -= speed` with `speed = lane.f(F_ZOMBIE_VELX)` (0.23–0.37 px/update).
The engine's live path is `aSpeed = GetTrackVelocity("_ground") * mScaleZombie`
(`Zombie.cpp:4131-4134`), which with `mAnimRate = mVelX * (frameCount/distance) * 47 / mScaleZombie`
(`Zombie.cpp:6691`) and `SECONDS_PER_UPDATE = 0.01` (`Reanimator.h:52`) averages **0.47 · mVelX**
px/update (§1.4).  The current core walks **2.13× too fast**, so every zombie reaches the house,
the mowers and every plant row ~2× early; win/loss and "zombie advance" — the headline leaf
statistics — cannot match.  The 2.13× is exactly `1/0.47`, which is the `47 × SECONDS_PER_UPDATE`
in those two lines.

**(3) Eating is gated on a counter that starts when the target is acquired, not on `mZombieAge % 4`.**
`SimTick.inc:352-353` increments `F_ZOMBIE_AGE` *only inside the `target >= 0` branch* and uses
`age % cEatInterval != 0`.  The engine increments `mZombieAge` on **every** update
(`Zombie.cpp:4309`) — including the walking-in updates and the updates spent eating — and tests
`mZombieAge % aTicksBetweenEats` (`:6867`).  Two differences: (a) the current core's bite phase is
always 0 relative to "first contact", whereas the original's is `contact_age % 4`, effectively
random per zombie; (b) the current core does not chill the cadence — the original doubles the
interval to 8 while `mChilledCounter > 0` (`:6863-6866`) **and additionally skips odd ages** in
`EatPlant` (`:7092-7095`), so a chilled zombie bites at a quarter the rate, not a half.

**(4) The eat reach window is 80 px wide; the engine's is 41 px wide with a 39-px dead gap.**
`SimTick.inc:341` computes `FeatureColumn(x + 40.0)` = `floor(mPosX / 80)` and eats whatever plant
is in that cell.  The engine requires `GetRectOverlap(mouth[50..70], plant_rect) >= 20`
(`Zombie.cpp:6435`) which is the mouth **fully inside** the plant's 60 px rect, i.e.
`mX ∈ [80c, 80c+40]` (§2 step 5).  The current core therefore starts biting ~40 px (≈ 2.3–3.7
engine-updates × the *current* wrong speed) too early and never has the inter-plant gap.  Also
`FeatureColumn` clamps to 0..9 while the lawn is `MAX_GRID_SIZE_X = 9` columns (`Board.h:42`);
`SimTick.inc:343`'s `col < kGridCols` guard saves it but the row-major cell index
`row * kCols + col` uses `kCols = 10` while the *legal* columns are 0..8 — worth an assertion.

**(5) Death duration is one constant where the engine has two stages.**
`SimTick.inc:33,166,425` use `cDeathFadeTicks = 100` as the whole dying lifetime.  The engine's
dying lifetime is **one loop of the death animation** at `aDeathAnimRate = RandRangeFloat(24,30)`
(`Zombie.cpp:9033`, chosen track `anim_death` / `anim_death2` (50 % of hits, `:9051`) /
`anim_superlongdeath` (1 % and `mLevel > 5`, i.e. active at 7, `:9039-9050`)) **plus 100** fade
updates (`:9308-9311`, `mZombieFade = mInPool ? 10 : 100`), the 100 then counted down at
`:4382-4388`.  Loop length in updates is `mFrameCount / aDeathAnimRate * 100` (same expression as
`Zombie.cpp:1730`).  With 100 updates/second, a 24-frame death anim at rate 24 is 100 updates, so
the true dying lifetime is roughly **150–200 updates**, random in both halves, and ~2× longer on
the 1 % super-long draw.

**(6) A DYING zombie is being excluded from the left-zone set.**
`SimTick.inc:605` drops `ZB_DEAD | ZB_DYING`; `LawnApp.cpp:1561` drops only `mDead`, which the
core already encodes as `ZB_DEAD`.  `ZB_DYING` must stay in the set — for 150–200 updates a
killed zombie at `x <= 160` is a member, so the wait round ends early on it.  This shifts
`zombie_entered_left_zone` timing, and `WaitRound` (`SimTick.inc:626-627`) tests the diff of that
very set.  `ZombieLive` (`SimTick.inc:94-97`) is used both for the event and for
`CheckVictory` (`:540-544`): the engine's victory sweep is `!mDead` (`Board.cpp:5076-5079`
pattern) — a fading corpse should **not** block a win — so those two uses need separating.

**(7) The mower trigger threshold.**
`SimTick.inc:49` `cMowerTriggerX = 30.0f`, applied as `lane.f(F_ZOMBIE_X) > cMowerTriggerX`
(`:402`).  There is no such test in the engine: `LawnMower::Update` triggers on rect overlap of
`Rect(mPosX, mPosY, 50, 80)` (`LawnMower.cpp:419`) with `Zombie::GetZombieRect()` (`:222-224`),
which for a READY mower parked at `mPosX = -21.0f` (`LawnMower.cpp:193`) and a hurt rect
`[mX+36, mX+78]` (`Zombie.cpp:199`) resolves to **`mX ∈ [-99, -8]`** (§2 step 7).  With the
house line at `-100` (`GameConstants.h:32`), a mower currently triggers ~130 px too early.
Two more differences: the mower kills by `MowZombie → Zombie::MowDown → DieWithLoot()`
(`LawnMower.cpp:228-231`, `Zombie.cpp:8584+`), not by zeroing the health fields
(`SimTick.inc:421-425`), and it only hits zombies passing `EffectedByDamage(127U)`
(`LawnMower.cpp:219`) which excludes dying ones unless `DAMAGES_DYING` is in that mask — decode
in `projectiles.md`.  Also, **mowers update after zombies in the same tick**
(`Board.cpp:5000-5005`), which `SimTick.inc:557-559` does get right.

**(8) `cSpawnJitter` / spawn x for the flag and pole-vaulter types is wrong.**
`SimTick.inc:65,498` give every zombie `780 + Rand(40)`.  Flag zombies spawn at exactly
`WIDE_BOARD_WIDTH = 800` (`Zombie.cpp:595`) and then `+40` if the wave is a flag wave
(`:236-239`); pole vaulters spawn at `870 + Rand(10)` (`:383`).  Vaulters also get a fixed
`mVelX` range (`0.66-0.68`, `:1202-1204`) not `def.velMin/velMax`-or-`0.3f`, and
`SimTick.inc:501-502` falls back to a hardcoded `0.3f` for anything with `velMax <= velMin`,
which is the wrong default for the flag zombie (0.45 exact, `:1193-1196`).

**(9) `mVelX` is treated as immutable, and `PickRandomSpeed` is re-rolled on resume.**
`SimTick.inc:502` draws the speed once and `:367` re-reads it every tick.  `StopEating` →
`StartWalkAnim(20)` → `PickRandomSpeed()` (`Zombie.cpp:6806,6739,1165`) re-draws it every time a
zombie finishes a chew.  For a NORMAL zombie that is a fresh uniform in `[0.23, 0.37]`
(≈ 60 % spread), so a long game with several eaten plants has a *materially* different x trace
under the current core.

**(10) Head loss and bleed-out are absent.**
`SimTick.inc:142-167` (`DamageZombie`) subtracts straight into `F_ZOMBIE_BODY` and dies at `<= 0`
with no intermediate state.  The engine drops the arm at `< 2·max/3` and the **head** at
`< max/3` (`Zombie.cpp:3910-3916`), after which the zombie keeps walking until either it is
finished off by bleed-out (`Rand(5) == 0` → 1 dmg, or 3 dmg when `mBodyMaxHealth >= 500`,
`:4667-4677`) or it walks to `mX <= -30` and takes a scripted 1800 (`:4566-4569`).  The headless
window is **90 body HP wide** for normal/cone/flag (270 → 90) and **166** for the vaulter, i.e.
5 pea hits.  Ignoring it makes peashooter DPS against the last 1/3 of a zombie's health effectively
free, and it removes a `Rand(5)` draw from every kill.

**(11) Minor / confirmed-correct.**
`cEatInterval = cEatDamage = 4` (`SimTick.inc:27-28`) — confirmed (`Zombie.cpp:57-58`).
`cChilledSpeedFactor = 0.4f` (`:30`) — confirmed (`:69`).  `cChilledTicks = 400` (`:31`) is
**wrong for a snow pea**: `ApplyChill(false)` gives 1000 (`:7534-7539`), 2000 only for an ice trap.
`cHouseX = -100.0f` (`:63`) is right for normal/cone/flag but wrong for the pole vaulter
(`-150`, `:4545-4548`), and it must be compared against `int(x)` (`:4390,4556`), not the float.
`cChewStepX = 0.0f` (`:74`) — right value, wrong cited reason: there is no step pair in the
position code; the advance is zero because `ZombieNotWalking()` returns true for `mIsEating`
(`:3956-3959`) and `UpdateZombieWalking` bails at `:4106`.
`cMowerSpeed = 3.33f` (`:47`) and `cMowerDespawnX = 800.0f` (`:48`) — confirmed
(`LawnMower.cpp:242,267`).  `cMowerRollInTicks = 60` (`:50`) — **wrong**, it is 100
(`LawnMower.cpp:192-197`), and the x is an ease-in-out curve from `-160` to `-21`, not linear.
`DamageZombie`'s helm-before-shield order (`SimTick.inc:149-161`) is inverted relative to the
engine (flying → **shield** → helm → body, `Zombie.cpp:7980-8002`); harmless at level 7 (no
shields) but do not let a later level make it load-bearing.

### Still not determined (do not guess these)

* The `_ground` track's per-frame x displacements for `anim_walk`/`anim_walk2`/`anim_eat`/
  `anim_death*` — they live in the `.reanim` data, not in `src/`.  Needed for bit-exact x; the
  *mean* (`0.47 · mVelX`) is derivable (§1.4).  Start at `src/PvzpLib/Reanimator.cpp` and the
  resource loader that produces `mDefinition->mTracks`.
* `mFrameCount` of each death track (the dying-lifetime loop length, §5 item 5).
* `Board::CountZombiesOnScreen()` and `HasLevelAwardDropped()` as evaluated inside a headless
  rollout — they gate the super-long death (`Zombie.cpp:9046-9048`).
* The `DamageFlags`/`DamageRangeFlags` bit decode behind `9U` and `127U`
  (`Zombie.cpp:4676,7129`, `LawnMower.cpp:219`) — belongs to `projectiles.md`.
* Whether the exporter ever hands a lane a zombie with `mFromWave < 1`
  (`Zombie.cpp:79-83`: `ZOMBIE_WAVE_CUTSCENE = -2`, `ZOMBIE_WAVE_UI = -3`), which is the only way
  `IsOnBoard()` can be false (§5 item 1).
