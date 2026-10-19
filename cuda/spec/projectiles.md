# Projectiles — flight, collision, impact (PROJECTILE_PEA / PROJECTILE_SNOWPEA)

## 2026-10-18 implementation corrections

`UpdateNormalMotion` moves mPosX and calls CheckForCollision (:735), which uses GetProjectileRect's
integer **mX from the previous tick**. Only after that return does UpdateMotion set mX=int(mPosX)
at :773. The CUDA core previously collided at the advanced position, one tick early. It now keeps
the pre-motion integer for the rectangle, while the off-board test still reads the advanced mPosX.
The off-board threshold is **800**, not 860 (:274). The hit scan must also honor
EffectedByDamage(1): skip dying/burned/mowed, off-board hurt rects and airborne pole-vaulters.
SimMechanicsTest pins a real pea at 383.87952 and zombie at 391.53726: first update cannot hit;
the second may hit. Short-wait original/host/GPU leaf comparisons verify the causal effect.

All citations are repo-relative to `/home/newbiexvwu/pvz-cuda`. Line numbers are the current working tree.
Every constant below is copied literally from the source; no value is inferred from commercial PvZ knowledge.
Where the source is silent, the line reads `NOT FOUND in <file>`.

Target environment assumed throughout: **adventure level 7, `BACKGROUND_1_DAY`** (daytime lawn, no pool, no roof, no fog),
deck = peashooter / snowpea / repeater. The consequences of that choice are listed in §0 and are what let several
branches be marked dead.

---

## 0. Scope: which projectile types can exist at all

`ProjectileType` values (`src/ConstEnums.h:800-817`) and every producer of each type (all `AddProjectile` call
sites in the tree: `src/Lawn/Plant.cpp:929`, `src/Lawn/Plant.cpp:4693`, `src/Lawn/Zombie.cpp:1540`,
`src/Lawn/Zombie.cpp:2344`, `src/Lawn/Zombie.cpp:2349`, `src/Lawn/Zombie.cpp:2353`, `src/Lawn/Zombie.cpp:2464`,
`src/Lawn/Zombie.cpp:2469`, `src/Lawn/Zombie.cpp:2473`, `src/Lawn/Board.cpp:2346`):

| type | producer | reachable with deck {peashooter, snowpea, repeater} on level 7 |
|---|---|---|
| `PROJECTILE_PEA` (0) | `Plant::Fire` — `SEED_PEASHOOTER`, `SEED_REPEATER` (and `THREEPEATER`/`SPLITPEA`/`GATLINGPEA`/`LEFTPEATER`) map to it (`src/Lawn/Plant.cpp:4513-4520`); also `Zombie::UpdateZombiePeaHead` / `UpdateZombieGatlingHead` when `mMindControlled` under `#ifdef DO_FIX_BUGS` (`src/Lawn/Zombie.cpp:2341-2346`, `:2464-2466`) | **YES** (the zombie-sourced variant needs mind control, which this deck has no source for) |
| `PROJECTILE_SNOWPEA` (1) | `Plant::Fire` — `SEED_SNOWPEA` (`src/Lawn/Plant.cpp:4521-4523`) | **YES** |
| `PROJECTILE_CABBAGE` (2) | `SEED_CABBAGEPULT` (`src/Lawn/Plant.cpp:4533-4534`) | NO — seed not in deck |
| `PROJECTILE_MELON` (3) | `SEED_MELONPULT` (`src/Lawn/Plant.cpp:4539-4540`) | NO |
| `PROJECTILE_PUFF` (4) | `SEED_PUFFSHROOM` / `SEED_SCAREDYSHROOM` / `SEED_SEASHROOM` (`src/Lawn/Plant.cpp:4524-4528`) | NO |
| `PROJECTILE_WINTERMELON` (5) | `SEED_WINTERMELON` (`src/Lawn/Plant.cpp:4542-4543`) | NO |
| `PROJECTILE_FIREBALL` (6) | **never spawned**; only `Projectile::ConvertToFireball` (`src/Lawn/Projectile.cpp:1197-1219`), called only from `Plant::UpdateTorchwood` (`src/Lawn/Plant.cpp:1393`) | NO — `SEED_TORCHWOOD` not in deck |
| `PROJECTILE_STAR` (7) | `Plant::StarFruitFire` (`src/Lawn/Plant.cpp:929`), `SEED_STARFRUIT` | NO |
| `PROJECTILE_SPIKE` (8) | `SEED_CACTUS` / `SEED_CATTAIL` (`src/Lawn/Plant.cpp:4529-4531`) | NO |
| `PROJECTILE_BASKETBALL` (9) | `Zombie.cpp:1540` (catapult zombie) | not deck-producible |
| `PROJECTILE_KERNEL` (10) | `SEED_KERNELPULT` (`src/Lawn/Plant.cpp:4536-4537`) | NO |
| `PROJECTILE_COBBIG` (11) ("cook"/corn cob) | `SEED_COBCANNON` (`src/Lawn/Plant.cpp:4545-4546`) | NO |
| `PROJECTILE_BUTTER` (12) | `SEED_KERNELPULT` + `WEAPON_SECONDARY` (`src/Lawn/Plant.cpp:4552-4555`) | NO |
| `PROJECTILE_ZOMBIE_PEA` (13) | `Zombie::UpdateZombiePeaHead` (`src/Lawn/Zombie.cpp:2349`, `:2353`) and `Zombie::UpdateZombieGatlingHead` (`src/Lawn/Zombie.cpp:2469`, `:2473`) | not deck-producible |

Butter / cabbage / corn-cob ("cook") / star paths, plus every `MOTION_LOBBED`, `MOTION_HOMING`, `MOTION_STAR`,
`MOTION_PUFF`, `MOTION_BACKWARDS`, `MOTION_FLOAT_OVER`, `MOTION_BEE*` branch, are **named here as unreachable and
are not specified further**. The only motion type a deck projectile can have is `MOTION_STRAIGHT`
(`src/Lawn/Projectile.cpp:76`; `Plant::Fire` overrides `mMotionType` only for the seed types at
`src/Lawn/Plant.cpp:4696-4782`, none of which are in this deck).

Terrain facts used repeatedly below:
* `Board::StageHasRoof()` is `BACKGROUND_5_ROOF || BACKGROUND_6_BOSS` (`src/Lawn/Board.cpp:8679-8682`) → false.
* `Board::StageHasPool()` is `BACKGROUND_3_POOL || BACKGROUND_4_FOG` (`src/Lawn/Board.cpp:8684-8687`) → false.
* `Board::GridToPixelY(gridX, gridY)` on a stage with neither roof nor pool returns `theGridY * 100 + LAWN_YMIN`
  (`src/Lawn/Board.cpp:8937-8944`, `LAWN_YMIN = 80` at `src/GameConstants.h:36`), minus 30 only for
  `GRIDSQUARE_HIGH_GROUND` (`src/Lawn/Board.cpp:8946-8949`). It therefore **does not depend on gridX** on this stage.
* Every cell of a daytime lawn is `GRIDSQUARE_GRASS` (`src/Lawn/Board.cpp:104`, and the override loop
  `src/Lawn/Board.cpp:1064-1081` only fires for DIRT/POOL/HIGH_GROUND rows) → `GRIDSQUARE_HIGH_GROUND` never occurs.
* `BOARD_WIDTH = WIDE_BOARD_WIDTH = 800`, `LAWN_XMIN = 40`, `HIGH_GROUND_HEIGHT = 30`
  (`src/GameConstants.h:28-37`). `MAX_GRID_SIZE_X = 9`, `MAX_GRID_SIZE_Y = 6` (`src/Lawn/Board.h:42-43`).

### 0.1 The "age 55" claim is FALSE

Every read of `mProjectileAge` in the whole tree (`grep mProjectileAge src/`):

| site | comparison | applies to pea/snowpea? |
|---|---|---|
| `src/Lawn/Projectile.cpp:266` | `mMotionType == MOTION_PUFF && mProjectileAge >= 75` → `Die()` | no (PUFF motion) |
| `src/Lawn/Projectile.cpp:244` | `PROJECTILE_STAR && mProjectileAge < 25 && mVelX >= 0.0f` | no (STAR) |
| `src/Lawn/Projectile.cpp:536` | `if (mProjectileAge > 20)` inside `UpdateLobMotion` | no (LOBBED) |
| `src/Lawn/Projectile.cpp:657` | `aMotion += aToTarget * (0.001f * mProjectileAge)` (HOMING) | no |
| `src/Lawn/Projectile.cpp:684`, `:703` | `if (mProjectileAge < 60)` (BEE / BEE_BACKWARDS) | no |
| `src/Lawn/Projectile.cpp:942` | `mProjectileAge++` (write) | **yes** |
| `src/Lawn/Projectile.cpp:960` | `if (mProjectileAge > aTime)` with `aTime = 0` for PEA/SNOWPEA (`:946-959`) → only rewrites `mRenderOrder` | **yes, presentation only** |
| `src/Lawn/Projectile.cpp:1005` | `PvzpAnimateCurveFloat(0, 30, mProjectileAge, ...)` in `Draw`, PUFF only | no |
| `src/Lawn/System/SaveGame.cpp:921` | serialize | n/a |
| `src/CudaExport.cpp:257`, `src/LawnApp.cpp:1900` | observation export | n/a |

**There is no threshold 55 anywhere.** `55` does not appear as a projectile-age comparison in
`src/Lawn/Projectile.cpp`, `src/Lawn/Board.cpp`, or `src/Lawn/Plant.cpp` — NOT FOUND in those files.
The stale claim lives in a comment in the new CUDA header (`cuda/CudaRollout.h:157`:
`int32_t age; // mProjectileAge (snow/pea decoy shot gate at 55)`) and is **not backed by the source**.
For PEA/SNOWPEA the age gate at `src/Lawn/Projectile.cpp:960` fires on the very first update
(`mProjectileAge` is incremented at `:942` before the test, so `1 > 0` is true immediately) and changes
nothing but `mRenderOrder`. Age gates **no gameplay logic at all** for these two types; the only role age plays is
being incremented and exported.

---

## 1. `Projectile::Update` — the flight model

`src/Lawn/Projectile.cpp:940-973`, called exactly once per logic tick per live projectile from
`Board::UpdateGameObjects` (`src/Lawn/Board.cpp:4988-4993`, skipping `mDead`). There is **no sub-stepping and no
multiplier**: `Board::UpdateGameObjects` is called once from `Board::UpdateGame` (`src/Lawn/Board.cpp:5662`),
which is called once from `Board::Update` (`src/Lawn/Board.cpp:5815`).

```
void Projectile::Update()                                                       // Projectile.cpp:940
{
    mProjectileAge++;                                                           // :942   int32 += 1
    if (mApp->mGameScene != SCENE_PLAYING && !mBoard->mCutScene->ShouldRunUpsellBoard())
        return;                                                                 // :943-944 age still counts, position does NOT move
    int aTime = 20;                                                             // :946   int
    if (type in {PEA, SNOWPEA, CABBAGE, MELON, WINTERMELON, KERNEL, BUTTER,
                 COBBIG, ZOMBIE_PEA, SPIKE}) aTime = 0;                          // :947-959  (PEA :947, SNOWPEA :948)
    if (mProjectileAge > aTime)                                                 // :960
        mRenderOrder = Board::MakeRenderOrder(RENDER_LAYER_PROJECTILE, mRow, 0); // :962  presentation
    if (mClickBackoffCounter > 0) mClickBackoffCounter--;                        // :965-968  dead code, see §6
    mRotation += mRotationSpeed;                                                 // :969   float += float (both 0.0f for pea/snowpea)
    UpdateMotion();                                                              // :971
    AttachmentUpdateAndMove(mAttachmentID, mPosX, mPosY + mPosZ);                 // :972   presentation, ignored
}
```

`UpdateMotion` (`src/Lawn/Projectile.cpp:739-775`) for a non-LOBED motion:

```
void Projectile::UpdateMotion()                                                 // :739
{
    if (mAnimTicksPerFrame > 0) { ... }                                         // :741-745  NEVER taken: mAnimTicksPerFrame == 0
                                                                                //           for every projectile (:98) and never written
                                                                                //           elsewhere -> NOT FOUND in src/
    int   aOldRow = mRow;                                                       // :747   int32
    float aOldY   = mBoard->GetPosYBasedOnRow(mPosX, mRow);                       // :748   float
    // mMotionType == MOTION_LOBBED ? no : ->
    UpdateNormalMotion();                                                       // :755
    float aSlopeHeightChange = mBoard->GetPosYBasedOnRow(mPosX, aOldRow) - aOldY; // :758
    // type != COBBIG so no override at :759-762
    // mMotionType != MOTION_FLOAT_OVER -> :763-766 skipped
    // mMotionType != MOTION_LOBBED     -> :767-771 skipped
    mShadowY += aSlopeHeightChange;                                              // :772   float += float
    mX = static_cast<int>(mPosX);                                                // :773   float -> int32, truncation toward zero
    mY = static_cast<int>(mPosY + mPosZ);                                        // :774   float add, then truncation
}
```

`GetPosYBasedOnRow(float thePosX, int theRow)` (`src/Lawn/Board.cpp:8893-8907`): on a non-roof stage it is
`return GridToPixelY(0, theRow);` → row-only, `thePosX` **ignored**. Both calls on `:748` and `:758` use the same
row (`aOldRow`, and the second uses `mRow == aOldRow` since straight motion never writes `mRow`), so
**`aSlopeHeightChange == 0.0f exactly` for a daytime-lawn straight pea, every tick.** (`:748` and `:758` both pass
`mPosX`, which changed in between, but the value is unused off-roof.)

`UpdateNormalMotion` (`src/Lawn/Projectile.cpp:640-737`) — the motion switch. For `MOTION_STRAIGHT` none of the
named branches match, so the fall-through `else` runs:

```
    else { mPosX += 3.33f; }            // src/Lawn/Projectile.cpp:716-719   float += float literal
```

Literal facts about the flight:

* **The velocity fields are NOT used for a straight pea.** `mVelX`, `mVelY`, `mVelZ`, `mAccZ` are all `0.0f`
  (`:70-73`) and are never written for PEA/SNOWPEA on this deck. `mVelX` is read only in
  `GetDamageFlags` (`:413`, MOTION_STAR only), `FindCollisionTarget` (`:244`, STAR only),
  `DoImpact` (`:842-843`, splat-particle position), and `UpdateNormalMotion`'s HOMING/STAR branches.
  **A CUDA implementation that advances projectiles by `mVelX` will produce frozen peas.** The speed is the
  hard-coded literal `3.33f` at `src/Lawn/Projectile.cpp:718`.
* `mPosY` is **never written** for `MOTION_STRAIGHT`: `:718` touches only `mPosX`; the `mPosY += mVelY` lines belong
  to BACKWARDS/HOMING/STAR/THREEPEATER branches (`:644`, `:666`, `:673`, `:712`), `mPosY -= 0.5f` to BEE (`:686`, `:705`),
  `mPosY += mVelZ` to the `GAMEMODE_CHALLENGE_HIGH_GRAVITY` block (`:721-733`, unreachable in adventure), and
  `mPosY += aSlopeHeightChange` to FLOAT_OVER/LOBBED only (`:765`, `:769`).
  **There is no gravity term in the straight-pea model.** `mAccZ` is only consumed in `UpdateLobMotion` (`:522`).
* `mPosZ` stays `0.0f`; so `mY == static_cast<int>(mPosY)` exactly.
* `mShadowY` stays constant (incremented by `0.0f`).
* Consequence: for a straight pea **only `mPosX`, `mX`, `mProjectileAge`, `mRenderOrder`, `mRotation` change per tick.**

Arithmetic types, per step: `mPosX += 3.33f` is `float += float` (the literal `3.33f` is `float`, i.e. the exact
value 3.33000004291534423828125). `mShadowY += aSlopeHeightChange` is `float += float`. `GetPosYBasedOnRow` returns
`float`, built from `int` (`GridToPixelY` returns `int32_t`, `src/Lawn/Board.cpp:8909`) promoted to `float`.
`mX`/`mY` are `int32_t` (`src/Lawn/GameObject.h:39-40`) assigned with `static_cast<int>`. `PI` is
`constexpr const double` (`src/GameConstants.h:25`) but is not touched by straight-pea flight.

### 1.1 Removal conditions

| condition | code | effect on PEA/SNOWPEA |
|---|---|---|
| out of bounds right/left | `if (mPosX > WIDE_BOARD_WIDTH || mPosX + mWidth < 0.0f) { Die(); return; }` — `src/Lawn/Projectile.cpp:272-276` | `mPosX > 800.0f` (float vs int 800) or `mPosX + 40.0f < 0.0f` → dies. This is the **only** lifetime limit for a straight pea: no age-based death exists (NOT FOUND in `src/Lawn/Projectile.cpp`). |
| PUFF lifetime | `src/Lawn/Projectile.cpp:266-270` (`MOTION_PUFF && age >= 75`) | unreachable |
| STAR off-screen y | `src/Lawn/Projectile.cpp:293-297` | unreachable |
| on hit | `DoImpact()` always ends with `Die();` — `src/Lawn/Projectile.cpp:937` | pea dies on every impact, including the `theZombie == nullptr` ground impacts |
| high-ground / altitude self-impact | `CheckForHighGround` → `DoImpact(nullptr)` — `src/Lawn/Projectile.cpp:351-388` | dead on this stage (§1.3) |

`Die()` (`src/Lawn/Projectile.cpp:1152-1165`) only sets `mDead = true` (`:1154`) and detaches the snow-pea trail
particle (`:1156-1160`, `SNOWPEA`/`PUFF` take the `AttachmentCrossFade("FadeOut") + AttachmentDetach` branch —
presentation, ignored). The object is returned to the `DataArray` later, in `Board::ProcessDeleteQueue`
(`src/Lawn/Board.cpp:8539-8547`), which runs at the **start of the next logic tick**
(`LawnApp::AdvanceLogicTick`, `src/LawnApp.cpp:1581-1589`: `++mAppCounter; mBoard->ProcessDeleteQueue(); …;
SexyApp::UpdateFrames();` → `WidgetManager::UpdateFrame()` → `Board::Update()`).

Tick order inside one tick (`src/Lawn/Board.cpp:4972-5010`): plants → zombies → **projectiles** → coins → mowers.
Because plants run first, a pea created by `Plant::Fire` during the plant phase **is updated in the same tick**
(`DataArray::begin()` is evaluated after the plant loop; iteration is index-ascending over live slots,
`src/PvzpLib/DataArray.h:135-171`). So a pea's first `+3.33f` and its first collision test happen on its spawn tick,
and `mProjectileAge == 1` at the end of that tick.

### 1.2 "A pea passes over a plant" — yes, unconditionally

A straight pea **never** collides with any plant, including a wallnut standing in the same lane as the shooting
peashooter. The deciding code:

* `Projectile::CheckForCollision` only tests plants when the type is `PROJECTILE_ZOMBIE_PEA`
  (`src/Lawn/Projectile.cpp:309-323`). `FindCollisionTargetPlant()` (`:157-194`) is called from exactly two places:
  that ZOMBIE_PEA branch (`:311`) and `UpdateLobMotion` for `BASKETBALL`/`ZOMBIE_PEA` (`:577-580`). Neither is
  reachable for PEA/SNOWPEA.
* After that branch, the only target search a straight pea performs is `FindCollisionTarget()` over
  `mBoard->mZombies` (`src/Lawn/Projectile.cpp:325`).
* There is no height/altitude test against plants for straight peas: `mPosZ` stays `0.0f` and `mPosY` never changes.
* The single plant that can affect a pea is torchwood: `Plant::UpdateTorchwood` iterates all live projectiles and
  converts `PROJECTILE_PEA → PROJECTILE_FIREBALL` / `PROJECTILE_SNOWPEA → PROJECTILE_PEA` when
  `GetRectOverlap(plantAttackRect, projectileRect) >= 10` and `aProjectile->mRow == mRow`
  (`src/Lawn/Plant.cpp:1377-1402`, conversions at `:1393` / `:1397`). With this deck `SEED_TORCHWOOD` cannot be
  planted, so this loop never matches and both conversions are unreachable.
* Its mirror-image guard, `Projectile::PeaAboutToHitTorchwood()` (`src/Lawn/Projectile.cpp:196-222`), is called at
  the top of `FindCollisionTarget` and returns `nullptr` (skipping all zombie collision, the "torchwood clip") when
  a same-row torchwood is within reach (`:226-227`). Its body requires `aPlant->mSeedType == SEED_TORCHWOOD`
  (`:208`), so **on this deck it always returns false after one pass over `mBoard->mPlants`**.
  (`:212` `aProjectileRect.mX += 40;` and `:214` `GetRectOverlap(...) > 10` are the only uses of those numbers;
  they cannot be reached here.)

### 1.3 The two height gates that depend on spawn geometry

`src/Lawn/Projectile.cpp:299-302` (inside `CheckForCollision`, before any zombie test):

```
if ((mProjectileType == PROJECTILE_PEA || mProjectileType == PROJECTILE_STAR) && mShadowY - mPosY > 90.0f)
    return;      // skip collision for this tick
```

**`PROJECTILE_SNOWPEA is not in this list** (`:299` names only PEA and STAR), so a snow pea is never gated by it.

`src/Lawn/Projectile.cpp:351-366` (inside `CheckForHighGround`, run after `CheckForCollision` — `:735-736`):

```
float aShadowDelta = mShadowY - mPosY;                                          // :353
if (type in {PEA, SNOWPEA, FIREBALL, SPIKE, COBBIG}) { if (aShadowDelta < 28.0f) { DoImpact(nullptr); return; } }  // :355-365
if (CantHitHighGround()) {                                                      // :380
    int aGridX = mBoard->PixelToGridXKeepOnBoard(mPosX + 30, mPosY);             // :382
    if (mBoard->mGridSquareType[aGridX][mRow] == GRIDSQUARE_HIGH_GROUND) DoImpact(nullptr);  // :383-386
}
```

`CantHitHighGround()` (`src/Lawn/Projectile.cpp:337-349`) is `true` for PEA and SNOWPEA whenever `!mOnHighGround`
and motion is not BACKWARDS/HOMING; `mOnHighGround` is set once at init from the spawn cell's grid type
(`:86`) and is false everywhere on a daytime lawn. `mGridSquareType[x][y]` is never `GRIDSQUARE_HIGH_GROUND` on this
stage (§0), so **the `:380-386` branch never fires here**.

Both gates read the same spawn-time constant `D = mShadowY - mPosY`, and for `MOTION_STRAIGHT` on this stage neither
`mShadowY` nor `mPosY` is ever written again — so **`D` is frozen for the projectile's whole life** and the gates are
decided once, at spawn. Derivation for a plant in row `R` firing into its own row:

```
mShadowY = GridToPixelY(aGridX, R) + 67.0f            // Projectile.cpp:74  == R*100 + 80 + 67 = R*100 + 147
                                                    // (roof-only -12.0f adjust at :87-90 does not apply)
mPosY    = aOriginY = mY + aOffsetY - 33             // Plant.cpp:4607-4608, mY = R*100 + 80 (Plant.cpp:114-115)
D        = 100.0f - aOffsetY                          // aOffsetY = (int) peashooter "anim_stem" track TransY
```

The literal value of `aOffsetY` is **animation-data driven and NOT FOUND in `src/`** —
`Plant::GetPeaHeadOffset` reads it from the reanimation at fire time
(`src/Lawn/Plant.cpp:3673-3705`; `aTransform.mTransY` (float) is assigned into an `int&`, i.e. truncated at
`:3703-3704`; if the body reanim is unresolvable both offsets fall back to `0`, `:3686-3687`). The observable
source behaviour then implies `28.0f <= D <= 90.0f` (i.e. `10 <= aOffsetY <= 72`): if `D < 28.0f` a snow pea would
`DoImpact(nullptr)` and vanish on its first tick (`:361-364`), and if `D > 90.0f` a pea would never collide with
anything (`:299-302`). **Recommended CUDA modelling: treat both gates as inert for PEA/SNOWPEA on the daytime lawn,
and store `D` as a spawn constant if you later want roof/pool stages.** Do not implement `:380-386` at all for
level 7.

---

## 2. `Board::AddProjectile` and what the plant passes

```
// src/Lawn/Board.h:271
Projectile* Board::AddProjectile(int theX, int theY, int theRenderOrder, int theRow, ProjectileType theProjectileType);

// src/Lawn/Board.cpp:2346-2351
Projectile* Board::AddProjectile(int theX, int theY, int theRenderOrder, int theRow, ProjectileType theProjectileType)
{
    Projectile* aProjectile = mProjectiles.DataArrayAlloc();     // :2348  capacity 1024 (Board.cpp:81)
    aProjectile->ProjectileInitialize(theX, theY, theRenderOrder, theRow, theProjectileType);  // :2349
    return aProjectile;                                          // :2350  nothing else is set here
}
```

All parameters are `int` (`theRenderOrder` `int`, `theRow` `int`) — **the spawn position is integer before it ever
becomes a float.**

### 2.1 The peashooter-family call site

`Plant::Fire(Zombie* theTargetZombie, int theRow, PlantWeapon thePlantWeapon)` (`src/Lawn/Plant.cpp:4491`).
Type selection: peashooter/repeater → `PROJECTILE_PEA` (`:4513-4520`), snowpea → `PROJECTILE_SNOWPEA`
(`:4521-4523`). Spawn offsets (`src/Lawn/Plant.cpp:4603-4609`):

```
else if (mSeedType == SEED_PEASHOOTER || mSeedType == SEED_SNOWPEA || mSeedType == SEED_REPEATER)
{
    int aOffsetX, aOffsetY;
    GetPeaHeadOffset(aOffsetX, aOffsetY);        // reanim "anim_stem" (else "anim_idle") track TransX/TransY
    aOriginX = mX + aOffsetX + 24;               // :4607   all int arithmetic
    aOriginY = mY + aOffsetY - 33;               // :4608
}
```
(Gatlingpea uses `+34` on x at `:4621`; threepeater uses `mX + 45, mY + 10` at `:4641-4642`; the generic fallback is
`mX + 10, mY + 5` at `:4669-4670`. None of these are in this deck.)

```
if (mBoard->GetFlowerPotAt(mPlantCol, mRow)) aOriginY -= 5;      // :4672-4675  roof/flowerpot only -> dead on level 7
// :4677-4681 snow pea muzzle particle PARTICLE_SNOWPEA_PUFF at (aOriginX + 8, aOriginY + 13) -- presentation, ignored
Projectile* aProjectile = mBoard->AddProjectile(aOriginX, aOriginY, mRenderOrder - 1, theRow, aProjectileType);  // :4693
aProjectile->mDamageRangeFlags = GetDamageRangeFlags(thePlantWeapon);                                              // :4694
```

`GetDamageRangeFlags` (`src/Lawn/Plant.cpp:604-639`): peashooter/snowpea/repeater fall to `default: return 1;`
(`:636-637`). **`mDamageRangeFlags == 1`** = bit `DAMAGES_GROUND` only
(`src/ConstEnums.h:257-267`, `DAMAGES_GROUND = 0`). `theRow` is always the plant's own `mRow` for this deck
(`Plant::UpdateShooter` → `FindTargetAndFire(mRow, WEAPON_PRIMARY)`, `src/Lawn/Plant.cpp:978`, and the repeater's
second pea at `:986-991`); `theTargetZombie` is **not used at all** for non-lobbed, non-homing seeds
(`:4696-4738` consumes it only for melon/cabbage/kernel-pult and cattail).

Spawn timing (needed to reproduce pea emission exactly; plant-side, quoted for reference):
`FindTargetAndFire` returns false when no target zombie is found (`src/Lawn/Plant.cpp:732-736`), else
`mShootingCounter` is set — 35 for peashooter/snowpea (`:758`), 26 for repeater (`:759-763`) — and `Fire()` is *not*
called there; it is called from `Plant::UpdateShooting` when the counter reaches 1 (`:3248`, `:3303-3306`), i.e.
**34 ticks after target acquisition for peashooter/snowpea, 25 ticks for the repeater's burst** (the counter is
decremented at `:3210` on each following tick, `UpdateShooting` being called at `src/Lawn/Plant.cpp:2544` inside
`Plant::Update`, before `UpdateShooter()` at `:2580`). The repeater's second pea is re-triggered by
`if (mLaunchCounter == 25) … FindTargetAndFire(mRow, WEAPON_PRIMARY)` (`src/Lawn/Plant.cpp:986-991`), with
`mLaunchCounter = mLaunchRate - Sexy::Rand(15)` (`:947-950`) — note the RNG.

### 2.2 Fields the projectile receives (`ProjectileInitialize`, `src/Lawn/Projectile.cpp:63-155`)

```
int aGridX = mBoard->PixelToGridXKeepOnBoard(theX, theY);   // :65   = max(PixelToGridX(theX,theY), 0);  (Board.cpp:8812-8834)
mProjectileType = theProjectileType;                        // :66
mPosX = theX;   mPosY = theY;   mPosZ = 0.0f;               // :67-69   int -> float
mVelX = 0.0f;   mVelY = 0.0f;   mVelZ = 0.0f;  mAccZ = 0.0f; // :70-73
mShadowY = mBoard->GridToPixelY(aGridX, theRow) + 67.0f;    // :74   int + float -> float
mHitTorchwoodGridX = -1;                                     // :75
mMotionType = MOTION_STRAIGHT;                               // :76
mFrame = 0;  mNumFrames = 1;                                 // :77-78
mRow = theRow;                                               // :79
mCobTargetX = 0.0f;  mCobTargetRow = 0;                       // :80, :84
mDamageRangeFlags = 0;                                        // :81   (overwritten by Plant::Fire:4694 to 1)
mDead = false;                                                // :82
mAttachmentID = ATTACHMENTID_NULL;                            // :83
mTargetZombieID = ZOMBIEID_NULL;                              // :85
mOnHighGround = (mGridSquareType[aGridX][theRow] == GRIDSQUARE_HIGH_GROUND);   // :86  -> false here
if (StageHasRoof() && theX < 480) mShadowY -= 12.0f;          // :87-90  dead here
mRenderOrder = theRenderOrder;                                // :91   (= plant mRenderOrder - 1)
mRotation = 0.0f;  mRotationSpeed = 0.0f;                     // :92-93
mWidth = 40;   mHeight = 40;                                  // :94-95
mProjectileAge = 0;  mClickBackoffCounter = 0;  mAnimTicksPerFrame = 0;         // :96-98
switch (type): SNOWPEA -> add PARTICLE_SNOWPEA_TRAIL at (mPosX+8.0f, mPosY+13.0f) and attach (8.0f, 13.0f)  // :116-121  presentation
               default -> nothing                            // :148-149   PEA takes the default branch
mAnimCounter = 0;                                             // :152
mX = static_cast<int>(mPosX);  mY = static_cast<int>(mPosY);  // :153-154
```

`PROJECTILE_PEA` and `PROJECTILE_SNOWPEA` both take `default:` in the init switch except for the snow-pea trail:
**no rotation, no velocity, no size change.** (Contrast: CABBAGE/BUTTER set rotation at `:102-106`, COBBIG resizes at
`:125-129`, PUFF attaches a trail at `:130-135`, STAR adjusts shadow at `:140-147` — all unreachable.)

### 2.3 Damage is **not** stored on the projectile

`class Projectile` has no `mDamage` member (`src/Lawn/Projectile.h:48-76` lists every field; NOT FOUND in
`Projectile.h`). Damage is looked up by type every time:

```
const ProjectileDefinition& Projectile::GetProjectileDef()            // src/Lawn/Projectile.cpp:1232-1238
{
    const ProjectileDefinition& aProjectileDef = gProjectileDefinition[mProjectileType];   // :1234
    PVZP_ASSERT(aProjectileDef.mProjectileType == mProjectileType);                         // :1235
    return aProjectileDef;
}
```

```
// src/Lawn/Projectile.h:37-44
class ProjectileDefinition { ProjectileType mProjectileType; int32_t mImageRow; int32_t mDamage; };
extern const ProjectileDefinition gProjectileDefinition[NUM_PROJECTILES];

// src/Lawn/Projectile.cpp:37-52  (constinit table; enum indices from src/ConstEnums.h:800-817)
{ PROJECTILE_PEA,         mImageRow 0, mDamage 20 }   // :38
{ PROJECTILE_SNOWPEA,     mImageRow 0, mDamage 20 }   // :39
{ PROJECTILE_CABBAGE,     mImageRow 0, mDamage 40 }   // :40
{ PROJECTILE_MELON,       mImageRow 0, mDamage 80 }   // :41
{ PROJECTILE_PUFF,        mImageRow 0, mDamage 20 }   // :42
{ PROJECTILE_WINTERMELON, mImageRow 0, mDamage 80 }   // :43
{ PROJECTILE_FIREBALL,    mImageRow 0, mDamage 40 }   // :44
{ PROJECTILE_STAR,        mImageRow 0, mDamage 20 }   // :45
{ PROJECTILE_SPIKE,       mImageRow 0, mDamage 20 }   // :46
{ PROJECTILE_BASKETBALL,  mImageRow 0, mDamage 75 }   // :47
{ PROJECTILE_KERNEL,      mImageRow 0, mDamage 20 }   // :48
{ PROJECTILE_COBBIG,      mImageRow 0, mDamage 300 }  // :49
{ PROJECTILE_BUTTER,      mImageRow 0, mDamage 40 }   // :50
{ PROJECTILE_ZOMBIE_PEA,  mImageRow 0, mDamage 20 }   // :51
```

**`mDamage(PEA) = 20`, `mDamage(SNOWPEA) = 20`** (both `int32_t`).

Velocity constants for the two live types:

| quantity | value | source |
|---|---|---|
| x advance per tick | `+3.33f` (`float`) | `src/Lawn/Projectile.cpp:718` |
| y advance per tick | none — no statement exists | NOT FOUND in `src/Lawn/Projectile.cpp` for MOTION_STRAIGHT |
| gravity | none for straight motion (`mAccZ` only read in `UpdateLobMotion:522`) | `src/Lawn/Projectile.cpp:509-530` |
| `mVelX / mVelY / mVelZ / mAccZ` | `0.0f` forever | `src/Lawn/Projectile.cpp:70-73` |

---

## 3. Collision: the exact hit test and what happens on a hit

### 3.1 The rectangles

```
Rect Projectile::GetProjectileRect()                                            // src/Lawn/Projectile.cpp:1167
{
    if (type == PEA || type == SNOWPEA || type == ZOMBIE_PEA)
        return Rect(mX - 15, mY, mWidth + 15, mHeight);                          // :1173
    ...
}
```

`Rect` is `TRect<int32_t>` (`src/SexyAppFramework/misc/Rect.h:128`, fields `mX, mY, mWidth, mHeight` at `:39-42`).
For a pea: **`x ∈ [mX-15, mX+40]`, i.e. an extra 15 px of reach in front of the integer position and 15 px of
trailing slack; `y ∈ [mY, mY+40]`** (the y part is never compared — see below). `mX`, `mY` are the truncated
positions from `UpdateMotion:773-774`.

Overlap measure (`src/Lawn/Board.cpp:8969-8973`, free function, both args `const Rect&`):

```
int GetRectOverlap(const Rect& rect1, const Rect& rect2)
{
    return std::min(rect1.mX + rect1.mWidth, rect2.mX + rect2.mWidth) - std::max(rect1.mX, rect2.mX);
}
```

**X-axis only — all `int32_t` arithmetic. There is no y term whatsoever.** "Overlap" here means horizontal
interval intersection width, and the pea test accepts `>= 0`, i.e. **touching edges count as a hit**. Any vertical
mismatch is tolerated: the only lane discipline is the `mRow` equality test.

### 3.2 The candidate scan

`Projectile::CheckForCollision` (`src/Lawn/Projectile.cpp:264-335`), straight-pea path in order:

```
1  if (mMotionType == MOTION_PUFF && mProjectileAge >= 75) { Die(); return; }                 // :266-270   dead
2  if (mPosX > WIDE_BOARD_WIDTH || mPosX + mWidth < 0.0f) { Die(); return; }                   // :272-276
3  if (mMotionType == MOTION_HOMING) { ... return; }                                            // :278-291   dead
4  if (type == STAR && (mPosY > 600.0f || mPosY < 40.0f)) { Die(); return; }                    // :293-297   dead
5  if ((type == PEA || type == STAR) && mShadowY - mPosY > 90.0f) return;                       // :299-302   §1.3
6  if (mMotionType == MOTION_FLOAT_OVER) return;                                                // :304-307   dead
7  if (type == ZOMBIE_PEA) { …plant damage…; return; }                                          // :309-323   dead
8  Zombie* aZombie = FindCollisionTarget();                                                     // :325
   if (aZombie) {
       if (aZombie->mOnHighGround && CantHitHighGround()) return;                               // :328-331   dead (§1.3)
       DoImpact(aZombie);                                                                        // :333
   }
```

`FindCollisionTarget` (`src/Lawn/Projectile.cpp:224-262`) reproduced literally:

```
Zombie* Projectile::FindCollisionTarget()
{
    if (PeaAboutToHitTorchwood()) return nullptr;                 // :226-227  always false for this deck (§1.2)
    Rect aProjectileRect = GetProjectileRect();                   // :229
    Zombie* aBestZombie = nullptr;                                // :230
    int aMinX = 0;                                               // :231  (int; 0 is a dummy, always overwritten on first accept)
    for (Zombie* aZombie : mBoard->mZombies)                      // :233  index-ascending over live DataArray slots
    {                                                            //      (src/PvzpLib/DataArray.h:135-171)
        if (aZombie->mDead) continue;                             // :235-236
        if ((aZombie->mZombieType == ZOMBIE_BOSS || aZombie->mRow == mRow)      // :237  ROW EQUALITY (int32 ==)
            && aZombie->EffectedByDamage(static_cast<unsigned int>(mDamageRangeFlags)))   // :237, flags = 1
        {
            if (aZombie->mZombiePhase == PHASE_SNORKEL_WALKING_IN_POOL && mPosZ <= 45.0f) // :239-240
                continue;                                          //           pool-only -> never here
            if (type == STAR && mProjectileAge < 25 && mVelX >= 0.0f
                && aZombie->mZombieType == ZOMBIE_DIGGER) continue; // :244-247  dead
            Rect aZombieRect = aZombie->GetZombieRect();            // :249
            if (GetRectOverlap(aProjectileRect, aZombieRect) >= 0)   // :250  horizontal interval intersection >= 0
            {
                if (aBestZombie == nullptr || aZombie->mX < aMinX)   // :252  strict less-than on int32 mX
                {
                    aBestZombie = aZombie;                            // :254
                    aMinX = aZombie->mX;                              // :255
                }
            }
        }
    }
    return aBestZombie;                                             // :261
}
```

**Winner when several zombies overlap: the one with the smallest `mX` (i.e. furthest left / closest to the house).**
It is an argmin over the scan, not "first in list" and not "nearest to the pea". `aZombie->mX` is the zombie's
`int32_t` render x (`src/Lawn/GameObject.h:39`, refreshed each tick from `mPosX` at `src/Lawn/Zombie.cpp:4389-4390`),
*not* `mPosX`. Exact ties are broken by **list order** (strict `<` keeps the earlier slot). Row is decided by
`aZombie->mRow == mRow` — a boss bypasses the row test (`src/Lawn/Projectile.cpp:237`).

### 3.3 On a hit: `DoImpact` (`src/Lawn/Projectile.cpp:823-938`)

```
void Projectile::DoImpact(Zombie* theZombie)
{
    PlayImpactSound(theZombie);                                    // :825   presentation, ignored (see below)
    if (IsSplashDamage(theZombie)) { … }                           // :827-835  NEVER for PEA/SNOWPEA
    else if (theZombie)                                            // :836
    {
        unsigned int aDamageFlags = GetDamageFlags(theZombie);     // :838
        theZombie->TakeDamage(GetProjectileDef().mDamage, aDamageFlags);   // :839   damage = 20 (int32), flags unsigned int
    }
    …particle/reanim switch…                                       // :842-935  presentation
    Die();                                                          // :937  always, hit or not
}
```

* **Splash: none.** `IsSplashDamage` (`src/Lawn/Projectile.cpp:390-399`) is true only for
  `PROJECTILE_MELON`, `PROJECTILE_WINTERMELON`, `PROJECTILE_FIREBALL`. So `DoSplashDamage`
  (`:460-507`), `IsZombieHitBySplash` (`:426-458`) and the `mDamage`/3 scaling are unreachable for this deck.
  A snow pea therefore **never** damages a neighbouring lane or a neighbouring zombie.
* `GetDamageFlags(theZombie)` (`src/Lawn/Projectile.cpp:401-424`):
  ```
  unsigned int aDamageFlags = 0U;                                          // :403
  if (IsSplashDamage)                       SetBit(HITS_SHIELD_AND_BODY=1) // :405-408  no
  else if (LOBBED || BACKWARDS)             SetBit(BYPASSES_SHIELD=0)      // :409-412  no (MOTION_STRAIGHT)
  else if (STAR && mVelX < 0.0f)            SetBit(BYPASSES_SHIELD=0)      // :413-416  no
  if (type == SNOWPEA || type == WINTERMELON) SetBit(DAMAGE_FREEZE=2)      // :418-421
  return aDamageFlags;                                                     // :423
  ```
  **`PROJECTILE_PEA → aDamageFlags = 0u`. `PROJECTILE_SNOWPEA → aDamageFlags = 4u` (bit 2).**
  Bit indices from `enum DamageFlags : int32_t` (`src/ConstEnums.h:248-256`):
  `DAMAGE_BYPASSES_SHIELD=0, DAMAGE_HITS_SHIELD_AND_BODY=1, DAMAGE_FREEZE=2, DAMAGE_DOESNT_CAUSE_FLASH=3,
  DAMAGE_DOESNT_LEAVE_BODY=4, DAMAGE_SPIKE=5`; `SetBit`/`TestBit` are `theNum |= 1<<idx` / `theNum & (1<<idx)`
  (`src/PvzpLib/PvzpCommon.h:211-212`).
  Note the returned value is a *DamageFlags* set, **not** `mDamageRangeFlags`; the range flags (value 1) were only
  used for target filtering in §3.2.
* `PlayImpactSound` (`src/Lawn/Projectile.cpp:777-821`): for PEA/SNOWPEA nothing is played by the type switch
  (`:781-802` match only KERNEL/BUTTER/FIREBALL/MELON/WINTERMELON), then `aPlayHelmSound && theZombie` may fire
  `FOLEY_SHIELD_HIT` / `FOLEY_PLASTIC_HIT` by helm type (`:804-815`) and `FOLEY_SPLAT` otherwise (`:817-820`).
  **Presentation, ignored.**
* Particle switch: PEA → `aSplatPosX -= 15.0f; aEffect = PARTICLE_PEA_SPLAT` (`:864-867`); SNOWPEA → same with
  `PARTICLE_SNOWPEA_SPLAT` (`:868-871`). **Presentation, ignored.**
* Ground impacts with `theZombie == nullptr` (only from `CheckForHighGround`, dead here) deal **no damage at all**
  (`:836` requires a non-null zombie).

### 3.4 The chill effect (snow pea only)

`Zombie::TakeDamage(int theDamage, unsigned int theDamageFlags)` (`src/Lawn/Zombie.cpp:7973-8000`) — reproduced in
§4's guards; the chill reaches it through two of the four stages, and **only where the damage actually lands**:

```
if (theDamageFlags has DAMAGE_FREEZE /*bit 2*/) ApplyChill(false);
```
* in `Zombie::TakeHelmDamage` — `src/Lawn/Zombie.cpp:7731-7734` (before the helm-break check at `:7735`)
* in `Zombie::TakeBodyDamage` — `src/Lawn/Zombie.cpp:7831-7834`
* **not** in `Zombie::TakeShieldDamage` (`src/Lawn/Zombie.cpp:7603-…`, no `DAMAGE_FREEZE` test) and **not** in
  `Zombie::TakeFlyingDamage` (`:7806-7822`). So a snow pea fully absorbed by a screen door / ladder shield chills
  nothing.

```
void Zombie::ApplyChill(bool theIsIceTrap)                       // src/Lawn/Zombie.cpp:7522-7540
{
    if (!CanBeChilled()) return;                                  // :7524-7525
    if (mChilledCounter == 0) mApp->PlayFoley(FOLEY_FROZEN);       // :7527-7530  presentation
    int aChillTime = 1000;                                          // :7532   int
    if (theIsIceTrap) aChillTime = 2000;                            // :7533-7536  (false here -> 1000)
    mChilledCounter = std::max(aChillTime, mChilledCounter);        // :7537   NOT an assignment: a max, i.e. it EXTENDS, never shortens
    UpdateAnimSpeed();                                              // :7539
}
```

* **Member written: `mChilledCounter` (`int32_t`, `src/Lawn/Zombie.h:115`) ← `std::max(1000, mChilledCounter)`.**
* It does **not** reset any other counter: `mIceTrapCounter` (`src/Lawn/Zombie.h:117`) and `mButteredCounter`
  (`:116`) are untouched by `ApplyChill`; the only writers of `mChilledCounter = 0` are `RemoveColdEffects`
  (`src/Lawn/Zombie.cpp:8673-8685`, called by fireball jalapeno-style paths and `:9779`/`:9821`/… — never by a pea)
  and the per-tick decrement `if (mChilledCounter > 0) { mChilledCounter--; if (mChilledCounter == 0)
  UpdateAnimSpeed(); }` (`src/Lawn/Zombie.cpp:4619-4626`, inside `Zombie::Update`).
* `CanBeChilled()` (`src/Lawn/Zombie.cpp:8035-8059`) blocks: `ZOMBIE_ZAMBONI`, bobsled-team-with-sled,
  `IsDeadOrDying()`, phases `DIGGER_TUNNELING / DIGGER_RISING / DIGGER_TUNNELING_PAUSE_WITHOUT_AXE /
  DIGGER_RISE_WITHOUT_AXE / RISING_FROM_GRAVE / DANCER_RISING`, `mMindControlled`, and the boss except in its three
  idle/spit head phases. **Level-7 normal/conehead/headover-armor zombies are all chillable.**
* Observable effect: `IsMovingAtChilledSpeed()` returns `mChilledCounter > 0`
  (`src/Lawn/Zombie.cpp:6590-6593`, plus dancer/bobsled group propagation at `:6595-6623`), and walking speed is
  scaled **at movement time, not by writing `mVelX`**: `aSpeed *= CHILLED_SPEED_FACTOR;` with
  `constexpr const float CHILLED_SPEED_FACTOR = 0.4f;` (`src/Lawn/Zombie.cpp:70`, used at `:4119`, `:4136`, `:4204`,
  `:3937`). Animation rate uses a different factor: `theAnimRate * 0.5f` (`src/Lawn/Zombie.cpp:6639`).
* Non-chill side effects of the hit, zombie state: `mJustGotShotCounter = 25` (`src/Lawn/Zombie.cpp:7828`, flash
  counter), shield `mShieldJustGotShotCounter = 25` / `mShieldRecoilCounter = 12` (`:7607`, `:7613`),
  helm/shield/body health decrements and `DropHelm`/`DropShield`, and on body death
  `mBodyHealth = 0; PlayDeathAnim(...); DropLoot();` (`src/Lawn/Zombie.cpp:7965-7970`).
* The projectile **always dies on impact** (`src/Lawn/Projectile.cpp:937`), for both types, hit or ground-impact.

---

## 4. The zombie-side "can this pea hit me" test

Entry point (called with the pea's `mDamageRangeFlags`):
`Zombie::EffectedByDamage(unsigned int theDamageRangeFlags)` — `src/Lawn/Zombie.cpp:8084-8177`, reproduced in full
with the level-7 outcome for `theDamageRangeFlags == 1u`:

```
bool Zombie::EffectedByDamage(unsigned int theDamageRangeFlags)
{
    if (!TestBit(flags, DAMAGES_DYING /*5*/) && IsDeadOrDying())          return false;   // :8086-8089  1 has no bit5 -> DYING ZOMBIES CANNOT BE HIT
    if (TestBit(flags, DAMAGES_ONLY_MINDCONTROLLED /*7*/)) { if (!mMindControlled) return false; }
    else if (mMindControlled)                                             return false;   // :8091-8101
    if (type == ZOMBIE_BUNGEE && mZombiePhase != PHASE_BUNGEE_AT_BOTTOM
                                 && mZombiePhase != PHASE_BUNGEE_GRABBING) return false;   // :8103-8106
    if (mZombieHeight == HEIGHT_GETTING_BUNGEE_DROPPED)                   return false;   // :8108-8111  (airborne airdrop)
    if (type == ZOMBIE_BOSS) { …phase tests… }                                            // :8113-8131
    if (type != ZOMBIE_BOBSLED && GetBobsledPosition() > 0)                return false;   // :8133-8136  (only the sled lead is hittable)
    if (mZombiePhase in { POLEVAULTER_IN_VAULT, IMP_GETTING_THROWN, DIGGER_RISING,
        DIGGER_TUNNELING_PAUSE_WITHOUT_AXE, DIGGER_RISE_WITHOUT_AXE, DOLPHIN_INTO_POOL,
        DOLPHIN_IN_JUMP, SNORKEL_INTO_POOL, BALLOON_POPPING, RISING_FROM_GRAVE,
        BOBSLED_CRASHING, DANCER_RISING })
        return TestBit(flags, DAMAGES_OFF_GROUND /*4*/);                                  // :8138-8152  == false for flags 1
    if (type != ZOMBIE_BOBSLED && GetZombieRect().mX > WIDE_BOARD_WIDTH)  return false;   // :8154-8157  OFF-BOARD (x > 800) CANNOT BE HIT
    bool submerged = (type == ZOMBIE_SNORKEL && mInPool && !mIsEating);                    // :8159
    if (TestBit(flags, DAMAGES_SUBMERGED /*2*/) && submerged)                             return true;   // :8160-8163  no
    bool underground = (mZombiePhase == PHASE_DIGGER_TUNNELING);                           // :8165
    if (TestBit(flags, DAMAGES_UNDERGROUND /*6*/) && underground)                         return true;   // :8166-8169  no
    if (TestBit(flags, DAMAGES_FLYING /*1*/) && IsFlying())                               return true;   // :8171-8174  no
    return TestBit(flags, DAMAGES_GROUND /*0*/) && !IsFlying() && !submerged && !underground;            // :8176  -> true
}
```

Supporting predicates, verbatim:

```
bool Zombie::IsDeadOrDying()                    // src/Lawn/Zombie.cpp:9643-9650
{ return mDead || mZombiePhase == PHASE_ZOMBIE_DYING || mZombiePhase == PHASE_ZOMBIE_BURNED
                   || mZombiePhase == PHASE_ZOMBIE_MOWERED; }

bool Zombie::IsFlying()                         // src/Lawn/Zombie.cpp:9601-9604
{ return mZombiePhase == PHASE_BALLOON_FLYING || mZombiePhase == PHASE_BALLOON_POPPING; }

bool Zombie::IsOnBoard()                        // src/Lawn/Zombie.cpp:4284-4293
{ if (mFromWave == ZOMBIE_WAVE_CUTSCENE || mFromWave == ZOMBIE_WAVE_UI) return false;
  PVZP_ASSERT(mBoard); return true; }
```

`mZombieRect`, the base of the hit rectangle (`src/Lawn/Zombie.cpp:8301-8319`):

```
Rect Zombie::GetZombieRect()
{
    Rect aZombieRect = mZombieRect;                                  // :8303   default Rect(36, 0, 42, 115)  (:201)
    if (IsWalkingBackwards()) aZombieRect.mX = mWidth - aZombieRect.mX - aZombieRect.mWidth;   // :8304-8307
    ZombieDrawPosition aDrawPos; GetDrawPos(aDrawPos);                // :8309-8310
    aZombieRect.Offset(mX, mY + aDrawPos.mBodyY);                     // :8311   int32 += int32  (Offset: Rect.h:103-107)
    if (aDrawPos.mClipHeight > CLIP_HEIGHT_LIMIT)                     // :8312   CLIP_HEIGHT_LIMIT = -100.0f (:71)
    { aZombieRect.mHeight -= aDrawPos.mClipHeight; aZombieRect.mHeight = std::max(aZombieRect.mHeight, 0); }  // :8313-8316
    return aZombieRect;                                              // :8318
}
```

Answers to the specific questions:

* **Which rectangles**: `GetProjectileRect()` (`[mX-15, mX+40] × [mY, mY+40]`, `Projectile.cpp:1173`) against
  `GetZombieRect()` (`mZombieRect` offset by `(mX, mY + mBodyY)`), with the pea's `mWidth + 15` as the radius term.
  The projectile's `mPosX`/`mPosY` floats are **not** used directly — only their integer truncations `mX`/`mY`
  (`Projectile.cpp:773-774` → `:1173`).
* **Row equality**: yes, `aZombie->mRow == mRow` (`Projectile.cpp:237`), and it is the **only** lane test.
* **y / range tolerance: there is none.** `GetRectOverlap` (`Board.cpp:8969-8973`) is x-only, and the straight-pea
  path contains no `mPosY` comparison against the zombie rect. (The only y comparison in the file is
  `mPosY > aZombieRect.mY && mPosY < aZombieRect.mY + aZombieRect.mHeight` at `Projectile.cpp:285`, inside the
  `MOTION_HOMING` branch — cattail, unreachable.) `aDrawPos.mBodyY = -mAltitude` for airborne zombies
  (`Zombie.cpp:6002`, `:6024`, `:6091`) therefore shifts the zombie rect vertically and **cannot affect a pea hit**.
* **Dying / off-board / airborne**: dying → no (bit `DAMAGES_DYING` absent, `Zombie.cpp:8086`); off-board x>800 →
  no (`:8154`); balloon flying → no (`:8176` needs `DAMAGES_FLYING`); vaulting/pogo-jump/dolphin/snorkel-entering/
  imp-thrown/digger-rising/grave-rising/bobsled-crashing/dancer-rising → no (`:8138-8152`); snorkel *walking* in the
  pool → additionally excluded by `mPosZ <= 45.0f` in `Projectile.cpp:239`. `IsOnBoard()` is **not** part of the hit
  test (NOT FOUND in `FindCollisionTarget`/`EffectedByDamage`); it is used by `Zombie::GetPosYBasedOnRow` (`:8004`)
  and `Zombie::UpdateAnimSpeed` (`:6645`). `mJustDied` **does not exist in this codebase** — NOT FOUND in
  `src/Lawn/Zombie.h` / `src/Lawn/Zombie.cpp` (0 hits tree-wide); the equivalent is `IsDeadOrDying()`.
  `TakeDamage` re-guards independently: `if (mZombiePhase == PHASE_JACK_IN_THE_BOX_POPPING || IsDeadOrDying()) return;`
  (`src/Lawn/Zombie.cpp:7975-7976`).
* **Empty lane**: nothing special — the scan just finds no candidate (§5).

---

## 5. Removal / `Board::UpdateProjectiles`

* **`Board::UpdateProjectiles` does not exist** — NOT FOUND in `src/Lawn/Board.h` / `src/Lawn/Board.cpp`
  (no declaration, no definition, no call site). The projectile driver is the third loop of
  `Board::UpdateGameObjects` (`src/Lawn/Board.cpp:4988-4993`), which is:

  ```
  for (Projectile* aProjectile : mProjectiles) { if (aProjectile->mDead) continue; aProjectile->Update(); }
  ```

  Nothing else in that loop touches board state: no counters, no spawns, no sound.
* **Projectiles are NOT removed when their row's zombie list is empty.** No such condition exists — NOT FOUND in
  `src/Lawn/Projectile.cpp` / `src/Lawn/Board.cpp`. An unopposed pea keeps advancing `+3.33f` per tick until
  `mPosX > 800.0f`, then `Die()` (`Projectile.cpp:272-276`).
* Actual de-allocation: `Board::ProcessDeleteQueue` (`src/Lawn/Board.cpp:8519-8547+`), third block
  (`:8539-8547`) `if (aProjectile->mDead) mProjectiles.DataArrayFree(aProjectile);`, invoked from
  `LawnApp::AdvanceLogicTick` before the tick's `UpdateFrames` (`src/LawnApp.cpp:1581-1589`);
  `DataArrayFree` pushes the slot on the free list and decrements `mSize`
  (`src/PvzpLib/DataArray.h:91-99`). A pea killed on tick T is still in the array (with `mDead == true`) for the
  remainder of T and is reclaimed at the start of T+1.
* Board state mutated anywhere along the straight-pea path:
  * `Projectile::Die` — sets `mDead`, detaches the snow-pea particle
    (`src/Lawn/Projectile.cpp:1152-1165`); no board counter touched.
  * `DoImpact` → `Zombie::TakeDamage` — mutates the **zombie** (health/shield/helm/`mChilledCounter`/
    `mJustGotShotCounter`), possibly `PlayDeathAnim` + `DropLoot` (`src/Lawn/Zombie.cpp:7965-7970`) which can create
    coins; that is the only board-object mutation reachable from a pea hit.
  * Presentation-only calls, ignored by the rollout: `mApp->PlayFoley` / `PlaySample`
    (`Projectile.cpp:318`, `:604`, `:611`, `:619`, `:783-800`, `:789`, `:1204`, `:1229`),
    `mApp->AddPvzpParticle` (`:118`, `:319`, `:850`, `:853`, `:858-859`, `:933`, `Plant.cpp:4680`),
    `AddReanimation` (`:876`), `Board::ShakeBoard` (`:861`, cob only), `AttachmentUpdateAndMove` /
    `AttachmentCrossFade` / `AttachmentDetach` / `AttachmentDie` (`:972`, `:1158-1163`, `:60`),
    `Board::MakeRenderOrder` writes to `mRenderOrder` (`:962`), and `ReportAchievement::GiveAchievement` (`:630`,
    cob only). `Board::mGargantuarsKillsByCornCob` (`:628`) is the only `Board` **counter** a projectile increments,
    and it is unreachable for this deck.
  * `Challenge::Update`-style portal teleport writes `mPosX/mPosY/mRow/mRenderOrder/mLastPortalX`
    (`src/Lawn/Challenge.cpp:3198-3215`) — `GAMEMODE_CHALLENGE_PORTAL_COMBAT` only, unreachable.

---

## 6. Per-tick live members (SoA completeness checklist)

`Projectile`'s full member list is `src/Lawn/Projectile.h:49-75` plus `GameObject`'s `src/Lawn/GameObject.h:36-45`
(`mApp, mBoard, mX, mY, mWidth, mHeight, mVisible, mRow, mRenderOrder`). For **PROJECTILE_PEA /
PROJECTILE_SNOWPEA on `MOTION_STRAIGHT`, level 7**, per tick:

| member | type | tick activity | cite |
|---|---|---|---|
| `mProjectileAge` | `int32_t` | **W** `++` ; R `> aTime` (gate, presentation), `>= 75`/`< 25`/`> 20`/`< 60` branches all false | :942, :960, :266, :244, :536, :684 |
| `mPosX` | `float` | **W** `+= 3.33f` ; R bounds, `PixelToGridXKeepOnBoard(mPosX+30, mPosY)`, `GetPosYBasedOnRow`, DoImpact splat | :718, :272, :382, :748/:758, :842 |
| `mX` | `int32_t` | **W** `= (int)mPosX` ; R `GetProjectileRect`, `mShadowY`/`mY` paths | :773, :1173 |
| `mY` | `int32_t` | **W** `= (int)(mPosY + mPosZ)` ; R `GetProjectileRect` | :774, :1173 |
| `mPosY` | `float` | R only (`mShadowY - mPosY` twice, `PixelToGridXKeepOnBoard` arg, `GetProjectileRect` uses `mY` not `mPosY`) | :299, :353, :382 |
| `mShadowY` | `float` | R `mShadowY - mPosY`; **W** `+= aSlopeHeightChange` which is `0.0f` exactly | :299, :353, :772 |
| `mPosZ` | `float` | R in `(int)(mPosY+mPosZ)`, `mPosZ <= 45.0f` snorkel test, splat y | :774, :239, :843 |
| `mVelX` | `float` | R only, and only in never-taken branches (`STAR && mVelX >= 0.0f`, `MOTION_STAR && mVelX < 0.0f`) + `aLastPosX` in DoImpact (presentation) | :244, :413, :842 |
| `mVelY` | `float` | R only in `aLastPosY` (presentation) / never-taken motion branches | :843, :666/:673/:712 |
| `mVelZ` | `float` | R only in `(int)(mPosY+mPosZ)`, `aLastPosY`, HIGH_GRAVITY (unreachable), FLOAT_OVER/LOBED (unreachable) | :774, :843, :692-696, :522-529 |
| `mAccZ` | `float` | **not touched on the straight path** (only `UpdateLobMotion`) | :522 |
| `mRotation` | `float` | **W** `+= mRotationSpeed` (both 0.0f → stays 0.0f) ; R in `Draw` only | :969, :1050/:1060 |
| `mRotationSpeed` | `float` | R (`:969`); never written for PEA/SNOWPEA | :93, :969 |
| `mMotionType` | `enum int32` | R at every branch of `UpdateNormalMotion`/`CheckForCollision`/`CantHitHighGround`/`GetDamageFlags`; never W after init | :76, :266/:278/:304/:339/:409, :642-716 |
| `mProjectileType` | `enum int32` | R everywhere (`:299`, `:343-347`, `:355-359`, `:396-398`, `:418`, `:947-956`, `:1169`); **W only** by `ConvertToFireball`/`ConvertToPea` (torchwood, unreachable) | :1202, :1227 |
| `mRow` | `int32_t` | R: row equality in the scan, render order, `PixelToGridXKeepOnBoard(..., mRow)`, `GetPosYBasedOnRow`; **W only** for HOMING/STAR/COBBIG | :237, :79, :962, :383, :669/:679, :515 |
| `mDamageRangeFlags` | `int32_t` | R `EffectedByDamage(static_cast<unsigned int>(mDamageRangeFlags))`; W once by `Plant::Fire` (value `1`) | :237, :314(no), :457/:496 (splash, dead), `Plant.cpp:4694` |
| `mWidth` / `mHeight` | `int32_t` | R: `mPosX + mWidth < 0.0f`, rect size (both `40`) | :272, :1173, :94-95 |
| `mDead` | `bool` | R by every owning loop; **W** `= true` in `Die()` | :235-236 (zombie side), `Board.cpp:4990`, `:1154` |
| `mRenderOrder` | `int32_t` | **W** at the age gate; R as `mRenderOrder + 1` in splat particles (presentation) | :962, :319/:850/:933 |
| `mHitTorchwoodGridX` | `int32_t` | R inside the torchwood scan only (`!= aPlant->mPlantCol`); W `-1` at init / by the two `ConvertTo*` | :208, :75, :1203/:1228 |
| `mOnHighGround` | `bool` | R in `CantHitHighGround()` (`!mOnHighGround`) and `DrawShadow`; W only at init (`false` here) | :348, :1087, :86 |
| `mFrame` / `mNumFrames` / `mAnimCounter` / `mAnimTicksPerFrame` | `int32_t` | **not touched** — the guard `if (mAnimTicksPerFrame > 0)` is never true for projectiles (`mAnimTicksPerFrame = 0` at `:98`, no writer anywhere: NOT FOUND in `src/`). Read only in `Draw` | :741-745, :1045/:1049 |
| `mClickBackoffCounter` | `int32_t` | R `> 0`, W `--` — but it is never set to anything but `0` (`:97`; no other writer: NOT FOUND in `src/`) ⇒ **dead field, safe to omit** | :965-968 |
| `mAttachmentID` | `AttachmentID` | R/W by particle attachment calls (`AttachmentUpdateAndMove`, `AttachmentDetach/CrossFade/Die`) — **presentation only** | :972, :1158-1163, :60 |
| `mCobTargetX` / `mCobTargetRow` | `float` / `int32_t` | untouched after init (LOBED cob path only) | :80, :84, :511-519 |
| `mTargetZombieID` | `ZombieID` | untouched (HOMING only) | :85, :280 |
| `mLastPortalX` | `int32_t` | untouched (Portal Combat only) | :75, `Challenge.cpp:3214` |
| `mApp`, `mBoard` | pointers | R for `mGameScene`, `mPlants`, `mZombies`, `mGridSquareType`, `GetProjectileDef` etc. | :943, :204, :233, :383, `Plant.cpp:3675` |
| `mVisible` | `bool` | not read on the update path (draw only) | — |

**Minimum SoA field set for a straight pea**: `type, motion, row, damageRangeFlags, age, posX, posY, posZ, shadowY,
mX, mY, width, height, dead`. Everything else in the table is either constant after spawn (`velX/velY/velZ/accZ,
rotation, rotationSpeed, onHighGround, hitTorchwoodGridX, cob*, targetZombieID, lastPortalX, anim*`) or
presentation (`renderOrder, attachmentID, clickBackoffCounter, frame/numFrames/animCounter/animTicksPerFrame`).
Damage must be looked up by `type` (`gProjectileDefinition[type].mDamage`), never stored per projectile (§2.3);
`src/CudaExport.cpp:255-256` already does exactly that and comments the reason.
