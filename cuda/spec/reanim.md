# cuda/spec/reanim.md — animation-derived timing numbers, measured from the shipped assets

Scope: **adventure level 7** (`BACKGROUND_1_DAY`, grass, 5 rows), zombie types
`ZOMBIE_NORMAL`/`ZOMBIE_TRAFFIC_CONE`/`ZOMBIE_FLAG`/`ZOMBIE_POLEVAULTER`, plant types
`PEASHOOTER`/`SUNFLOWER`/`CHERRYBOMB`/`WALLNUT`/`POTATOMINE`/`SNOWPEA`/`REPEATER`.
Everything here is either (M) **measured** out of `main.pak` or (S) **read** from `src/`.
Nothing in this file is guessed; §8 lists what could not be determined.

Sources of code: `src/PvzpLib/Reanimator.cpp` / `.h`, `src/PvzpLib/Definition.cpp`,
`src/Lawn/Zombie.cpp`, `src/Lawn/Plant.cpp`, `src/Lawn/Board.cpp`, `src/ConstEnums.h`,
`src/PvzpLib/EffectSystem.cpp`, `src/PvzpLib/Attachment.cpp`, `src/SexyAppFramework/misc/MTRand.cpp`.

### 0.1 Where the reanim data lives (M)

There is **no `data/` or `assets/` directory in this repo**. The `.reanim` definitions are inside
`main.pak` (`src/SexyAppFramework/SexyAppBase.cpp:3381` loads it from the resource dir; on this
machine the file is `~/main.pak`, 44 925 496 bytes). Two copies of every definition ship:

| form | pak key | example size | what it is |
|---|---|---|---|
| XML | `REANIM/ZOMBIE.REANIM` | 705 114 B | `<fps>` + `<track><name>…</name><t>…</t>` list |
| compiled | `COMPILED/REANIM/ZOMBIE.REANIM.COMPILED` | 92 615 B | `{u32 cookie 0xDEADFED4, u32 raw}` + zlib of the struct image |

Pak format (M, decoded): whole file XOR `0xF7` (`PakInterface.cpp:112-113`), magic `0xBAC04AC0`,
version 0, then index records `{u8 flags, u8 namelen, name, i32 size, i64 mtime}` with bit `0x80`
of `flags` ending the index; data blob starts right after the index (`PakInterface.cpp:158-196`).

XML element → field mapping is `gReanimatorTransformDefFields`
(`Definition.cpp:142-155`): `x`→`mTransX`, `y`→`mTransY`, `kx`→`mSkewX`, `ky`→`mSkewY`,
`sx`→`mScaleX`, `sy`→`mScaleY`, `f`→`mFrame`, `a`→`mAlpha`, `i`/`font`/`text`. Track container
fields `name`/`t` at `Definition.cpp:158-161`, definition fields `track`/`fps` at
`Definition.cpp:164-167`. An absent element stays at `DEFAULT_FIELD_PLACEHOLDER = -10000.0f`
(`Reanimator.h:51`, constructor `Reanimator.cpp:193-204`) and is then filled by the
inherit-from-previous pass `ReanimationFillInMissingData` (`Reanimator.cpp:206-213`, applied per
track at `Reanimator.cpp:221-261`). **The fill-in pass is what makes the `_ground` x-column
monotone: blank `<t></t>` frames repeat the previous x.**

Load order (S): `DefinitionLoadXML` → `DefinitionCompileAndLoad` (`Definition.cpp:1291-1319`)
tries the `.compiled` blob first and **falls back to the XML** if the cached struct sizes do not
match the running build. Cross-check (M): the shipped `.compiled` blob stores **32-bit** struct
images (`defSize` 12 for `ReanimatorTrack`, 44 for `ReanimatorTransform`); a 64-bit host build
wants 32/56, so on this host the blob is rejected at `Definition.cpp:383-387` and the XML is used.
Both were decoded and agree on every number below (`_ground` x[44] = −9.8, x[90] = 40.0 in both).
`cuda/ReanimDump.cpp` (untracked) is a C++ equivalent of the same dump.

---

## 1. (a) Frame-rate convention

| item | value / rule | cite | status |
|---|---|---|---|
| `SECONDS_PER_UPDATE` | `constexpr const double … = 0.01f` → the double `0.009999999776482582` (a float32 0.01 widened, **not** 0.01) | `Reanimator.h:52` | S |
| updates per second | 100 (1 update = 10 ms) | derived | S |
| `mAnimRate` units | **frames per second**, float. Initialised to the definition's `mFPS` (12.0 for every asset in scope; 30.0 for `CherryBomb.reanim`) | `Reanimator.cpp:405`, `Reanimator.h:84-89` | S/M |
| loop position | `mAnimTime` (float) ∈ [0,1) — **this fork has no `mFramePos`**; frame indices are derived | `Reanimator.h:192-198` | S |
| frame window | `mFrameStart`, `mFrameCount` (int32) = span of the played `anim_*` track | `Reanimator.h:197-198`, `Reanimator.cpp:1129-1136` | S |

Literal advance (`Reanimator.cpp:424-431`):

```cpp
void Reanimation::Update()
{
	if (mFrameCount == 0 || mDead) return;
	PVZP_ASSERT(std::isfinite(mAnimRate));
	mLastFrameTime = mAnimTime;                                  // previous loop position
	mAnimTime += SECONDS_PER_UPDATE * mAnimRate / mFrameCount;   // advance the loop position
```

so per tick `ΔmAnimTime = 0.01 · mAnimRate / mFrameCount`, i.e. **`mAnimRate/100` frames per tick**,
and a full loop costs `mFrameCount / (0.01 · mAnimRate)` ticks (`Zombie.cpp:1727` uses this
inverted form: `aAnimDuration = mFrameCount / mAnimRate * 100.0f`).

`mLoopCount` (S, `Reanimator.cpp:434-461`):

```cpp
	if (mAnimRate > 0) {
		switch (mLoopType) {
		case REANIM_LOOP: case REANIM_LOOP_FULL_LAST_FRAME:
			while (mAnimTime >= 1.0f) { mLoopCount++; mAnimTime -= 1.0f; }      // :437-443
			break;
		case REANIM_PLAY_ONCE: case REANIM_PLAY_ONCE_FULL_LAST_FRAME:
			if (mAnimTime >= 1.0f) { mLoopCount = 1; mAnimTime = 1.0f; mDead = true; }  // :447-452
			break;
		case REANIM_PLAY_ONCE_AND_HOLD: case REANIM_PLAY_ONCE_FULL_LAST_FRAME_AND_HOLD:
			if (mAnimTime >= 1.0f) { mLoopCount = 1; mAnimTime = 1.0f; }        // :456-459
			break;
```

Reverse playback mirrors it (`:470-495`). `PlayReanim` resets `mLoopCount = 0` and, for
`mAnimRate >= 0`, `mAnimTime = 0` (`Reanimator.cpp:1385-1395`, `1129-1136`).
Frame position for sampling (`Reanimator.cpp:906-927`):

```cpp
	int aFrameCount = (mLoopType == REANIM_PLAY_ONCE_FULL_LAST_FRAME || mLoopType == REANIM_LOOP_FULL_LAST_FRAME ||
			   mLoopType == REANIM_PLAY_ONCE_FULL_LAST_FRAME_AND_HOLD) ? mFrameCount : mFrameCount - 1;
	float aAnimPosition = mFrameStart + mAnimTime * aFrameCount;
	float aAnimFrameBefore = floor(aAnimPosition);
	theFrameTime->mAnimFrameBeforeInt = FloatRoundToInt(aAnimFrameBefore);   // == floor, PvzpCommon.h:203
	if (theFrameTime->mAnimFrameBeforeInt >= mFrameStart + mFrameCount - 1) { … before == after }
	else theFrameTime->mAnimFrameAfterInt = theFrameTime->mAnimFrameBeforeInt + 1;
```

**The `frameCount` vs `frameCount − 1` split between the advance and the sampler is the whole
origin of the K factor in §2.**

Tick ordering correction (2026-10-18 round 21, traced against real root tick 2400):
the original paragraph below missed `mIsAttachment=true` on zombie bodies (`Zombie.cpp:1041`).
`EffectSystem` skips them. `Zombie::UpdateReanim:5385` advances the body at the entity tail,
after movement and prey checks; `Plant::UpdateReanim:2791` similarly advances plant bodies.
The old claim that the zombie sees an already advanced body frame is false. Imported animation
times agree but the wrong ordering changes displacement at every frame boundary.

`Board::Update` (`:5722`) runs
`mApp->mEffectSystem->Update()` at `:5762` (which advances every non-attachment reanim,
`EffectSystem.cpp:128-130`) **before** `UpdateGame()` at `:5815` →
`UpdateGameObjects()` at `:5662`/`:4972`, where `Zombie::Update()` (`:4985`, body `:4304`) reads
the animation. This global phase advances only non-attachments, not the body animation.

---

## 2. (b) `anim_walk`, the `_ground` track, and K

### 2.1 Which anim and which file (M/S)

Body reanim comes from `gZombieDefs[].mReanimationType` (`Zombie.cpp:81-84`; NORMAL/FLAG/CONE →
`REANIM_ZOMBIE` → `reanim/Zombie.reanim` at `Reanimator.cpp:67`; POLEVAULTER →
`REANIM_POLEVAULTER` → `reanim/Zombie_polevaulter.reanim` at `Reanimator.cpp:100`).
`StartWalkAnim` (`Zombie.cpp:6748-6791`) picks the clip: `aWalkAnimVariant = Rand(2)` (`:6773`),
forced to 0 for `PEA_HEAD` (`:6774-6777`) and for `FLAG` (`:6778-6781`); variant 0 and the track
exists → `anim_walk2` (`:6783-6785`), otherwise `anim_walk` (`:6787-6789`). Play goes through
`PlayZombieReanim(name, REANIM_LOOP, blend, 0.0f)` (`:6785`/`:6789`) → `Reanimation::PlayReanim`
(`Reanimator.cpp:1385-1395`); rate 0 means "keep the current rate" (`:1389-1390`), `mLoopCount` is
reset to 0 (`:1393`), and `PlayZombieReanim` ends with `UpdateAnimSpeed()` (`Zombie.cpp:2904`).
So NORMAL / TRAFFIC_CONE get `anim_walk` or `anim_walk2` 50/50 per `StartWalkAnim` call, FLAG
always `anim_walk2`, POLEVAULTER always `anim_walk` (no `anim_walk2` in its file, §2.1).

Measured windows, `GetFramesForLayer` semantics (`Reanimator.cpp:1091-1127`, first/last frame with
`mFrame >= 0`), `Zombie.reanim` (M) — 42 tracks, 504 frames each, `fps = 12`:

| track | `frameStart` | `frameCount` |
|---|---|---|
| `anim_walk` | 44 | **47** |
| `anim_walk2` | 91 | **47** |
| `anim_eat` | 138 | 40 |
| `anim_death` | 178 | 39 |
| `anim_death2` | 217 | 33 |
| `anim_superlongdeath` | 292 | 137 |
| `anim_swim` | 250 | 42 |
| `anim_waterdeath` | 429 | 25 |
| `anim_idle` / `anim_idle2` | 0 / 29 | 29 / 15 |
| `anim_dance` | 454 | 50 |
| `_ground` (own span) | 44 | 460 |

`Zombie_polevaulter.reanim` (M) — 28 tracks, 193 frames each, `fps = 12`:

| track | `frameStart` | `frameCount` |
|---|---|---|
| `anim_idle` | 0 | 13 |
| `anim_run` | 13 | 37 |
| `anim_jump` | 50 | 43 |
| `anim_walk` | 93 | 45 |
| `anim_death` | 138 | 27 |
| `anim_eat` | 165 | 28 |
| `_ground` | 13 | 125 |

**No `anim_walk2`, no `anim_death2`, no `anim_superlongdeath` in the polevauler file** →
`TrackExists` fails and that zombie always walks `anim_walk` and always dies with `anim_death`.

### 2.2 The two formulas in the engine (S)

`Zombie::UpdateAnimSpeed`, `Zombie.cpp:6684-6693`:

```cpp
		else if (aBodyReanim->TrackExists("_ground"))
		{
			ReanimatorTrack* aTrack = &aBodyReanim->mDefinition->mTracks.tracks[aBodyReanim->FindTrackIndex("_ground")];
			float aDistance = aTrack->mTransforms.mTransforms[aBodyReanim->mFrameStart + aBodyReanim->mFrameCount - 1].mTransX
				            - aTrack->mTransforms.mTransforms[aBodyReanim->mFrameStart].mTransX;
			if (aDistance >= 1e-6f)
			{
				float aOneOverSpeed = aBodyReanim->mFrameCount / aDistance;
				float aAnimRate = mVelX * aOneOverSpeed * 47.0f / mScaleZombie;
				ApplyAnimRate(aAnimRate);
			}
		}
```

`ApplyAnimRate` halves it while chilled (`:6634-6641`, `IsMovingAtChilledSpeed` `:6590-6627`).

`Zombie::UpdateZombieWalking`, `Zombie.cpp:4127-4129` + `:4146`:

```cpp
		else if (aBodyReanim->TrackExists("_ground"))
		{
			aSpeed = aBodyReanim->GetTrackVelocity("_ground") * mScaleZombie;
		}
		…
		mPosX -= aSpeed;                       // walking left
```

`Reanimation::GetTrackVelocity`, `Reanimator.cpp:1315-1325`:

```cpp
	float aDis = aTrack->mTransforms.mTransforms[aFrameTime.mAnimFrameAfterInt].mTransX
	           - aTrack->mTransforms.mTransforms[aFrameTime.mAnimFrameBeforeInt].mTransX;
	return aDis * SECONDS_PER_UPDATE * mAnimRate;
```

### 2.3 K, closed form

Per tick the advance is `aDis_i · 0.01 · animRate · mScaleZombie`, where `aDis_i` is the `_ground`
x-step of the interval the loop position currently sits in. Over one loop the `frameCount − 1`
intervals each receive the same number of ticks (`mFrameCount/(0.01·animRate)` ticks spread over
`frameCount − 1` intervals), and `Σ aDis_i = aDistance`, so

```
mean advance per tick = 0.01 · animRate · aDistance / (frameCount − 1)
                      = 0.01 · [mVelX · frameCount / aDistance · 47 / mScaleZombie] · aDistance / (frameCount − 1)
                      = 0.47 · mVelX · frameCount/(frameCount − 1) · mScaleZombie / mScaleZombie
```

**K = 0.47 · frameCount/(frameCount − 1).** For level 7 `mScaleZombie = 1.0`
(`Zombie.cpp:206`) and `frameCount = 47` (walk/walk2) ⇒ **K = 0.47 · 47/46 = 0.480217**.
The bare `0.47` in `cuda/spec/zombies.md:246-251` is **2.17 % low**.

Measured by replaying the recurrence exactly as the engine computes it (float32 `mAnimTime`/
`mAnimRate`, `SECONDS_PER_UPDATE` as the double `0.010000000298023224`, `GetFrameTime`'s
`frameCount − 1` sampler, `PvzpCommon.h:203` rounding), one full loop from `mAnimTime = 0` (M):

| clip | `mVelX` | `aDistance` | `animRate` | ticks in loop | mean advance/tick | / `mVelX` | K closed form |
|---|---|---|---|---|---|---|---|
| `anim_walk` (Zombie) | 0.23 | 49.8 | 10.202210 | 461 | 0.110450 | **0.480217** | 0.480217 |
| `anim_walk` (Zombie) | 0.30 | 49.8 | 13.307230 | 354 | 0.143816 | **0.479387** | 0.480217 |
| `anim_walk` (Zombie) | 0.37 | 49.8 | 16.412249 | 287 | 0.177723 | **0.480332** | 0.480217 |
| `anim_walk` (Zombie) | 1.00 | 49.8 | 44.357430 | 106 | 0.479060 | **0.479060** | 0.480217 |
| `anim_walk2` (Zombie) | 0.30 | 49.8 | 13.307230 | 354 | 0.143929 | **0.479764** | 0.480217 |
| `anim_walk2` (Zombie) | 0.37 | 49.8 | 16.412249 | 287 | 0.176919 | **0.478161** | 0.480217 |
| `anim_walk2` (Zombie, FLAG) | 0.45 | 49.8 | 19.960844 | 236 | 0.215832 | **0.479627** | 0.480217 |
| `anim_walk2` (Zombie) | 1.00 | 49.8 | 44.357430 | 106 | 0.482440 | **0.482440** | 0.480217 |
| `anim_walk` (polevault) | 0.23 | 67.8 | 7.174779 | 628 | 0.110563 | **0.480707** | 0.480682 |
| `anim_walk` (polevault) | 0.30 | 67.8 | 9.358407 | 481 | 0.144451 | **0.481503** | 0.480682 |
| `anim_walk` (polevault) | 0.37 | 67.8 | 11.542035 | 390 | 0.177967 | **0.480992** | 0.480682 |
| `anim_run` (polevault) | 0.30 | 30.0 | 17.389999 | 213 | 0.144944 | **0.483147** | 0.483056 |
| `anim_run` (polevault, pre-vault) | 0.66–0.68 | 30.0 | 38.258003 | 97 | 0.320012 | **0.484867** | 0.483056 |
| any of the above, chilled | — | — | ×0.5 | ×2 | ×0.5 | **≈ K/2 = 0.240** | — |

"ticks in loop" = the update index on which `mAnimTime` first reaches `1.0f` (= the theoretical
`mFrameCount·100/animRate` rounded **up**, plus 1 when that value is an exact integer and the
float accumulation lands just under 1.0 — the mowed case in §4). Cycle net advance:
50.72–50.83 px (Zombie walk, `aDistance·47/46` = 50.88), 69.23–69.34 px (polevault walk,
`67.8·45/44` = 69.34), 30.72–30.80 px (polevault run, `30·37/36` = 30.83).

The measured ratio sits within ±0.9 % of the closed form and drifts with `mVelX` — the sampler
quantises `mFrameCount − 1` intervals onto `mFrameCount/(0.01·animRate)` ticks, so the interval
that gets the extra tick changes with the rate. **Use `K = 0.47·frameCount/(frameCount−1)`;
`0.47` alone is 2.17 % low, and no single constant reproduces per-tick motion (§2.4).**

### 2.4 Non-uniformity of `_ground` (M)

`anim_walk` window (frames 44…90), `_ground` `mTransX` after the fill-in pass, and the 46
per-interval steps (px):

```
x:  -9.8 -8.4 -7.0 -5.6 -4.1 -2.7 -1.3  0.0  1.4  2.8  4.2  5.7  7.1  7.9  8.8  9.7 10.5
   10.6 10.8 10.9 11.0 11.0 11.0 11.0 11.0 13.4 15.8 18.1 20.5 22.8 25.2 27.6 29.9 31.1
   32.3 33.5 34.6 35.9 37.0 38.2 39.4 39.5 39.6 39.7 39.8 39.9 40.0
dx:  1.4  1.4  1.4  1.5  1.4  1.4  1.3  1.4  1.4  1.4  1.5  1.4  0.8  0.9  0.9  0.8  0.1
      0.2  0.1  0.1  0.0  0.0  0.0  0.0  2.4  2.4  2.3  2.4  2.3  2.4  2.4  2.3  1.2  1.2
      1.2  1.1  1.3  1.1  1.2  1.2  0.1  0.1  0.1  0.1  0.1  0.1
```

`anim_walk2` window (91…137): same endpoints (−9.8 → 40.0, `aDistance` = 49.8) but a different
step profile (46 steps):

```
dx: 1.3 1.2 1.3 1.3 1.3 1.3 1.2 1.2 1.3 1.2 1.3 1.3 1.3 1.3 1.2 1.3 0.1 0.1 0.0 0.1 0.0 0.1
     0.1 1.8 1.7 1.8 1.8 1.8 1.7 1.8 1.8 1.8 1.8 1.7 1.8 1.7 1.9 1.7 1.8 0.1 0.0 0.1 0.2 0.1
     0.0 0.1
```

polevault `anim_walk` window (93…137), `_ground` −59.8 → 8.0, `aDistance` = 67.8 (44 steps):

```
dx: 0.7 0.8 0.7 0.7 1.9 2.0 1.9 1.9 2.0 2.1 2.0 2.0 2.1 2.0 2.0 2.1 2.0 2.0 1.9 2.0 2.0 2.0
     2.0 2.0 2.3 2.4 2.3 2.3 1.9 1.9 1.9 1.9 1.7 1.9 1.9 1.9 0.2 0.3 0.2 0.2 0.0 -0.1 -0.1
      0.0
```

| quantity | `anim_walk` | `anim_walk2` | polevault `anim_walk` |
|---|---|---|---|
| `Σ` of the 46 (resp. 44) steps | 49.8 = `aDistance` ✔ | 49.8 ✔ | 67.8 ✔ |
| steps == 0.0 | 4 of 46 | 4 of 46 | 2 of 44 (plus 2 negative: −0.1) |
| min / max step | 0.0 / 2.4 | 0.0 / 1.9 | −0.1 / 2.4 |
| max : min over **non-zero** steps | 24 : 1 (2.4 vs 0.1) | 19 : 1 | 12 : 1 |
| mean step (`aDistance/(frameCount−1)`) | 1.083 | 1.083 | 1.541 |
| instantaneous per-tick advance, `mVelX`=0.30 | 0 … 0.31937 (**0 → 1.065 × mVelX**) | 0 … 0.25284 (0 → 0.843) | −0.0094 … 0.22460 |
| fraction of ticks with **zero** advance | 4/46 ≈ 8.7 % | ≈ 8.7 % | 2/44 ≈ 4.5 % |

So: (1) the telescoping sum over the interval list *does* equal the track displacement, but
(2) the sum of **per-tick advances over one loop** is `aDistance · frameCount/(frameCount−1)`
(measured 50.72–50.83 px per 354-tick loop at `mVelX` = 0.30, vs `aDistance` = 49.8) — it does
**not** equal the displacement; and (3) the instantaneous speed is anything but constant: it
ranges from exactly 0 to `1.065 · mVelX` within one cycle. A CUDA core that advances `0.47·mVelX`
(or even `K·mVelX`) per tick will therefore not reproduce per-tick positions; it needs the
46-entry dx table above, or must accept sub-tick divergence in exchange for matching the cycle
mean.

---

## 3. (c) Death: which case, and how many ticks to `DieNoLoot`

`Zombie::PlayDeathAnim` (`Zombie.cpp:8926-9056`). `mVelX = 0.0f` at `:8988`,
`mZombiePhase = PHASE_ZOMBIE_DYING` at `:8989`. The `mInPool`/`aFallTime` switch inside
`UpdateDeath` only gates foley/daisies, never a tick count (S).

Rate selection (`:8998-9033`) — none of the level-7 types hit the fixed-rate cases
(FOOTBALL 24 `:9001`, GARGANTUAR/SNORKEL/YETI 14 `:9007/9012/9020`, DIGGER/BOSS 18):

```cpp
	else { aDeathAnimRate = RandRangeFloat(24.0f, 30.0f); }        // Zombie.cpp:9032
```

Track selection (`:9035-9051`):

```cpp
	const char* aDeathTrackName = "anim_death";
	int aDeathAnimHit = Rand(100);                                 // 0..99  (:9036)
	bool aCanDoSuperLongDeath = mApp->HasFinishedAdventure() || mBoard->mLevel > 5;   // :9037, level 7 => true
	if (mInPool && TrackExists("anim_waterdeath"))      aDeathTrackName = "anim_waterdeath";        // :9039-9042
	else if (aDeathAnimHit == 99 && TrackExists("anim_superlongdeath") && aCanDoSuperLongDeath
	         && mChilledCounter == 0 && mBoard->CountZombiesOnScreen() <= 5)
		{ aDeathAnimRate = 14.0f; aDeathTrackName = "anim_superlongdeath"; }                        // :9043-9047
	else if (aDeathAnimHit > 50 && TrackExists("anim_death2"))  aDeathTrackName = "anim_death2";    // :9048-9051
	PlayZombieReanim(aDeathTrackName, ReanimLoopType::REANIM_PLAY_ONCE_AND_HOLD, 20, aDeathAnimRate); // :9053
```

| `Rand(100)` | NORMAL / CONE / FLAG (has all 3 clips) | POLEVAULTER (only `anim_death`) |
|---|---|---|
| 0 … 50 (51 %) | `anim_death`, rate `RandRangeFloat(24,30)` | `anim_death`, same rate |
| 51 … 98 (48 %) | `anim_death2`, same rate | `anim_death` (`TrackExists` fails) |
| 99 (1 %) | `anim_superlongdeath`, rate **14.0** — needs `mChilledCounter==0` and ≤ 5 zombies on screen; else falls through to `anim_death2` | `anim_death` |
| `mInPool` | `anim_waterdeath` — **impossible on level 7 grass** | idem |

`RandRangeFloat(24,30)` is **continuous**, not integral: `RandRangeFloat` =
`Rand(theMax-theMin) + theMin` (`PvzpCommon.cpp:390-394`) and the `float` overload
(`Common.cpp:121-125`, `Common.h:149`) is `MTRand::Next(float)` =
`(float)((double)NextNoAssert() / 0x7FFFFFFF * range)` (`MTRand.cpp:189-198`), with the 32-bit
output masked to 31 bits at `MTRand.cpp:175`. So `aDeathAnimRate ∈ [24, 30]` (upper bound hit with
probability 2⁻³¹).

### 3.1 Tick accounting

`Zombie::UpdateDeath` (`Zombie.cpp:9095-9311`) never calls `DieNoLoot` itself for these types; the
only exit is the fade latch (`:9306-9309`):

```cpp
	else if (mZombieFade == -1 && aBodyReanim->mLoopCount > 0 && mZombieType != ZombieType::ZOMBIE_BOSS)
		mZombieFade = mInPool ? 10 : 100;
```

and `Zombie::Update` (`:4380-4386`, *after* the phase dispatch at `:4333-4336`) drains it:

```cpp
		if (mZombieFade > 0) { mZombieFade--; if (mZombieFade == 0) DieNoLoot(); }
```

`DieNoLoot` is `Zombie.cpp:7420`. With `k` = reanim updates until `mAnimTime ≥ 1.0`
(`REANIM_PLAY_ONCE_AND_HOLD` sets `mLoopCount = 1` and clamps `mAnimTime = 1.0f`,
`Reanimator.cpp:456-459`) and `T` = the tick on which `PlayDeathAnim` ran:
the latch fires on tick `T+k` (which also decrements 100 → 99), `DieNoLoot` on tick `T+k+99`
⇒ **total = k + 99 ticks** on grass. `k` was computed with float32 accumulation of the increment
(S formula, M values):

| case | clip / `frameCount` | `animRate` | k | ticks from `PHASE_ZOMBIE_DYING` to `DieNoLoot` |
|---|---|---|---|---|
| NORMAL/CONE/FLAG, `anim_death` @24 (slowest) | 39 | 24.0 | 163 | **262** |
| … @26 | 39 | 26.0 | 151 | 250 |
| … @28 | 39 | 28.0 | 140 | 239 |
| … @30 (fastest) | 39 | 30.0 | 130 | **229** |
| NORMAL/CONE/FLAG, `anim_death2` @24 | 33 | 24.0 | 138 | 237 |
| … @26 / @28 / @30 | 33 | 26/28/30 | 127 / 118 / 111 | 226 / 217 / **210** |
| super-long death (1 % of kills) | 137 | 14.0 | 979 | **1078** |
| POLEVAULTER `anim_death` @24 / @28 / @30 | 27 | 24/28/30 | 113 / 97 / 90 | 212 / 196 / **189** |
| any of the above while chilled | — | ×0.5 | ×2 | ≈ 2k + 99 |
| pool variants (`mInPool`) | — | — | — | k + 9 (fade 10) — **not reachable on level 7** |
| body reanim already gone | — | — | 0 | 0 (`UpdateDeath` → `DieNoLoot` immediately, `:9097-9101`) |

`k` is **not** reliably `ceil(frameCount·100/rate)`: `mAnimTime` accumulates in float32, so when
`frameCount·100/rate` is an exact integer the crossing may land one tick late, and which way it
goes is per-case. Measured: `anim_death`@24 (162.5) → 163 = ceil; @26 (150.0 exact) → **151**; @28
(139.29) → 140 = ceil; @30 (130.0 exact) → **130**; `anim_death2`@24 (137.5) → 138; @26 (126.92) →
127; @28 (117.86) → 118; @30 (110.0 exact) → **111**; polevault `anim_death`@24 (112.5) → 113; @28
(96.43) → 97; @30 (90.0 exact) → **90**; super-long@14 (978.57) → 979; mowed (100.0 exact) →
**101**. Table these — do not `ceilf` them in the kernel.

The "while chilled" row only bites if `mChilledCounter > 0` when `PlayDeathAnim` runs, because the
×0.5 is applied at rate-*apply* time (`:6658-6660` → `:6639`); a chill that expires mid-death
re-applies the full rate (`:4619-4625`) and shortens the remaining ticks.

Other death-time facts (S): `PlayDeathAnim` returns early (no dying phase) if already
DYING/BURNED/MOWERED (`:8928-8931`) or if the reanim lacks `anim_death` (`:8933-8937`);
`DAMAGE_DOESNT_LEAVE_BODY` → `DieNoLoot` at once (`:8962-8967`). A DYING zombie still runs
`UpdateZombieWalking` (`:4336`), and `UpdateAnimSpeed` takes the `IsDeadOrDying()` branch so the
clip keeps ticking at `ApplyAnimRate(mOriginalAnimRate)` (`:6658-6661`) — **not** at the
`mVelX`-derived walking rate (`mVelX` was zeroed at `:8988`). Walking still goes through the
`_ground` velocity branch (`:4127-4129`), so corpse motion = the death clip's `_ground` dx:
measured **net 0.0 px** for `anim_death` (38/38 transitions are exactly 0, i.e. the track is flat
over 178…216), `anim_death2`, and polevault `anim_death` ⇒ those corpses really do not move.
`anim_superlongdeath` is the exception: `_ground` 292…428 runs −28.5 → −35.4 (`aDistance` −6.9,
103 of 136 transitions zero, dx range −7.5 … +5.4), so that corpse twitches and drifts **+6.85 px
right** over its 978-tick death (only visible to anything that still reads a dying zombie's x).
`aFallTime` (foley only, `:9136-9180`): 0.788 super-long, 0.71 `anim_death2`, 0.77 `anim_death`,
0.68 `POLEVAULTER` — fraction-of-loop values matched with `ShouldTriggerTimedEvent`
(`Reanimator.cpp:1374-1384`).

---

## 4. (d) `anim_mowed`

`Zombie::MowDown` (`Zombie.cpp:8584-8670`) — the mowed clip is a **separate reanim object**, not
the body, and it is *not* started with `PlayReanim`, so the animation window comes from
`ReanimationInitialize` (`mFrameCount = tracks[0].mTransforms.count`, `Reanimator.cpp:410`):

```cpp
	Reanimation* aMoweredReanim = mApp->AddReanimation(0.0f, 0.0f, mRenderOrder, ReanimationType::REANIM_LAWN_MOWERED_ZOMBIE);
	aMoweredReanim->mAnimRate = 8.0f;                                       // Zombie.cpp:8665
	aMoweredReanim->mIsAttachment = false;                                  // :8666 (so EffectSystem::Update drives it)
	aMoweredReanim->mLoopType = ReanimLoopType::REANIM_PLAY_ONCE_AND_HOLD;   // :8667
	mMoweredReanimID = mApp->ReanimationGetID(aMoweredReanim);
	mZombiePhase = ZombiePhase::PHASE_ZOMBIE_MOWERED;                        // :8669
```

`REANIM_LAWN_MOWERED_ZOMBIE` = `reanim/LawnMoweredZombie.reanim` (`Reanimator.cpp:142`). Measured
(M): `fps = 12`, **one** track named `locator`, **8** frames, x column `[-10000(unset) → 0.0,
70.8, 134.9, 172.5, 175.5, 178.7, 181.8, 184.9]`, no `anim_*` clip tracks — hence no
`SetFramesForLayer`, `mFrameStart = 0` (`Reanimator.cpp:318`), `mFrameCount = 8`, `mAnimTime = 0`
(`Reanimator.cpp:312`). `UpdateAnimSpeed` never touches it (it is not the body reanim).

Advance per tick = `0.01 · 8.0 / 8 = 0.01` (float32 accumulation lands just under 1.0 after 100
adds) ⇒ **k = 101** reanim updates to `mLoopCount = 1` (`Reanimator.cpp:456-459`).

`Zombie::UpdateMowered` (`Zombie.cpp:9312-9321`), dispatched from `Zombie::Update:4329-4332`:

```cpp
	Reanimation* aMoweredReanim = mApp->ReanimationTryToGet(mMoweredReanimID);
	if (aMoweredReanim == nullptr || aMoweredReanim->mLoopCount > 0)
	{
		DropHead(0U);
		DropArm(0U);
		DieWithLoot();
	}
```

**Correction (round 21): 101 is animation updates, not simulation ticks.**
`Zombie::UpdateReanim:5365` also calls `aMoweredReanim->Update()` at the entity tail.
Because the separate object is not an attachment, it advances globally and at that tail.
It reaches loop 1 in the global phase of the 51st subsequent tick, and UpdateMowered removes it.
CUDA now models both updates and the imported progress; instant-removal exceptions remain.

The earlier **101-tick** claim for NORMAL / TRAFFIC_CONE / FLAG /
POLEVAULTER was incorrect. Clip and rate are type-independent. No fade is
involved. If the body reanim is missing the first update already kills (`mLoopCount` test passes
via the `nullptr` branch). Caveat (S): `MowDown` short-circuits for several types/phases
(`:8589-8641`: CATAPULT, ZAMBONI, DYING, IN_VAULT, flying, `mInPool` ⇒ instant `DieWithLoot`),
so 101 only applies to a zombie that actually enters `PHASE_ZOMBIE_MOWERED`.

Round 24 correction to the final caveat: 51 subsequent ticks apply to a newly mowed zombie.
`MowDown` keeps body health and eating status, drops helm/shield, and does not call StopEating.
Those retained fields remain observable until UpdateMowered removes the entity. The shared core's
native check covers initial ready-mower chomp=25, the slow-middle speed curve, and removal on tick 51.

---

## 5. (e) Speed table

`Zombie::PickRandomSpeed` (`Zombie.cpp:1165-1226`), evaluated in `if/else` order; it ends with
`UpdateAnimSpeed()` (`:1226`). Called at spawn (`:237`), on `StartWalkAnim` (`:6754`), on phase
changes (`:371/:387/:407/:1503/:1856/:1928/:2026/:2656/:2980/:4063`), and `ConvertToNormalZombie`
(`:6713`).

| type (engine id) | phase at level 7 | `mVelX` literal | cite |
|---|---|---|---|
| `ZOMBIE_NORMAL` (0) | `PHASE_ZOMBIE_NORMAL` | `RandRangeFloat(0.23f, 0.37f)` — continuous draw | `:1215` |
| `ZOMBIE_TRAFFIC_CONE` (2) | `PHASE_ZOMBIE_NORMAL` | `RandRangeFloat(0.23f, 0.37f)` | `:1215` |
| `ZOMBIE_FLAG` (1) | `PHASE_ZOMBIE_NORMAL` | **`0.45f` fixed** (grouped with DANCER/BACKUP/POGO) | `:1193-1197` |
| `ZOMBIE_POLEVAULTER` (3) | `PHASE_POLEVAULTER_PRE_VAULT` | `RandRangeFloat(0.66f, 0.68f)` | `:1199-1202` |
| `ZOMBIE_POLEVAULTER` (3) | `PHASE_POLEVAULTER_POST_VAULT` | falls through to `RandRangeFloat(0.23f, 0.37f)` | `:1215` |
| `ZOMBIE_POLEVAULTER` (3) | `PHASE_POLEVAULTER_IN_VAULT` | `mVelX = aJumpDistance / aAnimDuration`, `aJumpDistance = mX − aPlant->mX − 80`, `aAnimDuration = mFrameCount/mAnimRate·100` | `:1725/:1729/:1730/:1735` |

The ids in the task brief (`TRAFFIC_CONE(1)`, `FLAG(3)`, `POLEVAULTER(5)`) match neither
`enum ZombieType` (`ConstEnums.h:1340-1346`: NORMAL 0, FLAG 1, TRAFFIC_CONE 2, POLEVAULTER 3) nor
the `gZombieDefs` row order (`Zombie.cpp:81-84`, same sequence); no layer in this repo uses that
numbering — treat it as external and re-map before use.

* Side effects of the default branch (S): `mAnimTicksPerFrame = 12` when the draw `< 0.3f`,
  else `15` (`:1216-1223`). That field only feeds the legacy `mAnimCounter`/`mFrame` ladder
  (`:5013-5017`), **not** `mPosX`; the CUDA core can ignore it.
* `ZOMBIE_LIMP_SPEED_FACTOR = 2` (`Zombie.cpp:59`) is **not** a speed factor: it is used only as a
  multiplier on `mPhaseCounter` for `JACK_IN_THE_BOX` (`:506`) and `JALAPENO_HEAD` (`:797`), plus a
  bobsled rewind (`:5468`). No level-7 type touches it. There is no post-vault limp.
* Chilled animated walks: `ApplyAnimRate` ×0.5 (`:6639`, gate `IsMovingAtChilledSpeed()` =
  `mChilledCounter > 0`, `:6590-6593`), i.e. advance ×0.5 — **not** the `CHILLED_SPEED_FACTOR =
  0.4` (`Zombie.cpp:70`) used by the non-`_ground` branches (`:4117-4119`, `:4134-4136`). The two
  paths disagree for the four level-7 types only if a type loses its `_ground` track, which none of
  them do. The halving is applied only when `UpdateAnimSpeed()`/`SetAnimRate()` runs; the engine
  re-runs it when the chill **expires** (`:4619-4625`), so a chilled walk is a two-rate sequence
  (halved while `mChilledCounter > 0`, full after) rather than a blended speed, and the same ×0.5
  applies to a death clip started while chilled (`:6658-6660`). `IsImmobilizied()`
  (`mIceTrapCounter`/`mButteredCounter`, `:6585-6588`) pins the rate to 0 (`:6652-6655`) — total
  stop, x advance 0.
* `mVelY` **does not exist** on `Zombie` (`Zombie.h` has no such member; the only `mVelY` in
  `Zombie.cpp:1549` is on a `Projectile`). Vertical moves are teleports/curves (`mAltitude`,
  `mPosY = GetPosYBasedOnRow(mRow)`), never integrated velocity.
* Pole vault specifics (S): vault starts when `PHASE_POLEVAULTER_PRE_VAULT` finds a plant target
  (`:1709-1737`): phase set `:1725`, clip `anim_jump` `REANIM_PLAY_ONCE_AND_HOLD` @24.0 (`:1726`),
  window 50/43 (§2.1) ⇒ `aAnimDuration = 43/24·100 = 179.17` ticks (`:1729`), `mVelX =
  (mX − aPlant->mX − 80)/179.17` (`:1730`, `:1735`). During the vault `aSpeed = mVelX` with **no
  0.47/K factor** (`:4122-4125`) — `anim_jump`'s `_ground` has `aDistance == 0` (measured: x[50] =
  x[92] = −29.8), so `UpdateAnimSpeed`'s `_ground` branch is skipped by the `1e-6f` guard (`:6688`)
  and the rate stays 24.0. The jump ends when `mLoopCount > 0` (`:1765-1769`), which is
  **k = 180** updates at rate 24 (measured by the same replay), and then `mPosX -= 150.0f`
  (`:1768`), `mX = (int)mPosX` (`:1781`), phase → POST_VAULT (`:1782`), `StartWalkAnim(0)`
  (`:1785`) re-rolls `mVelX` to the 0.23–0.37 draw. Note the dead store at `:1789-1792`:
  `mPosX -= 150.0f·mAnimTime` is written and immediately restored from `aOldPosX`, so it is a pure
  no-op for x — do **not** implement it.
* `StartMindControlled` sets `mVelX = 0.17f` (`:4840`) — hypno path, out of level-7 scope.

---

## 6. Plant-side windows (id → reanim → measured clip spans)

`SeedType` ids from `ConstEnums.h:1035-1044`; `gPlantDefs` rows at `Plant.cpp:49-56`;
`ReanimationType` → file at `Reanimator.cpp:50-69`; `mReanimParamFlags` irrelevant here.

| seed (id) | reanim file (M) | fps | `anim_idle` | shoot clip | idle rate (S) | shoot rate / counter (S) |
|---|---|---|---|---|---|---|
| `PEASHOOTER` (0) | `reanim/PeaShooterSingle.reanim` (`:50`) | 12 | start 4, **25** | `anim_shooting` start 54, 25 | body `RandRangeFloat(15,20)` `Plant.cpp:226`; head same `:229`, window `anim_head_idle` (start 29, 25) | head `anim_shooting` @35, `mShootingCounter = 35` `Plant.cpp:751-760` |
| `SUNFLOWER` (1) | `reanim/SunFlower.reanim` (`:53`) | 12 | start 4, **25** | — | body `RandRangeFloat(10,15)` `Plant.cpp:167` (generic) | produces sun by `mLaunchRate = 2500` (`Plant.cpp:50`), no shoot clip |
| `CHERRYBOMB` (2) | `reanim/CherryBomb.reanim` (`:56`) | **30** | start 14, 13 | `anim_explode` start 0, **14** | generic 10–15 | detonation is a pure counter: `mDoSpecialCountdown = 100` `Plant.cpp:327`, decremented in `Plant::Update` `:2582-2588` → `DoSpecial()` `:4323-4335` (plant dies at `:4334`); `anim_explode` + `PLAY_ONCE_AND_HOLD` set at `:329-330`, no gameplay dependency on it. `Plant.cpp:4458` (`anim_explode` @26.0) is the *imitater* morph, not the cherry bomb. |
| `WALLNUT` (3) | `reanim/Wallnut.reanim` (`:51`) | 12 | start 0, **17** | — | generic 10–15 | blink tracks `anim_blink_twitch/twice/thrice` (17/5, 23/7, 31/11) |
| `POTATOMINE` (4) | `reanim/PotatoMine.reanim` (`:69`) | 12 | start 0, 1 (degenerate) | `anim_armed` start 20, **11** | **fixed `12.0f`** `Plant.cpp:341` | `mStateCountdown = 1500` armed timer `:346`; `anim_rise` start 1, 19 |
| `SNOWPEA` (5) | `reanim/SnowPea.reanim` (`:59`) | 12 | start 4, 25 | `anim_shooting` start 54, 25 | body `RandRangeFloat(15,20)` `:226` | same as peashooter (`@35`, counter 35) |
| `REPEATER` (7) | `reanim/PeaShooter.reanim` (`:60`) — note the swapped names | 12 | start 4, 25 | `anim_shooting` start 54, 25 | body `RandRangeFloat(15,20)` `:226` | head rate **45**, `mShootingCounter = 26` `:761-764` |

Two plant-side facts the CUDA core needs (S): a plant's body reanim is created with
`mIsAttachment = true` (`Plant.cpp:181`), so `EffectSystem::Update` **skips** it
(`EffectSystem.cpp:129`); attached reanims tick inside `Attachment::Update`
(`Attachment.cpp:104`, `:138-141`), reachable only through `AttachmentUpdateAndMove`, whose only
callers are `Coin.cpp:761`, `Projectile.cpp:972`, `Zombie.cpp:4392` — `Plant` never calls it.
Correction (round 21): `Plant::UpdateReanim:2791` explicitly calls `aBodyReanim->Update()`.
The earlier search limited to Attachment callers missed this direct call. Potato RISING does
depend on loop completion, and must inspect the prior tail's progress before advancing its body.

---

## 7. Consolidated numbers to hard-code

| name | value | status |
|---|---|---|
| tick → second | 0.01 (`SECONDS_PER_UPDATE`, float32-valued double) | S |
| `mAnimRate` unit | frames/s; `Zombie.reanim` / polevault / plant assets = 12, `CherryBomb` = 30 | S/M |
| walk `frameCount` | 47 (Zombie, both clips), 45 (polevault `anim_walk`), 37 (polevault `anim_run`) | M |
| walk `aDistance` (`_ground` x[90]−x[44]) | 49.8 px (Zombie), 67.8 px (polevault walk), 30.0 px (polevault run) | M |
| **K** (cycle mean) | `0.47 · frameCount/(frameCount−1)` = **0.480217** (Zombie walk, both clips), 0.480682 (pv walk), 0.483056 (pv run); measured cycle means fall in 0.4782–0.4849 (`mVelX`-dependent ±0.9 %) | M+S |
| walk cycle length | 461 ticks (`mVelX` 0.23), 354 (0.30), 287 (0.37), 236 (0.45, FLAG); polevault walk 628/481/390, run 213 at 0.30 and 97 at 0.66 | M |
| polevault vault | `anim_jump` 43 frames @24 ⇒ k = **180** ticks, `aSpeed = mVelX` (no K), landing `mPosX −= 150`, then speed re-roll | M+S |
| per-tick advance | 0 … 1.065·`mVelX`, ≈8.7 % of ticks exactly 0 (Zombie walk) | M |
| death fade | 100 ticks (`mInPool`: 10) | S |
| `anim_death` k | 163 / 151 / 140 / 130 (rates 24/26/28/30) → 262 / 250 / 239 / 229 ticks to `DieNoLoot` | M |
| `anim_death2` k | 138 / 127 / 118 / 111 → 237 / 226 / 217 / 210 | M |
| `anim_superlongdeath` k | 979 → 1078 (1 % of kills, `Rand(100) == 99`) | M |
| polevault `anim_death` k | 113 / 97 / 90 (24/28/30) → 212 / 196 / 189 | M |
| mowed | `frameCount = 8`, rate 8.0, k = **101** → `DieWithLoot` at tick T+101 | M+S |
| speeds | 0.23–0.37 (normal/cone/post-vault polevault), 0.45 fixed (flag), 0.66–0.68 (polevault pre-vault) | S |

---

## 8. Measured vs assumed vs unknown

* **Measured from `main.pak`** (XML, cross-checked against the shipped `.compiled` blob, which is
  byte-exactly decodable and agrees): every `frameStart`/`frameCount`, the `_ground` x columns and
  the derived `aDistance`, all `fps` values, the mowed clip's 8 frames, and every tick count in
  §2.3/§3.1/§4/§5 (`anim_jump` 180 included) — re-verified tick-by-tick in this session with the
  dump tool, no engine binary run.
* **Derived, not observed**: the closed form `K = 0.47·frameCount/(frameCount−1)` and the k/total
  formulas. They come from the literal code in §1–§4 and were then re-simulated with float32/double
  rounding matching C++ (`mAnimTime` float32, `SECONDS_PER_UPDATE * mAnimRate / mFrameCount`
  evaluated in double, `GetTrackVelocity` narrowed to float32, `FloatRoundToInt` per
  `PvzpCommon.h:203`).
* **Assumed**: that `PlayDeathAnim`/`MowDown` run early enough in tick `T` for the first reanim
  increment to land at `T+1` (tick order `Board.cpp:5762` before `:5815`). If damage is applied
  during the reanim pass instead, every death/mowed total shifts by ±1 tick. Cycle means are
  unaffected.
* **Unknown / not in this repo**: (1) the runtime RNG stream, so per-zombie `aDeathAnimRate`,
  `StartWalkAnim`'s `Rand(2)` clip choice and the `mVelX` draws cannot be predicted — only
  bracketed; (2) whether a planted body reanim (`mIsAttachment = true`) ticks at all, see §6;
  (3) `main.pak` is **not** in git and is not reproducible from repo contents, so the tables in
  §2.1/§2.4/§3/§6 must be re-extracted with the pak recipe in §0.1 if the asset set changes (the
  extraction/dump tool is `cuda/ReanimDump.cpp`, deliberately left untracked); (4) no `.reanim`
  file exists loose in the tree — anyone looking for one will find none.
