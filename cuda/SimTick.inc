// The tick.  Compiled by g++ (host, one lane, stride 1) and by nvcc (device, thread = lane) from this
// single source, so a disagreement between the two builds is a GPU bug and a disagreement between the
// host build and the original simulator is a mechanics bug -- never "which of my two rewrites is
// wrong".
//
// Coverage: adventure level 7 daytime grass with the deck peashooter/sunflower/cherry/wallnut/potato
// mine/snow pea/repeater against normal, cone-headed, flag and pole-vaulter zombies.  Anything the
// exporter marks unsupported never reaches a lane, so this file may assume those types.
//
// Source references and corrections live in cuda/spec/*.md. Ground movement and corpse lifetime
// use the imported animation state; earlier averaged movement / guessed lifetime constants were
// insufficient. MEMORY/docs/cuda-rollout-report.md records remaining gaps and measured evidence.
#ifndef PVZ_SIM_TICK_INC
#define PVZ_SIM_TICK_INC

#include "SimImpl.inc"
#include "Animation.inc"

namespace pvzc
{

namespace {

// ---------------------------------------------------------------------------
// Constants read out of the original
// ---------------------------------------------------------------------------

// Zombie.cpp:57/58 -- TICKS_BETWEEN_EATS is also the chew damage (DAMAGE_PER_EAT = TICKS_BETWEEN_EATS).
// Chilling doubles the interval to 8 (`:6863-6866`); the gate is on the zombie's *own age* (`:6867`).
constexpr int cEatInterval = 4;
constexpr int cEatDamage = 4;
// GroundAdvance uses the current reanimation frame, not a mean walking-speed factor.
// ApplyChill: mChilledCounter = max(aChillTime, current), aChillTime = 1000 (Zombie.cpp:7534).  Not 400.
constexpr int cChilledTicks = 1000;
// A snow pea does its damage plus the chill; the pea's own travel speed (Projectile MOTION_STRAIGHT).
constexpr float cPeaSpeed = 3.33f;
// Cherry bomb blast (radius 115, row range 1, burn -> instant kill, 100-tick mDoSpecialCountdown fuse)
// and potato mine (radius 60, row range 0, TakeDamage 1800, 1500-tick mStateCountdown arming --
// Plant.cpp:1177, NOT 500).  Radii are integers so BlastHit can square them without a float narrowing.
constexpr int cCherryFuseTicks = 100;
constexpr int cCherryRadius = 115;
constexpr int cPotatoArmTicks = 1500;
constexpr int cPotatoRadius = 60;
constexpr int cPotatoDamage = 1800;
// LawnMower::Update (LawnMower.cpp:193/242/267): roll-in is a 100-tick eased curve to x = -21; a
// triggered mower then crawls right at 3.33 px/update and despawns past 800.  There is no `x <= 30`
// trigger -- the READY mower sits at -21 and its 50-wide attack rect reaches a zombie's hurt rect only
// when the zombie is essentially at the house (mX in [-99,-8]), which the rect overlap below reproduces.
constexpr float cMowerSpeed = 3.33f;
constexpr float cMowerDespawnX = 800.0f;
constexpr int cMowerRollInTicks = 100;
constexpr float cMowerReadyX = -21.0f;
// Sky sun: Board::CreateCoin schedule -- min(950, 425 + 10n) + Rand(275), value 25.
constexpr int cSkySunMin = 425;
constexpr int cSkySunCap = 950;
constexpr int cSkySunStep = 10;
constexpr int cSkySunJitter = 275;
constexpr int cSkySunValue = 25;
// Plant.cpp:950 / :1034 -- the shooter's reset is mLaunchRate - Rand(15), the repeater's second pea
// fires when the same counter reads 25; a producer redraws [mLaunchRate-150, mLaunchRate] after its
// short first delay (the level table's prodMin/prodMax, [300, 1250] for the sunflower).
constexpr int cSecondPeaCounter = 25;
constexpr int cProduceRedrawJitter = 150;
// CheckForBoardEdge (Zombie.cpp:4545-4556): BOARD_EDGE is -100, and the pole vaulter uses -150.  The
// comparison is on the truncated int mX, so the loss fires the first update whose int(mPosX) <= edge.
constexpr float cHouseX = -100.0f;
constexpr float cHouseXVaulter = -150.0f;
// SpawnZombie (ZombieInitialize): mPosX = 780 + Rand(40), plus 40 on a flag wave.  IsOnBoard is
// provenance (a wave-spawned zombie is on board from birth), so it is set at spawn, not on a position
// test; 800 is only the right edge used to reject a pea that has flown past everything.
constexpr float cSpawnX = 780.0f;
constexpr int cSpawnJitter = 40;
constexpr int cFlagWaveSpawnBonus = 40;
constexpr float cRightEdgeX = 800.0f;
// UpdateZombieSpawning (Board.cpp:5346-5395): a whole wave is emitted in one tick, the countdown
// between waves is 2500 + Rand(600), the rush clamps it to 200, and a flag wave raises the banner 750
// ticks early.  lround(base * zombieCountMultiplier) zombies per wave, cycling mZombiesInWave on demand.
constexpr int cWaveInterval = 2500;
constexpr int cWaveIntervalRange = 600;
constexpr int cRushClamp = 200;
constexpr int cRushMinElapsed = 400;
constexpr int cHugeBannerTicks = 750;
constexpr int cFlagWavesPeriod = 10;       // GetNumWavesPerFlag == 10 in this scope
// Headless bleed-out (Zombie.cpp:4667-4677): 1 HP (3 if body max >= 500) on a Rand(5) == 0 draw.
constexpr int cBleedDraw = 5;
// GridToPixelX(col) == col * 80 + 40 (Board.cpp:8874): the coordinate the plant hurt rect and the
// zombie mouth rect are both measured in, so eating and the pea hit are computed on these numbers.
constexpr float cGridStepX = 80.0f;
constexpr float cGridOriginX = 40.0f;

// Rect widths of the two attack/hurt rectangles the eat and hit tests intersect, straight from
// GetZombieAttackRect / GetZombieRect / GetPlantRect / the projectile rect in projectiles.md.
//   normal/cone/flag mouth  [mX+20, mX+70]        (LoadPlainZombieReanim, Zombie.cpp:1011)
//   pole vaulter mouth      [mX-29, mX+41]        (Zombie.cpp:390, width 70)
//   zombie hurt rect        [mX+36, mX+78]        (mZombieRect = Rect(36,0,42,115), Zombie.cpp:199)
//   plant hurt rect         [mX+10, mX+70]        (Rect(mX+10, mY, 80-20, mHeight))
//   straight pea rect       [mX-15, mX+40]        (Rect(mX-15, mY, 40+15, mHeight), Projectile.cpp:1173)
constexpr int cMouthOffset = 20;
constexpr int cMouthWidth = 50;
constexpr int cEatMinimumOverlap = 20;
constexpr int cVaulterMouthBack = 29;      // mouth reaches 29 px behind the vaulter
constexpr int cVaulterMouthFront = 41;     // and 41 px ahead of it
constexpr int cHurtOffset = 36;
constexpr int cHurtWidth = 42;
constexpr int cPlantRectPad = 10;
constexpr int cPlantRectWidth = 80;
constexpr int cPeaBack = 15;
constexpr int cPeaWidth = 40;

// These three are defined below the tick functions that call them; declared here so the single source
// stays in reading order (plants, then projectiles, then zombies) without forward-use breaking the build.
PVZ_SIM int32_t RowHasZombieAhead(Lane lane, int32_t row, float muzzleX);
PVZ_SIM int32_t FindEatTarget(Lane lane, const LevelInfo& level, int32_t row, int32_t mz, int32_t vaulter);
PVZ_SIM void StopEating(Lane lane, const ZombieDef& def, int32_t slot, Rand& rand);

PVZ_SIM const PlantDef* TickPlantDef(const LevelInfo& level, int32_t seedType)
{
	for (int32_t index = 0; index < level.plantCount; ++index)
		if (level.plants[index].seedType == seedType) return &level.plants[index];
	return nullptr;
}

// Health of a freshly planted cell, from the level table the exporter filled.
PVZ_SIM int32_t FreshPlantHealth(const LevelInfo& level, int32_t seedType)
{
	const PlantDef* def = TickPlantDef(level, seedType);
	return def && def->health ? def->health : 300;
}

// A pole vaulter is the only level-7 type that walks at >= 0.6 px/update (pre-vault PickRandomSpeed is
// RandRangeFloat(0.66,0.68)); using the speed rather than a hard-coded seed id keeps this in step with
// whatever ids the exporter happens to use, and is the single property the vault mechanics hinge on.
PVZ_SIM int32_t IsVaulter(const ZombieDef& def) { return def.velMin >= 0.6f ? 1 : 0; }

// x-only GetRectOverlap (Board.cpp:8969): the width of the intersection of [aLo,aHi] and [bLo,bHi].
// Negative when they do not touch; the callers compare against 0 (peas) or 20 (eating) exactly as the
// original does.
PVZ_SIM int32_t RectOverlapX(int32_t aLo, int32_t aHi, int32_t bLo, int32_t bHi)
{
	const int32_t low = aLo > bLo ? aLo : bLo;
	const int32_t high = aHi < bHi ? aHi : bHi;
	return high - low;
}

// Board::GetCircleRectOverlap (:8975) specialised to an instant-kill blast (cherry / potato mine).
// Grass plants start at y=80+100r, zombies at y=50+100r; preserve their 30px offset.
// Normal-height zombies in this scope have mBodyY=0. Other heights need their actual hurt rectangle.
PVZ_SIM int32_t BlastHit(int32_t centerX, int32_t plantRow, int32_t zombieX, int32_t zombieRow,
	int32_t radius, int32_t rowRange)
{
	const int32_t d = zombieRow - plantRow;
	if (d > rowRange || d < -rowRange) return 0;
	// Zombie hurt rect is Rect(mX+36, mY, 42, 115); relative to the blast centre (plant mY + 40) the
	// vertical span is [d*100 - 70, d*100 + 45], and the centre sits at 0.
	// Plants start at y=80+100r, zombies at y=50+100r. Blast center is plant y+40.
	const int32_t topY = d * 100 - 70;
	const int32_t botY = d * 100 + 45;
	const int32_t nearY = topY > 0 ? topY : (botY < 0 ? botY : 0);
	const int32_t dy = -nearY;
	const int32_t lo = zombieX + cHurtOffset;
	const int32_t hi = zombieX + cHurtOffset + cHurtWidth;
	const int32_t nearX = centerX < lo ? lo : (centerX > hi ? hi : centerX);
	const int32_t dx = centerX - nearX;
	return dx * dx + dy * dy <= radius * radius ? 1 : 0;
}

// DieWithLoot/DieNoLoot share the board kill counter in this scope. Rendering-only charred
// animations are not zombie entities and must not occupy a simulation slot.
PVZ_SIM void RemoveZombie(Lane lane, int32_t slot)
{
	if (lane.i(F_ZOMBIE_FLAGS, slot) & ZB_ON_BOARD) ++lane.s(S_ZOMBIES_KILLED);
	lane.i(F_ZOMBIE_FLAGS, slot) |= ZB_DEAD;
	lane.i(F_ZOMBIE_TYPE, slot) = -1;
}

PVZ_SIM void ApplyBurn(Lane lane, int32_t slot)
{
	const int32_t flags = lane.i(F_ZOMBIE_FLAGS, slot);
	if (flags & (ZB_DEAD | ZB_BURNED)) return;
	// Zombie.cpp:8720-8830: ordinary headed zombies disappear immediately; headless
	// zombies stay burned for 300 ticks, while dying/airborne/mowed ones disappear too.
	if ((flags & (ZB_DYING | ZB_MOWERED)) || lane.i(F_ZOMBIE_PHASE, slot) == 12 || (flags & ZB_HAS_HEAD))
	{
		lane.i(F_ZOMBIE_BODY, slot) = 0;
		RemoveZombie(lane, slot);
		return;
	}
	lane.f(F_ZOMBIE_ANIM_RATE, slot) = 0.0f;
	lane.i(F_ZOMBIE_FLAGS, slot) = flags | ZB_BURNED;
	lane.i(F_ZOMBIE_PHASE, slot) = 2;
	lane.i(F_ZOMBIE_PHASE_COUNTER, slot) = 300;
}

// A zombie slot holds a live, still-playing zombie: present, not mDead and not in a terminal phase.
// IsDeadOrDying (Zombie.cpp:9643) = mDead || phase in {DYING, BURNED, MOWERED}.
PVZ_SIM int32_t ZombieLive(int32_t type, int32_t flags)
{
	return type >= 0 && !(flags & (ZB_DEAD | ZB_DYING | ZB_BURNED | ZB_MOWERED));
}

// The victory predicate, AreEnemyZombiesOnScreen (Board.cpp:262-274): a zombie only blocks the win if
// it is a live threat -- has its head, is not dead/dying/burned/mowed, is not mind-controlled and is
// on the board.  A headless or fading corpse does not count, which is what lets a board that has killed
// the last wave's heads reach the win on the tick the corpses would otherwise linger.
PVZ_SIM int32_t ZombieIsThreat(int32_t type, int32_t flags)
{
	if (type < 0) return 0;
	if (flags & (ZB_DEAD | ZB_DYING | ZB_BURNED | ZB_MOWERED)) return 0;
	if (!(flags & ZB_HAS_HEAD)) return 0;
	if (flags & ZB_MIND_CONTROLLED) return 0;
	if (!(flags & ZB_ON_BOARD)) return 0;
	return 1;
}

// TotalZombiesHealthInWave (Board.cpp, used by the rush test): the summed body + helm + shield * 0.2 +
// flying of the live, non-mind-controlled zombies that came from `wave`.  The comparison is float.
PVZ_SIM float TotalHealthInWave(Lane lane, int32_t wave)
{
	if (wave < 0) return 0.0f;
	float total = 0.0f;
	for (int32_t slot = 0; slot < lane.s(S_ZOMBIE_END); ++slot)
	{
		const int32_t type = lane.i(F_ZOMBIE_TYPE, slot);
		if (type < 0) continue;
		const int32_t flags = lane.i(F_ZOMBIE_FLAGS, slot);
		if (flags & (ZB_DEAD | ZB_DYING | ZB_BURNED | ZB_MOWERED | ZB_MIND_CONTROLLED)) continue;
		if (lane.i(F_ZOMBIE_FROM_WAVE, slot) != wave) continue;
		total += static_cast<float>(lane.i(F_ZOMBIE_BODY, slot))
			+ static_cast<float>(lane.i(F_ZOMBIE_HELM, slot))
			+ 0.2f * static_cast<float>(lane.i(F_ZOMBIE_SHIELD, slot));
	}
	return total;
}

// The headless environment banks every live sun one tick after it exists, so a coin never needs a
// position: it is a timer with a value.
PVZ_SIM void BankCoins(Lane lane)
{
	for (int32_t slot = 0; slot < kCoinSlots; ++slot)
	{
		if (lane.i(F_COIN_ACTIVE, slot) < 0) continue;
		const int32_t ticks = lane.i(F_COIN_TICKS, slot) - 1;
		if (ticks > 0) { lane.i(F_COIN_TICKS, slot) = ticks; continue; }
		lane.s(S_SUN) = Minimum(lane.s(S_SUN) + lane.i(F_COIN_VALUE, slot), 9990);
		lane.i(F_COIN_ACTIVE, slot) = -1;
		lane.i(F_COIN_VALUE, slot) = 0;
	}
}

PVZ_SIM int32_t AddCoin(Lane lane, int32_t value)
{
	const int32_t slot = FindFreeSlot(lane, F_COIN_ACTIVE, kCoinSlots);
	if (slot < 0) { lane.s(S_ERROR) |= ERR_CAPACITY; return -1; }
	lane.i(F_COIN_ACTIVE, slot) = 1;
	lane.i(F_COIN_VALUE, slot) = value;
	lane.i(F_COIN_TICKS, slot) = 1;
	// Board::AddCoin counts production at creation. Collection changes money, not production.
	lane.s(S_SUN_PRODUCED) = lane.s(S_SUN_PRODUCED) + value;
	return slot;
}

PVZ_SIM void AddProjectile(Lane lane, int32_t row, int32_t type, int32_t damage, int32_t chill,
	float x, float y)
{
	const int32_t slot = FindFreeSlot(lane, F_PROJ_TYPE, kProjectileSlots);
	if (slot < 0) { lane.s(S_ERROR) |= ERR_CAPACITY; return; }
	lane.i(F_PROJ_TYPE, slot) = type;
	lane.i(F_PROJ_ROW, slot) = row;
	lane.i(F_PROJ_DAMAGE, slot) = damage;
	lane.i(F_PROJ_CHILL, slot) = chill;
	lane.i(F_PROJ_AGE, slot) = 0;
	lane.f(F_PROJ_X, slot) = x;
	lane.f(F_PROJ_Y, slot) = y;
	lane.f(F_PROJ_VELX, slot) = cPeaSpeed;
	lane.f(F_PROJ_VELY, slot) = 0.0f;
}

// PlayDeathAnim selects only clips that exist on the zombie's resource.
PVZ_SIM void StartDeathAnimation(Lane lane, int32_t slot, int32_t pole, Rand& rand)
{
	float rate = 24.0f + 6.0f * rand.ZeroToOne();
	const int32_t hit = rand.Int(100);
	int32_t start = pole ? 138 : 178, frames = pole ? 27 : 39;
	if (!pole && hit == 99 && lane.i(F_ZOMBIE_CHILLED, slot) == 0)
	{
		int32_t onScreen = 0;
		for (int32_t z = 0; z < lane.s(S_ZOMBIE_END); ++z)
			if (ZombieIsThreat(lane.i(F_ZOMBIE_TYPE, z), lane.i(F_ZOMBIE_FLAGS, z))) ++onScreen;
		if (onScreen <= 5) { start = 292; frames = 137; rate = 14.0f; }
		else { start = 217; frames = 33; }
	}
	else if (!pole && hit > 50) { start = 217; frames = 33; }
	lane.i(F_ZOMBIE_PHASE, slot) = 1;
	lane.i(F_ZOMBIE_FADE, slot) = -1;
	PlayZombieAnimation(lane, slot, start, frames, rate, 3);
}

// Zombie::TakeDamage (Zombie.cpp:7973-8002): a flying hit is absorbed first, then the shield (unless it
// bypasses it), then the helm, then the body; excess damage carries through a broken layer.
// Body subtraction is unclamped with death tested as `<= 0` afterwards.
// Level 7 has no shields and no fliers, but the order is written out so a future type does not silently
// change it.
PVZ_SIM void ApplyBodyDamage(Lane lane, const LevelInfo& level, int32_t slot, int32_t damage, Rand& rand)
{
	int32_t left = damage;
	const int32_t shield = lane.i(F_ZOMBIE_SHIELD, slot);
	if (left > 0 && shield > 0)
	{
		const int32_t taken = Minimum(shield, left);
		lane.i(F_ZOMBIE_SHIELD, slot) = shield - taken;
		left -= taken;
	}
	const int32_t helm = lane.i(F_ZOMBIE_HELM, slot);
	if (left > 0 && helm > 0)
	{
		const int32_t taken = Minimum(helm, left);
		lane.i(F_ZOMBIE_HELM, slot) = helm - taken;
		left -= taken;
	}
	if (left <= 0) return;
	int32_t body = lane.i(F_ZOMBIE_BODY, slot) - left;       // unclamped, can go negative here
	const int32_t maxHealth = Maximum(lane.i(F_ZOMBIE_BODY_MAX, slot), 1);
	// UpdateDamageStates (Zombie.cpp:3904-3930): arm falls below 2/3, head below 1/3 (integer division).
	// Losing the head does NOT kill -- it only clears ZB_HAS_HEAD, which starts the bleed-out.
	if (body < (2 * maxHealth) / 3 && body > 0) lane.i(F_ZOMBIE_FLAGS, slot) &= ~ZB_HAS_ARM;
	if (body < maxHealth / 3) lane.i(F_ZOMBIE_FLAGS, slot) &= ~ZB_HAS_HEAD;
	if (body > 0) { lane.i(F_ZOMBIE_BODY, slot) = body; return; }
	// body <= 0: PlayDeathAnim -> PHASE_ZOMBIE_DYING, mVelX = 0, clamp body to 0, overkill discarded.
	lane.i(F_ZOMBIE_BODY, slot) = 0;
	lane.f(F_ZOMBIE_VELX, slot) = 0.0f;
	const int32_t flags = lane.i(F_ZOMBIE_FLAGS, slot);
	lane.i(F_ZOMBIE_FLAGS, slot) = (flags | ZB_DYING) & ~ZB_EATING;
	const int32_t zombieType = lane.i(F_ZOMBIE_TYPE, slot);
	const ZombieDef& def = level.zombies[Minimum(zombieType, kZombieTypes - 1)];
	StartDeathAnimation(lane, slot, IsVaulter(def), rand);
}

// Public entry the plant/mower/projectile callers use: it guards on IsDeadOrDying exactly as the
// original does (a dying zombie takes no further pea damage) and never on mind control (none at level 7).
PVZ_SIM void DamageZombie(Lane lane, const LevelInfo& level, int32_t slot, int32_t damage, Rand& rand)
{
	if (lane.i(F_ZOMBIE_TYPE, slot) < 0) return;
	const int32_t flags = lane.i(F_ZOMBIE_FLAGS, slot);
	if (flags & (ZB_DEAD | ZB_DYING | ZB_BURNED | ZB_MOWERED)) return;
	ApplyBodyDamage(lane, level, slot, damage, rand);
}

// The zombie a pea in this row is currently overlapping, i.e. the front-most (smallest int mX) live
// target whose hurt rect shares any x with the pea rect.  The original picks by argmin mX
// (projectiles.md: "the one with the smallest mX"), not by list order or by distance to the pea.
PVZ_SIM int32_t PeaTarget(Lane lane, int32_t row, int32_t peaX)
{
	int32_t best = -1;
	int32_t bestX = 0x7fffffff;
	for (int32_t slot = 0; slot < lane.s(S_ZOMBIE_END); ++slot)
	{
		if (lane.i(F_ZOMBIE_TYPE, slot) < 0) continue;
		if (lane.i(F_ZOMBIE_FLAGS, slot) & (ZB_DEAD | ZB_DYING | ZB_BURNED | ZB_MOWERED)) continue;
		// EffectedByDamage(1): ground peas pass over corpses and cannot hit a pole-vaulter in flight.
		if (lane.i(F_ZOMBIE_PHASE, slot) == 12) continue;
		if (lane.i(F_ZOMBIE_ROW, slot) != row) continue;
		const int32_t zx = static_cast<int32_t>(lane.f(F_ZOMBIE_X, slot));
		if (zx + cHurtOffset > 800) continue;
		if (RectOverlapX(peaX - cPeaBack, peaX + cPeaWidth, zx + cHurtOffset,
			zx + cHurtOffset + cHurtWidth) < 0) continue;           // the pea test accepts >= 0
		if (zx < bestX) { bestX = zx; best = slot; }
	}
	return best;
}

PVZ_SIM void UpdateProjectiles(Lane lane, const LevelInfo& level, Rand& rand)
{
	for (int32_t slot = 0; slot < lane.s(S_PROJECTILE_END); ++slot)
	{
		if (lane.i(F_PROJ_TYPE, slot) < 0) continue;
		// UpdateNormalMotion moves mPosX then checks collision, but GetProjectileRect still reads mX.
		// UpdateMotion synchronizes mX only after that call returns (Projectile.cpp:735, :773).
		const int32_t collisionX = static_cast<int32_t>(lane.f(F_PROJ_X, slot));
		lane.f(F_PROJ_X, slot) = lane.f(F_PROJ_X, slot) + lane.f(F_PROJ_VELX, slot);
		lane.i(F_PROJ_AGE, slot) = lane.i(F_PROJ_AGE, slot) + 1;
		// A pea that has flown past the right edge leaves the list (Projectile::Update off-board test).
		if (lane.f(F_PROJ_X, slot) > cRightEdgeX) { lane.i(F_PROJ_TYPE, slot) = -1; continue; }
		const int32_t hit = PeaTarget(lane, lane.i(F_PROJ_ROW, slot),
			collisionX);
		if (hit < 0) continue;
		DamageZombie(lane, level, hit, lane.i(F_PROJ_DAMAGE, slot), rand);
		if (lane.i(F_PROJ_CHILL, slot))
		{
			// ApplyChill keeps the longer of the two counters, so a re-chill refreshes a fading one.
			int32_t& chilled = lane.i(F_ZOMBIE_CHILLED, hit);
			if (chilled == 0) lane.f(F_ZOMBIE_ANIM_RATE, hit) *= 0.5f;
			if (chilled < cChilledTicks) chilled = cChilledTicks;
		}
		lane.i(F_PROJ_TYPE, slot) = -1;
	}
}

// PickRandomSpeed (Zombie.cpp:1165-1226): the def carries the type's range; FLAG is a fixed 0.45
// (velMin == velMax), NORMAL/CONE draw 0.23-0.37, and a pre-vault vaulter draws 0.66-0.68.  StopEating
// re-calls this (via StartWalkAnim), so an eating zombie that loses its target gets a fresh speed draw.
PVZ_SIM void RedrawZombieSpeed(Lane lane, const ZombieDef& def, int32_t slot, Rand& rand)
{
	const int32_t pole = IsVaulter(def);
	const float low = pole && lane.i(F_ZOMBIE_PHASE, slot) != 11 ? 0.23f : def.velMin;
	const float high = pole && lane.i(F_ZOMBIE_PHASE, slot) != 11 ? 0.37f : def.velMax;
	lane.f(F_ZOMBIE_VELX, slot) = high > low ? low + (high - low) * rand.ZeroToOne() : high;
	// StartWalkAnim: poles have only run/walk; normal/cone choose either walk, flags force walk2.
	const int32_t variant = rand.Int(2);
	if (pole && lane.i(F_ZOMBIE_PHASE, slot) == 11) PlayGroundAnimation(lane, slot, 13, 37);
	else if (pole) PlayGroundAnimation(lane, slot, 93, 45);
	else PlayGroundAnimation(lane, slot, def.velMin == def.velMax || variant == 0 ? 91 : 44, 47);
}

PVZ_SIM void UpdatePlant(Lane lane, const LevelInfo& level, int32_t slot, Rand& rand)
{
	const int32_t type = lane.i(F_PLANT_TYPE, slot);
	if (type < 0) return;
	const PlantDef* def = TickPlantDef(level, type);
	const int32_t behavior = def ? def->behavior : PB_NONE;
	const int32_t row = slot / kCols;
	const float plantX = lane.f(F_PLANT_X, slot);
	switch (behavior)
	{
	case PB_SHOOTER:
	{
		// Plant::UpdateAbilities runs UpdateShooting (:2544) before UpdateShooter (:2580), so the shot
		// countdown is stepped -- and the pea spawned -- before this tick's launch clock may arm a new one.
		//
		// The pea does not appear on the decision tick.  FindTargetAndFire (Plant.cpp:751-769) writes
		// mShootingCounter (35, or 26 for the repeater family) and returns without firing; UpdateShooting
		// fires when the countdown *decrements into* 1 (:3248 -> :3305).  That Fire is called with
		// theTargetZombie == nullptr, so the target test at the decision tick gates the animation and is
		// not an aim: a zombie that leaves the row in the meantime does not cancel the shot, and one that
		// walks in does not create it early.  plants.md §2.3, §2.6, §8 item 7.
		//
		// The tail at :3311-3417 may pin the counter back to 1 for the few ticks the head reanim has not
		// finished looping; §2.3 proves the pin cannot change any firing time and nothing here reads it as
		// a gate (only ScaredyShroom and Cactus do, both refused by KernelSupportsPlant), so the countdown
		// simply rests at 0 -- which is also the engine's steady state once the loop completes.
		int32_t& shooting = lane.i(F_PLANT_SHOOTING, slot);
		if (shooting != 0)
		{
			--shooting;
			if (shooting == 1)
				AddProjectile(lane, row, type == 5 ? 1 : 0, def->damage, def->chills,
					plantX + static_cast<float>(PeaHeadOffset(lane, slot, 0) + 24),
					lane.f(F_PLANT_Y, slot) + static_cast<float>(PeaHeadOffset(lane, slot, 1) - 33));
		}

		// Plant::UpdateShooter (Plant.cpp:941): the clock decrements whether or not the plant can see a
		// zombie, and resets to mLaunchRate - Rand(15) the moment it hits zero.  A row that was empty for
		// one cycle therefore *costs* that shot -- a kernel that held the clock at zero until a target
		// appeared would fire measurably more peas than the original.
		int32_t& launch = lane.i(F_PLANT_LAUNCH, slot);
		--launch;
		if (launch <= 0)
		{
			launch = def->launchRate - rand.Int(15);
			if (RowHasZombieAhead(lane, row, plantX)) shooting = def->shootCounter;
		}
		// The repeater's second pea is not a separate countdown: UpdateShooter arms again when the *same*
		// counter reads exactly 25, so it exists only for a plant that reset to something above 25.
		if (launch == cSecondPeaCounter && def->shots > 1 && RowHasZombieAhead(lane, row, plantX))
			shooting = def->shootCounter;
		break;
	}
	case PB_SUNPRODUCER:
	{
		// Plant::UpdateProduce (Plant.cpp:1026): decrement, and on reaching zero bank the sun and redraw
		// the clock from [mLaunchRate-150, mLaunchRate].  The first delay is shorter (Plant.cpp:192).
		int32_t& launch = lane.i(F_PLANT_LAUNCH, slot);
		--launch;
		if (launch > 0) break;
		AddCoin(lane, def->prodValue ? def->prodValue : 25);
		launch = rand.Range(def->launchRate - cProduceRedrawJitter, def->launchRate);
		break;
	}
	case PB_CHERRY:
	{
		// mDoSpecialCountdown fuse (Plant::UpdateAbilities step 10, :2582-2589).  On reaching zero DoSpecial
		// burns every zombie whose hurt rect overlaps the 115-px circle centred on the cell, row range 1
		// (Plant.cpp:4328 -> KillAllZombiesInRadius(..., 115, 1, true, ...)), then the plant Dies.
		int32_t& fuse = lane.i(F_PLANT_SPECIAL, slot);
		if (fuse > 0) { --fuse; }
		if (fuse != 0) break;
		const int32_t center = static_cast<int32_t>(plantX) + 40;   // DoSpecial aPosX = mX + mWidth/2
		for (int32_t zombie = 0; zombie < lane.s(S_ZOMBIE_END); ++zombie)
		{
			if (lane.i(F_ZOMBIE_TYPE, zombie) < 0) continue;
			const int32_t zflags = lane.i(F_ZOMBIE_FLAGS, zombie);
			// Cherry skips only mDead (the loop's guard) and an already-burned zombie (ApplyBurn's early
			// return); a dying or mowed one is burned through to DieWithLoot, which changes nothing on-screen.
			if (zflags & (ZB_DEAD | ZB_BURNED)) continue;
			if (static_cast<int32_t>(lane.f(F_ZOMBIE_X, zombie)) + cHurtOffset > 800
				&& lane.i(F_ZOMBIE_PHASE, zombie) != 12) continue; // EffectedByDamage(127)
			if (!BlastHit(center, row, static_cast<int32_t>(lane.f(F_ZOMBIE_X, zombie)),
				lane.i(F_ZOMBIE_ROW, zombie), cCherryRadius, 1)) continue;
			ApplyBurn(lane, zombie);
		}
		lane.i(F_PLANT_TYPE, slot) = -1;
		break;
	}
	case PB_POTATO:
	{
		// Plant::UpdateAbilities step 6 (:2546) decrements mStateCountdown before step 8's UpdatePotato
		// (:1166) reads it, so a mine planted with 1500 reaches RISING on the 1500th tick and arms once its
		// rise anim loops; an armed mine that finds a zombie on it detonates.  The kill is a 60-px circle at
		// mX+20 (aPosX = mX + mWidth/2 - 20, :4388), row range 0, NOT burn -> TakeDamage(1800, 18U).
		int32_t& arm = lane.i(F_PLANT_STATE_COUNTDOWN, slot);
		if (arm > 0) --arm;
		int32_t& state = lane.i(F_PLANT_STATE, slot);
		if (state == 0)
		{
			if (arm == 0)
			{
				state = 15; // STATE_POTATO_RISING, Plant::UpdatePotato :1155
				lane.f(F_PLANT_ANIM_TIME, slot) = 0.0f;
				lane.f(F_PLANT_ANIM_RATE, slot) = 18.0f;
				lane.i(F_PLANT_ANIM_FRAMES, slot) = 19; // measured PotatoMine anim_rise
				lane.i(F_PLANT_ANIM_LOOPS, slot) = 0;
			}
			break;
		}
		if (state == 15)
		{
			if (lane.i(F_PLANT_ANIM_LOOPS, slot) > 0) state = 16;
			break; // the armed branch first runs on the following tick
		}
		if (state != 16) break;
		// Trigger geometry is separate from the explosion circle (Plant.cpp:4864-4896).
		int32_t trigger = 0;
		int32_t attackX = static_cast<int32_t>(plantX), attackWidth = 55;
		for (int32_t zombie = 0; zombie < lane.s(S_ZOMBIE_END); ++zombie)
		{
			const int32_t flags = lane.i(F_ZOMBIE_FLAGS, zombie);
			if (lane.i(F_ZOMBIE_TYPE, zombie) < 0 || !(flags & ZB_HAS_HEAD)
				|| (flags & (ZB_DEAD | ZB_DYING | ZB_BURNED | ZB_MOWERED))) continue;
			if (lane.i(F_ZOMBIE_ROW, zombie) != row) continue;
			const int32_t phase = lane.i(F_ZOMBIE_PHASE, zombie);
			if (phase == 11 || phase == 12) continue;
			const int32_t zx = static_cast<int32_t>(lane.f(F_ZOMBIE_X, zombie));
			if (zx + cHurtOffset > 800) continue;
			if (lane.i(F_ZOMBIE_TYPE, zombie) == 3) { attackX += 40; attackWidth -= 40; }
			const int32_t extra = flags & ZB_EATING ? 30 : 0;
			if (RectOverlapX(attackX, attackX + attackWidth, zx + cHurtOffset,
				zx + cHurtOffset + cHurtWidth) >= -extra) { trigger = 1; break; }
		}
		if (!trigger) break;
		const int32_t center = static_cast<int32_t>(plantX) + 20;
		for (int32_t zombie = 0; zombie < lane.s(S_ZOMBIE_END); ++zombie)
		{
			if (!ZombieLive(lane.i(F_ZOMBIE_TYPE, zombie), lane.i(F_ZOMBIE_FLAGS, zombie))) continue;
			if (!BlastHit(center, row, static_cast<int32_t>(lane.f(F_ZOMBIE_X, zombie)),
				lane.i(F_ZOMBIE_ROW, zombie), cPotatoRadius, 0)) continue;
			if (lane.i(F_ZOMBIE_PHASE, zombie) == 12
				|| static_cast<int32_t>(lane.f(F_ZOMBIE_X, zombie)) + cHurtOffset > 800) continue;
			// KillAllZombiesInRadius uses TakeDamage(1800,18): DAMAGE_DOESNT_LEAVE_BODY.
			// Every supported type has at most 870 body+helm, so this removes it immediately.
			lane.i(F_ZOMBIE_BODY, zombie) = 0;
			RemoveZombie(lane, zombie);
		}
		lane.i(F_PLANT_TYPE, slot) = -1;
		break;
	}
	case PB_WALL:
	default:
		break;
	}
}

// The shooter only fires when a zombie is present in its row ahead of the muzzle; Plant::Fire is gated
// on a FindZombieInRow test, and holding the clock without checking would let peas spawn into an empty
// lane.  Reused by both the first and the repeater's second pea.
PVZ_SIM int32_t RowHasZombieAhead(Lane lane, int32_t row, float muzzleX)
{
	const int32_t from = static_cast<int32_t>(muzzleX);
	for (int32_t slot = 0; slot < lane.s(S_ZOMBIE_END); ++slot)
	{
		if (lane.i(F_ZOMBIE_TYPE, slot) < 0) continue;
		const int32_t flags = lane.i(F_ZOMBIE_FLAGS, slot);
		if (flags & (ZB_DEAD | ZB_DYING | ZB_BURNED | ZB_MOWERED)) continue;
		if (!(flags & ZB_ON_BOARD)) continue;
		if (lane.i(F_ZOMBIE_ROW, slot) != row) continue;
		const int32_t zx = static_cast<int32_t>(lane.f(F_ZOMBIE_X, slot));
		// The pea rect starts at muzzle-15; a zombie whose hurt rect reaches that far right can be hit.
		if (zx + cHurtOffset + cHurtWidth >= from - cPeaBack) return 1;
	}
	return 0;
}

PVZ_SIM void UpdateZombie(Lane lane, const LevelInfo& level, int32_t slot, Rand& rand)
{
	const int32_t type = lane.i(F_ZOMBIE_TYPE, slot);
	if (type < 0) return;
	int32_t& flags = lane.i(F_ZOMBIE_FLAGS, slot);
	if (flags & ZB_DEAD) return;
	const ZombieDef& def = level.zombies[Minimum(type, kZombieTypes - 1)];
	const int32_t pole = IsVaulter(def);
	++lane.i(F_ZOMBIE_AGE, slot);
	// Body animations advance at the entity tail; imported death progress is checked first.
	if (flags & ZB_BURNED)
	{
		if (--lane.i(F_ZOMBIE_PHASE_COUNTER, slot) == 0) RemoveZombie(lane, slot);
		return;
	}
	if (flags & ZB_MOWERED)
	{
		if (lane.i(F_ZOMBIE_ANIM_LOOPS, slot) > 0) RemoveZombie(lane, slot);
		return;
	}
	if (flags & ZB_DYING)
	{
		int32_t& fade = lane.i(F_ZOMBIE_FADE, slot);
		if (fade == -1 && lane.i(F_ZOMBIE_ANIM_LOOPS, slot) > 0) fade = 100;
		// Death and death2 have constant ground tracks; superlongdeath can slide in either direction.
		lane.f(F_ZOMBIE_X, slot) -= GroundAdvance(lane, slot);
		if (fade > 0 && --fade == 0) RemoveZombie(lane, slot);
		return;
	}

	int32_t& chill = lane.i(F_ZOMBIE_CHILLED, slot);
	if (chill > 0 && --chill == 0) lane.f(F_ZOMBIE_ANIM_RATE, slot) *= 2.0f;
	if (lane.i(F_ZOMBIE_PHASE_COUNTER, slot) > 0) --lane.i(F_ZOMBIE_PHASE_COUNTER, slot);
	const int32_t row = lane.i(F_ZOMBIE_ROW, slot);
	float& x = lane.f(F_ZOMBIE_X, slot);
	int32_t mz = static_cast<int32_t>(x); // the previous tick's integer mX
	int32_t& phase = lane.i(F_ZOMBIE_PHASE, slot);
	if (pole && phase == 11 && (flags & ZB_HAS_HEAD))
	{
		const int32_t target = FindEatTarget(lane, level, row, mz, 1);
		if (target >= 0)
		{
			phase = 12;
			PlayZombieAnimation(lane, slot, 50, 43, 24.0f, 3);
			const float duration = 43.0f / lane.f(F_ZOMBIE_ANIM_RATE, slot) * 100.0f;
			const int32_t distance = mz - static_cast<int32_t>(lane.f(F_PLANT_X, target)) - 80;
			lane.f(F_ZOMBIE_VELX, slot) = static_cast<float>(distance) / duration;
		}
	}
	else if (pole && phase == 12 && lane.i(F_ZOMBIE_ANIM_LOOPS, slot) > 0)
	{
		x -= 150.0f;
		mz = static_cast<int32_t>(x); // landing explicitly writes mX before movement/prey checks
		phase = 13;
		RedrawZombieSpeed(lane, def, slot, rand);
	}
	if (!(flags & ZB_EATING))
		x -= phase == 12 ? lane.f(F_ZOMBIE_VELX, slot) : GroundAdvance(lane, slot);

	const int32_t interval = chill > 0 ? cEatInterval * 2 : cEatInterval;
	if ((flags & ZB_HAS_HEAD) && phase != 11 && phase != 12 && lane.i(F_ZOMBIE_AGE, slot) % interval == 0)
	{
		const int32_t target = FindEatTarget(lane, level, row, mz, pole ? 2 : 0);
		if (target < 0)
		{
			if (flags & ZB_EATING) StopEating(lane, def, slot, rand);
		}
		else
		{
			if (!(flags & ZB_EATING))
			{
				flags |= ZB_EATING;
				PlayZombieAnimation(lane, slot, pole ? 165 : 138, pole ? 28 : 40, pole ? 20.0f : 36.0f, 0);
			}
			lane.i(F_ZOMBIE_TARGET, slot) = target;
			const PlantDef* prey = TickPlantDef(level, lane.i(F_PLANT_TYPE, target));
			if (!(prey && (prey->behavior == PB_CHERRY
				|| (prey->behavior == PB_POTATO && lane.i(F_PLANT_STATE, target) != 0))))
			{
				int32_t& hp = lane.i(F_PLANT_HP, target);
				hp -= cEatDamage;
				if (hp <= 0)
				{
					lane.i(F_PLANT_TYPE, target) = -1;
					++lane.s(S_PLANTS_EATEN);
					// StopEating happens only when a later chew check finds no target.
				}
			}
		}
	}
	const int32_t edge = pole ? static_cast<int32_t>(cHouseXVaulter) : static_cast<int32_t>(cHouseX);
	if (mz <= edge && (flags & ZB_HAS_HEAD))
	{
		if (flags & ZB_ON_BOARD) lane.s(S_RESULT) = -1;
	}
	else if (mz <= edge + 70 && !(flags & ZB_HAS_HEAD)) ApplyBodyDamage(lane, level, slot, 1800, rand);
	if (!(flags & ZB_HAS_HEAD) && ZombieLive(type, flags) && rand.Int(cBleedDraw) == 0)
		ApplyBodyDamage(lane, level, slot, lane.i(F_ZOMBIE_BODY_MAX, slot) >= 500 ? 3 : 1, rand);
}

// FindPlantTarget: overlap >=20. LoadPlainZombieReanim overwrites the constructor's 20-wide mouth
// with Rect(20,0,50,115), giving normal/cone/flag a window mX in [80c,80c+70]. A pre-vault pole's
// mouth is [-29,41]. Post-vault mouth restoration is handled with the vault state machine.
// Computed with the two rects on GridToPixelX coordinates so a vaulter and a normal zombie share one
// expression.  Row 5 (DIRT) and the off-lawn columns never have plantable cells, but the sweep is cheap.
PVZ_SIM int32_t FindEatTarget(Lane lane, const LevelInfo& level, int32_t row, int32_t mz, int32_t vaulter)
{
	if (row < 0 || row >= level.playableRows) return -1;
	const int32_t mouthLo = vaulter == 1 ? mz - cVaulterMouthBack : mz + (vaulter == 2 ? 50 : cMouthOffset);
	const int32_t mouthHi = vaulter == 1 ? mz + cVaulterMouthFront : mz + cMouthOffset + cMouthWidth;
	for (int32_t col = 0; col < kGridCols; ++col)
	{
		const int32_t cell = row * kCols + col;
		const int32_t plantType = lane.i(F_PLANT_TYPE, cell);
		if (plantType < 0) continue;
		// An armed/rising mine can be targeted and starts eating, but EatPlant does no damage to it.
		const int32_t plantX = static_cast<int32_t>(lane.f(F_PLANT_X, cell));
		const int32_t plantLo = plantX + cPlantRectPad;
		const int32_t plantHi = plantX + cPlantRectWidth - cPlantRectPad;
		if (RectOverlapX(mouthLo, mouthHi, plantLo, plantHi) >= cEatMinimumOverlap) return cell;
	}
	return -1;
}

PVZ_SIM void StopEating(Lane lane, const ZombieDef& def, int32_t slot, Rand& rand)
{
	lane.i(F_ZOMBIE_FLAGS, slot) &= ~ZB_EATING;
	lane.i(F_ZOMBIE_TARGET, slot) = -1;
	RedrawZombieSpeed(lane, def, slot, rand);
}

// Ready mowers.  A READY mower sits at -21 and kills any zombie whose hurt rect overlaps its 50-wide
// attack rect (that is mX in [-99,-8], not the old `x <= 30` guess); the first kill flips the row's
// mWaveRowGotLawnMowered to the current wave, which is what re-weights the row picker.
PVZ_SIM void UpdateMowers(Lane lane)
{
	for (int32_t slot = 0; slot < kMowerSlots; ++slot)
	{
		int32_t& state = lane.i(F_MOWER_STATE, slot);
		if (lane.i(F_MOWER_FLAGS, slot) & MB_GONE) continue;
		const int32_t row = slot;
		if (state == kMowerRollingIn)
		{
			int32_t& timer = lane.i(F_MOWER_TIMER, slot);
			if (timer > 0) --timer;
			if (timer == 0)
			{
				state = kMowerReady;
				lane.i(F_MOWER_FLAGS, slot) = MB_STATIONARY;
				lane.f(F_MOWER_X, slot) = cMowerReadyX;
			}
			continue;
		}
		const int32_t mowerX = static_cast<int32_t>(lane.f(F_MOWER_X, slot));
		// The attack-rect sweep runs in every non-squished state, READY included -- that is how a READY
		// mower is discovered by the zombie standing on it (LawnMower.cpp:207 skips only mDead).
		for (int32_t zombie = 0; zombie < lane.s(S_ZOMBIE_END); ++zombie)
		{
			if (lane.i(F_ZOMBIE_TYPE, zombie) < 0) continue;
			const int32_t zf = lane.i(F_ZOMBIE_FLAGS, zombie);
			if (zf & (ZB_DEAD | ZB_MOWERED)) continue;
			if (lane.i(F_ZOMBIE_ROW, zombie) != row) continue;
			const int32_t zx = static_cast<int32_t>(lane.f(F_ZOMBIE_X, zombie));
			if (RectOverlapX(mowerX, mowerX + 50, zx + cHurtOffset, zx + cHurtOffset + cHurtWidth) <= 0) continue;
			if (state == kMowerReady && !(zf & ZB_HAS_HEAD)) continue;   // bungee-style exception, moot here
			lane.i(F_MOWER_CHOMP, slot) = state == kMowerReady ? 25 : 50;
			if (state == kMowerReady)
			{
				lane.i(F_MOWER_FLAGS, slot) = MB_MOWING;
				state = kMowerTriggered;
				++lane.s(S_TRIGGERED_MOWERS);
				lane.i(F_ROW_GOT_MOWERED, row) = lane.s(S_WAVE);
			}
			// Dying/burned/airborne zombies disappear; walking ones use a separate, unchilled
			// LawnMoweredZombie animation. Imported roots carry that animation's actual progress.
			if ((zf & (ZB_DYING | ZB_BURNED)) || lane.i(F_ZOMBIE_PHASE, zombie) == 12)
			{
				RemoveZombie(lane, zombie);
				continue;
			}
			lane.i(F_ZOMBIE_HELM, zombie) = 0;
			lane.i(F_ZOMBIE_SHIELD, zombie) = 0;
			// MowDown drops helm/shield but retains body health and eating state until removal.
			lane.i(F_ZOMBIE_FLAGS, zombie) = zf | ZB_MOWERED;
			lane.i(F_ZOMBIE_PHASE, zombie) = 3;
			PlayZombieAnimation(lane, zombie, 0, 8, 8.0f, 3);
			lane.f(F_ZOMBIE_ANIM_RATE, zombie) = 8.0f;
		}
		if (state != kMowerTriggered) continue;
		float& mx = lane.f(F_MOWER_X, slot);
		float speed = cMowerSpeed;
		int32_t& chomp = lane.i(F_MOWER_CHOMP, slot);
		if (chomp > 0)
		{
			--chomp;
			const float time = static_cast<float>(chomp - 50) / -50.0f;
			if (time > 0.0f && time < 1.0f)
			{
				const float triangle = RoundedMulAdd(2.0f, time, -1.0f);
				const float bounce = 1.0f - (triangle < 0.0f ? -triangle : triangle);
				const float square = RoundedMulAdd(bounce, bounce, 0.0f);
				const float warped = 2.0f * bounce - square;
				speed = RoundedMulAdd(1.0f - cMowerSpeed, warped, cMowerSpeed);
			}
		}
		mx += speed;
		if (mx > cMowerDespawnX)
		{
			state = kMowerSquished;
			lane.i(F_MOWER_FLAGS, slot) = MB_GONE;
		}
	}
}

PVZ_SIM void UpdatePackets(Lane lane)
{
	for (int32_t slot = 0; slot < kPacketSlots; ++slot)
	{
		if (lane.i(F_PACKET_TYPE, slot) < 0) continue;
		if (lane.i(F_PACKET_ACTIVE, slot)) continue;
		int32_t& counter = lane.i(F_PACKET_COUNTER, slot);
		const int32_t refresh = lane.i(F_PACKET_REFRESH, slot);
		// SeedPacket::Update() gates on `!mActive && mRefreshing`, and a packet that was picked up but
		// never planted (or removed by a conveyor) sits at mRefreshTime == 0 with mRefreshing == false
		// (SeedPacket.cpp:140-146).  refresh == 0 is that flag: counting it down would revive a card the
		// original has dropped for good.
		if (refresh <= 0) continue;
		++counter;
		// Strictly greater (SeedPacket.cpp:175), so a card is back after refreshTime + 1 ticks, and the
		// counter is cleared rather than clamped (:177) -- a root exported from the engine carries 0 here
		// and a clamped value would not round-trip.
		if (counter > refresh)
		{
			counter = 0;
			lane.i(F_PACKET_ACTIVE, slot) = 1;
		}
	}
}

PVZ_SIM void SpawnSkySun(Lane lane, Rand& rand)
{
	int32_t& countdown = lane.s(S_SUN_COUNT_DOWN);
	--countdown;
	if (countdown != 0) return;
	AddCoin(lane, cSkySunValue);
	const int32_t fallen = lane.s(S_NUM_SUNS_FALLEN) + 1;
	lane.s(S_NUM_SUNS_FALLEN) = fallen;
	const int32_t base = Minimum(cSkySunMin + cSkySunStep * fallen, cSkySunCap);
	countdown = base + rand.Int(cSkySunJitter);
}

// Board::PickRowForNewZombie + PvzpPickFromSmoothArray (waves.md §3).  The temporary mWeight is
// rewritten every pick from the mower-protection rule; only mLastPicked / mSecondLastPicked persist
// across ticks (F_ROW_LAST_PICKED / F_ROW_SECOND_LAST_PICKED).  On level-7 grass rows 0-4 are eligible
// and row 5 (DIRT) has weight 0.
PVZ_SIM float SmoothWeight(float w, float lastPicked, float secondLastPicked)
{
	if (w < 1e-6f) return 0.0f;
	const float e1 = 1.0f / w;
	const float e2 = e1 * 2.0f;
	const float adv1 = lastPicked + 1.0f - e1;
	const float adv2 = secondLastPicked + 1.0f - e2;
	const float f1 = 1.0f + adv1 / e1 * 2.0f;             // literal order: divide, then *2, then +1
	const float f2 = 1.0f + adv2 / e2 * 2.0f;
	float ff = f1 * 0.75f + f2 * 0.25f;
	if (ff < 0.01f) ff = 0.01f;
	if (ff > 100.0f) ff = 100.0f;
	return w * ff;
}

PVZ_SIM int32_t PickRow(Lane lane, const LevelInfo& level, Rand& rand)
{
	float weight[kRows];
	float last[kRows];
	float second[kRows];
	float totalWeight = 0.0f;
	for (int32_t row = 0; row < kRows; ++row)
	{
		last[row] = lane.f(F_ROW_LAST_PICKED, row);
		second[row] = lane.f(F_ROW_SECOND_LAST_PICKED, row);
		if (row >= level.playableRows) { weight[row] = 0.0f; }     // PLANTROW_DIRT / off-lawn -> ineligible
		else
		{
			const int32_t wavesMowered = lane.s(S_WAVE) - lane.i(F_ROW_GOT_MOWERED, row);
			weight[row] = wavesMowered <= 1 ? 0.01f : (wavesMowered <= 2 ? 0.5f : 1.0f);
		}
		totalWeight += weight[row];
	}
	if (totalWeight <= 0.0f) return 0;
	const float normalize = 1.0f / totalWeight;
	float totalAdjusted = 0.0f;
	for (int32_t row = 0; row < kRows; ++row)
		totalAdjusted += SmoothWeight(weight[row] * normalize, last[row], second[row]);
	if (totalAdjusted <= 0.0f) return 0;
	const float draw = totalAdjusted * rand.ZeroToOne();
	float acc = 0.0f;
	int32_t pick = kRows - 1;
	for (int32_t row = 0; row < kRows - 1; ++row)
	{
		acc += SmoothWeight(weight[row] * normalize, last[row], second[row]);
		if (draw <= acc) { pick = row; break; }
	}
	// PvzpUpdateSmoothArrayPick: every eligible row ages its pick counters, the chosen one resets.
	for (int32_t row = 0; row < kRows; ++row)
	{
		if (weight[row] <= 0.0f) continue;
		lane.f(F_ROW_LAST_PICKED, row) = last[row] + 1.0f;
		lane.f(F_ROW_SECOND_LAST_PICKED, row) = second[row] + 1.0f;
	}
	lane.f(F_ROW_SECOND_LAST_PICKED, pick) = lane.f(F_ROW_LAST_PICKED, pick);
	lane.f(F_ROW_LAST_PICKED, pick) = 0.0f;
	return pick;
}

PVZ_SIM void SpawnZombie(Lane lane, const LevelInfo& level, int32_t zombieType, int32_t wave, int32_t flagWave,
	Rand& rand)
{
	const int32_t slot = FindFreeSlot(lane, F_ZOMBIE_TYPE, kZombieSlots);
	if (slot < 0) { lane.s(S_ERROR) |= ERR_CAPACITY; return; }
	const ZombieDef& def = level.zombies[Minimum(zombieType, kZombieTypes - 1)];
	// AddZombie picks the row before per-entity draws. Type-specific initialization overrides
	// the generic spawn x for flags and poles; flag-wave bonus must not survive those overrides.
	lane.i(F_ZOMBIE_ROW, slot) = PickRow(lane, level, rand);
	lane.i(F_ZOMBIE_TYPE, slot) = zombieType;
	lane.i(F_ZOMBIE_BODY, slot) = def.bodyHealth;
	lane.i(F_ZOMBIE_BODY_MAX, slot) = Maximum(def.bodyHealth, 1);
	lane.i(F_ZOMBIE_HELM, slot) = def.helmHealth;
	lane.i(F_ZOMBIE_SHIELD, slot) = def.shieldHealth;
	lane.i(F_ZOMBIE_FLYING, slot) = 0;
	lane.i(F_ZOMBIE_PHASE, slot) = IsVaulter(def) ? 11 : 0;
	lane.i(F_ZOMBIE_PHASE_COUNTER, slot) = 0;
	lane.i(F_ZOMBIE_FADE, slot) = -1;
	lane.i(F_ZOMBIE_FLAGS, slot) = ZB_HAS_HEAD | ZB_HAS_ARM | ZB_ON_BOARD;
	lane.i(F_ZOMBIE_CHILLED, slot) = 0;
	lane.i(F_ZOMBIE_AGE, slot) = 0;
	lane.i(F_ZOMBIE_FROM_WAVE, slot) = wave;
	lane.i(F_ZOMBIE_TARGET, slot) = -1;
	float spawnX = cSpawnX + static_cast<float>(rand.Int(cSpawnJitter));
	if (flagWave) spawnX += static_cast<float>(cFlagWaveSpawnBonus);
	if (zombieType == 1) spawnX = 800.0f;
	if (IsVaulter(def)) spawnX = 870.0f + static_cast<float>(rand.Int(10));
	lane.f(F_ZOMBIE_X, slot) = spawnX;
	lane.f(F_ZOMBIE_Y, slot) = 50.0f + 100.0f * static_cast<float>(lane.i(F_ZOMBIE_ROW, slot));
	RedrawZombieSpeed(lane, def, slot, rand);
	lane.f(F_ZOMBIE_VELY, slot) = 0.0f;
	UpdateZombieAnimation(lane, slot); // ZombieInitialize calls UpdateReanim once before returning
}

PVZ_SIM int32_t IsFlagWave(int32_t wave) { return wave >= 0 && wave % cFlagWavesPeriod == cFlagWavesPeriod - 1; }

// PickZombieWaves/PickZombieType for replay adventure level 7 (Board.cpp:578-783, :2416-2520).
// Only future waves reach this function.  Compose the wave from the request's future seed, never
// from the table saved in the real board.  Per-wave streams have the same sampling distribution,
// without storing 30 full tables per lane; they also keep candidates on common random futures even
// when their actions consume different amounts of in-play RNG.  Level 7 introduces no new type and
// has no survival caps or cross-wave type quotas.
PVZ_SIM int32_t DrawFutureWave(const LevelInfo& level, int32_t wave, int32_t futureSeed, int32_t* types)
{
	Rand waveRand;
	waveRand.Seed(wave, futureSeed);
	int32_t points = wave * 2 / 5 + 1;
	int32_t count = 0;
	if (IsFlagWave(wave))
	{
		const int32_t plain = Minimum(points, 8);
		points = static_cast<int32_t>(points * 2.5f);
		for (int32_t index = 0; index < plain; ++index) types[count++] = 0;
		types[count++] = 1;
		points -= plain * level.zombies[0].value + level.zombies[1].value;
	}
	// The full level's final wave, even when a task caps completion earlier (main.cpp:269-280).
	if (wave == kMaxWaves - 1)
	{
		for (int32_t type = 0; type < kZombieTypes; ++type)
		{
			if (!level.allowed[type]) continue;
			int32_t present = 0;
			for (int32_t index = 0; index < count; ++index) present |= types[index] == type;
			if (!present && count < kWaveZombieSlots)
			{
				types[count++] = type;
				points -= level.zombies[type].value;
			}
		}
	}
	while (points > 0 && count < kWaveZombieSlots)
	{
		int32_t total = 0;
		for (int32_t type = 0; type < kZombieTypes; ++type)
		{
			const ZombieDef& def = level.zombies[type];
			if (level.allowed[type] && wave + 1 >= def.firstAllowedWave && points >= def.value)
				total += def.pickWeight;
		}
		if (total <= 0) break;
		int32_t draw = waveRand.Int(total);
		int32_t picked = 0;
		for (int32_t type = 0; type < kZombieTypes; ++type)
		{
			const ZombieDef& def = level.zombies[type];
			if (!level.allowed[type] || wave + 1 < def.firstAllowedWave || points < def.value) continue;
			draw -= def.pickWeight;
			if (draw < 0) { picked = type; break; }
		}
		types[count++] = picked;
		points -= level.zombies[picked].value;
	}
	return count;
}

// Board::UpdateZombieSpawning (waves.md §1.4), the whole per-tick wave state machine: the huge-wave
// banner gate, the single countdown decrement, the 催波 clamp, the flag-wave banner trigger at 5, and the
// whole-wave emission at 0.  Emission rewrites mZombieHealthWaveStart / mZombieHealthToNextWave and the
// next wave's countdown from the rules at :5374-5398.
PVZ_SIM void UpdateWaves(Lane lane, const LevelInfo& level, const WaveTable& waves, Rand& rand)
{
	if (lane.s(S_RESULT) != 0) return;
	const int32_t numWaves = waves.waveCount;

	if (lane.s(S_HUGE_WAVE_COUNT_DOWN) > 0)
	{
		int32_t& banner = lane.s(S_HUGE_WAVE_COUNT_DOWN);
		--banner;
		if (banner == 0) lane.s(S_ZOMBIE_COUNT_DOWN) = 1;          // falls through and emits this tick
		else if (banner != 725) return;                            // only the 725 tick continues
	}

	if (lane.s(S_WAVE) >= numWaves) return;                        // last wave already emitted (Board.cpp:5340)

	int32_t& countdown = lane.s(S_ZOMBIE_COUNT_DOWN);
	--countdown;                                                   // the only per-tick decrement (:5346)

	// 催波: once 400 ticks into the interval, if the health left alive from the wave just spawned is at or
	// below the threshold set at emission, jump the wait to 200 so the next wave comes early.
	const int32_t wave = lane.s(S_WAVE);
	if (countdown > cRushClamp && lane.s(S_ZOMBIE_COUNT_DOWN_START) - countdown > cRushMinElapsed
		&& TotalHealthInWave(lane, wave - 1) <= static_cast<float>(lane.s(S_HEALTH_TO_NEXT_WAVE)))
		countdown = cRushClamp;

	if (countdown == 5)
	{
		if (IsFlagWave(wave)) { lane.s(S_HUGE_WAVE_COUNT_DOWN) = cHugeBannerTicks; return; }
	}

	if (countdown != 0) return;

	// SpawnZombieWave (waves.md §1.4:4918): lround(base * multiplier) zombies, cycling mZombiesInWave,
	// all emitted in this single tick.  std::lround rounds halves away from zero, which RoundHalfAway does.
	int32_t types[kWaveZombieSlots];
	const int32_t base = DrawFutureWave(level, wave, lane.s(S_FUTURE_SEED), types);
	const int32_t spawnCount = base > 0
		? static_cast<int32_t>(RoundHalfAway(static_cast<double>(base) * static_cast<double>(level.zombieCountMultiplier)))
		: 0;
	const int32_t flag = IsFlagWave(wave);
	for (int32_t index = 0; index < spawnCount; ++index)
		SpawnZombie(lane, level, types[base > 0 ? index % base : 0], wave, flag, rand);
	lane.s(S_WAVE) = wave + 1;

	// Post-emission timers (Board.cpp:5374-5398).  mZombieHealthWaveStart is the health of the wave that
	// just spawned; the next flag wave gets a fixed 4500 wait, otherwise 2500+Rand(600) and a 0.5-0.65
	// rush threshold truncated to int.
	const int32_t emitted = lane.s(S_WAVE) - 1;
	lane.s(S_HEALTH_WAVE_START) = static_cast<int32_t>(TotalHealthInWave(lane, emitted));
	const int32_t next = lane.s(S_WAVE);
	if (next < numWaves && IsFlagWave(next))
	{
		lane.s(S_HEALTH_TO_NEXT_WAVE) = 0;
		lane.s(S_ZOMBIE_COUNT_DOWN) = 4500;                        // ZOMBIE_COUNTDOWN_BEFORE_FLAG
	}
	else
	{
		const float threshold = (0.5f + 0.15f * rand.ZeroToOne()) * static_cast<float>(lane.s(S_HEALTH_WAVE_START));
		lane.s(S_HEALTH_TO_NEXT_WAVE) = static_cast<int32_t>(threshold);   // float -> int32 truncation
		lane.s(S_ZOMBIE_COUNT_DOWN) = cWaveInterval + rand.Int(cWaveIntervalRange);
	}
	lane.s(S_ZOMBIE_COUNT_DOWN_START) = lane.s(S_ZOMBIE_COUNT_DOWN);
}

// LawnApp's terminal test (board.md §530/§535): the board is WON once the last wave has been emitted and
// AreEnemyZombiesOnScreen() has gone false.  The loss path writes S_RESULT directly in UpdateZombie.
PVZ_SIM void CheckVictory(Lane lane, const WaveTable& waves)
{
	if (lane.s(S_RESULT) != 0) return;
	if (lane.s(S_WAVE) < waves.waveCount) return;
	for (int32_t slot = 0; slot < lane.s(S_ZOMBIE_END); ++slot)
	{
		if (ZombieIsThreat(lane.i(F_ZOMBIE_TYPE, slot), lane.i(F_ZOMBIE_FLAGS, slot))) return;
	}
	lane.s(S_RESULT) = 1;
}

}  // namespace

// The original's order inside one tick, as the environment drives it: coins banked, then plants,
// zombies, projectiles, mowers and seed packets, then mMainCounter++, then the sky-sun schedule and
// the wave emitter (cuda/spec/board.md).
PVZ_SIM void UpdateTick(Lane lane, const LevelInfo& level, const WaveTable& waves, Rand& rand)
{
	BankCoins(lane);
	UpdateMowedAnimations(lane); // EffectSystem skips attached plant/zombie bodies
	for (int32_t slot = 0; slot < kPlantSlots; ++slot)
	{
		UpdatePlant(lane, level, slot, rand);
		UpdatePlantAnimation(lane, slot);
	}
	for (int32_t slot = 0; slot < lane.s(S_ZOMBIE_END); ++slot)
	{
		UpdateZombie(lane, level, slot, rand);
		UpdateZombieAnimation(lane, slot);
	}
	UpdateProjectiles(lane, level, rand);
	UpdateMowers(lane);
	UpdatePackets(lane);
	lane.s(S_TICK) = lane.s(S_TICK) + 1;
	SpawnSkySun(lane, rand);
	UpdateWaves(lane, level, waves, rand);
	CheckVictory(lane, waves);
}

// ---------------------------------------------------------------------------
// The wait round
// ---------------------------------------------------------------------------

// The rows whose mower is still READY.  EnvironmentWaitEvents watches the *set* of ready defenses and
// ends the wait when one leaves it (protocol.md 1.5: the HasNew arguments are passed in reverse), so a
// bit per row is all the state the event needs.
PVZ_SIM int32_t DefenseMask(Lane lane)
{
	int32_t mask = 0;
	for (int32_t row = 0; row < kMowerSlots; ++row)
		if (lane.i(F_MOWER_STATE, row) == kMowerReady) mask |= 1 << row;
	return mask;
}

// The left zone threshold, on the coordinate the original publishes: LawnApp.cpp:1566 substitutes the
// six-significant-digit render only within 1 px of the line, so a zombie hovering at x = 160 crosses
// when the printed value says it does, not when the float does.
PVZ_SIM int32_t InLeftZone(float position)
{
	double value = static_cast<double>(position);
	if (AbsDouble(value - 160.0) < 1.0) value = RenderedPublicCoordinate(value);
	return value <= 160.0 ? 1 : 0;
}

// Live zombies standing at or left of x = 160, one bit per slot.  The membership filter the original
// applies is `!mDead` only (LawnApp.cpp:1561, zombies.md §3): a fading corpse -- DYING, MOWERED or
// burned -- stays a member until DieNoLoot sets mDead, so those are *not* skipped here.  Only the
// ZB_DEAD bit and the on-board provenance gate membership.
PVZ_SIM uint64_t LeftZoneMask(Lane lane)
{
	uint64_t mask = 0;
	for (int32_t slot = 0; slot < lane.s(S_ZOMBIE_END); ++slot)
	{
		if (lane.i(F_ZOMBIE_TYPE, slot) < 0) continue;
		const int32_t flags = lane.i(F_ZOMBIE_FLAGS, slot);
		if (!(flags & ZB_ON_BOARD)) continue;
		if (flags & ZB_DEAD) continue;
		if (InLeftZone(lane.f(F_ZOMBIE_X, slot))) mask |= 1ull << slot;
	}
	return mask;
}

PVZ_SIM int32_t WaitRound(Lane lane, const LevelInfo& level, const WaveTable& waves, Rand& rand,
	int32_t requested)
{
	// An already-terminal board advances nothing at all (EnvironmentWaitEvents.h: `if terminal, return`).
	if (lane.s(S_RESULT) != 0) return 0;
	int32_t beforeDefenses = DefenseMask(lane);
	uint64_t beforeLeft = LeftZoneMask(lane);
	int32_t steps = 0;
	while (steps < requested)
	{
		UpdateTick(lane, level, waves, rand);
		++steps;
		if (lane.s(S_RESULT) != 0) break;
		const int32_t defenses = DefenseMask(lane);
		if ((beforeDefenses & ~defenses) != 0) break;
		const uint64_t left = LeftZoneMask(lane);
		if ((left & ~beforeLeft) != 0) break;
		beforeDefenses = defenses;
		beforeLeft = left;
	}
	return steps;
}

// ---------------------------------------------------------------------------
// Decisions
// ---------------------------------------------------------------------------

// Mirrors EnvironmentAnyLegalPlant: a card that can be picked up, costs at most the current sun, and
// has at least one empty lawn cell.  Cells outside the playable rows are not plantable on grass.
PVZ_SIM int32_t AnyLegalPlant(Lane lane, const LevelInfo& level)
{
	const int32_t sun = lane.s(S_SUN);
	const int32_t rows = Maximum(level.playableRows, 1);
	for (int32_t packet = 0; packet < kPacketSlots; ++packet)
	{
		const int32_t type = lane.i(F_PACKET_TYPE, packet);
		if (type < 0 || !lane.i(F_PACKET_ACTIVE, packet)) continue;
		if (sun < lane.i(F_PACKET_COST, packet)) continue;
		for (int32_t row = 0; row < rows; ++row)
			for (int32_t col = 0; col < kGridCols; ++col)
				if (lane.i(F_PLANT_TYPE, row * kCols + col) < 0) return 1;
	}
	return 0;
}

// The legal cells of `seedType`, i.e. the empty playable cells.  Kept as a test rather than a list
// because the choice loop only ever needs "is this one legal" and "which packet has any".
PVZ_SIM int32_t CanPlantAt(Lane lane, const LevelInfo& level, int32_t row, int32_t col)
{
	if (row < 0 || row >= Maximum(level.playableRows, 1)) return 0;
	if (col < 0 || col >= kGridCols) return 0;
	return lane.i(F_PLANT_TYPE, row * kCols + col) < 0 ? 1 : 0;
}

// EnvironmentPlant's whole effect, in the original's order: re-check legality, deduct sun, create the
// plant with the constructor's state, and deactivate the card into its refresh clock.
PVZ_SIM int32_t ActionPlant(Lane lane, const LevelInfo& level, int32_t packet, int32_t row, int32_t col,
	Rand& rand)
{
	if (packet < 0 || packet >= kPacketSlots) return 0;
	const int32_t type = lane.i(F_PACKET_TYPE, packet);
	if (type < 0 || !lane.i(F_PACKET_ACTIVE, packet)) return 0;
	if (!CanPlantAt(lane, level, row, col)) return 0;
	const PlantDef* def = TickPlantDef(level, type);
	if (!def) return 0;
	const int32_t cost = lane.i(F_PACKET_COST, packet);
	if (lane.s(S_SUN) < cost) return 0;
	lane.s(S_SUN) = lane.s(S_SUN) - cost;
	const int32_t slot = row * kCols + col;
	lane.i(F_PLANT_TYPE, slot) = type;
	lane.i(F_PLANT_HP, slot) = def->health ? def->health : FreshPlantHealth(level, type);
	lane.i(F_PLANT_HP_MAX, slot) = def->health ? def->health : FreshPlantHealth(level, type);
	lane.i(F_PLANT_STATE, slot) = 0;
	lane.i(F_PLANT_STATE_COUNTDOWN, slot) = def->behavior == PB_POTATO ? cPotatoArmTicks : 0;
	lane.i(F_PLANT_SPECIAL, slot) = def->behavior == PB_CHERRY ? cCherryFuseTicks : 0;
	lane.i(F_PLANT_SHOOTING, slot) = 0;
	lane.i(F_PLANT_ANIM, slot) = 0;
	lane.i(F_PLANT_ONBOARD, slot) = 1;
	// GridToPixelX(col) = col * 80 + 40 -- the coordinate the eat and hit rects are measured against.
	lane.f(F_PLANT_X, slot) = cGridOriginX + cGridStepX * static_cast<float>(col);
	lane.f(F_PLANT_Y, slot) = 100.0f * static_cast<float>(row) + 80.0f;
	// Shooters keep anim_idle (start 4, 25 frames) on their body; only the head switches to shoot.
	lane.f(F_PLANT_ANIM_TIME, slot) = 0.0f;
	lane.i(F_PLANT_ANIM_LOOPS, slot) = 0;
	lane.i(F_PLANT_ANIM_FRAMES, slot) = def->behavior == PB_SHOOTER ? 25 : 0;
	lane.f(F_PLANT_ANIM_RATE, slot) = def->behavior == PB_SHOOTER ? 15.0f + 5.0f * rand.ZeroToOne() : 0.0f;
	// Shot clocks (Plant.cpp:188): a plant with mLaunchRate > 0 starts on a *drawn* delay -- a sun
	// producer a short first one, a shooter anywhere in [0, mLaunchRate].  A draw of 0 means the plant
	// fires on its first tick, which is why the countdown is allowed to start at zero.
	if (def->launchRate > 0)
		lane.i(F_PLANT_LAUNCH, slot) = def->behavior == PB_SUNPRODUCER
			? rand.Range(def->prodMin, def->prodMax)
			: rand.Range(0, def->launchRate);
	else
		lane.i(F_PLANT_LAUNCH, slot) = 0;
	lane.i(F_PLANT_LAUNCH_RATE, slot) = Maximum(def->launchRate, 1);
	lane.i(F_PACKET_ACTIVE, packet) = 0;
	lane.i(F_PACKET_COUNTER, packet) = 0;
	// SeedPacket::WasPlanted() (SeedPacket.cpp:872-894) starts the cooldown by writing
	// mRefreshTime = Plant::GetRefreshTime(type) right there; a packet dealt at level start carries
	// mRefreshTime == 0 (SeedBank::SetPacketType, :1111-1113), so this is where the number first exists.
	// LoadRoot used to guess it from the table on import instead, which is what made the cooldown channel
	// disagree with the engine's own feature build for a card the original had dropped.
	lane.i(F_PACKET_REFRESH, packet) = PlantRefresh(level, type);
	lane.i(F_PLANTED_BY_TYPE, packet) = lane.i(F_PLANTED_BY_TYPE, packet) + 1;
	lane.s(S_PLANTS_PLANTED) = lane.s(S_PLANTS_PLANTED) + 1;
	return 1;
}

}  // namespace pvzc

#endif
