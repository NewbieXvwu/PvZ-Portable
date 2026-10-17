// The tick.  Compiled by g++ (host, one lane, stride 1) and by nvcc (device, thread = lane) from this
// single source, so a disagreement between the two builds is a GPU bug and a disagreement between the
// host build and the original simulator is a mechanics bug -- never "which of my two rewrites is
// wrong".
//
// Coverage: adventure level 7 daytime grass with the deck peashooter/sunflower/cherry/wallnut/potato
// mine/snow pea/repeater against normal, cone-headed, flag and pole-vaulter zombies.  Anything the
// exporter marks unsupported never reaches a lane, so this file may assume those types.
//
// Every number marked "PROVISIONAL" is a constant whose Zombie.cpp / Plant.cpp / Board.cpp line the
// mechanic specs (cuda/spec/*.md) must confirm; the phase-1 agreement test is what catches the rest.
#ifndef PVZ_SIM_TICK_INC
#define PVZ_SIM_TICK_INC

#include "SimImpl.inc"

namespace pvzc
{

namespace {

// ---------------------------------------------------------------------------
// Constants read out of the original
// ---------------------------------------------------------------------------

// Zombie.cpp:67/68 -- TICKS_BETWEEN_EATS is also the chew damage (DAMAGE_PER_EAT = TICKS_BETWEEN_EATS).
constexpr int cEatInterval = 4;
constexpr int cEatDamage = 4;
// Zombie.cpp:70 CHILLED_SPEED_FACTOR; the counter a snow pea sets is PROVISIONAL.
constexpr float cChilledSpeedFactor = 0.4f;
constexpr int cChilledTicks = 400;
// A zombie fades for this many ticks before the slot is freed (death anim), PROVISIONAL.
constexpr int cDeathFadeTicks = 100;
// Projectile::Update: peas travel right at 3.33 px per tick, PROVISIONAL on the exact literal.
constexpr float cPeaSpeed = 3.33f;
// The horizontal window in which a pea is considered to have reached a zombie, PROVISIONAL.
constexpr float cHitWindow = 20.0f;
// Cherry bomb (radius 115, 1800 damage) and potato mine (radius 60, 1800, 500-tick arming).
constexpr int cCherryFuseTicks = 100;
constexpr float cCherryRadius = 115.0f;
constexpr float cCherryRowGap = 80.0f;
constexpr int cCherryDamage = 1800;
constexpr int cPotatoArmTicks = 500;
constexpr float cPotatoRadius = 60.0f;
constexpr int cPotatoDamage = 1800;
// Lawn mower rolling speed and the x at which it leaves the board, PROVISIONAL.
constexpr float cMowerSpeed = 3.33f;
constexpr float cMowerDespawnX = 800.0f;
constexpr float cMowerTriggerX = 30.0f;
constexpr int cMowerRollInTicks = 60;
// Sky sun: Board::CreateCoin schedule -- min(950, 425 + 10n) + Rand(275), value 25.
constexpr int cSkySunMin = 425;
constexpr int cSkySunCap = 950;
constexpr int cSkySunStep = 10;
constexpr int cSkySunJitter = 275;
constexpr int cSkySunValue = 25;
// Plant.cpp:950 / :1034 -- the shooter's reset is mLaunchRate - Rand(15), a producer's redraw is
// [mLaunchRate-150, mLaunchRate], and the repeater's second pea fires when the counter reads 25.
// The producer's short *first* delay is the level table's prodMin/prodMax (300 .. mLaunchRate/2).
constexpr int cSecondPeaCounter = 25;
constexpr int cProduceRedrawJitter = 150;
// A zombie that walks this far left has reached the house.
constexpr float cHouseX = -100.0f;
// Spawn x: 780 + Rand(40), and a zombie counts as on board once it is left of the right edge.
constexpr float cSpawnX = 780.0f;
constexpr int cSpawnJitter = 40;
constexpr float cOnBoardX = 800.0f;
// One zombie spawned per tick while a wave is emitting, PROVISIONAL (the original staggers by a table).
constexpr int cWaveSpawnInterval = 1;
// Plant::UpdateShooter fires the repeater's second pea this many ticks after the first.
constexpr int cSecondPeaDelay = 25;
// The chomp step pair (Zombie.cpp:3476 / Plant.cpp:4232) nets to zero inside one tick, so the walking
// position is unchanged by eating; the spec confirms the sign and magnitude.
constexpr float cChewStepX = 0.0f;
// A plant's cell-centre offset from the left edge of its grid square (Board::ConvertGridXtoScreenX).
constexpr float cPlantOffsetX = 40.0f;
constexpr float cPlantOffsetY = 40.0f;

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

// A zombie slot holds a live, still-playing zombie: not yet removed and not still fading out.
PVZ_SIM int32_t ZombieLive(int32_t type, int32_t flags)
{
	return type >= 0 && !(flags & ZB_DEAD) && !(flags & ZB_DYING);
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
		lane.s(S_SUN) += lane.i(F_COIN_VALUE, slot);
		lane.s(S_SUN_PRODUCED) += lane.i(F_COIN_VALUE, slot);
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

// helm before body: a cone-headed zombie absorbs pea damage into the cone until it is gone.
PVZ_SIM void DamageZombie(Lane lane, int32_t slot, int32_t damage)
{
	if (lane.i(F_ZOMBIE_TYPE, slot) < 0) return;
	const int32_t flags = lane.i(F_ZOMBIE_FLAGS, slot);
	if (flags & (ZB_DEAD | ZB_DYING)) return;
	int32_t left = damage;
	const int32_t helm = lane.i(F_ZOMBIE_HELM, slot);
	if (helm > 0)
	{
		const int32_t taken = Minimum(helm, left);
		lane.i(F_ZOMBIE_HELM, slot) = helm - taken;
		left -= taken;
	}
	const int32_t shield = lane.i(F_ZOMBIE_SHIELD, slot);
	if (left > 0 && shield > 0)
	{
		const int32_t taken = Minimum(shield, left);
		lane.i(F_ZOMBIE_SHIELD, slot) = shield - taken;
		left -= taken;
	}
	lane.i(F_ZOMBIE_BODY, slot) = lane.i(F_ZOMBIE_BODY, slot) - left;
	if (lane.i(F_ZOMBIE_BODY, slot) > 0) return;
	// Death is not removal: the zombie keeps occupying its slot (and stops acting) for the fade.
	lane.i(F_ZOMBIE_FLAGS, slot) = (flags | ZB_DYING) & ~ZB_EATING;
	lane.i(F_ZOMBIE_PHASE_COUNTER, slot) = cDeathFadeTicks;
}

// The leftmost live zombie in a row that is still in front of `xFrom`, i.e. a shot that can reach it.
PVZ_SIM int32_t TargetZombie(Lane lane, int32_t row, float xFrom)
{
	int32_t best = -1;
	float bestX = 1e30f;
	for (int32_t slot = 0; slot < kZombieSlots; ++slot)
	{
		if (!ZombieLive(lane.i(F_ZOMBIE_TYPE, slot), lane.i(F_ZOMBIE_FLAGS, slot))) continue;
		if (lane.i(F_ZOMBIE_ROW, slot) != row) continue;
		if (!(lane.i(F_ZOMBIE_FLAGS, slot) & ZB_ON_BOARD)) continue;
		const float x = lane.f(F_ZOMBIE_X, slot);
		if (x < xFrom - cHitWindow) continue;
		if (x < bestX) { bestX = x; best = slot; }
	}
	return best;
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
		// Plant::UpdateShooter (Plant.cpp:941): the clock decrements whether or not the plant can see a
		// zombie, and resets to mLaunchRate - Rand(15) the moment it hits zero.  A row that was empty for
		// one cycle therefore *costs* that shot -- a kernel that held the clock at zero until a target
		// appeared would fire measurably more peas than the original.
		int32_t& launch = lane.i(F_PLANT_LAUNCH, slot);
		--launch;
		if (launch <= 0)
		{
			launch = def->launchRate - rand.Int(15);
			if (TargetZombie(lane, row, plantX) >= 0)
				AddProjectile(lane, row, type, def->damage, def->chills, plantX, lane.f(F_PLANT_Y, slot));
		}
		// The repeater's second pea is not a separate countdown: UpdateShooter fires again when the *same*
		// counter reads exactly 25, so it exists only for a plant that reset to something above 25.
		if (launch == cSecondPeaCounter && def->shots > 1 && TargetZombie(lane, row, plantX) >= 0)
			AddProjectile(lane, row, type, def->damage, def->chills, plantX, lane.f(F_PLANT_Y, slot));
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
		int32_t& fuse = lane.i(F_PLANT_SPECIAL, slot);
		if (fuse > 0) { --fuse; }
		if (fuse != 0) break;
		for (int32_t zombie = 0; zombie < kZombieSlots; ++zombie)
		{
			if (lane.i(F_ZOMBIE_TYPE, zombie) < 0) continue;
			const int32_t zrow = lane.i(F_ZOMBIE_ROW, zombie);
			if (zrow < row - 1 || zrow > row + 1) continue;
			if (row == zrow)
			{
				if (AbsDouble(static_cast<double>(lane.f(F_ZOMBIE_X, zombie)) - static_cast<double>(plantX))
					> cCherryRadius) continue;
			}
			else if (AbsDouble(static_cast<double>(lane.f(F_ZOMBIE_X, zombie)) - static_cast<double>(plantX))
				> cCherryRadius + cCherryRowGap) continue;
			DamageZombie(lane, zombie, cCherryDamage);
			lane.i(F_ZOMBIE_FLAGS, zombie) |= ZB_BURNED;
		}
		lane.i(F_PLANT_TYPE, slot) = -1;
		break;
	}
	case PB_POTATO:
	{
		int32_t& arm = lane.i(F_PLANT_STATE_COUNTDOWN, slot);
		if (arm > 0) { --arm; break; }
		for (int32_t zombie = 0; zombie < kZombieSlots; ++zombie)
		{
			if (!ZombieLive(lane.i(F_ZOMBIE_TYPE, zombie), lane.i(F_ZOMBIE_FLAGS, zombie))) continue;
			if (lane.i(F_ZOMBIE_ROW, zombie) != row) continue;
			if (AbsDouble(static_cast<double>(lane.f(F_ZOMBIE_X, zombie)) - static_cast<double>(plantX))
				> cPotatoRadius) continue;
			DamageZombie(lane, zombie, cPotatoDamage);
			lane.i(F_PLANT_TYPE, slot) = -1;
			break;
		}
		break;
	}
	case PB_WALL:
	default:
		break;
	}
}

// A pea moving left-to-right hits the front-most zombie it has reached.  Scanning by smallest x keeps
// the choice independent of slot order; the original iterates its list, which the spec must confirm
// agrees with this whenever two zombies overlap the same window.
PVZ_SIM void UpdateProjectiles(Lane lane)
{
	for (int32_t slot = 0; slot < kProjectileSlots; ++slot)
	{
		if (lane.i(F_PROJ_TYPE, slot) < 0) continue;
		lane.f(F_PROJ_X, slot) = lane.f(F_PROJ_X, slot) + lane.f(F_PROJ_VELX, slot);
		lane.i(F_PROJ_AGE, slot) = lane.i(F_PROJ_AGE, slot) + 1;
		const float x = lane.f(F_PROJ_X, slot);
		if (x > cOnBoardX + 40.0f) { lane.i(F_PROJ_TYPE, slot) = -1; continue; }
		const int32_t row = lane.i(F_PROJ_ROW, slot);
		int32_t hit = -1;
		float hitX = 1e30f;
		for (int32_t zombie = 0; zombie < kZombieSlots; ++zombie)
		{
			if (!ZombieLive(lane.i(F_ZOMBIE_TYPE, zombie), lane.i(F_ZOMBIE_FLAGS, zombie))) continue;
			if (lane.i(F_ZOMBIE_ROW, zombie) != row) continue;
			if (!(lane.i(F_ZOMBIE_FLAGS, zombie) & ZB_ON_BOARD)) continue;
			const float zx = lane.f(F_ZOMBIE_X, zombie);
			if (x < zx - cHitWindow || x > zx + cHitWindow) continue;
			if (zx < hitX) { hitX = zx; hit = zombie; }
		}
		if (hit < 0) continue;
		DamageZombie(lane, hit, lane.i(F_PROJ_DAMAGE, slot));
		if (lane.i(F_PROJ_CHILL, slot)) lane.i(F_ZOMBIE_CHILLED, hit) = cChilledTicks;
		lane.i(F_PROJ_TYPE, slot) = -1;
	}
}

PVZ_SIM void UpdateZombie(Lane lane, const LevelInfo& level, int32_t slot, Rand& rand)
{
	const int32_t type = lane.i(F_ZOMBIE_TYPE, slot);
	if (type < 0) return;
	int32_t& flags = lane.i(F_ZOMBIE_FLAGS, slot);
	if (flags & ZB_DEAD) return;
	if (flags & ZB_DYING)
	{
		int32_t& counter = lane.i(F_ZOMBIE_PHASE_COUNTER, slot);
		if (counter > 0) --counter;
		if (counter != 0) return;
		// Freed: the fade ended, and this is where mZombiesKilled moves (Zombie::Update's death tail
		// records it when the reanim finishes and the list prunes the zombie).
		if (flags & ZB_ON_BOARD) lane.s(S_ZOMBIES_KILLED) = lane.s(S_ZOMBIES_KILLED) + 1;
		lane.i(F_ZOMBIE_TYPE, slot) = -1;
		for (int32_t field = F_ZOMBIE_ROW; field <= F_ZOMBIE_TARGET; ++field)
			lane.i(static_cast<IntField>(field), slot) = 0;
		return;
	}

	const int32_t row = lane.i(F_ZOMBIE_ROW, slot);
	float& x = lane.f(F_ZOMBIE_X, slot);
	if (row >= 0 && row < kRows && !(flags & ZB_ON_BOARD))
	{
		// Walking in from off the right edge; IsOnBoard() flips once it is inside the lawn.
		if (x < cOnBoardX) flags |= ZB_ON_BOARD;
	}
	if (lane.i(F_ZOMBIE_CHILLED, slot) > 0) --lane.i(F_ZOMBIE_CHILLED, slot);

	int32_t target = lane.i(F_ZOMBIE_TARGET, slot);
	if (target >= 0 && (target < 0 || target >= kPlantSlots || lane.i(F_PLANT_TYPE, target) < 0))
	{
		target = -1;
		lane.i(F_ZOMBIE_TARGET, slot) = -1;
	}
	if (target < 0 && row >= 0 && row < level.playableRows)
	{
		// Mouth reach: the plant in the cell the zombie is standing on.  PROVISIONAL on the offset.
		const int32_t col = FeatureColumn(static_cast<double>(x) + 40.0);
		const int32_t cell = row * kCols + col;
		if (col < kGridCols && lane.i(F_PLANT_TYPE, cell) >= 0)
		{
			target = cell;
			lane.i(F_ZOMBIE_TARGET, slot) = cell;
			flags |= ZB_EATING;
		}
	}
	if (target >= 0)
	{
		lane.i(F_ZOMBIE_AGE, slot) = lane.i(F_ZOMBIE_AGE, slot) + 1;
		if (lane.i(F_ZOMBIE_AGE, slot) % cEatInterval != 0) return;
		int32_t& hp = lane.i(F_PLANT_HP, target);
		hp -= cEatDamage;
		// Zombie.cpp:3476 / Plant.cpp:4232 pair the chomp with a small step back and forth.
		x += cChewStepX;
		if (hp > 0) return;
		lane.i(F_PLANT_TYPE, target) = -1;
		lane.s(S_PLANTS_EATEN) = lane.s(S_PLANTS_EATEN) + 1;
		lane.i(F_ZOMBIE_TARGET, slot) = -1;
		flags &= ~ZB_EATING;
		return;
	}
	flags &= ~ZB_EATING;

	float speed = lane.f(F_ZOMBIE_VELX, slot);
	if (lane.i(F_ZOMBIE_CHILLED, slot) > 0) speed *= cChilledSpeedFactor;
	x -= speed;
	if (x <= cHouseX && (flags & ZB_HAS_HEAD) && (flags & ZB_ON_BOARD))
	{
		lane.s(S_RESULT) = -1;
		// The original ends the board on this test; the zombie itself is left in place.
		flags |= ZB_DEAD;
	}
	(void)rand;
}

// Ready mowers only.  A mower that has already rolled is gone; the row's weight change lives in
// F_ROW_GOT_MOWERED, which the wave row picker reads.
PVZ_SIM void UpdateMowers(Lane lane)
{
	for (int32_t slot = 0; slot < kMowerSlots; ++slot)
	{
		int32_t& state = lane.i(F_MOWER_STATE, slot);
		if (lane.i(F_MOWER_FLAGS, slot) & MB_GONE) continue;
		if (state == kMowerRollingIn)
		{
			int32_t& timer = lane.i(F_MOWER_TIMER, slot);
			if (timer > 0) --timer;
			if (timer == 0) { state = kMowerReady; lane.i(F_MOWER_FLAGS, slot) = MB_STATIONARY; }
			continue;
		}
		const int32_t row = slot;
		if (state == kMowerReady)
		{
			for (int32_t zombie = 0; zombie < kZombieSlots; ++zombie)
			{
				if (!ZombieLive(lane.i(F_ZOMBIE_TYPE, zombie), lane.i(F_ZOMBIE_FLAGS, zombie))) continue;
				if (lane.i(F_ZOMBIE_ROW, zombie) != row) continue;
				if (!(lane.i(F_ZOMBIE_FLAGS, zombie) & ZB_ON_BOARD)) continue;
				if (lane.f(F_ZOMBIE_X, zombie) > cMowerTriggerX) continue;
				state = kMowerTriggered;
				lane.i(F_MOWER_FLAGS, slot) = MB_MOWING;
				lane.s(S_TRIGGERED_MOWERS) = lane.s(S_TRIGGERED_MOWERS) + 1;
				lane.i(F_ROW_GOT_MOWERED, row) = 1;
				break;
			}
			continue;
		}
		if (state != kMowerTriggered) continue;
		float& mx = lane.f(F_MOWER_X, slot);
		mx += cMowerSpeed;
		for (int32_t zombie = 0; zombie < kZombieSlots; ++zombie)
		{
			if (!ZombieLive(lane.i(F_ZOMBIE_TYPE, zombie), lane.i(F_ZOMBIE_FLAGS, zombie))) continue;
			if (lane.i(F_ZOMBIE_ROW, zombie) != row) continue;
			if (AbsDouble(static_cast<double>(lane.f(F_ZOMBIE_X, zombie)) - static_cast<double>(mx))
				> cHitWindow) continue;
			// A mower kills outright rather than dealing a damage number.
			lane.i(F_ZOMBIE_BODY, zombie) = 0;
			lane.i(F_ZOMBIE_HELM, zombie) = 0;
			lane.i(F_ZOMBIE_SHIELD, zombie) = 0;
			lane.i(F_ZOMBIE_FLAGS, zombie) |= ZB_DYING;
			lane.i(F_ZOMBIE_PHASE_COUNTER, zombie) = cDeathFadeTicks;
		}
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
		++counter;
		const int32_t refresh = lane.i(F_PACKET_REFRESH, slot);
		if (refresh > 0 && counter >= refresh)
		{
			counter = refresh;
			lane.i(F_PACKET_ACTIVE, slot) = 1;
		}
	}
}

PVZ_SIM void SpawnSkySun(Lane lane, Rand& rand)
{
	int32_t& countdown = lane.s(S_SUN_COUNT_DOWN);
	if (countdown > 0) { --countdown; return; }
	if (lane.s(S_NUM_SUNS_FALLEN) == 0 && countdown == 0 && lane.s(S_TICK) < 100) return;
	AddCoin(lane, cSkySunValue);
	const int32_t fallen = lane.s(S_NUM_SUNS_FALLEN) + 1;
	lane.s(S_NUM_SUNS_FALLEN) = fallen;
	const int32_t base = Minimum(cSkySunMin + cSkySunStep * fallen, cSkySunCap);
	countdown = base + rand.Int(cSkySunJitter);
}

// The row picker: the original weights a row down once its mower has been used and keeps the two most
// recent picks to avoid repeating a row.  PROVISIONAL on the exact weights; the shape (uniform over
// playable rows, last-pick penalty) is what matters for the aggregate agreement test.
PVZ_SIM int32_t PickRow(Lane lane, const LevelInfo& level, Rand& rand)
{
	const int32_t rows = Maximum(level.playableRows, 1);
	const int32_t pick = rand.Int(rows);
	for (int32_t row = 0; row < rows; ++row)
	{
		const float value = static_cast<float>(row == pick ? 1.0 : 0.0);
		lane.f(F_ROW_SECOND_LAST_PICKED, row) = lane.f(F_ROW_LAST_PICKED, row);
		lane.f(F_ROW_LAST_PICKED, row) = value;
	}
	return pick;
}

PVZ_SIM void SpawnZombie(Lane lane, const LevelInfo& level, int32_t zombieType, int32_t wave, Rand& rand)
{
	const int32_t slot = FindFreeSlot(lane, F_ZOMBIE_TYPE, kZombieSlots);
	if (slot < 0) { lane.s(S_ERROR) |= ERR_CAPACITY; return; }
	const ZombieDef& def = level.zombies[Minimum(zombieType, kZombieTypes - 1)];
	lane.i(F_ZOMBIE_TYPE, slot) = zombieType;
	lane.i(F_ZOMBIE_ROW, slot) = PickRow(lane, level, rand);
	lane.i(F_ZOMBIE_BODY, slot) = def.bodyHealth;
	lane.i(F_ZOMBIE_HELM, slot) = def.helmHealth;
	lane.i(F_ZOMBIE_SHIELD, slot) = def.shieldHealth;
	lane.i(F_ZOMBIE_FLYING, slot) = 0;
	lane.i(F_ZOMBIE_PHASE, slot) = 0;
	lane.i(F_ZOMBIE_PHASE_COUNTER, slot) = 0;
	lane.i(F_ZOMBIE_FLAGS, slot) = ZB_HAS_HEAD | ZB_HAS_ARM;
	lane.i(F_ZOMBIE_CHILLED, slot) = 0;
	lane.i(F_ZOMBIE_AGE, slot) = 0;
	lane.i(F_ZOMBIE_FROM_WAVE, slot) = wave;
	lane.i(F_ZOMBIE_TARGET, slot) = -1;
	lane.f(F_ZOMBIE_X, slot) = cSpawnX + static_cast<float>(rand.Int(cSpawnJitter));
	lane.f(F_ZOMBIE_Y, slot) = 0.0f;
	// PickRandomSpeed: the def carries the type's range; a fixed-speed type carries the literal instead.
	lane.f(F_ZOMBIE_VELX, slot) = def.velMax > def.velMin
		? def.velMin + (def.velMax - def.velMin) * rand.ZeroToOne() : 0.3f;
	lane.f(F_ZOMBIE_VELY, slot) = 0.0f;
}

// Wave scheduling.  The countdown ticks down every tick; at zero the wave's imported zombie list
// emits one zombie at a time, then the next wave's clock restarts from the imported start value.
PVZ_SIM void UpdateWaves(Lane lane, const LevelInfo& level, const WaveTable& waves, Rand& rand)
{
	int32_t wave = lane.s(S_WAVE);
	if (wave >= waves.waveCount) return;
	int32_t& cursor = lane.s(S_WAVE_SPAWN_CURSOR);
	const int32_t total = waves.zombieCount[wave];
	if (cursor < total)
	{
		if (lane.s(S_TICK) % cWaveSpawnInterval != 0) return;
		SpawnZombie(lane, level, waves.type[wave][cursor], wave, rand);
		cursor = cursor + 1;
		return;
	}
	int32_t& countdown = lane.s(S_ZOMBIE_COUNT_DOWN);
	if (countdown > 0)
	{
		--countdown;
		// 催波: the original shortens the wait once the health of the zombies that must die before the
		// next wave has been burned down.  PROVISIONAL on the arithmetic.
		if (lane.s(S_HEALTH_TO_NEXT_WAVE) <= 0 && countdown > 1) countdown = 1;
		if (countdown > 0) return;
	}
	lane.s(S_WAVE) = wave + 1;
	lane.s(S_WAVE_SPAWN_CURSOR) = 0;
	lane.s(S_ZOMBIE_COUNT_DOWN) = lane.s(S_ZOMBIE_COUNT_DOWN_START);
}

PVZ_SIM void CheckVictory(Lane lane, const WaveTable& waves)
{
	if (lane.s(S_RESULT) != 0) return;
	if (lane.s(S_WAVE) < waves.waveCount) return;
	if (lane.s(S_WAVE_SPAWN_CURSOR) < waves.zombieCount[Minimum(lane.s(S_WAVE), waves.waveCount - 1)]) return;
	for (int32_t slot = 0; slot < kZombieSlots; ++slot)
	{
		if (!ZombieLive(lane.i(F_ZOMBIE_TYPE, slot), lane.i(F_ZOMBIE_FLAGS, slot))) continue;
		return;
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
	for (int32_t slot = 0; slot < kPlantSlots; ++slot) UpdatePlant(lane, level, slot, rand);
	for (int32_t slot = 0; slot < kZombieSlots; ++slot) UpdateZombie(lane, level, slot, rand);
	UpdateProjectiles(lane);
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

// Live zombies standing at or left of x = 160, one bit per slot.  The slot index stands in for the
// original's zombie id: a freed slot cannot be in the set, and a slot reused by a fresh spawn is at
// the right edge (780+), which is outside the zone, so the diff cannot be fooled by the reuse.
// PROVISIONAL on `DYING`: whether a fading corpse still counts for the event is a Zombie::IsLive
// question for cuda/spec/zombies.md to settle; it only shifts when a wait ends early.
PVZ_SIM uint64_t LeftZoneMask(Lane lane)
{
	uint64_t mask = 0;
	for (int32_t slot = 0; slot < kZombieSlots; ++slot)
	{
		if (lane.i(F_ZOMBIE_TYPE, slot) < 0) continue;
		const int32_t flags = lane.i(F_ZOMBIE_FLAGS, slot);
		if (!(flags & ZB_ON_BOARD)) continue;
		if (flags & (ZB_DEAD | ZB_DYING)) continue;
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
	lane.f(F_PLANT_X, slot) = 40.0f + 80.0f * static_cast<float>(col) + cPlantOffsetX;
	lane.f(F_PLANT_Y, slot) = 80.0f * static_cast<float>(row) + cPlantOffsetY;
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
	lane.i(F_PLANTED_BY_TYPE, packet) = lane.i(F_PLANTED_BY_TYPE, packet) + 1;
	lane.s(S_PLANTS_PLANTED) = lane.s(S_PLANTS_PLANTED) + 1;
	return 1;
}

}  // namespace pvzc

#endif
