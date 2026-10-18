// Implementation of the parts of the core that no mechanic depends on: root import, leaf export,
// student feature build.
//
// Included exactly once by cuda/SimHost.cpp (plain C++, one lane, for debugging mechanics against the
// original simulator) and once by cuda/RolloutCuda.cu (nvcc, the batched device build).  Everything
// here is a transcription, not a reinterpretation: the comments name the original's members.
#ifndef PVZ_SIM_IMPL_INC
#define PVZ_SIM_IMPL_INC

#include "SimCore.h"

namespace pvzc
{

namespace {

// Deck lookups.  The exporter puts a plant's cost/refresh into the level block, but a record written
// by an older exporter (or a packet slot the original left blank) can carry zeros, so the tick code
// falls back to the level table rather than dividing by a zero refresh clock.
PVZ_SIM int32_t PlantCost(const LevelInfo& level, int32_t seedType)
{
	for (int32_t index = 0; index < level.plantCount; ++index)
		if (level.plants[index].seedType == seedType) return level.plants[index].cost;
	return 0;
}

PVZ_SIM int32_t PlantRefresh(const LevelInfo& level, int32_t seedType)
{
	for (int32_t index = 0; index < level.plantCount; ++index)
		if (level.plants[index].seedType == seedType) return level.plants[index].refreshTime;
	return 0;
}

PVZ_SIM int32_t PlantHealth(const LevelInfo& level, int32_t seedType)
{
	for (int32_t index = 0; index < level.plantCount; ++index)
		if (level.plants[index].seedType == seedType) return level.plants[index].health;
	return 300;
}

PVZ_SIM const PlantDef* FindPlantDef(const LevelInfo& level, int32_t seedType)
{
	for (int32_t index = 0; index < level.plantCount; ++index)
		if (level.plants[index].seedType == seedType) return &level.plants[index];
	return nullptr;
}

}  // namespace

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

// Empty means "no entity in this slot", tested by the type field being negative.  Clearing every
// field of a slot (rather than only the type) keeps the sweeps branch-free and makes a stale value
// impossible to read back.
PVZ_SIM int32_t LoadRoot(Lane lane, const RootRecord& root, const LevelInfo& level)
{
	if (!root.supported)
	{
		lane.s(S_ERROR) = ERR_BAD_ROOT;
		return 0;
	}
	lane.s(S_ZOMBIE_END) = lane.s(S_PROJECTILE_END) = 0;
	for (int32_t slot = 0; slot < kPlantSlots; ++slot)
	{
		for (int32_t field = F_PLANT_TYPE; field <= F_PLANT_ONBOARD; ++field)
			lane.i(static_cast<IntField>(field), slot) = 0;
		lane.i(F_PLANT_TYPE, slot) = -1;
		lane.f(F_PLANT_X, slot) = 0.0f;
		lane.f(F_PLANT_Y, slot) = 0.0f;
		lane.f(F_PLANT_ANIM_TIME, slot) = lane.f(F_PLANT_ANIM_RATE, slot) = 0.0f;
	}
	for (int32_t slot = 0; slot < kZombieSlots; ++slot)
	{
		for (int32_t field = F_ZOMBIE_TYPE; field <= F_ZOMBIE_TARGET; ++field)
			lane.i(static_cast<IntField>(field), slot) = 0;
		lane.i(F_ZOMBIE_TYPE, slot) = -1;
		lane.f(F_ZOMBIE_X, slot) = 0.0f;
		lane.f(F_ZOMBIE_Y, slot) = 0.0f;
		lane.f(F_ZOMBIE_VELX, slot) = 0.0f;
		lane.f(F_ZOMBIE_VELY, slot) = 0.0f;
		lane.f(F_ZOMBIE_ANIM_TIME, slot) = lane.f(F_ZOMBIE_ANIM_RATE, slot) = 0.0f;
	}
	for (int32_t slot = 0; slot < kProjectileSlots; ++slot)
	{
		lane.i(F_PROJ_TYPE, slot) = -1;
		lane.i(F_PROJ_ROW, slot) = 0;
		lane.i(F_PROJ_AGE, slot) = 0;
		lane.i(F_PROJ_DAMAGE, slot) = 0;
		lane.i(F_PROJ_CHILL, slot) = 0;
		lane.f(F_PROJ_X, slot) = 0.0f;
		lane.f(F_PROJ_Y, slot) = 0.0f;
		lane.f(F_PROJ_VELX, slot) = 0.0f;
		lane.f(F_PROJ_VELY, slot) = 0.0f;
	}
	for (int32_t slot = 0; slot < kCoinSlots; ++slot)
	{
		lane.i(F_COIN_ACTIVE, slot) = -1;
		lane.i(F_COIN_VALUE, slot) = 0;
		lane.i(F_COIN_TICKS, slot) = 0;
	}
	for (int32_t slot = 0; slot < kMowerSlots; ++slot)
	{
		lane.i(F_MOWER_STATE, slot) = kMowerSquished;
		lane.i(F_MOWER_FLAGS, slot) = MB_GONE;
		lane.i(F_MOWER_TIMER, slot) = 0;
		lane.i(F_MOWER_CHOMP, slot) = 0;
		lane.f(F_MOWER_X, slot) = 0.0f;
	}
	for (int32_t slot = 0; slot < kPacketSlots; ++slot)
	{
		lane.i(F_PACKET_TYPE, slot) = -1;
		lane.i(F_PACKET_ACTIVE, slot) = 0;
		lane.i(F_PACKET_COUNTER, slot) = 0;
		lane.i(F_PACKET_REFRESH, slot) = 0;
		lane.i(F_PACKET_COST, slot) = 0;
		lane.i(F_PLANTED_BY_TYPE, slot) = 0;
	}

	lane.s(S_TICK) = root.tick;
	lane.s(S_SUN) = root.sun;
	lane.s(S_WAVE) = root.wave;
	lane.s(S_NUM_WAVES) = root.numWaves;
	lane.s(S_ZOMBIE_COUNT_DOWN) = root.zombieCountDown;
	lane.s(S_ZOMBIE_COUNT_DOWN_START) = root.zombieCountDownStart;
	lane.s(S_HEALTH_WAVE_START) = root.zombieHealthWaveStart;
	lane.s(S_HEALTH_TO_NEXT_WAVE) = root.zombieHealthToNextWave;
	lane.s(S_HUGE_WAVE_COUNT_DOWN) = root.hugeWaveCountDown;
	lane.s(S_SUN_COUNT_DOWN) = root.sunCountDown;
	lane.s(S_NUM_SUNS_FALLEN) = root.numSunsFallen;
	lane.s(S_SUN_PRODUCED) = root.sunMoneyProduced;
	lane.s(S_ZOMBIES_KILLED) = root.zombiesKilled;
	lane.s(S_PLANTS_EATEN) = root.plantsEaten;
	lane.s(S_PLANTS_PLANTED) = 0;
	lane.s(S_TRIGGERED_MOWERS) = root.triggeredLawnMowers;
	lane.s(S_RESULT) = root.result;
	lane.s(S_DECISIONS) = 0;
	lane.s(S_PRIOR_TICK) = -1;
	lane.s(S_PRIOR_PLANT) = 0;
	lane.s(S_INCOME_WINDOW_START) = root.incomeWindowStart;
	lane.s(S_INCOME_COUNT) = root.incomeCount < 0 ? 0 : Minimum(root.incomeCount, kIncomeEvents);
	lane.s(S_INCOME_HEAD) = 0;
	lane.s(S_WAVE_SPAWN_CURSOR) = 0;
	lane.s(S_TICKS_SINCE_EVENT) = 0;
	lane.s(S_ERROR) = 0;
	lane.s(S_ROOT_SUN_PRODUCED) = root.sunMoneyProduced;
	lane.s(S_ROOT_ZOMBIES_KILLED) = root.zombiesKilled;
	lane.s(S_ROOT_PLANTS_EATEN) = root.plantsEaten;
	lane.s(S_ROOT_TRIGGERED_MOWERS) = root.triggeredLawnMowers;
	lane.s(S_ROOT_TICK) = root.tick;
	lane.s(S_ROOT_INDEX) = 0;
	for (int32_t row = 0; row < kRows; ++row)
	{
		lane.f(F_ROW_LAST_PICKED, row) = root.rowLastPicked[row];
		lane.f(F_ROW_SECOND_LAST_PICKED, row) = root.rowSecondLastPicked[row];
		lane.i(F_ROW_GOT_MOWERED, row) = root.rowGotMowered[row];
	}
	for (int32_t index = 0; index < lane.s(S_INCOME_COUNT); ++index)
	{
		lane.i(F_INCOME_TICK, index) = root.incomeTick[index];
		lane.i(F_INCOME_VALUE, index) = root.incomeValue[index];
	}

	// Plants live in the slot of their own cell (row * 10 + col), which is how the original's feature
	// grid addresses them; a second plant in one cell cannot exist on grass.
	for (int32_t index = 0; index < root.plantCount; ++index)
	{
		const RootPlant& src = root.plants[index];
		if (src.type < 0 || src.cell < 0 || src.cell >= kPlantSlots) continue;
		const int32_t slot = src.cell;
		lane.i(F_PLANT_TYPE, slot) = src.type;
		lane.i(F_PLANT_HP, slot) = src.health;
		lane.i(F_PLANT_HP_MAX, slot) = src.maxHealth ? src.maxHealth : PlantHealth(level, src.type);
		lane.i(F_PLANT_STATE, slot) = src.state;
		lane.i(F_PLANT_STATE_COUNTDOWN, slot) = src.stateCountdown;
		lane.i(F_PLANT_LAUNCH, slot) = src.launchCounter;
		lane.i(F_PLANT_LAUNCH_RATE, slot) = src.launchRate;
		lane.i(F_PLANT_SPECIAL, slot) = src.special;
		lane.i(F_PLANT_SHOOTING, slot) = src.extraA;
		lane.i(F_PLANT_ANIM, slot) = src.extraB;
		lane.i(F_PLANT_ONBOARD, slot) = src.onBoard;
		lane.f(F_PLANT_X, slot) = src.posX;
		lane.f(F_PLANT_Y, slot) = src.posY;
		lane.f(F_PLANT_ANIM_TIME, slot) = src.animTime;
		lane.f(F_PLANT_ANIM_RATE, slot) = src.animRate;
		lane.i(F_PLANT_ANIM_FRAMES, slot) = src.animFrames;
		lane.i(F_PLANT_ANIM_LOOPS, slot) = src.animLoops;
	}
	for (int32_t index = 0; index < root.zombieCount; ++index)
	{
		const RootZombie& src = root.zombies[index];
		if (src.type < 0) continue;
		const int32_t slot = FindFreeSlot(lane, F_ZOMBIE_TYPE, kZombieSlots);
		if (slot < 0) { lane.s(S_ERROR) |= ERR_CAPACITY; continue; }
		lane.i(F_ZOMBIE_TYPE, slot) = src.type;
		lane.i(F_ZOMBIE_ROW, slot) = src.row;
		lane.i(F_ZOMBIE_BODY, slot) = src.body;
		// The head/arm loss thresholds are fractions of the *starting* health (Zombie::UpdateDamageStates,
		// Zombie.cpp:3904-3930), so the import has to carry it; an older record leaves it zero.
		lane.i(F_ZOMBIE_BODY_MAX, slot) = src.bodyMax ? src.bodyMax : (src.body > 0 ? src.body : 1);
		lane.i(F_ZOMBIE_HELM, slot) = src.helm;
		lane.i(F_ZOMBIE_SHIELD, slot) = src.shield;
		lane.i(F_ZOMBIE_FLYING, slot) = src.flying;
		lane.i(F_ZOMBIE_PHASE, slot) = src.phase;
		lane.i(F_ZOMBIE_PHASE_COUNTER, slot) = src.phaseCounter;
		lane.i(F_ZOMBIE_FLAGS, slot) = src.stateBits;
		lane.i(F_ZOMBIE_CHILLED, slot) = src.chilled;
		lane.i(F_ZOMBIE_AGE, slot) = src.age;
		lane.i(F_ZOMBIE_FROM_WAVE, slot) = src.fromWave;
		// The engine has no target latch on the chew path (FindPlantTarget re-runs every 4 ticks,
		// Zombie.cpp:6879) and the exporter cannot read mTargetPlantID, so the imported cell is always -1
		// (cuda/spec/host.md, corpus finding 3).  The tick re-derives it from the mouth rectangle.
		lane.i(F_ZOMBIE_TARGET, slot) = -1;
		lane.f(F_ZOMBIE_X, slot) = src.posX;
		lane.f(F_ZOMBIE_Y, slot) = src.posY;
		lane.f(F_ZOMBIE_VELX, slot) = src.velX;
		lane.f(F_ZOMBIE_ANIM_TIME, slot) = src.animTime;
		lane.f(F_ZOMBIE_ANIM_RATE, slot) = src.animRate;
		lane.i(F_ZOMBIE_ANIM_START, slot) = src.animStart;
		lane.i(F_ZOMBIE_ANIM_FRAMES, slot) = src.animFrames;
		lane.i(F_ZOMBIE_ANIM_LOOP_TYPE, slot) = src.animLoopType;
		lane.i(F_ZOMBIE_ANIM_LOOPS, slot) = src.animLoops;
		lane.i(F_ZOMBIE_FADE, slot) = src.fade;
		// A zombie imported mid-vault carries no vertical velocity; the arc is recomputed from the
		// phase countdown, so this only has to be a finite number.
		lane.f(F_ZOMBIE_VELY, slot) = 0.0f;
	}
	for (int32_t index = 0; index < root.projectileCount; ++index)
	{
		const RootProjectile& src = root.projectiles[index];
		if (src.type < 0) continue;
		const int32_t slot = FindFreeSlot(lane, F_PROJ_TYPE, kProjectileSlots);
		if (slot < 0) { lane.s(S_ERROR) |= ERR_CAPACITY; continue; }
		lane.i(F_PROJ_TYPE, slot) = src.type;
		lane.i(F_PROJ_ROW, slot) = src.row;
		lane.i(F_PROJ_AGE, slot) = src.age;
		lane.i(F_PROJ_DAMAGE, slot) = src.damage;
		lane.i(F_PROJ_CHILL, slot) = src.chill;
		lane.f(F_PROJ_X, slot) = src.posX;
		lane.f(F_PROJ_Y, slot) = src.posY;
		// RootProjectile::velX is structurally 0 (the exporter does not read the field: straight peas are
		// advanced by the literal `mPosX += mVelX` where mVelX came from the projectile definition,
		// cuda/spec/projectiles.md).  3.33 is that definition value and must stay in step with
		// SimTick.inc's cPeaSpeed.
		lane.f(F_PROJ_VELX, slot) = src.velX ? src.velX : 3.33f;
		lane.f(F_PROJ_VELY, slot) = src.velY;
	}
	// Sun coins carry no position: LawnApp::EnvironmentWait banks every live sun at the top of the
	// next tick, so the exporter's ticksToCollect is the whole state that matters.
	for (int32_t index = 0; index < root.coinCount; ++index)
	{
		const RootCoin& src = root.coins[index];
		const int32_t slot = FindFreeSlot(lane, F_COIN_ACTIVE, kCoinSlots);
		if (slot < 0) { lane.s(S_ERROR) |= ERR_CAPACITY; continue; }
		lane.i(F_COIN_ACTIVE, slot) = 1;
		lane.i(F_COIN_VALUE, slot) = src.value;
		lane.i(F_COIN_TICKS, slot) = src.ticksToCollect > 0 ? src.ticksToCollect : 1;
	}
	for (int32_t index = 0; index < root.mowerCount; ++index)
	{
		const RootMower& src = root.mowers[index];
		const int32_t slot = src.row >= 0 && src.row < kMowerSlots ? src.row : index;
		if (slot >= kMowerSlots) continue;
		lane.i(F_MOWER_STATE, slot) = src.state;
		// Exporter stores mRollingInCounter (elapsed), while the tick uses ticks remaining.
		lane.i(F_MOWER_TIMER, slot) = src.state == kMowerRollingIn ? Maximum(100 - src.timer, 0) : src.timer;
		lane.i(F_MOWER_FLAGS, slot) = src.flags;
		lane.i(F_MOWER_CHOMP, slot) = src.chompCounter;
		lane.f(F_MOWER_X, slot) = src.posX;
	}
	for (int32_t index = 0; index < root.packetCount && index < kPacketSlots; ++index)
	{
		const RootPacket& src = root.packets[index];
		lane.i(F_PACKET_TYPE, index) = src.type;
		lane.i(F_PACKET_ACTIVE, index) = src.active;
		lane.i(F_PACKET_COUNTER, index) = src.refreshCounter;
		// Import the engine's mRefreshTime verbatim.  Filling a 0 in here invented a cooldown for cards the
		// original had dropped (SeedPacket.cpp:144) and made the cooldown scalar disagree with the
		// engine-derived encoder, which reads the raw value (python/pvz_value_net.py:112).  The number a
		// fresh packet really needs is written by ActionPlant, the way SeedPacket::WasPlanted does it.
		lane.i(F_PACKET_REFRESH, index) = src.refreshTime;
		lane.i(F_PACKET_COST, index) = src.cost ? src.cost : PlantCost(level, src.type);
	}
	(void)FindPlantDef;
	return 1;
}

// ---------------------------------------------------------------------------
// Leaf export
// ---------------------------------------------------------------------------

PVZ_SIM void StoreResult(Lane lane, RolloutResult& out)
{
	out.rootIndex = lane.s(S_ROOT_INDEX);
	out.decisions = lane.s(S_DECISIONS);
	out.netCalls = lane.s(S_NET_CALLS);
	out.ticks = lane.s(S_TICK) - lane.s(S_ROOT_TICK);
	out.endTick = lane.s(S_TICK);
	out.result = lane.s(S_RESULT);
	out.sun = lane.s(S_SUN);
	out.wave = lane.s(S_WAVE);
	out.sunProduced = lane.s(S_SUN_PRODUCED) - lane.s(S_ROOT_SUN_PRODUCED);
	out.zombiesKilled = lane.s(S_ZOMBIES_KILLED) - lane.s(S_ROOT_ZOMBIES_KILLED);
	out.plantsEaten = lane.s(S_PLANTS_EATEN) - lane.s(S_ROOT_PLANTS_EATEN);
	out.plantsPlanted = lane.s(S_PLANTS_PLANTED);
	out.mowersTriggered = lane.s(S_TRIGGERED_MOWERS) - lane.s(S_ROOT_TRIGGERED_MOWERS);
	out.errorFlags = lane.s(S_ERROR);
	for (int32_t index = 0; index < kPacketSlots; ++index)
		out.plantedByType[index] = lane.i(F_PLANTED_BY_TYPE, index);

	int32_t zombiesAlive = 0, plantsAlive = 0, mowersReady = 0;
	for (int32_t row = 0; row < kRows; ++row) out.rowMinZombieX[row] = INT32_MIN;
	for (int32_t slot = 0; slot < lane.s(S_ZOMBIE_END); ++slot)
	{
		if (lane.i(F_ZOMBIE_TYPE, slot) < 0) continue;
		const int32_t flags = lane.i(F_ZOMBIE_FLAGS, slot);
		// EnvironmentLeanObservation lists a zombie iff `!mDead` (LawnApp.cpp:1655), so a fading corpse
		// (DYING/MOWERED/BURNED, all still on the list until the slot frees) counts as present. Mowed and
		// burned entities can retain positive health. Filtering on DYING here would undercount
		// every board the moment a zombie is shot.
		if (flags & ZB_DEAD) continue;
		++zombiesAlive;
		if (!(flags & ZB_ON_BOARD)) continue;
		const int32_t row = lane.i(F_ZOMBIE_ROW, slot);
		if (row < 0 || row >= kRows) continue;
		const int32_t scaled = static_cast<int32_t>(lane.f(F_ZOMBIE_X, slot) * 4.0f);
		if (out.rowMinZombieX[row] == INT32_MIN || scaled < out.rowMinZombieX[row]) out.rowMinZombieX[row] = scaled;
	}
	for (int32_t slot = 0; slot < kPlantSlots; ++slot)
		if (lane.i(F_PLANT_TYPE, slot) >= 0) ++plantsAlive;
	for (int32_t slot = 0; slot < kMowerSlots; ++slot)
		if (lane.i(F_MOWER_STATE, slot) == kMowerReady) ++mowersReady;
	out.zombiesAlive = zombiesAlive;
	out.plantsAlive = plantsAlive;
	// The ROLLOUT response's "mowers_left" is the count of READY mowers, not the trigger delta.
	out.mowersLeft = mowersReady;
}

// ---------------------------------------------------------------------------
// Leaf features for the value net
// ---------------------------------------------------------------------------

// board[CHANNEL][ROW][COL] in numpy's C order: the exported block is reshaped to (92, 6, 10) by the
// reader, so this is the layout python/pvz_value_net.py writes.
PVZ_SIM int32_t ValueBoardIndex(int32_t channel, int32_t row, int32_t col)
{
	return (channel * kRows + row) * kCols + col;
}

// python/pvz_value_net.py::featurize_numpy + observation_context, element for element.  It is *not*
// BuildFeatures: the value net was trained on its own encoder, and the two differ in three ways that
// each change a score.
//   * the zombie column is floor((x - 40) / 80) of the coordinate as the observation printed it
//     (six significant digits, LawnApp.cpp:1658) with no snap to a cell boundary -- that snap is part
//     of EnvironmentRollout::Features (cuda/SimImpl.inc, BuildFeatures) and not of this encoder;
//   * there is no prior-action pair, so the scalar vector is 113 long instead of 115;
//   * the task context is a separate 35-vector the caller concatenates, not part of the scalars.
// Runs once per lane at the leaf, so its cost is not on the tick path.
PVZ_SIM void StoreLeafValueFeatures(Lane lane, const LevelInfo& level, const float* waveLog, float* out)
{
	for (int32_t index = 0; index < kBoardChannels * kCells; ++index) out[index] = 0.0f;

	// Plants: health ratio in the channel of its seed type, columns 0..8 only (pvz_value_net.py:83).
	// The observation already dropped dead and squished plants (LawnApp.cpp:1644, :1648), which here is
	// the slot-emptiness test; a plant leaves the slot when it dies.
	for (int32_t slot = 0; slot < kPlantSlots; ++slot)
	{
		const int32_t type = lane.i(F_PLANT_TYPE, slot);
		if (type < 0) continue;
		const int32_t row = slot / kCols;
		const int32_t col = slot - row * kCols;
		if (col >= kCols - 1) continue;
		const int32_t maxHealth = Maximum(lane.i(F_PLANT_HP_MAX, slot), 1);
		out[ValueBoardIndex(Minimum(type, kPlantTypes - 1), row, col)] +=
			static_cast<float>(static_cast<double>(lane.i(F_PLANT_HP, slot)) / static_cast<double>(maxHealth));
	}

	// Zombies: health sum / 200, eating, chilled -- channels 54..89, the same four the student encodes,
	// filtered the same way (the lean observation lists a zombie iff !mDead, LawnApp.cpp:1655).
	for (int32_t slot = 0; slot < lane.s(S_ZOMBIE_END); ++slot)
	{
		const int32_t type = lane.i(F_ZOMBIE_TYPE, slot);
		if (type < 0) continue;
		const int32_t flags = lane.i(F_ZOMBIE_FLAGS, slot);
		const int32_t row = lane.i(F_ZOMBIE_ROW, slot);
		if ((flags & ZB_DEAD) || !(flags & ZB_ON_BOARD) || row < 0 || row >= kRows) continue;
		const int32_t col =
			FeatureColumn(RenderedPublicCoordinate(static_cast<double>(lane.f(F_ZOMBIE_X, slot))));
		const double health = static_cast<double>(lane.i(F_ZOMBIE_BODY, slot))
			+ static_cast<double>(lane.i(F_ZOMBIE_HELM, slot)) + static_cast<double>(lane.i(F_ZOMBIE_SHIELD, slot));
		const int32_t channel = kPlantTypes + Minimum(type, kZombieTypes - 1);
		out[ValueBoardIndex(channel, row, col)] += static_cast<float>(health / 200.0);
		out[ValueBoardIndex(kPlantTypes + kZombieTypes, row, col)] += (flags & ZB_EATING) ? 1.0f : 0.0f;
		out[ValueBoardIndex(kPlantTypes + kZombieTypes + 1, row, col)] +=
			lane.i(F_ZOMBIE_CHILLED, slot) > 0 ? 1.0f : 0.0f;
	}

	// Mowers: a READY mower (LawnMowerState 1) marks column 0 of its row; anything else leaves 0.
	for (int32_t row = 0; row < kRows && row < kMowerSlots; ++row)
		out[ValueBoardIndex(90, row, 0)] = lane.i(F_MOWER_STATE, row) == kMowerReady ? 1.0f : 0.0f;

	// Channel 91: whole playable rows.  The Python derives the row list from the terrain grid (a row
	// counts when its 9 lawn cells are all grass or all pool); the kernel takes it from the level, which
	// is the same set on the day-grass tasks this corpus comes from.
	for (int32_t row = 0; row < level.playableRows; ++row)
		for (int32_t col = 0; col < kCols; ++col) out[ValueBoardIndex(91, row, col)] = 1.0f;

	float* scalars = out + kBoardChannels * kCells;
	for (int32_t index = 0; index < kValueScalars; ++index) scalars[index] = 0.0f;
	const int32_t sun = lane.s(S_SUN);
	const int32_t waves = Maximum(lane.s(S_NUM_WAVES), 1);
	scalars[0] = static_cast<float>(static_cast<double>(Minimum(sun, 2000)) / 1000.0);
	scalars[1] = static_cast<float>(static_cast<double>(lane.s(S_WAVE)) / static_cast<double>(waves));
	scalars[2] = static_cast<float>(static_cast<double>(lane.s(S_TICK)) / 60000.0);
	scalars[3] = static_cast<float>(IncomeRate(lane) / 100.0);
	scalars[4] = waveLog[Minimum(waves, kMaxWaves)];
	for (int32_t index = 0; index < kPacketSlots; ++index)
	{
		const int32_t type = lane.i(F_PACKET_TYPE, index);
		if (type < 0) continue;
		const int32_t kind = Minimum(type, kPlantTypes - 1);
		const int32_t active = lane.i(F_PACKET_ACTIVE, index);
		const int32_t refresh = lane.i(F_PACKET_REFRESH, index);
		const int32_t counter = lane.i(F_PACKET_COUNTER, index);
		scalars[5 + kind] = (active && sun >= lane.i(F_PACKET_COST, index)) ? 1.0f : 0.0f;
		scalars[5 + kPlantTypes + kind] = static_cast<float>(MinimumDouble(1.0,
			active || !refresh ? 1.0 : static_cast<double>(counter) / static_cast<double>(refresh)));
	}

	// observation_context: the wave-size multiplier and the zombie roster, constant for the task, so the
	// reader can take it from the level instead of the frame.
	float* context = scalars + kValueScalars;
	context[0] = static_cast<float>(static_cast<double>(level.zombieCountMultiplier) / 3.0);
	for (int32_t type = 0; type < kZombieTypes; ++type)
		context[1 + type] = level.allowed[type] ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------------

// Padded plane element, and per-lane vector element, of this lane's activation slot -- the same
// formulas the batched GEMMs address by, and the same ones NetForward.inc's PVZ_PLANE/PVZ_VECTOR use.
PVZ_SIM float& CellChannel(Lane lane, float* board, int32_t channel, int32_t cell)
{
	return board[PlaneIndex(kBoardChannels, lane.netSlot, channel, PaddedCell(cell))];
}

PVZ_SIM float& ScalarChannel(Lane lane, float* scalars, int32_t index)
{
	return scalars[VectorIndex(kScalarFeatures, lane.netSlot, index)];
}

PVZ_SIM void BuildFeatures(Lane lane, const LevelInfo& level, const NetState& buffers, const float* waveLog,
	bool previousValid)
{
	for (int32_t channel = 0; channel < kBoardChannels; ++channel)
		for (int32_t cell = 0; cell < kCells; ++cell)
			CellChannel(lane, buffers.board, channel, cell) = 0.0f;
	// The scalar vector is cleared too, not only filled: the original does
	// `scalars.assign(kScalarFeatures, 0.0f)` before writing the deck-dependent entries, and an
	// activation slot is reused by whichever lane reaches a decision next, so an uncleared tail would
	// make a lane's decision depend on which lane ran last (cuda/spec/host.md finding 3).
	for (int32_t index = 0; index < kScalarFeatures; ++index)
		ScalarChannel(lane, buffers.scalars, index) = 0.0f;

	// Plants: health ratio in the channel of its seed type (EnvironmentRollout.cpp:112).  The original
	// skips dead and squished plants; a squished plant leaves the slot immediately here, so the type
	// test is the whole filter.
	for (int32_t slot = 0; slot < kPlantSlots; ++slot)
	{
		const int32_t type = lane.i(F_PLANT_TYPE, slot);
		if (type < 0) continue;
		const int32_t col = slot % kCols;
		if (col >= kGridCols) continue;
		const int32_t channel = Minimum(type, kPlantTypes - 1);
		const int32_t maxHealth = Maximum(lane.i(F_PLANT_HP_MAX, slot), 1);
		CellChannel(lane, buffers.board, channel, slot) +=
			static_cast<float>(static_cast<double>(lane.i(F_PLANT_HP, slot)) / static_cast<double>(maxHealth));
	}

	// Zombies: health sum / 200, plus the eating and chilled channels (EnvironmentRollout.cpp:116).
	for (int32_t slot = 0; slot < lane.s(S_ZOMBIE_END); ++slot)
	{
		const int32_t type = lane.i(F_ZOMBIE_TYPE, slot);
		if (type < 0) continue;
		const int32_t flags = lane.i(F_ZOMBIE_FLAGS, slot);
		const int32_t row = lane.i(F_ZOMBIE_ROW, slot);
		if ((flags & ZB_DEAD) || !(flags & ZB_ON_BOARD) || row < 0 || row >= kRows) continue;
		double position = static_cast<double>(lane.f(F_ZOMBIE_X, slot));
		const double boundary = 40.0 + 80.0 * RoundHalfAway((position - 40.0) / 80.0);
		if (AbsDouble(position - boundary) < 0.001) position = RenderedPublicCoordinate(position);
		const int32_t cell = row * kCols + FeatureColumn(position);
		const double health = static_cast<double>(lane.i(F_ZOMBIE_BODY, slot))
			+ static_cast<double>(lane.i(F_ZOMBIE_HELM, slot)) + static_cast<double>(lane.i(F_ZOMBIE_SHIELD, slot));
		CellChannel(lane, buffers.board, kPlantTypes + Minimum(type, kZombieTypes - 1), cell) +=
			static_cast<float>(health / 200.0);
		CellChannel(lane, buffers.board, 88, cell) += (flags & ZB_EATING) ? 1.0f : 0.0f;
		CellChannel(lane, buffers.board, 89, cell) += lane.i(F_ZOMBIE_CHILLED, slot) > 0 ? 1.0f : 0.0f;
	}

	// Mowers: channel 90, ready mowers only, always column 0 (EnvironmentRollout.cpp:129).
	for (int32_t row = 0; row < kRows && row < kMowerSlots; ++row)
		CellChannel(lane, buffers.board, 90, row * kCols) =
			lane.i(F_MOWER_STATE, row) == kMowerReady ? 1.0f : 0.0f;

	// Channel 91: a row whose 9 lawn cells are all the same playable terrain.  On day grass that is
	// every playable row, decided by the level rather than by the position.
	for (int32_t row = 0; row < level.playableRows; ++row)
		for (int32_t col = 0; col < kCols; ++col)
			CellChannel(lane, buffers.board, 91, row * kCols + col) = 1.0f;

	float* scalars = buffers.scalars;
	const int32_t sun = lane.s(S_SUN);
	const int32_t waves = Maximum(lane.s(S_NUM_WAVES), 1);
	// EnvironmentRollout.cpp:141 divides by the double literal 1000.0, so this is a real quotient and not
	// the int truncation the mSunMoney type might suggest.
	ScalarChannel(lane, scalars, 0) = static_cast<float>(Minimum(sun, 2000) / 1000.0);
	ScalarChannel(lane, scalars, 1) = static_cast<float>(static_cast<double>(lane.s(S_WAVE)) / static_cast<double>(waves));
	ScalarChannel(lane, scalars, 2) = static_cast<float>(static_cast<double>(lane.s(S_TICK)) / 60000.0);
	ScalarChannel(lane, scalars, 3) = static_cast<float>(IncomeRate(lane) / 100.0);
	// log1p(waves)/log1p(50) is computed by glibc on the host and uploaded, so the device reads the
	// same bits the CPU feature build produced instead of recomputing a libm difference.
	ScalarChannel(lane, scalars, 4) = waveLog[Minimum(waves, kMaxWaves)];
	for (int32_t index = 0; index < kPacketSlots; ++index)
	{
		const int32_t type = lane.i(F_PACKET_TYPE, index);
		if (type < 0) continue;
		const int32_t kind = Minimum(type, kPlantTypes - 1);
		const int32_t active = lane.i(F_PACKET_ACTIVE, index);
		const int32_t refresh = lane.i(F_PACKET_REFRESH, index);
		const int32_t counter = lane.i(F_PACKET_COUNTER, index);
		ScalarChannel(lane, scalars, 5 + kind) = (active && sun >= lane.i(F_PACKET_COST, index)) ? 1.0f : 0.0f;
		ScalarChannel(lane, scalars, 59 + kind) = static_cast<float>(
			MinimumDouble(1.0, active || !refresh ? 1.0 :
				static_cast<double>(counter) / static_cast<double>(refresh)));
	}
	if (previousValid)
	{
		ScalarChannel(lane, scalars, 113) = lane.s(S_PRIOR_PLANT) ? 1.0f : 0.0f;
		ScalarChannel(lane, scalars, 114) =
			static_cast<float>(static_cast<double>(lane.s(S_TICK) - lane.s(S_PRIOR_TICK)) / 300.0);
	}
	else
	{
		ScalarChannel(lane, scalars, 113) = 0.0f;
		ScalarChannel(lane, scalars, 114) = 0.0f;
	}
	// Task context (EnvironmentRollout.cpp:161): the multiplier and the roster, constant per task, so
	// a batch from one task does not recompute them.
	ScalarChannel(lane, scalars, kScalars) = static_cast<float>(static_cast<double>(level.zombieCountMultiplier) / 3.0);
	for (int32_t type = 0; type < kZombieTypes; ++type)
		ScalarChannel(lane, scalars, kScalars + 1 + type) = level.allowed[type] ? 1.0f : 0.0f;
}

}  // namespace pvzc

#endif
