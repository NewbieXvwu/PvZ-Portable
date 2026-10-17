// Implementation of the spec-independent core: root import, leaf export, feature build.
//
// Included exactly once by cuda/SimHost.cpp (plain C++, one lane, for debugging mechanics against the
// original simulator) and once by cuda/RolloutCuda.cu (nvcc, the batched device build).  Everything
// here is a transcription, not a reinterpretation: the field names in comments are the original's.
#ifndef PVZ_SIM_IMPL_INC
#define PVZ_SIM_IMPL_INC

#include "SimCore.h"

namespace pvzc
{

namespace {

// Slot field access helpers: the empty marker is a negative entity type, so slot 0 is a real entity
// exactly when its type is >= 0.  Counts are recomputed on import and maintained by the tick logic.
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

PVZ_SIM const ZombieDef* FindZombieDef(const LevelInfo& level, int32_t type)
{
	if (type >= 0 && type < kZombieTypes && level.zombies[type].type == type) return &level.zombies[type];
	return nullptr;
}

}  // namespace

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

PVZ_SIM int32_t LoadRoot(Lane lane, const RootRecord& root, const LevelInfo& level)
{
	if (!root.supported)
	{
		lane.s(S_ERROR) = ERR_BAD_ROOT;
		return 0;
	}
	for (int32_t slot = 0; slot < kPlantSlots; ++slot)
	{
		lane.i(F_PLANT_TYPE, slot) = -1;
		lane.i(F_PLANT_HP, slot) = 0;
		lane.i(F_PLANT_STATE, slot) = 0;
		lane.i(F_PLANT_STATE_COUNTDOWN, slot) = 0;
		lane.i(F_PLANT_LAUNCH, slot) = 0;
		lane.i(F_PLANT_LAUNCH_RATE, slot) = 0;
		lane.i(F_PLANT_SPECIAL, slot) = 0;
		lane.i(F_PLANT_EXTRA_A, slot) = 0;
		lane.i(F_PLANT_EXTRA_B, slot) = 0;
	}
	for (int32_t slot = 0; slot < kZombieSlots; ++slot)
	{
		lane.i(F_ZOMBIE_TYPE, slot) = -1;
		for (int32_t field = F_ZOMBIE_TYPE; field <= F_ZOMBIE_TARGET; ++field)
			lane.i(static_cast<IntField>(field), slot) = -1;
		lane.f(F_ZOMBIE_X, slot) = 0.0f;
		lane.f(F_ZOMBIE_Y, slot) = 0.0f;
		lane.f(F_ZOMBIE_VELX, slot) = 0.0f;
		lane.f(F_ZOMBIE_VELY, slot) = 0.0f;
	}
	for (int32_t slot = 0; slot < kProjectileSlots; ++slot)
	{
		lane.i(F_PROJ_TYPE, slot) = -1;
		lane.i(F_PROJ_ROW, slot) = -1;
		lane.i(F_PROJ_AGE, slot) = 0;
		lane.i(F_PROJ_DAMAGE, slot) = 0;
		lane.f(F_PROJ_X, slot) = 0.0f;
		lane.f(F_PROJ_Y, slot) = 0.0f;
		lane.f(F_PROJ_VELX, slot) = 0.0f;
		lane.f(F_PROJ_VELY, slot) = 0.0f;
	}
	for (int32_t slot = 0; slot < kCoinSlots; ++slot)
	{
		lane.i(F_COIN_TYPE, slot) = -1;
		lane.i(F_COIN_VALUE, slot) = 0;
		lane.i(F_COIN_TICKS, slot) = 0;
		lane.f(F_COIN_X, slot) = 0.0f;
		lane.f(F_COIN_Y, slot) = 0.0f;
		lane.f(F_COIN_VELX, slot) = 0.0f;
		lane.f(F_COIN_VELY, slot) = 0.0f;
	}
	for (int32_t slot = 0; slot < kMowerSlots; ++slot)
	{
		lane.i(F_MOWER_STATE, slot) = MOWER_GONE;
		lane.i(F_MOWER_TIMER, slot) = 0;
		lane.i(F_MOWER_FLAGS, slot) = 0;
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
	lane.s(S_ERROR) = 0;
	lane.s(S_ROOT_SUN_PRODUCED) = root.sunMoneyProduced;
	lane.s(S_ROOT_ZOMBIES_KILLED) = root.zombiesKilled;
	lane.s(S_ROOT_PLANTS_EATEN) = root.plantsEaten;
	lane.s(S_ROOT_TRIGGERED_MOWERS) = root.triggeredLawnMowers;
	lane.s(S_ROOT_TICK) = root.tick;
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

	for (int32_t index = 0; index < root.plantCount; ++index)
	{
		const RootPlant& src = root.plants[index];
		if (src.type < 0) continue;
		const int32_t slot = src.cell < 0 || src.cell >= kPlantSlots ? kPlantSlots - 1 : src.cell;
		lane.i(F_PLANT_TYPE, slot) = src.type;
		lane.i(F_PLANT_HP, slot) = src.health;
		lane.i(F_PLANT_HP_MAX, slot) = src.maxHealth;
		lane.i(F_PLANT_STATE, slot) = src.state;
		lane.i(F_PLANT_STATE_COUNTDOWN, slot) = src.stateCountdown;
		lane.i(F_PLANT_LAUNCH, slot) = src.launchCounter;
		lane.i(F_PLANT_LAUNCH_RATE, slot) = src.launchRate;
		lane.i(F_PLANT_SPECIAL, slot) = src.special;
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
		lane.i(F_ZOMBIE_HELM, slot) = src.helm;
		lane.i(F_ZOMBIE_SHIELD, slot) = src.shield;
		lane.i(F_ZOMBIE_PHASE, slot) = src.phase;
		lane.i(F_ZOMBIE_PHASE_COUNTER, slot) = src.phaseCounter;
		lane.i(F_ZOMBIE_FLAGS, slot) = src.stateBits;
		lane.i(F_ZOMBIE_CHILLED, slot) = src.chilled;
		lane.i(F_ZOMBIE_AGE, slot) = src.age;
		lane.i(F_ZOMBIE_FROM_WAVE, slot) = src.fromWave;
		lane.i(F_ZOMBIE_TARGET, slot) = src.targetCell;
		lane.f(F_ZOMBIE_X, slot) = src.posX;
		lane.f(F_ZOMBIE_Y, slot) = src.posY;
		lane.f(F_ZOMBIE_VELX, slot) = src.velX;
		// A zombie imported mid-vault has no vertical velocity of its own; the arc is recomputed from
		// the phase countdown, so the imported value only has to be finite.
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
		lane.f(F_PROJ_X, slot) = src.posX;
		lane.f(F_PROJ_Y, slot) = src.posY;
		lane.f(F_PROJ_VELX, slot) = src.velX;
		lane.f(F_PROJ_VELY, slot) = src.velY;
	}
	for (int32_t index = 0; index < root.coinCount; ++index)
	{
		const RootCoin& src = root.coins[index];
		const int32_t slot = FindFreeSlot(lane, F_COIN_TYPE, kCoinSlots);
		if (slot < 0) { lane.s(S_ERROR) |= ERR_CAPACITY; continue; }
		lane.i(F_COIN_TYPE, slot) = src.type;
		lane.i(F_COIN_VALUE, slot) = src.value;
		lane.i(F_COIN_TICKS, slot) = src.ticksToCollect;
		lane.f(F_COIN_X, slot) = src.posX;
		lane.f(F_COIN_Y, slot) = src.posY;
		lane.f(F_COIN_VELX, slot) = src.velX;
		lane.f(F_COIN_VELY, slot) = src.velY;
	}
	for (int32_t index = 0; index < root.mowerCount; ++index)
	{
		const RootMower& src = root.mowers[index];
		const int32_t slot = src.row >= 0 && src.row < kMowerSlots ? src.row : index;
		lane.i(F_MOWER_STATE, slot) = src.state;
		lane.i(F_MOWER_TIMER, slot) = src.timer;
		lane.i(F_MOWER_FLAGS, slot) = src.flags;
		lane.i(F_MOWER_CHOMP, slot) = src.chompCounter;
		lane.f(F_MOWER_X, slot) = src.posX;
	}
	for (int32_t index = 0; index < root.packetCount; ++index)
	{
		const RootPacket& src = root.packets[index];
		lane.i(F_PACKET_TYPE, index) = src.type;
		lane.i(F_PACKET_ACTIVE, index) = src.active;
		lane.i(F_PACKET_COUNTER, index) = src.refreshCounter;
		// The kernel keeps the clock in the same units the original does: mRefreshTime is
		// Plant::GetRefreshTime, and SeedPacket::Update activates the packet when the counter passes it.
		lane.i(F_PACKET_REFRESH, index) = src.refreshTime ? src.refreshTime : PlantRefresh(level, src.type);
		lane.i(F_PACKET_COST, index) = src.cost ? src.cost : PlantCost(level, src.type);
	}
	(void)FindZombieDef;
	return 1;
}

// ---------------------------------------------------------------------------
// Leaf export
// ---------------------------------------------------------------------------

PVZ_SIM void StoreResult(Lane lane, const RootRecord& root, RolloutResult& out)
{
	out.rootIndex = 0;
	out.decisions = lane.s(S_DECISIONS);
	out.ticks = lane.s(S_TICK) - lane.s(S_ROOT_TICK);
	out.endTick = lane.s(S_TICK);
	out.result = lane.s(S_RESULT);
	out.sun = lane.s(S_SUN);
	out.wave = lane.s(S_WAVE);
	out.sunProduced = lane.s(S_SUN_PRODUCED) - lane.s(S_ROOT_SUN_PRODUCED);
	out.zombiesKilled = lane.s(S_ZOMBIES_KILLED) - lane.s(S_ROOT_ZOMBIES_KILLED);
	out.plantsEaten = lane.s(S_PLANTS_EATEN) - lane.s(S_ROOT_PLANTS_EATEN);
	out.plantsPlanted = lane.s(S_PLANTS_PLANTED);
	out.mowersLeft = lane.s(S_TRIGGERED_MOWERS) - lane.s(S_ROOT_TRIGGERED_MOWERS);
	out.errorFlags = lane.s(S_ERROR);

	int32_t zombiesAlive = 0, plantsAlive = 0, mowersReady = 0;
	for (int32_t slot = 0; slot < kRows; ++slot) out.rowMinZombieX[slot] = INT32_MIN;
	const int32_t* zombieX = nullptr;
	for (int32_t slot = 0; slot < kZombieSlots; ++slot)
	{
		if (lane.i(F_ZOMBIE_TYPE, slot) < 0) continue;
		const int32_t flags = lane.i(F_ZOMBIE_FLAGS, slot);
		if (flags & ZB_DEAD) continue;
		++zombiesAlive;
		const int32_t row = lane.i(F_ZOMBIE_ROW, slot);
		if (row < 0 || row >= kRows) continue;
		// Fixed-point ×4, the same scaling the corpus uses for the original simulator's floats.
		const int32_t scaled = static_cast<int32_t>(lane.f(F_ZOMBIE_X, slot) * 4.0f);
		if (out.rowMinZombieX[row] == INT32_MIN || scaled < out.rowMinZombieX[row]) out.rowMinZombieX[row] = scaled;
	}
	for (int32_t slot = 0; slot < kPlantSlots; ++slot)
		if (lane.i(F_PLANT_TYPE, slot) >= 0) ++plantsAlive;
	for (int32_t slot = 0; slot < kMowerSlots; ++slot)
		if (lane.i(F_MOWER_STATE, slot) == MOWER_READY) ++mowersReady;
	out.zombiesAlive = zombiesAlive;
	out.plantsAlive = plantsAlive;
	// The response's "mowers_left" is the count of READY mowers, not the trigger delta; keep both,
	// the agreement script compares the count and the search only reads the count.
	out.mowersReady = mowersReady;
	(void)root;
	(void)zombieX;
}

// ---------------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------------

// (channel * kCells + cell) * stride + lane, the layout the convolutions read without transposing.
PVZ_SIM float& CellChannel(Lane lane, float* board, int32_t channel, int32_t cell)
{
	return board[(channel * kCells + cell) * lane.stride + lane.lane];
}

PVZ_SIM float& ScalarChannel(Lane lane, float* scalars, int32_t index)
{
	return scalars[index * lane.stride + lane.lane];
}

PVZ_SIM void BuildFeatures(Lane lane, const LevelInfo& level, const NetBuffers& buffers, const float* waveLog,
	bool previousValid)
{
	for (int32_t channel = 0; channel < kBoardChannels; ++channel)
		for (int32_t cell = 0; cell < kCells; ++cell)
			CellChannel(lane, buffers.board, channel, cell) = 0.0f;

	// Plants: health ratio in the channel of its seed type (EnvironmentRollout.cpp:112).
	for (int32_t slot = 0; slot < kPlantSlots; ++slot)
	{
		const int32_t type = lane.i(F_PLANT_TYPE, slot);
		if (type < 0) continue;
		const int32_t row = slot / kCols, col = slot % kCols;
		// mSquished plants are excluded by the original, which reads live Plant objects; the kernel
		// drops a squished plant the moment it is squished, so no extra test is needed here.
		if (row >= kRows || col >= kGridCols) continue;
		const int32_t channel = Minimum(type, kPlantTypes - 1);
		const int32_t maxHealth = Maximum(lane.i(F_PLANT_HP_MAX, slot), 1);
		CellChannel(lane, buffers.board, channel, slot) +=
			static_cast<float>(static_cast<double>(lane.i(F_PLANT_HP, slot)) / maxHealth);
	}

	// Zombies: health sum / 200, plus the eating and chilled channels (EnvironmentRollout.cpp:116).
	for (int32_t slot = 0; slot < kZombieSlots; ++slot)
	{
		const int32_t type = lane.i(F_ZOMBIE_TYPE, slot);
		if (type < 0) continue;
		const int32_t flags = lane.i(F_ZOMBIE_FLAGS, slot);
		const int32_t row = lane.i(F_ZOMBIE_ROW, slot);
		if ((flags & ZB_DEAD) || !(flags & ZB_ON_BOARD) || row < 0 || row >= kRows) continue;
		double position = lane.f(F_ZOMBIE_X, slot);
		const double boundary = 40.0 + 80.0 * RoundHalfAway((position - 40.0) / 80.0);
		if (AbsDouble(position - boundary) < 0.001) position = RenderedPublicCoordinate(position);
		int32_t col = static_cast<int32_t>(FloorDivide80(position));
		if (col < 0) col = 0;
		if (col > 9) col = 9;
		const int32_t cell = row * kCols + col;
		const double health = static_cast<double>(lane.i(F_ZOMBIE_BODY, slot))
			+ lane.i(F_ZOMBIE_HELM, slot) + lane.i(F_ZOMBIE_SHIELD, slot);
		CellChannel(lane, buffers.board, kPlantTypes + Minimum(type, kZombieTypes - 1), cell) +=
			static_cast<float>(health / 200.0);
		CellChannel(lane, buffers.board, 88, cell) += (flags & ZB_EATING) ? 1.0f : 0.0f;
		CellChannel(lane, buffers.board, 89, cell) += lane.i(F_ZOMBIE_CHILLED, slot) > 0 ? 1.0f : 0.0f;
	}

	// Mowers: channel 90, ready mowers only, always column 0 (EnvironmentRollout.cpp:129).
	for (int32_t slot = 0; slot < kMowerSlots; ++slot)
	{
		const int32_t row = slot;
		if (row >= kRows) continue;
		CellChannel(lane, buffers.board, 90, row * kCols) =
			lane.i(F_MOWER_STATE, slot) == MOWER_READY ? 1.0f : 0.0f;
	}

	// Channel 91: a row whose 9 lawn cells are all the same playable terrain.  On day grass that is
	// every row the level plays in, decided by the level, not by the position.
	for (int32_t row = 0; row < level.playableRows; ++row)
		for (int32_t col = 0; col < kCols; ++col)
			CellChannel(lane, buffers.board, 91, row * kCols + col) = 1.0f;

	float* scalars = buffers.scalars;
	const int32_t sun = lane.s(S_SUN);
	const int32_t waves = Maximum(lane.s(S_NUM_WAVES), 1);
	ScalarChannel(lane, scalars, 0) = static_cast<float>(Minimum(sun, 2000) / 1000.0);
	ScalarChannel(lane, scalars, 1) = static_cast<float>(static_cast<double>(lane.s(S_WAVE)) / waves);
	ScalarChannel(lane, scalars, 2) = static_cast<float>(lane.s(S_TICK) / 60000.0);
	ScalarChannel(lane, scalars, 3) = static_cast<float>(IncomeRate(lane) / 100.0);
	ScalarChannel(lane, scalars, 4) = waveLog[Minimum(waves, 63)];
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
			Minimum(1.0, active || !refresh ? 1.0 : static_cast<double>(counter) / refresh));
	}
	if (previousValid)
	{
		ScalarChannel(lane, scalars, 113) = lane.s(S_PRIOR_PLANT) ? 1.0f : 0.0f;
		ScalarChannel(lane, scalars, 114) = static_cast<float>((lane.s(S_TICK) - lane.s(S_PRIOR_TICK)) / 300.0);
	}
	else
	{
		ScalarChannel(lane, scalars, 113) = 0.0f;
		ScalarChannel(lane, scalars, 114) = 0.0f;
	}
	// Task context (EnvironmentRollout.cpp:161): the multiplier and the roster, constant per task, so
	// the search can batch positions from one task without recomputing them.
	ScalarChannel(lane, scalars, kScalars) = static_cast<float>(level.zombieCountMultiplier / 3.0);
	for (int32_t type = 0; type < kZombieTypes; ++type)
		ScalarChannel(lane, scalars, kScalars + 1 + type) = level.allowed[type] ? 1.0f : 0.0f;
}

}  // namespace pvzc

#endif
