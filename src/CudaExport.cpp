// Export a live original-simulator position into the layout cuda/CudaRollout.h defines.
//
// This file deliberately has no CUDA dependency: it is compiled into pvz-portable always, so a
// machine without nvcc can still produce the .cudaroot files that the GPU box consumes.
//
// The contract is "whatever the kernel needs, and nothing it cannot simulate".  Every field here
// exists because a tick of the CUDA simulator reads it.  Anything presentation-only (reanimations,
// attachments, particles, colours) is skipped -- sim-skip 29 already proved those bits do not change
// the game, and the rollouts only ever need logic.
#include "CudaExport.h"

#include "LawnApp.h"
#include "Lawn/Board.h"
#include "Lawn/Coin.h"
#include "Lawn/LawnMower.h"
#include "Lawn/Plant.h"
#include "Lawn/Projectile.h"
#include "Lawn/SeedPacket.h"
#include "Lawn/Zombie.h"

#include <cstring>

namespace pvzc
{

// The kernel implements this subset only.  A position outside it is exported with supported = 0 and
// a reason, so the caller skips the root instead of measuring a silently wrong simulation.
static bool KernelSupportsPlant(SeedType type)
{
	switch (type)
	{
	case SeedType::SEED_PEASHOOTER: case SeedType::SEED_SUNFLOWER: case SeedType::SEED_CHERRYBOMB:
	case SeedType::SEED_WALLNUT: case SeedType::SEED_POTATOMINE: case SeedType::SEED_SNOWPEA:
	case SeedType::SEED_REPEATER:
		return true;
	default:
		return false;
	}
}

static bool KernelSupportsZombie(ZombieType type)
{
	switch (type)
	{
	case ZombieType::ZOMBIE_NORMAL: case ZombieType::ZOMBIE_FLAG:
	case ZombieType::ZOMBIE_TRAFFIC_CONE: case ZombieType::ZOMBIE_POLEVAULTER:
		return true;
	default:
		return false;
	}
}

// Max health for the types the kernel spawns.  Mirrors the switch in Zombie::ZombieInitialize; the
// phase-1 agreement check compares live-spawned zombies against these numbers, so a drift here is
// caught by measurement rather than by reading.
static void FillZombieHealth(ZombieType type, ZombieDef& def)
{
	def.bodyHealth = 270; def.helmHealth = 0; def.shieldHealth = 0;
	switch (type)
	{
	case ZombieType::ZOMBIE_FLAG: def.bodyHealth = 270; break;
	case ZombieType::ZOMBIE_TRAFFIC_CONE: def.bodyHealth = 270; def.helmHealth = 370; break;
	case ZombieType::ZOMBIE_POLEVAULTER: def.bodyHealth = 500; break;
	default: break;
	}
}

// Health of a freshly planted cell.  Mirrors the switch in the Plant constructor (Plant.cpp:133 sets
// the 300 default, Plant.cpp:288 the wallnut's 4000); the phase-1 agreement check compares
// live-spawned plants against these, so drift shows up as a measurement and not as a reading error.
static int PlantBaseHealth(SeedType type)
{
	switch (type)
	{
	case SeedType::SEED_WALLNUT: return 4000;
	default: return 300;
	}
}

void FillLevelInfo(LawnApp* app, LevelInfo& info)
{
	Board* board = app->mBoard;
	std::memset(&info, 0, sizeof(info));
	info.level = board->mLevel;
	info.numWaves = board->mNumWaves;
	info.playableRows = 5;                      // rows 0..4; row 5 is GRIDSQUARE_DIRT on grass
	info.zombieCountMultiplier = app->mEnvironmentTaskSpec.zombieCountMultiplier;
	for (int type = 0; type < kZombieTypes && type < NUM_ZOMBIE_TYPES; ++type)
	{
		ZombieDef& def = info.zombies[type];
		def.type = type;
		if (!board->mZombieAllowed[type]) continue;
		const ZombieDefinition& source = gZombieDefs[type];
		info.allowed[type] = 1;
		def.value = source.mZombieValue;
		def.firstAllowedWave = source.mFirstAllowedWave;
		def.pickWeight = source.mPickWeight;
		def.variantCount = 5;                   // Zombie::GetVariantCount() for the non-special types
		FillZombieHealth(static_cast<ZombieType>(type), def);
		// PickRandomSpeed(): NORMAL/CONE/FLAG walk in 0.23..0.37 px per tick, the pole vaulter
		// overrides it with 0.66..0.68 before its first jump (Zombie::PickRandomSpeed).
		def.velMin = 0.23f; def.velMax = 0.37f;
		if (static_cast<ZombieType>(type) == ZombieType::ZOMBIE_POLEVAULTER)
		{
			def.velMin = 0.66f; def.velMax = 0.68f;
			def.velX = 0;                       // still drawn from the range, not a fixed value
		}
	}
	static const SeedType deck[] = { SeedType::SEED_PEASHOOTER, SeedType::SEED_SUNFLOWER,
		SeedType::SEED_CHERRYBOMB, SeedType::SEED_WALLNUT, SeedType::SEED_POTATOMINE,
		SeedType::SEED_SNOWPEA, SeedType::SEED_REPEATER };
	info.plantCount = 0;
	for (SeedType seed : deck)
	{
		PlantDef& def = info.plants[info.plantCount++];
		std::memset(&def, 0, sizeof(def));
		const PlantDefinition& source = GetPlantDefinition(seed);
		def.seedType = static_cast<int>(seed);
		def.cost = Plant::GetCost(seed, SeedType::SEED_NONE);
		def.health = PlantBaseHealth(seed);
		def.refreshTime = Plant::GetRefreshTime(seed, SeedType::SEED_NONE);
		def.launchRate = source.mLaunchRate;
		// Behaviour class the kernel branches on, so it never has to re-derive "is this a shooter" from
		// a subclass enum.  Plant::Update dispatches on exactly these five cases for this deck.
		switch (source.mSubClass)
		{
		case PlantSubClass::SUBCLASS_SHOOTER:  def.behavior = PB_SHOOTER; break;
		default:                               def.behavior = PB_NONE; break;
		}
		switch (seed)
		{
		case SeedType::SEED_SUNFLOWER:   def.behavior = PB_SUNPRODUCER;
			// Plant.cpp:192 -- the first delay is RandRangeInt(300, mLaunchRate/2), later ones redraw
			// from [mLaunchRate-150, mLaunchRate] in the kernel.
			def.prodMin = 300; def.prodMax = source.mLaunchRate / 2; def.prodValue = 25; break;
		case SeedType::SEED_CHERRYBOMB:  def.behavior = PB_CHERRY; break;
		case SeedType::SEED_POTATOMINE:  def.behavior = PB_POTATO; break;
		case SeedType::SEED_WALLNUT:     def.behavior = PB_WALL; break;
		default: break;
		}
		// Shooters create a pea (peashooter/snowpea) or two peas (repeater, the second one at
		// mLaunchCounter == 25 -- Plant::UpdateShooter); snow peas carry the chill on the same 20 damage.
		if (source.mSubClass == PlantSubClass::SUBCLASS_SHOOTER)
		{
			const bool snow = seed == SeedType::SEED_SNOWPEA;
			def.damage = gProjectileDefinition[snow ? ProjectileType::PROJECTILE_SNOWPEA
				: ProjectileType::PROJECTILE_PEA].mDamage;
			def.chills = snow ? 1 : 0;
			def.shots = seed == SeedType::SEED_REPEATER ? 2 : 1;
		}
	}
}

// The env's 6000-tick sun income window lives in Python (python/pvz_env.py::_annotate_observation),
// not in the board, so the caller hands the exporter the (tick, produced) events it accumulated.
// They are copied verbatim rather than folded into buckets: ROLLOUT computes the rate as
// sum(tick >= now - 6000) * 1000 / min(6000, tick - historyStart) (main.cpp), and the GPU has to
// reproduce that number exactly or the student sees a different board than the CPU did.
static void FillIncome(const std::vector<std::pair<int, uint32_t>>& events, int tick, int windowStart,
	RootRecord& record)
{
	record.incomeWindowStart = windowStart;
	record.incomeHead = 0;
	record.incomeCount = 0;
	const int cutoff = tick - kIncomeWindowTicks;
	for (const auto& event : events)
	{
		if (event.first < cutoff || event.second == 0) continue;
		if (record.incomeCount >= kIncomeEvents) { record.supported = 0; record.unsupportedReason = 5000; return; }
		const int slot = record.incomeCount++;
		record.incomeTick[slot] = event.first;
		record.incomeValue[slot] = static_cast<int32_t>(event.second);
	}
}

bool BuildCudaRoot(LawnApp* app, const std::vector<std::pair<int, uint32_t>>& incomeEvents,
	int incomeWindowStart, RootRecord& record)
{
	std::memset(&record, 0, sizeof(record));
	record.magic = kMagic;
	record.version = kVersion;
	Board* board = app->mBoard;
	if (!board) return false;

	record.tick = board->mMainCounter;
	record.sun = board->mSunMoney;
	record.wave = board->mCurrentWave;
	record.numWaves = board->mNumWaves;
	record.zombieCountDown = board->mZombieCountDown;
	record.zombieCountDownStart = board->mZombieCountDownStart;
	record.zombieHealthWaveStart = board->mZombieHealthWaveStart;
	record.zombieHealthToNextWave = board->mZombieHealthToNextWave;
	record.hugeWaveCountDown = board->mHugeWaveCountDown;
	record.sunCountDown = board->mSunCountDown;
	record.numSunsFallen = board->mNumSunsFallen;
	record.sunMoneyProduced = static_cast<int32_t>(board->mSunMoneyProduced);
	record.zombiesKilled = static_cast<int32_t>(board->mZombiesKilled);
	record.plantsEaten = static_cast<int32_t>(board->mPlantsEaten);
	record.plantsPlanted = 0;                 // the original does not count plantings
	record.triggeredLawnMowers = board->mTriggeredLawnMowers;
	record.result = app->EnvironmentTerminal()
		? (app->mBoardResult == BoardResult::BOARDRESULT_WON ? 1 :
			(app->mBoardResult == BoardResult::BOARDRESULT_LOST ? -1 : 0)) : 0;
	for (int row = 0; row < kRows; ++row)
	{
		record.rowGotMowered[row] = row < MAX_GRID_SIZE_Y ? board->mWaveRowGotLawnMowered[row] : -100;
		record.rowLastPicked[row] = row < MAX_GRID_SIZE_Y ? board->mRowPickingArray[row].mLastPicked : 0.0f;
		record.rowSecondLastPicked[row] = row < MAX_GRID_SIZE_Y ? board->mRowPickingArray[row].mSecondLastPicked : 0.0f;
	}
	record.supported = 1;
	record.unsupportedReason = 0;
	FillIncome(incomeEvents, record.tick, incomeWindowStart, record);
	for (const Plant* plant : board->mPlants)
	{
		if (plant->mDead || plant->mRow < 0 || plant->mRow >= kRows || plant->mPlantCol < 0 ||
			plant->mPlantCol >= kGridCols) continue;
		if (record.plantCount >= kPlantSlots) { record.supported = 0; record.unsupportedReason = 1000; break; }
		RootPlant& out = record.plants[record.plantCount++];
		out.cell = plant->mRow * kCols + plant->mPlantCol;
		out.type = static_cast<int>(plant->mSeedType);
		out.health = plant->mPlantHealth;
		out.maxHealth = plant->mPlantMaxHealth;
		out.state = static_cast<int>(plant->mState);
		out.stateCountdown = plant->mStateCountdown;
		out.launchCounter = plant->mLaunchCounter;
		out.launchRate = plant->mLaunchRate;
		out.special = plant->mDoSpecialCountdown;
		out.extraA = plant->mShootingCounter;
		out.extraB = plant->mAnimCounter;
		out.onBoard = plant->mIsOnBoard ? 1 : 0;
		// GameObject::mX/mY are int32_t (a plant sits on a grid cell and never moves), unlike a zombie's
		// float mPosX; the record keeps one float type for all positions and the integers survive exactly.
		out.posX = static_cast<float>(plant->mX);
		out.posY = static_cast<float>(plant->mY);
		if (!KernelSupportsPlant(plant->mSeedType) || plant->mSquished ||
			plant->mImitaterType != SeedType::SEED_NONE)
		{
			record.supported = 0;
			record.unsupportedReason = out.type;
		}
	}
	for (Zombie* zombie : board->mZombies)
	{
		if (zombie->mDead || zombie->mRow < 0 || zombie->mRow >= kRows) continue;
		if (record.zombieCount >= kZombieSlots) { record.supported = 0; record.unsupportedReason = 2000; break; }
		RootZombie& out = record.zombies[record.zombieCount++];
		out.type = static_cast<int>(zombie->mZombieType);
		out.row = zombie->mRow;
		out.body = zombie->mBodyHealth; out.bodyMax = zombie->mBodyMaxHealth;
		out.helm = zombie->mHelmHealth; out.helmMax = zombie->mHelmMaxHealth;
		out.shield = zombie->mShieldHealth; out.shieldMax = zombie->mShieldMaxHealth;
		out.flying = zombie->mFlyingHealth;
		out.phase = static_cast<int>(zombie->mZombiePhase);
		out.phaseCounter = zombie->mPhaseCounter;
		out.chilled = zombie->mChilledCounter;
		out.age = zombie->mZombieAge;
		out.fromWave = zombie->mFromWave;
		out.targetCell = zombie->mTargetCol >= 0 && zombie->mTargetRow >= 0
			? zombie->mTargetRow * kCols + zombie->mTargetCol : -1;
		out.posX = zombie->mPosX; out.posY = zombie->mPosY;
		out.velX = zombie->mVelX;
		out.stateBits = (zombie->mIsEating ? ZB_EATING : 0) | (zombie->mHasHead ? ZB_HAS_HEAD : 0) |
			(zombie->mHasArm ? ZB_HAS_ARM : 0) | (zombie->IsDeadOrDying() ? ZB_DYING : 0) |
			(zombie->mMindControlled ? ZB_MIND_CONTROLLED : 0) |
			(zombie->IsOnBoard() ? ZB_ON_BOARD : 0);
		if (!KernelSupportsZombie(zombie->mZombieType) || zombie->mMindControlled ||
			zombie->mZombieType == ZombieType::ZOMBIE_INVALID)
		{
			record.supported = 0;
			record.unsupportedReason = 10000 + out.type;
		}
	}
	for (Projectile* projectile : board->mProjectiles)
	{
		if (projectile->mDead || projectile->mRow < 0 || projectile->mRow >= kRows) continue;
		if (record.projectileCount >= kProjectileSlots) { record.supported = 0; record.unsupportedReason = 3000; break; }
		RootProjectile& out = record.projectiles[record.projectileCount++];
		out.type = static_cast<int>(projectile->mProjectileType);
		out.row = projectile->mRow;
		// Projectiles do not carry damage as state: Projectile::GetProjectileDef() looks it up by type.
		out.damage = gProjectileDefinition[projectile->mProjectileType].mDamage;
		// Whether this projectile chills what it hits, resolved here so the kernel never needs the
		// ProjectileType enum: the only chilling projectile in this deck's rows is the snow pea's.
		out.chill = projectile->mProjectileType == ProjectileType::PROJECTILE_SNOWPEA ? 1 : 0;
		out.age = projectile->mProjectileAge;
		out.posX = projectile->mPosX; out.posY = projectile->mPosY;
		out.velX = projectile->mVelX; out.velY = projectile->mVelY;
	}
	for (Coin* coin : board->mCoins)
	{
		if (coin->mDead || !coin->IsSun()) continue;
		if (record.coinCount >= kCoinSlots) { record.supported = 0; record.unsupportedReason = 4000; break; }
		RootCoin& out = record.coins[record.coinCount++];
		out.value = coin->GetSunValue();
		// LawnApp::EnvironmentWait scores every live sun coin at the top of the next tick, so an
		// imported coin is exactly "one tick from being banked" regardless of where it is falling.
		out.ticksToCollect = 1;
	}
	for (const LawnMower* mower : board->mLawnMowers)
	{
		if (!mower || mower->mRow < 0 || mower->mRow >= kMowerSlots) continue;
		RootMower& out = record.mowers[record.mowerCount++];
		out.row = mower->mRow;
		out.state = static_cast<int>(mower->mMowerState);
		out.timer = mower->mChompCounter > 0 ? mower->mChompCounter : mower->mRollingInCounter;
		out.flags = mower->mDead ? MB_GONE : (mower->mMowerState == LawnMowerState::MOWER_READY ? MB_STATIONARY :
			(mower->mMowerState == LawnMowerState::MOWER_TRIGGERED ? MB_MOWING : 0));
		out.chompCounter = mower->mChompCounter;
		out.posX = mower->mPosX;
	}
	if (board->mSeedBank)
		for (int index = 0; index < board->mSeedBank->mNumPackets && index < kPacketSlots; ++index)
		{
			const SeedPacket& packet = board->mSeedBank->mSeedPackets[index];
			RootPacket& out = record.packets[record.packetCount++];
			out.type = static_cast<int>(packet.mPacketType);
			out.active = packet.mActive > 0 ? 1 : 0;
			out.refreshCounter = packet.mRefreshCounter;
			out.refreshTime = packet.mRefreshTime;
			out.cost = packet.mPacketType == SeedType::SEED_NONE ? 0 :
				Plant::GetCost(packet.mPacketType, packet.mImitaterType);
			if (!KernelSupportsPlant(packet.mPacketType))
			{
				record.supported = 0;
				record.unsupportedReason = 20000 + out.type;
			}
		}
	// The wave future the original already committed to.  RESEED re-seeds only the draws made from
	// here on (rows, spawn jitter, sun timing) and never redraws this table, so a continuation that
	// re-drew it would be measuring a different game than the one the teacher sampled.
	record.waveCount = 0;
	{
		const int waves = board->mNumWaves < kMaxWaves ? board->mNumWaves : kMaxWaves;
		for (int wave = 0; wave < waves; ++wave)
		{
			const int count = board->NumberZombiesInWave(wave);
			if (count > kWaveZombieSlots)
			{
				record.supported = 0;
				record.unsupportedReason = 6000 + wave;
				break;
			}
			record.waveZombieCount[wave] = count;
			for (int index = 0; index < count; ++index)
				record.waveTable[wave][index] = static_cast<int>(board->mZombiesInWave[wave][index]);
			record.waveCount = wave + 1;
		}
		if (board->mNumWaves > kMaxWaves)
		{
			record.supported = 0;
			record.unsupportedReason = 6100;
		}
	}
	return true;
}

bool WriteCudaRecord(std::FILE* out, int32_t kind, const void* data, size_t size)
{
	const int32_t tag[3] = { kMagic, kVersion, kind };
	return std::fwrite(tag, sizeof(tag), 1, out) == 1 &&
		std::fwrite(data, size, 1, out) == 1;
}

bool WriteCudaLevel(std::FILE* out, LawnApp* app)
{
	LevelInfo info;
	FillLevelInfo(app, info);
	return WriteCudaRecord(out, 1, &info, sizeof(info));
}

bool WriteCudaRoot(std::FILE* out, LawnApp* app, const std::vector<std::pair<int, uint32_t>>& incomeEvents,
	int incomeWindowStart)
{
	RootRecord record;
	if (!BuildCudaRoot(app, incomeEvents, incomeWindowStart, record)) return false;
	return WriteCudaRecord(out, 2, &record, sizeof(record));
}

}  // namespace pvzc
