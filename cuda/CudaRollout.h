// Shared POD layout for the CUDA rollout simulator.
//
// This header is compiled by plain C++ (the exporter in src/main.cpp) and by nvcc (the kernel in
// cuda/).  It therefore contains no CUDA syntax at all: only constants, plain structs and constexpr
// offset arithmetic.  Keeping the single definition on both sides is what makes "import an arbitrary
// original position" a memcpy instead of a translation layer that can drift.
//
// Two shapes are defined here:
//   RootRecord  -- one exported original position, slot-major (an array of entity records).  Written
//                  by Board->record, uploaded, then transposed into the device SoA by a setup kernel.
//   device SoA   -- field-major arrays with the lane index innermost, so that a warp of 32 lanes
//                  touching "zombie 7's health" reads 32 consecutive floats.
#ifndef PVZ_CUDA_ROLLOUT_H
#define PVZ_CUDA_ROLLOUT_H

#include <cstdint>

namespace pvzc
{

enum {
	kMagic = 0x50564331,          // "PVC1"
	kRows = 6,                    // feature grid rows (only 0..4 are playable on grass)
	kCols = 10,                   // 9 lawn columns + column 9 = "beyond the lawn"
	kCells = kRows * kCols,
	kGridCols = 9,                // lawn columns
	kBoardChannels = 92,
	kScalars = 115,
	kPlantTypes = 54,
	kZombieTypes = 34,
	kMaxWaves = 30,
};

// Fixed capacity per lane.  Everything is a bounded slot sweep: no allocation, no linked lists, and
// every lane in a warp runs the same loop trip counts (divergence would serialize the whole warp).
enum {
	kMaxLanes = 256,              // lanes per block; also the stride of every SoA field
	kPlantSlots = 60,             // indexed by cell: one plant per grass cell
	kZombieSlots = 64,
	kProjectileSlots = 96,
	kCoinSlots = 24,
	kMowerSlots = 6,
	kPacketSlots = 10,
	kIncomeBuckets = 60,          // 100-tick buckets covering the student's 6000-tick income window
	kHistorySlots = 8,            // recent (tick, plant) decisions for scalars[113..114]
};

// ---------------------------------------------------------------------------
// Exported position (slot-major, mirrors the C++ object arrays)
// ---------------------------------------------------------------------------

// SeedType ids we simulate.  Anything else arriving in a RootRecord is a bug in the exporter and is
// rejected by the kernel (it sets an error flag rather than silently mis-simulating).
enum PlantKind {
	PLANT_NONE = -1,
	PLANT_PEASHOOTER = 0, PLANT_SUNFLOWER = 1, PLANT_CHERRY = 2, PLANT_WALLNUT = 3,
	PLANT_POTATO = 4, PLANT_SNOWPEA = 5, PLANT_REPEATER = 7,
};

enum ZombieKind {
	ZOMBIE_NONE = -1,
	ZOMBIE_NORMAL = 0, ZOMBIE_FLAG = 1, ZOMBIE_CONEHEAD = 2, ZOMBIE_POLEVAULTER = 4,
};

struct RootPlant
{
	int32_t cell;                 // row * 10 + col
	int32_t type;                 // SeedType
	int32_t health, maxHealth;
	int32_t state;                // Plant::mState
	int32_t stateCountdown;       // Plant::mStateCountdown
	int32_t launchCounter;        // Plant::mLaunchCounter
	int32_t launchRate;           // Plant::mLaunchRate
	int32_t shootingCounter;      // Plant::mShootingCounter
};

struct RootZombie
{
	int32_t type, row;
	int32_t body, bodyMax, helm, helmMax, shield, shieldMax, flying;
	int32_t phase;                // ZombiePhase
	int32_t phaseCounter;
	int32_t stateBits;            // ZB_DEAD|ZB_EATING|ZB_HAS_HEAD|ZB_HAS_ARM|ZB_DYING|ZB_MIND_CONTROLLED|...
	int32_t chilled;              // mChilledCounter
	int32_t eatCounter;           // mEatingNormalTime / mEatingFrozenTime
	int32_t fromWave;             // mFromWave, feeds the 催波 health test
	int32_t targetCol;            // mTargetCol of the plant being eaten
	int32_t jumpCounter;          // mJumpTime for the pole vault
	int32_t groanDelay;           // ZombieStartGroaningSoundDelay
	float posX, posY, velX, velY;
};

struct RootProjectile
{
	int32_t type, row;            // ProjectileType
	int32_t damage;
	float posX, posY, velX, velY;
};

struct RootCoin
{
	int32_t type;                 // CoinType; only sun matters here
	int32_t state;                // CoinMotion
	float posX, posY, velX, velY;
};

struct RootMower
{
	int32_t row, state;
	float posX;
};

struct RootPacket
{
	int32_t type, imitaterType;
	int32_t active;
	int32_t refreshCounter, refreshTime;
	int32_t cost;
};

// One real position, exactly as the original simulator holds it at a decision point.
struct RootRecord
{
	int32_t magic;
	int32_t version;
	int32_t level;
	int32_t tick;
	int32_t sun;
	int32_t wave;                 // mCurrentWave (next wave to spawn)
	int32_t numWaves;             // 30 on level 7 replay
	int32_t zombieCountDown;
	int32_t zombieCountDownStart;
	int32_t mZombieHealthWaveStart;
	int32_t zombieHealthToNextWave;
	int32_t hugeWaveCountDown;
	int32_t flagRaiseCounter;
	int32_t sunCountDown;
	int32_t numSunsFallen;
	int32_t sunMoneyProduced;     // running total, for the income feature
	int32_t zombiesKilled, plantsEaten, triggeredLawnMowers;
	int32_t rowGotMowered[kRows]; // mWaveRowGotLawnMowered
	int32_t lastPicked[kRows];    // Board's smooth row picker
	int32_t secondLastPicked[kRows];
	int32_t zombieAllowed[kZombieTypes];
	int32_t incomeWindowStart;    // tick at which the 6000-tick window began
	int32_t incomeTick[kIncomeBuckets];
	int32_t incomeValue[kIncomeBuckets];
	int32_t plantCount, zombieCount, projectileCount, coinCount, mowerCount, packetCount;
	RootPlant plants[kPlantSlots];
	RootZombie zombies[kZombieSlots];
	RootProjectile projectiles[kProjectileSlots];
	RootCoin coins[kCoinSlots];
	RootMower mowers[kMowerSlots];
	RootPacket packets[kPacketSlots];
	float zombieCountMultiplier;
	int32_t terminalResult;       // 0 ongoing, 1 won, -1 lost
};

// ---------------------------------------------------------------------------
// Device side request / result
// ---------------------------------------------------------------------------

// One rollout = (root, first action, random future, horizon).  The first action mirrors what
// python/pvz_env.py sends with ROLLOUT: "A" = let the student choose immediately, or an encoded
// PLANT/WAIT command that is executed before the student takes over.
struct RolloutRequest
{
	int32_t rootIndex;            // which RootRecord to start from
	uint32_t futureSeed;          // Philox counter seed: one independent future per rollout
	int32_t untilTick;            // absolute tick to stop at (horizon)
	int32_t maxActions;
	int32_t firstActionKind;      // 0 = student decides, 1 = plant, 2 = wait
	int32_t firstPacket, firstRow, firstCol;
	int32_t firstWaitTicks;
	int32_t wantFeatures;         // dump the value-net features of the leaf into the result buffer
};

// Per-rollout leaf statistics.  Everything phase-1 agreement checks lives here.
struct RolloutResult
{
	int32_t rootIndex;
	int32_t decisions;            // student forwards actually run
	int32_t ticks;                // ticks simulated
	int32_t endTick;
	int32_t result;               // 0 ongoing, 1 won, -1 lost, -2 internal error
	int32_t sun;
	int32_t wave;                 // waves started
	int32_t zombiesAlive, zombiesKilled, plantsAlive, plantsEaten, plantsPlanted;
	int32_t mowersLeft;
	int32_t plantedByType[kPacketSlots];
	int32_t rowMinZombieX[kRows]; // fixed-point (x * 4), INT32_MIN when the row is clear
	int32_t sunProduced;
	int32_t errorFlags;           // PVZ_ERROR_* bitmask
};

enum {
	ERR_UNSUPPORTED_PLANT = 1 << 0,
	ERR_UNSUPPORTED_ZOMBIE = 1 << 1,
	ERR_CAPACITY = 1 << 2,
	ERR_BAD_ROOT = 1 << 3,
	ERR_NET_WIDTH = 1 << 4,
};

// Value-net / student-net features of a leaf: board[kBoardChannels][kCells] + scalars[kScalars].
// Kept separate from RolloutResult because it is 22 KB per rollout: only sampled leaves ask for it.
struct LeafFeatures
{
	float board[kBoardChannels * kCells];
	float scalars[kScalars];
};

// ---------------------------------------------------------------------------
// Student network (PVZRN dump format, see python/pvz_rollout_policy.py::export_student)
// ---------------------------------------------------------------------------

// The kernel runs one forward per lane, so the weight tensors stay in their native PyTorch layout and
// are read directly; no repacking.  Layer order matches EnvironmentRollout::Load's `counts`.
enum Weights {
	W_CONV1 = 0, W_CONV1_BIAS, W_CONV2, W_CONV2_BIAS, W_CONV3, W_CONV3_BIAS,
	W_SCALAR, W_SCALAR_BIAS,               // context branch from the global scalars
	W_PLANT, W_PLANT_BIAS,
	W_PACKET, W_PACKET_BIAS,
	W_PACKET_CELL,                          // cell head: 1x1 conv selected by the packet
	W_CELL_CONV, W_CELL_CONV_BIAS, W_CELL_CONTEXT, W_CELL_OUT,
	W_COUNT
};

}  // namespace pvzc

#endif
