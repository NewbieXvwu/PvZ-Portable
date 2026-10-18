// Shared POD layout for the CUDA rollout simulator.
//
// Compiled by plain C++ (the exporter in src/CudaExport.cpp) and by nvcc (cuda/RolloutCuda.cu).
// It contains no CUDA syntax at all: only constants, plain structs and constexpr offset arithmetic.
// One definition on both sides is what makes "import an arbitrary original position" a memcpy
// instead of a translation layer that can drift.
//
// Two shapes:
//   RootRecord -- one exported original position, slot-major (arrays of entity records).  Written by
//                  the exporter, uploaded verbatim, then transposed into the device SoA.
//   device SoA -- field-major arrays with the lane index innermost, so a warp of 32 lanes touching
//                  "zombie 7's health" reads 32 consecutive elements.
#ifndef PVZ_CUDA_ROLLOUT_H
#define PVZ_CUDA_ROLLOUT_H

#include <cstdint>

namespace pvzc
{

enum {
	kMagic = 0x50564334,          // "PVC4" -- the trailing digit is the tag's own literal, not the
	                              // version below; staleness of a .cudaroot is carried by kVersion.
	kVersion = 7,                 // import animation progress for rises, vaults, ground motion and corpses
	kRows = 6,                    // feature grid rows (only 0..4 are playable on grass)
	kCols = 10,                   // 9 lawn columns + column 9 = "beyond the lawn"
	kCells = kRows * kCols,
	kGridCols = 9,                // lawn columns

	// Activations live in a *padded* grid: valid cell (row, col) sits at (row + kPadRow, col + kPadCol)
	// of a kRowsP x kColsP plane, and the halo stays zero forever.  Padding 1 row / 2 columns is what
	// both a 3x3 with padding 1 and a 1x5 with padding (0, 2) need to read only in-bounds cells, and it
	// is what makes a convolution a batched GEMM: with the halo present, each filter tap is a constant
	// pointer offset -- no bounds test, no gather.  The host build uses the same layout, so the two
	// builds differ only in who does the multiply-accumulate.
	kPadRow = 1,
	kPadCol = 2,
	kRowsP = kRows + 2 * kPadRow,     // 8
	kColsP = kCols + 2 * kPadCol,     // 14
	kPlaneCells = kRowsP * kColsP,    // 112 cells per channel, halo included
	kValidOrigin = kPadRow * kColsP + kPadCol,
	kBoardChannels = 92,
	kScalars = 115,

	// Activations are chunked along lanes.  A batch is ceil(lanes / kNetChunk) blocks, and inside a
	// block the lane index is innermost, so one padded grid row of one channel is kColsP * kNetChunk
	// contiguous floats of which the first kCols * kNetChunk (from kPadCol on) are the valid columns.
	// That contiguous run is what makes a filter tap a single strided-batched GEMM with
	// M = kCols * kNetChunk and kRows batches (cuda/BlasProbe.cu measures this shape).  The host build
	// has one lane and passes netSlot = 0, so it uses block 0 of a one-block buffer: same layout, same
	// semantics, only the multiply-accumulate differs.
	kNetChunk = 1024,
	kNetChunkBits = 10,
	kNetChunkMask = kNetChunk - 1,
	kPlantTypes = 54,
	kZombieTypes = 34,
	kMaxWaves = 30,
	kIncomeWindowTicks = 6000,

	// The value net's encoding (python/pvz_value_net.py::featurize_numpy) is a *different* vector from
	// the student's above: 113 scalars instead of 115 (no prior-action pair) and the task context is a
	// separate 35 floats the caller appends rather than part of the scalar branch.  The leaf exports it
	// so a GPU continuation can be scored by the same net the teacher uses.
	kValueScalars = 5 + 2 * kPlantTypes,                    // 113
	kValueContext = 1 + kZombieTypes,                       // 35
	kValueFloats = kBoardChannels * kCells + kValueScalars + kValueContext,   // 5668
};

// Fixed capacity per lane.  Everything is a bounded slot sweep: no allocation, no linked lists, and
// every lane in a warp runs identical loop trip counts (divergence would serialize the whole warp).
enum {
	kMaxLanes = 256,              // lanes per block; also the default stride of every SoA field
	kPlantSlots = 60,             // indexed by cell: one plant per grass cell
	kZombieSlots = 64,
	kProjectileSlots = 96,
	kCoinSlots = 24,
	kMowerSlots = 6,
	kPacketSlots = 10,
	kIncomeEvents = 160,          // sun-production events kept per lane (see RootRecord)
	kScalarSlots = 64,            // single-value per-lane integers (tick, sun, wave, timers, ...)
	kWaveZombieSlots = 32,        // zombies exported per wave; a bigger wave is reported unsupported
};

// ---------------------------------------------------------------------------
// Static level data (identical for every lane of a task family)
// ---------------------------------------------------------------------------

struct PlantDef
{
	int32_t seedType;             // gPlantDefs index
	int32_t cost;
	int32_t refreshTime;          // ticks: Plant::GetRefreshTime(), the seed-packet cooldown clock
	int32_t health;               // mPlantHealth at planting time (Plant constructor)
	int32_t launchRate;           // gPlantDefs mLaunchRate: the shooters' shot clock, 0 = never fires
	int32_t damage;               // per projectile
	int32_t behavior;             // PB_*: which tick branch the kernel runs for this plant
	int32_t shots;                // projectiles per launch (repeater fires two)
	int32_t prodMin, prodMax;     // sun producer: the steady-state delay range in ticks
	int32_t prodValue;            // sun producer: the coin's value
	int32_t chills;               // its projectile applies the chill (snow pea)
	// The value Plant::FindTargetAndFire writes to mShootingCounter when it arms a shot (Plant.cpp:751-769:
	// 26 for the repeater family, 35 for the rest of the head-reanim shooters, 100 for gatling).  The pea
	// spawns when that countdown decrements into 1 in Plant::UpdateShooting (:3248 -> :3305), i.e.
	// shootCounter - 1 ticks after the decision tick -- plants.md §2.3/§2.6.  Zero would mean "fire on the
	// decision tick", which only the :815 branch (a shooter with no anim_shooting track) does.
	int32_t shootCounter;
};

// Which per-tick branch a plant runs.  Chosen by the exporter from PlantDefinition::mSubClass plus the
// few types whose behaviour is special-cased in Plant::Update, so the kernel never has to know seed ids.
enum { PB_NONE = 0, PB_SHOOTER = 1, PB_SUNPRODUCER = 2, PB_CHERRY = 3, PB_POTATO = 4, PB_WALL = 5 };

struct ZombieDef
{
	int32_t type;                 // ZombieType id
	int32_t value;                // wave-point cost
	int32_t firstAllowedWave;     // 1-based
	int32_t pickWeight;           // 0 = never drawn by weight (flag zombie)
	int32_t bodyHealth;
	int32_t helmHealth;
	int32_t shieldHealth;
	int32_t variantCount;         // gZombieVariants size, feeds the "extra 15% if variant 0" rule
	int32_t velX;                 // 1/4096 px per tick; 0 = draw from velMin..velMax like ZOMBIE_NORMAL
	float velMin, velMax;         // PickRandomSpeed range (px per tick)
};

// Everything the kernel needs that does not change during a rollout.
struct LevelInfo
{
	int32_t level;
	int32_t numWaves;
	int32_t playableRows;         // 5 on grass (row 5 is GRIDSQUARE_DIRT)
	int32_t allowed[kZombieTypes];
	ZombieDef zombies[kZombieTypes];
	int32_t plantCount;
	PlantDef plants[16];
	float zombieCountMultiplier;
};

// ---------------------------------------------------------------------------
// Exported position (slot-major, mirrors the C++ object arrays)
// ---------------------------------------------------------------------------

enum
{
	ZB_DEAD = 1 << 0,
	ZB_EATING = 1 << 1,
	ZB_HAS_HEAD = 1 << 2,
	ZB_HAS_ARM = 1 << 3,
	ZB_DYING = 1 << 4,
	ZB_MIND_CONTROLLED = 1 << 5,
	ZB_SHIELD_DIED = 1 << 6,
	ZB_VAULTED = 1 << 7,          // the pole vaulter has already jumped
	ZB_ON_BOARD = 1 << 8,         // Zombie::IsOnBoard(): provenance, not position (mFromWave is a real wave)
	ZB_BURNED = 1 << 9,
	ZB_ENTERING = 1 << 10,        // still walking in from off the right edge, not yet playable
	ZB_MOWERED = 1 << 11,         // PHASE_ZOMBIE_MOWERED: run over, waiting for the mowed reanim to loop
};

enum
{
	MB_MOWING = 1 << 0,           // mTriggeredLawnMowers semantics: rolling and killing
	MB_STATIONARY = 1 << 1,       // counts as a ready mower for the feature channel
	MB_FALLING = 1 << 2,
	MB_GONE = 1 << 3,
};

// LawnMowerState (src/ConstEnums.h:569).  The kernel stores the original's numbers, so an imported
// mower state needs no translation.
enum { kMowerRollingIn = 0, kMowerReady = 1, kMowerTriggered = 2, kMowerSquished = 3 };

struct RootPlant
{
	int32_t cell;                 // row * 10 + col
	int32_t type;                 // SeedType
	int32_t health, maxHealth;
	int32_t state;                // PlantState (potato arming, cherry burn)
	int32_t stateCountdown;       // mStateCountdown (potato 1500-tick delay before its rise animation)
	int32_t launchCounter;        // mLaunchCounter: counts down to the next shot / sun
	int32_t launchRate;           // mLaunchRate: the reset value drawn from by the shot clock
	int32_t special;              // mDoSpecialCountdown (cherry fuse = 100 ticks)
	int32_t extraA;               // mShootingCounter: the repeater's second-pea clock
	int32_t extraB;               // mAnimCounter: the sprout/produce animation clock
	int32_t onBoard;              // mIsOnBoard: a plant only takes damage once it has landed
	float posX, posY;             // mPosX/mPosY, so blast and bite rectangles use the original's numbers
	float animTime, animRate;
	int32_t animFrames, animLoops;
};

struct RootZombie
{
	int32_t type, row;
	int32_t body, bodyMax, helm, helmMax, shield, shieldMax, flying;
	int32_t phase;                // ZombiePhase
	int32_t phaseCounter;         // remaining ticks of the current phase (vault, death, ...)
	int32_t stateBits;            // ZB_*
	int32_t chilled;              // mChilledCounter
	int32_t age;                  // mZombieAge: the eat cadence clock (age % 4 == 0 -> a chew)
	int32_t fromWave;             // mFromWave, feeds the 催波 health test
	int32_t targetCell;           // the plant being eaten, row * 10 + col (-1 none)
	float posX, posY, velX;
	float animTime, animRate;
	int32_t animStart, animFrames, animLoopType, animLoops, fade;
};

struct RootProjectile
{
	int32_t type, row;            // ProjectileType
	int32_t damage;               // gProjectileDefinition[type].mDamage
	int32_t chill;                // 1 = this projectile applies the slow (the kernel must not need the enum)
	int32_t age;                  // mProjectileAge (snow/pea decoy shot gate at 55)
	float posX, posY, velX, velY;
};

struct RootCoin
{
	int32_t value;                // sun points granted on collection
	int32_t ticksToCollect;       // headless auto-collect: LawnApp scores live sun on the next tick
};

struct RootMower
{
	int32_t row, state, timer, flags, chompCounter;
	float posX;
};

struct RootPacket
{
	int32_t type;                 // SeedType in the slot
	int32_t active;               // mActive > 0
	int32_t refreshCounter;       // counts up every tick, activates when it passes refreshTime
	int32_t refreshTime;          // mRefreshTime: Plant::GetRefreshTime in ticks
	int32_t cost;
};

// One real position, exactly as the original simulator holds it at a decision point.
struct RootRecord
{
	int32_t magic, version;
	// --- scalars
	int32_t tick;                 // mMainCounter
	int32_t sun;
	int32_t wave;                 // mCurrentWave: the next wave to spawn
	int32_t numWaves;
	int32_t zombieCountDown;
	int32_t zombieCountDownStart;
	int32_t zombieHealthWaveStart;
	int32_t zombieHealthToNextWave;
	int32_t hugeWaveCountDown;    // >0 = the huge-wave banner is running
	int32_t sunCountDown;
	int32_t numSunsFallen;
	int32_t sunMoneyProduced;
	int32_t zombiesKilled, plantsEaten;
	int32_t plantsPlanted;        // the original has no such counter: always 0, kernel-side only
	int32_t triggeredLawnMowers;
	int32_t result;               // 0 ongoing, 1 won, -1 lost
	int32_t rowGotMowered[kRows]; // mWaveRowGotLawnMowered
	float rowLastPicked[kRows];
	float rowSecondLastPicked[kRows];
	int32_t incomeWindowStart;    // tick at which the 6000-tick window began (elapsed = tick - start)
	int32_t incomeCount;          // events in the list; the kernel keeps it as a ring
	int32_t incomeHead;           // oldest live event
	int32_t incomeTick[kIncomeEvents];
	int32_t incomeValue[kIncomeEvents];
	int32_t plantCount, zombieCount, projectileCount, coinCount, mowerCount, packetCount;
	RootPlant plants[kPlantSlots];
	RootZombie zombies[kZombieSlots];
	RootProjectile projectiles[kProjectileSlots];
	RootCoin coins[kCoinSlots];
	RootMower mowers[kMowerSlots];
	RootPacket packets[kPacketSlots];
	// Board::mZombiesInWave, the future the original already committed to.  RESEED changes only the
	// draws made from here on (rows, spawn jitter, sun timing), never this table, so a continuation
	// that redrew it would be measuring a different game than the one the teacher sampled.
	int32_t waveCount;              // == numWaves, capped at kMaxWaves
	int32_t waveZombieCount[kMaxWaves];
	int32_t waveTable[kMaxWaves][kWaveZombieSlots];
	int32_t supported;            // exporter's verdict: 1 = the kernel can simulate this position
	int32_t unsupportedReason;
};

// ---------------------------------------------------------------------------
// Device side request / result
// ---------------------------------------------------------------------------

// The committed wave future, indexed by root rather than by lane.  A continuation never rewrites it
// (only the per-lane spawn cursor advances), so 200 roots cost 200 tables instead of one per lane --
// at 64 lanes per block that saves about 250 KB of the working set.
struct WaveTable
{
	int32_t waveCount;
	int32_t zombieCount[kMaxWaves];
	int32_t type[kMaxWaves][kWaveZombieSlots];
};

// One rollout = (root, first action, random future, horizon).  Mirrors what python/pvz_env.py sends
// with ROLLOUT: kind 0 = the student decides at once, otherwise the encoded action runs first.
struct RolloutRequest
{
	int32_t rootIndex;            // which RootRecord to start from
	int32_t futureSeed;           // Philox counter seed: one independent future per rollout
	int32_t untilTick;            // absolute tick to stop at (the search's horizon)
	int32_t maxActions;
	int32_t firstActionKind;      // 0 = student, 1 = plant, 2 = wait
	int32_t firstPacket, firstRow, firstCol;
	int32_t firstWaitTicks;
	int32_t wantFeatures;         // dump the leaf's value-net features into the feature buffer
};

// Per-rollout leaf statistics.  Everything the phase-1 agreement check reads lives here.  The
// counters are deltas over the rollout (the root's own counters subtracted), which is what the
// ROLLOUT response's "events" block reports for the original simulator.
struct RolloutResult
{
	int32_t rootIndex;
	int32_t decisions;            // student decision points reached
	int32_t netCalls;             // the ones that actually ran the network, the CPU's networkCalls
	int32_t ticks;                // ticks simulated
	int32_t endTick;
	int32_t result;               // 0 ongoing, 1 won, -1 lost
	int32_t sun;                  // leaf sun money (absolute)
	int32_t wave;                 // waves started (absolute board counter at the leaf)
	int32_t zombiesAlive, zombiesKilled, plantsAlive, plantsEaten, plantsPlanted;
	int32_t mowersLeft;           // mowers still READY at the leaf, as ROLLOUT reports it
	int32_t mowersTriggered;      // delta of mTriggeredLawnMowers over the continuation
	int32_t sunProduced;          // delta over the rollout
	int32_t errorFlags;           // ERR_* bitmask
	int32_t plantedByType[kPacketSlots];
	int32_t rowMinZombieX[kRows]; // fixed-point (x * 4), INT32_MIN when the row is clear
};

enum {
	ERR_UNSUPPORTED_PLANT = 1 << 0,
	ERR_UNSUPPORTED_ZOMBIE = 1 << 1,
	ERR_CAPACITY = 1 << 2,
	ERR_BAD_ROOT = 1 << 3,
	ERR_NET_WIDTH = 1 << 4,
};

// ---------------------------------------------------------------------------
// Device SoA field layout
// ---------------------------------------------------------------------------

// An entity slot is empty exactly when its type field is negative.  Dead entities keep their slot
// until the sweep that removes them, so the slot order stays the spawn order the original iterates.
enum IntField {
	F_PLANT_TYPE, F_PLANT_HP, F_PLANT_HP_MAX, F_PLANT_STATE, F_PLANT_STATE_COUNTDOWN,
	F_PLANT_LAUNCH, F_PLANT_LAUNCH_RATE, F_PLANT_SPECIAL, F_PLANT_SHOOTING, F_PLANT_ANIM,
	F_PLANT_ANIM_FRAMES, F_PLANT_ANIM_LOOPS,
	F_PLANT_ONBOARD,
	F_ZOMBIE_TYPE, F_ZOMBIE_ROW, F_ZOMBIE_BODY, F_ZOMBIE_BODY_MAX, F_ZOMBIE_HELM, F_ZOMBIE_SHIELD,
	F_ZOMBIE_FLYING,
	F_ZOMBIE_PHASE, F_ZOMBIE_PHASE_COUNTER, F_ZOMBIE_FLAGS, F_ZOMBIE_CHILLED,
	F_ZOMBIE_ANIM_START, F_ZOMBIE_ANIM_FRAMES, F_ZOMBIE_ANIM_LOOP_TYPE, F_ZOMBIE_ANIM_LOOPS, F_ZOMBIE_FADE,
	F_ZOMBIE_AGE, F_ZOMBIE_FROM_WAVE, F_ZOMBIE_TARGET,
	F_PROJ_TYPE, F_PROJ_ROW, F_PROJ_AGE, F_PROJ_DAMAGE, F_PROJ_CHILL,
	F_COIN_ACTIVE, F_COIN_VALUE, F_COIN_TICKS,
	F_MOWER_STATE, F_MOWER_TIMER, F_MOWER_FLAGS, F_MOWER_CHOMP,
	F_PACKET_TYPE, F_PACKET_ACTIVE, F_PACKET_COUNTER, F_PACKET_REFRESH, F_PACKET_COST,
	F_PLANTED_BY_TYPE,
	F_ROW_GOT_MOWERED,
	F_INCOME_TICK, F_INCOME_VALUE,
	F_SCALAR,                       // kScalarSlots single-value fields, see ScalarField
	F_INT_FIELD_COUNT
};

enum FloatField {
	F_PLANT_X, F_PLANT_Y,
	F_PLANT_ANIM_TIME, F_PLANT_ANIM_RATE,
	F_ZOMBIE_X, F_ZOMBIE_Y, F_ZOMBIE_VELX, F_ZOMBIE_VELY,
	F_ZOMBIE_ANIM_TIME, F_ZOMBIE_ANIM_RATE,
	F_PROJ_X, F_PROJ_Y, F_PROJ_VELX, F_PROJ_VELY,
	F_MOWER_X,
	F_ROW_LAST_PICKED, F_ROW_SECOND_LAST_PICKED,
	F_FLOAT_FIELD_COUNT
};

// Indices into the F_SCALAR block.  `tick` and `sun` live here too: they are read on every feature
// build, so keeping them in the same array as everything else costs one less pointer chase.
// The S_ROOT_* entries are the counters the position arrived with, because RolloutResult reports
// deltas over the continuation exactly like the ROLLOUT response's "events" block.
enum ScalarField {
	S_TICK, S_SUN, S_WAVE, S_NUM_WAVES, S_ZOMBIE_COUNT_DOWN, S_ZOMBIE_COUNT_DOWN_START,
	S_HEALTH_WAVE_START, S_HEALTH_TO_NEXT_WAVE, S_HUGE_WAVE_COUNT_DOWN,
	S_SUN_COUNT_DOWN, S_NUM_SUNS_FALLEN, S_SUN_PRODUCED,
	S_ZOMBIES_KILLED, S_PLANTS_EATEN, S_PLANTS_PLANTED, S_TRIGGERED_MOWERS,
	S_RESULT, S_DECISIONS, S_PRIOR_TICK, S_PRIOR_PLANT, S_INCOME_WINDOW_START,
	S_INCOME_HEAD, S_INCOME_COUNT, S_FIRST_ACTION, S_ERROR, S_WAVE_SPAWN_CURSOR,
	S_ROOT_TICK, S_ROOT_SUN_PRODUCED, S_ROOT_ZOMBIES_KILLED, S_ROOT_PLANTS_EATEN,
	S_ROOT_TRIGGERED_MOWERS, S_TICKS_SINCE_EVENT,
	S_NET_CALLS,                    // forwards that actually ran the net, the CPU path's networkCalls
	S_ROOT_INDEX,                   // which RootRecord this lane loaded, for indexing the wave table
	S_RNG_HIGH, S_RNG_LOW,
	// Lane lifecycle for the three-kernel round (sim → net+choose → apply).  A lane is LANE_RUNNING
	// while the sim kernel may still advance it, LANE_DECIDING when it stopped at a decision point, and
	// LANE_DONE once it hit the horizon, the action cap or a terminal result.
	//
	// `S_STOP_REASON` is declared and never written.  It is not an oversight to delete: the entries below
	// it are positional, and a `.cudaroot` payload stores the lane's initial scalars in this order, so
	// dropping the slot would shift every later index and invalidate every corpus on disk for a field
	// that says nothing `result`/`ticks`/`decisions` don't already say (ongoing + ticks == horizon is the
	// horizon cut; ongoing + decisions == max_actions is the cap).  Kept as padding, priced against that.
	S_STATE, S_UNTIL_TICK, S_MAX_ACTIONS, S_STOP_REASON,
	// A decision round is split across two kernels when the trunk runs batched: RoundPre observes the
	// clock and the sun produced so far, RoundPost finishes the round with them two ticks later.  They
	// are per-lane state for the duration of one round, so they live in the scalar block rather than in
	// registers the second kernel cannot see.
	S_ROUND_TICK, S_ROUND_PRODUCED, S_ROUNDS,
	// The teacher's forced first action (Rollout.inc's RolloutBegin).  Kind 0 = the student decides at
	// once, 1 = plant this packet at (row, col), 2 = wait `ticks`.  Written by whoever builds the lane's
	// workload, before RolloutBegin; a lane that runs no forced action still has to be written, because
	// the device's scalar block is reused across batches and repeats.
	S_FIRST_KIND, S_FIRST_PACKET, S_FIRST_ROW, S_FIRST_COL, S_FIRST_TICKS,
	S_FUTURE_SEED,                // immutable seed for resampling future wave composition
	S_SCALAR_COUNT
};

enum { LANE_RUNNING = 0, LANE_DECIDING = 1, LANE_DONE = 2 };

constexpr int IntSlotCount(IntField field)
{
	switch (field)
	{
	case F_PLANT_TYPE: case F_PLANT_HP: case F_PLANT_HP_MAX: case F_PLANT_STATE:
	case F_PLANT_STATE_COUNTDOWN: case F_PLANT_LAUNCH: case F_PLANT_LAUNCH_RATE:
	case F_PLANT_SPECIAL: case F_PLANT_SHOOTING: case F_PLANT_ANIM: case F_PLANT_ONBOARD:
	case F_PLANT_ANIM_FRAMES: case F_PLANT_ANIM_LOOPS:
		return kPlantSlots;
	case F_ZOMBIE_TYPE: case F_ZOMBIE_ROW: case F_ZOMBIE_BODY: case F_ZOMBIE_BODY_MAX:
	case F_ZOMBIE_HELM:
	case F_ZOMBIE_SHIELD: case F_ZOMBIE_FLYING: case F_ZOMBIE_PHASE: case F_ZOMBIE_PHASE_COUNTER:
	case F_ZOMBIE_FLAGS: case F_ZOMBIE_CHILLED: case F_ZOMBIE_AGE: case F_ZOMBIE_FROM_WAVE:
	case F_ZOMBIE_ANIM_START: case F_ZOMBIE_ANIM_FRAMES: case F_ZOMBIE_ANIM_LOOP_TYPE:
	case F_ZOMBIE_ANIM_LOOPS: case F_ZOMBIE_FADE:
	case F_ZOMBIE_TARGET:
		return kZombieSlots;
	case F_PROJ_TYPE: case F_PROJ_ROW: case F_PROJ_AGE: case F_PROJ_DAMAGE: case F_PROJ_CHILL:
		return kProjectileSlots;
	case F_COIN_ACTIVE: case F_COIN_VALUE: case F_COIN_TICKS:
		return kCoinSlots;
	case F_MOWER_STATE: case F_MOWER_TIMER: case F_MOWER_FLAGS: case F_MOWER_CHOMP:
		return kMowerSlots;
	case F_PACKET_TYPE: case F_PACKET_ACTIVE: case F_PACKET_COUNTER: case F_PACKET_REFRESH:
	case F_PACKET_COST: case F_PLANTED_BY_TYPE:
		return kPacketSlots;
	case F_ROW_GOT_MOWERED:
		return kRows;
	case F_INCOME_TICK: case F_INCOME_VALUE:
		return kIncomeEvents;
	case F_SCALAR:
		return kScalarSlots;
	default:
		return 0;
	}
}

constexpr int FloatSlotCount(FloatField field)
{
	switch (field)
	{
	case F_PLANT_X: case F_PLANT_Y: case F_PLANT_ANIM_TIME: case F_PLANT_ANIM_RATE:
		return kPlantSlots;
	case F_ZOMBIE_X: case F_ZOMBIE_Y: case F_ZOMBIE_VELX: case F_ZOMBIE_VELY:
	case F_ZOMBIE_ANIM_TIME: case F_ZOMBIE_ANIM_RATE:
		return kZombieSlots;
	case F_PROJ_X: case F_PROJ_Y: case F_PROJ_VELX: case F_PROJ_VELY:
		return kProjectileSlots;
	case F_MOWER_X:
		return kMowerSlots;
	case F_ROW_LAST_PICKED: case F_ROW_SECOND_LAST_PICKED:
		return kRows;
	default:
		return 0;
	}
}

constexpr int IntFieldOffset(IntField field)
{
	int offset = 0;
	for (int index = 0; index < static_cast<int>(field); ++index)
		offset += IntSlotCount(static_cast<IntField>(index));
	return offset;
}

constexpr int FloatFieldOffset(FloatField field)
{
	int offset = 0;
	for (int index = 0; index < static_cast<int>(field); ++index)
		offset += FloatSlotCount(static_cast<FloatField>(index));
	return offset;
}

constexpr int IntSlotsTotal() { return IntFieldOffset(F_INT_FIELD_COUNT); }
constexpr int FloatSlotsTotal() { return FloatFieldOffset(F_FLOAT_FIELD_COUNT); }

// ---------------------------------------------------------------------------
// Student network (the PVZRN dump produced by python/pvz_rollout_policy.py::export_student)
// ---------------------------------------------------------------------------

enum { kMaxNetWidth = 128 };     // v14 is width 64; larger dumps are rejected, not truncated

}  // namespace pvzc

#endif
