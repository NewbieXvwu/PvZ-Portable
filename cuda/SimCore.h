// The lane state container for the re-implemented rollout simulator.
//
// This header is compiled twice: into `cuda/RolloutCuda.cu` (nvcc, one thread = one lane, the SoA
// lives in global memory) and into `cuda/SimHost.cpp` (plain g++, one lane, the SoA lives in a stack
// array).  The mechanics in SimTick.inc therefore exist exactly once, which is what makes the
// agreement test a test of the *mechanics* rather than of two independent rewrites: if the host build
// and the GPU build disagree, that is a GPU bug; if the host build and the original simulator
// disagree, that is a mechanics bug.
#ifndef PVZ_SIM_CORE_H
#define PVZ_SIM_CORE_H

#include "CudaRollout.h"

#include <cstdint>

#ifdef __CUDACC__
#define PVZ_SIM __host__ __device__ inline
#else
#define PVZ_SIM inline
#endif

namespace pvzc
{

// One lane's view of the batch SoA.  Field arrays are field-major with the lane index innermost, so
// a warp reading "zombie 7's body health" reads 32 consecutive elements.
struct Lane
{
	int32_t* ints;
	float* floats;
	int32_t lane;
	int32_t stride;      // padded lane count; 1 in the single-lane host build

	PVZ_SIM int32_t* base(IntField field) { return ints + IntFieldOffset(field) * stride + lane; }
	PVZ_SIM float* base(FloatField field) { return floats + FloatFieldOffset(field) * stride + lane; }
	PVZ_SIM int32_t& i(IntField field, int slot) { return base(field)[slot * stride]; }
	PVZ_SIM float& f(FloatField field, int slot) { return base(field)[slot * stride]; }
	PVZ_SIM int32_t& s(ScalarField which) { return base(F_SCALAR)[which]; }
};

// ---------------------------------------------------------------------------
// Small arithmetic helpers.  Deliberately hand written: the device and the host build must round the
// same way, and the original's own arithmetic is float/double IEEE with no fast-math shortcuts.
// ---------------------------------------------------------------------------

PVZ_SIM int32_t Minimum(int32_t a, int32_t b) { return a < b ? a : b; }
PVZ_SIM int32_t Maximum(int32_t a, int32_t b) { return a > b ? a : b; }
PVZ_SIM int32_t ClampInt(int32_t value, int32_t lo, int32_t hi)
{
	return value < lo ? lo : (value > hi ? hi : value);
}
PVZ_SIM double AbsDouble(double value) { return value < 0.0 ? -value : value; }
PVZ_SIM int64_t Truncate(double value) { return static_cast<int64_t>(value); }
PVZ_SIM double FloorDouble(double value)
{
	const int64_t whole = static_cast<int64_t>(value);
	return value < static_cast<double>(whole) ? static_cast<double>(whole - 1) : static_cast<double>(whole);
}
PVZ_SIM double RoundHalfAway(double value)
{
	return value >= 0.0 ? static_cast<double>(Truncate(value + 0.5))
		: -static_cast<double>(Truncate(-value + 0.5));
}

// The feature grid column of a world x: floor((x - 40) / 80) clamped to 0..9 (EnvironmentRollout.cpp:123).
PVZ_SIM int32_t FeatureColumn(double position)
{
	return ClampInt(static_cast<int32_t>(FloorDouble((position - 40.0) / 80.0)), 0, 9);
}

// First free entity slot, scanning the low slots first so a batch keeps its entity arrays dense.
PVZ_SIM int32_t FindFreeSlot(Lane lane, IntField typeField, int32_t slots)
{
	int32_t* column = lane.base(typeField);
	for (int32_t slot = 0; slot < slots; ++slot)
		if (column[slot * lane.stride] < 0) return slot;
	return -1;
}

// Clearing a slot means "no entity here", which the sweeps test by type.
PVZ_SIM void ClearIntFields(Lane lane, IntField first, IntField last, int32_t slot, int32_t value)
{
	for (int32_t field = first; field <= last; ++field)
		lane.i(static_cast<IntField>(field), slot) = value;
}

// ---------------------------------------------------------------------------
// Per-lane counter-based RNG.  The brief allows a random stream different from the original; what
// must not change is the distribution of what is drawn.
// ---------------------------------------------------------------------------

struct Rand
{
	uint64_t state;

	PVZ_SIM void Seed(int32_t counter, int32_t stream)
	{
		state = (static_cast<uint64_t>(static_cast<uint32_t>(counter)) << 32)
			| static_cast<uint32_t>(static_cast<uint32_t>(stream) ^ 0x9E3779B9u);
		state += 0x9E3779B97F4A7C15ull;
		Next();
	}

	PVZ_SIM uint32_t Next()
	{
		state += 0x9E3779B97F4A7C15ull;
		uint64_t mixed = state;
		mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
		mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
		mixed ^= mixed >> 31;
		return static_cast<uint32_t>(mixed >> 32);
	}

	// Uniform 0..n-1 by truncation, the same shape as the original SexRand::Random(n).
	PVZ_SIM int32_t Int(int32_t n) { return n <= 0 ? 0 : static_cast<int32_t>(Next() % static_cast<uint32_t>(n)); }
	PVZ_SIM int32_t Range(int32_t lo, int32_t hi) { return hi <= lo ? lo : lo + Int(hi - lo + 1); }
	PVZ_SIM float ZeroToOne() { return static_cast<float>(Next() >> 8) * (1.0f / 16777216.0f); }
};

// The lane's own stream, kept in the scalar block so the tick logic needs no extra pointer.  Two
// uint32 halves live in two int slots; the RNG advances a fixed number of times per decision, not per
// tick, so it does not have to be reloaded from disk.
PVZ_SIM uint64_t RandState(Lane lane)
{
	return (static_cast<uint64_t>(static_cast<uint32_t>(lane.s(S_RNG_HIGH))) << 32)
		| static_cast<uint32_t>(lane.s(S_RNG_LOW));
}

// ---------------------------------------------------------------------------
// Six significant decimal digits, matching what std::ostream writes for a double.  The original
// observation path renders zombie coordinates that way, and EnvironmentRollout::Features applies the
// same rounding to a zombie standing within 0.001 px of a cell boundary; reproducing it keeps the
// student's zombie-position channel identical on both sides.  Only boundary cases pay for it.
// ---------------------------------------------------------------------------

PVZ_SIM double RenderedPublicCoordinate(double value)
{
	const double magnitude = AbsDouble(value);
	if (magnitude <= 0.0) return value;
	// 6 significant digits: the last kept place is 10^(e-5) where 10^e <= |value| < 10^(e+1).
	static const double kPowers[9] = { 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0,
		10000000.0, 100000000.0 };
	int exponent = 0;
	while (exponent < 8 && magnitude >= kPowers[exponent + 1]) ++exponent;
	const int keep = exponent - 5;
	const double scale = keep >= 0 ? kPowers[keep] : kPowers[0] / kPowers[0 - keep];
	const double rounded = RoundHalfAway(magnitude / scale) * scale;
	return value < 0.0 ? -rounded : rounded;
}

// ---------------------------------------------------------------------------
// Income window (student feature scalar[3]).
//
// main.cpp keeps a per-decision-step list: after each step it appends (tick_after_step, sun produced
// during that step); before each prediction it drops entries older than 6000 ticks, sums the rest and
// divides by min(6000, tick - window_start).  The ring below is that list, kept exactly rather than
// bucketed, so this feature carries no approximation error.
// ---------------------------------------------------------------------------

PVZ_SIM void PushIncomeEvent(Lane lane, int32_t tick, int32_t produced)
{
	if (produced <= 0) return;
	const int32_t count = lane.s(S_INCOME_COUNT);
	if (count < kIncomeEvents)
	{
		lane.i(F_INCOME_TICK, count) = tick;
		lane.i(F_INCOME_VALUE, count) = produced;
		lane.s(S_INCOME_COUNT) = count + 1;
		return;
	}
	// Full ring: drop the oldest quarter.  Filling 160 slots inside a 6000-tick window would take a
	// decision every 37 ticks, so this is a slow path, not a per-step one.
	const int32_t drop = kIncomeEvents / 4;
	for (int32_t slot = 0; slot + drop < kIncomeEvents; ++slot)
	{
		lane.i(F_INCOME_TICK, slot) = lane.i(F_INCOME_TICK, slot + drop);
		lane.i(F_INCOME_VALUE, slot) = lane.i(F_INCOME_VALUE, slot + drop);
	}
	const int32_t tail = kIncomeEvents - drop;
	lane.i(F_INCOME_TICK, tail) = tick;
	lane.i(F_INCOME_VALUE, tail) = produced;
	lane.s(S_INCOME_COUNT) = tail + 1;
}

PVZ_SIM double IncomeRate(Lane lane)
{
	const int32_t tick = lane.s(S_TICK);
	const int32_t count = lane.s(S_INCOME_COUNT);
	double produced = 0.0;
	for (int32_t slot = 0; slot < count; ++slot)
		if (lane.i(F_INCOME_TICK, slot) >= tick - kIncomeWindowTicks) produced += lane.i(F_INCOME_VALUE, slot);
	const int32_t elapsed = ClampInt(tick - lane.s(S_INCOME_WINDOW_START), 0, kIncomeWindowTicks);
	return elapsed ? produced * 1000.0 / elapsed : 0.0;
}

// ---------------------------------------------------------------------------
// Student features — a transcription of EnvironmentRollout::Features (src/EnvironmentRollout.cpp:106)
// ---------------------------------------------------------------------------

// Activation layout for the batched network: index = (channel * kCells + cell) * stride + lane, so
// one warp reading one channel of one cell touches 128 contiguous bytes.
struct NetBuffers
{
	float* board;      // kBoardChannels * kCells * stride
	float* scalars;    // (kScalars + 1 + kZombieTypes) * stride
};

void BuildFeatures(Lane lane, const LevelInfo& level, const NetBuffers& buffers, const float* waveLog,
	bool previousValid);

// ---------------------------------------------------------------------------
// Import / export
// ---------------------------------------------------------------------------

// Transpose one exported position into this lane's SoA slots and reset the rollout bookkeeping.
// Returns 0 when the record holds something outside the kernel's coverage (the exporter marks those,
// and a batch built from a stale file is rejected lane by lane rather than silently mis-simulated).
int32_t LoadRoot(Lane lane, const RootRecord& root, const LevelInfo& level);

// Leaf statistics in the same units as the ROLLOUT response's "events" block.
void StoreResult(Lane lane, RolloutResult& out);

// ---------------------------------------------------------------------------
// One tick of the game, and the decision step around it (cuda/SimTick.inc)
// ---------------------------------------------------------------------------

struct Decision
{
	int32_t packet;               // -1 = wait
	int32_t row, col;
};

// Advance exactly one tick, in the original's order (see cuda/spec/protocol.md).
void UpdateTick(Lane lane, const LevelInfo& level, Rand& rand);

// Whether any card can be picked up and planted somewhere right now.  The rollout loop uses this to
// skip the network entirely when the answer is no, so it has to match EnvironmentAnyLegalPlant.
bool AnyLegalPlant(Lane lane, const LevelInfo& level);

// Planting, exactly as EnvironmentPlant does it: legality re-check, sun deduction, AddPlant state,
// card deactivated into its refresh clock.  Returns false when the action would have been rejected.
bool ActionPlant(Lane lane, const LevelInfo& level, int32_t packet, int32_t row, int32_t col, Rand& rand);

}  // namespace pvzc

#endif
