// Phase 0: the CUDA rollout core, run against real exported positions.
//
// One lane = one rollout = one continuation of a real position to the horizon.  Nothing about the game
// logic lives in this file: the tick, the features, the network arithmetic and the decision all come
// from the .inc sources that the host build compiles too, so a disagreement with the original
// simulator is a mechanics bug rather than "which rewrite is wrong".
//
// What *is* here is the schedule.  A lane's decision round is split across kernels so the trunk of the
// student network can run over the whole batch at once as strided-batched GEMMs:
//
//   InitKernel   one independent future per lane, its root transposed into the SoA
//   PreKernel    loop top + the short circuit; lanes that reach a forward are compacted into
//                activation slots 0..netCount-1 and write their own features there
//   [cuBLAS]     three convolutions over NetBlocks(netCount) chunks of 1024 lanes, bias and ReLU
//                folded by a pass per layer, then the cell head's 1x1 convolution (packet-independent)
//   PostKernel   global context, heads, the choice, and the rest of the round (the wait or the plant)
//   StoreKernel  leaf statistics
//
// The split is only equivalent to the single-threaded loop because nothing between PreKernel and
// PostKernel draws from a lane's RNG, and every kernel hands the stream back through the scalar block
// (cuda/SimCore.h's LoadRandState / StoreRandState).
//
// Build (no CMake needed for the measurement):
//   /usr/local/cuda/bin/nvcc -arch=sm_120 -O3 -std=c++17 --expt-relaxed-constexpr \
//       -I. -lcublas -o build/rollout_cuda cuda/RolloutCuda.cu
// Run:
//   ./build/rollout_cuda --roots artifacts/cuda/corpus_smoke/roots.cudaroot \
//                        --net artifacts/cuda/student/v14.pvzrn --lanes 21504 --repeat 224
//   ./build/rollout_cuda --roots ... --net ... --pairs pairs.txt --out leaves.bin --leaf-out leaves_feat.bin
//                        (the explicit workload: one lane per corpus line, for the phase-1 comparison)
#ifndef PVZ_ROLLOUT_CUDA_CU
#define PVZ_ROLLOUT_CUDA_CU

#include "SimCore.h"
#include "SimImpl.inc"
#include "SimTick.inc"
#include "NetForward.inc"
#include "Rollout.inc"
#include "PairList.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Only the server's front door uses these (cuda/spec/host.md "serve mode"): the kernels and the shared
// core never see a socket.
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Named rather than anonymous: nvcc's host-stub generator gives every anonymous namespace in this
// translation unit (the ones inside SimImpl.inc / SimTick.inc too) the same synthetic identifier, and
// a __global__ declared in one of them then fails to mangle.
namespace cudarun {

using namespace pvzc;

[[noreturn]] void Fail(const std::string& message)
{
	std::fprintf(stderr, "rollout-cuda: %s\n", message.c_str());
	std::exit(1);
}

void Checked(cudaError_t error, const char* what)
{
	if (error == cudaSuccess) return;
	std::string message = std::string(what) + ": " + cudaGetErrorString(error);
	Fail(message);
}

void Checked(cublasStatus_t status, const char* what)
{
	if (status == CUBLAS_STATUS_SUCCESS) return;
	Fail(std::string(what) + ": cublas status " + std::to_string(static_cast<int>(status)));
}

#define CALL(expression) Checked((expression), #expression)
#define CHECK_LAUNCH() Checked(cudaGetLastError(), "kernel launch")

// An asynchronous fault is reported by whichever call comes next, so SYNC=1 puts a device sync after
// every GEMM and every ReLU pass and makes the failing one name itself.
bool SyncEachCall()
{
	static const bool value = std::getenv("SYNC") != nullptr;
	return value;
}

// ---------------------------------------------------------------------------
// The corpus: one LEVEL record and N ROOT records, each written as the raw struct it mirrors
// ---------------------------------------------------------------------------

enum { kRecordLevel = 1, kRecordRoot = 2 };

struct Corpus
{
	LevelInfo level{};
	std::vector<RootRecord> roots;
	bool haveLevel = false;
};

Corpus ReadCorpus(const std::string& path)
{
	std::FILE* stream = std::fopen(path.c_str(), "rb");
	if (!stream) Fail("cannot open " + path);
	Corpus corpus;
	int32_t tag[3];
	while (std::fread(tag, sizeof(tag), 1, stream) == 1)
	{
		if (tag[0] != kMagic || tag[1] != kVersion)
			Fail("record tag " + std::to_string(tag[0]) + "/" + std::to_string(tag[1]) + " is not PVC"
				+ std::to_string(kVersion));
		if (tag[2] == kRecordLevel)
		{
			if (std::fread(&corpus.level, sizeof(LevelInfo), 1, stream) != 1) Fail("short level record");
			corpus.haveLevel = true;
		}
		else if (tag[2] == kRecordRoot)
		{
			RootRecord root;
			if (std::fread(&root, sizeof(RootRecord), 1, stream) != 1) Fail("short root record");
			// The exporter marks positions it cannot represent; a batch of them would measure an empty
			// kernel, so they are dropped here rather than simulated as no-ops.
			if (root.supported) corpus.roots.push_back(root);
		}
		else Fail("unknown record kind " + std::to_string(tag[2]));
	}
	std::fclose(stream);
	if (!corpus.haveLevel) Fail(path + " holds no level record");
	if (corpus.roots.empty()) Fail(path + " holds no supported roots");
	return corpus;
}

// ---------------------------------------------------------------------------
// The student dump, in EnvironmentRollout::Load's order
// ---------------------------------------------------------------------------

std::vector<size_t> WeightCounts(int32_t width, int32_t meanmax, int32_t withContext)
{
	const size_t w = static_cast<size_t>(width);
	const size_t context = (meanmax ? 3 : 2) * w;
	const size_t scalars = static_cast<size_t>(kScalars) + (withContext ? kTaskContext : 0);
	return {w * kBoardChannels * 9, w, w * w * 9, w, w * w * 5, w, w * scalars, w,
		context, 1, 10 * context, 10, 10 * w, w * (2 * w + context), w, w, 1};
}

struct NetImage
{
	std::vector<float> values;
	NetParams params{};
};

NetImage ReadNet(const std::string& path)
{
	std::FILE* stream = std::fopen(path.c_str(), "rb");
	if (!stream) Fail("cannot open " + path);
	char magic[9] = {0};
	uint32_t width = 0, pool = 0, withContext = 0;
	if (std::fread(magic, 8, 1, stream) != 1) Fail(path + " is too short");
	const std::string version(magic, 8);
	if (version != "PVZRN002" && version != "PVZRN003") Fail("unsupported network dump " + version);
	if (std::fread(&width, 4, 1, stream) != 1 || std::fread(&pool, 4, 1, stream) != 1) Fail("short header");
	if (version == "PVZRN003" && std::fread(&withContext, 4, 1, stream) != 1) Fail("short header");
	if (width < 1 || width > kMaxNetWidth) Fail("network width " + std::to_string(width) + " out of range");

	NetImage image;
	const std::vector<size_t> counts = WeightCounts(static_cast<int32_t>(width), pool ? 1 : 0,
		withContext ? 1 : 0);
	size_t total = 0;
	for (size_t count : counts) total += count;
	image.values.resize(total);
	size_t offset = 0;
	for (size_t count : counts)
	{
		if (std::fread(image.values.data() + offset, sizeof(float), count, stream) != count)
			Fail(path + " holds fewer weights than the header promises");
		offset += count;
	}
	if (std::fgetc(stream) != EOF) Fail(path + " holds trailing data");
	std::fclose(stream);

	// The tensors are laid out in the file order Load() reads them: conv1, bias1, conv2, bias2, conv3,
	// bias3, scalar, scalarBias, plant, plantBias, packet, packetBias, embedding, cell, cellBias,
	// cellOut, cellOutBias.  The host pointers are only ever read for their offsets into the uploaded
	// buffer, so the device side re-derives the same layout with the same cursor walk.
	const float* base = image.values.data();
	size_t cursor = 0;
	auto take = [base, &cursor](size_t count)
	{
		const float* pointer = base + cursor;
		cursor += count;
		return pointer;
	};
	NetParams& params = image.params;
	params.width = static_cast<int32_t>(width);
	params.meanmax = pool ? 1 : 0;
	params.taskContext = withContext ? 1 : 0;
	params.version = version == "PVZRN003" ? 3 : 2;
	params.contextSize = (params.meanmax ? 3 : 2) * params.width;
	params.scalarCount = kScalars + (params.taskContext ? kTaskContext : 0);
	params.conv1 = take(counts[0]);
	params.bias1 = take(counts[1]);
	params.conv2 = take(counts[2]);
	params.bias2 = take(counts[3]);
	params.conv3 = take(counts[4]);
	params.bias3 = take(counts[5]);
	params.scalarWeight = take(counts[6]);
	params.scalarBias = take(counts[7]);
	params.plantWeight = take(counts[8]);
	params.plantBias = take(counts[9]);
	params.packetWeight = take(counts[10]);
	params.packetBias = take(counts[11]);
	params.packetEmbedding = take(counts[12]);
	params.cellWeight = take(counts[13]);
	params.cellBias = take(counts[14]);
	params.cellOut = take(counts[15]);
	params.cellOutBias = take(counts[16]);
	if (cursor != total) Fail("weight counts do not add up");
	return image;
}

// The trunk's weights, re-laid out for the GEMMs.  The dump stores [outChannel][inChannel][tap] (what
// torch's Conv2d weight is, row major), while a column-major GEMM's B block for one tap has to be
// [outChannel][inChannel] with inChannel fastest -- see cuda/BlasProbe.cu for the same transposition.
// Only the batched path reads this buffer; NetForward.inc's per-lane loops keep the dump order, and
// the host build never sees a repacked pointer.  `tapCount` is 9 for a 3x3 (row major: dr then dc) and
// 5 for the 1x5, matching the shift tables below.
size_t PackedCount(int32_t inChannels, int32_t outChannels, int32_t tapCount)
{
	return static_cast<size_t>(tapCount) * inChannels * outChannels;
}

void RepackConv(float* destination, const float* source, int32_t inChannels, int32_t outChannels,
	int32_t tapCount)
{
	for (int32_t tap = 0; tap < tapCount; ++tap)
		for (int32_t oc = 0; oc < outChannels; ++oc)
			for (int32_t ic = 0; ic < inChannels; ++ic)
				destination[static_cast<size_t>(tap) * inChannels * outChannels + oc * inChannels + ic]
					= source[static_cast<size_t>(oc) * (inChannels * tapCount)
						+ static_cast<size_t>(ic) * tapCount + tap];
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// Benchmark modes.  The gate is a throughput number, so the driver has to be able to say *which half*
// of a continuation produced it: `sim` runs the same rounds with no forward at all.
//
// `lane` is a second schedule for the whole rollout, not a diagnostic split of the first one: one kernel
// launch, every lane running its own forward through NetForward.inc's per-lane loops.  It was written
// against the worry that a round of the batched schedule costs ~12 ms of pure launch latency at 1 lane
// (24 cuBLAS calls plus the readback), so a narrow batch might not amortize it.  Measured (rollouts/s,
// z13 corpus, post-fix binary): `full` 29/56/124/316 versus `lane` 1/2/3/8 at 10/30/100/300 lanes, i.e.
// the per-lane forward is 10x to 30x slower across the whole range the server sees -- one thread doing
// 64x64 convolutions cannot beat 84 SMs doing them in batches, whatever the launch latency.  There is no
// crossover to pick at: the serve path always uses the batched schedule, and `lane` is kept only to
// answer "what would it cost if the trunk were not batched".
enum BenchMode { kBenchFull = 0, kBenchSim = 1, kBenchLane = 2 };

// Per-lane state is ~17 KB, so a lane's kernels are limited by how many independent load chains the
// machine can keep in flight, not by arithmetic.  At 256 threads a block already fills an SM's register
// file (see kBlockThreads), so a wave of fewer than ~21,500 lanes leaves most of the 84 SMs idle: 1,000
// lanes is four blocks, and `heads+apply` over those four blocks costs 14.7 ms a round -- more than the
// same kernel over 27,200 lanes (9.2 ms).  Splitting the same lanes across narrow blocks puts them on
// more SMs.  Measured (rollouts/s, z13 corpus, `--bench full`):
//   lanes    10    50   100   150   256   512   1000  3000  8192  27200
//   256      26    80    116   153   236   463   831   2221  4514  15518
//   64       26    78    113   164   304   594   1123  2959  5558  15259
// The narrow block wins from ~150 lanes up (+7% .. +33%) and costs ~3% below it (a tiny wave pays the
// level-table staging per block, and 3% of nothing is nothing), so the wave size picks the block.
enum { kNarrowWaveLanes = 128 };
constexpr int32_t kNarrowBlock = 64;

// Occupancy control.  At 248 registers a thread the SM fits exactly one 256-thread block, i.e. 8 warps
// to cover a ~700-cycle global load; building with -DPVZ_MIN_BLOCKS=2 caps the budget at 128 registers
// and trades spills for warps.  --block may not exceed kBlockThreads because of this contract.
enum { kBlockThreads = 256 };
#ifndef PVZ_MIN_BLOCKS
#define PVZ_MIN_BLOCKS 1
#endif
enum { kMinBlocks = PVZ_MIN_BLOCKS };

// Per-batch bookkeeping in device memory.  `netCount` is memset at the top of every round (it is the
// atomicAdd target of the compaction); `running` is decremented by whichever kernel observes a lane
// reaching LANE_DONE, and the round loop stops when the host reads it back as zero.
struct Counters
{
	int32_t netCount;
	int32_t running;
};

struct Arguments
{
	int32_t* ints;
	float* floats;
	const LevelInfo* level;
	const RootRecord* roots;
	const WaveTable* waves;
	const float* waveLog;
	NetParams net;
	// The same weights in *dump* order.  The batched trunk reads the repacked buffer through
	// `net.conv1..3`, but NetForward.inc's per-lane loops index [outChannel][inChannel][tap], so the
	// one-launch schedule has to be handed the untouched pointers or every lane would convolve against
	// a transposed kernel.
	NetParams netLane;
	NetState state;
	RolloutResult* results;
	// Per lane, the activation slot RoundPre compacted it into, or -1 for a lane that did not reach a
	// forward this round.  PostKernel runs over *all* lanes and returns on -1: launching it over just the
	// compacted ones left 30 blocks for 84 SMs, and that phase was the most expensive half of a round.
	int32_t* laneSlot;
	Counters* counters;
	// The value net's encoding of each leaf (kValueFloats contiguous floats per lane), or null when
	// --leaf-out was not asked for.  Sized to one batch, not to the whole workload: it is written back
	// and streamed to the file batch by batch.
	float* leafFeatures;
	int32_t lanes;
	int32_t firstRollout;
	int32_t perRoot;
	int32_t perGame;                  // rootCount * perRoot; the rollout index wraps over repeats
	int32_t horizon;
	int32_t maxActions;
	int32_t seedBase;
	int32_t bench;                    // BenchMode
	// The explicit workload (cuda/PairList.h): when pairCount > 0 every lane takes its root and its
	// future from these arrays instead of from RolloutOf's formula, because a phase-1 comparison has to
	// run *the same* continuation the CPU reference ran.
	const int32_t* pairRoot;
	const int32_t* pairSeed;
	// Five int32 per pair -- kind, packet, row, col, ticks -- the teacher's candidate action for that
	// lane (cuda/spec/host.md finding 1).  Null when the workload names no actions, which is what the
	// phase-1 corpus files are.
	const int32_t* pairFirst;
	int32_t pairCount;
};

// rollout -> (root, future).  The workload holds `perGame = rootCount * perRoot` lanes and a benchmark
// may run several batches of them, but a phase-0 gate has to measure *distinct* continuations: keying
// the seed off `slot % perRoot` would replay the identical future on every repeat (the old 21,504
// rollouts were 96 distinct positions times 224 identical replays).  The root rotates over the corpus as
// before, while the future seed advances with the global rollout index, so pass k of a root continues it
// down a different RNG stream than pass k-1.
PVZ_SIM void RolloutOf(const Arguments& arguments, int32_t index, int32_t& rootIndex,
	int32_t& futureSeed, int32_t& pairSlot)
{
	pairSlot = -1;
	if (arguments.pairCount > 0)
	{
		const int32_t slot = (arguments.firstRollout + index) % arguments.pairCount;
		rootIndex = arguments.pairRoot[slot];
		futureSeed = arguments.pairSeed[slot];
		pairSlot = slot;
		return;
	}
	const int32_t global = arguments.firstRollout + index;
	const int32_t slot = global % arguments.perGame;
	const int32_t rootCount = arguments.perRoot > 0 ? arguments.perGame / arguments.perRoot : 1;
	rootIndex = rootCount > 0 ? slot / arguments.perRoot : 0;
	futureSeed = RolloutFutureSeed(global, rootCount > 0 ? rootCount : 1, arguments.perRoot,
		arguments.seedBase);
}

// The plant and zombie tables are scanned on the hot path (TickPlantDef runs for every live plant every
// tick), so the block stages them once in shared memory instead of broadcasting global loads.  Every
// thread of the block has to take part, including the ones past the lane count, so a kernel calls this
// before its first early return.  The array is the caller's __shared__ block: a __device__ function may
// not declare one itself.
__device__ __forceinline__ void StageLevel(const LevelInfo* source, int32_t* destination)
{
	const int32_t* words = reinterpret_cast<const int32_t*>(source);
	for (int32_t word = threadIdx.x; word < sizeof(LevelInfo) / sizeof(int32_t); word += blockDim.x)
		destination[word] = words[word];
	__syncthreads();
}

__global__ void __launch_bounds__(kBlockThreads, kMinBlocks) InitKernel(Arguments arguments)
{
	const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int32_t levelRaw[sizeof(LevelInfo) / sizeof(int32_t)];
	StageLevel(arguments.level, levelRaw);
	const LevelInfo& level = *reinterpret_cast<const LevelInfo*>(levelRaw);
	if (index >= arguments.lanes) return;

	int32_t rootIndex = 0, futureSeed = 0, pairSlot = 0;
	RolloutOf(arguments, index, rootIndex, futureSeed, pairSlot);

	Lane lane{arguments.ints, arguments.floats, index, arguments.lanes, 0};
	arguments.results[index] = RolloutResult{};
	arguments.results[index].rootIndex = rootIndex;
	// Written for every lane, before LoadRoot can fail and before RolloutBegin reads them: the scalar
	// block is reused across batches and --repeat, so a lane that inherits the previous batch's action
	// would plant the previous candidate.
	if (pairSlot >= 0 && arguments.pairFirst != nullptr)
	{
		const int32_t* words = arguments.pairFirst
			+ static_cast<size_t>(pairSlot) * pvzhost::kFirstActionWords;
		lane.s(S_FIRST_KIND) = words[0];
		lane.s(S_FIRST_PACKET) = words[1];
		lane.s(S_FIRST_ROW) = words[2];
		lane.s(S_FIRST_COL) = words[3];
		lane.s(S_FIRST_TICKS) = words[4];
	}
	else lane.s(S_FIRST_KIND) = 0;
	if (!LoadRoot(lane, arguments.roots[rootIndex], level))
	{
		// A lane that never started is not simulated and not stored; dropping it out of `running` is what
		// lets the round loop finish even when every lane of the batch came from a stale corpus file.
		arguments.results[index].errorFlags = ERR_BAD_ROOT;
		atomicSub(&arguments.counters->running, 1);
		return;
	}
	lane.s(S_ROOT_INDEX) = rootIndex;
	lane.s(S_UNTIL_TICK) = arguments.roots[rootIndex].tick + arguments.horizon;
	RolloutBegin(lane, futureSeed, arguments.maxActions, level, arguments.waves[rootIndex]);
	// A forced wait can reach the horizon inside InitKernel. PreKernel skips done lanes, so it
	// cannot retire this one later; leaving it in running would stall the entire batch forever.
	if (lane.s(S_STATE) == LANE_DONE) atomicSub(&arguments.counters->running, 1);
}

// Round semantics are shared by the SoA schedule and the narrow-wave shared-memory schedule.
__device__ __forceinline__ void PreLane(const Arguments& arguments, Lane lane,
	const LevelInfo& level, int32_t index)
{
	arguments.laneSlot[index] = -1;
	if (lane.s(S_STATE) == LANE_DONE) return;
	const WaveTable& waves = arguments.waves[lane.s(S_ROOT_INDEX)];
	Rand rand = LoadRandState(lane);
	const int32_t stage = RoundPre(lane, level, waves, rand);
	StoreRandState(lane, rand);
	if (stage == PRE_NEEDS_NET)
	{
		arguments.laneSlot[index] = atomicAdd(&arguments.counters->netCount, 1);
		lane.netSlot = arguments.laneSlot[index];
		BuildFeatures(lane, level, arguments.state, arguments.waveLog, lane.s(S_PRIOR_TICK) >= 0);
		return;
	}
	if (lane.s(S_STATE) == LANE_DONE) atomicSub(&arguments.counters->running, 1);
}

__device__ __forceinline__ void PostLane(const Arguments& arguments, Lane lane,
	const LevelInfo& level)
{
	const WaveTable& waves = arguments.waves[lane.s(S_ROOT_INDEX)];
	Rand rand = LoadRandState(lane);
	RoundPost(lane, level, waves, arguments.net, arguments.state, rand);
	StoreRandState(lane, rand);
	if (lane.s(S_STATE) == LANE_DONE) atomicSub(&arguments.counters->running, 1);
}

__global__ void __launch_bounds__(kBlockThreads, kMinBlocks) PreKernel(Arguments arguments)
{
	const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int32_t levelRaw[sizeof(LevelInfo) / sizeof(int32_t)];
	StageLevel(arguments.level, levelRaw);
	if (index >= arguments.lanes) return;
	Lane lane{arguments.ints, arguments.floats, index, arguments.lanes, 0};
	PreLane(arguments, lane, *reinterpret_cast<const LevelInfo*>(levelRaw), index);
}

__global__ void __launch_bounds__(kBlockThreads, kMinBlocks) PostKernel(Arguments arguments)
{
	const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int32_t levelRaw[sizeof(LevelInfo) / sizeof(int32_t)];
	StageLevel(arguments.level, levelRaw);
	if (index >= arguments.lanes || arguments.laneSlot[index] < 0) return;
	Lane lane{arguments.ints, arguments.floats, index, arguments.lanes, arguments.laneSlot[index]};
	PostLane(arguments, lane, *reinterpret_cast<const LevelInfo*>(levelRaw));
}

// A small SoA wave can occupy only one or two SMs. Give each continuation a block, cooperatively
// transpose its state to shared memory, and run the identical serial core on thread 0. This changes
// storage and placement, not mechanics, RNG order, or network arithmetic. Large waves retain SoA.
template <bool post, bool cooperative = false>
__global__ void PrivateRoundKernel(Arguments arguments)
{
	const int32_t index = blockIdx.x;
	if (index >= arguments.lanes) return;
	const int32_t slot = post ? arguments.laneSlot[index] : 0;
	if (post && slot < 0) return;
	__shared__ int32_t levelRaw[sizeof(LevelInfo) / sizeof(int32_t)];
	__shared__ int32_t ints[IntSlotsTotal()];
	__shared__ float floats[FloatSlotsTotal()];
	__shared__ Choice choice;
	for (int32_t word = threadIdx.x; word < IntSlotsTotal(); word += blockDim.x)
		ints[word] = arguments.ints[static_cast<size_t>(word) * arguments.lanes + index];
	for (int32_t word = threadIdx.x; word < FloatSlotsTotal(); word += blockDim.x)
		floats[word] = arguments.floats[static_cast<size_t>(word) * arguments.lanes + index];
	StageLevel(arguments.level, levelRaw);
	Lane lane{ints, floats, 0, 1, slot};
	const LevelInfo& level = *reinterpret_cast<const LevelInfo*>(levelRaw);
	if constexpr (post && cooperative)
	{
		// Parallelize independent channels/cells, preserving every sum's original operand order.
		for (int32_t oc = threadIdx.x; oc < arguments.net.width; oc += blockDim.x)
			NetContextChannel(lane, arguments.net, arguments.state, oc);
		__syncthreads();
		if (threadIdx.x == 0) choice = NetPickPacket(lane, level, arguments.net, arguments.state);
		__syncthreads();
		if (choice.packet >= 0)
		{
			for (int32_t oc = threadIdx.x; oc < arguments.net.width; oc += blockDim.x)
				CellLogitsChannel(lane, arguments.net, arguments.state, choice.packet, oc);
			__syncthreads();
			for (int32_t cell = threadIdx.x; cell < kLogitCells; cell += blockDim.x)
				CellLogitCell(lane, arguments.net, arguments.state, cell);
			__syncthreads();
			if (threadIdx.x == 0) choice = NetPickCell(lane, level, arguments.state, choice);
		}
		if (threadIdx.x == 0)
		{
			Rand rand = LoadRandState(lane);
			RoundApply(lane, level, arguments.waves[lane.s(S_ROOT_INDEX)], rand, choice);
			StoreRandState(lane, rand);
			if (lane.s(S_STATE) == LANE_DONE) atomicSub(&arguments.counters->running, 1);
		}
	}
	else if (threadIdx.x == 0)
	{
		if (post) PostLane(arguments, lane, level);
		else PreLane(arguments, lane, level, index);
	}
	__syncthreads();
	for (int32_t word = threadIdx.x; word < IntSlotsTotal(); word += blockDim.x)
		arguments.ints[static_cast<size_t>(word) * arguments.lanes + index] = ints[word];
	for (int32_t word = threadIdx.x; word < FloatSlotsTotal(); word += blockDim.x)
		arguments.floats[static_cast<size_t>(word) * arguments.lanes + index] = floats[word];
}

// Bias and ReLU over the valid cells of one layer's output.  SGEMM has no epilogue, so the bias that
// NetForward.inc's loops fold into the accumulator is applied here instead -- and only on valid cells,
// because the halo has to stay zero forever for the padding to work.
__global__ void ReluBiasKernel(float* plane, int32_t channels, const float* bias)
{
	const int32_t total = channels * kCells * kNetChunk;
	for (int32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
		index += blockDim.x * gridDim.x)
	{
		const int32_t cellSlot = index >> kNetChunkBits;
		const int32_t channel = cellSlot / kCells;
		const int32_t cell = cellSlot - channel * kCells;
		const size_t offset = (static_cast<size_t>(channel) * kPlaneCells + PaddedCell(cell)) * kNetChunk
			+ (index & kNetChunkMask);
		const float value = plane[offset] + bias[channel];
		plane[offset] = value > 0.0f ? value : 0.0f;
	}
}

// One rollout start to start.  With `--bench sim` it runs no forward at all, which is what measures the
// tick loop alone; with `--bench lane` it runs the per-lane forward, and then this single launch is the
// entire batch -- no round loop, no readback, no cuBLAS call.  Each lane addresses its activations by its
// own index, so the chunked layout keeps 30 concurrent lanes 30 separate planes apart.
__global__ void __launch_bounds__(kBlockThreads, kMinBlocks) RolloutKernel(Arguments arguments)
{
	const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int32_t levelRaw[sizeof(LevelInfo) / sizeof(int32_t)];
	StageLevel(arguments.level, levelRaw);
	const LevelInfo& level = *reinterpret_cast<const LevelInfo*>(levelRaw);
	if (index >= arguments.lanes) return;

	int32_t rootIndex = 0, futureSeed = 0, pairSlot = 0;
	RolloutOf(arguments, index, rootIndex, futureSeed, pairSlot);
	Lane lane{arguments.ints, arguments.floats, index, arguments.lanes, index};
	// InitKernel's failed root import leaves the SoA holding whatever the previous batch put there, and
	// unlike the round loop this schedule gets no second chance to notice.
	if (lane.s(S_STATE) == LANE_DONE || arguments.results[index].errorFlags & ERR_BAD_ROOT) return;
	const WaveTable& waves = arguments.waves[rootIndex];
	RunRollout(lane, level, waves, arguments.netLane, arguments.state, arguments.waveLog, futureSeed,
		arguments.maxActions, arguments.bench == kBenchSim ? 0 : 1);
	StoreResult(lane, arguments.results[index]);
}

__global__ void StoreKernel(Arguments arguments)
{
	const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int32_t levelRaw[sizeof(LevelInfo) / sizeof(int32_t)];
	StageLevel(arguments.level, levelRaw);
	const LevelInfo& level = *reinterpret_cast<const LevelInfo*>(levelRaw);
	if (index >= arguments.lanes) return;
	Lane lane{arguments.ints, arguments.floats, index, arguments.lanes, 0};
	float* leaf = arguments.leafFeatures ? arguments.leafFeatures + static_cast<size_t>(index) * kValueFloats
		: nullptr;
	if (arguments.results[index].errorFlags & ERR_BAD_ROOT)
	{
		// Zeroed rather than skipped: the buffer is reused by the next batch, so leaving the previous
		// batch's block here would put real features under a lane that never ran.
		if (leaf) for (int32_t place = 0; place < kValueFloats; ++place) leaf[place] = 0.0f;
		return;
	}
	StoreResult(lane, arguments.results[index]);
	// Written after StoreResult so a lane's aggregates and its features describe the same leaf.  The
	// per-lane block is contiguous (these are never read by a GEMM), and the reader pairs block i with
	// record i of --out.
	if (leaf) StoreLeafValueFeatures(lane, level, arguments.waveLog, leaf);
}

// ---------------------------------------------------------------------------
// The batched trunk
// ---------------------------------------------------------------------------

// One convolution as `tapCount` strided-batched GEMMs per lane chunk, accumulating into the same
// output with beta = 0 on the first tap.  cuBLAS is column-major, so the spatial index (10 valid
// columns x kNetChunk lanes, contiguous in the chunked layout) is the matrix row and the channel is
// the column; the 6 grid rows are the batch dimension, one padded grid row apart.
struct ConvLayer
{
	const float* input;
	float* output;
	const float* taps;                  // [tap][outChannel][inChannel], in the shift order below
	const float* bias;
	int32_t inChannels;
	int32_t outChannels;
	int32_t tapCount;
	const int32_t* shifts;              // padded cells from the chunk's valid origin
};

void RunConv(cublasHandle_t handle, const ConvLayer& layer, int32_t blocks)
{
	const int32_t leading = kPlaneCells * kNetChunk;
	const long long rowStride = static_cast<long long>(kColsP) * kNetChunk;
	const long long inLead = static_cast<long long>(layer.inChannels) * kPlaneCells * kNetChunk;
	const long long outLead = static_cast<long long>(layer.outChannels) * kPlaneCells * kNetChunk;
	const float one = 1.0f, zero = 0.0f;
	const int32_t reluTotal = layer.outChannels * kCells * kNetChunk;
	const int32_t reluBlocks = (reluTotal + kBlockThreads - 1) / kBlockThreads;
	for (int32_t block = 0; block < blocks; ++block)
	{
		const float* input = layer.input + static_cast<long long>(block) * inLead;
		float* output = layer.output + static_cast<long long>(block) * outLead;
		for (int32_t tap = 0; tap < layer.tapCount; ++tap)
		{
			const cublasStatus_t status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
				kCols * kNetChunk, layer.outChannels, layer.inChannels, &one,
				input + static_cast<size_t>(layer.shifts[tap]) * kNetChunk, leading, rowStride,
				layer.taps + static_cast<size_t>(tap) * layer.inChannels * layer.outChannels,
				layer.inChannels, 0LL,
				tap == 0 ? &zero : &one,
				output + static_cast<size_t>(kValidOrigin) * kNetChunk, leading, rowStride, kRows);
			if (status != CUBLAS_STATUS_SUCCESS)
				Fail("sgemm: cublas status " + std::to_string(static_cast<int>(status))
					+ " in tap " + std::to_string(tap) + " of block " + std::to_string(block));
			if (SyncEachCall())
			{
				CHECK_LAUNCH();
				CALL(cudaDeviceSynchronize());
			}
		}
		// One pass per layer over the chunk this block just wrote; the grid is capped because the loop
		// behind it is bandwidth-bound long before 4096 blocks.
		ReluBiasKernel<<<reluBlocks < 4096 ? reluBlocks : 4096, kBlockThreads>>>(output,
			layer.outChannels, layer.bias);
		CHECK_LAUNCH();
		if (SyncEachCall()) CALL(cudaDeviceSynchronize());
	}
}

// The cell head's 1x1 convolution, batched.  It does not depend on which seed card the plant head
// picks, so unlike the rest of the head it can run over every slot of the batch as one GEMM -- the same
// m / k / n / batch shape as a trunk tap, into `scratch`, whose conv2 result the third convolution has
// by now consumed.  beta = 0 and no bias pass: the head's bias is the packet embedding plus the
// context, per lane, and CellLogits adds it (cuda/NetForward.inc).
void RunCellPre(cublasHandle_t handle, const NetState& state, const NetParams& net, int32_t blocks)
{
	const int32_t leading = kPlaneCells * kNetChunk;
	const long long rowStride = static_cast<long long>(kColsP) * kNetChunk;
	const long long lead = static_cast<long long>(net.width) * kPlaneCells * kNetChunk;
	const int32_t step = 2 * net.width + net.contextSize;
	const float one = 1.0f, zero = 0.0f;
	for (int32_t block = 0; block < blocks; ++block)
	{
		const cublasStatus_t status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			kCols * kNetChunk, net.width, net.width, &one,
			state.hidden + static_cast<long long>(block) * lead + kValidOrigin * kNetChunk, leading,
			rowStride,
			net.cellWeight, step, 0LL, &zero,
			state.scratch + static_cast<long long>(block) * lead + kValidOrigin * kNetChunk, leading,
			rowStride, kRows);
		if (status != CUBLAS_STATUS_SUCCESS)
			Fail("sgemm: cublas status " + std::to_string(static_cast<int>(status))
				+ " in cell head block " + std::to_string(block));
		if (SyncEachCall())
		{
			CHECK_LAUNCH();
			CALL(cudaDeviceSynchronize());
		}
	}
}

// ---------------------------------------------------------------------------
// Driver: one device context, two front doors
// ---------------------------------------------------------------------------
//
// `main` either runs one workload described by the command line (the phase-0 gate, the phase-1 corpus
// comparison) or serves many small ones from a resident process (`--serve`, the search's continuation
// backend).  Both doors go through the same Device, the same kernels and the same RunBatch, so a
// measurement made through one is a measurement of the other's code path.
//
// Why a server is needed at all: a continuation costs ~49 rounds, and round 9 of
// ~/pvz-logs/cuda_status.md measured ~10 ms of *device* time per round whether the batch holds 10 lanes
// or 1000 (24 cuBLAS calls total 38 us of it; the rest is PreKernel/PostKernel latency with the machine
// nearly empty).  So the batched schedule needs ~100 lanes to match one CPU thread and ~300 to beat four.
// A search decision is 10-40 lanes, which only becomes a batch if positions from several games are held in
// flight together -- and that needs a process that outlives the request, because starting one costs 1.4 s,
// more than the decision it would serve.

struct Options
{
	std::string roots, net, out;
	// Value-net features for every leaf, in the same lane order as --out.  22.7 KB per lane, so it is
	// opt-in and streamed batch by batch rather than held for the whole workload.
	std::string leafOut;
	// One line per lane: `<rootIndex> <futureSeed>` (cuda/PairList.h).  Replaces the derived pairing.
	std::string pairs;
	std::string socket;               // --serve: the Unix socket to listen on
	int32_t perRoot = 8;
	int32_t horizon = 2500;
	int32_t maxActions = 3000;
	int32_t lanes = 8192;             // batch capacity, in lanes
	int32_t block = 0;                // 0 = choose from the wave size (see kNarrowWaveLanes)
	int32_t repeat = 1;
	int32_t seedBase = 1600000000;
	int32_t warm = 1;
	int32_t bench = kBenchFull;
	int32_t serve = 0;
	// How long the server waits for more requests before launching a wave.  A wave costs hundreds of ms,
	// so a modest window is cheap: merging four 30-lane requests into one 120-lane wave finishes the
	// slowest of them in ~800 ms instead of ~2000 ms.  The window closes early once the wave is half full.
	// How long a wave may be held open.  A wave costs its rounds (~50 x ~7 ms), so holding one for 80 ms to
	// merge a second worker into it is a trade the server wins whenever there is more than one worker; with
	// a single client it is pure latency, which is why `--gather-ms 0` is the honest benchmark setting.
	int32_t gatherMs = 80;
	// Give up rather than hold a few hundred MB of VRAM forever if no client ever appears.
	int32_t startTimeoutMs = 60000;
};

std::string Value(int argc, char** argv, int& index)
{
	if (++index >= argc) Fail("missing value for " + std::string(argv[index - 1]));
	return argv[index];
}

Options Parse(int argc, char** argv)
{
	Options options;
	for (int index = 1; index < argc; ++index)
	{
		const std::string flag = argv[index];
		if (flag == "--roots") options.roots = Value(argc, argv, index);
		else if (flag == "--net") options.net = Value(argc, argv, index);
		else if (flag == "--out") options.out = Value(argc, argv, index);
		else if (flag == "--leaf-out") options.leafOut = Value(argc, argv, index);
		else if (flag == "--pairs") options.pairs = Value(argc, argv, index);
		else if (flag == "--socket") options.socket = Value(argc, argv, index);
		else if (flag == "--serve") options.serve = 1;
		else if (flag == "--bench")
		{
			const std::string mode = Value(argc, argv, index);
			if (mode == "sim") options.bench = kBenchSim;
			else if (mode == "full") options.bench = kBenchFull;
			else if (mode == "lane") options.bench = kBenchLane;
			else Fail("--bench takes full|sim|lane");
		}
		else if (flag == "--per-root") options.perRoot = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--horizon") options.horizon = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--max-actions") options.maxActions = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--lanes") options.lanes = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--block") options.block = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--repeat") options.repeat = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--seed-base") options.seedBase = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--warm") options.warm = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--gather-ms") options.gatherMs = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--start-timeout-ms")
			options.startTimeoutMs = std::atoi(Value(argc, argv, index).c_str());
		else Fail("unknown flag " + flag);
	}
	if (options.net.empty() || (!options.serve && options.roots.empty()))
		Fail("usage: --net FILE (--roots FILE [--per-root 8] [--horizon 2500] [--lanes 8192] "
			"[--block 0] [--repeat 1] [--max-actions 3000] [--seed-base N] [--bench full|sim|lane] "
			"[--pairs FILE] [--out results.bin] [--leaf-out leaves.bin]) | (--serve --socket PATH)");
	if (options.lanes < 1 || options.block < 0 || options.block > kBlockThreads || options.perRoot < 1)
		Fail("lanes and per-root must be positive, --block in [0, 256] (0 picks the wave size)");
	// A leaf feature block is meaningless without the record it pairs with (the result and error flags
	// say which lanes never started), and --out is what carries them.
	if (!options.leafOut.empty() && options.out.empty()) Fail("--leaf-out needs --out");
	if (!options.leafOut.empty() && options.bench == kBenchSim)
		Fail("--leaf-out needs a schedule that runs the network (--bench full|lane)");
	if (options.serve && options.socket.empty()) Fail("--serve needs --socket PATH");
	return options;
}

// ---------------------------------------------------------------------------
// The device, sized once by the batch capacity
// ---------------------------------------------------------------------------

// One batch's statistics, so both doors can report the same numbers.
struct BatchStats
{
	int32_t rounds = 0;
	int32_t maxNet = 0;
	float milliseconds = 0.0f;
	float milliPre = 0.0f, milliConv = 0.0f, milliPost = 0.0f;
};

struct Device
{
	int32_t lanes = 0;                       // capacity, in lanes
	int32_t blocks = 0;                       // NetBlocks(capacity)
	int32_t width = 0;
	int32_t options_block = 0;                // 0 = pick per wave (RunBatch)
	int32_t bench = kBenchFull;
	int32_t horizon = 2500;
	int32_t maxActions = 3000;
	int32_t perRoot = 8;
	int32_t perGame = 0;
	int32_t seedBase = 1600000000;

	int32_t* ints = nullptr;
	float* floats = nullptr;
	LevelInfo* level = nullptr;
	RootRecord* roots = nullptr;
	WaveTable* waves = nullptr;
	float* waveLog = nullptr;
	float* netBase = nullptr;
	float* packedBase = nullptr;
	float* board = nullptr;
	float* scalarPlane = nullptr;
	float* hidden = nullptr;
	float* scratch = nullptr;
	float* context = nullptr;
	float* logits = nullptr;
	int32_t* laneSlot = nullptr;
	Counters* counters = nullptr;
	RolloutResult* results = nullptr;
	float* leafFeatures = nullptr;
	int32_t* pairRoot = nullptr;
	int32_t* pairSeed = nullptr;
	int32_t* pairFirst = nullptr;
	size_t pairCapacity = 0;
	int32_t rootCapacity = 0;

	cublasHandle_t handle = nullptr;
	int32_t shifts3[9];
	int32_t shifts1x5[5];
	NetParams net{};
	NetParams netLane{};
	NetState state{};
	ConvLayer conv1{nullptr, nullptr, nullptr, nullptr, 0, 0, 0, nullptr};
	ConvLayer conv2{nullptr, nullptr, nullptr, nullptr, 0, 0, 0, nullptr};
	ConvLayer conv3{nullptr, nullptr, nullptr, nullptr, 0, 0, 0, nullptr};
	Arguments arguments{};

	cudaEvent_t start = nullptr, stop = nullptr, phaseStart[3] = {}, phaseStop[3] = {};
	int32_t* hostCounters = nullptr;         // pinned, 2 words
	std::vector<RolloutResult> hostResults;
	std::vector<float> hostLeaf;
};

// The root and pair arrays are re-uploaded per wave by the server and once by the batch door, so growing
// them never has to preserve anything -- and must not be called between an upload and the batch that
// reads it.
void GrowRoots(Device& dev, int32_t count)
{
	if (count <= dev.rootCapacity) return;
	const int32_t capacity = count < 256 ? 256 : count;
	if (dev.roots) CALL(cudaFree(dev.roots));
	if (dev.waves) CALL(cudaFree(dev.waves));
	CALL(cudaMalloc(&dev.roots, static_cast<size_t>(capacity) * sizeof(RootRecord)));
	CALL(cudaMalloc(&dev.waves, static_cast<size_t>(capacity) * sizeof(WaveTable)));
	dev.rootCapacity = capacity;
	// The kernels reach both arrays through `arguments`, which DeviceOpen filled from the pointers that
	// this just freed.  Refresh them here or every grow leaves the round loop reading the old mapping:
	// the host writes the new one, so the lanes simulate against whatever the reused address holds.  On
	// the 425-root z15 corpus that was a garbage wave table for most roots and an illegal access in
	// UpdateWaves for root 35 (compute-sanitizer, 2026-10-18); corpora of 256 roots or fewer never grow
	// the arrays, which is why the phase-1 comparison over the z13 corpus did not show it.
	dev.arguments.roots = dev.roots;
	dev.arguments.waves = dev.waves;
}

void GrowPairs(Device& dev, size_t count)
{
	if (count <= dev.pairCapacity) return;
	const size_t capacity = count < 4096 ? 4096 : count;
	if (dev.pairRoot) CALL(cudaFree(dev.pairRoot));
	if (dev.pairSeed) CALL(cudaFree(dev.pairSeed));
	if (dev.pairFirst) CALL(cudaFree(dev.pairFirst));
	CALL(cudaMalloc(&dev.pairRoot, capacity * sizeof(int32_t)));
	CALL(cudaMalloc(&dev.pairSeed, capacity * sizeof(int32_t)));
	CALL(cudaMalloc(&dev.pairFirst, capacity * pvzhost::kFirstActionWords * sizeof(int32_t)));
	dev.pairCapacity = capacity;
	dev.arguments.pairRoot = dev.pairRoot;
	dev.arguments.pairSeed = dev.pairSeed;
	dev.arguments.pairFirst = dev.pairFirst;
}

void DeviceOpen(Device& dev, const Options& options, const NetImage& image)
{
	const int32_t lanes = options.lanes;
	const int32_t width = image.params.width;
	dev.lanes = lanes;
	dev.blocks = NetBlocks(lanes);
	dev.width = width;
	dev.options_block = options.block;
	dev.bench = options.bench;
	dev.horizon = options.horizon;
	dev.maxActions = options.maxActions;
	dev.perRoot = options.perRoot;
	dev.seedBase = options.seedBase;

	// The SoA is sized by the batch, not by the rollout count: every batch reuses the same buffers, so the
	// working set stays under the 8 GB budget the task sets for this machine.  Activations are sized in
	// whole chunks of kNetChunk lanes, padded grid included -- 112 cells per plane instead of 60, and the
	// halo is what lets a filter tap be a constant pointer offset.
	const size_t chunkFloats = static_cast<size_t>(kNetChunk) * kPlaneCells * dev.blocks;
	const size_t vectorFloats = static_cast<size_t>(kNetChunk) * dev.blocks;
	const size_t intBytes = static_cast<size_t>(IntSlotsTotal()) * lanes * sizeof(int32_t);
	const size_t floatBytes = static_cast<size_t>(FloatSlotsTotal()) * lanes * sizeof(float);
	const size_t boardBytes = kBoardChannels * chunkFloats * sizeof(float);
	const size_t planeBytes = static_cast<size_t>(width) * chunkFloats * sizeof(float);
	const size_t scalarBytes = kScalarFeatures * vectorFloats * sizeof(float);
	const size_t contextBytes = static_cast<size_t>(image.params.contextSize) * vectorFloats * sizeof(float);
	const size_t logitsBytes = kLogitCells * vectorFloats * sizeof(float);
	const size_t activationBytes = boardBytes + 2 * planeBytes + scalarBytes + contextBytes + logitsBytes;
	const size_t weightsBytes = image.values.size() * sizeof(float);
	// The server always takes leaves (that is how the search scores a continuation without paying for an
	// observation round trip), so the buffer is part of its capacity rather than an opt-in.
	const size_t leafBytes = options.serve || !options.leafOut.empty()
		? static_cast<size_t>(lanes) * kValueFloats * sizeof(float) : 0u;

	// Three trunk layers, one after another in the order the shift tables walk their taps.
	const size_t packedCount = PackedCount(kBoardChannels, width, 9)
		+ PackedCount(width, width, 9) + PackedCount(width, width, 5);
	std::vector<float> packed(packedCount);
	{
		float* cursor = packed.data();
		RepackConv(cursor, image.params.conv1, kBoardChannels, width, 9);
		cursor += PackedCount(kBoardChannels, width, 9);
		RepackConv(cursor, image.params.conv2, width, width, 9);
		cursor += PackedCount(width, width, 9);
		RepackConv(cursor, image.params.conv3, width, width, 5);
		if (cursor + PackedCount(width, width, 5) != packed.data() + packedCount)
			Fail("packed trunk size does not add up");
	}
	const size_t total_bytes = intBytes + floatBytes + activationBytes + weightsBytes
		+ packedCount * 4 + static_cast<size_t>(lanes) * (sizeof(RolloutResult) + sizeof(int32_t))
		+ leafBytes;
	std::fprintf(stderr, "batch lanes %d in %d chunks of %d, device working set %.0f MB (limit 8192 MB)\n",
		lanes, dev.blocks, kNetChunk, static_cast<double>(total_bytes) / 1048576.0);
	if (total_bytes > 8ull * 1024 * 1024 * 1024) Fail("device working set above the 8 GB budget");

	CALL(cudaSetDevice(0));
	CALL(cudaMalloc(&dev.ints, intBytes));
	CALL(cudaMalloc(&dev.floats, floatBytes));
	CALL(cudaMalloc(&dev.level, sizeof(LevelInfo)));
	CALL(cudaMalloc(&dev.waveLog, (kMaxWaves + 1) * sizeof(float)));
	CALL(cudaMalloc(&dev.netBase, weightsBytes));
	CALL(cudaMalloc(&dev.packedBase, packed.size() * sizeof(float)));
	CALL(cudaMalloc(&dev.board, boardBytes));
	CALL(cudaMalloc(&dev.scalarPlane, scalarBytes));
	CALL(cudaMalloc(&dev.hidden, planeBytes));
	CALL(cudaMalloc(&dev.scratch, planeBytes));
	CALL(cudaMalloc(&dev.context, contextBytes));
	CALL(cudaMalloc(&dev.logits, logitsBytes));
	CALL(cudaMalloc(&dev.laneSlot, static_cast<size_t>(lanes) * sizeof(int32_t)));
	CALL(cudaMalloc(&dev.counters, sizeof(Counters)));
	CALL(cudaMalloc(&dev.results, static_cast<size_t>(lanes) * sizeof(RolloutResult)));
	GrowRoots(dev, 1);
	GrowPairs(dev, 1);
	if (leafBytes)
	{
		CALL(cudaMalloc(&dev.leafFeatures, leafBytes));
		// Zeroed once: a lane that failed its root import is skipped by StoreKernel, and a dump whose
		// untouched blocks held whatever malloc returned would not be comparable between runs.
		CALL(cudaMemset(dev.leafFeatures, 0, leafBytes));
	}
	CALL(cudaMemcpy(dev.netBase, image.values.data(), weightsBytes, cudaMemcpyHostToDevice));
	CALL(cudaMemcpy(dev.packedBase, packed.data(), packed.size() * sizeof(float), cudaMemcpyHostToDevice));
	// The halo of every plane is zero from here on and nothing ever writes it, which is what makes the
	// padding free: a source cell off the lawn is a halo cell holding 0.
	CALL(cudaMemset(dev.board, 0, boardBytes));
	CALL(cudaMemset(dev.hidden, 0, planeBytes));
	CALL(cudaMemset(dev.scratch, 0, planeBytes));

	std::vector<float> hostWaveLog(kMaxWaves + 1);
	for (int32_t index = 0; index <= kMaxWaves; ++index)
		hostWaveLog[index] = static_cast<float>(std::log1p(static_cast<double>(index))
			/ std::log1p(50.0));
	CALL(cudaMemcpy(dev.waveLog, hostWaveLog.data(), hostWaveLog.size() * sizeof(float),
		cudaMemcpyHostToDevice));

	// The device NetParams point at the uploaded weight buffer at the offsets the file order implies.
	NetParams deviceNet = image.params;
	{
		size_t place = 0;
		auto take = [&dev, &place](size_t count)
		{
			float* pointer = dev.netBase + place;
			place += count;
			return pointer;
		};
		const std::vector<size_t> counts = WeightCounts(deviceNet.width, deviceNet.meanmax,
			deviceNet.taskContext);
		deviceNet.conv1 = take(counts[0]);
		deviceNet.bias1 = take(counts[1]);
		deviceNet.conv2 = take(counts[2]);
		deviceNet.bias2 = take(counts[3]);
		deviceNet.conv3 = take(counts[4]);
		deviceNet.bias3 = take(counts[5]);
		deviceNet.scalarWeight = take(counts[6]);
		deviceNet.scalarBias = take(counts[7]);
		deviceNet.plantWeight = take(counts[8]);
		deviceNet.plantBias = take(counts[9]);
		deviceNet.packetWeight = take(counts[10]);
		deviceNet.packetBias = take(counts[11]);
		deviceNet.packetEmbedding = take(counts[12]);
		deviceNet.cellWeight = take(counts[13]);
		deviceNet.cellBias = take(counts[14]);
		deviceNet.cellOut = take(counts[15]);
		deviceNet.cellOutBias = take(counts[16]);
	}
	// The batched trunk reads its weights repacked; NetForward.inc's per-lane loops want the dump order.
	// `netLane` is the copy from before the replacement, which is what `--bench lane` runs on.
	dev.netLane = deviceNet;
	deviceNet.conv1 = dev.packedBase;
	deviceNet.conv2 = dev.packedBase + PackedCount(kBoardChannels, width, 9);
	deviceNet.conv3 = dev.packedBase + PackedCount(kBoardChannels, width, 9) + PackedCount(width, width, 9);
	dev.net = deviceNet;
	dev.state = NetState{dev.board, dev.scalarPlane, dev.hidden, dev.scratch, dev.context, dev.logits};

	CALL(cublasCreate(&dev.handle));
	// TF32 tensor cores would roughly double the trunk, and change the rounding.  Phase 1 compares
	// decisions against the CPU forward, so reduced precision is a measured opt-in, not a default.
	CALL(cublasSetMathMode(dev.handle, CUBLAS_DEFAULT_MATH));

	for (int32_t dr = 0; dr < 3; ++dr)
		for (int32_t dc = 0; dc < 3; ++dc)
			dev.shifts3[dr * 3 + dc] = kValidOrigin + (dr - 1) * kColsP + (dc - 1);
	for (int32_t dc = 0; dc < 5; ++dc) dev.shifts1x5[dc] = kValidOrigin + (dc - 2);
	// ConvLayer keeps a pointer to the shift table, so the tables have to live as long as the device.
	dev.conv1 = ConvLayer{dev.board, dev.hidden, dev.net.conv1, dev.net.bias1, kBoardChannels, width, 9,
		dev.shifts3};
	dev.conv2 = ConvLayer{dev.hidden, dev.scratch, dev.net.conv2, dev.net.bias2, width, width, 9,
		dev.shifts3};
	dev.conv3 = ConvLayer{dev.scratch, dev.hidden, dev.net.conv3, dev.net.bias3, width, width, 5,
		dev.shifts1x5};

	Arguments& arguments = dev.arguments;
	arguments.ints = dev.ints;
	arguments.floats = dev.floats;
	arguments.level = dev.level;
	arguments.roots = dev.roots;
	arguments.waves = dev.waves;
	arguments.waveLog = dev.waveLog;
	arguments.net = dev.net;
	arguments.netLane = dev.netLane;
	arguments.state = dev.state;
	arguments.results = dev.results;
	arguments.laneSlot = dev.laneSlot;
	arguments.counters = dev.counters;
	arguments.leafFeatures = dev.leafFeatures;
	arguments.pairRoot = dev.pairRoot;
	arguments.pairSeed = dev.pairSeed;
	arguments.pairFirst = dev.pairFirst;

	CALL(cudaEventCreate(&dev.start));
	CALL(cudaEventCreate(&dev.stop));
	for (int32_t index = 0; index < 3; ++index)
	{
		CALL(cudaEventCreate(&dev.phaseStart[index]));
		CALL(cudaEventCreate(&dev.phaseStop[index]));
	}
	CALL(cudaMallocHost(&dev.hostCounters, 2 * sizeof(int32_t)));
	dev.hostResults.resize(static_cast<size_t>(lanes));
	if (dev.leafFeatures) dev.hostLeaf.resize(static_cast<size_t>(lanes) * kValueFloats);
}

// The wave tables come from the roots themselves: the future the original already committed to, indexed
// by root, so 200 roots cost 200 tables instead of one per lane.
std::vector<WaveTable> BuildWaves(const std::vector<RootRecord>& roots)
{
	std::vector<WaveTable> tables(roots.size());
	for (size_t index = 0; index < roots.size(); ++index)
	{
		const RootRecord& root = roots[index];
		WaveTable& table = tables[index];
		table.waveCount = root.waveCount;
		for (int32_t wave = 0; wave < kMaxWaves; ++wave)
		{
			table.zombieCount[wave] = root.waveZombieCount[wave];
			for (int32_t slot = 0; slot < kWaveZombieSlots; ++slot)
				table.type[wave][slot] = root.waveTable[wave][slot];
		}
	}
	return tables;
}

// Upload `roots` at `base` (which must fit the grown capacity) and return nothing: the caller has already
// translated its pair indices.  `dev.uploadedRoots` is the running cursor the server advances per request.
void UploadRoots(Device& dev, int32_t base, const std::vector<RootRecord>& roots,
	const std::vector<WaveTable>& waves)
{
	if (roots.empty()) return;
	CALL(cudaMemcpy(dev.roots + base, roots.data(), roots.size() * sizeof(RootRecord),
		cudaMemcpyHostToDevice));
	CALL(cudaMemcpy(dev.waves + base, waves.data(), waves.size() * sizeof(WaveTable),
		cudaMemcpyHostToDevice));
}

// One batch of at most `dev.lanes` lanes, from InitKernel to StoreKernel, timed with events.  The round
// loop's structure is the phase-0 one and unchanged: the only difference between the doors is who decides
// what the lanes mean.
BatchStats RunBatch(Device& dev, int32_t firstRollout, int32_t batchLanes)
{
	Arguments& arguments = dev.arguments;
	BatchStats stats;
	const bool phaseTiming = std::getenv("PHASE") != nullptr;
	const char* privateLimitText = std::getenv("PVZ_CUDA_PRIVATE_LIMIT");
	const int32_t privateLimit = privateLimitText ? std::atoi(privateLimitText) : 0;
	const bool privateWave = batchLanes <= privateLimit;
	const bool cooperative = std::getenv("PVZ_CUDA_COOPERATIVE_HEADS") != nullptr;
	const char* privateThreadsText = std::getenv("PVZ_CUDA_PRIVATE_THREADS");
	const int32_t privateThreads = privateThreadsText ? std::atoi(privateThreadsText) : 128;
	if (privateThreads != 32 && privateThreads != 64 && privateThreads != 128 && privateThreads != 256)
		Fail("PVZ_CUDA_PRIVATE_THREADS must be 32, 64, 128, or 256");
	// The round loop reaches the root, wave-table and pair arrays through `arguments`, and GrowRoots /
	// GrowPairs move them, so a grow that did not refresh the copy would leave the host writing one
	// buffer while the kernels read a freed one.  Whether that reads as garbage or as the same bytes
	// depends on which virtual addresses the allocator hands back, so it can pass a test by luck: check
	// it instead of trusting the coincidence (one of the captured pointers did go stale, 2026-10-18).
	if (arguments.roots != dev.roots || arguments.waves != dev.waves
		|| (arguments.pairCount > 0 && (arguments.pairRoot != dev.pairRoot
			|| arguments.pairSeed != dev.pairSeed || arguments.pairFirst != dev.pairFirst)))
		Fail("the batch would read a root or pair array that a grow moved without refreshing Arguments");
	arguments.lanes = batchLanes;
	arguments.firstRollout = firstRollout;
	arguments.perGame = dev.perGame;
	arguments.perRoot = dev.perRoot;
	arguments.seedBase = dev.seedBase;
	arguments.horizon = dev.horizon;
	arguments.maxActions = dev.maxActions;
	arguments.bench = dev.bench;
	const Counters initial{0, batchLanes};
	CALL(cudaMemcpy(dev.counters, &initial, sizeof(Counters), cudaMemcpyHostToDevice));
	CALL(cudaEventRecord(dev.start));

	// The block is chosen from the wave that is about to run, not from the capacity the device was opened
	// with: a served wave of 300 lanes and a benchmark wave of 27,200 share this code path and want
	// opposite settings (see kNarrowWaveLanes).  It only moves threads between launches -- the lane-to-
	// thread map is `blockIdx * blockDim + threadIdx` either way, so no decision changes.
	const int32_t block = dev.options_block > 0 ? dev.options_block
		: static_cast<int32_t>(batchLanes >= kNarrowWaveLanes ? kNarrowBlock : kBlockThreads);
	const int32_t grid = (batchLanes + block - 1) / block;
	InitKernel<<<grid, block>>>(arguments);
	CHECK_LAUNCH();
	if (dev.bench != kBenchFull)
	{
		RolloutKernel<<<grid, block>>>(arguments);
		CHECK_LAUNCH();
		if (dev.bench == kBenchLane)
		{
			StoreKernel<<<grid, block>>>(arguments);
			CHECK_LAUNCH();
		}
	}
	else
	{
		// A round applies at most one action, and RoundTop ends a lane at maxActions of them, so the loop
		// provably drains every lane within maxActions + 1 rounds.  The guard turns a broken S_STATE into a
		// stalled run with a message instead of an infinite launch sequence.
		const int32_t roundGuard = dev.maxActions + 8;
		int32_t round = 0;
		for (; round < roundGuard; ++round)
		{
			CALL(cudaEventRecord(dev.phaseStart[0]));
			CALL(cudaMemsetAsync(dev.counters, 0, sizeof(int32_t)));     // netCount; running carries over
			if (privateWave) PrivateRoundKernel<false><<<batchLanes, privateThreads>>>(arguments);
			else PreKernel<<<grid, block>>>(arguments);
			CHECK_LAUNCH();
			CALL(cudaEventRecord(dev.phaseStop[0]));
			// One 8-byte readback per round, and the only host/device synchronisation the loop needs
			// without the phase timer: it is what lets the batch stop as soon as the last lane reaches
			// its horizon.
			CALL(cudaMemcpyAsync(dev.hostCounters, dev.counters, 2 * sizeof(int32_t),
				cudaMemcpyDeviceToHost));
			CALL(cudaStreamSynchronize(0));
			if (dev.hostCounters[1] == 0) break;
			const int32_t netCount = dev.hostCounters[0];
			if (netCount > stats.maxNet) stats.maxNet = netCount;
			const int32_t netBlocks = NetBlocks(netCount);
			CALL(cudaEventRecord(dev.phaseStart[1]));
			RunConv(dev.handle, dev.conv1, netBlocks);
			RunConv(dev.handle, dev.conv2, netBlocks);
			RunConv(dev.handle, dev.conv3, netBlocks);
			RunCellPre(dev.handle, dev.state, dev.net, netBlocks);
			CALL(cudaEventRecord(dev.phaseStop[1]));
			CALL(cudaEventRecord(dev.phaseStart[2]));
			// Over all lanes, not over the compacted ones: a round that needs a forward for 7,500 of
			// 21,504 lanes would otherwise launch 30 blocks into 84 SMs, and heads+apply was then the
			// most expensive third of the round.  The threads that find -1 retire immediately.
			if (privateWave && cooperative) PrivateRoundKernel<true, true><<<batchLanes, privateThreads>>>(arguments);
			else if (privateWave) PrivateRoundKernel<true><<<batchLanes, privateThreads>>>(arguments);
			else PostKernel<<<grid, block>>>(arguments);
			CHECK_LAUNCH();
			CALL(cudaEventRecord(dev.phaseStop[2]));
			if (phaseTiming)
			{
				CALL(cudaEventSynchronize(dev.phaseStop[2]));
				float elapsed = 0.0f;
				for (int32_t phase = 0; phase < 3; ++phase)
				{
					CALL(cudaEventElapsedTime(&elapsed, dev.phaseStart[phase], dev.phaseStop[phase]));
					if (phase == 0) stats.milliPre += elapsed;
					else if (phase == 1) stats.milliConv += elapsed;
					else stats.milliPost += elapsed;
				}
			}
		}
		if (round >= roundGuard) Fail("round guard hit: lanes are not draining");
		stats.rounds = round;
		StoreKernel<<<grid, block>>>(arguments);
		CHECK_LAUNCH();
	}
	CALL(cudaEventRecord(dev.stop));
	CALL(cudaEventSynchronize(dev.stop));
	CALL(cudaEventElapsedTime(&stats.milliseconds, dev.start, dev.stop));
	CALL(cudaMemcpy(dev.hostResults.data(), dev.results, static_cast<size_t>(batchLanes)
		* sizeof(RolloutResult), cudaMemcpyDeviceToHost));
	return stats;
}

// The leaf blocks of the batch just run, `batchLanes` rows of kValueFloats.  Empty when the device was
// opened without them.
const std::vector<float>& BatchLeaves(Device& dev, int32_t batchLanes)
{
	if (!dev.leafFeatures) dev.hostLeaf.clear();
	else
	{
		CALL(cudaMemcpy(dev.hostLeaf.data(), dev.leafFeatures,
			static_cast<size_t>(batchLanes) * kValueFloats * sizeof(float), cudaMemcpyDeviceToHost));
	}
	return dev.hostLeaf;
}

// ---------------------------------------------------------------------------
// Door one: one workload from the command line
// ---------------------------------------------------------------------------

int RunWorkload(const Options& options)
{
	Corpus corpus = ReadCorpus(options.roots);
	NetImage image = ReadNet(options.net);
	const int32_t rootCount = static_cast<int32_t>(corpus.roots.size());
	std::vector<int32_t> pairRoot, pairSeed, pairFirstFlat;
	if (!options.pairs.empty())
	{
		std::vector<pvzhost::FirstAction> pairFirst;
		std::string error;
		if (!pvzhost::ReadPairList(options.pairs, pairRoot, pairSeed, pairFirst, error)) Fail(error);
		pvzhost::FlattenFirstActions(pairFirst, pairFirstFlat);
		for (size_t index = 0; index < pairRoot.size(); ++index)
			if (pairRoot[index] >= rootCount)
				Fail("pair " + std::to_string(index) + " names root " + std::to_string(pairRoot[index])
					+ " but the corpus holds " + std::to_string(rootCount));
	}
	// --per-root and --repeat describe the *derived* pairing; an enumerated corpus already says which
	// future belongs to which root, so the workload is exactly its line count.
	const int32_t perGame = pairRoot.empty() ? rootCount * options.perRoot
		: static_cast<int32_t>(pairRoot.size());
	const int32_t total = pairRoot.empty() ? perGame * options.repeat : perGame;

	Device dev;
	DeviceOpen(dev, options, image);
	GrowRoots(dev, rootCount);
	GrowPairs(dev, pairRoot.size());
	UploadRoots(dev, 0, corpus.roots, BuildWaves(corpus.roots));
	CALL(cudaMemcpy(dev.level, &corpus.level, sizeof(LevelInfo), cudaMemcpyHostToDevice));
	dev.perGame = perGame;
	dev.arguments.pairRoot = pairRoot.empty() ? nullptr : dev.pairRoot;
	dev.arguments.pairSeed = pairRoot.empty() ? nullptr : dev.pairSeed;
	dev.arguments.pairFirst = pairRoot.empty() ? nullptr : dev.pairFirst;
	dev.arguments.pairCount = static_cast<int32_t>(pairRoot.size());
	if (!pairRoot.empty())
	{
		const size_t bytes = pairRoot.size() * sizeof(int32_t);
		CALL(cudaMemcpy(dev.pairRoot, pairRoot.data(), bytes, cudaMemcpyHostToDevice));
		CALL(cudaMemcpy(dev.pairSeed, pairSeed.data(), bytes, cudaMemcpyHostToDevice));
		CALL(cudaMemcpy(dev.pairFirst, pairFirstFlat.data(),
			pairFirstFlat.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
	}

	std::vector<RolloutResult> collected;
	if (!options.out.empty()) collected.resize(static_cast<size_t>(total));
	std::FILE* leafStream = nullptr;
	if (!options.leafOut.empty())
	{
		leafStream = std::fopen(options.leafOut.c_str(), "wb");
		if (!leafStream) Fail("cannot write " + options.leafOut);
		// "PVZLEAF1", int32 lane count, int32 floats per lane, then one block per lane in the same order
		// as --out.  Which blocks are real is only readable from those records: a lane that failed its
		// root import is skipped by StoreKernel and stays all zero here.
		const char magic[8] = { 'P', 'V', 'Z', 'L', 'E', 'A', 'F', '1' };
		const int32_t counts[2] = { total, kValueFloats };
		std::fwrite(magic, 1, sizeof(magic), leafStream);
		std::fwrite(counts, sizeof(int32_t), 2, leafStream);
	}

	int64_t ticks = 0, decisions = 0, netCalls = 0, planted = 0, eaten = 0, killed = 0, errors = 0;
	int64_t rounds = 0;
	int32_t maxNet = 0, won = 0, lost = 0, ongoing = 0, completed = 0;
	float milliseconds = 0.0f, milliPre = 0.0f, milliConv = 0.0f, milliPost = 0.0f;
	const int32_t batchCount = (total + dev.lanes - 1) / dev.lanes;
	for (int32_t batch = -options.warm; batch < batchCount; ++batch)
	{
		const int32_t first = batch < 0 ? 0 : batch * dev.lanes;
		// A batch never launches more lanes than there are rollouts left: a lane whose rollout duplicates
		// one already in the batch would do real work that the rollouts/s figure could not count.
		const int32_t batchLanes = dev.lanes < total - first ? dev.lanes : total - first;
		// The warm-up batch runs the same round loop; counting its rounds would divide the timed work by a
		// larger denominator than it did.
		if (batch == 0) { rounds = 0; maxNet = 0; milliPre = milliConv = milliPost = 0.0f; }
		const BatchStats stats = RunBatch(dev, first, batchLanes);
		if (batch < 0) continue;                       // the timed run starts after the warm-up
		milliseconds += stats.milliseconds;
		rounds += stats.rounds;
		if (stats.maxNet > maxNet) maxNet = stats.maxNet;
		milliPre += stats.milliPre;
		milliConv += stats.milliConv;
		milliPost += stats.milliPost;
		for (int32_t index = 0; index < batchLanes; ++index)
		{
			const RolloutResult& result = dev.hostResults[index];
			ticks += result.ticks;
			decisions += result.decisions;
			netCalls += result.netCalls;
			planted += result.plantsPlanted;
			eaten += result.plantsEaten;
			killed += result.zombiesKilled;
			errors += result.errorFlags != 0 ? 1 : 0;
			if (result.result > 0) ++won;
			else if (result.result < 0) ++lost;
			else ++ongoing;
			if (!options.out.empty()) collected[static_cast<size_t>(first + index)] = result;
		}
		if (leafStream)
		{
			// hostLeaf is sized to the lane *capacity*, so the count has to come from the batch, not from the
			// vector, or a short last batch writes the previous batch's leftovers behind the header's count.
			const std::vector<float>& leaves = BatchLeaves(dev, batchLanes);
			const size_t floats = static_cast<size_t>(batchLanes) * kValueFloats;
			if (std::fwrite(leaves.data(), sizeof(float), floats, leafStream) != floats)
				Fail("short write to " + options.leafOut);
		}
		completed += batchLanes;
		if (batchCount > 1) std::printf("  batch %d/%d done\n", batch + 1, batchCount);
	}

	if (!pairRoot.empty())
		std::printf("\n%d rollouts from %s (explicit root/future pairs), horizon %d ticks, max-actions %d\n",
			total, options.pairs.c_str(), options.horizon, options.maxActions);
	else
		std::printf("\nroots %d x %d futures x %d repeats = %d rollouts, horizon %d ticks, max-actions %d\n",
			rootCount, options.perRoot, options.repeat, total, options.horizon, options.maxActions);
	std::printf("student %s: width %d, %s, %s context, %zu weights\n", image.params.version == 3 ? "003" : "002",
		image.params.width, image.params.meanmax ? "meanmax" : "mean",
		image.params.taskContext ? "with" : "without", image.values.size());
	std::printf("wall %.0f ms for %d rollouts -> %.0f rollouts/s, %.2f Gticks/s\n", milliseconds, completed,
		completed * 1000.0 / milliseconds, static_cast<double>(ticks) / milliseconds / 1.0e6);
	std::printf("per rollout: %.1f ticks, %.1f decisions, %.1f net forwards, %.1f planted, %.1f eaten, "
		"%.1f killed\n",
		static_cast<double>(ticks) / completed, static_cast<double>(decisions) / completed,
		static_cast<double>(netCalls) / completed,
		static_cast<double>(planted) / completed, static_cast<double>(eaten) / completed,
		static_cast<double>(killed) / completed);
	if (options.bench == kBenchFull)
	{
		std::printf("rounds %lld, lanes per forward round %.0f (peak %d of %d)\n",
			static_cast<long long>(rounds), static_cast<double>(netCalls) / (rounds ? rounds : 1),
			maxNet, dev.lanes);
		if (milliPre + milliConv + milliPost > 0.0f)
		{
			const float accounted = milliPre + milliConv + milliPost;
			std::printf("per round of %.1f ms: sim %.1f, trunk %.1f, heads+apply %.1f, sync and launch %.1f\n",
				rounds ? milliseconds / rounds : 0.0f, rounds ? milliPre / rounds : 0.0f,
				rounds ? milliConv / rounds : 0.0f, rounds ? milliPost / rounds : 0.0f,
				rounds ? (milliseconds - accounted) / rounds : 0.0f);
		}
	}
	std::printf("leaf results: won %d, lost %d, ongoing %d; lanes with an error flag %lld\n", won, lost,
		ongoing, static_cast<long long>(errors));
	std::printf("gate: CPU baseline 90.18/s/thread x 20 = 1804/s, needed 5x = 9018/s -> %.2fx\n",
		completed * 1000.0 / milliseconds / 9018.0);

	if (!options.out.empty())
	{
		std::FILE* stream = std::fopen(options.out.c_str(), "wb");
		if (!stream) Fail("cannot write " + options.out);
		std::fwrite(collected.data(), sizeof(RolloutResult), collected.size(), stream);
		std::fclose(stream);
		std::printf("wrote %zu leaf records to %s\n", collected.size(), options.out.c_str());
	}
	if (leafStream)
	{
		std::fclose(leafStream);
		std::printf("wrote %d leaf feature blocks of %d floats (%.1f MB) to %s\n", total, kValueFloats,
			static_cast<double>(total) * kValueFloats * sizeof(float) / 1048576.0,
			options.leafOut.c_str());
	}
	cublasDestroy(dev.handle);
	return 0;
}

// ---------------------------------------------------------------------------
// Door two: the resident server
// ---------------------------------------------------------------------------
//
// Protocol, all of it on one Unix stream socket per client:
//
//   RUN horizon=<n> max_actions=<n> leaf=<0|1> pairs=<n> roots=<path>\n
//     followed by pairs * 7 little-endian int32: root, futureSeed, kind, packet, row, col, ticks
//     (the same seven columns cuda/PairList.h reads from a file; kind 0 lets the student decide)
//   -> OK <lanes> <resultBytes> <leafBytes>\n then the records, then the leaf blocks in lane order
//   PING\n -> PONG <capacity> <lanesDone> <wavesDone>\n
//   QUIT\n -> BYE\n
//   Bad request or unreadable corpus -> ERR <one-line reason>\n
//
// The roots and pairs travel as a path plus an inline array because the position is written by the
// simulator (CUDA_DUMP) and cannot be re-expressed here, while the pairs are the caller's own arithmetic.
// Nothing about a request's *content* is trusted beyond what a file read already validates: the corpus
// path is opened, parsed and range-checked before a single lane runs, and a lane whose root import fails
// comes back with ERR_BAD_ROOT in its record rather than as a crash.

struct Client
{
	int fd = -1;
	std::string input;
};

struct Request
{
	int fd = -1;
	std::string rootsPath;
	int32_t horizon = 2500;
	int32_t maxActions = 3000;
	int32_t leaf = 1;
	std::vector<int32_t> pairRoot, pairSeed, pairFirst;
	std::vector<RootRecord> roots;
	std::vector<WaveTable> waves;
	LevelInfo level{};
	int32_t bad = 0;                     // the request was rejected; answer once, drop it
	size_t next = 0;                     // lanes already computed
	std::vector<RolloutResult> results;
	std::vector<float> leaves;
};

std::string Field(const std::string& line, const std::string& key)
{
	const size_t at = line.find(key);
	if (at == std::string::npos) return std::string();
	size_t start = at + key.size();
	const size_t end = line.find(' ', start);
	if (end == std::string::npos) return line.substr(start);
	return line.substr(start, end - start);
}

// Everything after `roots=` to the end of the line: a path may contain spaces, so it is the last field.
std::string RootField(const std::string& line)
{
	const size_t at = line.find("roots=");
	if (at == std::string::npos) return std::string();
	std::string path = line.substr(at + 6);
	while (!path.empty() && (path.back() == '\n' || path.back() == '\r' || path.back() == ' '))
		path.pop_back();
	return path;
}

bool SendAll(int fd, const char* data, size_t size)
{
	while (size)
	{
		const ssize_t sent = ::send(fd, data, size, MSG_NOSIGNAL);
		if (sent <= 0)
		{
			if (sent < 0 && (errno == EINTR || errno == EAGAIN)) continue;
			return false;
		}
		data += sent;
		size -= static_cast<size_t>(sent);
	}
	return true;
}

// A non-exiting corpus read: the server answers a bad request, it does not take the machine down with it.
bool TryReadCorpus(const std::string& path, Corpus& corpus, std::string& error)
{
	std::FILE* stream = std::fopen(path.c_str(), "rb");
	if (!stream) { error = "cannot open " + path; return false; }
	int32_t tag[3];
	while (std::fread(tag, sizeof(tag), 1, stream) == 1)
	{
		if (tag[0] != kMagic || tag[1] != kVersion)
		{
			error = "record tag is not a position of this build";
			std::fclose(stream);
			return false;
		}
		if (tag[2] == kRecordLevel)
		{
			if (std::fread(&corpus.level, sizeof(LevelInfo), 1, stream) != 1)
			{ error = "short level record"; std::fclose(stream); return false; }
			corpus.haveLevel = true;
		}
		else if (tag[2] == kRecordRoot)
		{
			RootRecord root;
			if (std::fread(&root, sizeof(RootRecord), 1, stream) != 1)
			{ error = "short root record"; std::fclose(stream); return false; }
			if (root.supported) corpus.roots.push_back(root);
		}
		else { error = "unknown record kind"; std::fclose(stream); return false; }
	}
	std::fclose(stream);
	if (!corpus.haveLevel) { error = path + " holds no level record"; return false; }
	if (corpus.roots.empty()) { error = path + " holds no supported roots"; return false; }
	return true;
}

// Answer a finished request: the header line, then its records, then its leaf blocks in lane order.
// Callers erase in reverse index order, so a queue index read before this call is still the right one.
void CompleteRequest(std::vector<Request>& queue, size_t index)
{
	Request& request = queue[index];
	const size_t leafBytes = request.leaf ? request.leaves.size() * sizeof(float) : 0;
	const std::string header = "OK " + std::to_string(request.pairRoot.size()) + " "
		+ std::to_string(request.results.size() * sizeof(RolloutResult)) + " "
		+ std::to_string(leafBytes) + "\n";
	const char* records = reinterpret_cast<const char*>(request.results.data());
	const char* leaves = reinterpret_cast<const char*>(request.leaves.data());
	if (!SendAll(request.fd, header.data(), header.size())
		|| !SendAll(request.fd, records, request.results.size() * sizeof(RolloutResult))
		|| (leafBytes && !SendAll(request.fd, leaves, leafBytes)))
		std::fprintf(stderr, "serve: client %d went away mid-answer\n", request.fd);
	queue.erase(queue.begin() + index);
}

int Serve(const Options& options)
{
	NetImage image = ReadNet(options.net);
	Device dev;
	DeviceOpen(dev, options, image);
	// Every request brings its own level and roots, so nothing is uploaded here; the batch fields the
	// derived pairing would need are irrelevant when every lane is enumerated by a pair.
	dev.arguments.pairRoot = dev.pairRoot;
	dev.arguments.pairSeed = dev.pairSeed;
	dev.arguments.pairFirst = dev.pairFirst;
	dev.perGame = 0;

	::unlink(options.socket.c_str());          // a stale socket from a killed server is ours to replace
	int listener = ::socket(AF_UNIX, SOCK_STREAM, 0);
	if (listener < 0) Fail("socket: " + std::string(std::strerror(errno)));
	sockaddr_un address{};
	address.sun_family = AF_UNIX;
	if (options.socket.size() >= sizeof(address.sun_path))
		Fail("socket path too long: " + options.socket);
	std::strncpy(address.sun_path, options.socket.c_str(), sizeof(address.sun_path) - 1);
	if (::bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0)
		Fail("bind " + options.socket + ": " + std::strerror(errno));
	if (::listen(listener, 16) != 0) Fail("listen: " + std::string(std::strerror(errno)));
	// The client reads this line to know the server is up (and to learn the lane capacity, which caps how
	// wide a request it may send in one go).
	std::printf("READY %s %d\n", options.socket.c_str(), dev.lanes);
	std::fflush(stdout);

	// Per-wave tracing for the service-shape measurements: which requests merged into one wave and
	// what that wave cost.  Off by default; the numbers only go to stderr when asked for.
	const bool serveTrace = std::getenv("PVZ_SERVE_TRACE") != nullptr;
	std::vector<Client> clients;
	std::vector<Request> queue;
	int64_t requests = 0, waves = 0, lanesDone = 0;
	double gpuMs = 0.0;
	int64_t roundsDone = 0;
	bool running = true;
	// Whether any client has ever connected.  Before that the server is waiting for its launcher and
	// gives up on a timeout; after that it lives until the last worker goes away.
	bool greeted = false;
	// When the oldest unserved request arrived; see the gather window in the poll timeout.
	std::chrono::steady_clock::time_point queueSince{};
	// A blocking worker cannot produce another request until its queued one is answered. Once every
	// connected worker is waiting, a gather timeout can only add latency; there is nobody left to gather.
	auto allClientsWaiting = [&clients, &queue]()
	{
		bool any = false;
		for (const Client& client : clients)
		{
			if (client.fd < 0) continue;
			any = true;
			bool waiting = false;
			for (const Request& request : queue)
				if (request.fd == client.fd && request.next < request.pairRoot.size()) { waiting = true; break; }
			if (!waiting) return false;
		}
		return any;
	};

	while (running)
	{
		std::vector<pollfd> fds;
		fds.push_back(pollfd{listener, POLLIN, 0});
		for (const Client& client : clients) fds.push_back(pollfd{client.fd, POLLIN, 0});
		// The gather window: a wave is worth holding together only while more requests may still arrive
		// inside it.  Holding costs the waiting worker at most `--gather-ms`, and buys a whole wave --
		// ~50 rounds of ~7 ms, which is what the window is measured against (round 9 of
		// ~/pvz-logs/cuda_status.md: a wave costs its rounds, not its lanes).  Workers that were answered
		// by the same wave come back within tens of milliseconds of each other, so the window is what
		// turns "one request per worker per decision" into "one wave per decision for all of them".
		const auto now = std::chrono::steady_clock::now();
		if (queue.empty()) queueSince = std::chrono::steady_clock::time_point();
		const int64_t held = queueSince == std::chrono::steady_clock::time_point()
			? 0 : std::chrono::duration_cast<std::chrono::milliseconds>(now - queueSince).count();
		int32_t queued = 0;
		for (const Request& request : queue)
			queued += static_cast<int32_t>(request.pairRoot.size()) - static_cast<int32_t>(request.next);
		// A full house does not need to wait for more.
		const bool packed = queued * 2 >= dev.lanes || (!queue.empty() && allClientsWaiting());
		// `--gather-ms 0` means "do not wait", but an empty queue still needs a bounded poll or the server
		// spins a core between requests.
		const int32_t idle = options.gatherMs < 1 ? 1 : options.gatherMs;
		const int32_t timeout = !queue.empty()
			? (packed || held >= options.gatherMs ? 0 : options.gatherMs - static_cast<int32_t>(held))
			: (greeted ? idle : options.startTimeoutMs);
		const int ready = ::poll(fds.data(), fds.size(), timeout);
		if (ready < 0 && errno != EINTR) Fail("poll: " + std::string(std::strerror(errno)));
		int32_t parsed = 0;

		if (ready > 0)
		{
			if (fds[0].revents & POLLIN)
			{
				const int fd = ::accept(listener, nullptr, nullptr);
				if (fd >= 0)
				{
					clients.push_back(Client{});
					clients.back().fd = fd;
					greeted = true;
					// The first client to connect clears the way for the start-up timeout to stop counting;
					// after that an idle server whose clients have all gone is an orphan and gives its
					// VRAM back.
					std::fprintf(stderr, "serve: client %d connected\n", fd);
				}
			}
			for (size_t c = 1; c < fds.size(); ++c)
			{
				Client& client = clients[c - 1];
				if (client.fd < 0) continue;
				if (!(fds[c].revents & POLLIN) && !(fds[c].revents & (POLLHUP | POLLERR))) continue;
				char buffer[4096];
				const ssize_t got = ::recv(client.fd, buffer, sizeof(buffer), 0);
				if (got <= 0)
				{
					// The worker exited (normally or not).  Its queued lanes are dropped; a wave may still
					// be holding a slice of them, which is harmless: nobody is reading that answer.
					if (client.fd >= 0) ::close(client.fd);
					client.fd = -1;
					continue;
				}
				client.input.append(buffer, static_cast<size_t>(got));
				for (;;)
				{
					const size_t newline = client.input.find('\n');
					if (newline == std::string::npos) break;
					const std::string line = client.input.substr(0, newline);
					if (line.rfind("PING", 0) == 0)
					{
						const std::string answer = "PONG " + std::to_string(dev.lanes) + " "
							+ std::to_string(lanesDone) + " " + std::to_string(waves) + "\n";
						SendAll(client.fd, answer.data(), answer.size());
						client.input.erase(0, newline + 1);
						continue;
					}
					if (line.rfind("QUIT", 0) == 0)
					{
						const std::string answer = "BYE\n";
						SendAll(client.fd, answer.data(), answer.size());
						::close(client.fd);
						client.fd = -1;
						client.input.clear();
						break;
					}
					if (line.rfind("RUN", 0) != 0)
					{
						const std::string answer = "ERR unknown command\n";
						SendAll(client.fd, answer.data(), answer.size());
						client.input.erase(0, newline + 1);
						continue;
					}
					const std::string pairsText = Field(line, "pairs=");
					const int32_t pairs = pairsText.empty() ? -1 : std::atoi(pairsText.c_str());
					if (pairs < 1 || pairs > 1 << 20)
					{
						const std::string answer = "ERR bad pairs count\n";
						SendAll(client.fd, answer.data(), answer.size());
						client.input.erase(0, newline + 1);
						continue;
					}
					const size_t payloadBytes = static_cast<size_t>(pairs) * 7 * sizeof(int32_t);
					if (client.input.size() < newline + 1 + payloadBytes) break;    // incomplete: wait
					Request request;
					request.fd = client.fd;
					request.horizon = std::atoi(Field(line, "horizon=").c_str());
					request.maxActions = std::atoi(Field(line, "max_actions=").c_str());
					const std::string leafText = Field(line, "leaf=");
					request.leaf = leafText.empty() ? 1 : std::atoi(leafText.c_str());
					request.rootsPath = RootField(line);
					if (request.horizon < 1) request.horizon = 2500;
					if (request.maxActions < 1) request.maxActions = 3000;
					const char* payload = client.input.data() + newline + 1;
					std::vector<int32_t> words(static_cast<size_t>(pairs) * 7);
					std::memcpy(words.data(), payload, payloadBytes);
					client.input.erase(0, newline + 1 + payloadBytes);
					for (int32_t index = 0; index < pairs; ++index)
					{
						request.pairRoot.push_back(words[index * 7]);
						request.pairSeed.push_back(words[index * 7 + 1]);
						request.pairFirst.push_back(words[index * 7 + 2]);
						request.pairFirst.push_back(words[index * 7 + 3]);
						request.pairFirst.push_back(words[index * 7 + 4]);
						request.pairFirst.push_back(words[index * 7 + 5]);
						request.pairFirst.push_back(words[index * 7 + 6]);
					}
					std::string error;
					Corpus corpus;
					if (!TryReadCorpus(request.rootsPath, corpus, error))
						request.bad = 1;
					else
					{
						request.roots = corpus.roots;
						request.level = corpus.level;
						request.waves = BuildWaves(request.roots);
						for (int32_t index = 0; index < pairs; ++index)
							if (request.pairRoot[index] < 0
								|| request.pairRoot[index] >= static_cast<int32_t>(request.roots.size()))
							{ request.bad = 1; break; }
					}
					if (request.bad)
					{
						const std::string answer = "ERR " + (error.empty() ? "pair names an unknown root"
							: error) + "\n";
						SendAll(request.fd, answer.data(), answer.size());
					}
					else
					{
						request.results.resize(static_cast<size_t>(pairs));
						if (request.leaf) request.leaves.resize(static_cast<size_t>(pairs) * kValueFloats);
						if (queue.empty()) queueSince = std::chrono::steady_clock::now();
						queue.push_back(std::move(request));
					}
					++requests;
					parsed = 1;
				}
			}
		}
		if (ready == 0 && !greeted)
		{
			std::fprintf(stderr, "serve: no client appeared within %d ms, exiting\n",
				options.startTimeoutMs);
			break;
		}

		// Hold the window open for a request that arrived during *this* poll.  The timeout above was
		// computed from the queue as it looked before parsing, so without this a fresh queue spends its
		// window on nothing and runs one wave per request (measured 2026-10-18: 506 requests, 506 waves,
		// 9 lanes per wave with two workers).  Waiting is bounded: the next poll's timeout is the window
		// that is left, so either another worker lands inside it or the wave runs at `--gather-ms`.
		if (parsed && !queue.empty() && options.gatherMs > 0)
		{
			const int64_t age = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::steady_clock::now() - queueSince).count();
			int32_t waiting = 0;
			for (const Request& request : queue)
				waiting += static_cast<int32_t>(request.pairRoot.size()) - static_cast<int32_t>(request.next);
			if (age < options.gatherMs && waiting * 2 < dev.lanes && !allClientsWaiting()) continue;
		}

		// Serve every queued request in waves of at most the lane capacity.  Requests are only merged when
		// they agree on horizon, max-actions and level, because those three are per-batch constants of the
		// kernels (S_UNTIL_TICK is derived from the horizon, and the level is staged once per block).
		while (!queue.empty())
		{
			struct Slice { size_t request; int32_t offset; int32_t lanes; int32_t lane; };
			std::vector<Slice> slices;
			const Request& head = queue.front();
			int32_t used = 0;
			int32_t waveRoots = 0;
			int32_t waveLeaf = 0;
			std::vector<int32_t> base(queue.size(), -1);
			for (size_t index = 0; index < queue.size(); ++index)
			{
				const Request& request = queue[index];
				if (request.horizon != head.horizon || request.maxActions != head.maxActions
					|| std::memcmp(&request.level, &head.level, sizeof(LevelInfo)) != 0) continue;
				const int32_t left = static_cast<int32_t>(request.pairRoot.size()) - static_cast<int32_t>(request.next);
				if (left <= 0) continue;
				if (base[index] < 0)
				{
					if (waveRoots + static_cast<int32_t>(request.roots.size()) > dev.rootCapacity)
						GrowRoots(dev, waveRoots + static_cast<int32_t>(request.roots.size()));
					base[index] = waveRoots;
					waveRoots += static_cast<int32_t>(request.roots.size());
				}
				const int32_t take = dev.lanes - used < left ? dev.lanes - used : left;
				// `lane` is where this slice starts in the wave's packed lane numbering, `offset` where it
				// starts in the request's own list.  They only coincide for the first slice of the first
				// request, and reading results by `offset` instead silently pairs a candidate with another
				// worker's continuation (caught 2026-10-18: two merged games both lost that both won alone).
				slices.push_back(Slice{index, static_cast<int32_t>(request.next), take, used});
				if (request.leaf) waveLeaf = 1;
				used += take;
				if (used >= dev.lanes) break;
			}
			if (slices.empty())
			{
				// Nothing in the queue can run together yet (mixed keys, or every request is finished).
				// Finish the completed ones and wait for the rest.
				for (size_t index = queue.size(); index-- > 0;)
					if (queue[index].next == queue[index].pairRoot.size())
						CompleteRequest(queue, index);
				if (queue.empty()) break;
				break;
			}
			GrowPairs(dev, static_cast<size_t>(used));
			std::vector<int32_t> hostPairRoot, hostPairSeed, hostPairFirst;
			hostPairRoot.reserve(used);
			hostPairSeed.reserve(used);
			hostPairFirst.reserve(static_cast<size_t>(used) * pvzhost::kFirstActionWords);
			for (const Slice& slice : slices)
			{
				Request& request = queue[slice.request];
				UploadRoots(dev, base[slice.request], request.roots, request.waves);
				for (int32_t lane = 0; lane < slice.lanes; ++lane)
				{
					const size_t from = static_cast<size_t>(slice.offset) + lane;
					hostPairRoot.push_back(request.pairRoot[from] + base[slice.request]);
					hostPairSeed.push_back(request.pairSeed[from]);
					for (int32_t word = 0; word < pvzhost::kFirstActionWords; ++word)
						hostPairFirst.push_back(request.pairFirst[from * pvzhost::kFirstActionWords + word]);
				}
			}
			CALL(cudaMemcpy(dev.pairRoot, hostPairRoot.data(), hostPairRoot.size() * sizeof(int32_t),
				cudaMemcpyHostToDevice));
			CALL(cudaMemcpy(dev.pairSeed, hostPairSeed.data(), hostPairSeed.size() * sizeof(int32_t),
				cudaMemcpyHostToDevice));
			CALL(cudaMemcpy(dev.pairFirst, hostPairFirst.data(), hostPairFirst.size() * sizeof(int32_t),
				cudaMemcpyHostToDevice));
			dev.horizon = head.horizon;
			dev.maxActions = head.maxActions;
			CALL(cudaMemcpy(dev.level, &head.level, sizeof(LevelInfo), cudaMemcpyHostToDevice));
			dev.arguments.pairCount = used;
			const BatchStats stats = RunBatch(dev, 0, used);
			if (waveLeaf) BatchLeaves(dev, used);
			if (serveTrace)
				std::fprintf(stderr, "serve-trace: wave %lld lanes %d requests %zu rounds %d ms %.1f\n",
					static_cast<long long>(waves), used, slices.size(), stats.rounds, stats.milliseconds);
			++waves;
			lanesDone += used;
			gpuMs += stats.milliseconds;
			roundsDone += stats.rounds;
			for (const Slice& slice : slices)
			{
				Request& request = queue[slice.request];
				std::memcpy(request.results.data() + request.next, dev.hostResults.data() + slice.lane,
					static_cast<size_t>(slice.lanes) * sizeof(RolloutResult));
				// Kernels index the concatenated root buffer; the client names roots in its own corpus.
				for (int32_t lane = 0; lane < slice.lanes; ++lane)
					request.results[request.next + lane].rootIndex = request.pairRoot[request.next + lane];
				if (request.leaf)
					std::memcpy(request.leaves.data() + request.next * kValueFloats,
						dev.hostLeaf.data() + static_cast<size_t>(slice.lane) * kValueFloats,
						static_cast<size_t>(slice.lanes) * kValueFloats * sizeof(float));
				request.next += static_cast<size_t>(slice.lanes);
			}
			for (size_t index = queue.size(); index-- > 0;)
				if (queue[index].next == queue[index].pairRoot.size())
					CompleteRequest(queue, index);
		}
		for (size_t index = clients.size(); index-- > 0;)
			if (clients[index].fd < 0) clients.erase(clients.begin() + index);
		if (greeted && clients.empty())
		{
			std::fprintf(stderr, "serve: all clients gone, exiting\n");
			break;
		}
	}
	// What the gather window bought: lanes per wave is the batching the search could not get on its own.
	std::fprintf(stderr, "serve: %lld requests in %lld waves, %lld lanes, %.0f ms of GPU time, %.1f ms per "
		"round over %lld rounds, %.0f lanes per wave\n", static_cast<long long>(requests),
		static_cast<long long>(waves), static_cast<long long>(lanesDone), gpuMs,
		roundsDone ? gpuMs / roundsDone : 0.0, static_cast<long long>(roundsDone),
		waves ? static_cast<double>(lanesDone) / waves : 0.0);
	::close(listener);
	::unlink(options.socket.c_str());
	cublasDestroy(dev.handle);
	return 0;
}

}  // namespace cudarun

using namespace cudarun;
using namespace pvzc;

int main(int argc, char** argv)
{
	Options options = Parse(argc, argv);
	return options.serve ? Serve(options) : RunWorkload(options);
}

#endif
