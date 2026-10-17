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
#ifndef PVZ_ROLLOUT_CUDA_CU
#define PVZ_ROLLOUT_CUDA_CU

#include "SimCore.h"
#include "SimImpl.inc"
#include "SimTick.inc"
#include "NetForward.inc"
#include "Rollout.inc"

#include <cublas_v2.h>
#include <cuda_runtime.h>

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
enum BenchMode { kBenchFull = 0, kBenchSim = 1 };

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
	NetState state;
	RolloutResult* results;
	// Per lane, the activation slot RoundPre compacted it into, or -1 for a lane that did not reach a
	// forward this round.  PostKernel runs over *all* lanes and returns on -1: launching it over just the
	// compacted ones left 30 blocks for 84 SMs, and that phase was the most expensive half of a round.
	int32_t* laneSlot;
	Counters* counters;
	int32_t lanes;
	int32_t firstRollout;
	int32_t perRoot;
	int32_t perGame;                  // rootCount * perRoot; the rollout index wraps over repeats
	int32_t horizon;
	int32_t maxActions;
	int32_t seedBase;
	int32_t bench;                    // BenchMode
};

// rollout -> (root, future).  The workload holds `perGame = rootCount * perRoot` lanes and a benchmark
// may run several batches of them, but a phase-0 gate has to measure *distinct* continuations: keying
// the seed off `slot % perRoot` would replay the identical future on every repeat (the old 21,504
// rollouts were 96 distinct positions times 224 identical replays).  The root rotates over the corpus as
// before, while the future seed advances with the global rollout index, so pass k of a root continues it
// down a different RNG stream than pass k-1.
PVZ_SIM void RolloutOf(const Arguments& arguments, int32_t index, int32_t& rootIndex,
	int32_t& futureSeed)
{
	const int32_t global = arguments.firstRollout + index;
	const int32_t slot = global % arguments.perGame;
	const int32_t rootCount = arguments.perRoot > 0 ? arguments.perGame / arguments.perRoot : 1;
	rootIndex = rootCount > 0 ? slot / arguments.perRoot : 0;
	futureSeed = arguments.seedBase + global / (rootCount > 0 ? rootCount : 1);
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

	int32_t rootIndex = 0, futureSeed = 0;
	RolloutOf(arguments, index, rootIndex, futureSeed);

	Lane lane{arguments.ints, arguments.floats, index, arguments.lanes, 0};
	arguments.results[index] = RolloutResult{};
	arguments.results[index].rootIndex = rootIndex;
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
	RolloutBegin(lane, futureSeed, arguments.maxActions);
}

// The head of one round for every lane still in flight.
__global__ void __launch_bounds__(kBlockThreads, kMinBlocks) PreKernel(Arguments arguments)
{
	const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int32_t levelRaw[sizeof(LevelInfo) / sizeof(int32_t)];
	StageLevel(arguments.level, levelRaw);
	const LevelInfo& level = *reinterpret_cast<const LevelInfo*>(levelRaw);
	if (index >= arguments.lanes) return;

	Lane lane{arguments.ints, arguments.floats, index, arguments.lanes, 0};
	// Written before any early exit, so a lane that finished cannot leave last round's slot behind.
	arguments.laneSlot[index] = -1;
	if (lane.s(S_STATE) == LANE_DONE) return;
	const WaveTable& waves = arguments.waves[lane.s(S_ROOT_INDEX)];

	Rand rand = LoadRandState(lane);
	const int32_t stage = RoundPre(lane, level, waves, rand);
	StoreRandState(lane, rand);
	if (stage == PRE_NEEDS_NET)
	{
		// Compaction: the trunk then runs over slots 0..netCount-1 instead of over all lanes, so a round
		// in which a third of the batch can plant costs a third of the GEMMs.  The slot is this lane's
		// activation address for the rest of the round and nothing else; laneSlot hands it back.
		arguments.laneSlot[index] = atomicAdd(&arguments.counters->netCount, 1);
		lane.netSlot = arguments.laneSlot[index];
		BuildFeatures(lane, level, arguments.state, arguments.waveLog, lane.s(S_PRIOR_TICK) >= 0);
		return;
	}
	if (lane.s(S_STATE) == LANE_DONE) atomicSub(&arguments.counters->running, 1);
}

// The tail of one round.  One thread per lane, most of them done with their work in PreKernel: the
// lanes that reached a forward are spread over the whole machine this way, which is what this phase
// needed (see Arguments::laneSlot).
__global__ void __launch_bounds__(kBlockThreads, kMinBlocks) PostKernel(Arguments arguments)
{
	const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int32_t levelRaw[sizeof(LevelInfo) / sizeof(int32_t)];
	StageLevel(arguments.level, levelRaw);
	const LevelInfo& level = *reinterpret_cast<const LevelInfo*>(levelRaw);
	if (index >= arguments.lanes) return;

	const int32_t slot = arguments.laneSlot[index];
	if (slot < 0) return;
	Lane lane{arguments.ints, arguments.floats, index, arguments.lanes, slot};
	const WaveTable& waves = arguments.waves[lane.s(S_ROOT_INDEX)];

	Rand rand = LoadRandState(lane);
	RoundPost(lane, level, waves, arguments.net, arguments.state, rand);
	StoreRandState(lane, rand);
	if (lane.s(S_STATE) == LANE_DONE) atomicSub(&arguments.counters->running, 1);
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

// One rollout start to start, no forward at all: the tick loop's own throughput.  The rounds run in a
// single kernel because nothing has to be batched between them.
__global__ void __launch_bounds__(kBlockThreads, kMinBlocks) RolloutKernel(Arguments arguments)
{
	const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int32_t levelRaw[sizeof(LevelInfo) / sizeof(int32_t)];
	StageLevel(arguments.level, levelRaw);
	const LevelInfo& level = *reinterpret_cast<const LevelInfo*>(levelRaw);
	if (index >= arguments.lanes) return;

	int32_t rootIndex = 0, futureSeed = 0;
	RolloutOf(arguments, index, rootIndex, futureSeed);
	Lane lane{arguments.ints, arguments.floats, index, arguments.lanes, 0};
	if (lane.s(S_STATE) == LANE_DONE) return;
	const WaveTable& waves = arguments.waves[rootIndex];
	RunRollout(lane, level, waves, arguments.net, arguments.state, arguments.waveLog, futureSeed,
		arguments.maxActions, 0);
	StoreResult(lane, arguments.results[index]);
}

__global__ void StoreKernel(Arguments arguments)
{
	const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= arguments.lanes) return;
	if (arguments.results[index].errorFlags & ERR_BAD_ROOT) return;
	Lane lane{arguments.ints, arguments.floats, index, arguments.lanes, 0};
	StoreResult(lane, arguments.results[index]);
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
// Driver
// ---------------------------------------------------------------------------

struct Options
{
	std::string roots, net, out;
	int32_t perRoot = 8;
	int32_t horizon = 2500;
	int32_t maxActions = 3000;
	int32_t lanes = 8192;
	int32_t block = 256;
	int32_t repeat = 1;
	int32_t seedBase = 1600000000;
	int32_t warm = 1;
	int32_t bench = kBenchFull;
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
		else if (flag == "--bench")
		{
			const std::string mode = Value(argc, argv, index);
			if (mode == "sim") options.bench = kBenchSim;
			else if (mode == "full") options.bench = kBenchFull;
			else Fail("--bench takes full|sim");
		}
		else if (flag == "--per-root") options.perRoot = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--horizon") options.horizon = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--max-actions") options.maxActions = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--lanes") options.lanes = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--block") options.block = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--repeat") options.repeat = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--seed-base") options.seedBase = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--warm") options.warm = std::atoi(Value(argc, argv, index).c_str());
		else Fail("unknown flag " + flag);
	}
	if (options.roots.empty() || options.net.empty())
		Fail("usage: --roots FILE --net FILE [--per-root 8] [--horizon 2500] [--lanes 8192] "
			"[--block 256] [--repeat 1] [--max-actions 3000] [--seed-base N] [--bench full|sim] "
			"[--out results.bin]");
	if (options.lanes < 1 || options.block < 32 || options.block > kBlockThreads || options.perRoot < 1)
		Fail("lanes and per-root must be positive, block in [32, 256] (launch bounds)");
	return options;
}

}  // namespace cudarun

using namespace cudarun;
using namespace pvzc;

int main(int argc, char** argv)
{
	Options options = Parse(argc, argv);
	Corpus corpus = ReadCorpus(options.roots);
	NetImage image = ReadNet(options.net);

	const int32_t rootCount = static_cast<int32_t>(corpus.roots.size());
	const int32_t perGame = rootCount * options.perRoot;
	const int32_t total = perGame * options.repeat;

	// The SoA is sized by the batch, not by the rollout count: every batch reuses the same buffers, so
	// the working set stays under the 8 GB budget the task sets for this machine.  Activations are sized
	// in whole chunks of kNetChunk lanes, padded grid included -- 112 cells per plane instead of 60, and
	// the halo is what lets a filter tap be a constant pointer offset.
	const int32_t lanes = options.lanes;
	const int32_t blocks = NetBlocks(lanes);
	const size_t chunkFloats = static_cast<size_t>(kNetChunk) * kPlaneCells * blocks;
	const size_t vectorFloats = static_cast<size_t>(kNetChunk) * blocks;
	const size_t intBytes = static_cast<size_t>(IntSlotsTotal()) * lanes * sizeof(int32_t);
	const size_t floatBytes = static_cast<size_t>(FloatSlotsTotal()) * lanes * sizeof(float);
	const size_t boardBytes = kBoardChannels * chunkFloats * sizeof(float);
	const size_t planeBytes = static_cast<size_t>(image.params.width) * chunkFloats * sizeof(float);
	const size_t scalarBytes = kScalarFeatures * vectorFloats * sizeof(float);
	const size_t contextBytes = static_cast<size_t>(image.params.contextSize) * vectorFloats * sizeof(float);
	const size_t logitsBytes = kLogitCells * vectorFloats * sizeof(float);
	const size_t activationBytes = boardBytes + 2 * planeBytes + scalarBytes + contextBytes + logitsBytes;
	// The three trunk layers, one after another in the order the shift tables walk their taps.
	const int32_t width = image.params.width;
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
	const size_t weightsBytes = image.values.size() * sizeof(float);
	const size_t total_bytes = intBytes + floatBytes + activationBytes + weightsBytes + packedCount * 4
		+ static_cast<size_t>(rootCount) * (sizeof(RootRecord) + sizeof(WaveTable))
		+ static_cast<size_t>(lanes) * (sizeof(RolloutResult) + sizeof(int32_t));
	std::printf("batch lanes %d in %d chunks of %d, device working set %.0f MB (limit 8192 MB)\n",
		lanes, blocks, kNetChunk, static_cast<double>(total_bytes) / 1048576.0);
	if (total_bytes > 8ull * 1024 * 1024 * 1024) Fail("device working set above the 8 GB budget");

	CALL(cudaSetDevice(0));
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
	CALL(cudaMalloc(&ints, intBytes));
	CALL(cudaMalloc(&floats, floatBytes));
	CALL(cudaMalloc(&level, sizeof(LevelInfo)));
	CALL(cudaMalloc(&roots, static_cast<size_t>(rootCount) * sizeof(RootRecord)));
	CALL(cudaMalloc(&waves, static_cast<size_t>(rootCount) * sizeof(WaveTable)));
	CALL(cudaMalloc(&waveLog, (kMaxWaves + 1) * sizeof(float)));
	CALL(cudaMalloc(&netBase, weightsBytes));
	CALL(cudaMalloc(&packedBase, packed.size() * sizeof(float)));
	CALL(cudaMalloc(&board, boardBytes));
	CALL(cudaMalloc(&scalarPlane, scalarBytes));
	CALL(cudaMalloc(&hidden, planeBytes));
	CALL(cudaMalloc(&scratch, planeBytes));
	CALL(cudaMalloc(&context, contextBytes));
	CALL(cudaMalloc(&logits, logitsBytes));
	CALL(cudaMalloc(&laneSlot, static_cast<size_t>(lanes) * sizeof(int32_t)));
	CALL(cudaMalloc(&counters, sizeof(Counters)));
	CALL(cudaMalloc(&results, static_cast<size_t>(lanes) * sizeof(RolloutResult)));
	CALL(cudaMemcpy(level, &corpus.level, sizeof(LevelInfo), cudaMemcpyHostToDevice));
	CALL(cudaMemcpy(roots, corpus.roots.data(), static_cast<size_t>(rootCount) * sizeof(RootRecord),
		cudaMemcpyHostToDevice));
	// The wave table is the future the original already committed to; it is indexed by root, so 200
	// roots cost 200 tables instead of one per lane.
	std::vector<WaveTable> hostWaves(rootCount);
	for (int32_t index = 0; index < rootCount; ++index)
	{
		const RootRecord& root = corpus.roots[index];
		WaveTable& table = hostWaves[index];
		table.waveCount = root.waveCount;
		for (int32_t wave = 0; wave < kMaxWaves; ++wave)
		{
			table.zombieCount[wave] = root.waveZombieCount[wave];
			for (int32_t slot = 0; slot < kWaveZombieSlots; ++slot)
				table.type[wave][slot] = root.waveTable[wave][slot];
		}
	}
	CALL(cudaMemcpy(waves, hostWaves.data(), static_cast<size_t>(rootCount) * sizeof(WaveTable),
		cudaMemcpyHostToDevice));
	std::vector<float> hostWaveLog(kMaxWaves + 1);
	for (int32_t index = 0; index <= kMaxWaves; ++index)
		hostWaveLog[index] = static_cast<float>(std::log1p(static_cast<double>(index))
			/ std::log1p(50.0));
	CALL(cudaMemcpy(waveLog, hostWaveLog.data(), hostWaveLog.size() * sizeof(float), cudaMemcpyHostToDevice));
	CALL(cudaMemcpy(netBase, image.values.data(), weightsBytes, cudaMemcpyHostToDevice));
	CALL(cudaMemcpy(packedBase, packed.data(), packed.size() * sizeof(float), cudaMemcpyHostToDevice));
	// The halo of every plane is zero from here on and nothing ever writes it, which is what makes the
	// padding free: a source cell off the lawn is a halo cell holding 0.
	CALL(cudaMemset(board, 0, boardBytes));
	CALL(cudaMemset(hidden, 0, planeBytes));
	CALL(cudaMemset(scratch, 0, planeBytes));

	// The device NetParams point at the uploaded weight buffer at the offsets the file order implies.
	NetParams deviceNet = image.params;
	{
		size_t place = 0;
		auto take = [&netBase, &place](size_t count)
		{
			float* pointer = netBase + place;
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
	// The batched path reads the trunk's weights from the repacked buffer; the per-lane code in
	// NetForward.inc (which the device never calls for the trunk) would read the dump order from these
	// same three pointers, so the replacement is deliberate and lives only here.
	deviceNet.conv1 = packedBase;
	deviceNet.conv2 = packedBase + PackedCount(kBoardChannels, width, 9);
	deviceNet.conv3 = packedBase + PackedCount(kBoardChannels, width, 9) + PackedCount(width, width, 9);

	cublasHandle_t handle = nullptr;
	CALL(cublasCreate(&handle));
	// TF32 tensor cores would roughly double the trunk, and change the rounding.  Phase 1 compares
	// decisions against the CPU forward, so reduced precision is a measured opt-in, not a default.
	CALL(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

	int32_t shifts3[9], shifts1x5[5];
	for (int32_t dr = 0; dr < 3; ++dr)
		for (int32_t dc = 0; dc < 3; ++dc) shifts3[dr * 3 + dc] = kValidOrigin + (dr - 1) * kColsP + (dc - 1);
	for (int32_t dc = 0; dc < 5; ++dc) shifts1x5[dc] = kValidOrigin + (dc - 2);

	Arguments arguments;
	arguments.ints = ints;
	arguments.floats = floats;
	arguments.level = level;
	arguments.roots = roots;
	arguments.waves = waves;
	arguments.waveLog = waveLog;
	arguments.net = deviceNet;
	arguments.state = NetState{board, scalarPlane, hidden, scratch, context, logits};
	arguments.results = results;
	arguments.laneSlot = laneSlot;
	arguments.counters = counters;
	arguments.lanes = lanes;
	arguments.firstRollout = 0;
	arguments.perRoot = options.perRoot;
	arguments.perGame = perGame;
	arguments.horizon = options.horizon;
	arguments.maxActions = options.maxActions;
	arguments.seedBase = options.seedBase;
	arguments.bench = options.bench;

	// One round's three layers, with the ping-pong the host build uses too: hidden and scratch alternate
	// as input and output so only two width-plane buffers are live.
	const ConvLayer conv1{board, hidden, deviceNet.conv1, deviceNet.bias1, kBoardChannels, width, 9,
		shifts3};
	const ConvLayer conv2{hidden, scratch, deviceNet.conv2, deviceNet.bias2, width, width, 9, shifts3};
	const ConvLayer conv3{scratch, hidden, deviceNet.conv3, deviceNet.bias3, width, width, 5, shifts1x5};

	int32_t* hostCounters = nullptr;
	CALL(cudaMallocHost(&hostCounters, 2 * sizeof(int32_t)));

	std::vector<RolloutResult> hostResults(static_cast<size_t>(lanes));
	std::vector<RolloutResult> collected;
	if (!options.out.empty()) collected.resize(static_cast<size_t>(total));

	cudaEvent_t start, stop;
	CALL(cudaEventCreate(&start));
	CALL(cudaEventCreate(&stop));
	cudaEvent_t phaseStart[3], phaseStop[3];
	for (int32_t index = 0; index < 3; ++index)
	{
		CALL(cudaEventCreate(&phaseStart[index]));
		CALL(cudaEventCreate(&phaseStop[index]));
	}
	// The extra event sync per round is measurable, so the breakdown is opt-in: PHASE=1 to find out
	// where a round went, unset for the number that goes into the gate.
	const bool phaseTiming = std::getenv("PHASE") != nullptr;

	int64_t ticks = 0, decisions = 0, netCalls = 0, planted = 0, eaten = 0, killed = 0, errors = 0;
	int64_t rounds = 0;
	int32_t maxNet = 0;
	// Where a round goes.  Three event pairs per round, read after the round's own synchronisation: the
	// split is what turns "the batch is slow" into "1.4 ms of GEMM, 18 ms of sim, 3 ms of wait".
	float milliPre = 0.0f, milliConv = 0.0f, milliPost = 0.0f;
	int32_t won = 0, lost = 0, ongoing = 0;
	int32_t completed = 0;
	float milliseconds = 0.0f;
	const int32_t batchCount = (total + lanes - 1) / lanes;
	// A round applies at most one action, and RoundTop ends a lane at maxActions of them, so the loop
	// provably drains every lane within maxActions + 1 rounds.  The guard turns a broken S_STATE into a
	// stalled benchmark with a message instead of an infinite launch sequence.
	const int32_t roundGuard = options.maxActions + 8;
	for (int32_t batch = -options.warm; batch < batchCount; ++batch)
	{
		// The warm-up batch runs the same round loop; counting its rounds would divide the timed work by a
		// larger denominator than it did.
		if (batch == 0)
		{
			rounds = 0;
			maxNet = 0;
			milliPre = milliConv = milliPost = 0.0f;
		}
		const int32_t first = batch < 0 ? 0 : batch * lanes;
		// A batch never launches more lanes than there are rollouts left: a lane whose rollout duplicates
		// one already in the batch would do real work that the rollouts/s figure could not count.
		const int32_t batchLanes = lanes < total - first ? lanes : total - first;
		arguments.lanes = batchLanes;
		arguments.firstRollout = first;
		const Counters initial{0, batchLanes};
		CALL(cudaMemcpy(counters, &initial, sizeof(Counters), cudaMemcpyHostToDevice));
		CALL(cudaEventRecord(start));

		InitKernel<<<(batchLanes + options.block - 1) / options.block, options.block>>>(arguments);
		CHECK_LAUNCH();
		if (options.bench == kBenchSim)
		{
			RolloutKernel<<<(batchLanes + options.block - 1) / options.block, options.block>>>(arguments);
			CHECK_LAUNCH();
		}
		else
		{
			int32_t round = 0;
			for (; round < roundGuard; ++round)
			{
				CALL(cudaEventRecord(phaseStart[0]));
				CALL(cudaMemsetAsync(counters, 0, sizeof(int32_t)));   // netCount; running carries over
				PreKernel<<<(batchLanes + options.block - 1) / options.block, options.block>>>(arguments);
				CHECK_LAUNCH();
				CALL(cudaEventRecord(phaseStop[0]));
				// One 8-byte readback per round, and the only host/device synchronisation the loop needs
				// without the phase timer: it is what lets the batch stop as soon as the last lane reaches
				// its horizon.
				CALL(cudaMemcpyAsync(hostCounters, counters, 2 * sizeof(int32_t),
					cudaMemcpyDeviceToHost));
				CALL(cudaStreamSynchronize(0));
				if (hostCounters[1] == 0) break;
				const int32_t netCount = hostCounters[0];
				if (netCount > maxNet) maxNet = netCount;
				const int32_t netBlocks = NetBlocks(netCount);
				CALL(cudaEventRecord(phaseStart[1]));
				RunConv(handle, conv1, netBlocks);
				RunConv(handle, conv2, netBlocks);
				RunConv(handle, conv3, netBlocks);
				RunCellPre(handle, arguments.state, deviceNet, netBlocks);
				CALL(cudaEventRecord(phaseStop[1]));
				CALL(cudaEventRecord(phaseStart[2]));
				// Over all lanes, not over the compacted ones: a round that needs a forward for 7,500 of
				// 21,504 lanes would otherwise launch 30 blocks into 84 SMs, and heads+apply was then the
				// most expensive third of the round.  The threads that find -1 retire immediately.
				PostKernel<<<(batchLanes + options.block - 1) / options.block, options.block>>>(arguments);
				CHECK_LAUNCH();
				CALL(cudaEventRecord(phaseStop[2]));
				if (phaseTiming)
				{
					CALL(cudaEventSynchronize(phaseStop[2]));
					float elapsed = 0.0f;
					for (int32_t phase = 0; phase < 3; ++phase)
					{
						CALL(cudaEventElapsedTime(&elapsed, phaseStart[phase], phaseStop[phase]));
						if (phase == 0) milliPre += elapsed;
						else if (phase == 1) milliConv += elapsed;
						else milliPost += elapsed;
					}
				}
			}
			if (round >= roundGuard) Fail("round guard hit: lanes are not draining");
			rounds += round;
			StoreKernel<<<(batchLanes + options.block - 1) / options.block, options.block>>>(arguments);
			CHECK_LAUNCH();
		}
		CALL(cudaEventRecord(stop));
		CALL(cudaEventSynchronize(stop));
		if (batch < 0) continue;                       // the timed run starts after the warm-up
		float elapsed = 0.0f;
		CALL(cudaEventElapsedTime(&elapsed, start, stop));
		milliseconds += elapsed;
		const int32_t inBatch = batchLanes;
		CALL(cudaMemcpy(hostResults.data(), results, static_cast<size_t>(inBatch) * sizeof(RolloutResult),
			cudaMemcpyDeviceToHost));
		for (int32_t index = 0; index < inBatch; ++index)
		{
			const RolloutResult& result = hostResults[index];
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
		completed += inBatch;
		if (batchCount > 1) std::printf("  batch %d/%d done\n", batch + 1, batchCount);
	}

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
			maxNet, lanes);
		if (phaseTiming)
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
	cublasDestroy(handle);
	return 0;
}

#endif
