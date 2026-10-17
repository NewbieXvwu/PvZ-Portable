// Phase 0: the CUDA rollout core, run against real exported positions.
//
// One thread = one rollout = one lane.  The thread transposes its root record into the batch SoA,
// plays the ROLLOUT loop (cuda/Rollout.inc) to the horizon, and writes the leaf statistics the
// agreement test reads.  Nothing about the game logic lives in this file: the tick, the features, the
// network and the decision all come from the .inc sources that the host build compiles too, so a
// disagreement with the original simulator is a mechanics bug rather than "which rewrite is wrong".
//
// Build (no CMake needed for the measurement):
//   /usr/local/cuda/bin/nvcc -arch=sm_120 -O3 --use_fast_math=false -std=c++17 \
//       -I. -I../src -o rollout_cuda RolloutCuda.cu
// Run:
//   ./rollout_cuda --roots ../artifacts/cuda/corpus_smoke/roots.cudaroot \
//                  --net ../artifacts/cuda/student/v14.pvzrn --lanes 8192 --repeat 4
#ifndef PVZ_ROLLOUT_CUDA_CU
#define PVZ_ROLLOUT_CUDA_CU

#include "SimCore.h"
#include "SimImpl.inc"
#include "SimTick.inc"
#include "NetForward.inc"
#include "Rollout.inc"

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
 cudaGetLastError();
	Fail(message);
}

#define CALL(expression) Checked((expression), #expression)
#define CHECK_LAUNCH() Checked(cudaGetLastError(), "kernel launch")

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

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// Benchmark modes.  The gate is a throughput number, so the driver has to be able to say *which half*
// of a continuation produced it.
enum BenchMode { kBenchFull = 0, kBenchNet = 1, kBenchSim = 2 };

// Occupancy control.  At 248 registers a thread the SM fits exactly one 256-thread block, i.e. 8 warps
// to cover a ~700-cycle global load; building with -DPVZ_MIN_BLOCKS=2 caps the budget at 128 registers
// and trades spills for warps.  --block may not exceed kBlockThreads because of this contract.
enum { kBlockThreads = 256 };
#ifndef PVZ_MIN_BLOCKS
#define PVZ_MIN_BLOCKS 1
#endif
enum { kMinBlocks = PVZ_MIN_BLOCKS };

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
	int32_t lanes;
	int32_t firstRollout;
	int32_t perRoot;
	int32_t perGame;                  // rootCount * perRoot; the rollout index wraps over repeats
	int32_t horizon;
	int32_t maxActions;
	int32_t seedBase;
	int32_t bench;                    // BenchMode
	int32_t netCallsPerRollout;       // what the measured full run does, so kBenchNet is comparable
};

__global__ void __launch_bounds__(kBlockThreads, kMinBlocks) RolloutKernel(Arguments arguments)
{
	const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= arguments.lanes) return;

	// The plant and zombie tables are scanned on the hot path (TickPlantDef runs for every live plant
	// every tick), so the block stages them once in shared memory instead of broadcasting global loads.
	__shared__ int32_t levelRaw[sizeof(LevelInfo) / sizeof(int32_t)];
	const int32_t* levelWords = reinterpret_cast<const int32_t*>(arguments.level);
	for (int32_t word = threadIdx.x; word < sizeof(LevelInfo) / sizeof(int32_t); word += blockDim.x)
		levelRaw[word] = levelWords[word];
	__syncthreads();
	const LevelInfo& level = *reinterpret_cast<const LevelInfo*>(levelRaw);

	// rollout -> (root, future).  The modulo is what lets --repeat replay the same workload through a
	// buffer that only holds one batch of lanes.
	const int32_t slot = (arguments.firstRollout + index) % arguments.perGame;
	const int32_t rootIndex = slot / arguments.perRoot;

	Lane lane{arguments.ints, arguments.floats, index, arguments.lanes};
	RolloutResult& result = arguments.results[index];
	result = RolloutResult{};
	result.rootIndex = rootIndex;
	if (!LoadRoot(lane, arguments.roots[rootIndex], level))
	{
		result.errorFlags = ERR_BAD_ROOT;
		return;
	}
	lane.s(S_ROOT_INDEX) = rootIndex;
	lane.s(S_UNTIL_TICK) = arguments.roots[rootIndex].tick + arguments.horizon;
	if (arguments.bench == kBenchNet)
	{
		// The net forward on its own: the 43 measured forwards per continuation, run against the root's
		// own features.  Splits "the forward is slow" from "the tick loop is slow".
		for (int32_t step = 0; step < arguments.netCallsPerRollout; ++step)
		{
			BuildFeatures(lane, level, arguments.state, arguments.waveLog, true);
			NetEncode(lane, arguments.net, arguments.state);
		}
		result.ticks = arguments.horizon;
		result.decisions = arguments.netCallsPerRollout;
		result.netCalls = arguments.netCallsPerRollout;
		result.result = 0;
		return;
	}
#ifdef PVZ_LANE_CHECK
	const int32_t entryTick = lane.s(S_TICK), entryIncome = lane.s(S_INCOME_COUNT);
	std::printf("lane %d root %d stride %d record.tick %d s(S_TICK) %d raw %d sun %d wave %d\n", index,
		rootIndex, lane.stride, arguments.roots[rootIndex].tick, lane.s(S_TICK),
		arguments.ints[IntFieldOffset(F_SCALAR) * lane.stride + lane.lane + S_TICK], lane.s(S_SUN),
		lane.s(S_WAVE));
#endif
	RunRollout(lane, level, arguments.waves[rootIndex], arguments.net, arguments.state,
		arguments.waveLog, arguments.seedBase + slot % arguments.perRoot, arguments.maxActions,
		arguments.bench == kBenchSim ? 0 : 1);
	StoreResult(lane, result);
#ifdef PVZ_LANE_CHECK
	if (lane.s(S_INCOME_COUNT) >= kIncomeEvents || lane.s(S_DECISIONS) > arguments.maxActions)
		std::printf("lane %d root %d: tick %d -> %d until %d income %d -> %d decisions %d error %d\n",
			index, rootIndex, entryTick, lane.s(S_TICK), lane.s(S_UNTIL_TICK), entryIncome,
			lane.s(S_INCOME_COUNT), lane.s(S_DECISIONS), lane.s(S_ERROR));
#endif
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
	int32_t netCallsPerRollout = 43;
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
		const std::string value = index + 1 < argc ? argv[index + 1] : std::string();
		if (flag == "--roots") options.roots = Value(argc, argv, index);
		else if (flag == "--net") options.net = Value(argc, argv, index);
		else if (flag == "--out") options.out = Value(argc, argv, index);
		else if (flag == "--bench")
		{
			const std::string mode = Value(argc, argv, index);
			options.bench = mode == "net" ? kBenchNet : (mode == "sim" ? kBenchSim : kBenchFull);
			if (mode != "net" && mode != "sim" && mode != "full") Fail("--bench takes full|net|sim");
		}
		else if (flag == "--net-calls") options.netCallsPerRollout = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--per-root") options.perRoot = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--horizon") options.horizon = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--max-actions") options.maxActions = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--lanes") options.lanes = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--block") options.block = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--repeat") options.repeat = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--seed-base") options.seedBase = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--warm") options.warm = std::atoi(Value(argc, argv, index).c_str());
		else Fail("unknown flag " + flag);
		(void)value;
	}
	if (options.roots.empty() || options.net.empty())
		Fail("usage: --roots FILE --net FILE [--per-root 8] [--horizon 2500] [--lanes 8192] "
			"[--block 256] [--repeat 1] [--max-actions 3000] [--seed-base N] [--out results.bin]");
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
	// the working set stays under the 8 GB budget the task sets for this machine.
	const int32_t lanes = options.lanes;
	const size_t intBytes = static_cast<size_t>(IntSlotsTotal()) * lanes * sizeof(int32_t);
	const size_t floatBytes = static_cast<size_t>(FloatSlotsTotal()) * lanes * sizeof(float);
	const size_t boardBytes = static_cast<size_t>(kBoardChannels) * kCells * lanes * sizeof(float);
	const size_t hiddenBytes = static_cast<size_t>(image.params.width) * kCells * lanes * sizeof(float);
	const size_t contextBytes = static_cast<size_t>(image.params.contextSize) * lanes * sizeof(float);
	const size_t scalarBytes = static_cast<size_t>(image.params.scalarCount) * lanes * sizeof(float);
	const size_t logitsBytes = static_cast<size_t>(kRows * kGridCols) * lanes * sizeof(float);
	const size_t activationBytes = boardBytes + 2 * hiddenBytes + contextBytes + scalarBytes + logitsBytes;
	const size_t weightsBytes = image.values.size() * sizeof(float);
	const size_t total_bytes = intBytes + floatBytes + activationBytes + weightsBytes
		+ static_cast<size_t>(rootCount) * (sizeof(RootRecord) + sizeof(WaveTable))
		+ static_cast<size_t>(lanes) * sizeof(RolloutResult);
	std::printf("batch lanes %d, device working set %.0f MB (limit 8192 MB)\n", lanes,
		static_cast<double>(total_bytes) / 1048576.0);
	if (total_bytes > 8ull * 1024 * 1024 * 1024) Fail("device working set above the 8 GB budget");

	CALL(cudaSetDevice(0));
	int32_t* ints = nullptr;
	float* floats = nullptr;
	LevelInfo* level = nullptr;
	RootRecord* roots = nullptr;
	WaveTable* waves = nullptr;
	float* waveLog = nullptr;
	float* netBase = nullptr;
	float* board = nullptr;
	float* scalarPlane = nullptr;
	float* hidden = nullptr;
	float* scratch = nullptr;
	float* context = nullptr;
	float* logits = nullptr;
	RolloutResult* results = nullptr;
	CALL(cudaMalloc(&ints, intBytes));
	CALL(cudaMalloc(&floats, floatBytes));
	CALL(cudaMalloc(&level, sizeof(LevelInfo)));
	CALL(cudaMalloc(&roots, static_cast<size_t>(rootCount) * sizeof(RootRecord)));
	CALL(cudaMalloc(&waves, static_cast<size_t>(rootCount) * sizeof(WaveTable)));
	CALL(cudaMalloc(&waveLog, (kMaxWaves + 1) * sizeof(float)));
	CALL(cudaMalloc(&netBase, weightsBytes));
	CALL(cudaMalloc(&board, boardBytes));
	CALL(cudaMalloc(&scalarPlane, scalarBytes));
	CALL(cudaMalloc(&hidden, hiddenBytes));
	CALL(cudaMalloc(&scratch, hiddenBytes));
	CALL(cudaMalloc(&context, contextBytes));
	CALL(cudaMalloc(&logits, logitsBytes));
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
	arguments.perRoot = options.perRoot;
	arguments.perGame = perGame;
	arguments.horizon = options.horizon;
	arguments.maxActions = options.maxActions;
	arguments.seedBase = options.seedBase;
	arguments.bench = options.bench;
	arguments.netCallsPerRollout = options.netCallsPerRollout;

	std::vector<RolloutResult> hostResults(static_cast<size_t>(lanes));
	std::vector<RolloutResult> collected;
	if (!options.out.empty()) collected.resize(static_cast<size_t>(total));

	cudaEvent_t start, stop;
	CALL(cudaEventCreate(&start));
	CALL(cudaEventCreate(&stop));

	int64_t ticks = 0, decisions = 0, netCalls = 0, planted = 0, eaten = 0, killed = 0, errors = 0;
	int32_t won = 0, lost = 0, ongoing = 0;
	int32_t completed = 0;
	float milliseconds = 0.0f;
	const int32_t batchCount = (total + lanes - 1) / lanes;
	for (int32_t batch = -options.warm; batch < batchCount; ++batch)
	{
		const int32_t first = batch < 0 ? 0 : batch * lanes;
		arguments.lanes = lanes;
		arguments.firstRollout = first;
		CALL(cudaEventRecord(start));
		RolloutKernel<<<(lanes + options.block - 1) / options.block, options.block>>>(arguments);
		CHECK_LAUNCH();
		CALL(cudaEventRecord(stop));
		CALL(cudaEventSynchronize(stop));
		if (batch < 0) continue;                       // the timed run starts after the warm-up
		float elapsed = 0.0f;
		CALL(cudaEventElapsedTime(&elapsed, start, stop));
		milliseconds += elapsed;
		const int32_t inBatch = lanes < total - first ? lanes : total - first;
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
	std::printf("wall %.0f ms for %d rollouts -> %.0f rollouts/s, %.3f Gticks/s\n", milliseconds, completed,
		completed * 1000.0 / milliseconds, static_cast<double>(ticks) / milliseconds / 1.0e6);
	std::printf("per rollout: %.1f ticks, %.1f decisions, %.1f net forwards, %.1f planted, %.1f eaten, "
		"%.1f killed\n",
		static_cast<double>(ticks) / completed, static_cast<double>(decisions) / completed,
		static_cast<double>(netCalls) / completed,
		static_cast<double>(planted) / completed, static_cast<double>(eaten) / completed,
		static_cast<double>(killed) / completed);
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
	return 0;
}

#endif
