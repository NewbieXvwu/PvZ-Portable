// Single-lane host build of the re-implemented core: one lane, stride 1, the SoA in a heap block.
//
// Its purpose is diagnosis.  The same SimImpl.inc / SimTick.inc / NetForward.inc / Rollout.inc the
// device runs is compiled here by plain g++, so an out-of-bounds slot index, a wrong loop bound or a
// bad constant shows up under -fsanitize=address with a source line, without a GPU round trip and
// without nvcc's inlining making the attribution ambiguous.  It is also the reference the phase-1
// agreement test compares the GPU against when a disagreement must be split into "GPU bug" vs
// "mechanics bug": host build disagrees with the GPU -> GPU bug; host build disagrees with the
// original simulator -> mechanics bug.
//
// It runs the whole rollout, not just the tick: it reads a .cudaroot corpus and a .pvzrn student dump,
// transposes one RootRecord into the SoA (pvzc::LoadRoot, the same function the device's InitKernel
// calls), then calls pvzc::RunRollout(...) with useNet = 1, which is the single-threaded stringing
// together of the same RoundTop / NetDecide / RoundApply pieces the device splits across kernels.
//
// build: g++ -fsanitize=address -g -O1 -std=c++17 -I. -o build/sim_host cuda/SimHost.cpp
// run:   ./build/sim_host --roots artifacts/cuda/corpus_smoke/roots.cudaroot
//        --net artifacts/cuda/student/v14.pvzrn --per-root 8 --horizon 2500
//        --pairs pairs.txt --leaf-out host_leaf.bin    (the device's workload, then its leaf encoding)
//        --root-features host_root.bin                 (encode every root and stop: the encoder check)
//
// The activations use the device's chunked, padded layout (cuda/CudaRollout.h), so the host allocates
// one whole chunk of kNetChunk lanes to run a single lane.  That is deliberate: the layout is what
// makes the batched GEMMs and these per-lane loops address the same float, and the phase-1 test is
// about the mechanics, not about the host build being fast.  See cuda/spec/host.md.

// Before the .inc chain: cuda/NetCore.h names `size_t` unqualified but only includes <cstdint>, which
// libstdc++ does not have to define at global scope.  nvcc gets it for free from its own preincludes,
// plain g++ does not, so the host build pulls it in first rather than editing the shared header.
#include <cstddef>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cuda/SimTick.inc"
#include "cuda/NetForward.inc"
#include "cuda/Rollout.inc"
#include "cuda/PairList.h"

namespace pvzhost
{

using namespace pvzc;

enum { kRecordLevel = 1, kRecordRoot = 2 };

[[noreturn]] void Fail(const std::string& message)
{
	std::fprintf(stderr, "sim-host: %s\n", message.c_str());
	std::exit(1);
}

// ---------------------------------------------------------------------------
// The corpus: one LEVEL record and N ROOT records, each written as the raw struct it mirrors
// ---------------------------------------------------------------------------

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
				+ std::to_string(kVersion) + " in " + path);
		if (tag[2] == kRecordLevel)
		{
			if (std::fread(&corpus.level, sizeof(LevelInfo), 1, stream) != 1) Fail("short level record");
			corpus.haveLevel = true;
		}
		else if (tag[2] == kRecordRoot)
		{
			RootRecord root;
			if (std::fread(&root, sizeof(RootRecord), 1, stream) != 1) Fail("short root record");
			// The exporter marks positions it cannot represent; they are dropped here exactly as the
			// device driver drops them, so both sides index roots the same way.
			if (root.supported) corpus.roots.push_back(root);
		}
		else Fail("unknown record kind " + std::to_string(tag[2]));
	}
	// A trailing partial record means a truncated transfer, and the loop above would have stopped
	// mid-tag: say so rather than reporting a corpus that silently lost its last root.
	const long here = std::ftell(stream);
	std::fseek(stream, 0, SEEK_END);
	if (std::ftell(stream) != here) Fail(path + " ends with a partial record");
	std::fclose(stream);
	if (!corpus.haveLevel) Fail(path + " holds no level record");
	if (corpus.roots.empty()) Fail(path + " holds no supported roots");
	return corpus;
}

// ---------------------------------------------------------------------------
// The student dump, in EnvironmentRollout::Load's order (the device driver's ReadNet, same counts)
// ---------------------------------------------------------------------------

std::vector<size_t> WeightCounts(int32_t width, int32_t meanmax, int32_t withContext, int32_t observation)
{
	const size_t w = static_cast<size_t>(width);
	const size_t context = (meanmax ? 3 : 2) * w;
	const size_t v2lite = observation == 2 ? 1 : 0;
	const size_t scalars = static_cast<size_t>(kScalars) + (withContext ? kTaskContext : 0) + v2lite;
	const size_t channels = static_cast<size_t>(kBoardChannels) + 4 * v2lite;
	return {w * channels * 9, w, w * w * 9, w, w * w * 5, w, w * scalars, w,
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
	uint32_t width = 0, pool = 0, withContext = 0, observation = 1;
	if (std::fread(magic, 8, 1, stream) != 1) Fail(path + " is too short");
	const std::string version(magic, 8);
	if (version != "PVZRN002" && version != "PVZRN003" && version != "PVZRN004")
		Fail("unsupported network dump " + version);
	if (std::fread(&width, 4, 1, stream) != 1 || std::fread(&pool, 4, 1, stream) != 1) Fail("short header");
	if ((version == "PVZRN003" || version == "PVZRN004") && std::fread(&withContext, 4, 1, stream) != 1)
		Fail("short header");
	if (version == "PVZRN004" && std::fread(&observation, 4, 1, stream) != 1) Fail("short header");
	if (observation < 1 || observation > 2) Fail("bad observation version");
	if (width < 1 || width > kMaxNetWidth) Fail("network width " + std::to_string(width) + " out of range");

	NetImage image;
	const std::vector<size_t> counts = WeightCounts(static_cast<int32_t>(width), pool ? 1 : 0,
		withContext ? 1 : 0, static_cast<int32_t>(observation));
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

	// File order = Load order = conv1, bias1, conv2, bias2, conv3, bias3, scalar, scalarBias, plant,
	// plantBias, packet, packetBias, embedding, cell, cellBias, cellOut, cellOutBias.  Unlike the
	// device, these pointers are the real ones the per-lane loops dereference, and they stay in dump
	// order: cuda/RolloutCuda.cu repacks the trunk for cuBLAS, the host build never does.
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
	params.version = version == "PVZRN004" ? 4 : (version == "PVZRN003" ? 3 : 2);
	params.contextSize = (params.meanmax ? 3 : 2) * params.width;
	params.scalarCount = kScalars + (params.taskContext ? kTaskContext : 0) + (observation == 2 ? 1 : 0);
	params.boardChannels = kBoardChannels + (observation == 2 ? 4 : 0);
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
// Activations: one chunk of kNetChunk lanes, padded grid, zero halo, exactly as the device allocates
// ---------------------------------------------------------------------------

// The largest index each buffer is addressed by, derived from the very formulas NetForward.inc and
// SimImpl.inc use, so a layout change makes the allocation check below fail loudly instead of turning
// into an out-of-bounds write that ASan attributes to the wrong line.
size_t PlaneCapacity(int32_t channels)
{
	return PlaneIndex(channels, kNetChunk - 1, channels - 1, kPlaneCells - 1) + 1;
}

size_t VectorCapacity(int32_t width)
{
	return VectorIndex(width, kNetChunk - 1, width - 1) + 1;
}

struct Activations
{
	std::vector<float> board, scalars, hidden, scratch, context, logits;
	NetState state{};

	void Build(int32_t width, int32_t contextSize)
	{
		board.assign(PlaneCapacity(kStudentBoardChannels), 0.0f);
		scalars.assign(VectorCapacity(kScalarFeatures), 0.0f);
		hidden.assign(PlaneCapacity(width), 0.0f);
		scratch.assign(PlaneCapacity(width), 0.0f);
		context.assign(VectorCapacity(contextSize), 0.0f);
		logits.assign(VectorCapacity(kLogitCells), 0.0f);
		state.board = board.data();
		state.scalars = scalars.data();
		state.hidden = hidden.data();
		state.scratch = scratch.data();
		state.context = context.data();
		state.logits = logits.data();
	}

	size_t bytes() const
	{
		return (board.size() + scalars.size() + hidden.size() + scratch.size() + context.size()
			+ logits.size()) * sizeof(float);
	}
};

// log1p(waves)/log1p(50), uploaded by the device driver and read here: same table, same bits.
std::vector<float> WaveLogTable()
{
	std::vector<float> table(kMaxWaves + 1);
	for (int32_t index = 0; index <= kMaxWaves; ++index)
		table[index] = static_cast<float>(std::log1p(static_cast<double>(index)) / std::log1p(50.0));
	return table;
}

// ---------------------------------------------------------------------------
// Leaf statistics, aggregated the way the device driver aggregates them
// ---------------------------------------------------------------------------

struct Leaf
{
	int64_t rollouts = 0;
	int64_t ticks = 0, decisions = 0, netCalls = 0, planted = 0, eaten = 0, killed = 0;
	int64_t sun = 0, wave = 0, mowers = 0, zombiesAlive = 0, plantsAlive = 0, sunProduced = 0;
	int64_t won = 0, lost = 0, ongoing = 0, errors = 0;

	void Add(const RolloutResult& result)
	{
		++rollouts;
		ticks += result.ticks;
		decisions += result.decisions;
		netCalls += result.netCalls;
		planted += result.plantsPlanted;
		eaten += result.plantsEaten;
		killed += result.zombiesKilled;
		sun += result.sun;
		wave += result.wave;
		mowers += result.mowersLeft;
		zombiesAlive += result.zombiesAlive;
		plantsAlive += result.plantsAlive;
		sunProduced += result.sunProduced;
		if (result.result > 0) ++won;
		else if (result.result < 0) ++lost;
		else ++ongoing;
		if (result.errorFlags != 0) ++errors;
	}

	void Merge(const Leaf& other)
	{
		rollouts += other.rollouts; ticks += other.ticks; decisions += other.decisions;
		netCalls += other.netCalls; planted += other.planted; eaten += other.eaten;
		killed += other.killed; sun += other.sun; wave += other.wave; mowers += other.mowers;
		zombiesAlive += other.zombiesAlive; plantsAlive += other.plantsAlive;
		sunProduced += other.sunProduced; won += other.won; lost += other.lost;
		ongoing += other.ongoing; errors += other.errors;
	}

	// One row of the table.  `scale` divides the sums, so a row can be printed as a mean per rollout.
	// The last two live-entity counts are not in cuda/RolloutCuda.cu's printed set, but they are in the
	// original simulator's reference dump (artifacts/cuda/corpus_smoke/corpus.jsonl), which is the other
	// half of the agreement test.
	void Print(const char* label, double scale) const
	{
		if (!rollouts) { std::printf("%-11s %6d  (no rollouts)\n", label, 0); return; }
		const double n = scale > 0.0 ? scale : 1.0;
		std::printf("%-11s %6lld  %8.1f  %7.1f  %7.1f  %7.1f  %7.1f  %7.1f  %8.1f  %6.2f  %5.2f"
			"  %6.2f  %6.2f   %5lld/%5lld/%6lld  %4lld\n",
			label, static_cast<long long>(rollouts), static_cast<double>(ticks) / n,
			static_cast<double>(decisions) / n, static_cast<double>(netCalls) / n,
			static_cast<double>(planted) / n, static_cast<double>(eaten) / n,
			static_cast<double>(killed) / n, static_cast<double>(sun) / n,
			static_cast<double>(wave) / n, static_cast<double>(mowers) / n,
			static_cast<double>(plantsAlive) / n, static_cast<double>(zombiesAlive) / n,
			static_cast<long long>(won), static_cast<long long>(lost),
			static_cast<long long>(ongoing), static_cast<long long>(errors));
	}
};

void PrintHeader()
{
	std::printf("%-11s %6s  %8s  %7s  %7s  %7s  %7s  %7s  %8s  %6s  %5s  %6s  %6s   %19s  %4s\n",
		"scope", "rolls", "ticks", "decis", "net", "plant", "eaten", "kill", "sun", "wave", "mower",
		"palive", "zalive", "w/lost/ongoing", "err");
}

}  // namespace pvzhost

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

namespace
{

using namespace pvzhost;

struct Options
{
	std::string roots = "artifacts/cuda/corpus_smoke/roots.cudaroot";
	std::string net = "artifacts/cuda/student/v14.pvzrn";
	int32_t perRoot = 8;
	int32_t horizon = 2500;
	int32_t maxActions = 3000;
	int32_t repeat = 1;
	int32_t seedBase = 1600000000;
	int32_t useNet = 1;
	int32_t rootLimit = 0;          // 0 = every supported root
	// The explicit workload and the leaf feature dump, both shared with cuda/RolloutCuda.cu: run the same
	// (root, future) pairs through this build and through the device, then diff the two dumps.  A
	// difference is a GPU bug, because it is the same code compiled twice.
	std::string pairs, leafOut;
	// Dump the value net's encoding of every root as the core holds it, before a single tick runs, and
	// exit.  This is the encoder check: python/pvz_cuda_root.py builds the same frame straight from the
	// .cudaroot and feeds it to pvz_value_net.featurize_numpy, so a differing block is a bug in one of the
	// two encoders with no random stream left to blame it on.
	std::string rootFeatures;
	// The *student's* encoding of every root (v1: 92 channels + 115 scalars; v2lite: 96 + 116), built by
	// the same BuildFeatures the rollout uses.  Python reconstructs the same vector from the .cudaroot
	// through StudentHistory, so a differing float is an encoder bug, not a different future.
	std::string studentFeatures;
	// Fill the scalar activation vector with this value before the first forward.  BuildFeatures writes
	// only the indices its deck and its scalars cover; everything it never writes but NetContext still
	// reads keeps this value instead of 0.  Left at 0 it is a no-op; see cuda/spec/host.md item 3.
	float poison = 0.0f;
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
		else if (flag == "--per-root") options.perRoot = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--horizon") options.horizon = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--max-actions") options.maxActions = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--repeat") options.repeat = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--seed-base") options.seedBase = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--root-limit") options.rootLimit = std::atoi(Value(argc, argv, index).c_str());
		else if (flag == "--pairs") options.pairs = Value(argc, argv, index);
		else if (flag == "--leaf-out") options.leafOut = Value(argc, argv, index);
		else if (flag == "--root-features") options.rootFeatures = Value(argc, argv, index);
		else if (flag == "--student-features") options.studentFeatures = Value(argc, argv, index);
		else if (flag == "--poison") options.poison = std::strtof(Value(argc, argv, index).c_str(), nullptr);
		else if (flag == "--bench")
		{
			const std::string mode = Value(argc, argv, index);
			// Same meaning as cuda/RolloutCuda.cu's --bench sim: the tick loop and the wait rounds, no
			// forward at all.  Isolates the mechanics from the network when a number looks wrong.
			if (mode == "full") options.useNet = 1;
			else if (mode == "sim") options.useNet = 0;
			else Fail("--bench takes full|sim");
		}
		else Fail("unknown flag " + flag);
	}
	if (options.perRoot < 1 || options.repeat < 1 || options.horizon < 1 || options.maxActions < 1)
		Fail("--per-root, --repeat, --horizon and --max-actions must all be positive");
	return options;
}

double Seconds(const std::chrono::steady_clock::time_point& from,
	const std::chrono::steady_clock::time_point& to)
{
	return std::chrono::duration<double, std::ratio<1>>(to - from).count();
}

// One PVZLEAF1 block per root, in root order: the value net's encoding of the position as the core
// holds it the instant it imported it.  An unsupported root still gets a block (all zeroes) because the
// reader pairs blocks with root indices -- dropping one would shift every later root onto the wrong
// frame, which is the exact failure mode this check exists to catch.
// The student's own encoder at the root: BuildFeatures with the loaded net's widths, then a compact
// [channels][cells] + [scalars] block per root.  Unlike WriteRootFeatures this reads the history fields
// the root carries, so a v2lite net sees the birth ticks the exporter copied.
int WriteStudentFeatures(const Options& options, const Corpus& corpus, Lane lane,
	std::vector<int32_t>& ints, std::vector<float>& floats, const std::vector<float>& waveLog,
	Activations& activations, const NetParams& net)
{
	const int32_t used = static_cast<int32_t>(corpus.roots.size());
	const int32_t perRoot = net.boardChannels * kCells + net.scalarCount;
	std::FILE* stream = std::fopen(options.studentFeatures.c_str(), "wb");
	if (!stream) Fail("cannot write " + options.studentFeatures);
	const char magic[8] = { 'P', 'V', 'Z', 'S', 'T', 'U', '0', '1' };
	const int32_t counts[2] = { used, perRoot };
	std::fwrite(magic, 1, sizeof(magic), stream);
	std::fwrite(counts, sizeof(int32_t), 2, stream);
	std::vector<float> block(static_cast<size_t>(perRoot), 0.0f);
	int32_t encoded = 0;
	for (int32_t index = 0; index < used; ++index)
	{
		std::fill(ints.begin(), ints.end(), 0);
		std::fill(floats.begin(), floats.end(), 0.0f);
		std::fill(block.begin(), block.end(), 0.0f);
		if (LoadRoot(lane, corpus.roots[static_cast<size_t>(index)], corpus.level))
		{
			BuildFeatures(lane, corpus.level, activations.state, waveLog.data(), false,
				net.boardChannels, net.scalarCount, net.version == 4);
			for (int32_t channel = 0; channel < net.boardChannels; ++channel)
				for (int32_t cell = 0; cell < kCells; ++cell)
					block[static_cast<size_t>(channel) * kCells + cell] =
						CellChannel(lane, activations.state.board, net.boardChannels, channel, cell);
			for (int32_t scalar = 0; scalar < net.scalarCount; ++scalar)
				block[static_cast<size_t>(net.boardChannels) * kCells + scalar] =
					ScalarChannel(lane, activations.state.scalars, scalar);
			++encoded;
		}
		std::fwrite(block.data(), sizeof(float), block.size(), stream);
	}
	if (std::fclose(stream) != 0) Fail("write failed: " + options.studentFeatures);
	std::printf("student features: %d of %d roots encoded into %s (%d floats each, %d channels)\n",
		encoded, used, options.studentFeatures.c_str(), perRoot, net.boardChannels);
	return encoded == used ? 0 : 1;
}
int WriteRootFeatures(const Options& options, const Corpus& corpus, Lane lane,
	std::vector<int32_t>& ints, std::vector<float>& floats, const std::vector<float>& waveLog)
{
	const int32_t used = static_cast<int32_t>(corpus.roots.size());
	std::FILE* stream = std::fopen(options.rootFeatures.c_str(), "wb");
	if (!stream) Fail("cannot write " + options.rootFeatures);
	const char magic[8] = { 'P', 'V', 'Z', 'L', 'E', 'A', 'F', '1' };
	const int32_t counts[2] = { used, kValueFloats };
	std::fwrite(magic, 1, sizeof(magic), stream);
	std::fwrite(counts, sizeof(int32_t), 2, stream);
	std::vector<float> block(static_cast<size_t>(kValueFloats), 0.0f);
	int32_t encoded = 0;
	for (int32_t index = 0; index < used; ++index)
	{
		std::fill(ints.begin(), ints.end(), 0);
		std::fill(floats.begin(), floats.end(), 0.0f);
		std::fill(block.begin(), block.end(), 0.0f);
		if (LoadRoot(lane, corpus.roots[static_cast<size_t>(index)], corpus.level))
		{
			StoreLeafValueFeatures(lane, corpus.level, waveLog.data(), block.data());
			++encoded;
		}
		std::fwrite(block.data(), sizeof(float), block.size(), stream);
	}
	if (std::fclose(stream) != 0) Fail("write failed: " + options.rootFeatures);
	std::printf("root features: %d of %d roots encoded into %s (%.1f MB, %d floats each)\n",
		encoded, used, options.rootFeatures.c_str(),
		(static_cast<double>(kValueFloats) * 4.0 * used + 16.0) / 1048576.0, kValueFloats);
	return encoded == used ? 0 : 1;
}

}  // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
	const Options options = Parse(argc, argv);
	Corpus corpus = ReadCorpus(options.roots);
	NetImage image = ReadNet(options.net);

	const int32_t rootCount = static_cast<int32_t>(corpus.roots.size());
	if (options.rootLimit > 0 && options.rootLimit < rootCount)
	{
		corpus.roots.resize(static_cast<size_t>(options.rootLimit));
	}
	const int32_t used = static_cast<int32_t>(corpus.roots.size());
	std::vector<int32_t> pairRoot, pairSeed;
	std::vector<pvzhost::FirstAction> pairFirst;
	if (!options.pairs.empty())
	{
		std::string error;
		if (!ReadPairList(options.pairs, pairRoot, pairSeed, pairFirst, error)) Fail(error);
		for (size_t index = 0; index < pairRoot.size(); ++index)
			if (pairRoot[index] >= used)
				Fail("pair " + std::to_string(index) + " names root " + std::to_string(pairRoot[index])
					+ " but this run uses " + std::to_string(used) + " (see --root-limit)");
	}
	// Same rule as the device driver: an enumerated workload is one rollout per line.
	const int32_t distinct = pairRoot.empty() ? used * options.perRoot : static_cast<int32_t>(pairRoot.size());
	const int32_t total = pairRoot.empty() ? distinct * options.repeat : distinct;

	Activations activations;
	activations.Build(image.params.width, image.params.contextSize);
	// Only the scalar vector is poisoned: the board's halo and the planes' halo must stay zero forever or
	// the padded-grid trick that makes a tap a constant offset stops being equivalent to zero padding.
	if (options.poison != 0.0f)
	{
		std::fill(activations.scalars.begin(), activations.scalars.end(), options.poison);
		std::printf("diagnostic: scalars pre-filled with %.6g (board/plane halo left at 0)\n",
			static_cast<double>(options.poison));
	}
	const std::vector<float> waveLog = WaveLogTable();

	// The wave table is the future the original already committed to; it is indexed by root, so the host
	// keeps the same array the device uploads.
	std::vector<WaveTable> waves(static_cast<size_t>(used));
	for (int32_t index = 0; index < used; ++index)
	{
		const RootRecord& root = corpus.roots[index];
		WaveTable& table = waves[index];
		table.waveCount = root.waveCount;
		for (int32_t wave = 0; wave < kMaxWaves; ++wave)
		{
			table.zombieCount[wave] = root.waveZombieCount[wave];
			for (int32_t slot = 0; slot < kWaveZombieSlots; ++slot)
				table.type[wave][slot] = root.waveTable[wave][slot];
		}
	}

	// ASan needs red zones around the SoA, so the arrays are heap allocated at exactly their declared
	// size rather than padded: an out-of-range slot index then lands in a red zone.
	std::vector<int32_t> ints(static_cast<size_t>(IntSlotsTotal()), 0);
	std::vector<float> floats(static_cast<size_t>(FloatSlotsTotal()), 0.0f);
	// One lane, stride 1, and the activation slot is 0 too: the batched trunk never runs here, so the
	// single lane always writes chunk 0, local lane 0.
	Lane lane{ints.data(), floats.data(), 0, 1, 0};

	std::printf("host core: %d supported roots used of %d, IntSlotsTotal %d, FloatSlotsTotal %d\n",
		used, rootCount, IntSlotsTotal(), FloatSlotsTotal());
	std::printf("student %s: width %d, %s, %s context, %zu weights; activations %.1f MB (one chunk of %d lanes)\n",
		image.params.version == 4 ? "PVZRN004" : (image.params.version == 3 ? "PVZRN003" : "PVZRN002"),
		image.params.width,
		image.params.meanmax ? "meanmax" : "mean", image.params.taskContext ? "with" : "without",
		image.values.size(), static_cast<double>(activations.bytes()) / 1048576.0, kNetChunk);
	std::printf("level %d, %d waves, %d playable rows, %d deck cards; workload %d roots x %d futures"
		" x %d repeats = %d rollouts, horizon %d ticks, max-actions %d, seeds %d..%d, bench %s\n",
		corpus.level.level, corpus.level.numWaves, corpus.level.playableRows, corpus.level.plantCount,
		used, options.perRoot, options.repeat, total, options.horizon, options.maxActions,
		options.seedBase, options.seedBase + options.perRoot - 1, options.useNet ? "full" : "sim");

	// The encoder check says nothing about the tick loop and the ASan batch costs seconds per rollout, so
	// it runs alone and returns.
	if (!options.rootFeatures.empty())
		return WriteRootFeatures(options, corpus, lane, ints, floats, waveLog);
	if (!options.studentFeatures.empty())
		return WriteStudentFeatures(options, corpus, lane, ints, floats, waveLog, activations, image.params);

	Leaf aggregate;
	std::vector<Leaf> perRoot(static_cast<size_t>(used));
	int32_t rejected = 0;

	// The same "PVZLEAF1" dump cuda/RolloutCuda.cu writes, so the two files can be diffed block by block.
	std::FILE* leafStream = nullptr;
	std::vector<float> leafBlock;
	if (!options.leafOut.empty())
	{
		leafBlock.assign(static_cast<size_t>(kValueFloats), 0.0f);
		leafStream = std::fopen(options.leafOut.c_str(), "wb");
		if (!leafStream) Fail("cannot write " + options.leafOut);
		const char magic[8] = { 'P', 'V', 'Z', 'L', 'E', 'A', 'F', '1' };
		const int32_t counts[2] = { total, kValueFloats };
		std::fwrite(magic, 1, sizeof(magic), leafStream);
		std::fwrite(counts, sizeof(int32_t), 2, leafStream);
	}

	const auto startTime = std::chrono::steady_clock::now();
	for (int32_t repeat = 0; repeat < options.repeat; ++repeat)
	{
		// Same root-major mapping as RolloutCuda. Derived repeats advance each root's future set;
		// explicit --pairs repeat the named workload verbatim, which is
		// how a GPU continuation gets compared against the CPU reference row that ran the same future.
		for (int32_t slot = 0; slot < distinct; ++slot)
		{
			const int32_t rootIndex = pairRoot.empty() ? slot / options.perRoot
				: pairRoot[static_cast<size_t>(slot)];
			const int32_t futureSeed = pairRoot.empty()
				? RolloutFutureSeed(repeat * distinct + slot, rootCount, options.perRoot, options.seedBase)
				: pairSeed[static_cast<size_t>(slot)];
			const RootRecord& root = corpus.roots[static_cast<size_t>(rootIndex)];

			// The SoA is reset wholesale rather than relying on LoadRoot to clear every field it writes:
			// a field LoadRoot forgets would otherwise carry the previous rollout's value into this one,
			// which is exactly the class of bug this build exists to catch.
			std::fill(ints.begin(), ints.end(), 0);
			std::fill(floats.begin(), floats.end(), 0.0f);
			if (!LoadRoot(lane, root, corpus.level))
			{
				++rejected;
				std::fprintf(stderr, "root %d rejected by LoadRoot\n", rootIndex);
				// A zero block keeps this dump lane-aligned with the device's, which also emits a block for
				// a lane whose root never loaded.
				if (leafStream) std::fwrite(leafBlock.data(), sizeof(float), leafBlock.size(), leafStream);
				continue;
			}
			lane.s(S_ROOT_INDEX) = rootIndex;
			lane.s(S_UNTIL_TICK) = root.tick + options.horizon;
			// The teacher's candidate, if this line named one.  The SoA was zeroed above, so an
			// enumerated workload without actions is already "kind 0 = the student decides"; with actions it
			// is the same five writes cuda/RolloutCuda.cu's InitKernel makes from the uploaded array.
			if (!pairFirst.empty())
			{
				const pvzhost::FirstAction& action = pairFirst[static_cast<size_t>(slot)];
				lane.s(S_FIRST_KIND) = action.kind;
				lane.s(S_FIRST_PACKET) = action.packet;
				lane.s(S_FIRST_ROW) = action.row;
				lane.s(S_FIRST_COL) = action.col;
				lane.s(S_FIRST_TICKS) = action.ticks;
			}
			RunRollout(lane, corpus.level, waves[static_cast<size_t>(rootIndex)], image.params,
				activations.state, waveLog.data(), futureSeed, options.maxActions, options.useNet);

			RolloutResult result{};
			StoreResult(lane, result);
			if (leafStream)
			{
				StoreLeafValueFeatures(lane, corpus.level, waveLog.data(), leafBlock.data());
				std::fwrite(leafBlock.data(), sizeof(float), leafBlock.size(), leafStream);
			}
			perRoot[static_cast<size_t>(rootIndex)].Add(result);
			aggregate.Add(result);
		}
	}
	const auto stopTime = std::chrono::steady_clock::now();
	const double elapsed = Seconds(startTime, stopTime);
	if (leafStream)
	{
		std::fclose(leafStream);
		std::printf("wrote %d leaf feature blocks of %d floats to %s\n", total, kValueFloats,
			options.leafOut.c_str());
	}

	PrintHeader();
	for (int32_t index = 0; index < used; ++index)
	{
		char label[32];
		// The tick the position arrived at goes in the label, so a row whose mean ticks is nowhere near
		// the horizon can be read against a root that was already past it.
		std::snprintf(label, sizeof(label), "root%d@%d", index, corpus.roots[static_cast<size_t>(index)].tick);
		Leaf& bucket = perRoot[static_cast<size_t>(index)];
		// The divisor is what this bucket actually holds: with --pairs a root is not necessarily visited
		// perRoot times.
		bucket.Print(label, static_cast<double>(bucket.rollouts ? bucket.rollouts : 1));
	}
	aggregate.Print("ALL", static_cast<double>(total));

	const int32_t completed = static_cast<int32_t>(aggregate.rollouts);
	if (completed > 0)
	{
		const double perSecond = completed / (elapsed > 0.0 ? elapsed : 1e-9);
		std::printf("\nwall %.1f s for %d rollouts -> %.2f rollouts/s, %.3f Mticks/s (one thread)\n",
			elapsed, completed, perSecond,
			static_cast<double>(aggregate.ticks) / elapsed / 1.0e6);
		std::printf("per rollout: %.1f ticks, %.1f decisions, %.1f net forwards, %.1f planted, %.1f eaten,"
			" %.1f killed, %.1f sun, %.2f wave, %.2f mowers left, %.1f zombies alive, %.1f plants alive\n",
			static_cast<double>(aggregate.ticks) / completed,
			static_cast<double>(aggregate.decisions) / completed,
			static_cast<double>(aggregate.netCalls) / completed,
			static_cast<double>(aggregate.planted) / completed,
			static_cast<double>(aggregate.eaten) / completed,
			static_cast<double>(aggregate.killed) / completed,
			static_cast<double>(aggregate.sun) / completed,
			static_cast<double>(aggregate.wave) / completed,
			static_cast<double>(aggregate.mowers) / completed,
			static_cast<double>(aggregate.zombiesAlive) / completed,
			static_cast<double>(aggregate.plantsAlive) / completed);
		std::printf("leaf results: won %lld, lost %lld, ongoing %lld; rollouts with an error flag %lld;"
			" rejected roots %d\n",
			static_cast<long long>(aggregate.won), static_cast<long long>(aggregate.lost),
			static_cast<long long>(aggregate.ongoing), static_cast<long long>(aggregate.errors), rejected);
		std::printf("reference: CPU baseline 90.18 rollouts/s/thread (original simulator + OpenBLAS)\n");
	}
	return completed > 0 && rejected == 0 ? 0 : 1;
}
