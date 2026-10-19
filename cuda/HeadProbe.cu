// Where the per-lane half of the student forward actually goes.
//
// The round-split driver (cuda/RolloutCuda.cu) attributes ~20 ms of a 38 ms round to "heads+apply", and
// only a part of that can be RoundApply's wait ticks (the whole batch's tick volume runs in 395 ms in
// --bench sim).  The head loops are ~4% of a forward's arithmetic yet cost several times the batched
// trunk, so something about them is pathological -- and the game is too entangled a harness to find out,
// because ablating a head also changes the decisions and therefore the number of ticks.  This probe runs
// each piece on its own over a batch of activation slots and prints nanoseconds per lane-forward.
//
// Weights and activations are synthetic: the arithmetic, and therefore the timing, does not depend on
// the values, and keeping the checkpoint out of it means the probe still measures the same thing after a
// change to the net.
//
// Build:
//   /usr/local/cuda/bin/nvcc -arch=sm_120 -O3 -std=c++17 --expt-relaxed-constexpr -I. \
//       -o build/head_probe cuda/HeadProbe.cu
// Run:
//   LD_LIBRARY_PATH=/usr/local/cuda/lib64 ./build/head_probe [lanes] [iterations]
#ifndef PVZ_HEAD_PROBE_CU
#define PVZ_HEAD_PROBE_CU

#include "SimCore.h"
#include "NetForward.inc"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace headprobe {

using namespace pvzc;

void Checked(cudaError_t error, const char* what)
{
	if (error == cudaSuccess) return;
	std::fprintf(stderr, "head-probe: %s: %s\n", what, cudaGetErrorString(error));
	std::exit(1);
}

#define CALL(expression) Checked((expression), #expression)

// The pieces PostKernel runs per lane, in the order it runs them.  The scalar linear is separated from
// NetContext because it is the one part of the context whose input is per-lane contiguous.
enum Stage { kStageMeanmax = 0, kStageScalar = 1, kStageLogits = 2, kStageCell = 3, kStageCount = 4 };

struct Buffers
{
	NetParams net;
	NetState state;
	int32_t lanes;
};

__global__ void __launch_bounds__(256) RunKernel(Buffers buffers, int32_t stage, int32_t iterations,
	float* sink)
{
	const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffers.lanes) return;
	// netSlot = index: the probe gives every lane its own activation slot, which is what RoundPre's
	// compaction produces.  No lane or level state is needed -- none of these functions read it.
	Lane lane{nullptr, nullptr, index, buffers.lanes, index};
	const NetParams& net = buffers.net;
	const NetState& state = buffers.state;
	float acc = 0.0f;
	for (int32_t step = 0; step < iterations; ++step)
	{
		if (stage == kStageMeanmax)
		{
			const int32_t width = net.width;
			const int32_t context = net.contextSize;
			for (int32_t oc = 0; oc < width; ++oc)
			{
				float total = 0.0f;
				float largest = 0.0f;
				for (int32_t cell = 0; cell < kCells; ++cell)
				{
					const float value = buffers.state.hidden[PlaneIndex(width, lane.netSlot, oc,
						PaddedCell(cell))];
					total += value;
					if (value > largest) largest = value;
				}
				state.context[VectorIndex(context, lane.netSlot, oc)] =
					total / static_cast<float>(kCells);
				if (net.meanmax) state.context[VectorIndex(context, lane.netSlot, width + oc)] = largest;
			}
		}
		else if (stage == kStageScalar)
		{
			const int32_t width = net.width;
			const int32_t context = net.contextSize;
			for (int32_t oc = 0; oc < width; ++oc)
			{
				const float* weight = net.scalarWeight + oc * net.scalarCount;
				float sum = net.scalarBias[oc];
				for (int32_t item = 0; item < net.scalarCount; ++item)
					sum += state.scalars[VectorIndex(kScalarFeatures, lane.netSlot, item)] * weight[item];
				state.context[VectorIndex(context, lane.netSlot, context - width + oc)] = Relu(sum);
			}
		}
		else if (stage == kStageLogits)
		{
			acc += PlantLogit(lane, net, state);
			for (int32_t packet = 0; packet < kPacketSlots; ++packet)
				acc += PacketLogit(lane, net, state, packet);
		}
		else
			CellLogits(lane, net, state, index % kPacketSlots);
	}
	if (acc == 12345.0f) sink[index] = acc;
}

}  // namespace headprobe

int main(int argc, char** argv)
{
	using namespace headprobe;

	const int32_t lanes = argc > 1 ? std::atoi(argv[1]) : 7519;
	const int32_t iterations = argc > 2 ? std::atoi(argv[2]) : 50;
	// The forward count of the measured run (21,504 rollouts x 18.5 net forwards), so a stage's number
	// can be multiplied straight out into the driver's phase split.
	const double forwardsPerRun = 21504.0 * 18.5;

	const int32_t width = 64, meanmax = 1, taskContext = 0;
	const int32_t scalarCount = taskContext ? kScalars + kTaskContext : kScalarFeatures;
	const int32_t context = meanmax ? 3 * width : 2 * width;
	const int32_t blocks = (lanes + kNetChunk - 1) / kNetChunk;

	// Tensor sizes in the order NetForward.inc indexes them.
	std::vector<size_t> sizes;
	sizes.push_back(static_cast<size_t>(width) * kBoardChannels * 9);   // conv1
	sizes.push_back(width);                                             // bias1
	sizes.push_back(static_cast<size_t>(width) * width * 9);            // conv2
	sizes.push_back(width);                                             // bias2
	sizes.push_back(static_cast<size_t>(width) * width * 5);            // conv3
	sizes.push_back(width);                                             // bias3
	sizes.push_back(static_cast<size_t>(width) * scalarCount);          // scalarWeight
	sizes.push_back(width);                                             // scalarBias
	sizes.push_back(context);                                           // plantWeight
	sizes.push_back(1);                                                 // plantBias
	sizes.push_back(static_cast<size_t>(kPacketSlots) * context);       // packetWeight
	sizes.push_back(kPacketSlots);                                      // packetBias
	sizes.push_back(static_cast<size_t>(kPacketSlots) * width);         // packetEmbedding
	sizes.push_back(static_cast<size_t>(width) * (2 * width + context));  // cellWeight
	sizes.push_back(width);                                             // cellBias
	sizes.push_back(width);                                             // cellOut
	sizes.push_back(1);                                                 // cellOutBias
	size_t total = 0;
	for (size_t size : sizes) total += size;

	std::vector<float> hostWeights(total);
	for (size_t index = 0; index < total; ++index)
		hostWeights[index] = 0.01f * static_cast<float>(index % 97);
	float* weights = nullptr;
	CALL(cudaMalloc(&weights, total * sizeof(float)));
	CALL(cudaMemcpy(weights, hostWeights.data(), total * sizeof(float), cudaMemcpyHostToDevice));
	float* cursor = weights;
	auto take = [&cursor](size_t count)
	{
		float* pointer = cursor;
		cursor += count;
		return pointer;
	};

	Buffers buffers;
	buffers.lanes = lanes;
	buffers.net.width = width;
	buffers.net.meanmax = meanmax;
	buffers.net.taskContext = taskContext;
	buffers.net.scalarCount = scalarCount;
	buffers.net.contextSize = context;
	buffers.net.version = taskContext ? 3 : 2;
	buffers.net.conv1 = take(sizes[0]); buffers.net.bias1 = take(sizes[1]);
	buffers.net.conv2 = take(sizes[2]); buffers.net.bias2 = take(sizes[3]);
	buffers.net.conv3 = take(sizes[4]); buffers.net.bias3 = take(sizes[5]);
	buffers.net.scalarWeight = take(sizes[6]); buffers.net.scalarBias = take(sizes[7]);
	buffers.net.plantWeight = take(sizes[8]); buffers.net.plantBias = take(sizes[9]);
	buffers.net.packetWeight = take(sizes[10]); buffers.net.packetBias = take(sizes[11]);
	buffers.net.packetEmbedding = take(sizes[12]);
	buffers.net.cellWeight = take(sizes[13]); buffers.net.cellBias = take(sizes[14]);
	buffers.net.cellOut = take(sizes[15]); buffers.net.cellOutBias = take(sizes[16]);

	auto plane = [&](size_t channels)
	{
		float* pointer = nullptr;
		const size_t bytes = static_cast<size_t>(blocks) * channels * kPlaneCells * kNetChunk * sizeof(float);
		CALL(cudaMalloc(&pointer, bytes));
		// 0x30303030 is a finite normal float: an all-zero plane would let a compiler-visible shortcut
		// skip work, and an uninitialised one can be -nan/inf and take a different path.
		CALL(cudaMemset(pointer, 0x30, bytes));
		return pointer;
	};
	// A vector buffer is chunked the same way but has no padded-cell dimension.
	auto vector = [&](size_t items)
	{
		float* pointer = nullptr;
		const size_t bytes = static_cast<size_t>(blocks) * items * kNetChunk * sizeof(float);
		CALL(cudaMalloc(&pointer, bytes));
		CALL(cudaMemset(pointer, 0x30, bytes));
		return pointer;
	};
	buffers.state.board = plane(kBoardChannels);
	buffers.state.hidden = plane(width);
	buffers.state.scratch = plane(width);
	buffers.state.scalars = vector(kScalarFeatures);
	buffers.state.context = vector(context);
	buffers.state.logits = vector(kLogitCells);
	float* sink = vector(1);

	cudaEvent_t start, stop;
	CALL(cudaEventCreate(&start));
	CALL(cudaEventCreate(&stop));
	const int32_t block = 256;
	const int32_t grid = (lanes + block - 1) / block;

	const char* names[kStageCount] = {"context mean/max", "context scalar linear",
		"plant + packet logits", "cell head"};
	// MACs per lane-forward of each stage, so the second column says how far from the machine's FP32
	// peak the stage is running.  The cell head's is width * cells * width for the 1x1 plus the
	// width-way combine of each lawn cell and the embedding/context bases.
	const double macs[kStageCount] = {
		static_cast<double>(width) * kCells,
		static_cast<double>(width) * scalarCount,
		static_cast<double>(context) * (1 + kPacketSlots),
		static_cast<double>(width) * (2 * width + context + kCells * width)
			+ kLogitCells * width
	};

	std::printf("lanes %d in %d blocks of %d, %d iterations, width %d, context %d\n\n", lanes, grid,
		block, iterations, width, context);
	double sumPerRun = 0.0;
	for (int32_t stage = 0; stage < kStageCount; ++stage)
	{
		RunKernel<<<grid, block>>>(buffers, stage, 1, sink);   // one untimed pass to page in the buffers
		CALL(cudaEventRecord(start));
		RunKernel<<<grid, block>>>(buffers, stage, iterations, sink);
		CALL(cudaEventRecord(stop));
		CALL(cudaEventSynchronize(stop));
		float elapsed = 0.0f;
		CALL(cudaEventElapsedTime(&elapsed, start, stop));
		Checked(cudaGetLastError(), "launch");
		const double perForwardNs = static_cast<double>(elapsed) * 1e6 / iterations;   // per batch pass
		const double amortised = perForwardNs / lanes;                                 // per lane-forward
		const double gFlops = 2.0 * macs[stage] / 1e9;
		const double achieved = amortised > 0 ? gFlops / amortised * 1e6 : 0.0;   // GFLOP/s
		const double perRun = amortised * forwardsPerRun / 1e6;                        // ms for the run
		sumPerRun += perRun;
		std::printf("%-24s %8.1f ns/forward  %7.1f ms/run  %6.0f GFLOP/s  (%.0f MAC/forward)\n",
			names[stage], amortised, perRun, achieved, macs[stage]);
	}
	std::printf("\ntotal of the per-lane stages: %.0f ms per run of %.0f forwards\n", sumPerRun,
		forwardsPerRun);
	std::printf("the batched trunk measured in the driver: 408 ms for the same forwards (6.62 MMAC each)\n");
	return 0;
}

#endif
