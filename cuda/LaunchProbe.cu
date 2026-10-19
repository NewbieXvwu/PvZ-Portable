// Is the ~12 ms per rollout round the price of *talking to the GPU*, or the price of running our kernels?
//
// The measurement that motivated this file: a round of cuda/RolloutCuda.cu costs 13.4 ms at 10 lanes and
// 12 ms at 8192 -- 800x the work for the same time.  Something fixed is being paid ~50 times per rollout,
// and until it is named the server seam is just guessing.  Three candidates, measured here one at a time:
//
//   launch      the cost of enqueueing a kernel (30 per round: 24 cuBLAS GEMMs, 4 ReLU passes, kernels)
//   sync        the cost of asking "is it done yet" (1 per round, and it cannot be removed -- the loop
//               needs netCount), which under WSL2 may be a hypervisor round trip rather than a spin
//   graph       the same 30 launches captured once and replayed as a single cudaGraphLaunch
//
// Build:
//   /usr/local/cuda/bin/nvcc -arch=sm_120 -O3 -std=c++17 -o build/launch_probe cuda/LaunchProbe.cu
// Run:
//   ./build/launch_probe [--rounds 49] [--spinsync]
#ifndef PVZ_LAUNCH_PROBE_CU
#define PVZ_LAUNCH_PROBE_CU

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

__global__ void Nop(float* p)
{
	if (threadIdx.x == 0) p[blockIdx.x] += 1.0f;
}

struct Timer
{
	std::chrono::steady_clock::time_point began = std::chrono::steady_clock::now();
	double Ms() const
	{
		return std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - began).count();
	}
};

#define CALL(expression) do { const cudaError_t error = (expression); \
	if (error != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #expression, \
		cudaGetErrorString(error)); return 1; } } while (0)

int main(int argc, char** argv)
{
	int32_t rounds = 49;                       // what a 2500-tick continuation takes
	int32_t launches = 30;                    // what a round of the batched trunk enqueues
	int32_t repeats = 20;                     // whole rollouts, so a steady-state number is reported
	bool spin = false;
	for (int index = 1; index < argc; ++index)
	{
		const std::string flag = argv[index];
		if (flag == "--rounds") rounds = std::atoi(argv[++index]);
		else if (flag == "--launches") launches = std::atoi(argv[++index]);
		else if (flag == "--repeats") repeats = std::atoi(argv[++index]);
		else if (flag == "--spinsync") spin = true;
		else { std::fprintf(stderr, "unknown flag %s\n", flag.c_str()); return 1; }
	}

	// The flag has to be set before the context exists, so it is set here rather than in the driver.
	if (spin) CALL(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
	CALL(cudaSetDevice(0));
	float* device = nullptr;
	float* host = nullptr;
	CALL(cudaMalloc(&device, 1024 * sizeof(float)));
	CALL(cudaMallocHost(&host, sizeof(float)));
	CALL(cudaMemset(device, 0, 1024 * sizeof(float)));

	// One command each, timed alone: launch-only, sync-only, and the pair, so the round's cost can be
	// attributed instead of asserted.
	std::printf("rounds %d x launches %d, %d rollouts, schedule %s\n", rounds, launches, repeats,
		spin ? "spin" : "auto");

	// (a) enqueue without waiting: the per-launch cost as seen by the host thread.
	CALL(cudaDeviceSynchronize());
	{
		Timer timer;
		for (int32_t repeat = 0; repeat < repeats; ++repeat)
			for (int32_t round = 0; round < rounds; ++round)
				for (int32_t launch = 0; launch < launches; ++launch) Nop<<<1, 32>>>(device);
		const double enqueued = timer.Ms();
		CALL(cudaDeviceSynchronize());
		const double total = timer.Ms();
		std::printf("enqueue only      : %7.1f ms per rollout, %.1f us per launch"
			" (drain included %.1f ms)\n", total / repeats, enqueued * 1000.0
			/ (repeats * static_cast<double>(rounds) * launches), (total - enqueued) / repeats);
	}

	// (b) one launch then a sync: the round-trip, which is what the round loop pays per round.
	CALL(cudaDeviceSynchronize());
	{
		Timer timer;
		for (int32_t repeat = 0; repeat < repeats; ++repeat)
			for (int32_t round = 0; round < rounds; ++round)
			{
				Nop<<<1, 32>>>(device);
				CALL(cudaStreamSynchronize(0));
			}
		const double total = timer.Ms();
		std::printf("launch + sync     : %7.1f ms per rollout, %.2f ms per round\n",
			total / repeats, total / (repeats * static_cast<double>(rounds)));
	}

	// (c) the round's own shape: `launches` kernels then one sync.
	CALL(cudaDeviceSynchronize());
	{
		Timer timer;
		for (int32_t repeat = 0; repeat < repeats; ++repeat)
			for (int32_t round = 0; round < rounds; ++round)
			{
				for (int32_t launch = 0; launch < launches; ++launch) Nop<<<1, 32>>>(device);
				CALL(cudaStreamSynchronize(0));
			}
		const double total = timer.Ms();
		std::printf("%d launches + sync : %7.1f ms per rollout, %.2f ms per round\n", launches,
			total / repeats, total / (repeats * static_cast<double>(rounds)));
	}

	// (d) the same, captured into a graph: one cudaGraphLaunch and one sync per round.
	{
		cudaGraph_t graph = nullptr;
		cudaGraphExec_t executable = nullptr;
		CALL(cudaStreamBeginCapture(0, cudaStreamCaptureModeThreadLocal));
		for (int32_t launch = 0; launch < launches; ++launch) Nop<<<1, 32>>>(device);
		CALL(cudaStreamEndCapture(0, &graph));
		CALL(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
		// The first replay builds whatever the instantiation deferred; do not time it.
		for (int32_t warm = 0; warm < 3; ++warm)
		{
			CALL(cudaGraphLaunch(executable, 0));
			CALL(cudaStreamSynchronize(0));
		}
		Timer timer;
		for (int32_t repeat = 0; repeat < repeats; ++repeat)
			for (int32_t round = 0; round < rounds; ++round)
			{
				CALL(cudaGraphLaunch(executable, 0));
				CALL(cudaStreamSynchronize(0));
			}
		const double total = timer.Ms();
		std::printf("graph + sync      : %7.1f ms per rollout, %.2f ms per round\n",
			total / repeats, total / (repeats * static_cast<double>(rounds)));
		CALL(cudaGraphExecDestroy(executable));
		CALL(cudaGraphDestroy(graph));
	}

	// (e) is the machine asleep?  A back-to-back stream of work has no gaps for the clock to fall out of,
	// so if this is far cheaper per launch than (a) the fixed cost above is DVFS, not the API.
	{
		CALL(cudaDeviceSynchronize());
		Timer timer;
		int32_t count = 0;
		while (timer.Ms() < 400.0)
		{
			for (int32_t launch = 0; launch < 200; ++launch) Nop<<<1, 32>>>(device);
			count += 200;
			CALL(cudaDeviceSynchronize());
		}
		const double total = timer.Ms();
		std::printf("saturated enqueue : %.1f us per launch over %d launches\n",
			total * 1000.0 / count, count);
	}
	CALL(cudaGetLastError());
	return 0;
}

#endif
