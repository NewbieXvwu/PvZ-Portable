// Does a batched-GEMM student trunk beat the per-lane forward?  Measure before refactoring.
//
// The trunk is three convolutions over the 6x10 feature grid.  Lay the activations out as
// [channel][paddedRow][paddedCol][lane]; then for one filter tap and one grid row, the 10 valid
// columns of that row are a contiguous run of 10*lanes floats, so the whole tap is a single
// strided-batched GEMM batched over the 6 rows.  cuBLAS is column-major and our activations are
// channel-outermost, so the matrices are the transpose of the usual statement: the spatial position
// is the row, the channel is the column, and the weight block per tap holds [outChannel][inChannel]
// with inChannel fastest.  A 3x3 conv is 9 such calls, a 1x5 is 5.  Bias is not modelled here -- the
// real kernel adds it through one extra "constant 1" input channel, which is 1 channel out of 64 and
// does not change the arithmetic intensity this probe is about.
//
// This is a performance probe only: random weights, no simulation, no features.  It answers one
// question -- how many GFLOP/s does the batched trunk get at a realistic lane count -- so the
// three-kernel round design can be chosen from a number instead of a guess.
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace {

#define CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
	std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(2); } } while (0)
#define CHECK_BLAS(call) do { const cublasStatus_t s = (call); if (s != CUBLAS_STATUS_SUCCESS) { \
	std::fprintf(stderr, "%s: cublas status %d\n", #call, (int)s); std::exit(2); } } while (0)

// The grid is 6 rows x 10 columns.  Valid cell (R, C) sits at padded (R + 1, C + 2) so that a 3x3
// with padding 1 and a 1x5 with padding (0, 2) both read inside the buffer, including the halo.
enum { kRows = 6, kCols = 10, kRowsP = 8, kColsP = 14 };
enum { kBoardChannels = 92, kWidth = 64 };
enum { kOutputBase = 1 * kColsP + 2 };

float* Malloc(size_t floats)
{
	float* p = nullptr;
	CHECK(cudaMalloc(&p, floats * sizeof(float)));
	return p;
}

// `shift` is the tap's offset from the output's valid origin, in padded cells.
struct Tap
{
	int32_t shift;
};

Tap Taps3(int32_t dr, int32_t dc) { return {kOutputBase + (dr - 1) * kColsP + (dc - 1)}; }
Tap Taps1x5(int32_t dc) { return {kOutputBase + (dc - 2)}; }

// One convolution: `tapCount` strided-batched GEMMs accumulating into the same output block.
//
// cuBLAS is column-major, so a matrix element (i, k) sits at ptr[i + k * ld].  Our activations put a
// whole [column, lane] run of one channel contiguously and step `plane` floats per channel, which is
// exactly C(spatial, channel) with ld = plane -- so the GEMM computes the trunk *transposed*: the
// spatial index is the matrix row and the channel is the column.  Reading it this way also means the
// weight block per tap must hold [outChannel][inChannel] with inChannel fastest.
// batchCount = the 6 grid rows; per row, M = 10 valid columns x lanes.
void Conv(cublasHandle_t handle, const float* input, const float* weight, int32_t inChannels,
	int32_t outChannels, const Tap* taps, int32_t tapCount, int32_t laneStride, int32_t offset,
	int32_t lanes, float* output)
{
	const int32_t m = kCols * lanes;
	// Channel-to-channel distance = leading dimension; grid-row-to-grid-row = batch stride.  Both are
	// set by the *whole* batch's lane count, because a chunk of lanes is a strided slice of the plane,
	// not a separate buffer.
	const int32_t plane = kRowsP * kColsP * laneStride;
	const long long rowStride = static_cast<long long>(kColsP) * laneStride;
	const float one = 1.0f, zero = 0.0f;
	for (int32_t tap = 0; tap < tapCount; ++tap)
	{
		const cublasStatus_t status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			m, outChannels, inChannels, &one,
			input + static_cast<size_t>(taps[tap].shift) * laneStride + offset, plane, rowStride,
			weight + static_cast<size_t>(tap) * inChannels * outChannels, inChannels, 0LL,
			tap == 0 ? &zero : &one,
			output + static_cast<size_t>(kOutputBase) * laneStride + offset, plane, rowStride, kRows);
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			std::fprintf(stderr,
				"cublas status %d: m %d n %d k %d lda %d ldb %d stride %lld shift %d tap %d of %d, cuda %s\n",
				static_cast<int>(status), m, outChannels, inChannels, plane, inChannels, rowStride,
				taps[tap].shift, tap, tapCount, cudaGetErrorString(cudaGetLastError()));
			std::exit(2);
		}
		// SYNC=1 makes an out-of-range access report the call that caused it instead of the next one.
		if (getenv("SYNC"))
		{
			const cudaError_t e = cudaDeviceSynchronize();
			if (e != cudaSuccess)
			{
				std::fprintf(stderr, "sync after k %d tap %d: %s\n", inChannels, tap,
					cudaGetErrorString(e));
				std::exit(2);
			}
		}
	}
}

// ReLU over the valid cells of one lane chunk.  The halo cells a convolution never writes stay zero,
// which is what the next convolution's padding reads.
__global__ void Relu(float* plane, int32_t laneStride, int32_t offset, int32_t lanes)
{
	const int32_t total = kWidth * kRows * kCols * lanes;
	for (int32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
		index += blockDim.x * gridDim.x)
	{
		const int32_t lane = index % lanes;
		const int32_t cell = index / lanes;
		const int32_t channel = cell / (kRows * kCols);
		const int32_t row = (cell / kCols) % kRows, column = cell % kCols;
		float* slot = plane + static_cast<size_t>((channel * kRowsP + row + 1) * kColsP + column + 2)
			* laneStride + offset + lane;
		*slot = *slot > 0.0f ? *slot : 0.0f;
	}
}

}  // namespace

int main(int argc, char** argv)
{
	// A chunk of lanes smaller than the batch keeps the 9-tap read-modify-write of the output plane
	// inside L2 instead of DRAM.  0 = one chunk over the whole batch.
	const int32_t chunk = argc > 5 ? std::atoi(argv[5]) : 0;
	const int32_t lanes = argc > 1 ? std::atoi(argv[1]) : 21504;
	const int32_t generations = argc > 2 ? std::atoi(argv[2]) : 43;
	const int32_t withRelu = argc > 3 ? std::atoi(argv[3]) : 1;
	// 1 = allow TF32 tensor cores.  Not free: it changes the rounding, so phase 1 agreement has to be
	// measured with it before it can be used.  Measured here to know how much headroom is behind FP32.
	const int32_t tf32 = argc > 4 ? std::atoi(argv[4]) : 0;

	const int32_t chunkWidth = chunk <= 0 || chunk > lanes ? lanes : chunk;
	const size_t boardFloats = static_cast<size_t>(kBoardChannels) * kRowsP * kColsP * lanes;
	const size_t planeFloats = static_cast<size_t>(kWidth) * kRowsP * kColsP * lanes;
	float* board = Malloc(boardFloats);
	float* left = Malloc(planeFloats);
	float* right = Malloc(planeFloats);

	// Weights are read per tap as [outChannel][inChannel] with inChannel fastest (the column-major
	// K x N block cuBLAS wants for B); the real importer transposes the checkpoint's
	// [outChannel][inChannel][tap] into that.  Values are irrelevant to the timing.
	const size_t weightFloats =
		(9ull * kBoardChannels * kWidth + 9ull * kWidth * kWidth + 5ull * kWidth * kWidth) * kWidth;
	float* weights = nullptr;
	CHECK(cudaMalloc(&weights, weightFloats * sizeof(float)));
	{
		float* host = new float[weightFloats];
		for (size_t i = 0; i < weightFloats; ++i) host[i] = 0.01f * static_cast<float>(i % 97);
		CHECK(cudaMemcpy(weights, host, weightFloats * sizeof(float), cudaMemcpyHostToDevice));
		delete[] host;
	}

	// Finite inputs: an uninitialised -nan/inf pattern can push cuBLAS down different code paths.
	CHECK(cudaMemset(board, 0, boardFloats * sizeof(float)));
	CHECK(cudaMemset(left, 0, planeFloats * sizeof(float)));
	CHECK(cudaMemset(right, 0, planeFloats * sizeof(float)));

	cublasHandle_t handle = nullptr;
	CHECK_BLAS(cublasCreate(&handle));
	CHECK_BLAS(cublasSetMathMode(handle,
		tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));

	Tap taps3[9], taps1x5[5];
	for (int32_t dr = 0; dr < 3; ++dr)
		for (int32_t dc = 0; dc < 3; ++dc) taps3[dr * 3 + dc] = Taps3(dr, dc);
	for (int32_t dc = 0; dc < 5; ++dc) taps1x5[dc] = Taps1x5(dc);

	const double flops = 2.0 * lanes * kRows * kCols
		* (kWidth * kBoardChannels * 9 + kWidth * kWidth * 9 + kWidth * kWidth * 5);

	const int32_t block = 256;
	const int32_t grid = 2048;
	auto relu = [&](float* plane, int32_t offset, int32_t width)
	{
		if (!withRelu) return;
		Relu<<<grid, block>>>(plane, lanes, offset, width);
	};

	auto runOnce = [&]()
	{
		for (int32_t offset = 0; offset < lanes; offset += chunkWidth)
		{
			const int32_t width = chunkWidth < lanes - offset ? chunkWidth : lanes - offset;
			Conv(handle, board, weights, kBoardChannels, kWidth, taps3, 9, lanes, offset, width, left);
			relu(left, offset, width);
			Conv(handle, left, weights + 9ull * kBoardChannels * kWidth, kWidth, kWidth, taps3, 9, lanes,
				offset, width, right);
			relu(right, offset, width);
			Conv(handle, right, weights + (9ull * kBoardChannels * kWidth + 9ull * kWidth * kWidth),
				kWidth, kWidth, taps1x5, 5, lanes, offset, width, left);
			relu(left, offset, width);
		}
	};
	for (int32_t i = 0; i < 2; ++i) runOnce();
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start));
	for (int32_t i = 0; i < generations; ++i) runOnce();
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	float ms = 0.0f;
	CHECK(cudaEventElapsedTime(&ms, start, stop));

	const double seconds = ms / 1000.0;
	std::printf("lanes %d chunk %d, %d decision rounds, relu %d tf32 %d, %.1f ms total\n",
		lanes, chunkWidth, generations, withRelu, tf32, ms);
	std::printf("trunk: %.2f TFLOP/s sustained (%.1f GFLOP per round, %.3f ms per round)\n",
		flops * generations / seconds / 1e12, flops / 1e9, ms / generations);
	std::printf("=> %.2f us of trunk per forward per lane, %.0f rollouts/s from the trunk alone\n",
		1e3 * ms / generations / lanes, lanes * 1000.0 / ms);
	std::printf("   (per-lane in-kernel forward measured 9.6 us; a rollout is 43 forwards)\n");
	return 0;
}
