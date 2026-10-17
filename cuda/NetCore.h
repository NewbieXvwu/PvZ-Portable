// The student network, as a plain struct of device pointers.
//
// Same dual-compile rule as the rest of the core: this header has no CUDA syntax, and NetForward.inc
// is compiled by nvcc (thread = lane) and by g++ (single lane) from one source.  The reference
// implementation is src/EnvironmentRollout.cpp -- the CPU path the original simulator uses -- and the
// weights come from the PVZRN dump of python/pvz_rollout_policy.py::export_student, whose tensor order
// and element counts are transcribed below from EnvironmentRollout::Load so the two builds read the
// same bytes.
//
// Layout of the per-lane activations follows the SoA rule used everywhere else: an element of a
// [channel][cell] plane for lane L sits at (channel * kCells + cell) * stride + L, so a warp reading
// "channel 7 of cell 12" touches 32 consecutive floats.
#ifndef PVZ_NET_CORE_H
#define PVZ_NET_CORE_H

#include "CudaRollout.h"

#include <cstdint>

#ifdef __CUDACC__
#ifndef PVZ_SIM
#define PVZ_SIM __host__ __device__ inline
#endif
#else
#ifndef PVZ_SIM
#define PVZ_SIM inline
#endif
#endif

namespace pvzc
{

// kScalars already includes the two "previous action" features (113, 114); a PVZRN003 dump appends the
// 1 + kZombieTypes task context after them.
enum { kTaskContext = 1 + kZombieTypes };

struct NetParams
{
	int32_t width;
	int32_t meanmax;
	int32_t taskContext;
	int32_t scalarCount;                  // kScalars or kScalars + kTaskContext
	int32_t contextSize;                  // (meanmax ? 3 : 2) * width
	int32_t version;                      // 2 = PVZRN002, 3 = PVZRN003

	const float* conv1;                   // [width][92][3][3]
	const float* bias1;
	const float* conv2;                   // [width][width][3][3]
	const float* bias2;
	const float* conv3;                   // [width][width][1][5]
	const float* bias3;
	const float* scalarWeight;            // [width][scalarCount]
	const float* scalarBias;
	const float* plantWeight;             // [contextSize]
	const float* plantBias;
	const float* packetWeight;            // [10][contextSize]
	const float* packetBias;
	const float* packetEmbedding;         // [10][width]
	const float* cellWeight;              // [width][2*width + contextSize]
	const float* cellBias;
	const float* cellOut;                 // [width]
	const float* cellOutBias;
};

// One lane's slice of the batched activations.  `hidden` and `scratch` ping-pong through the three
// convolutions, so only two [width][kCells] planes are live at a time.
struct NetState
{
	float* board;                         // kBoardChannels * kCells
	float* scalars;                       // scalarCount
	float* hidden;                        // width * kCells
	float* scratch;                       // width * kCells
	float* context;                       // contextSize
	float* logits;                        // kRows * kGridCols, the cell head's lawn logits
};

}  // namespace pvzc

#endif
