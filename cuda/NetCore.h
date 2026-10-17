// The student network, as a plain struct of device pointers.
//
// Same dual-compile rule as the rest of the core: this header has no CUDA syntax, and NetForward.inc
// is compiled by nvcc (thread = lane) and by g++ (single lane) from one source.  The reference
// implementation is src/EnvironmentRollout.cpp -- the CPU path the original simulator uses -- and the
// weights come from the PVZRN dump of python/pvz_rollout_policy.py::export_student, whose tensor order
// and element counts are transcribed below from EnvironmentRollout::Load so the two builds read the
// same bytes.
//
// Layout of the per-lane activations follows the SoA rule used everywhere else, with the padded grid
// and the lane chunk of cuda/CudaRollout.h: element (channel, padded position pos) of activation slot
// S sits at ((S / kNetChunk) * channels + channel) * kPlaneCells * kNetChunk + pos * kNetChunk +
// (S % kNetChunk), so a warp touching one channel of one cell reads 32 consecutive floats and one tap
// of the batched trunk is a contiguous kCols * kNetChunk run.
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
// The scalars buffer is *addressed* by this capacity on both sides, whether or not the loaded network
// reads the task-context tail: chunked addressing needs a width that every accessor agrees on, and
// net.scalarCount (115 or 150) is only the loop bound of the scalar linear.
enum { kScalarFeatures = kScalars + kTaskContext };
enum { kLogitCells = kRows * kGridCols };

// ---------------------------------------------------------------------------
// Activation addressing
// ---------------------------------------------------------------------------
//
// One formula, used by the host build and the device build and by cuBLAS's pointer arithmetic, so the
// batched trunk and the per-lane forward read the same floats.  A buffer holding `channels` planes is
// [block][channel][paddedRow][paddedCol][lane], each dimension of size
// (batch blocks, channels, kRowsP, kColsP, kNetChunk).

PVZ_SIM int32_t NetBlock(int32_t netSlot) { return netSlot >> kNetChunkBits; }
PVZ_SIM int32_t NetLocal(int32_t netSlot) { return netSlot & kNetChunkMask; }

// Valid cell (row, col) of the 6x10 feature grid, in padded coordinates.
PVZ_SIM int32_t PaddedCell(int32_t cell)
{
	return (cell / kCols + kPadRow) * kColsP + cell % kCols + kPadCol;
}

// Distance from one channel plane to the next, and from one padded grid row to the next: cuBLAS's
// leading dimension and batch stride for a plane buffer of `channels` channels.
PVZ_SIM size_t PlaneLeadingDimension(int32_t channels)
{
	return static_cast<size_t>(channels) * kPlaneCells * kNetChunk;
}
PVZ_SIM size_t PlaneRowStride() { return static_cast<size_t>(kColsP) * kNetChunk; }

// Element (channel, padded position `pos`) of lane `netSlot` in a `channels`-plane buffer.
PVZ_SIM size_t PlaneIndex(int32_t channels, int32_t netSlot, int32_t channel, int32_t pos)
{
	return (static_cast<size_t>(NetBlock(netSlot) * channels + channel) * kPlaneCells
		+ static_cast<size_t>(pos)) * kNetChunk + static_cast<size_t>(NetLocal(netSlot));
}

// Element `index` of a per-lane vector (scalars, context, logits) of `width` entries.
PVZ_SIM size_t VectorIndex(int32_t width, int32_t netSlot, int32_t index)
{
	return static_cast<size_t>(NetBlock(netSlot) * width + index) * kNetChunk
		+ static_cast<size_t>(NetLocal(netSlot));
}

// How many chunks a batch of `lanes` lanes needs, i.e. how many blocks to loop over per round.
PVZ_SIM int32_t NetBlocks(int32_t lanes) { return (lanes + kNetChunkMask) >> kNetChunkBits; }

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

// One batch's activations, chunked as described above.  `hidden` and `scratch` ping-pong through the
// three convolutions, so only two width-plane buffers are live at a time.  Each pointer covers
// NetBlocks(lanes) blocks of the buffer's own capacity: board kBoardChannels planes, hidden and
// scratch net.width, scalars kScalarFeatures, context net.contextSize, logits kLogitCells.
struct NetState
{
	float* board;                         // kBoardChannels * kPlaneCells per block
	float* scalars;                       // kScalarFeatures
	float* hidden;                        // width * kPlaneCells
	float* scratch;                       // width * kPlaneCells
	float* context;                       // contextSize
	float* logits;                        // kLogitCells, the cell head's lawn logits
};

// What the student decided, in the shape Predict's caller consumes.  Lives here rather than in
// NetForward.inc because the round split in cuda/Rollout.inc passes it between two kernels.
struct Choice
{
	int32_t packet;               // -1 = wait
	int32_t row, col;
	int32_t forwarded;            // 1 = the network ran; net_calls counts these, not decisions
	// How long a wait waits.  Zero means the student's own cadence (60 ticks, EnvironmentRollout's wait);
	// only the teacher's forced first action ever names another number (src/main.cpp:453-457 reads the
	// ticks out of the request instead of assuming 60).
	int32_t waitTicks;
};

}  // namespace pvzc

#endif
