// Export a live original-simulator position into the layout cuda/CudaRollout.h defines.
// Plain C++, no CUDA dependency: a machine without nvcc can still produce .cudaroot files.
#ifndef PVZ_CUDA_EXPORT_H
#define PVZ_CUDA_EXPORT_H

#include "CudaRollout.h"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

class LawnApp;

namespace pvzc
{

// The student's v2lite history, as the caller holds it *before* this position is observed.  The
// original keeps it in EnvironmentRollout::History; the GPU kernel keeps it per lane and updates it
// on the first feature build, so the root only has to carry the state that build starts from.
struct CudaHistoryPlant
{
	int row, col, kind, born;
};
struct CudaHistory
{
	int wave = -1;                // -1 = the student has not observed a wave yet
	int waveTick = 0;
	std::vector<CudaHistoryPlant> plants;
};

// incomeEvents are the (tick, mSunMoneyProduced delta) pairs the caller accumulated, needed because
// the env's 6000-tick sun income window is a Python-side rolling counter, not a board field.
// history is optional: without it the root is a v1 position (no plant is older than the root).
bool BuildCudaRoot(LawnApp* app, const std::vector<std::pair<int, uint32_t>>& incomeEvents,
	int incomeWindowStart, RootRecord& record, const CudaHistory* history = nullptr);
void FillLevelInfo(LawnApp* app, LevelInfo& info);
bool WriteCudaLevel(std::FILE* out, LawnApp* app);
bool WriteCudaRoot(std::FILE* out, LawnApp* app,
	const std::vector<std::pair<int, uint32_t>>& incomeEvents, int incomeWindowStart,
	const CudaHistory* history = nullptr);

}  // namespace pvzc

#endif
