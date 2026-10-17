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

// incomeEvents are the (tick, mSunMoneyProduced delta) pairs the caller accumulated, needed because
// the env's 6000-tick sun income window is a Python-side rolling counter, not a board field.
bool BuildCudaRoot(LawnApp* app, const std::vector<std::pair<int, uint32_t>>& incomeEvents,
	int incomeWindowStart, RootRecord& record);
void FillLevelInfo(LawnApp* app, LevelInfo& info);
bool WriteCudaLevel(std::FILE* out, LawnApp* app);
bool WriteCudaRoot(std::FILE* out, LawnApp* app,
	const std::vector<std::pair<int, uint32_t>>& incomeEvents, int incomeWindowStart);

}  // namespace pvzc

#endif
