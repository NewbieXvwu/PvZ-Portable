#ifndef PVZ_ENVIRONMENT_SNAPSHOT_H
#define PVZ_ENVIRONMENT_SNAPSHOT_H

#include "ConstEnums.h"
#include <cstdint>
#include <string>
#include <vector>

class LawnApp;

// The existing in-memory search transaction; the portable save format is unchanged.
struct EnvironmentSnapshot
{
	std::vector<unsigned char> board;
	std::string randState;
	int appRandSeed;
	uint32_t randSeed;
	uint32_t appCounter;
	uint32_t zombiesKilled;
	uint32_t plantsEaten;
	uint32_t plantsEatenFree;
	uint32_t sunProduced;
	int triggeredLawnMowers;
	GameScenes gameScene;
	BoardResult boardResult;
};

bool SaveEnvironmentSnapshot(LawnApp* app, EnvironmentSnapshot& snapshot);
bool RestoreEnvironmentSnapshot(LawnApp* app, const EnvironmentSnapshot& snapshot);

#endif
