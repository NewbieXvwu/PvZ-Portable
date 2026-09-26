/*
 * Copyright (C) 2026 Zhou Qiankang <wszqkzqk@qq.com>
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * This file is part of PvZ-Portable.
 *
 * PvZ-Portable is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PvZ-Portable is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with PvZ-Portable. If not, see <https://www.gnu.org/licenses/>.
 */

#include "LawnApp.h"
#include "Lawn/Board.h"
#include "Lawn/System/SaveGame.h"
#include "Resources.h"
#include "PvzpLib/PvzpStringFile.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
using namespace Sexy;

struct EnvCounters
{
	int sun = 0;
	uint32_t sunProduced = 0;
	uint32_t zombiesKilled = 0;
	uint32_t plantsEaten = 0;
	uint32_t mowers = 0;
	int waves = 0;
};

struct EnvironmentSnapshot
{
	std::vector<unsigned char> board;
	std::string randState;
	int appRandSeed;
	uint32_t randSeed;
	uint32_t zombiesKilled;
	uint32_t plantsEaten;
	uint32_t sunProduced;
	int triggeredLawnMowers;
	GameScenes gameScene;
	BoardResult boardResult;
};

static EnvCounters ReadCounters(LawnApp* app)
{
	if (!app->mBoard)
		return {};
	return { app->mBoard->mSunMoney, app->mBoard->mSunMoneyProduced, app->mBoard->mZombiesKilled,
		app->mBoard->mPlantsEaten, static_cast<uint32_t>(app->mBoard->mTriggeredLawnMowers), app->mBoard->mTotalSpawnedWaves };
}

static void RunEnvironment(LawnApp* app)
{
	std::string line;
	std::unordered_map<int, EnvironmentSnapshot> snapshots;
	int nextSnapshotId = 1;
	std::cout << "PVZENV {\"ready\":true}" << std::endl;
	while (std::getline(std::cin, line))
	{
		std::istringstream input(line);
		std::string command;
		input >> command;
		EnvCounters before = ReadCounters(app);
		bool ok = false;
		bool privileged = false;
		bool hasSnapshotId = false;
		int snapshotId = 0;
		int ticksAdvanced = -1;
		if (command == "RESET")
		{
			int level = 0;
			uint32_t seed = 0;
			std::string deckText;
			input >> level >> seed >> deckText;
			std::vector<SeedType> deck;
			std::istringstream deckInput(deckText);
			std::string type;
			while (std::getline(deckInput, type, ','))
			{
				std::istringstream typeInput(type);
				int value = -1;
				if (!(typeInput >> value) || value < 0 || value >= SeedType::NUM_SEED_TYPES)
				{
					deck.clear();
					break;
				}
				deck.push_back(static_cast<SeedType>(value));
			}
			ok = !deck.empty() && app->EnvironmentReset(level, seed, deck);
			if (ok)
			{
				snapshots.clear();
				nextSnapshotId = 1;
			}
		}
		else if (command == "PLANT")
		{
			int packet = -1, col = -1, row = -1;
			input >> packet >> col >> row;
			ok = app->EnvironmentPlant(packet, col, row);
		}
		else if (command == "SHOVEL")
		{
			int col = -1, row = -1;
			input >> col >> row;
			ok = app->EnvironmentShovel(col, row);
		}
		else if (command == "WAIT")
		{
			int ticks = -1;
			input >> ticks;
			if (ticks >= 0 && ticks <= 1000000)
			{
				app->EnvironmentWait(ticks);
				ok = true;
			}
		}
		else if (command == "WAIT_DECISION")
		{
			int maxTicks = -1;
			input >> maxTicks;
			ticksAdvanced = app->EnvironmentWaitDecision(maxTicks);
			ok = ticksAdvanced > 0;
		}
		else if (command == "OBS" || command == "PRIV")
		{
			ok = app->mBoard != nullptr;
			privileged = command == "PRIV";
		}
		else if (command == "SNAPSHOT")
		{
			EnvironmentSnapshot snapshot{};
			ok = app->mBoard && LawnSaveGameToMemory(app->mBoard, snapshot.board);
			if (ok)
			{
				snapshot.randState = GetRandState();
				snapshot.appRandSeed = app->mAppRandSeed;
				snapshot.randSeed = app->mRandSeed;
				snapshot.zombiesKilled = app->mBoard->mZombiesKilled;
				snapshot.plantsEaten = app->mBoard->mPlantsEaten;
				snapshot.sunProduced = app->mBoard->mSunMoneyProduced;
				snapshot.triggeredLawnMowers = app->mBoard->mTriggeredLawnMowers;
				snapshot.gameScene = app->mGameScene;
				snapshot.boardResult = app->mBoardResult;
				snapshotId = nextSnapshotId++;
				snapshots.emplace(snapshotId, std::move(snapshot));
				hasSnapshotId = true;
			}
		}
		else if (command == "RESTORE")
		{
			input >> snapshotId;
			auto it = snapshots.find(snapshotId);
			if (it != snapshots.end() && app->mBoard)
			{
				const EnvironmentSnapshot& snapshot = it->second;
				ok = LawnLoadGameFromMemory(app->mBoard, snapshot.board);
				if (ok)
				{
					SetRandState(snapshot.randState);
					app->mAppRandSeed = snapshot.appRandSeed;
					app->mRandSeed = snapshot.randSeed;
					app->mBoard->mZombiesKilled = snapshot.zombiesKilled;
					app->mBoard->mPlantsEaten = snapshot.plantsEaten;
					app->mBoard->mSunMoneyProduced = snapshot.sunProduced;
					app->mBoard->mTriggeredLawnMowers = snapshot.triggeredLawnMowers;
					app->mGameScene = snapshot.gameScene;
					app->mBoardResult = snapshot.boardResult;
					before = ReadCounters(app);
				}
			}
		}
		else if (command == "QUIT")
		{
			std::cout << "PVZENV {\"ok\":true,\"closed\":true}" << std::endl;
			break;
		}

		std::cout << "PVZENV {\"ok\":" << (ok ? "true" : "false") << ",\"observation\":" << app->EnvironmentObservation(privileged);
		if (ticksAdvanced >= 0)
			std::cout << ",\"ticks_advanced\":" << ticksAdvanced;
		if (hasSnapshotId)
			std::cout << ",\"snapshot_id\":" << snapshotId;
		if (app->mBoard)
		{
			EnvCounters after = ReadCounters(app);
			std::cout << ",\"events\":{\"zombies_killed\":" << after.zombiesKilled - before.zombiesKilled
				<< ",\"plants_eaten\":" << after.plantsEaten - before.plantsEaten
				<< ",\"sun_produced\":" << after.sunProduced - before.sunProduced
				<< ",\"sun_spent\":" << std::max(before.sun - after.sun, 0)
				<< ",\"mower_triggered\":" << after.mowers - before.mowers
				<< ",\"waves_started\":" << after.waves - before.waves
				<< ",\"level_won\":" << (app->mBoard->mLevelComplete ? "true" : "false")
				<< ",\"level_lost\":" << (app->mGameScene == GameScenes::SCENE_ZOMBIES_WON ? "true" : "false") << '}';
		}
		std::cout << '}' << std::endl;
	}
}

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

#ifdef __SWITCH__
#include <switch.h>
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#ifdef _WIN32
static std::vector<std::string> gUtf8ArgsStorage;
static std::vector<char*> gUtf8Argv;

static void BuildUtf8ArgsFromWin32(int& argc, char**& argv)
{
	int aWideArgc = 0;
	LPWSTR* aWideArgv = CommandLineToArgvW(GetCommandLineW(), &aWideArgc);
	if (aWideArgv == nullptr || aWideArgc <= 0)
		return;

	gUtf8ArgsStorage.clear();
	gUtf8Argv.clear();
	gUtf8ArgsStorage.reserve(static_cast<size_t>(aWideArgc));
	gUtf8Argv.reserve(static_cast<size_t>(aWideArgc));

	for (int i = 0; i < aWideArgc; ++i)
	{
		const wchar_t* aWide = aWideArgv[i];
		int aLen = WideCharToMultiByte(CP_UTF8, 0, aWide, -1, nullptr, 0, nullptr, nullptr);
		if (aLen <= 0)
		{
			gUtf8ArgsStorage.emplace_back();
		}
		else
		{
			std::string aUtf8;
			aUtf8.resize(static_cast<size_t>(aLen - 1));
			WideCharToMultiByte(CP_UTF8, 0, aWide, -1, aUtf8.data(), aLen, nullptr, nullptr);
			gUtf8ArgsStorage.emplace_back(std::move(aUtf8));
		}
	}

	for (auto& aStr : gUtf8ArgsStorage)
		gUtf8Argv.push_back(const_cast<char*>(aStr.c_str()));

	argc = static_cast<int>(gUtf8Argv.size());
	argv = gUtf8Argv.data();

	LocalFree(aWideArgv);
}
#endif

int main(int argc, char** argv)
{
#ifdef __SWITCH__
	consoleDebugInit(debugDevice_SVC);
#endif

#ifdef _WIN32
	BuildUtf8ArgsFromWin32(argc, argv);
#endif

	PvzpStringListSetColors(gLawnStringFormats, gLawnStringFormatCount);
	gExtractResourcesByName = Sexy::ExtractResourcesByName;
	gLawnApp = new LawnApp();
	gLawnApp->SetArgs(argc, argv);
	gLawnApp->Init();
	if (gLawnApp->mEnvironmentMode && !gLawnApp->mLoadingFailed)
		RunEnvironment(gLawnApp);
	else if (!gLawnApp->mHeadlessMode)
		gLawnApp->Start();
	else if (!gLawnApp->mLoadingFailed)
		std::printf("headless-ready\n");
#ifndef __EMSCRIPTEN__
	gLawnApp->Shutdown();
	delete gLawnApp;
#endif

	return 0;
};
