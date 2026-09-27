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
#include <charconv>
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
	uint32_t appCounter;
	uint32_t zombiesKilled;
	uint32_t plantsEaten;
	uint32_t sunProduced;
	int triggeredLawnMowers;
	GameScenes gameScene;
	BoardResult boardResult;
};

static bool ParseInt(const std::string& text, int& value)
{
	const char* first = text.data();
	const char* last = first + text.size();
	auto parsed = std::from_chars(first, last, value);
	return parsed.ec == std::errc() && parsed.ptr == last;
}

static bool ParseIntList(const std::string& text, std::vector<int>& values)
{
	values.clear();
	if (text.empty() || text == "-") return true;
	std::istringstream input(text);
	std::string item;
	while (std::getline(input, item, ','))
	{
		int value = 0;
		if (!ParseInt(item, value)) return false;
		values.push_back(value);
	}
	return !values.empty();
}

static bool ParseDeck(const std::string& text, std::vector<EnvironmentSeed>& deck)
{
	deck.clear();
	if (text.empty() || text == "-") return false;
	std::istringstream input(text);
	std::string item;
	while (std::getline(input, item, ','))
	{
		size_t separator = item.find(':');
		int type = 0;
		int imitaterType = static_cast<int>(SeedType::SEED_NONE);
		if (!ParseInt(item.substr(0, separator), type) ||
			(separator != std::string::npos && !ParseInt(item.substr(separator + 1), imitaterType)) ||
			type < 0 || type >= SeedType::NUM_SEED_TYPES ||
			(imitaterType != static_cast<int>(SeedType::SEED_NONE) &&
				(imitaterType < 0 || imitaterType >= SeedType::NUM_SEED_TYPES)))
		{
			deck.clear();
			return false;
		}
		deck.push_back({ static_cast<SeedType>(type), static_cast<SeedType>(imitaterType) });
	}
	return !deck.empty();
}

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
	std::cout << "PVZENV {\"ready\":true,\"protocol_version\":1}" << std::endl;
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
		if (command == "RESET_V1")
		{
			int level = 0, playthrough = 0, slots = 0, imitater = 0, firstAid = 0, poolCleaner = 0, roofCleaner = 0, rake = 0;
			uint32_t seed = 0;
			std::string upgradesText, forcedText, deckText, multiplierText;
			input >> level >> seed >> playthrough >> slots >> imitater >> firstAid >> poolCleaner >> roofCleaner >> rake
				>> upgradesText >> forcedText >> deckText;
			std::vector<int> upgrades, forced;
			std::vector<EnvironmentSeed> deck;
			EnvironmentTaskSpec task;
			bool parsed = !input.fail() && ParseIntList(upgradesText, upgrades) && ParseIntList(forcedText, forced) && ParseDeck(deckText, deck);
			if (parsed && input >> multiplierText)
			{
				std::istringstream multiplierInput(multiplierText);
				parsed = static_cast<bool>(multiplierInput >> task.zombieCountMultiplier) && multiplierInput.peek() == std::char_traits<char>::eof();
				std::string extra;
				if (input >> extra) parsed = false;
			}
			if (parsed && (imitater == 0 || imitater == 1) && (firstAid == 0 || firstAid == 1) &&
				(poolCleaner == 0 || poolCleaner == 1) && (roofCleaner == 0 || roofCleaner == 1))
			{
				task.playthrough = playthrough;
				task.seedSlotCount = slots;
				task.imitaterOwned = imitater != 0;
				task.firstAidOwned = firstAid != 0;
				task.poolCleanerOwned = poolCleaner != 0;
				task.roofCleanerOwned = roofCleaner != 0;
				task.rakeCharges = rake;
				for (int value : upgrades) task.ownedUpgradePlants.push_back(static_cast<SeedType>(value));
				for (int value : forced) task.forcedSeeds.push_back(static_cast<SeedType>(value));
				ok = app->EnvironmentReset(level, seed, deck, task);
			}
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
					snapshot.appCounter = app->mAppCounter;
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
						app->mAppCounter = snapshot.appCounter;
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
		else if (command == "DROP_SNAPSHOT")
		{
			input >> snapshotId;
			ok = snapshots.erase(snapshotId) > 0;
		}
		else if (command == "QUIT")
		{
			std::cout << "PVZENV {\"ok\":true,\"closed\":true}" << std::endl;
			break;
		}

		std::cout << "PVZENV {\"protocol_version\":1,\"ok\":" << (ok ? "true" : "false") << ",\"observation\":" << app->EnvironmentObservation(privileged);
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
