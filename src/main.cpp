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
#include "Lawn/Coin.h"
#include "Lawn/Plant.h"
#include "Lawn/SeedPacket.h"
#include "Lawn/System/SaveGame.h"
#include "Lawn/Zombie.h"
#include "Resources.h"
#include "PvzpLib/PvzpStringFile.h"
#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
using namespace Sexy;

static constexpr int kEnvironmentProtocolVersion = 4;

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

static bool ParsePreplanted(const std::string& text, std::vector<EnvironmentPreplanted>& plants)
{
	plants.clear();
	if (text == "-") return true;
	if (text.empty() || text.front() == ',' || text.back() == ',') return false;
	std::istringstream input(text);
	std::string item;
	while (std::getline(input, item, ','))
	{
		size_t firstSeparator = item.find(':');
		size_t secondSeparator = firstSeparator == std::string::npos ? std::string::npos : item.find(':', firstSeparator + 1);
		int type = 0, row = 0, col = 0;
		if (firstSeparator == std::string::npos || secondSeparator == std::string::npos ||
			item.find(':', secondSeparator + 1) != std::string::npos ||
			!ParseInt(item.substr(0, firstSeparator), type) ||
			!ParseInt(item.substr(firstSeparator + 1, secondSeparator - firstSeparator - 1), row) ||
			!ParseInt(item.substr(secondSeparator + 1), col) ||
			type < 0 || type > static_cast<int>(SeedType::SEED_IMITATER) ||
			row < 0 || row >= MAX_GRID_SIZE_Y || col < 0 || col >= MAX_GRID_SIZE_X)
		{
			plants.clear();
			return false;
		}
		plants.push_back({ static_cast<SeedType>(type), row, col });
	}
	return !plants.empty();
}

static EnvCounters ReadCounters(LawnApp* app)
{
	if (!app->mBoard)
		return {};
	return { app->mBoard->mSunMoney, app->mBoard->mSunMoneyProduced, app->mBoard->mZombiesKilled,
		app->mBoard->mPlantsEaten, static_cast<uint32_t>(app->mBoard->mTriggeredLawnMowers), app->mBoard->mTotalSpawnedWaves };
}

static bool SaveEnvironmentSnapshot(LawnApp* app, EnvironmentSnapshot& snapshot)
{
	if (!app->mBoard || !LawnSaveGameToMemory(app->mBoard, snapshot.board))
		return false;
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
	return true;
}

// Where the transposition key comes from.  The byte-at-a-time FNV-1a this replaced spent
// 77 us on an 86 KB payload -- about a third of one ``BRANCH_SNAPSHOT_FAST`` branch --
// because every input byte is one more link in a serial xor/multiply chain.  Folding a
// whole 64-bit word per step keeps FNV's "xor then multiply" mixing and cuts the chain
// eightfold.  The multiply only ever spreads a difference *upwards*, so the last words
// leave the low bits under-mixed; ``finalise`` is MurmurHash3's fmix64 avalanche, which
// fixes that.  Callers only ever compare hashes for equality, so the value itself is
// opaque and changing it is not a compatibility break.
static uint64_t FinaliseHash(uint64_t theKey)
{
	theKey ^= theKey >> 33;
	theKey *= 0xff51afd7ed558ccdULL;
	theKey ^= theKey >> 33;
	theKey *= 0xc4ceb9fe1a85ec53ULL;
	theKey ^= theKey >> 33;
	return theKey;
}

static uint64_t EnvironmentSnapshotHash(const EnvironmentSnapshot& snapshot)
{
	constexpr uint64_t kPrime = 1099511628211ULL;
	uint64_t hash = 1469598103934665603ULL;
	auto mixWord = [&hash](uint64_t value)
	{
		hash ^= value;
		hash *= kPrime;
	};
	auto mixBytes = [&hash, &mixWord](const void* theData, size_t theSize)
	{
		const unsigned char* aBytes = static_cast<const unsigned char*>(theData);
		size_t aPosition = 0;
		// ``memcpy`` of eight bytes compiles to a single unaligned load on every target
		// that has one, so this is portable without a strict-aliasing or alignment worry.
		for (; aPosition + sizeof(uint64_t) <= theSize; aPosition += sizeof(uint64_t))
		{
			uint64_t aWord;
			memcpy(&aWord, aBytes + aPosition, sizeof(aWord));
			mixWord(aWord);
		}
		if (aPosition != theSize)
		{
			// Big-endian tail so the same bytes give the same word whatever the machine is.
			uint64_t aTail = 0;
			for (; aPosition < theSize; ++aPosition)
				aTail = (aTail << 8) | aBytes[aPosition];
			mixWord(aTail);
		}
	};
	// Lengths first, so a payload that merely ends in zero words cannot collide with a
	// shorter one.
	mixWord(static_cast<uint64_t>(snapshot.board.size()));
	mixBytes(snapshot.board.data(), snapshot.board.size());
	mixWord(static_cast<uint64_t>(snapshot.randState.size()));
	mixBytes(snapshot.randState.data(), snapshot.randState.size());
	mixWord(static_cast<uint32_t>(snapshot.appRandSeed));
	mixWord(snapshot.randSeed);
	mixWord(snapshot.appCounter);
	mixWord(snapshot.zombiesKilled);
	mixWord(snapshot.plantsEaten);
	mixWord(snapshot.sunProduced);
	mixWord(static_cast<uint32_t>(snapshot.triggeredLawnMowers));
	mixWord(static_cast<uint32_t>(snapshot.gameScene));
	mixWord(static_cast<uint32_t>(snapshot.boardResult));
	return FinaliseHash(hash);
}

static bool RestoreEnvironmentSnapshot(LawnApp* app, const EnvironmentSnapshot& snapshot)
{
	if (!app->mBoard || !LawnLoadGameFromMemory(app->mBoard, snapshot.board))
		return false;
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
	return true;
}

static bool ExecuteEnvironmentAction(LawnApp* app, const std::string& command,
	std::istringstream& input)
{
	if (command == "PLANT")
	{
		int packet = -1, col = -1, row = -1;
		input >> packet >> col >> row;
		return !input.fail() && app->EnvironmentPlant(packet, col, row);
	}
	if (command == "SHOVEL")
	{
		int col = -1, row = -1;
		input >> col >> row;
		return !input.fail() && app->EnvironmentShovel(col, row);
	}
	if (command == "WAIT")
	{
		int ticks = -1;
		input >> ticks;
		if (input.fail() || ticks < 0 || ticks > 1000000)
			return false;
		app->EnvironmentWait(ticks);
		return true;
	}
	return false;
}

static bool DecodeBranchAction(const std::string& spec, std::string& command, std::string& arguments)
{
	if (spec.size() < 3 || spec[1] != ':')
		return false;
	const char kind = spec[0];
	if (kind == 'P') command = "PLANT";
	else if (kind == 'S') command = "SHOVEL";
	else if (kind == 'W') command = "WAIT";
	else return false;
	arguments = spec.substr(2);
	std::replace(arguments.begin(), arguments.end(), ':', ' ');
	return true;
}

static void RunEnvironment(LawnApp* app)
{
	std::string line;
	std::unordered_map<int, EnvironmentSnapshot> snapshots;
	int nextSnapshotId = 1;
	std::cout << "PVZENV {\"ready\":true,\"protocol_version\":" << kEnvironmentProtocolVersion << '}' << std::endl;
	while (std::getline(std::cin, line))
	{
		std::istringstream input(line);
		std::string command;
		input >> command;
		EnvCounters before = ReadCounters(app);
		bool ok = false;
		bool privileged = false;
		bool hasSnapshotId = false;
		bool minimalResponse = false;
		int snapshotId = 0;
		if (command == "RESET_V2")
		{
			int level = 0, playthrough = 0, slots = 0, imitater = 0, firstAid = 0, poolCleaner = 0, roofCleaner = 0, rake = 0;
			uint32_t seed = 0;
			std::string upgradesText, forcedText, deckText, multiplierText, preplantedText;
			input >> level >> seed >> playthrough >> slots >> imitater >> firstAid >> poolCleaner >> roofCleaner >> rake
				>> upgradesText >> forcedText >> deckText;
			std::vector<int> upgrades, forced;
			std::vector<EnvironmentSeed> deck;
			EnvironmentTaskSpec task;
			bool parsed = !input.fail() && ParseIntList(upgradesText, upgrades) && ParseIntList(forcedText, forced) && ParseDeck(deckText, deck);
			if (parsed)
			{
				parsed = static_cast<bool>(input >> multiplierText >> task.waveCap >> preplantedText);
				if (parsed)
				{
					std::istringstream multiplierInput(multiplierText);
					parsed = static_cast<bool>(multiplierInput >> task.zombieCountMultiplier) &&
						multiplierInput.peek() == std::char_traits<char>::eof() && ParsePreplanted(preplantedText, task.preplanted);
				}
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
				// RESET starts a new counter history. Subtracting the previous
				// board's uint32 counters wraps into billions of fake events.
				before = ReadCounters(app);
			}
		}
		else if (command == "PLANT" || command == "SHOVEL" || command == "WAIT")
		{
			ok = ExecuteEnvironmentAction(app, command, input);
		}
		else if (command == "BENCH_SNAPSHOT")
		{
			// Split the cost of one BRANCH_SNAPSHOT_FAST branch into its ingredients.  The
			// search is dominated by that round trip, and "save" versus "restore" versus
			// "hash" need very different fixes -- so an optimisation should be attributed by
			// measurement, not by guesswork.  ``scripts/branch_benchmark.py`` times the whole
			// command from Python; this is the in-process view that separates the parts the
			// Python timer lumps together with transport and JSON parsing.
			int reps = 0;
			input >> reps;
			reps = std::clamp(reps, 1, 20000);
			using Clock = std::chrono::steady_clock;
			auto measure = [&](auto&& body)
			{
				std::vector<double> samples;
				samples.reserve(static_cast<size_t>(reps));
				for (int i = 0; i < reps; ++i)
				{
					auto started = Clock::now();
					body();
					samples.push_back(std::chrono::duration<double, std::micro>(Clock::now() - started).count());
				}
				std::sort(samples.begin(), samples.end());
				double total = 0.0;
				for (double sample : samples) total += sample;
				return std::pair<double, double>{ samples[samples.size() / 2], total / static_cast<double>(samples.size()) };
			};
			EnvironmentSnapshot parent{};
			bool haveParent = app->mBoard && SaveEnvironmentSnapshot(app, parent);
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion
				<< ",\"ok\":" << (haveParent ? "true" : "false") << ",\"reps\":" << reps;
			if (haveParent)
			{
				std::vector<unsigned char> reused;
				auto emit = [&](const char* name, const std::pair<double, double>& stats)
				{
					std::cout << ",\"" << name << "\":{\"median_us\":" << stats.first << ",\"mean_us\":" << stats.second << '}';
				};
				// Every body has to feed a volatile sink: the compiler is entitled to delete a
				// call whose result is unused and it happily did, turning "hash" into 0 us.
				static volatile uint64_t aSink = 0;
				emit("save_fresh", measure([&] { std::vector<unsigned char> out; LawnSaveGameToMemory(app->mBoard, out); aSink += out.size(); }));
				emit("save_reused", measure([&] { reused.clear(); LawnSaveGameToMemory(app->mBoard, reused); aSink += reused.size(); }));
				emit("restore", measure([&] { aSink += LawnLoadGameFromMemory(app->mBoard, parent.board) ? 1 : 0; }));
				emit("hash", measure([&] { aSink += EnvironmentSnapshotHash(parent); }));
				emit("observation", measure([&] { aSink += app->EnvironmentObservation(false).size(); }));
				std::cout << ",\"payload_bytes\":" << parent.board.size()
					<< ",\"rand_state_bytes\":" << parent.randState.size();
			}
			std::cout << '}' << std::endl;
			continue;
		}
		else if (command == "BRANCH_SNAPSHOT_FAST")
		{
			int branchCount = 0;
			input >> snapshotId >> branchCount;
			std::vector<std::string> specs;
			if (!input.fail() && snapshots.find(snapshotId) != snapshots.end() && branchCount >= 1 && branchCount <= 128)
			{
				specs.reserve(static_cast<size_t>(branchCount));
				for (int i = 0; i < branchCount; ++i)
				{
					std::string spec;
					input >> spec;
					if (input.fail()) break;
					specs.push_back(std::move(spec));
				}
			}
			ok = static_cast<int>(specs.size()) == branchCount;
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":" << (ok ? "true" : "false");
			if (ok)
			{
				std::cout << ",\"branches\":[";
				for (int i = 0; i < branchCount; ++i)
				{
					if (i != 0) std::cout << ',';
					auto parent = snapshots.find(snapshotId);
					bool branchOk = parent != snapshots.end() && RestoreEnvironmentSnapshot(app, parent->second);
					int childSnapshotId = 0;
					uint64_t childStateHash = 0;
					bool hasChildSnapshot = false;
					EnvCounters branchBefore = ReadCounters(app);
					std::string observation = "null";
					if (branchOk)
					{
						std::string actionCommand, actionArguments;
						branchOk = DecodeBranchAction(specs[static_cast<size_t>(i)], actionCommand, actionArguments);
						if (branchOk)
						{
							std::istringstream actionInput(actionArguments);
							branchOk = ExecuteEnvironmentAction(app, actionCommand, actionInput);
						}
					}
					EnvCounters branchAfter = ReadCounters(app);
					if (branchOk)
					{
						observation = app->EnvironmentObservation(false);
						if (!app->EnvironmentTerminal())
						{
							EnvironmentSnapshot child{};
							branchOk = SaveEnvironmentSnapshot(app, child);
							if (branchOk)
							{
								childStateHash = EnvironmentSnapshotHash(child);
								childSnapshotId = nextSnapshotId++;
								snapshots.emplace(childSnapshotId, std::move(child));
								hasChildSnapshot = true;
							}
						}
					}
					std::cout << "{\"ok\":" << (branchOk ? "true" : "false") << ",\"observation\":" << observation;
					if (hasChildSnapshot)
						std::cout << ",\"snapshot_id\":" << childSnapshotId << ",\"state_hash\":\"" << childStateHash << '\"';
					std::cout << ",\"events\":{\"zombies_killed\":" << branchAfter.zombiesKilled - branchBefore.zombiesKilled
						<< ",\"plants_eaten\":" << branchAfter.plantsEaten - branchBefore.plantsEaten
						<< ",\"sun_produced\":" << branchAfter.sunProduced - branchBefore.sunProduced
						<< ",\"sun_spent\":" << std::max(branchBefore.sun - branchAfter.sun, 0)
						<< ",\"mower_triggered\":" << branchAfter.mowers - branchBefore.mowers
						<< ",\"waves_started\":" << branchAfter.waves - branchBefore.waves
						<< ",\"level_won\":" << (app->mBoard && app->mBoard->mLevelComplete ? "true" : "false")
						<< ",\"level_lost\":" << (app->mGameScene == GameScenes::SCENE_ZOMBIES_WON ? "true" : "false") << "}}";
				}
				std::cout << ']';
			}
			std::cout << '}' << std::endl;
			continue;
		}
		else if (command == "OBS" || command == "PRIV")
		{
			ok = app->mBoard != nullptr;
			privileged = command == "PRIV";
		}
		else if (command == "SNAPSHOT" || command == "SNAPSHOT_FAST")
		{
			minimalResponse = command == "SNAPSHOT_FAST";
			EnvironmentSnapshot snapshot{};
			ok = SaveEnvironmentSnapshot(app, snapshot);
			if (ok)
			{
				snapshotId = nextSnapshotId++;
				snapshots.emplace(snapshotId, std::move(snapshot));
				hasSnapshotId = true;
			}
		}
		else if (command == "RESTORE" || command == "RESTORE_FAST")
		{
			minimalResponse = command == "RESTORE_FAST";
			input >> snapshotId;
			auto it = snapshots.find(snapshotId);
			if (it != snapshots.end())
			{
				ok = RestoreEnvironmentSnapshot(app, it->second);
				if (ok)
					before = ReadCounters(app);
			}
		}
		else if (command == "DROP_SNAPSHOT" || command == "DROP_SNAPSHOT_FAST")
		{
			minimalResponse = command == "DROP_SNAPSHOT_FAST";
			input >> snapshotId;
			ok = snapshots.erase(snapshotId) > 0;
		}
		else if (command == "CRITIC_INPUTS")
		{
			int waveIndex = 0;
			input >> waveIndex;
			ok = !input.fail() && app->mBoard != nullptr;
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion
				<< ",\"ok\":" << (ok ? "true" : "false");
			if (ok)
			{
				Board* board = app->mBoard;
				std::cout << ",\"wave_timer\":" << board->mZombieCountDown << ",\"wave_zombies\":[";
				if (board->mNumWaves > 0)
				{
					const int selectedWave = std::clamp(waveIndex, 0, board->mNumWaves - 1);
					bool firstZombie = true;
					for (int i = 0; i < std::min(15, MAX_ZOMBIES_IN_WAVE) &&
						board->mZombiesInWave[selectedWave][i] != ZombieType::ZOMBIE_INVALID; ++i)
					{
						if (!firstZombie) std::cout << ',';
						firstZombie = false;
						std::cout << static_cast<int>(board->mZombiesInWave[selectedWave][i]);
					}
				}
				std::cout << ']';
			}
			std::cout << '}' << std::endl;
			continue;
		}
		else if (command == "QUIT")
		{
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":true,\"closed\":true}" << std::endl;
			break;
		}

		if (minimalResponse)
		{
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":" << (ok ? "true" : "false");
			if (hasSnapshotId)
				std::cout << ",\"snapshot_id\":" << snapshotId;
			std::cout << '}' << std::endl;
			continue;
		}

		std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":" << (ok ? "true" : "false") << ",\"observation\":" << app->EnvironmentObservation(privileged);
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
