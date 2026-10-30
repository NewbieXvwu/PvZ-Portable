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
#include "Lawn/CursorObject.h"
#include "SexyAppFramework/widget/WidgetManager.h"
#include <SDL.h>
#include <filesystem>
#include "Lawn/System/SaveGame.h"
#include "Lawn/Zombie.h"
#include "Lawn/LawnMower.h"
#include "EnvironmentRollout.h"
#include "EnvironmentValueNet.h"
#include "Resources.h"
#include "PvzpLib/PvzpStringFile.h"
#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <deque>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <tuple>
#include <vector>
using namespace Sexy;

static constexpr int kEnvironmentProtocolVersion = 4;

struct EnvCounters
{
	int sun = 0;
	uint32_t sunProduced = 0;
	uint32_t zombiesKilled = 0;
	uint32_t plantsEaten = 0;
	uint32_t plantsEatenFree = 0;   // task65: plantsEaten whose actual planting cost is 0
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
	uint32_t plantsEatenFree;   // task65: not in SaveGame (save format unchanged); rides the in-memory snapshot
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
	// "-" is the empty deck: only the original conveyor start accepts it (EnvironmentReset
	// rejects an empty deck for every other task).
	if (text == "-") return true;
	if (text.empty()) return false;
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
		app->mBoard->mPlantsEaten, app->mBoard->mPlantsEatenFree,
		static_cast<uint32_t>(app->mBoard->mTriggeredLawnMowers), app->mBoard->mTotalSpawnedWaves };
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
	snapshot.plantsEatenFree = app->mBoard->mPlantsEatenFree;
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
	app->mBoard->mPlantsEatenFree = snapshot.plantsEatenFree;
	app->mBoard->mSunMoneyProduced = snapshot.sunProduced;
	app->mBoard->mTriggeredLawnMowers = snapshot.triggeredLawnMowers;
	app->mGameScene = snapshot.gameScene;
	app->mBoardResult = snapshot.boardResult;
	return true;
}

// Debug aid for "two runs disagree" reports: dump every field StableEnvironmentHash folds in,
// so the two runs can be compared field by field instead of only by a 64-bit digest.  The board
// payload is the same SaveGame V4 image the hash uses (header + TLV chunks), so a diff can be
// narrowed to a chunk and then to a field.  Layout:
//   "PVZSB1\n" | u64 board size | board bytes | u64 rand size | rand bytes |
//   u32 appRandSeed | u32 randSeed | u32 appCounter | u32 zombiesKilled | u32 plantsEaten |
//   u32 sunProduced | u32 triggeredLawnMowers | u32 gameScene | u32 boardResult
static bool WriteEnvironmentStateBytes(LawnApp* app, const std::string& thePath)
{
	EnvironmentSnapshot snapshot{};
	if (!app->mBoard)
		return false;
	// Same presentation zeroing StableEnvironmentHash does, so the dumped image is byte-for-byte
	// what the digest covered.
	Board* board = app->mBoard;
	auto presentation = std::make_tuple(board->mDrawCount, board->mStartDrawTime,
		board->mIntervalDrawTime, board->mIntervalDrawCountStart, board->mMinFPS, board->mPreloadTime);
	std::tie(board->mDrawCount, board->mStartDrawTime, board->mIntervalDrawTime,
		board->mIntervalDrawCountStart, board->mMinFPS, board->mPreloadTime) = std::make_tuple(0U, 0LL, 0LL, 0U, 0.0f, 0);
	bool saved = SaveEnvironmentSnapshot(app, snapshot);
	std::tie(board->mDrawCount, board->mStartDrawTime, board->mIntervalDrawTime,
		board->mIntervalDrawCountStart, board->mMinFPS, board->mPreloadTime) = presentation;
	if (!saved)
		return false;
	std::ofstream out(thePath, std::ios::binary);
	if (!out)
		return false;
	auto writeU64 = [&out](uint64_t value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); };
	auto writeU32 = [&out](uint32_t value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); };
	out.write("PVZSB1\n", 7);
	writeU64(snapshot.board.size());
	out.write(reinterpret_cast<const char*>(snapshot.board.data()), static_cast<std::streamsize>(snapshot.board.size()));
	writeU64(snapshot.randState.size());
	out.write(reinterpret_cast<const char*>(snapshot.randState.data()), static_cast<std::streamsize>(snapshot.randState.size()));
	writeU32(snapshot.appRandSeed);
	writeU32(snapshot.randSeed);
	writeU32(snapshot.appCounter);
	writeU32(snapshot.zombiesKilled);
	writeU32(snapshot.plantsEaten);
	writeU32(snapshot.sunProduced);
	writeU32(snapshot.triggeredLawnMowers);
	writeU32(static_cast<uint32_t>(snapshot.gameScene));
	writeU32(static_cast<uint32_t>(snapshot.boardResult));
	return static_cast<bool>(out);
}

static uint64_t StableEnvironmentHash(LawnApp* app)
{
	Board* board = app->mBoard;
	auto presentation = std::make_tuple(board->mDrawCount, board->mStartDrawTime,
		board->mIntervalDrawTime, board->mIntervalDrawCountStart, board->mMinFPS, board->mPreloadTime);
	std::tie(board->mDrawCount, board->mStartDrawTime, board->mIntervalDrawTime,
		board->mIntervalDrawCountStart, board->mMinFPS, board->mPreloadTime) = std::make_tuple(0U, 0LL, 0LL, 0U, 0.0f, 0);
	EnvironmentSnapshot snapshot{};
	bool saved = SaveEnvironmentSnapshot(app, snapshot);
	std::tie(board->mDrawCount, board->mStartDrawTime, board->mIntervalDrawTime,
		board->mIntervalDrawCountStart, board->mMinFPS, board->mPreloadTime) = presentation;
	return saved ? EnvironmentSnapshotHash(snapshot) : 0;
}

struct HumanInput
{
	int packet = -1;
	bool shovel = false;
	bool paused = false;
	bool closed = false;
	int mouseX = 0;
	int mouseY = 0;
	std::string title;

	std::string Poll(LawnApp* app)
	{
		SDL_Event event;
		std::string action;
		while (SDL_PollEvent(&event))
		{
			if (event.type == SDL_QUIT || (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE)) closed = true;
			if (event.type == SDL_KEYDOWN && !event.key.repeat)
			{
				// task80: Esc (and Space, deliberately the same thing) toggles the operation
				// pause: game time stands still, but planting/shoveling/collecting below keep
				// working so a slow turn is playable.  The old two-pause split (Space froze and
				// ate every click, Esc quit) was a trap -- clicking into a dead screen -- and the
				// recorder loop treats both flags identically anyway.  The recording ends by
				// closing the window.  The title bar carries the "paused" hint: it cannot cover
				// the lawn, unlike any overlay.
				if (event.key.keysym.sym == SDLK_ESCAPE || event.key.keysym.sym == SDLK_SPACE)
				{
					paused = !paused;
					SDL_Window* window = static_cast<SDL_Window*>(app->mWindow);
					if (window != nullptr)
					{
						if (title.empty()) title = SDL_GetWindowTitle(window);
						SDL_SetWindowTitle(window, paused ? (title + " — 已暂停（可继续操作）").c_str()
														  : title.c_str());
					}
				}
			}
			if (event.type != SDL_MOUSEMOTION && event.type != SDL_MOUSEBUTTONDOWN) continue;
			mouseX = event.type == SDL_MOUSEMOTION ? event.motion.x : event.button.x;
			mouseY = event.type == SDL_MOUSEMOTION ? event.motion.y : event.button.y;
			app->mWidgetManager->RemapMouse(mouseX, mouseY);
			// Clicks stay live while paused -- that is the whole point of the operation pause.
			if (event.type != SDL_MOUSEBUTTONDOWN) continue;
			if (event.button.button == SDL_BUTTON_RIGHT) { packet = -1; shovel = false; continue; }
			if (event.button.button != SDL_BUTTON_LEFT) continue;
			Board* board = app->mBoard;
			const int x = mouseX - board->mX, y = mouseY - board->mY;
			HitResult hit;
			board->MouseHitTest(x, y, &hit);
			if (hit.mObjectType == GameObjectType::OBJECT_TYPE_SEEDPACKET)
			{
				packet = static_cast<SeedPacket*>(hit.mObject)->mIndex;
				shovel = false;
			}
			else if (hit.mObjectType == GameObjectType::OBJECT_TYPE_SHOVEL) { packet = -1; shovel = true; }
			else if (hit.mObjectType == GameObjectType::OBJECT_TYPE_COIN)
			{
				action = "{\"type\":\"collect\",\"x\":" + std::to_string(x) + ",\"y\":" + std::to_string(y) + "}";
				break;
			}
			else if (packet >= 0 || shovel)
			{
				SeedType type = packet >= 0 ? board->mSeedBank->mSeedPackets[packet].mPacketType : SeedType::SEED_NONE;
				if (type == SeedType::SEED_IMITATER) type = board->mSeedBank->mSeedPackets[packet].mImitaterType;
				int col = board->PlantingPixelToGridX(x, y, type), row = board->PlantingPixelToGridY(x, y, type);
				action = "{\"type\":\"" + std::string(shovel ? "shovel" : "plant") + "\",\"col\":" + std::to_string(col) + ",\"row\":" + std::to_string(row);
				if (!shovel) action += ",\"packet\":" + std::to_string(packet);
				action += "}";
				break;
			}
		}
		// Nothing clicked: running time offers one tick, paused time offers none -- the record
		// loop simply steps whatever this returns, so a paused poll must be a no-op.
		if (action.empty()) action = paused ? "{\"type\":\"wait\",\"ticks\":0}" : "{\"type\":\"wait\",\"ticks\":1}";
		EnvironmentSnapshot snapshot{};
		if (!SaveEnvironmentSnapshot(app, snapshot)) throw std::runtime_error("could not save presentation state");
		auto* manager = app->mWidgetManager.get();
		auto mouse = std::make_tuple(manager->mLastMouseX, manager->mLastMouseY, manager->mMouseIn);
		manager->mLastMouseX = mouseX;
		manager->mLastMouseY = mouseY;
		manager->mMouseIn = true;
		auto* cursor = app->mBoard->mCursorObject.get();
		if (packet >= 0)
		{
			cursor->mCursorType = CursorType::CURSOR_TYPE_PLANT_FROM_BANK;
			cursor->mType = app->mBoard->mSeedBank->mSeedPackets[packet].mPacketType;
			cursor->mImitaterType = app->mBoard->mSeedBank->mSeedPackets[packet].mImitaterType;
			cursor->mSeedBankIndex = packet;
		}
		if (shovel) cursor->mCursorType = CursorType::CURSOR_TYPE_SHOVEL;
		cursor->Update();
		app->mBoard->MarkDirty();
		app->EnvironmentPresent();
		std::tie(manager->mLastMouseX, manager->mLastMouseY, manager->mMouseIn) = mouse;
		if (!RestoreEnvironmentSnapshot(app, snapshot)) throw std::runtime_error("could not restore presentation state");
		return action;
	}
};

// Search must not peek at the real future (DESIGN.md §13.3).  A restored snapshot carries the
// exact RNG state *and* the whole wave table drawn at level start, so every rollout from it
// would replay the true spawns.  Reseeding draws a fresh, equally plausible future instead:
// waves that have not started yet are re-picked from the level's own rules, waves already
// sent keep their zombies, and the wave count (including a task's wave cap) is unchanged.
// In-play randomness (rows, timings, sky sun) then continues on an independent stream.
static void ReseedEnvironment(LawnApp* app, uint32_t seed)
{
	Board* board = app->mBoard;
	const int numWaves = board->mNumWaves;
	const int sent = std::clamp(board->mCurrentWave, 0, static_cast<int>(MAX_ZOMBIE_WAVES));
	std::vector<ZombieType> kept(&board->mZombiesInWave[0][0],
		&board->mZombiesInWave[0][0] + static_cast<size_t>(sent) * MAX_ZOMBIES_IN_WAVE);
	SRand(seed);
	board->PickZombieWaves();
	board->mNumWaves = numWaves;
	std::copy(kept.begin(), kept.end(), &board->mZombiesInWave[0][0]);
	SRand(seed ^ 0x9E3779B9u);
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
	if (command == "COLLECT")
	{
		int x = -1, y = -1;
		input >> x >> y;
		if (input.fail() || !app->mBoard) return false;
		for (Coin* coin : app->mBoard->mCoins)
		{
			HitResult hit;
			if (!coin->mDead && (coin->IsSun() || coin->IsMoney()) && coin->MouseHitTest(x, y, &hit))
			{
				coin->ScoreCoin();
				return true;
			}
		}
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

static bool ReadStudentOptions(std::istream& input, int tick, EnvironmentRollout::History& history, std::string& tracePath)
{
	std::string token;
	if (!(input >> token)) return input.eof();
	if (token == "V2H")
	{
		int count = 0;
		if (!(input >> history.wave >> history.waveTick >> count) || history.wave < -1 || history.waveTick < 0 || history.waveTick > tick || count < 0 || count > 6 * 9 * 54) return false;
		for (int index = 0; index < count; ++index)
		{
			int row, col, kind, born;
			if (!(input >> row >> col >> kind >> born) || row < 0 || row >= 6 || col < 0 || col >= 9 || kind < 0 || kind >= 54 || born < 0 || born > tick ||
				!history.plants.emplace(std::make_tuple(row, col, kind), born).second) return false;
		}
		if (!(input >> token)) return input.eof();
	}
	return token == "TRACE" && static_cast<bool>(input >> std::quoted(tracePath)) && !tracePath.empty() && !(input >> token);
}

static void TraceStudent(std::ostream& output, EnvironmentRollout& rollout, LawnApp* app, double income,
	int priorTick, bool priorPlant, EnvironmentRollout::History& history)
{
	std::vector<float> board, scalars;
	rollout.Features(app, income, priorTick, priorPlant, board, scalars, history);
	output << std::setprecision(17) << "{\"observation\":" << app->EnvironmentLeanObservation()
		<< ",\"income\":" << income << ",\"previous_tick\":" << priorTick << ",\"previous_plant\":" << priorPlant << ",\"board\":[";
	for (size_t index = 0; index < board.size(); ++index) output << (index ? "," : "") << board[index];
	output << "],\"scalars\":[";
	for (size_t index = 0; index < scalars.size(); ++index) output << (index ? "," : "") << scalars[index];
	output << "]}\n";
}

static void RunEnvironment(LawnApp* app)
{
	std::string line;
	std::unordered_map<int, EnvironmentSnapshot> snapshots;
	int nextSnapshotId = 1;
	EnvironmentRollout rollout, rootPolicy;
	EnvironmentValueNet valueNets[2]; // leaf, static
	HumanInput human;
	bool demo = false;
	const bool visible = !app->mHeadlessMode;
	double actionSeconds = 0, observationSeconds = 0, controlSeconds = 0;
	std::cout << "PVZENV {\"ready\":true,\"protocol_version\":" << kEnvironmentProtocolVersion
		<< ",\"wait_events_version\":" << EnvironmentWaitEvents::Version
		<< ",\"rollout_version\":6,\"sparse_tracks_version\":1,\"generic_forward_version\":1}" << std::endl;
	while (std::getline(std::cin, line))
	{
		const auto commandStart = std::chrono::steady_clock::now();
		std::istringstream input(line);
		std::string command;
		input >> command;
		EnvCounters before = ReadCounters(app);
		bool ok = false;
		bool privileged = false;
		bool hasSnapshotId = false;
		bool minimalResponse = false;
		bool hasWaitResult = false;
		EnvironmentWaitEvents::Result waitResult;
		int snapshotId = 0;
		bool hasSubsteps = false;
		int previousTick = 0;
		std::vector<std::pair<int, uint32_t>> substeps;
		if (command == "DEMO_BEGIN")
		{
			demo = true;
			if (visible) SDL_GL_SetSwapInterval(0);
			app->mHeadlessMode = true;
			app->mWidgetManager->mMouseIn = false;
			app->mWidgetManager->mLastMouseX = app->mWidgetManager->mLastMouseY = 0;
			minimalResponse = ok = true;
		}
		else if (command == "HUMAN_POLL")
		{
			ok = demo && visible && app->mBoard;
			const std::string action = ok ? human.Poll(app) : "null";
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":" << (ok ? "true" : "false")
				<< ",\"action\":" << action << ",\"paused\":" << (human.paused ? "true" : "false")
				<< ",\"closed\":" << (human.closed ? "true" : "false") << '}' << std::endl;
			continue;
		}
			else if (command == "DEMO_INPUT")
			{
			// task80 test seam: push a synthetic event into the very SDL queue the recorder
			// reads, so a scripted regression drives the real Esc/Space/window-close path
			// instead of a parallel one.  Never accepted outside a visible demo session.
			std::string key;
			input >> key;
			ok = demo && visible && app->mBoard &&
				(key == "escape" || key == "space" || key == "close");
			if (ok)
			{
				SDL_Event event{};
				if (key == "close") event.type = SDL_QUIT;
				else
				{
					event.type = SDL_KEYDOWN;
					event.key.keysym.sym = key == "escape" ? SDLK_ESCAPE : SDLK_SPACE;
				}
				SDL_PushEvent(&event);
			}
				minimalResponse = true;
			}
			else if (command == "DEMO_CLICK_PACKET")
			{
				int index = -1;
				ok = static_cast<bool>(input >> index) && demo && visible && app->mBoard &&
					index >= 0 && index < app->mBoard->mSeedBank->mNumPackets;
				if (ok)
				{
					human.packet = index;
					human.shovel = false;
				}
				minimalResponse = true;
			}
			else if (command == "DEMO_CLICK_CELL")
			{
				int col = -1, row = -1;
				ok = static_cast<bool>(input >> col >> row) && demo && visible && app->mBoard &&
					col >= 0 && col < MAX_GRID_SIZE_X && row >= 0 && row < MAX_GRID_SIZE_Y;
				if (ok)
				{
					int x = app->mBoard->mX + app->mBoard->GridToPixelX(col, row) + 40;
					int y = app->mBoard->mY + app->mBoard->GridToPixelY(col, row) + 40;
					x = (x - app->mWidgetManager->mMouseDestRect.mX) * app->mWidgetManager->mMouseSourceRect.mWidth /
						app->mWidgetManager->mMouseDestRect.mWidth + app->mWidgetManager->mMouseSourceRect.mX;
					y = (y - app->mWidgetManager->mMouseDestRect.mY) * app->mWidgetManager->mMouseSourceRect.mHeight /
						app->mWidgetManager->mMouseDestRect.mHeight + app->mWidgetManager->mMouseSourceRect.mY;
					SDL_Event motion{};
					motion.type = SDL_MOUSEMOTION;
					motion.motion.x = x;
					motion.motion.y = y;
					SDL_PushEvent(&motion);
					SDL_Event click{};
					click.type = SDL_MOUSEBUTTONDOWN;
					click.button.button = SDL_BUTTON_LEFT;
					click.button.x = x;
					click.button.y = y;
					SDL_PushEvent(&click);
				}
				minimalResponse = true;
			}
		else if (command == "STATE_HASH")
		{
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":" << (app->mBoard ? "true" : "false");
			if (app->mBoard) std::cout << ",\"state_hash\":\"" << std::hex << StableEnvironmentHash(app) << std::dec << '"';
			std::cout << '}' << std::endl;
			continue;
		}
		else if (command == "STATE_BYTES")
		{
			std::string path;
			input >> std::quoted(path);
			ok = !input.fail() && !path.empty() && app->mBoard != nullptr && WriteEnvironmentStateBytes(app, path);
			minimalResponse = true;
		}
		else if (command == "SCREENSHOT")
		{
			// task66: visible runs only.  The frame is read back from the GL framebuffer inside
			// the process, so a sleeping display or a locked desktop cannot black it out.
			std::string path;
			input >> std::quoted(path);
			ok = !input.fail() && !path.empty() && visible && app->mBoard != nullptr &&
				app->EnvironmentScreenshot(path);
			minimalResponse = true;
		}
		else if (command == "ENV_PROFILE")
		{
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":true"
				<< ",\"action_seconds\":" << actionSeconds << ",\"observation_seconds\":" << observationSeconds
				<< ",\"control_seconds\":" << controlSeconds << ",\"value\":[";
			for (int slot = 0; slot < 2; ++slot)
			{
				const auto& net = valueNets[slot];
				std::cout << (slot ? "," : "") << "{\"calls\":" << net.calls << ",\"features\":" << net.featureSeconds
					<< ",\"forward\":" << net.forwardSeconds << ",\"conv\":[" << net.convolutionSeconds[0] << ','
					<< net.convolutionSeconds[1] << ',' << net.convolutionSeconds[2] << "]}";
			}
			std::cout << "]}" << std::endl;
			continue;
		}
		else if (command == "DECK_OPTIONS")
		{
			const std::string options = app->EnvironmentDeckOptions();
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ',' << options.substr(1) << std::endl;
			continue;
		}
		else if (command == "DECK_WARNINGS")
		{
			int count = 0;
			std::vector<int> deck;
			ok = static_cast<bool>(input >> count) && count >= 0 && count <= 10;
			for (int index = 0; ok && index < count; ++index)
			{
				int type = -1;
				ok = static_cast<bool>(input >> type) && type >= 0 && type < 64;
				if (ok) deck.push_back(type);
			}
			const std::string warnings = ok ? app->EnvironmentDeckWarnings(deck) : std::string("{\"ok\":false}");
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ',' << warnings.substr(1) << std::endl;
			continue;
		}
		else if (command == "DECK_POOL")
		{
			const std::string pool = app->EnvironmentDeckPool();
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ',' << pool.substr(1) << std::endl;
			continue;
		}
		else if (command == "LOAD_VALUE_NET")
		{
			int slot = -1;
			std::string path, extra;
			if (input >> slot >> std::quoted(path) && (slot == 0 || slot == 1) && !(input >> extra))
			{
				try { valueNets[slot].Load(path); ok = true; }
				catch (const std::exception& error) { std::cerr << error.what() << std::endl; }
			}
			minimalResponse = true;
		}
		else if (command == "LEAF_VALUES")
		{
			// Board mode scores without serializing the board. Feature-file mode also serves
			// external simulator leaves and the Python oracle; it runs the same forward.
			int slot = -1, count = 0;
			double income = 0;
			std::string mode, path, extra;
			ok = static_cast<bool>(input >> slot >> mode) && slot >= 0 && slot < 2 && valueNets[slot].Loaded();
			std::vector<float> logits;
			if (ok && mode == "BOARD")
			{
				ok = static_cast<bool>(input >> income) && std::isfinite(income) && app->mBoard;
				if (input >> std::quoted(path)) { if (input >> extra) ok = false; }
				if (ok)
				{
					std::vector<float> board, scalars;
					valueNets[slot].Features(app, income, board, scalars);
					logits.push_back(valueNets[slot].Logit(board, scalars));
					if (!path.empty())
					{
						std::ofstream trace(path, std::ios::binary);
						trace.write(reinterpret_cast<const char*>(board.data()), board.size() * sizeof(float));
						trace.write(reinterpret_cast<const char*>(scalars.data()), scalars.size() * sizeof(float));
						ok = trace.good();
					}
				}
			}
			else if (ok && mode == "FEATURES")
			{
				ok = static_cast<bool>(input >> count >> std::quoted(path)) && count > 0 && count <= 4096 && !(input >> extra);
				if (ok)
				{
					std::ifstream features(path, std::ios::binary);
					std::vector<float> board(valueNets[slot].ChannelCount() * EnvironmentValueNet::Cells), scalars(valueNets[slot].ScalarCount());
					for (int index = 0; ok && index < count; ++index)
					{
						features.read(reinterpret_cast<char*>(board.data()), board.size() * sizeof(float));
						features.read(reinterpret_cast<char*>(scalars.data()), scalars.size() * sizeof(float));
						ok = features.good() && std::all_of(board.begin(), board.end(), [](float x) { return std::isfinite(x); }) &&
							std::all_of(scalars.begin(), scalars.end(), [](float x) { return std::isfinite(x); });
						if (ok) logits.push_back(valueNets[slot].Logit(board, scalars));
					}
					ok = ok && features.peek() == std::char_traits<char>::eof();
				}
			}
			else ok = false;
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":" << (ok ? "true" : "false") << ",\"logits\":[" << std::setprecision(9);
			for (size_t i = 0; ok && i < logits.size(); ++i) std::cout << (i ? "," : "") << logits[i];
			std::cout << "],\"values\":[";
			for (size_t i = 0; ok && i < logits.size(); ++i) std::cout << (i ? "," : "") << EnvironmentValueNet::Probability(logits[i]);
			std::cout << "]}" << std::endl << std::setprecision(6);
			continue;
		}
		else if (command == "STATIC_SCORES")
		{
			int count = 0;
			double income = 0;
			std::string extra;
			ok = static_cast<bool>(input >> snapshotId >> income >> count) && snapshots.contains(snapshotId) &&
				std::isfinite(income) && count > 0 && count <= 1024 && valueNets[1].Loaded();
			std::vector<std::string> specs;
			for (int i = 0; ok && i < count; ++i)
			{
				std::string spec, actionCommand, actionArguments;
				ok = static_cast<bool>(input >> spec) && DecodeBranchAction(spec, actionCommand, actionArguments);
				if (ok) specs.push_back(spec);
			}
			ok = ok && !(input >> extra);
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":" << (ok ? "true" : "false") << ",\"values\":[" << std::setprecision(9);
			for (int i = 0; ok && i < count; ++i)
			{
				if (i) std::cout << ',';
				bool branchOk = RestoreEnvironmentSnapshot(app, snapshots.at(snapshotId));
				std::string actionCommand, actionArguments;
				if (branchOk) branchOk = DecodeBranchAction(specs[i], actionCommand, actionArguments);
				if (branchOk) { std::istringstream arguments(actionArguments); branchOk = ExecuteEnvironmentAction(app, actionCommand, arguments); }
				if (branchOk) std::cout << valueNets[1].Predict(app, income);
				else std::cout << "null";
			}
			std::cout << "]}" << std::endl << std::setprecision(6);
			continue;
		}
		else if (command == "LOAD_POLICY_CONFIG")
		{
			std::string path, extra;
			if (input >> std::quoted(path) && !(input >> extra))
			{
				try {
					std::ifstream config(path);
					std::string magic, key, model, rootPath, rolloutPath;
					config >> magic;
					if (magic != "PVZPOL001") throw std::runtime_error("policy config magic");
					while (config >> key) {
						if (!(config >> std::quoted(model))) throw std::runtime_error("policy config path");
						auto resolved = (std::filesystem::path(path).parent_path() / model).string();
						if (key == "root" && rootPath.empty()) rootPath = resolved;
						else if (key == "rollout" && rolloutPath.empty()) rolloutPath = resolved;
						else throw std::runtime_error("unknown/duplicate policy config key");
					}
					if (rootPath.empty() || rolloutPath.empty()) throw std::runtime_error("policy config requires root and rollout");
					EnvironmentRollout nextRoot, nextRollout;
					nextRoot.Load(rootPath); nextRollout.Load(rolloutPath);
					rootPolicy = std::move(nextRoot); rollout = std::move(nextRollout); ok = true;
				} catch (const std::exception& error) { std::cerr << error.what() << std::endl; }
			}
			minimalResponse = true;
		}
		else if (command == "LOAD_ROLLOUT_NET")
		{
			std::string path, extra;
			if (input >> std::quoted(path) && !(input >> extra))
			{
				try { rollout.Load(path); ok = true; }
				catch (const std::exception& error) { std::cerr << error.what() << std::endl; }
			}
			minimalResponse = true;
		}
		else if (command == "ROLLOUT_PREDICT" || command == "ROOT_PREDICT")
		{
			auto& policy = command == "ROOT_PREDICT" ? rootPolicy : rollout;
			int priorTick = -1, priorPlant = 0;
			double income = 0;
			std::string tracePath;
			EnvironmentRollout::History studentHistory;
			ok = static_cast<bool>(input >> priorTick >> priorPlant >> income) &&
				policy.Loaded() && app->mBoard && std::isfinite(income) && (priorPlant == 0 || priorPlant == 1);
			ok = ok && ReadStudentOptions(input, app->mBoard->mMainCounter, studentHistory, tracePath);
			std::ofstream trace;
			if (ok && !tracePath.empty()) { trace.open(tracePath); ok = trace.good(); }
			EnvironmentRollout::Prediction prediction{};
			if (ok)
			{
				try {
					if (trace.is_open()) TraceStudent(trace, policy, app, income, priorTick, priorPlant, studentHistory);
					prediction = policy.Predict(app, income, priorTick, priorPlant, studentHistory, true);
				} catch (const std::exception& error) { std::cerr << error.what() << std::endl; ok = false; }
			}
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":" << (ok ? "true" : "false");
			if (ok)
			{
				std::cout << std::setprecision(9) << ",\"plant\":" << prediction.plant << ",\"packet\":[";
				for (int index = 0; index < 10; ++index) std::cout << (index ? "," : "") << prediction.packets[index];
				std::cout << "],\"cell_packet\":" << prediction.cellPacket << ",\"cell\":[";
				for (int index = 0; index < 54; ++index) std::cout << (index ? "," : "") << prediction.cells[index];
				std::cout << "],\"action\":{";
				if (prediction.action.packet < 0) std::cout << "\"type\":\"wait\",\"ticks\":60,\"until\":\"timeout\"";
				else std::cout << "\"type\":\"plant\",\"packet\":" << prediction.action.packet << ",\"row\":" << prediction.action.row << ",\"col\":" << prediction.action.col;
				std::cout << '}';
			}
			std::cout << '}' << std::endl << std::setprecision(6);
			continue;
		}
		else if (command == "ROLLOUT" || command == "ROLLOUT_VALUE")
		{
			const bool scoreValue = command == "ROLLOUT_VALUE";
			int untilTick = 0, mask = 0, maxActions = 0, eventWait = 0, historyStart = 0, historyCount = 0;
			int rootId = 0;
			int64_t seed = -1;
			std::string first, tracePath;
			EnvironmentRollout::History studentHistory;
			std::ofstream trace;
			std::deque<std::pair<int, uint32_t>> history;
			ok = static_cast<bool>(input >> rootId >> seed >> untilTick >> mask >> maxActions >> eventWait >> first >> historyStart >> historyCount) &&
				rollout.Loaded() && (!scoreValue || valueNets[0].Loaded()) && app->mBoard && mask >= 0 && !(mask & ~ENV_SKIP_ALLOWED) &&
				maxActions > 0 && maxActions <= 1000000 && (eventWait == 0 || eventWait == 1) && historyCount >= 0 && historyCount <= 100000 &&
				rootId >= 0 && (rootId == 0 || snapshots.contains(rootId)) && seed >= -1 && seed <= UINT32_MAX;
			if (ok && rootId) ok = RestoreEnvironmentSnapshot(app, snapshots.at(rootId));
			if (ok && seed >= 0) ReseedEnvironment(app, static_cast<uint32_t>(seed));
			// The events block must describe the rollout, not the board the previous command left
			// behind: re-read the counters from the restored root, so the deltas are per-rollout.
			if (ok) before = ReadCounters(app);
			for (int index = 0; ok && index < historyCount; ++index)
			{
				int tick = 0;
				uint32_t produced = 0;
				ok = static_cast<bool>(input >> tick >> produced) && tick >= historyStart && tick <= app->mBoard->mMainCounter &&
					(history.empty() || tick >= history.back().first);
				if (ok) history.emplace_back(tick, produced);
			}
			std::string actionCommand, actionArguments;
			ok = ok && ReadStudentOptions(input, app->mBoard->mMainCounter, studentHistory, tracePath) && historyStart >= 0 && historyStart <= app->mBoard->mMainCounter &&
				(first == "A" || DecodeBranchAction(first, actionCommand, actionArguments));
			if (ok && !tracePath.empty()) { trace.open(tracePath); ok = trace.good(); }
			int decisions = 0, mowersLeft = 0, estimatedResult = -1, startTick = 0;
			std::array<int, 10> planted{};
			std::array<int, 54> plantedByType{};
			double simulationSeconds = 0, networkSeconds = 0;
			if (ok)
			{
				const int savedMask = gEnvSimSkip;
				gEnvSimSkip = mask;
				startTick = app->mBoard->mMainCounter;
				const bool finiteHorizon = untilTick > startTick;
				if (untilTick <= app->mBoard->mMainCounter) untilTick = std::numeric_limits<int>::max();
				int priorTick = -1;
				bool priorPlant = false;
				auto readyMowers = [&] { int count = 0; for (const LawnMower* mower : app->mBoard->mLawnMowers)
					if (!mower->mDead && mower->mMowerState == LawnMowerState::MOWER_READY) ++count; return count; };
				mowersLeft = readyMowers();
				rollout.ResetNetworkCalls();
				while (!app->EnvironmentTerminal() && decisions < maxActions)
				{
					const int tick = app->mBoard->mMainCounter;
					const uint32_t producedBefore = app->mBoard->mSunMoneyProduced;
					EnvironmentRollout::Action action;
					bool isFirst = decisions == 0 && first != "A";
					while (!history.empty() && history.front().first < tick - 6000) history.pop_front();
					double totalProduced = 0;
					for (auto entry : history) totalProduced += entry.second;
					const int elapsed = std::max(0, std::min(6000, tick - historyStart));
					const double income = elapsed ? totalProduced * 1000.0 / elapsed : 0;
					if (trace.is_open()) TraceStudent(trace, rollout, app, income, priorTick, priorPlant, studentHistory);
					if (isFirst && rollout.HasHistory()) studentHistory.Update(app);
					if (!isFirst)
					{
						const auto start = std::chrono::steady_clock::now();
						action = rollout.Predict(app, income, priorTick, priorPlant, studentHistory).action;
						networkSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
					}
					const auto start = std::chrono::steady_clock::now();
					bool accepted = true;
					int plantedType = -1;
					int plantSlot = action.packet;
					if (isFirst && actionCommand == "PLANT")
					{
						std::istringstream arguments(actionArguments);
						arguments >> plantSlot;
					}
					if ((!isFirst || actionCommand == "PLANT") && plantSlot >= 0 && plantSlot < app->mBoard->mSeedBank->mNumPackets)
					{
						const SeedPacket& packet = app->mBoard->mSeedBank->mSeedPackets[plantSlot];
						plantedType = static_cast<int>(packet.mPacketType == SeedType::SEED_IMITATER ? packet.mImitaterType : packet.mPacketType);
					}
					if (isFirst)
					{
						std::istringstream arguments(actionArguments);
						if (actionCommand == "WAIT" && eventWait)
						{
							int ticks = -1;
							accepted = static_cast<bool>(arguments >> ticks) && ticks >= 0 && ticks <= 1000000;
							if (accepted) app->EnvironmentWaitUntil(ticks, 0);
						}
						else accepted = ExecuteEnvironmentAction(app, actionCommand, arguments);
						priorPlant = actionCommand == "PLANT" && accepted;
						if (priorPlant) { arguments.clear(); arguments.str(actionArguments); arguments >> action.packet; }
					}
					else
					{
						priorPlant = action.packet >= 0;
						if (priorPlant) accepted = app->EnvironmentPlant(action.packet, action.col, action.row);
						else app->EnvironmentWaitUntil(60, 0);
					}
					if (!accepted) { app->EnvironmentWait(60); priorPlant = false; }
					if (priorPlant)
					{
						++planted[action.packet];
						if (plantedType >= 0 && plantedType < static_cast<int>(plantedByType.size())) ++plantedByType[plantedType];
					}
					priorTick = tick;
					const int now = app->mBoard->mMainCounter;
					const uint32_t produced = app->mBoard->mSunMoneyProduced - producedBefore;
					if (produced) history.emplace_back(now, produced);
					substeps.emplace_back(now, produced);
					simulationSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
					++decisions;
					// 胜利会放出剩余小推车，与 Python 一样保留最后非终局观测的数量。
					if (!app->EnvironmentTerminal()) mowersLeft = readyMowers();
					if (now >= untilTick) break;
					if (!app->EnvironmentTerminal() && (mask & ENV_EARLY_RESULT) && app->mBoard->mLevel == 7)
					{
						bool protectedRows[6]{};
						for (const LawnMower* mower : app->mBoard->mLawnMowers)
							if (!mower->mDead && mower->mRow >= 0 && mower->mRow < 6 && mower->mMowerState == LawnMowerState::MOWER_READY)
								protectedRows[mower->mRow] = true;
						bool safe = app->mBoard->mCurrentWave == app->mBoard->mNumWaves && app->mBoard->mRiseFromGraveCounter == 0;
						for (Zombie* zombie : app->mBoard->mZombies)
						{
							if (zombie->mDead || zombie->IsDeadOrDying() || zombie->mMindControlled) continue;
							const bool protectedRow = zombie->mRow >= 0 && zombie->mRow < 6 && protectedRows[zombie->mRow];
							safe = safe && protectedRow;
							if (!protectedRow && zombie->mPosX < 80.0f && zombie->mHasHead)
							{
								estimatedResult = 0;
								break;
							}
						}
						if (safe) estimatedResult = 1;
						if (estimatedResult >= 0) break;
					}
					if (finiteHorizon && (mask & ENV_ADAPTIVE_HORIZON) && now - startTick >= 600)
					{
						bool threatened = false;
						for (Zombie* zombie : app->mBoard->mZombies)
							if (!zombie->mDead && !zombie->IsDeadOrDying() && !zombie->mMindControlled && zombie->mPosX < 400.0f)
							{
								threatened = true;
								break;
							}
						if (!threatened) break;
					}
				}
				gEnvSimSkip = savedMask;
				if (trace.is_open()) { trace.flush(); ok = trace.good(); }
			}
			actionSeconds += simulationSeconds;
			if (scoreValue)
			{
				double value = 0;
				if (ok)
				{
					const int now = app->mBoard->mMainCounter;
					double produced = 0;
					for (auto sample : history) if (sample.first >= now - 6000) produced += sample.second;
					const int elapsed = std::max(0, std::min(6000, now - historyStart));
					if (estimatedResult >= 0) value = estimatedResult;
					else if (app->EnvironmentTerminal()) value = app->mBoardResult == BoardResult::BOARDRESULT_WON ? 1.0 : 0.0;
					else value = valueNets[0].Predict(app, elapsed ? produced * 1000.0 / elapsed : 0.0);
				}
				std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":" << (ok ? "true" : "false")
					<< ",\"observation\":{\"tick\":" << (app->mBoard ? app->mBoard->mMainCounter : 0)
					<< ",\"wave_timer\":0,\"terminal\":" << (app->EnvironmentTerminal() ? "true" : "false")
					<< ",\"result\":" << static_cast<int>(app->mBoardResult) << ",\"value\":" << std::setprecision(9) << value;
				if (estimatedResult >= 0) std::cout << ",\"rollout_result\":" << estimatedResult;
				std::cout << "},\"decisions\":" << decisions << ",\"simulation_seconds\":" << simulationSeconds
					<< ",\"network_seconds\":" << networkSeconds << "}" << std::endl << std::setprecision(6);
				continue;
			}
			const auto observationStart = std::chrono::steady_clock::now();
			const std::string observation = app->mBoard ? app->EnvironmentLeanObservation() : "null";
			observationSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - observationStart).count();
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":" << (ok ? "true" : "false")
				<< ",\"observation\":" << observation
				<< ",\"estimated_result\":" << estimatedResult
				<< ",\"decisions\":" << decisions << ",\"mowers_left\":" << mowersLeft << ",\"planted_slots\":[";
			for (int index = 0; index < 10; ++index) std::cout << (index ? "," : "") << planted[index];
			std::cout << "],\"planted_types\":{";
			bool firstType = true;
			for (size_t index = 0; index < plantedByType.size(); ++index)
				if (plantedByType[index]) { std::cout << (firstType ? "" : ",") << '\"' << index << "\":" << plantedByType[index]; firstType = false; }
			std::cout << "},\"simulation_seconds\":" << simulationSeconds << ",\"network_seconds\":" << networkSeconds
				<< ",\"net_calls\":" << rollout.NetworkCalls() << ",\"ticks\":" << app->mBoard->mMainCounter - startTick << ",\"substeps\":[";
			for (size_t index = 0; index < substeps.size(); ++index)
				std::cout << (index ? ",[" : "[") << substeps[index].first << ',' << substeps[index].second << ']';
			std::cout << "]}" << std::endl;
			continue;
		}
		else if (command == "RESET_V2")
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
				// Optional trailing token, absent in every pre-task20 task: the command text of
				// those tasks therefore stays byte-identical.
				std::string extra;
				while (input >> extra)
				{
					if (extra == "conveyor_original") task.originalConveyorStart = true;
					// fog_human keeps the pre-task66 cell rule so old archives replay byte for
					// byte; fog_human_v2 is the calibrated opacity rule (task66).
					else if (extra == "fog_human") task.fogObservation = EnvironmentFogObservation::Cell;
					else if (extra == "fog_human_v2") task.fogObservation = EnvironmentFogObservation::Opacity;
					else if (extra == "fog_storm") task.fogObservation = EnvironmentFogObservation::Storm;
					else if (extra == "fog_xray") task.fogObservation = EnvironmentFogObservation::None;
					else parsed = false;
				}
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
		else if (command == "PLANT" || command == "SHOVEL" || command == "WAIT" || command == "COLLECT")
		{
			ok = ExecuteEnvironmentAction(app, command, input);
		}
		else if (command == "WAIT_EVENT_V1")
		{
			int ticks = -1, condition = -1;
			std::string extra;
			input >> ticks >> condition;
			if (!input.fail() && !(input >> extra) && ticks >= 0 && ticks <= 1000000 && condition >= 0 && condition < 6)
			{
				waitResult = app->EnvironmentWaitUntil(ticks, condition);
				hasWaitResult = ok = true;
			}
		}
		else if (command == "WAIT_UNTIL_LEGAL")
		{
			// A search rollout's student can only wait while no plant is legal, so instead of one
			// round trip per ``WAIT_EVENT_V1 60 0`` this repeats exactly that command here until a
			// plant becomes legal, the level ends, or the board clock reaches ``untilTick``.  The
			// response is the last sub-step's, plus every sub-step's (end tick, sun produced) so the
			// caller's sun-income history is the same as after the individual commands, and the
			// tick the last sub-step started from (the caller's "previous observation").
			int untilTick = 0;
			std::string extra;
			input >> untilTick;
			ok = !input.fail() && !(input >> extra) && app->mBoard != nullptr;
			if (ok)
			{
				hasSubsteps = hasWaitResult = true;
				do
				{
					previousTick = app->mBoard->mMainCounter;
					const uint32_t producedBefore = ReadCounters(app).sunProduced;
					waitResult = app->EnvironmentWaitUntil(60, 0);
					if (!app->mBoard)
						break;
					substeps.emplace_back(app->mBoard->mMainCounter, ReadCounters(app).sunProduced - producedBefore);
				} while (!app->EnvironmentTerminal() && app->mBoard->mMainCounter < untilTick &&
					!app->EnvironmentAnyLegalPlant());
			}
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
		else if (command == "BRANCH_SNAPSHOT_FAST" || command == "BRANCH_OBSERVE")
		{
			const bool observeOnly = command == "BRANCH_OBSERVE";
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
						observation = observeOnly ? app->EnvironmentLeanObservation() : app->EnvironmentObservation(false);
						if (!observeOnly && !app->EnvironmentTerminal())
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
						<< ",\"plants_eaten_free\":" << branchAfter.plantsEatenFree - branchBefore.plantsEatenFree
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
		else if (command == "SIM_SKIP")
		{
			// Approximate simulation for search rollouts (EnvSimSkip in PvzpCommon.h); 0 = exact.
			// Process-wide and sticky until changed, so callers set it around rollouts only.
			int mask = -1;
			std::string extra;
			input >> mask;
			ok = !input.fail() && !(input >> extra) && mask >= 0 && (mask & ~ENV_SKIP_ALLOWED) == 0;
			if (ok)
				gEnvSimSkip = mask;
			minimalResponse = true;
		}
		else if (command == "OBS_V3")
		{
			int enabled = -1;
			std::string extra;
			input >> enabled;
			ok = !input.fail() && !(input >> extra) && (enabled == 0 || enabled == 1);
			if (ok) app->mEnvironmentTokenObservation = enabled == 1;
			minimalResponse = true;
		}
		else if (command == "OBS_LEAN")
		{
			// Compact observations (LawnApp::EnvironmentLeanObservation) for search rollouts.
			// Sticky until changed; PRIV and every caller that does not opt in see the full form.
			int lean = -1;
			std::string extra;
			input >> lean;
			ok = !input.fail() && !(input >> extra) && (lean == 0 || lean == 1);
			if (ok)
				app->mEnvironmentLeanObservation = lean == 1;
			minimalResponse = true;
		}
		else if (command == "RESEED")
		{
			uint32_t seed = 0;
			std::string extra;
			input >> seed;
			ok = !input.fail() && !(input >> extra) && app->mBoard != nullptr;
			if (ok)
				ReseedEnvironment(app, seed);
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

		if (demo && ok && (command == "PLANT" || command == "SHOVEL"))
		{
			human.packet = -1;
			human.shovel = false;
		}
		const auto observationStart = std::chrono::steady_clock::now();
		const double executionSeconds = std::chrono::duration<double>(observationStart - commandStart).count();
		if (command == "PLANT" || command == "SHOVEL" || command == "WAIT" || command == "WAIT_EVENT_V1" || command == "WAIT_UNTIL_LEGAL")
			actionSeconds += executionSeconds;
		else controlSeconds += executionSeconds;
		const std::string observation = app->EnvironmentObservation(privileged);
		observationSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - observationStart).count();
		std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":" << (ok ? "true" : "false") << ",\"observation\":" << observation;
		if (demo && app->mBoard) std::cout << ",\"state_hash\":\"" << std::hex << StableEnvironmentHash(app) << std::dec << '"';
		if (hasWaitResult)
		{
			std::cout << ",\"wait_result\":{\"version\":" << EnvironmentWaitEvents::Version
				<< ",\"condition\":\"" << EnvironmentWaitEvents::Conditions[waitResult.condition] << "\""
				<< ",\"requested_ticks\":" << waitResult.requestedTicks
				<< ",\"actual_ticks\":" << waitResult.actualTicks
				<< ",\"logic_steps\":" << waitResult.logicSteps
				<< ",\"stalled_clock_steps\":" << waitResult.stalledClockSteps
				<< ",\"initial_condition_satisfied\":" << (waitResult.initialConditionSatisfied ? "true" : "false")
				<< ",\"reason\":\"" << waitResult.reason << "\",\"triggered\":[";
			for (size_t i = 0; i < waitResult.triggered.size(); ++i)
			{
				if (i) std::cout << ',';
				std::cout << '"' << waitResult.triggered[i] << '"';
			}
			std::cout << "]}";
		}
		if (hasSubsteps)
		{
			std::cout << ",\"previous_tick\":" << previousTick << ",\"substeps\":[";
			for (size_t i = 0; i < substeps.size(); ++i)
				std::cout << (i ? ",[" : "[") << substeps[i].first << ',' << substeps[i].second << ']';
			std::cout << ']';
		}
		if (hasSnapshotId)
			std::cout << ",\"snapshot_id\":" << snapshotId;
		if (app->mBoard)
		{
			EnvCounters after = ReadCounters(app);
			std::cout << ",\"events\":{\"zombies_killed\":" << after.zombiesKilled - before.zombiesKilled
				<< ",\"plants_eaten\":" << after.plantsEaten - before.plantsEaten
			<< ",\"plants_eaten_free\":" << after.plantsEatenFree - before.plantsEatenFree
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
