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
#include <map>
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

struct RolloutRequest
{
	int rootId = 0;
	int64_t seed = -1;
	int untilTick = 0;
	int mask = 0;
	int maxActions = 0;
	bool eventWait = false;
	bool scoreValue = false;
	std::string first;
	int historyStart = 0;
	std::vector<std::pair<int, uint32_t>> history;
	const EnvironmentRollout::History* studentHistory = nullptr;
	const std::string* tracePath = nullptr;
};

struct RolloutOutcome
{
	bool ok = false;
	int decisions = 0, mowersLeft = 0, estimatedResult = -1, startTick = 0;
	std::array<int, 10> planted{};
	std::array<int, 54> plantedByType{};
	double simulationSeconds = 0, networkSeconds = 0;
	std::vector<std::pair<int, uint32_t>> substeps;
	double value = 0;
	bool terminal = false;
};

// One continuation: restore the root, reseed, play the first action ("A" = let the student pick it)
// and then let the student continue until untilTick, and score the leaf out of its income history.
// ROLLOUT and SEARCH both go through here, so the continuation loop exists exactly once.
static RolloutOutcome RunRollout(LawnApp* app, EnvironmentRollout& rollout, EnvironmentValueNet& leafValue,
	const RolloutRequest& request, std::unordered_map<int, EnvironmentSnapshot>& snapshots)
{
	RolloutOutcome outcome;
	std::ofstream trace;
	outcome.ok = true;
	if (request.rootId) outcome.ok = RestoreEnvironmentSnapshot(app, snapshots.at(request.rootId));
	if (!outcome.ok) return outcome;
	if (request.seed >= 0) ReseedEnvironment(app, static_cast<uint32_t>(request.seed));
	if (request.tracePath && !request.tracePath->empty())
	{
		trace.open(*request.tracePath);
		if (!trace.good()) { outcome.ok = false; return outcome; }
	}
	EnvironmentRollout::History studentHistory;
	if (request.studentHistory) studentHistory = *request.studentHistory;
	std::string actionCommand, actionArguments;
	// "A" lets the student pick the first action too; otherwise the spec must decode.
	if (request.first != "A" && !DecodeBranchAction(request.first, actionCommand, actionArguments))
	{
		outcome.ok = false;
		return outcome;
	}
	std::deque<std::pair<int, uint32_t>> history(request.history.begin(), request.history.end());
	const int savedMask = gEnvSimSkip;
	gEnvSimSkip = request.mask;
	outcome.startTick = app->mBoard->mMainCounter;
	const bool finiteHorizon = request.untilTick > outcome.startTick;
	int untilTick = request.untilTick;
	if (untilTick <= app->mBoard->mMainCounter) untilTick = std::numeric_limits<int>::max();
	int priorTick = -1;
	bool priorPlant = false;
	auto readyMowers = [&] { int count = 0; for (const LawnMower* mower : app->mBoard->mLawnMowers)
		if (!mower->mDead && mower->mMowerState == LawnMowerState::MOWER_READY) ++count; return count; };
	outcome.mowersLeft = readyMowers();
	rollout.ResetNetworkCalls();
	while (!app->EnvironmentTerminal() && outcome.decisions < request.maxActions)
	{
		const int tick = app->mBoard->mMainCounter;
		const uint32_t producedBefore = app->mBoard->mSunMoneyProduced;
		EnvironmentRollout::Action action;
		bool isFirst = outcome.decisions == 0 && request.first != "A";
		while (!history.empty() && history.front().first < tick - 6000) history.pop_front();
		double totalProduced = 0;
		for (auto entry : history) totalProduced += entry.second;
		const int elapsed = std::max(0, std::min(6000, tick - request.historyStart));
		const double income = elapsed ? totalProduced * 1000.0 / elapsed : 0;
		if (trace.is_open()) TraceStudent(trace, rollout, app, income, priorTick, priorPlant, studentHistory);
		if (isFirst && rollout.HasHistory()) studentHistory.Update(app);
		if (!isFirst)
		{
			const auto start = std::chrono::steady_clock::now();
			action = rollout.Predict(app, income, priorTick, priorPlant, studentHistory).action;
			outcome.networkSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
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
			if (actionCommand == "WAIT" && request.eventWait)
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
			++outcome.planted[action.packet];
			if (plantedType >= 0 && plantedType < static_cast<int>(outcome.plantedByType.size())) ++outcome.plantedByType[plantedType];
		}
		priorTick = tick;
		const int now = app->mBoard->mMainCounter;
		const uint32_t produced = app->mBoard->mSunMoneyProduced - producedBefore;
		if (produced) history.emplace_back(now, produced);
		outcome.substeps.emplace_back(now, produced);
		outcome.simulationSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
		++outcome.decisions;
		// 胜利会放出剩余小推车，与 Python 一样保留最后非终局观测的数量。
		if (!app->EnvironmentTerminal()) outcome.mowersLeft = readyMowers();
		if (now >= untilTick) break;
		if (!app->EnvironmentTerminal() && (request.mask & ENV_EARLY_RESULT) && app->mBoard->mLevel == 7)
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
					outcome.estimatedResult = 0;
					break;
				}
			}
			if (safe) outcome.estimatedResult = 1;
			if (outcome.estimatedResult >= 0) break;
		}
		if (finiteHorizon && (request.mask & ENV_ADAPTIVE_HORIZON) && now - outcome.startTick >= 600)
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
	if (trace.is_open()) { trace.flush(); outcome.ok = trace.good(); }
	outcome.terminal = app->EnvironmentTerminal();
	if (!request.scoreValue) return outcome;
	const int now = app->mBoard->mMainCounter;
	double produced = 0;
	for (auto sample : history) if (sample.first >= now - 6000) produced += sample.second;
	const int elapsed = std::max(0, std::min(6000, now - request.historyStart));
	if (std::getenv("PVZ_COMBO_DEBUG"))
		std::cerr << "leafnow=" << now << " produced=" << produced << " elapsed=" << elapsed
			<< " income=" << (elapsed ? produced * 1000.0 / elapsed : 0.0)
			<< " plants=" << app->mBoard->mPlants.mSize << " hist=" << history.size()
			<< " start=" << request.historyStart << std::endl;
	if (outcome.estimatedResult >= 0) outcome.value = outcome.estimatedResult;
	else if (app->EnvironmentTerminal()) outcome.value = app->mBoardResult == BoardResult::BOARDRESULT_WON ? 1.0 : 0.0;
	else outcome.value = leafValue.Predict(app, elapsed ? produced * 1000.0 / elapsed : 0.0);
	return outcome;
}

// The teacher's search, moved out of Python (task217).  One command answers a whole decision:
// candidates from the static value net plus the student's own top-k, successive halving over the
// same reseeded futures, the override margin, and the verification futures.  Everything below runs
// in-process on the primitives that already exist (RunRollout, EnvironmentRollout::Rank, the two
// value nets, the snapshot table), so a decision no longer needs a round trip per candidate.
// scripts/search_play.py keeps the Python version next to it until the two have been compared.
struct SearchCandidate
{
	char kind = 'W';   // 'P' plant with packet/col/row, 'W' the no-op wait (NOOP in Python), 'S' a
	                   // save-plan wait of `ticks` (an event wait until the timeout fires)
	int packet = -1, col = 0, row = 0;
	int upgradePacket = 0;   // --combo-candidates: the second step this plan forces (0 = base alone)
	int baseType = 0;        // the base's SeedType, so the run can tell a live base from a dead one
	int ticks = 0;           // 'S' only: the ticks the plan waited
};

static std::string CandidateKey(const SearchCandidate& candidate)
{
	if (candidate.kind != 'P') return "W";
	std::string key = "P " + std::to_string(candidate.packet) + " " + std::to_string(candidate.col) + " "
		+ std::to_string(candidate.row);
	if (candidate.upgradePacket) key += " u" + std::to_string(candidate.upgradePacket);
	return key;
}

static std::string CandidateJson(const SearchCandidate& candidate)
{
	if (candidate.kind == 'S')
		return "{\"type\":\"wait\",\"ticks\":" + std::to_string(candidate.ticks) + ",\"until\":\"timeout\"}";
	if (candidate.kind != 'P') return "{\"type\":\"wait\",\"ticks\":60,\"until\":\"timeout\"}";
	std::string json = "{\"type\":\"plant\",\"packet\":" + std::to_string(candidate.packet) + ",\"row\":"
		+ std::to_string(candidate.row) + ",\"col\":" + std::to_string(candidate.col);
	if (candidate.upgradePacket)
		json += ",\"combo\":{\"upgrade_packet\":" + std::to_string(candidate.upgradePacket)
			+ ",\"base_type\":" + std::to_string(candidate.baseType) + "}";
	return json + "}";
}

// The sun income rate the value nets read, at `tick`, from the decision's sun-production history:
// the same 6000-tick window pvz_env builds sun_income_rate with.
static double IncomeAt(const std::vector<std::pair<int, uint32_t>>& history, int historyStart, int tick)
{
	if (tick - historyStart <= 0) return 0.0;
	double produced = 0;
	for (const auto& sample : history) if (sample.first >= tick - 6000) produced += sample.second;
	const int elapsed = std::max(0, std::min(6000, tick - historyStart));
	return elapsed ? produced * 1000.0 / elapsed : 0.0;
}

static bool DecodeSearchCandidate(const std::string& spec, SearchCandidate& candidate)
{
	// "W" is the no-op wait, otherwise the branch-action form "P:packet:col:row[:upgrade]".
	if (spec == "W") { candidate = SearchCandidate(); return true; }
	std::string command, arguments;
	if (!DecodeBranchAction(spec, command, arguments) || command != "PLANT") return false;
	std::istringstream input(arguments);
	int packet = -1, col = -1, row = -1, upgrade = 0;
	if (!(input >> packet >> col >> row) || packet < 0 || col < 0 || row < 0) return false;
	if (!(input >> upgrade)) upgrade = 0;
	if (upgrade < 0) return false;
	candidate = SearchCandidate();
	candidate.kind = 'P';
	candidate.packet = packet;
	candidate.col = col;
	candidate.row = row;
	candidate.upgradePacket = upgrade;
	return true;
}

struct SearchParameters
{
	int horizon = 2500;
	int perCard = 1;
	int studentCandidates = 0;
	int samples = 3;
	int verifySamples = 0;
	int halvingFirstSamples = 1;
	int minGap = 300;
	int lastSearchTick = -1000000000;
	int comboBuffer = 25;
	int comboInterval = 600;
	int simSkip = 61;
	double marginRelative = 0.0;
	bool halving = true;
	int comboCandidates = 0;
	int candidateMode = 0;   // 0 = static scoring (production), 1 = the student's own top-k (task42 B3 arm)
	int studentTopk = 3;
	int comboMaxWait = 5000;
	int saveCandidates = 0;
	int saveChunk = 300;
	int saveMaxWait = 1800;
	int saveInterval = 600;
	double saveShare = 0.5;
};

// State a decision passes to the next one (--combo-candidates and the --save-* probes): the pending
// upgrade with the sun held back for it, and when each card was last probed.  RESET_V2 clears it.
// Upgrade SeedType -> the base SeedType it may be planted on (Plant::IsUpgradableTo).  The
// --combo-candidates list runs it the other way round: plant the base now, force the upgrade later.
static const std::vector<std::pair<int, int>> kComboBases = { { 40, 7 }, { 41, 1 }, { 42, 10 }, { 43, 16 },
	{ 44, 39 }, { 45, 31 }, { 46, 21 }, { 47, 34 } };

struct SearchState
{
	int pendingUpgrade = 0, pendingBase = 0, pendingCol = -1, pendingRow = -1, pendingExpires = 0;
	int lastComboTick = -1000000000;   // --combo-interval: when a base last armed a plan
	int pendingArmedTick = -1;         // decision that armed the plan (its base is planted after it)
	std::map<int, int> lastSaveProbe;
	void Reset()
	{
		pendingUpgrade = pendingBase = 0;
		pendingCol = pendingRow = -1;
		pendingExpires = 0;
		lastComboTick = -1000000000;
		pendingArmedTick = -1;
		lastSaveProbe.clear();
	}
};

// One --save-candidates plan: wait `ticks`, then plant `packet` on the best cell of the future board.
struct SaveCandidate
{
	bool ok = false;
	int packet = -1, cost = 0, ticks = 0, col = -1, row = -1, samples = 0;
	double value = 0, sun = 0;
};

struct SearchResult
{
	bool searched = false;
	bool overridden = false;
	SearchCandidate chosen;
	std::vector<SearchCandidate> candidates;
	std::vector<double> totals;
	std::vector<bool> studentCandidate;
	int policyIndex = 0, best = 0;
	std::string verification;
	int overrides = 0, halvingCandidates = 0, halvingRollouts = 0, halvingSelectedLanes = 0;
	int studentCandidateAdded = 0, studentCandidateBest = 0, studentCandidateOverride = 0;
	int rolloutBatches = 0, rolloutLanes = 0, demoCandidateAdded = 0;
	int verifyRollouts = 0;
	int comboCandidateAdded = 0, comboUpgrades = 0, comboHeld = 0;
	int saveProbes = 0, saveOverrides = 0, saveWaitTicks = 0;
};

// Where one decision's time goes (task217b): the search is deterministic, so the sections are the
// only way to see what a speed-up would even have to attack.
struct SearchProfile
{
	double totalSeconds = 0, staticSeconds = 0, restoreSeconds = 0, simulationSeconds = 0, networkSeconds = 0;
	double legalSeconds = 0;
	long long ticksSimulated = 0, studentForwards = 0, leafForwards = 0, rollouts = 0, verifyRollouts = 0;
	int staticEvaluations = 0, restores = 0, legalCalls = 0;
};

// RunRollout is shared with ROLLOUT, so the profiler is a scope pointer the search sets.
static SearchProfile* gProfile = nullptr;

struct ProfileScope
{
	SearchProfile* saved;
	explicit ProfileScope(SearchProfile* profile) : saved(gProfile) { gProfile = profile; }
	~ProfileScope() { gProfile = saved; }
};

// The override margin: absolute by default, opt-in relative (--margin-relative).  With the default
// 0 this is MARGIN unchanged, so the historical decision rule keeps the same float comparison.
static double Elapsed(const std::chrono::steady_clock::time_point& started)
{
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

static double SearchMargin(const SearchParameters& parameters, double bestValue)
{
	if (parameters.marginRelative <= 0) return 0.02;
	return std::max(0.001, parameters.marginRelative * bestValue);
}

class EnvironmentSearch
{
public:
	EnvironmentSearch(SearchState& theState, int& theNextSnapshotId) : state(theState), nextSnapshotId(theNextSnapshotId) {}

	// Scores one candidate plant from the root; leaves the engine on the branch it scored.
	bool ScoreStatic(LawnApp* app, EnvironmentValueNet& staticValue, int rootId, int packet, int col, int row,
		double income, std::unordered_map<int, EnvironmentSnapshot>& snapshots, double& score)
	{
		const auto startedScore = std::chrono::steady_clock::now();
		if (!RestoreEnvironmentSnapshot(app, snapshots.at(rootId))) return false;
		++profile.restores;
		if (!app->EnvironmentPlant(packet, col, row)) return false;
		score = staticValue.Predict(app, income);
		profile.staticSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - startedScore).count();
		++profile.staticEvaluations;
		return true;
	}

	// One continuation for a candidate, from the root, on the given future.
	RolloutOutcome Continue(LawnApp* app, EnvironmentRollout& rollout, EnvironmentValueNet& leafValue,
		std::unordered_map<int, EnvironmentSnapshot>& snapshots, int rootId, double income,
		const SearchParameters& parameters, uint32_t seed, int untilTick,
		const std::vector<std::pair<int, uint32_t>>& history, int historyStart,
		const EnvironmentRollout::History& studentHistory, const SearchCandidate& candidate)
	{
		RolloutRequest request;
		request.rootId = rootId;
		request.seed = seed;
		request.untilTick = untilTick;
		request.mask = parameters.simSkip;
		request.maxActions = 1000000;
		request.eventWait = true;
		request.scoreValue = true;
		request.first = candidate.kind == 'P'
			? "P:" + std::to_string(candidate.packet) + ":" + std::to_string(candidate.col) + ":" + std::to_string(candidate.row)
			: "W:60";
		request.historyStart = historyStart;
		request.history = history;
		request.studentHistory = &studentHistory;
		(void)income;
		return RunRollout(app, rollout, leafValue, request, snapshots);
	}

	static bool UpgradeLegal(const std::vector<std::array<int, 3>>& legal, int packet, int col, int row)
	{
		for (const auto& action : legal)
			if (action[0] == packet && action[1] == col && action[2] == row) return true;
		return false;
	}

	// One continuation for a --combo-candidates plan: plant the base, wait in the engine until the
	// upgrade is legal on that cell, plant it, then score the leaf --horizon ticks after the last
	// step.  Waiting in the engine (not letting the student act) mirrors the run, which holds the
	// upgrade's cost in reserve; with the student playing, sun never reached the upgrade and every
	// plan scored as "base + --combo-max-wait wasted ticks" (task217b, L33).
	RolloutOutcome ContinueCombo(LawnApp* app, EnvironmentRollout& rollout, EnvironmentValueNet& leafValue,
		std::unordered_map<int, EnvironmentSnapshot>& snapshots, int rootId, const SearchParameters& parameters,
		uint32_t seed, int untilTick, const std::vector<std::pair<int, uint32_t>>& history, int historyStart,
		const EnvironmentRollout::History& studentHistory, const SearchCandidate& candidate)
	{
		RolloutOutcome outcome;
		const auto restored = std::chrono::steady_clock::now();
		outcome.ok = RestoreEnvironmentSnapshot(app, snapshots.at(rootId));
		profile.restoreSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - restored).count();
		++profile.restores;
		if (!outcome.ok) return outcome;
		// No reseed before the wait loop: the run restores the root and waits, it does not reseed
		// either.  The final rollout reseeds in RunRollout.
		const int rootPlants = app->mBoard->mPlants.mSize;
		if (!app->EnvironmentPlant(candidate.packet, candidate.col, candidate.row))
		{
			outcome.ok = false;
			return outcome;
		}
		if (std::getenv("PVZ_COMBO_DEBUG"))
			std::cerr << "combo root plants=" << rootPlants << " after=" << app->mBoard->mPlants.mSize << std::endl;
		// The sun produced while waiting joins the history: the leaf is scored out of it, and the
		// future snapshot's history in the run already has it.
		std::vector<std::pair<int, uint32_t>> futureHistory = history;
		std::vector<std::array<int, 3>> legal;
		int waited = 0;
		bool placed = false;
		while (waited < parameters.comboMaxWait && !app->EnvironmentTerminal())
		{
			legal.clear();
			LegalCells(app, legal);
			if (UpgradeLegal(legal, candidate.upgradePacket, candidate.col, candidate.row)) { placed = true; break; }
			const int chunk = std::min(250, parameters.comboMaxWait - waited);
			const int before = app->mBoard->mMainCounter;
			const uint32_t producedBefore = ReadCounters(app).sunProduced;
			app->EnvironmentWait(chunk);
			const int now = app->mBoard->mMainCounter;
			waited += now - before;
			const uint32_t produced = ReadCounters(app).sunProduced - producedBefore;
			if (produced) futureHistory.emplace_back(now, produced);
			while (!futureHistory.empty() && futureHistory.front().first < now - 6000)
				futureHistory.erase(futureHistory.begin());
		}
		if (placed) app->EnvironmentPlant(candidate.upgradePacket, candidate.col, candidate.row);
		// Every plan gets --horizon ticks after its last step: the engine reads a deadline in the past
		// as "no limit", so a plan that waited out --combo-max-wait would otherwise run to the end of
		// the game, score 0 on every loss and always look worthless.
		untilTick = std::max(untilTick, static_cast<int>(app->mBoard->mMainCounter) + parameters.horizon);
		if (std::getenv("PVZ_SEARCH_DEBUG"))
			std::cerr << "combo end base=" << candidate.packet << '@' << candidate.col << ',' << candidate.row
				<< " waited=" << waited << " placed=" << placed << " until=" << untilTick
				<< " now=" << app->mBoard->mMainCounter << " sun=" << app->mBoard->mSunMoney << std::endl;
		RolloutRequest request;
		request.rootId = 0;   // the board already sits on the future state
		request.seed = seed;   // int64: a uint32 seed above 2**31 must stay positive
		request.untilTick = untilTick;
		request.mask = parameters.simSkip;
		request.maxActions = 1000000;
		request.eventWait = true;
		request.scoreValue = true;
		request.first = "A";
		request.historyStart = historyStart;
		request.history = futureHistory;
		request.studentHistory = &studentHistory;
		RolloutOutcome scored = RunRollout(app, rollout, leafValue, request, snapshots);
		if (std::getenv("PVZ_SEARCH_DEBUG"))
			std::cerr << "combo leaf ok=" << scored.ok << " tick=" << app->mBoard->mMainCounter
				<< " sun=" << app->mBoard->mSunMoney << " plants=" << app->mBoard->mPlants.mSize
				<< " value=" << std::setprecision(17) << scored.value << std::setprecision(6) << std::endl;
		return scored;
	}

	// A candidate is "terminal" when its leaf carries a decided or estimated result instead of a
	// value-net score; those accumulate first, exactly like the original CPU order.
	static bool TerminalLeaf(const RolloutOutcome& outcome)
	{
		return outcome.estimatedResult >= 0 || outcome.terminal;
	}

	void Run(LawnApp* app, EnvironmentRollout& rollout, EnvironmentValueNet& leafValue, EnvironmentValueNet& staticValue,
		std::unordered_map<int, EnvironmentSnapshot>& snapshots, int rootId, int64_t decisionSeed, double income,
		int previousTick, bool previousPlant, const SearchParameters& parameters,
		const std::vector<std::pair<int, uint32_t>>& history, int historyStart,
		const EnvironmentRollout::History& studentHistory, const SearchCandidate& policy,
		const SearchCandidate* demo)
	{
		result = SearchResult();
		profile = SearchProfile();
		ProfileScope scope(&profile);
		const auto started = std::chrono::steady_clock::now();
		result.chosen = policy;
		const int tick = app->mBoard->mMainCounter;
		const int untilTick = tick + parameters.horizon;
		// The engine only lists plant actions the player can pay for right now.
		std::vector<std::array<int, 3>> legal;
		LegalCells(app, legal);
		if (legal.empty()) { profile.totalSeconds = Elapsed(started); return; }
		if (parameters.minGap && policy.kind != 'P' && tick - parameters.lastSearchTick < parameters.minGap)
		{ profile.totalSeconds = Elapsed(started); return; }
		result.searched = true;

		std::vector<std::pair<double, SearchCandidate>> scored;
		std::map<std::string, double> candidateScore;
		std::vector<SearchCandidate> candidates;
		if (parameters.candidateMode == 1)
		{
			// The experimental arm (task42 B3) ranks the legal plants with the student instead of the
			// static net: its global top-k, plus wait and the student's own action.  No per-card floor.
			RestoreEnvironmentSnapshot(app, snapshots.at(rootId));
			std::vector<EnvironmentRollout::Action> rankable;
			for (const auto& action : legal) rankable.push_back({ action[0], action[1], action[2] });
			std::vector<EnvironmentRollout::Ranked> ranked;
			EnvironmentRollout::History rankHistory = studentHistory;
			rollout.Rank(app, income, previousTick, previousPlant, rankHistory, rankable, ranked);
			std::vector<std::string> keys;
			for (size_t index = 0; index < ranked.size() && static_cast<int>(index) < parameters.studentTopk; ++index)
			{
				SearchCandidate candidate;
				candidate.kind = 'P';
				candidate.packet = ranked[index].action.packet;
				candidate.col = ranked[index].action.col;
				candidate.row = ranked[index].action.row;
				const std::string key = CandidateKey(candidate);
				if (std::find(keys.begin(), keys.end(), key) != keys.end()) continue;
				keys.push_back(key);
				candidates.push_back(candidate);
			}
			candidates.push_back({ 'W', -1, 0, 0 });
		}
		else
		{
		// The best cells of *every* card, not the overall top-k: the static net was trained on games
		// where some cards were never played, so it cannot rank them against shooters right after
		// planting -- only the rollout can show what they are worth.
		for (const auto& action : legal)
		{
			double score = 0;
			SearchCandidate candidate;
			candidate.kind = 'P';
			candidate.packet = action[0];
			candidate.col = action[1];
			candidate.row = action[2];
			if (ScoreStatic(app, staticValue, rootId, candidate.packet, candidate.col, candidate.row, income, snapshots, score))
			{
				scored.emplace_back(score, candidate);
				candidateScore[CandidateKey(candidate)] = score;
			}
		}
		std::stable_sort(scored.begin(), scored.end(),
			[](const auto& left, const auto& right) { return left.first > right.first; });
		{
			std::vector<int> cardOrder;
			std::map<int, std::vector<SearchCandidate>> byCard;
			for (const auto& entry : scored)
			{
				auto found = byCard.find(entry.second.packet);
				if (found == byCard.end()) cardOrder.push_back(entry.second.packet);
				byCard[entry.second.packet].push_back(entry.second);
			}
			for (int packet : cardOrder)
			{
				auto& cells = byCard[packet];
				for (int index = 0; index < parameters.perCard && index < static_cast<int>(cells.size()); ++index)
					candidates.push_back(cells[static_cast<size_t>(index)]);
			}
			candidates.push_back({ 'W', -1, 0, 0 });
		}
		}
		auto present = [&] { std::vector<std::string> keys; for (const auto& candidate : candidates) keys.push_back(CandidateKey(candidate)); return keys; };
		const std::string policyKey = CandidateKey(policy);
		{
			std::vector<std::string> keys = present();
			if (policy.kind == 'P' && std::find(keys.begin(), keys.end(), policyKey) == keys.end())
				candidates.push_back(policy);
		}
		result.studentCandidate.assign(candidates.size(), parameters.candidateMode == 1);
		// --student-candidates k: the student's own top-k plants join the candidates.  The static net
		// only offers one cell per card, so a cell the student would actually pick is reachable
		// otherwise only by shovelling first.  The net reads the root, so restore it first.
		if (parameters.studentCandidates > 0)
		{
			RestoreEnvironmentSnapshot(app, snapshots.at(rootId));
			std::vector<EnvironmentRollout::Action> rankable;
			for (const auto& action : legal) rankable.push_back({ action[0], action[1], action[2] });
			std::vector<EnvironmentRollout::Ranked> ranked;
			// The student reads its previous action too, so the probe must see the same one it did.
			EnvironmentRollout::History historyCopy = studentHistory;
			rollout.Rank(app, income, previousTick, previousPlant, historyCopy, rankable, ranked);
			if (std::getenv("PVZ_SEARCH_DEBUG"))
			{
				std::cerr << "rank(" << previousTick << "):";
				for (size_t index = 0; index < ranked.size() && index < 8; ++index)
					std::cerr << ' ' << ranked[index].action.packet << '/' << ranked[index].action.row << '/'
						<< ranked[index].action.col << '=' << ranked[index].score;
				std::cerr << std::endl;
			}
			std::vector<std::string> keys = present();
			int added = 0;
			for (const auto& entry : ranked)
			{
				if (added >= parameters.studentCandidates) break;
				SearchCandidate candidate{ 'P', entry.action.packet, entry.action.col, entry.action.row };
				const std::string key = CandidateKey(candidate);
				if (std::find(keys.begin(), keys.end(), key) != keys.end()) continue;
				candidates.push_back(candidate);
				result.studentCandidate.push_back(true);
				keys.push_back(key);
				++added;
			}
			result.studentCandidateAdded += added;
		}
		// --combo-candidates k: a base plant now plus a forced upgrade later, scored as one plan.  Only
		// pairs whose upgrade card is not already legal (a legal upgrade is its own candidate) and only
		// cells the static net already ranked.  k = 0 leaves the set untouched.
		if (parameters.comboCandidates > 0 && tick - state.lastComboTick >= parameters.comboInterval)
		{
			std::map<int, int> packetType;   // packet slot -> SeedType
			for (int slot = 0; slot < app->mBoard->mSeedBank->mNumPackets; ++slot)
			{
				const SeedType type = app->mBoard->mSeedBank->mSeedPackets[slot].mPacketType;
				if (type != SeedType::SEED_NONE) packetType[slot] = static_cast<int>(type);
			}
			auto isLegalPacket = [&](int packet)
			{
				for (const auto& action : legal) if (action[0] == packet) return true;
				return false;
			};
			int added = 0;
			for (const auto& pair : kComboBases)
			{
				std::vector<int> upgradePackets;
				for (const auto& entry : packetType)
					if (entry.second == pair.first) upgradePackets.push_back(entry.first);
				if (added >= parameters.comboCandidates || upgradePackets.empty() || isLegalPacket(upgradePackets[0]))
					continue;
				const SearchCandidate* bestBase = nullptr;
				double bestScore = -std::numeric_limits<double>::infinity();
				for (const auto& candidate : candidates)
				{
					if (candidate.kind != 'P') continue;
					auto found = packetType.find(candidate.packet);
					if (found == packetType.end() || found->second != pair.second) continue;
					auto score = candidateScore.find(CandidateKey(candidate));
					const double value = score == candidateScore.end()
						? -std::numeric_limits<double>::infinity() : score->second;
					if (!bestBase || value > bestScore) { bestBase = &candidate; bestScore = value; }
				}
				if (!bestBase) continue;
				SearchCandidate combo = *bestBase;
				combo.upgradePacket = upgradePackets[0];
				combo.baseType = pair.second;
				candidates.push_back(combo);
				result.studentCandidate.push_back(false);
				state.lastComboTick = tick;
				++added;
			}
			result.comboCandidateAdded += added;
		}
		// A demonstration start offers its recorded plant once more, like the Python path does.
		if (demo && demo->kind == 'P')
		{
			std::vector<std::string> keys = present();
			const std::string key = CandidateKey(*demo);
			if (std::find(keys.begin(), keys.end(), key) == keys.end())
			{
				candidates.push_back(*demo);
				result.studentCandidate.push_back(false);
				++result.demoCandidateAdded;
			}
		}
		result.candidates = candidates;
		for (size_t index = 0; index < candidates.size(); ++index)
			if (CandidateKey(candidates[index]) == policyKey) { result.policyIndex = static_cast<int>(index); break; }

		const long long rolloutsBefore = profile.rollouts;
		if (parameters.halving)
			HalvingChoice(app, rollout, leafValue, staticValue, snapshots, rootId, decisionSeed, income, parameters,
				history, historyStart, studentHistory, untilTick, policy, tick);
		else
			FullChoice(app, rollout, leafValue, staticValue, snapshots, rootId, decisionSeed, income, parameters,
				history, historyStart, studentHistory, untilTick, policy, tick);
		// The response is emitted by the caller, so the profile is closed out here.
		profile.totalSeconds = Elapsed(started);
		(void)rolloutsBefore;
		if (result.chosen.kind == 'P' && result.chosen.upgradePacket)
		{
			// The caller executes the chosen plant right away, so the plan locks here.
			state.pendingUpgrade = result.chosen.upgradePacket;
			state.pendingBase = result.chosen.baseType;
			state.pendingCol = result.chosen.col;
			state.pendingRow = result.chosen.row;
			state.pendingExpires = -1;   // armed; the next decision stamps tick + --combo-max-wait
			state.pendingArmedTick = tick;
			state.lastComboTick = tick;
		}
	}

	// Runs after Run().  The pending upgrade is checked at *every* decision, including the ones the
	// search itself skips (no legal plant, --min-gap), exactly like the Python loop, which checks it
	// outside the search.
	void FinishDecision(LawnApp* app, const SearchParameters& parameters, int tick)
	{
		// A plan armed by this decision's own action has not been planted yet (the caller executes it
		// after this response), so it cannot fire here -- Python arms it after the step, too.
		if (state.pendingUpgrade && state.pendingArmedTick == tick) return;
		if (state.pendingUpgrade && state.pendingExpires < 0)
			state.pendingExpires = tick + parameters.comboMaxWait;
		ForcedUpgrade(app, parameters, tick);
	}

	SearchResult result;
	SearchProfile profile;

private:
	// The cell scan walks every plant against every cell; it is timed on its own because it is the
	// one bucket a candidate generator can remove entirely.
	void LegalCells(LawnApp* app, std::vector<std::array<int, 3>>& out)
	{
		const auto started = std::chrono::steady_clock::now();
		app->EnvironmentLegalPlants(out);
		profile.legalSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
		++profile.legalCalls;
	}

	SearchState& state;
	int& nextSnapshotId;

	void AppendBatch(LawnApp* app, EnvironmentRollout& rollout, EnvironmentValueNet& leafValue,
		std::unordered_map<int, EnvironmentSnapshot>& snapshots, int rootId, int64_t decisionSeed, double income,
		const SearchParameters& parameters, const std::vector<std::pair<int, uint32_t>>& history, int historyStart,
		const EnvironmentRollout::History& studentHistory, int untilTick,
		const std::vector<std::pair<int, int>>& order, std::vector<std::pair<int, bool>>& terminal, std::vector<double>& values)
	{
		result.rolloutBatches++;
		result.rolloutLanes += static_cast<int>(order.size());
		terminal.clear();
		values.clear();
		for (const auto& job : order)
		{
			// Same futures as the Python path: (decision_seed * 1000 + k) % 2**32.
			const uint32_t seed = static_cast<uint32_t>((decisionSeed * 1000 + job.second) & 0xffffffff);
			const SearchCandidate& candidate = result.candidates[static_cast<size_t>(job.first)];
			RolloutOutcome outcome = candidate.upgradePacket
				? ContinueCombo(app, rollout, leafValue, snapshots, rootId, parameters, seed, untilTick, history,
					historyStart, studentHistory, candidate)
				: Continue(app, rollout, leafValue, snapshots, rootId, income, parameters, seed, untilTick,
					history, historyStart, studentHistory, candidate);
			terminal.emplace_back(job.first, TerminalLeaf(outcome));
			values.push_back(outcome.value);
			profile.rollouts++;
			profile.simulationSeconds += outcome.simulationSeconds;
			profile.networkSeconds += outcome.networkSeconds;
			profile.leafForwards++;
			profile.studentForwards += std::max(0, outcome.decisions - 1);
			if (!outcome.substeps.empty())
				profile.ticksSimulated += outcome.substeps.back().first - outcome.startTick;
		}
	}

	void HalvingChoice(LawnApp* app, EnvironmentRollout& rollout, EnvironmentValueNet& leafValue,
		EnvironmentValueNet& staticValue, std::unordered_map<int, EnvironmentSnapshot>& snapshots, int rootId,
		int64_t decisionSeed, double income,
		const SearchParameters& parameters, const std::vector<std::pair<int, uint32_t>>& history, int historyStart,
		const EnvironmentRollout::History& studentHistory, int untilTick, const SearchCandidate& policy, int tick)
	{
		std::vector<std::vector<double>> values(result.candidates.size());
		std::vector<int> alive;
		for (int index = 0; index < static_cast<int>(result.candidates.size()); ++index) alive.push_back(index);
		result.halvingCandidates += static_cast<int>(result.candidates.size());
		int selectedLanes = 0, done = 0;
		// Round 1 gives every candidate one future before the first cut; later rounds double.
		int target = std::min(parameters.samples, std::max(1, parameters.halvingFirstSamples));
		while (true)
		{
			target = std::min(target, parameters.samples);
			std::vector<std::pair<int, int>> order;
			for (int index : alive)
				for (int future = done; future < target; ++future) order.emplace_back(index, future);
			selectedLanes += static_cast<int>(order.size());
			std::vector<std::pair<int, bool>> terminal;
			std::vector<double> batch;
			AppendBatch(app, rollout, leafValue, snapshots, rootId, decisionSeed, income, parameters, history,
				historyStart, studentHistory, untilTick, order, terminal, batch);
			std::vector<int> ranking(order.size());
			for (size_t index = 0; index < ranking.size(); ++index) ranking[index] = static_cast<int>(index);
			std::stable_sort(ranking.begin(), ranking.end(), [&](int left, int right) {
				return terminal[static_cast<size_t>(left)].second && !terminal[static_cast<size_t>(right)].second; });
			for (size_t index = 0; index < ranking.size(); ++index)
			{
				const auto& job = order[static_cast<size_t>(ranking[index])];
				values[static_cast<size_t>(job.first)].push_back(batch[ranking[index]]);
			}
			done = target;
			std::vector<int> ranked = alive;
			std::stable_sort(ranked.begin(), ranked.end(), [&](int left, int right) {
				return Mean(values[static_cast<size_t>(left)]) > Mean(values[static_cast<size_t>(right)]); });
			if (done >= parameters.samples) break;
			std::vector<int> keep(ranked.begin(), ranked.begin() + std::max(2, (static_cast<int>(ranked.size()) + 1) / 2));
			if (std::find(keep.begin(), keep.end(), result.policyIndex) == keep.end()) keep.push_back(result.policyIndex);
			std::sort(keep.begin(), keep.end());
			alive = keep;
			target = done * 2;
		}
		int rollouts = 0;
		for (const auto& entry : values) rollouts += static_cast<int>(entry.size());
		result.halvingRollouts += rollouts;
		result.halvingSelectedLanes += selectedLanes;
		result.totals.assign(result.candidates.size(), -std::numeric_limits<double>::infinity());
		std::vector<int> counts(result.candidates.size(), 0);
		for (size_t index = 0; index < values.size(); ++index)
		{
			if (values[index].empty()) continue;
			result.totals[index] = Mean(values[index]);
			counts[index] = static_cast<int>(values[index].size());
		}
		int best = alive.front();
		for (int index : alive) if (result.totals[static_cast<size_t>(index)] > result.totals[static_cast<size_t>(best)]) best = index;
		VerifiedChoice(app, rollout, leafValue, staticValue, snapshots, rootId, decisionSeed, income, parameters,
			history, historyStart, studentHistory, untilTick, policy, best, counts, false, tick);
	}

	void FullChoice(LawnApp* app, EnvironmentRollout& rollout, EnvironmentValueNet& leafValue,
		EnvironmentValueNet& staticValue, std::unordered_map<int, EnvironmentSnapshot>& snapshots, int rootId,
		int64_t decisionSeed, double income,
		const SearchParameters& parameters, const std::vector<std::pair<int, uint32_t>>& history, int historyStart,
		const EnvironmentRollout::History& studentHistory, int untilTick, const SearchCandidate& policy, int tick)
	{
		std::vector<int> counts(result.candidates.size(), parameters.samples);
		result.totals.assign(result.candidates.size(), 0.0);
		for (size_t index = 0; index < result.candidates.size(); ++index)
		{
			std::vector<std::pair<int, bool>> terminal;
			std::vector<double> batch;
			std::vector<std::pair<int, int>> order;
			for (int future = 0; future < parameters.samples; ++future) order.emplace_back(static_cast<int>(index), future);
			AppendBatch(app, rollout, leafValue, snapshots, rootId, decisionSeed, income, parameters, history,
				historyStart, studentHistory, untilTick, order, terminal, batch);
			// Terminal futures first, then the value-net leaves: the original accumulation order.
			std::vector<double> ordered;
			for (size_t job = 0; job < batch.size(); ++job) if (terminal[job].second) ordered.push_back(batch[job]);
			for (size_t job = 0; job < batch.size(); ++job) if (!terminal[job].second) ordered.push_back(batch[job]);
			double total = 0;
			for (double value : ordered) total += value;
			result.totals[index] = total / static_cast<double>(ordered.size());
		}
		int best = 0;
		for (size_t index = 1; index < result.totals.size(); ++index)
			if (result.totals[index] > result.totals[static_cast<size_t>(best)]) best = static_cast<int>(index);
		VerifiedChoice(app, rollout, leafValue, staticValue, snapshots, rootId, decisionSeed, income, parameters,
			history, historyStart, studentHistory, untilTick, policy, best, counts, false, tick);
	}

	void VerifiedChoice(LawnApp* app, EnvironmentRollout& rollout, EnvironmentValueNet& leafValue,
		EnvironmentValueNet& staticValue, std::unordered_map<int, EnvironmentSnapshot>& snapshots, int rootId,
		int64_t decisionSeed, double income,
		const SearchParameters& parameters, const std::vector<std::pair<int, uint32_t>>& history, int historyStart,
		const EnvironmentRollout::History& studentHistory, int untilTick, const SearchCandidate& policy,
		int best, std::vector<int> counts, bool fresh, int tick)
	{
		const int policyIndex = result.policyIndex;
		const long long beforeVerificationRollouts = profile.rollouts;
		if (!fresh && parameters.saveCandidates > 0 &&
			SavePlan(app, rollout, leafValue, staticValue, snapshots, rootId, decisionSeed, income, parameters,
				history, historyStart, studentHistory, best, tick))
			return;
		bool before = fresh || (best != policyIndex &&
			result.totals[static_cast<size_t>(best)] - result.totals[static_cast<size_t>(policyIndex)] >
				SearchMargin(parameters, result.totals[static_cast<size_t>(best)]));
		const int samples = parameters.verifySamples + (fresh ? parameters.samples : 0);
		if (samples && before)
		{
			const double beforePolicy = result.totals[static_cast<size_t>(policyIndex)];
			const double beforeBest = result.totals[static_cast<size_t>(best)];
			const int originalPolicy = counts[static_cast<size_t>(policyIndex)];
			const int originalBest = counts[static_cast<size_t>(best)];
			const auto started = std::chrono::steady_clock::now();
			for (int index : { policyIndex, best })
			{
				std::vector<std::pair<int, bool>> terminal;
				std::vector<double> batch;
				std::vector<std::pair<int, int>> order;
				for (int future = parameters.samples; future < parameters.samples + samples; ++future)
					order.emplace_back(index, future);
				AppendBatch(app, rollout, leafValue, snapshots, rootId, decisionSeed, income, parameters, history,
					historyStart, studentHistory, untilTick, order, terminal, batch);
				std::vector<double> ordered;
				for (size_t job = 0; job < batch.size(); ++job) if (terminal[job].second) ordered.push_back(batch[job]);
				for (size_t job = 0; job < batch.size(); ++job) if (!terminal[job].second) ordered.push_back(batch[job]);
				double added = 0;
				for (double value : ordered) added += value;
				result.totals[static_cast<size_t>(index)] =
					(result.totals[static_cast<size_t>(index)] * counts[static_cast<size_t>(index)] + added) /
					(counts[static_cast<size_t>(index)] + samples);
				counts[static_cast<size_t>(index)] += samples;
			}
			// The original value batches and terminal-first sum order, so the last bits match too.
			const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
			result.verifyRollouts += static_cast<int>(profile.rollouts - beforeVerificationRollouts);
			std::ostringstream verification;
			verification << std::setprecision(17) << "{\"samples\":" << samples << ",\"student\":" << CandidateJson(policy)
				<< ",\"challenger\":" << CandidateJson(result.candidates[static_cast<size_t>(best)])
				<< ",\"before_values\":[" << beforePolicy << ',' << beforeBest << "],\"after_values\":["
				<< result.totals[static_cast<size_t>(policyIndex)] << ',' << result.totals[static_cast<size_t>(best)]
				<< "],\"before_override\":true,\"after_override\":"
				<< ((result.totals[static_cast<size_t>(best)] - result.totals[static_cast<size_t>(policyIndex)] >
					SearchMargin(parameters, result.totals[static_cast<size_t>(best)])) ? "true" : "false")
				<< ",\"original_samples\":[" << originalPolicy << ',' << originalBest << "],\"decision_seed\":" << decisionSeed
				<< ",\"mode\":\"" << (fresh ? "fresh" : "reproduced") << "\",\"seconds\":" << seconds << '}';
			result.verification = verification.str();
		}
		if (result.studentCandidate[static_cast<size_t>(best)])
		{
			result.studentCandidateBest++;
			if (best != policyIndex &&
				result.totals[static_cast<size_t>(best)] - result.totals[static_cast<size_t>(policyIndex)] >
					SearchMargin(parameters, result.totals[static_cast<size_t>(best)]))
				result.studentCandidateOverride++;
		}
		result.best = best;
		if (best != policyIndex &&
			result.totals[static_cast<size_t>(best)] - result.totals[static_cast<size_t>(policyIndex)] >
				SearchMargin(parameters, result.totals[static_cast<size_t>(best)]))
		{
			result.overrides++;
			result.overridden = true;
			result.chosen = result.candidates[static_cast<size_t>(best)];
		}
		else result.chosen = policy;
	}

	// --save-candidates k: what would a card be worth if the teacher saved up for it?  The engine
	// only lists actions the player can pay for, so a card above the sun the policy ever holds is
	// invisible to the candidate set (task120: a 125 card had a legal cell in 0.5% of decisions, a 300
	// card in none).  Each probe waits in the engine until the card is affordable, scores its best
	// cell from there and averages --samples + --verify-samples futures.
	bool SavePlan(LawnApp* app, EnvironmentRollout& rollout, EnvironmentValueNet& leafValue,
		EnvironmentValueNet& staticValue, std::unordered_map<int, EnvironmentSnapshot>& snapshots, int rootId,
		int64_t decisionSeed, double income, const SearchParameters& parameters,
		const std::vector<std::pair<int, uint32_t>>& history, int historyStart,
		const EnvironmentRollout::History& studentHistory, int best, int tick)
	{
		(void)income;
		// The halving rollouts left the engine on the last future; the plan reads the root's sun.
		if (!RestoreEnvironmentSnapshot(app, snapshots.at(rootId))) return false;
		std::vector<int> present;
		for (const auto& candidate : result.candidates)
			if (candidate.kind == 'P') present.push_back(candidate.packet);
		// obs["packets"] is in deck order: the slot is the action's packet, the entry's type is the
		// SeedType (a different numbering).
		std::vector<std::pair<int, int>> reachable;   // (packet, cost)
		for (int slot = 0; slot < app->mBoard->mSeedBank->mNumPackets; ++slot)
		{
			if (app->mBoard->mSeedBank->mSeedPackets[slot].mPacketType == SeedType::SEED_NONE) continue;
			const int cost = PacketCost(app, slot);
			if (std::find(present.begin(), present.end(), slot) != present.end()) continue;
			if (cost <= app->mBoard->mSunMoney || app->mBoard->mSunMoney < parameters.saveShare * cost) continue;
			auto last = state.lastSaveProbe.find(slot);
			if (last != state.lastSaveProbe.end() && tick - last->second < parameters.saveInterval) continue;
			reachable.emplace_back(slot, cost);
		}
		if (std::getenv("PVZ_SEARCH_DEBUG"))
		{
			std::cerr << "save reachable=" << reachable.size() << " sun=" << app->mBoard->mSunMoney
				<< " packets=" << app->mBoard->mSeedBank->mNumPackets << " present=" << present.size();
			for (const auto& entry : reachable) std::cerr << " (" << entry.first << ':' << entry.second << ')';
			std::cerr << std::endl;
		}
		if (reachable.empty()) return false;
		std::stable_sort(reachable.begin(), reachable.end(),
			[](const auto& left, const auto& right) { return left.second < right.second; });
		std::vector<SaveCandidate> plans;
		for (size_t index = 0; index < reachable.size() && static_cast<int>(index) < parameters.saveCandidates; ++index)
		{
			const auto& entry = reachable[index];
			state.lastSaveProbe[entry.first] = tick;
			SaveCandidate plan;
			const bool probed = SaveProbe(app, rollout, leafValue, staticValue, snapshots, rootId, decisionSeed,
				income, parameters, history, historyStart, studentHistory, entry.first, entry.second, tick, plan);
			RestoreEnvironmentSnapshot(app, snapshots.at(rootId));
			if (std::getenv("PVZ_SEARCH_DEBUG"))
				std::cerr << "save probe packet=" << entry.first << " cost=" << entry.second << " ok=" << probed
					<< " seconds=" << plan.value << std::endl;
			if (probed) plans.push_back(plan);
		}
		result.saveProbes += static_cast<int>(plans.size());
		if (plans.empty()) return false;
		double bestFinite = -std::numeric_limits<double>::infinity();
		for (double total : result.totals) if (std::isfinite(total) && total > bestFinite) bestFinite = total;
		if (!std::isfinite(bestFinite)) return false;
		const SaveCandidate* plan = &plans.front();
		for (const auto& candidate : plans) if (candidate.value > plan->value) plan = &candidate;
		if (!(plan->value > bestFinite + SearchMargin(parameters, bestFinite))) return false;
		result.best = best;
		result.saveOverrides++;
		result.saveWaitTicks += plan->ticks;
		result.chosen = SearchCandidate();
		result.chosen.kind = 'S';
		result.chosen.packet = plan->packet;
		result.chosen.col = plan->col;
		result.chosen.row = plan->row;
		result.chosen.ticks = plan->ticks;
		return true;
	}

	bool SaveProbe(LawnApp* app, EnvironmentRollout& rollout, EnvironmentValueNet& leafValue,
		EnvironmentValueNet& staticValue, std::unordered_map<int, EnvironmentSnapshot>& snapshots, int rootId,
		int64_t decisionSeed, double income, const SearchParameters& parameters,
		const std::vector<std::pair<int, uint32_t>>& history, int historyStart,
		const EnvironmentRollout::History& studentHistory, int packet, int cost, int tick, SaveCandidate& plan)
	{
		(void)income;
		const bool restored = RestoreEnvironmentSnapshot(app, snapshots.at(rootId));
		if (std::getenv("PVZ_SEARCH_DEBUG"))
			std::cerr << "probe start restore=" << restored << " tick=" << app->mBoard->mMainCounter
				<< " sun=" << app->mBoard->mSunMoney << " term=" << app->EnvironmentTerminal() << std::endl;
		if (!restored) return false;
		int waited = 0;
		while (app->mBoard->mSunMoney < cost && waited < parameters.saveMaxWait && !app->EnvironmentTerminal())
		{
			const int step = std::min(parameters.saveChunk, parameters.saveMaxWait - waited);
			const int before = app->mBoard->mMainCounter;
			app->EnvironmentWait(step);
			const int advanced = app->mBoard->mMainCounter - before;
			if (std::getenv("PVZ_SEARCH_DEBUG"))
				std::cerr << "probe wait step=" << step << " advanced=" << advanced
					<< " sun=" << app->mBoard->mSunMoney << " term=" << app->EnvironmentTerminal() << std::endl;
			if (advanced <= 0) break;
			waited += advanced;
		}
		if (app->EnvironmentTerminal() || app->mBoard->mSunMoney < cost) return false;
		std::vector<std::array<int, 3>> legal;
		LegalCells(app, legal);
		std::vector<SearchCandidate> cells;
		for (const auto& action : legal)
		{
			if (action[0] != packet) continue;
			SearchCandidate action2;
			action2.kind = 'P';
			action2.packet = action[0];
			action2.col = action[1];
			action2.row = action[2];
			cells.push_back(action2);
		}
		if (std::getenv("PVZ_SEARCH_DEBUG"))
			std::cerr << "probe cells packet=" << packet << " n=" << cells.size() << " waited=" << waited
				<< " sun=" << app->mBoard->mSunMoney << " legal=" << legal.size() << std::endl;
		if (cells.empty()) return false;
		const int untilTick = static_cast<int>(app->mBoard->mMainCounter) + parameters.horizon;
		const int futureTick = app->mBoard->mMainCounter;
		const double futureIncome = IncomeAt(history, historyStart, futureTick);
		const int futureId = nextSnapshotId++;
		if (!SaveEnvironmentSnapshot(app, snapshots[futureId])) { snapshots.erase(futureId); return false; }
		std::vector<std::pair<double, SearchCandidate>> scored;
		for (const auto& cell : cells)
		{
			double score = 0;
			if (ScoreStatic(app, staticValue, futureId, cell.packet, cell.col, cell.row, futureIncome, snapshots, score))
				scored.emplace_back(score, cell);
		}
		if (std::getenv("PVZ_SEARCH_DEBUG"))
			std::cerr << "probe scored=" << scored.size() << " of " << cells.size() << std::endl;
		if (scored.empty()) { snapshots.erase(futureId); return false; }
		std::stable_sort(scored.begin(), scored.end(),
			[](const auto& left, const auto& right) { return left.first > right.first; });
		const SearchCandidate target = scored.front().second;
		// The normal candidates are decided after --samples + --verify-samples futures; a plan with
		// fewer would win against them on noise alone.
		const int samples = std::max(1, parameters.samples + std::max(0, parameters.verifySamples));
		double total = 0;
		bool ok = true;
		for (int future = 0; future < samples && ok; ++future)
		{
			const uint32_t seed = static_cast<uint32_t>((decisionSeed * 1000 + future) & 0xffffffff);
			RolloutOutcome outcome = Continue(app, rollout, leafValue, snapshots, futureId,
				IncomeAt(history, historyStart, futureTick), parameters, seed, untilTick, history, historyStart,
				studentHistory, target);
			if (std::getenv("PVZ_SEARCH_DEBUG"))
				std::cerr << "probe future=" << future << " ok=" << outcome.ok << " value=" << outcome.value
					<< " terminal=" << outcome.terminal << std::endl;
			if (!outcome.ok) ok = false;
			else total += outcome.value;
		}
		snapshots.erase(futureId);
		if (!ok) return false;
		plan.ok = true;
		plan.packet = packet;
		plan.cost = cost;
		plan.ticks = waited;
		plan.col = target.col;
		plan.row = target.row;
		plan.samples = samples;
		plan.value = total / samples;
		plan.sun = app->mBoard->mSunMoney;
		return true;
	}

	// The upgrade goes in as soon as the base cell offers it; the plan is dropped when the base is
	// gone or the wait budget is spent, so a dead base never blocks the teacher.
	void ForcedUpgrade(LawnApp* app, const SearchParameters& parameters, int tick)
	{
		if (!state.pendingUpgrade) return;
		bool planted = false;
		for (const Plant* plant : app->mBoard->mPlants)
			if (!plant->mDead && !plant->mSquished && plant->mRow == state.pendingRow &&
				plant->mPlantCol == state.pendingCol && static_cast<int>(plant->mSeedType) == state.pendingBase)
			{
				planted = true;
				break;
			}
		if (!planted || (state.pendingExpires >= 0 && tick > state.pendingExpires))
		{
			state.pendingUpgrade = 0;
			return;
		}
		std::vector<std::array<int, 3>> legal;
		LegalCells(app, legal);
		if (!UpgradeLegal(legal, state.pendingUpgrade, state.pendingCol, state.pendingRow)) return;
		// Leave --combo-buffer sun behind: a hard bank deadlocked poor night levels (task214d).
		if (app->mBoard->mSunMoney - PacketCost(app, state.pendingUpgrade) < parameters.comboBuffer) return;
		SearchCandidate upgrade;
		upgrade.kind = 'P';
		upgrade.packet = state.pendingUpgrade;
		upgrade.col = state.pendingCol;
		upgrade.row = state.pendingRow;
		result.chosen = upgrade;
		++result.comboUpgrades;
		state.pendingUpgrade = 0;
	}

	static int PacketCost(LawnApp* app, int packet)
	{
		if (packet < 0 || packet >= app->mBoard->mSeedBank->mNumPackets) return 0;
		SeedPacket& card = app->mBoard->mSeedBank->mSeedPackets[packet];
		return card.mPacketType == SeedType::SEED_NONE ? 0
			: Plant::GetCost(card.mPacketType, card.mImitaterType);
	}

	static double Mean(const std::vector<double>& values)
	{
		double total = 0;
		for (double value : values) total += value;
		return total / static_cast<double>(values.size());
	}
};

static void RunEnvironment(LawnApp* app)
{
	std::string line;
	std::unordered_map<int, EnvironmentSnapshot> snapshots;
	int nextSnapshotId = 1;
	SearchState searchState;
	EnvironmentRollout rollout;
	EnvironmentValueNet valueNets[2]; // leaf, static
	HumanInput human;
	bool demo = false;
	const bool visible = !app->mHeadlessMode;
	double actionSeconds = 0, observationSeconds = 0, controlSeconds = 0;
	std::cout << "PVZENV {\"ready\":true,\"protocol_version\":" << kEnvironmentProtocolVersion
		<< ",\"wait_events_version\":" << EnvironmentWaitEvents::Version
		<< ",\"rollout_version\":6,\"sparse_tracks_version\":1,\"search_version\":1}" << std::endl;
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
		else if (command == "ROLLOUT_PREDICT")
		{
			int priorTick = -1, priorPlant = 0;
			double income = 0;
			std::string tracePath;
			EnvironmentRollout::History studentHistory;
			ok = static_cast<bool>(input >> priorTick >> priorPlant >> income) &&
				rollout.Loaded() && app->mBoard && std::isfinite(income) && (priorPlant == 0 || priorPlant == 1);
			ok = ok && ReadStudentOptions(input, app->mBoard->mMainCounter, studentHistory, tracePath);
			std::ofstream trace;
			if (ok && !tracePath.empty()) { trace.open(tracePath); ok = trace.good(); }
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":" << (ok ? "true" : "false");
			if (ok)
			{
				if (trace.is_open()) TraceStudent(trace, rollout, app, income, priorTick, priorPlant, studentHistory);
				auto prediction = rollout.Predict(app, income, priorTick, priorPlant, studentHistory, true);
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
			std::string firstCommand, firstArguments;
			ok = ok && ReadStudentOptions(input, app->mBoard->mMainCounter, studentHistory, tracePath) && historyStart >= 0 && historyStart <= app->mBoard->mMainCounter &&
				(first == "A" || DecodeBranchAction(first, firstCommand, firstArguments));
			RolloutRequest request;
			request.rootId = rootId;
			request.seed = seed;
			request.untilTick = untilTick;
			request.mask = mask;
			request.maxActions = maxActions;
			request.eventWait = eventWait != 0;
			request.scoreValue = scoreValue;
			request.first = first;
			request.historyStart = historyStart;
			request.history.assign(history.begin(), history.end());
			request.studentHistory = &studentHistory;
			request.tracePath = &tracePath;
			RolloutOutcome outcome;
			if (ok) outcome = RunRollout(app, rollout, valueNets[0], request, snapshots);
			ok = ok && outcome.ok;
			actionSeconds += outcome.simulationSeconds;
			if (scoreValue)
			{
				std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":" << (ok ? "true" : "false")
					<< ",\"observation\":{\"tick\":" << (app->mBoard ? app->mBoard->mMainCounter : 0)
					<< ",\"wave_timer\":0,\"terminal\":" << (app->EnvironmentTerminal() ? "true" : "false")
					<< ",\"result\":" << static_cast<int>(app->mBoardResult) << ",\"value\":" << std::setprecision(17) << outcome.value;
				if (outcome.estimatedResult >= 0) std::cout << ",\"rollout_result\":" << outcome.estimatedResult;
				std::cout << "},\"decisions\":" << outcome.decisions << ",\"simulation_seconds\":" << outcome.simulationSeconds
					<< ",\"network_seconds\":" << outcome.networkSeconds << "}" << std::endl << std::setprecision(6);
				continue;
			}
			const auto observationStart = std::chrono::steady_clock::now();
			const std::string observation = app->mBoard ? app->EnvironmentLeanObservation() : "null";
			observationSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - observationStart).count();
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":" << (ok ? "true" : "false")
				<< ",\"observation\":" << observation
				<< ",\"estimated_result\":" << outcome.estimatedResult
				<< ",\"decisions\":" << outcome.decisions << ",\"mowers_left\":" << outcome.mowersLeft << ",\"planted_slots\":[";
			for (int index = 0; index < 10; ++index) std::cout << (index ? "," : "") << outcome.planted[index];
			std::cout << "],\"planted_types\":{";
			bool firstType = true;
			for (size_t index = 0; index < outcome.plantedByType.size(); ++index)
				if (outcome.plantedByType[index]) { std::cout << (firstType ? "" : ",") << '\"' << index << "\":" << outcome.plantedByType[index]; firstType = false; }
			std::cout << "},\"simulation_seconds\":" << outcome.simulationSeconds << ",\"network_seconds\":" << outcome.networkSeconds
				<< ",\"net_calls\":" << rollout.NetworkCalls() << ",\"ticks\":" << (app->mBoard ? app->mBoard->mMainCounter - outcome.startTick : 0) << ",\"substeps\":[";
			for (size_t index = 0; index < outcome.substeps.size(); ++index)
				std::cout << (index ? ",[" : "[") << outcome.substeps[index].first << ',' << outcome.substeps[index].second << ']';
			std::cout << "]}" << std::endl;
			continue;
		}
		else if (command == "RANK_PROBE")
		{
			// Student preference order on the current board, for the migration comparison against
			// pvz_rollout_policy.ranked_plants (task217).  No state change.
			double income = 0;
			int previousTick = -1, previousPlant = 0;
			std::string tracePath;
			EnvironmentRollout::History studentHistory;
			ok = static_cast<bool>(input >> income >> previousTick >> previousPlant) && rollout.Loaded() && app->mBoard &&
				std::isfinite(income) && previousTick >= -1 && (previousPlant == 0 || previousPlant == 1) &&
				ReadStudentOptions(input, app->mBoard->mMainCounter, studentHistory, tracePath);
			std::vector<std::array<int, 3>> legal;
			if (ok) app->EnvironmentLegalPlants(legal);
			std::vector<EnvironmentRollout::Action> actions;
			for (const auto& action : legal) actions.push_back({ action[0], action[1], action[2] });
			std::vector<EnvironmentRollout::Ranked> ranked;
			double seconds = 0;
			if (ok)
			{
				const auto started = std::chrono::steady_clock::now();
				rollout.Rank(app, income, previousTick, previousPlant != 0, studentHistory, actions, ranked);
				seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
			}
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":"
				<< (ok ? "true" : "false") << ",\"seconds\":" << seconds << ",\"ranked\":[";
			for (size_t index = 0; index < ranked.size(); ++index)
			{
				if (index) std::cout << ',';
				std::cout << '[' << ranked[index].action.packet << ',' << ranked[index].action.row << ','
					<< ranked[index].action.col << ',' << std::setprecision(17) << ranked[index].score << ']';
			}
			std::cout << "]}" << std::endl << std::setprecision(6);
			continue;
		}
		else if (command == "SEARCH")
		{
			// One whole teacher decision (task217): the candidate set, the halving rounds, the
			// verification futures and the override rule all run here, so the driver no longer needs a
			// round trip per candidate.  The caller passes the decision root's snapshot, its sun-income
			// history, the student history and the student's own action; the algorithm is
			// MEMORY/docs/task217-csearch.md.
			int rootId = 0, mask = 0, horizon = 0, perCard = 0, studentCandidates = 0, samples = 0;
			int verifySamples = 0, halving = 0, halvingFirstSamples = 0, minGap = 0, lastSearchTick = 0;
			int historyStart = 0, historyCount = 0, previousTick = -1, previousPlant = 0;
			int comboCandidates = 0, comboMaxWait = 0, comboBuffer = 25, comboInterval = 600;
			int saveCandidates = 0, saveChunk = 0, saveMaxWait = 0;
			int saveInterval = 0, candidateMode = 0, studentTopk = 3;
			double income = 0, marginRelative = 0, saveShare = 0;
			int64_t decisionSeed = 0;
			std::string policyToken, demoTarget, tracePath;
			EnvironmentRollout::History studentHistory;
			std::deque<std::pair<int, uint32_t>> history;
			ok = static_cast<bool>(input >> rootId >> decisionSeed >> income >> previousTick >> previousPlant
				>> mask >> horizon >> perCard
				>> studentCandidates >> samples >> verifySamples >> halving >> halvingFirstSamples >> marginRelative
				>> minGap >> lastSearchTick >> comboCandidates >> comboMaxWait >> comboBuffer >> comboInterval
				>> saveCandidates
				>> saveShare >> saveInterval >> saveChunk >> saveMaxWait >> candidateMode >> studentTopk
				>> policyToken >> demoTarget >> historyStart >> historyCount) &&
				rollout.Loaded() && valueNets[0].Loaded() && valueNets[1].Loaded() && app->mBoard &&
				snapshots.contains(rootId) && decisionSeed >= 0 && std::isfinite(income) && mask >= 0 &&
				!(mask & ~ENV_SKIP_ALLOWED) && horizon >= 0 && horizon <= 1000000 && perCard >= 1 && perCard <= 32 &&
				studentCandidates >= 0 && studentCandidates <= 64 && samples >= 1 && samples <= 64 &&
				verifySamples >= 0 && verifySamples <= 64 && (halving == 0 || halving == 1) && halvingFirstSamples >= 1 &&
				marginRelative >= 0 && std::isfinite(marginRelative) && minGap >= 0 && previousTick >= -1 &&
				(previousPlant == 0 || previousPlant == 1) &&
				historyStart >= 0 && historyCount >= 0 && historyCount <= 100000 &&
				comboCandidates >= 0 && comboCandidates <= 64 && comboMaxWait >= 0 && comboBuffer >= 0 &&
				comboInterval >= 0 &&
				saveCandidates >= 0 && saveCandidates <= 64 && saveShare >= 0 && std::isfinite(saveShare) &&
				saveInterval >= 0 && saveChunk > 0 && saveMaxWait > 0 &&
				(candidateMode == 0 || candidateMode == 1) && studentTopk >= 1 && studentTopk <= 64;
			if (ok) ok = RestoreEnvironmentSnapshot(app, snapshots.at(rootId));
			if (ok) before = ReadCounters(app);
			for (int index = 0; ok && index < historyCount; ++index)
			{
				int tick = 0;
				uint32_t produced = 0;
				ok = static_cast<bool>(input >> tick >> produced) && tick >= historyStart &&
					tick <= app->mBoard->mMainCounter && (history.empty() || tick >= history.back().first);
				if (ok) history.emplace_back(tick, produced);
			}
			ok = ok && ReadStudentOptions(input, app->mBoard->mMainCounter, studentHistory, tracePath) &&
				historyStart <= app->mBoard->mMainCounter;
			SearchCandidate policy;
			SearchCandidate demo;
			bool hasDemo = false;
			ok = ok && DecodeSearchCandidate(policyToken, policy);
			if (ok && demoTarget != "-")
			{
				hasDemo = DecodeSearchCandidate(demoTarget, demo) && demo.kind == 'P';
				ok = hasDemo;
			}
			SearchParameters parameters;
			parameters.horizon = horizon;
			parameters.perCard = perCard;
			parameters.studentCandidates = studentCandidates;
			parameters.samples = samples;
			parameters.verifySamples = verifySamples;
			parameters.halvingFirstSamples = halvingFirstSamples;
			parameters.minGap = minGap;
			parameters.lastSearchTick = lastSearchTick;
			parameters.comboBuffer = comboBuffer;
			parameters.comboInterval = comboInterval;
			parameters.simSkip = mask;
			parameters.marginRelative = marginRelative;
			parameters.halving = halving != 0;
			parameters.comboCandidates = comboCandidates;
			parameters.comboMaxWait = comboMaxWait;
			parameters.saveCandidates = saveCandidates;
			parameters.saveChunk = saveChunk;
			parameters.saveMaxWait = saveMaxWait;
			parameters.saveInterval = saveInterval;
			parameters.candidateMode = candidateMode;
			parameters.studentTopk = studentTopk;
			parameters.saveShare = saveShare;
			EnvironmentSearch search(searchState, nextSnapshotId);
			const int decisionTick = app->mBoard ? static_cast<int>(app->mBoard->mMainCounter) : 0;
			if (ok)
				search.Run(app, rollout, valueNets[0], valueNets[1], snapshots, rootId, decisionSeed, income,
					previousTick, previousPlant != 0, parameters,
					std::vector<std::pair<int, uint32_t>>(history.begin(), history.end()), historyStart,
					studentHistory, policy, hasDemo ? &demo : nullptr);
			if (ok) search.FinishDecision(app, parameters, decisionTick);
			const SearchResult& result = search.result;
			ok = ok && result.chosen.kind != 0;
			std::cout << "PVZENV {\"protocol_version\":" << kEnvironmentProtocolVersion << ",\"ok\":"
				<< (ok ? "true" : "false") << ",\"searched\":" << (result.searched ? "true" : "false")
				<< ",\"action\":" << CandidateJson(result.chosen) << ",\"best\":" << result.best
				<< ",\"policy_index\":" << result.policyIndex << ",\"overridden\":"
				<< (result.overridden ? "true" : "false") << ",\"stats\":{\"overrides\":" << result.overrides
				<< ",\"halving_candidates\":" << result.halvingCandidates << ",\"halving_rollouts\":"
				<< result.halvingRollouts << ",\"halving_selected_lanes\":" << result.halvingSelectedLanes
				<< ",\"rollout_batches\":" << result.rolloutBatches << ",\"rollout_lanes\":" << result.rolloutLanes
				<< ",\"student_candidate_added\":" << result.studentCandidateAdded << ",\"student_candidate_best\":"
				<< result.studentCandidateBest << ",\"student_candidate_override\":" << result.studentCandidateOverride
				<< ",\"demo_candidate_added\":" << result.demoCandidateAdded
				<< ",\"combo_candidate_added\":" << result.comboCandidateAdded
				<< ",\"combo_upgrades\":" << result.comboUpgrades
				<< ",\"combo_held\":" << result.comboHeld
				<< ",\"save_probes\":" << result.saveProbes
				<< ",\"save_overrides\":" << result.saveOverrides
				<< ",\"save_wait_ticks\":" << result.saveWaitTicks
				<< ",\"last_combo_tick\":" << searchState.lastComboTick << ",\"save_last_probe\":{";
			for (auto entry = searchState.lastSaveProbe.begin(); entry != searchState.lastSaveProbe.end(); ++entry)
			{
				if (entry != searchState.lastSaveProbe.begin()) std::cout << ',';
				std::cout << '"' << entry->first << "\":" << entry->second;
			}
			std::cout << "}";
			if (std::getenv("PVZ_SEARCH_PROFILE"))
			{
				const SearchProfile& p = search.profile;
				const double other = p.totalSeconds - p.staticSeconds - p.restoreSeconds - p.legalSeconds -
					p.simulationSeconds - p.networkSeconds;
				std::cout << ",\"profile\":{\"total_seconds\":" << p.totalSeconds
					<< ",\"static_seconds\":" << p.staticSeconds << ",\"restore_seconds\":" << p.restoreSeconds
					<< ",\"simulation_seconds\":" << p.simulationSeconds
					<< ",\"student_seconds\":" << p.networkSeconds
					<< ",\"other_seconds\":" << other << ",\"ticks_simulated\":" << p.ticksSimulated
					<< ",\"student_forwards\":" << p.studentForwards << ",\"leaf_forwards\":" << p.leafForwards
					<< ",\"rollouts\":" << p.rollouts << ",\"verify_rollouts\":" << result.verifyRollouts
					<< ",\"static_evaluations\":" << p.staticEvaluations << ",\"restores\":" << p.restores
					<< ",\"legal_seconds\":" << p.legalSeconds << ",\"legal_calls\":" << p.legalCalls << '}';
			}
			std::cout << ",\"combo_pending\":";
			if (searchState.pendingUpgrade)
				std::cout << "{\"packet\":" << searchState.pendingUpgrade << ",\"base_type\":"
					<< searchState.pendingBase << ",\"col\":" << searchState.pendingCol << ",\"row\":"
					<< searchState.pendingRow << ",\"expires\":" << searchState.pendingExpires << '}';
			else std::cout << "null";
			if (result.overridden)
				std::cout << ",\"override_kind\":\"" << (result.chosen.kind == 'P' ? "plant" : "wait") << "_over_"
					<< (policy.kind == 'P' ? "plant" : "wait") << '"';
			std::cout << "},\"targets\":[" << std::setprecision(17);
			for (size_t index = 0; index < result.candidates.size(); ++index)
			{
				if (index) std::cout << ',';
				const SearchCandidate& candidate = result.candidates[index];
				std::cout << '[' << (candidate.kind == 'P' ? candidate.packet : -1) << ','
					<< (candidate.kind == 'P' ? candidate.row : -1) << ','
					<< (candidate.kind == 'P' ? candidate.col : -1) << ',';
				if (std::isfinite(result.totals[index])) std::cout << result.totals[index];
				else std::cout << "null";
				std::cout << ',' << candidate.upgradePacket << ',' << candidate.baseType << ']';
			}
			std::cout << "],\"verification\":" << (result.verification.empty() ? "null" : result.verification.c_str())
				<< '}' << std::endl << std::setprecision(6);
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
				searchState.Reset();
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
