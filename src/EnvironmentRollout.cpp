#include "EnvironmentRollout.h"
#include "LawnApp.h"
#include "Lawn/Board.h"
#include "Lawn/Plant.h"
#include "Lawn/Zombie.h"
#include "Lawn/LawnMower.h"
#include "Lawn/SeedPacket.h"
#include "Lawn/Projectile.h"
#include "Lawn/Coin.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

#ifdef PVZ_ROLLOUT_BLAS
#ifdef __APPLE__
#include <vecLib/cblas.h>
#else
#include <cblas.h>
#endif
extern "C" void sgemm_(const char*, const char*, const int*, const int*, const int*, const float*,
	const float*, const int*, const float*, const int*, const float*, float*, const int*);
#endif

#ifndef PVZ_ROLLOUT_BLAS
static void MultiplyRows(const float* matrix, const float* input, float* output, int rows, int count, int stride)
{
	constexpr int cells = EnvironmentRollout::Cells;
#if defined(__aarch64__)
	constexpr int block = 20;
#else
	constexpr int block = 4;
#endif
	int target = 0;
	for (; target + 4 <= rows; target += 4)
		for (int cell = 0; cell < cells; cell += block)
		{
			float first[block], second[block], third[block], fourth[block];
			std::copy_n(output + target * cells + cell, block, first);
			std::copy_n(output + (target + 1) * cells + cell, block, second);
			std::copy_n(output + (target + 2) * cells + cell, block, third);
			std::copy_n(output + (target + 3) * cells + cell, block, fourth);
			for (int index = 0; index < count; ++index)
			{
				const float* source = input + index * cells + cell;
				const float weight0 = matrix[target * stride + index];
				const float weight1 = matrix[(target + 1) * stride + index];
				const float weight2 = matrix[(target + 2) * stride + index];
				const float weight3 = matrix[(target + 3) * stride + index];
				for (int offset = 0; offset < block; ++offset)
				{
					first[offset] += weight0 * source[offset];
					second[offset] += weight1 * source[offset];
					third[offset] += weight2 * source[offset];
					fourth[offset] += weight3 * source[offset];
				}
			}
			std::copy_n(first, block, output + target * cells + cell);
			std::copy_n(second, block, output + (target + 1) * cells + cell);
			std::copy_n(third, block, output + (target + 2) * cells + cell);
			std::copy_n(fourth, block, output + (target + 3) * cells + cell);
		}
	for (; target < rows; ++target)
		for (int index = 0; index < count; ++index)
		{
			const float factor = matrix[target * stride + index];
			for (int cell = 0; cell < cells; ++cell)
				output[target * cells + cell] += factor * input[index * cells + cell];
		}
}
#endif

void EnvironmentRollout::Load(const std::string& path)
{
	std::ifstream input(path, std::ios::binary);
	char magic[8]{};
	uint32_t size = 0, pool = 0, withContext = 0, observation = 1, withRowPool = 0;
	uint32_t packetHead = 0, embedding = 0;
	input.read(magic, 8);
	input.read(reinterpret_cast<char*>(&size), 4);
	input.read(reinterpret_cast<char*>(&pool), 4);
	const std::string version(magic, 8);
	const bool v6 = version == "PVZRN006";
	if (version == "PVZRN003" || version == "PVZRN004" || version == "PVZRN005" || v6) input.read(reinterpret_cast<char*>(&withContext), 4);
	if (version == "PVZRN004" || version == "PVZRN005" || v6) input.read(reinterpret_cast<char*>(&observation), 4);
	if (version == "PVZRN005" || v6) input.read(reinterpret_cast<char*>(&withRowPool), 4);
	if (v6)
	{
		input.read(reinterpret_cast<char*>(&packetHead), 4);
		input.read(reinterpret_cast<char*>(&embedding), 4);
	}
	else embedding = size;
	if (!input || (version != "PVZRN002" && version != "PVZRN003" && version != "PVZRN004" && version != "PVZRN005" && !v6) || std::endian::native != std::endian::little || size < 1 || size > 512 || pool > 1 || withContext > 1 || observation < 1 || observation > 4 || withRowPool > 1 || packetHead > 1 || embedding < 1 || embedding > 512 || (!packetHead && embedding != size))
		throw std::runtime_error("invalid rollout network header");
	const size_t nextWidth = size;
	const size_t nextContext = (pool ? 3 : 2) * nextWidth;
	std::vector<size_t> counts = {
		nextWidth * (Channels + (observation == 4 ? 10 : observation == 3 ? 9 : observation == 2 ? 4 : 0)) * 9, nextWidth, nextWidth * nextWidth * 9, nextWidth,
		nextWidth * nextWidth * 5, nextWidth, nextWidth * (Scalars + (withContext ? TaskContext : 0) + (observation >= 3 ? 7 : observation == 2)), nextWidth,
		nextContext, 1, packetHead ? nextWidth * (nextContext + embedding + 3) : 10 * nextContext,
		packetHead ? nextWidth : 10, (packetHead ? 54 : 10) * embedding,
		nextWidth * (nextWidth + embedding + nextContext), nextWidth, nextWidth, 1};
	if (withRowPool) { counts.push_back(2 * nextWidth * nextWidth); counts.push_back(nextWidth); }
	if (packetHead) { counts.push_back(nextWidth); counts.push_back(1); }
	std::vector<std::vector<float>> loaded;
	for (size_t count : counts)
	{
		auto& values = loaded.emplace_back(count);
		input.read(reinterpret_cast<char*>(values.data()), count * sizeof(float));
		if (!input || !std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }))
			throw std::runtime_error("invalid rollout network weights");
	}
	if (input.peek() != std::char_traits<char>::eof())
		throw std::runtime_error("unexpected rollout network data");
	width = size;
	identityPacket = packetHead != 0;
	plantEmbedding = embedding;
	meanmax = pool != 0;
	rowPool = withRowPool != 0;
	taskContext = withContext != 0;
	v2lite = observation == 2;
	v2 = observation >= 3;
	fogChannel = observation == 4;
	weights = std::move(loaded);
	// LOAD_ROLLOUT_NET can replace a model in an existing process, including with
	// another model of the same layout. Channel sets and dimensions do not identify
	// weights: invalidate every derived weight/padding cache on a successful load.
	kernelChannels.clear();
	kernelBuffer.clear();
	packetWeightDouble.clear();
	columnsCount = -1;
	packetLogitsValid = false;
	hidden.resize(Cells * width);
	scratch.resize(Cells * width);
	context.resize(nextContext);
}

void EnvironmentRollout::History::Update(LawnApp* app)
{
	const Board* state = app->mBoard;
	if (wave != state->mCurrentWave) { wave = state->mCurrentWave; waveTick = state->mMainCounter; }
	decltype(plants) current;
	for (const Plant* plant : state->mPlants)
		if (!plant->mDead && !plant->mSquished && plant->mRow >= 0 && plant->mRow < 6 && plant->mPlantCol >= 0 && plant->mPlantCol < 9)
		{
			const auto key = std::make_tuple(plant->mRow, plant->mPlantCol, static_cast<int>(plant->mSeedType));
			const auto found = plants.find(key);
			current[key] = found == plants.end() ? state->mMainCounter : found->second;
		}
	plants = std::move(current);
}

void EnvironmentRollout::Features(LawnApp* app, double income, int previousTick, bool previousPlant,
	std::vector<float>& board, std::vector<float>& scalars, History& history)
{
	Board* state = app->mBoard;
	board.assign((Channels + (fogChannel ? 10 : v2 ? 9 : v2lite ? 4 : 0)) * Cells, 0.0f);
	scalars.assign(Scalars + (taskContext ? TaskContext : 0) + (v2 ? 7 : v2lite), 0.0f);
	for (const Plant* plant : state->mPlants)
		if (!plant->mDead && !plant->mSquished && plant->mRow >= 0 && plant->mRow < 6 && plant->mPlantCol >= 0 && plant->mPlantCol < 9)
		{
			// task70 B5: a fog-hidden plant's health is "unknown" (-1) -- the same fixed value the
			// observations carry, so the type channel reads a presence marker, never real internals.
			const double health = app->EnvironmentPlantFogHidden(plant) ? -1.0 :
				static_cast<double>(plant->mPlantHealth) / std::max(1, plant->mPlantMaxHealth);
			board[std::min(static_cast<int>(plant->mSeedType), 53) * Cells + plant->mRow * 10 + plant->mPlantCol] += HasHistory() ? static_cast<float>(health) : health;
		}
	for (Zombie* zombie : state->mZombies)
	{
		if (zombie->mDead || !zombie->IsOnBoard() || zombie->mRow < 0 || zombie->mRow >= 6 || !app->EnvironmentZombieVisible(zombie)) continue;
		double position = zombie->mPosX;
		const double boundary = 40.0 + 80.0 * std::round((position - 40.0) / 80.0);
		if (std::abs(position - boundary) < 0.001)
			position = EnvironmentWaitEvents::RenderedPublicCoordinate(position);
		const int cell = zombie->mRow * 10 + std::clamp(static_cast<int>(std::floor((position - 40.0) / 80.0)), 0, 9);
		const double health = (zombie->mBodyHealth + zombie->mHelmHealth + zombie->mShieldHealth) / 200.0;
		board[(54 + std::min(static_cast<int>(zombie->mZombieType), 33)) * Cells + cell] += HasHistory() ? static_cast<float>(health) : health;
		board[88 * Cells + cell] += zombie->mIsEating;
		board[89 * Cells + cell] += zombie->mChilledCounter > 0;
	}
	for (const LawnMower* mower : state->mLawnMowers)
		if (!mower->mDead && mower->mRow >= 0 && mower->mRow < 6)
			board[90 * Cells + mower->mRow * 10] = mower->mMowerState == LawnMowerState::MOWER_READY;
	for (int row = 0; row < 6; ++row)
	{
		const auto terrain = state->mGridSquareType[0][row];
		bool uniform = true;
		for (int col = 1; col < 9; ++col) uniform &= state->mGridSquareType[col][row] == terrain;
		if (uniform && (terrain == GridSquareType::GRIDSQUARE_GRASS || terrain == GridSquareType::GRIDSQUARE_POOL))
			std::fill_n(board.data() + 91 * Cells + row * 10, 10, 1.0f);
	}
	if (fogChannel)
		for (int row = 0; row < 6; ++row)
			for (int col = 0; col < 10; ++col)
				board[101 * Cells + row * 10 + col] = app->EnvironmentFogAmount(row, std::min(col, 8)) / 255.0f;
	const int waves = std::max(1, state->mNumWaves);
	scalars[0] = std::min(state->mSunMoney, 2000) / 1000.0;
	scalars[1] = static_cast<double>(state->mCurrentWave) / waves;
	scalars[2] = state->mMainCounter / 60000.0;
	scalars[3] = income / 100.0;
	scalars[4] = std::log1p(waves) / std::log1p(50);
	for (auto& info : packetInfo) info = {-1, 0, 0, 0};
	for (int index = 0; index < state->mSeedBank->mNumPackets; ++index)
	{
		const SeedPacket& packet = state->mSeedBank->mSeedPackets[index];
		const int kind = std::min(static_cast<int>(packet.mPacketType), 53);
		const int cost = packet.mPacketType == SeedType::SEED_NONE ? 0 : Plant::GetCost(packet.mPacketType, packet.mImitaterType);
		if (kind < 0) continue;
		// Identity packet inputs follow the original-start engine flag, just like packet_features.
		// Legacy slot students keep their established scalar encoding on every task.
		const bool affordable = state->mSunMoney >= cost ||
			(identityPacket && app->mEnvironmentTaskSpec.originalConveyorStart && state->HasConveyorBeltSeedBank());
		const float ready = packet.mActive && affordable;
		const float progress = std::min(1.0, packet.mActive || !packet.mRefreshTime ? 1.0 :
			static_cast<double>(packet.mRefreshCounter) / packet.mRefreshTime);
		packetInfo[index] = {static_cast<float>(kind), static_cast<float>(packet.mActive), progress,
			static_cast<float>(affordable)};
		scalars[5 + kind] = identityPacket ? std::max(scalars[5 + kind], ready) : ready;
		scalars[59 + kind] = identityPacket ? std::max(scalars[59 + kind], progress) : progress;
	}
	if (previousTick >= 0)
	{
		scalars[113] = previousPlant;
		scalars[114] = (state->mMainCounter - previousTick) / 300.0;
	}
	if (taskContext)
	{
		scalars[Scalars] = app->mEnvironmentTaskSpec.zombieCountMultiplier / 3.0;
		for (int type = 0; type < NUM_ZOMBIE_TYPES && type < 100; ++type)
			if (state->mZombieAllowed[type]) scalars[Scalars + 1 + std::min(type, ZombieTypes - 1)] = 1.0f;
	}
	if (v2lite)
	{
		history.Update(app);
		std::array<double, Cells> ages, health{}, positions{}, speeds{};
		ages.fill(1.0);
		for (const auto& [key, born] : history.plants)
		{
			const auto [row, col, kind] = key;
			const int cell = row * 10 + col;
			ages[cell] = std::min(ages[cell], (state->mMainCounter - born) / 1500.0);
			board[95 * Cells + cell] = ages[cell];
		}
		for (Zombie* zombie : state->mZombies)
		{
			if (zombie->mDead || !zombie->IsOnBoard() || zombie->mRow < 0 || zombie->mRow >= 6 || !app->EnvironmentZombieVisible(zombie)) continue;
			const double position = EnvironmentWaitEvents::RenderedPublicCoordinate(zombie->mPosX);
			const double velocity = EnvironmentWaitEvents::RenderedPublicCoordinate(zombie->mVelX);
			const int col = std::clamp(static_cast<int>(std::floor((position - 40.0) / 80.0)), 0, 9);
			const int cell = zombie->mRow * 10 + col;
			const double amount = zombie->mBodyHealth + zombie->mHelmHealth + zombie->mShieldHealth;
			positions[cell] += amount * std::clamp((position - 40.0 - col * 80.0) / 80.0, 0.0, 1.0);
			speeds[cell] += amount * velocity;
			health[cell] += amount;
			board[94 * Cells + cell] = std::max(static_cast<double>(board[94 * Cells + cell]), velocity);
		}
		for (int cell = 0; cell < Cells; ++cell)
			if (health[cell] > 0)
			{
				board[92 * Cells + cell] = positions[cell] / health[cell];
				board[93 * Cells + cell] = speeds[cell] / health[cell];
			}
		scalars.back() = std::min(1.0, (state->mMainCounter - history.waveTick) / 3000.0);
	}
	if (v2)
	{
		history.Update(app);
		std::array<double, Cells> health{}, positions{}, speeds{}, maximum;
		maximum.fill(-std::numeric_limits<double>::infinity());
		for (Zombie* zombie : state->mZombies)
		{
			if (zombie->mDead || !zombie->IsOnBoard() || zombie->mRow < 0 || zombie->mRow >= 6 || !app->EnvironmentZombieVisible(zombie)) continue;
			const double position = EnvironmentWaitEvents::RenderedPublicCoordinate(zombie->mPosX);
			const double velocity = EnvironmentWaitEvents::RenderedPublicCoordinate(zombie->mVelX);
			const int col = std::clamp(static_cast<int>(std::floor((position - 40.0) / 80.0)), 0, 9);
			const int cell = zombie->mRow * 10 + col;
			const double amount = zombie->mBodyHealth + zombie->mHelmHealth + zombie->mShieldHealth;
			positions[cell] += amount * std::clamp((position - 40.0 - col * 80.0) / 80.0, 0.0, 1.0);
			speeds[cell] += amount * velocity / 0.7;
			health[cell] += amount;
			maximum[cell] = std::max(maximum[cell], velocity / 0.7);
			if (zombie->mZombieType == ZombieType::ZOMBIE_POLEVAULTER && zombie->mHasObject)
				board[95 * Cells + cell] += static_cast<float>(amount / 200.0);
			board[96 * Cells + cell] += static_cast<float>((zombie->mHelmHealth + zombie->mShieldHealth) / 370.0);
			board[97 * Cells + cell] = std::max(static_cast<double>(board[97 * Cells + cell]), zombie->mChilledCounter / 2000.0);
		}
		for (int cell = 0; cell < Cells; ++cell)
		{
			if (health[cell] > 0)
			{
				board[92 * Cells + cell] = positions[cell] / health[cell];
				board[93 * Cells + cell] = speeds[cell] / health[cell];
			}
			if (std::isfinite(maximum[cell])) board[94 * Cells + cell] = maximum[cell];
		}
		for (const Plant* plant : state->mPlants)
		{
			if (plant->mDead || plant->mSquished || plant->mRow < 0 || plant->mRow >= 6 || plant->mPlantCol < 0 || plant->mPlantCol >= 9) continue;
			const int cell = plant->mRow * 10 + plant->mPlantCol;
			// task70 B5: a fog-hidden plant contributes exactly what the JSON decoders compute
			// from the "unknown" (-1) fields the observations carry, so the C++ student input
			// and the python student input stay identical on fog frames.
			const bool aHidden = app->EnvironmentPlantFogHidden(plant);
			const PlantState aState = aHidden ? static_cast<PlantState>(-1) : plant->mState;
			const int aCountdown = aHidden ? -1 : plant->mStateCountdown;
			const int aSpecial = aHidden ? -1 : plant->mDoSpecialCountdown;
			const int aLaunch = aHidden ? -1 : plant->mLaunchCounter;
			double progress = 0;
			if (plant->mSeedType == SeedType::SEED_POTATOMINE)
			{
				if (aState == PlantState::STATE_NOTREADY) progress = 1.0 - aCountdown / 1500.0;
				else if (aState == PlantState::STATE_POTATO_RISING || aState == PlantState::STATE_POTATO_ARMED) progress = 1.0;
			}
			else if (plant->mSeedType == SeedType::SEED_CHERRYBOMB || plant->mSeedType == SeedType::SEED_JALAPENO)
				progress = 1.0 - aSpecial / 100.0;
			board[98 * Cells + cell] = std::max(static_cast<double>(board[98 * Cells + cell]), progress);
			if (plant->mLaunchRate > 0)
				board[99 * Cells + cell] = std::max(static_cast<double>(board[99 * Cells + cell]), 1.0 - static_cast<double>(aLaunch) / plant->mLaunchRate);
		}
		for (Projectile* projectile : state->mProjectiles)
		{
			if (projectile->mDead || projectile->mRow < 0 || projectile->mRow >= 6) continue;
			if (app->EnvironmentFogMasked() && app->EnvironmentFogAmount(projectile->mRow,
				std::clamp(static_cast<int>((projectile->mPosX - 40) / 80), 0, 8)) > 0) continue;
			const double position = EnvironmentWaitEvents::RenderedPublicCoordinate(projectile->mPosX);
			const int col = std::clamp(static_cast<int>(std::floor((position - 40.0) / 80.0)), 0, 9);
			board[100 * Cells + projectile->mRow * 10 + col] += static_cast<float>(projectile->GetProjectileDef().mDamage / 200.0);
		}
		int pending = 0;
		for (Coin* coin : state->mCoins)
			if (!coin->mDead && coin->IsSun()) pending += coin->GetSunValue();
		const int start = Scalars + (taskContext ? TaskContext : 0);
		scalars[start] = state->mZombieCountDown / 4500.0;
		scalars[start + 1] = state->mHugeWaveCountDown > 0;
		scalars[start + 2] = !app->EnvironmentFogMasked() && state->mZombieHealthWaveStart ? static_cast<double>(state->TotalZombiesHealthInWave(state->mCurrentWave - 1)) / state->mZombieHealthWaveStart : 0.0;
		scalars[start + 3] = state->mSunCountDown / 1224.0;
		scalars[start + 4] = pending / 200.0;
		scalars[start + 5] = std::min(1.0, (state->mMainCounter - history.waveTick) / 3000.0);
		scalars[start + 6] = 1.0f;
	}
}

// A float is zero exactly when its bit pattern is zero, so a channel of the observation is
// empty when the bitwise OR of its cells is zero.  Testing 64 bits at a time replaced a
// per-cell value comparison (0.7% of the rollout CPU on 2026-10-23).  Unlike the value test
// this also counts -0.0f as non-zero, which only ever adds an all-zero channel to the
// convolution and therefore cannot change the result.
static bool CellsAreEmpty(const float* theCells)
{
	constexpr int Words = EnvironmentRollout::Cells * static_cast<int>(sizeof(float)) / static_cast<int>(sizeof(uint64_t));
	uint64_t aBits = 0;
	for (int word = 0; word < Words; ++word)
	{
		uint64_t aWord;
		std::memcpy(&aWord, theCells + word * (sizeof(uint64_t) / sizeof(float)), sizeof(aWord));
		aBits |= aWord;
	}
	return aBits == 0;
}

void EnvironmentRollout::Convolve(const std::vector<float>& input, int channels, int kernelRows, int kernelCols,
	int layer, std::vector<float>& output, bool sparse)
{
	const auto& kernel = weights[layer];
	const auto& bias = weights[layer + 1];
	activeChannels.clear();
	for (int channel = 0; channel < channels; ++channel)
		if (!sparse || !CellsAreEmpty(input.data() + channel * Cells)) activeChannels.push_back(channel);
	const int count = static_cast<int>(activeChannels.size()) * kernelRows * kernelCols;
	const size_t needed = static_cast<size_t>(count) * Cells;
	// im2col writes every in-bounds cell of every feature; the padding border must be zero.  It
	// only has to be zeroed when the feature->channel mapping changes, not on every forward.
	if (columns.size() < needed) columns.resize(needed, 0.0f);
	if (count != columnsCount) std::fill_n(columns.data(), needed, 0.0f);
	columnsCount = count;
	// The transposed kernel depends only on which channels are active.
	const bool transpose = sparse && activeChannels != kernelChannels;
	if (transpose) kernelBuffer.resize(static_cast<size_t>(width) * count);
	int feature = 0;
	for (int channel : activeChannels)
		for (int kernelRow = 0; kernelRow < kernelRows; ++kernelRow)
			for (int kernelCol = 0; kernelCol < kernelCols; ++kernelCol, ++feature)
			{
				const int deltaRow = kernelRow - kernelRows / 2, deltaCol = kernelCol - kernelCols / 2;
				// The in-bounds columns of a row are one contiguous run: copy it as a run instead of
				// testing every cell (this im2col was 5.2% of the rollout CPU on 2026-10-23).  The
				// run is the full ten-float row for every interior kernel column, and a variable
				// length memcpy costs a PLT round trip (~9 ns for 40 bytes, 3.1% of the rollout CPU
				// on 2026-10-23); the fixed-size copy is inlined as two vector moves.
				const int colStart = std::max(0, -deltaCol), colEnd = std::min(10, 10 - deltaCol);
				const int rowBegin = std::max(0, -deltaRow), rowEnd = std::min(6, 6 - deltaRow);
				const int run = colEnd - colStart;
				for (int row = rowBegin; row < rowEnd; ++row)
				{
					float* dest = &columns[feature * Cells + row * 10 + colStart];
					const float* source = &input[channel * Cells + (row + deltaRow) * 10 + colStart + deltaCol];
					// A 3x3 kernel only ever leaves 10 or 9 in-bounds columns and a 1x5 kernel 10, 9
					// or 8, so the run length is one of three values: copy it with a constant size so
					// the compiler inlines the copy instead of calling memcpy (~9 ns a call).
					switch (run)
					{
					case 10: std::memcpy(dest, source, 10 * sizeof(float)); break;
					case 9: std::memcpy(dest, source, 9 * sizeof(float)); break;
					case 8: std::memcpy(dest, source, 8 * sizeof(float)); break;
					default: std::memcpy(dest, source, static_cast<size_t>(run) * sizeof(float)); break;
					}
				}
				if (transpose)
					for (int target = 0; target < width; ++target)
						kernelBuffer[target * count + feature] = kernel[((target * channels + channel) * kernelRows + kernelRow) * kernelCols + kernelCol];
			}
	if (transpose) kernelChannels = activeChannels;
	output.resize(width * Cells);
	const float* matrix = sparse ? kernelBuffer.data() : kernel.data();
	for (int target = 0; target < width; ++target)
		std::fill_n(output.data() + target * Cells, Cells, bias[target]);
#ifdef PVZ_ROLLOUT_BLAS
	if (count)
	{
		const char normal = 'N';
		const float one = 1.0f;
		const int cells = Cells;
		sgemm_(&normal, &normal, &cells, &width, &count, &one, columns.data(), &cells, matrix, &count,
			&one, output.data(), &cells);
	}
#else
	MultiplyRows(matrix, columns.data(), output.data(), width, count, count);
#endif
	for (float& value : output) value = std::max(0.0f, value);
}

static float Linear(const float* input, const float* weight, int count, float bias)
{
	for (int index = 0; index < count; ++index) bias += input[index] * weight[index];
	return bias;
}

static float IdentityLinear(const float* input, const float* weight, int count, float bias)
{
	double total = bias;
	for (int index = 0; index < count; ++index) total += static_cast<double>(input[index]) * weight[index];
	return static_cast<float>(total);
}

void EnvironmentRollout::Encode(const std::vector<float>& board, const std::vector<float>& scalars)
{
	Convolve(board, Channels + (fogChannel ? 10 : v2 ? 9 : v2lite ? 4 : 0), 3, 3, 0, hidden, true);
	Convolve(hidden, width, 3, 3, 2, scratch, false);
	Convolve(scratch, width, 1, 5, 4, hidden, false);
	if (rowPool)
	{
		// A buffer of its own: the im2col buffer above keeps its zero border between forwards.
		rowPoolColumns.resize(6 * 2 * width);
		for (int row = 0; row < 6; ++row)
			for (int channel = 0; channel < width; ++channel)
			{
				const float* values = hidden.data() + channel * Cells + row * 10;
				float total = 0;
				for (int col = 0; col < 10; ++col) total += values[col];
				rowPoolColumns[row * 2 * width + channel] = total / 10;
				rowPoolColumns[row * 2 * width + width + channel] = *std::max_element(values, values + 10);
			}
	#ifdef PVZ_ROLLOUT_BLAS
		for (int row = 0; row < 6; ++row)
			std::copy_n(weights[18].data(), width, scratch.data() + row * width);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 6, width, 2 * width, 1.0f,
			rowPoolColumns.data(), 2 * width, weights[17].data(), 2 * width, 1.0f, scratch.data(), width);
	#endif
		for (int channel = 0; channel < width; ++channel)
			for (int row = 0; row < 6; ++row)
			{
			#ifdef PVZ_ROLLOUT_BLAS
				const float projected = scratch[row * width + channel];
			#else
				const float projected = Linear(rowPoolColumns.data() + row * 2 * width,
					weights[17].data() + channel * 2 * width, 2 * width, weights[18][channel]);
			#endif
				for (int col = 0; col < 10; ++col)
				{
					float& value = hidden[channel * Cells + row * 10 + col];
					value = std::max(0.0f, value + projected);
				}
			}
	}
	for (int channel = 0; channel < width; ++channel)
	{
		float total = 0;
		for (int cell = 0; cell < Cells; ++cell) total += hidden[channel * Cells + cell];
		context[channel] = total / Cells;
		if (meanmax)
			context[width + channel] = *std::max_element(hidden.data() + channel * Cells, hidden.data() + (channel + 1) * Cells);
		context[context.size() - width + channel] = std::max(0.0f, Linear(scalars.data(), weights[6].data() + channel * scalars.size(), scalars.size(), weights[7][channel]));
	}
}

EnvironmentRollout::Prediction EnvironmentRollout::Heads(int packet)
{
	Prediction result{};
	const int global = context.size(), stride = width + plantEmbedding + global;
	result.cellPacket = packet;
	const auto linear = identityPacket ? IdentityLinear : Linear;
	result.plant = linear(context.data(), weights[8].data(), global, weights[9][0]);
	if (!identityPacket)
		for (int index = 0; index < 10; ++index)
			result.packets[index] = Linear(context.data(), weights[10].data() + index * global, global, weights[11][index]);
	else if (packetLogitsValid)
		// Predict() asks for the same packet logits twice per decision (Heads(-1) to choose a
		// card, then Heads(card) for the cells); they only depend on the context and packetInfo.
		result.packets = packetLogits;
	else
	{
		result.packets.fill(-1e9f);
		const int inputSize = global + plantEmbedding + 3;
		packetInput.resize(inputSize);
		std::copy_n(context.begin(), global, packetInput.begin());
		packetActivation.resize(width);
		const int last = rowPool ? 19 : 17;
#ifdef PVZ_ROLLOUT_BLAS
		// The packet head is one matrix-matrix product (up to 10 inputs x width), not ten
		// independent dot products: this loop was 16.7% of the rollout CPU as scalar double
		// arithmetic (2026-10-23 profile).  The weights are widened to double once.
		if (packetWeightDouble.empty())
		{
			packetWeightDouble.resize(static_cast<size_t>(width) * inputSize);
			std::copy(weights[10].begin(), weights[10].begin() + static_cast<long>(packetWeightDouble.size()),
				packetWeightDouble.begin());
		}
		int rows = 0;
		std::array<int, 10> rowPacket{};
		packetMatrix.resize(static_cast<size_t>(10) * inputSize);
		for (int index = 0; index < 10; ++index)
		{
			const auto& info = packetInfo[index];
			if (info[0] < 0) continue;
			std::copy_n(weights[12].data() + static_cast<int>(info[0]) * plantEmbedding, plantEmbedding, packetInput.data() + global);
			std::copy_n(info.data() + 1, 3, packetInput.data() + global + plantEmbedding);
			std::copy(packetInput.begin(), packetInput.end(), packetMatrix.begin() + static_cast<long>(rows) * inputSize);
			rowPacket[rows++] = index;
		}
		if (rows)
		{
			packetOutput.resize(static_cast<size_t>(rows) * width);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, width, inputSize, 1.0,
				packetMatrix.data(), inputSize, packetWeightDouble.data(), inputSize, 0.0,
				packetOutput.data(), width);
		}
		for (int row = 0; row < rows; ++row)
		{
			for (int channel = 0; channel < width; ++channel)
				packetActivation[channel] = std::max(0.0f,
					static_cast<float>(packetOutput[static_cast<size_t>(row) * width + channel] + weights[11][channel]));
			result.packets[rowPacket[row]] = IdentityLinear(packetActivation.data(), weights[last].data(), width, weights[last + 1][0]);
		}
#else
		for (int index = 0; index < 10; ++index)
		{
			const auto& info = packetInfo[index];
			if (info[0] < 0) continue;
			std::copy_n(weights[12].data() + static_cast<int>(info[0]) * plantEmbedding, plantEmbedding, packetInput.data() + global);
			std::copy_n(info.data() + 1, 3, packetInput.data() + global + plantEmbedding);
			for (int channel = 0; channel < width; ++channel)
				packetActivation[channel] = std::max(0.0f, IdentityLinear(packetInput.data(), weights[10].data() + channel * inputSize, inputSize, weights[11][channel]));
			result.packets[index] = IdentityLinear(packetActivation.data(), weights[last].data(), width, weights[last + 1][0]);
		}
#endif
		packetLogits = result.packets;
		packetLogitsValid = true;
	}
	if (packet < 0) return result;
	result.cells.fill(weights[16][0]);
	for (int channel = 0; channel < width; ++channel)
	{
		const float* kernel = weights[13].data() + channel * stride;
		const int kind = identityPacket ? std::max(0, static_cast<int>(packetInfo[packet][0])) : packet;
		float base = linear(weights[12].data() + kind * plantEmbedding, kernel + width, plantEmbedding, weights[14][channel]);
		base = linear(context.data(), kernel + width + plantEmbedding, global, base);
		std::fill_n(scratch.data() + channel * Cells, Cells, base);
	}
#ifdef PVZ_ROLLOUT_BLAS
	const char normal = 'N';
	const float one = 1.0f;
	const int cells = Cells;
	sgemm_(&normal, &normal, &cells, &width, &width, &one, hidden.data(), &cells, weights[13].data(), &stride,
		&one, scratch.data(), &cells);
#else
	MultiplyRows(weights[13].data(), hidden.data(), scratch.data(), width, width, stride);
#endif
	if (identityPacket)
	{
		for (int cell = 0; cell < 54; ++cell)
		{
			double total = weights[16][0];
			for (int channel = 0; channel < width; ++channel)
				total += static_cast<double>(std::max(0.0f, scratch[channel * Cells + cell / 9 * 10 + cell % 9])) * weights[15][channel];
			result.cells[cell] = static_cast<float>(total);
		}
	}
	else
		for (int channel = 0; channel < width; ++channel)
			for (int cell = 0; cell < 54; ++cell)
				result.cells[cell] += std::max(0.0f, scratch[channel * Cells + cell / 9 * 10 + cell % 9]) * weights[15][channel];
	return result;
}

EnvironmentRollout::Prediction EnvironmentRollout::Predict(LawnApp* app, double income, int previousTick, bool previousPlant, History& history, bool diagnostics)
{
	Prediction result{};
	if (!diagnostics && !app->EnvironmentAnyLegalPlant())
	{
		if (HasHistory()) history.Update(app);
		return result;
	}
	++networkCalls;
	packetLogitsValid = false;
	Features(app, income, previousTick, previousPlant, featureBoard, featureScalars, history);
	Encode(featureBoard, featureScalars);
	result = Heads(-1);
	if (result.plant <= 0 && !diagnostics) return result;
	float bestPacket = -std::numeric_limits<float>::infinity();
	int bestKind = std::numeric_limits<int>::max();
	std::vector<Action> legal;
	for (int index = 0; index < app->mBoard->mSeedBank->mNumPackets; ++index)
	{
		SeedPacket& packet = app->mBoard->mSeedBank->mSeedPackets[index];
		const int kind = static_cast<int>(packetInfo[index][0]);
		if (!packet.CanPickUp() || result.packets[index] < bestPacket ||
			(result.packets[index] == bestPacket && (!identityPacket || kind >= bestKind))) continue;
		const SeedType seed = packet.mPacketType == SeedType::SEED_IMITATER ? packet.mImitaterType : packet.mPacketType;
		std::vector<Action> cells;
		for (int row = 0; row < 6; ++row)
			for (int col = 0; col < 9; ++col)
				if (app->mBoard->CanPlantAt(col, row, seed) == PlantingReason::PLANTING_OK)
					cells.push_back({index, col, row});
		if (!cells.empty()) { bestPacket = result.packets[index]; bestKind = kind; legal = std::move(cells); }
	}
	if (legal.empty()) return result;
	result = Heads(legal[0].packet);
	if (result.plant > 0)
		result.action = *std::max_element(legal.begin(), legal.end(), [&result](const Action& left, const Action& right) {
			return result.cells[left.row * 9 + left.col] < result.cells[right.row * 9 + right.col]; });
	return result;
}
