#include "EnvironmentValueNet.h"
#include "EnvironmentRollout.h"
#include "LawnApp.h"
#include "Lawn/Board.h"
#include "Lawn/Plant.h"
#include "Lawn/Zombie.h"
#include "Lawn/LawnMower.h"
#include "Lawn/SeedPacket.h"
#include "Lawn/GridItem.h"
#include "Lawn/Projectile.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <stdexcept>

#ifdef __APPLE__
#include <vecLib/cblas.h>
#else
#include <cblas.h>
#endif

namespace
{
float Clamp01(double value)
{
	return static_cast<float>(std::clamp(value, 0.0, 1.0));
}

// Value-net v6 (encoder 6).  Only current state that changes the future is encoded; plant age
// and invested sun are deliberately absent.  A cell can hold a pumpkin and a normal plant, so
// state/timers take the maximum and health takes the sum, like the v5 board cells.
void InternalFeatures(LawnApp* app, float* out)
{
	Board* state = app->mBoard;
	auto cell = [out](int channel, int row, int col) -> float&
	{
		return out[channel * EnvironmentValueNet::Cells + row * 10 + col];
	};
	auto bump = [&cell](int channel, int row, int col, float value)
	{
		float& target = cell(channel, row, col);
		target = std::max(target, value);
	};
	auto add = [&cell](int channel, int row, int col, float value)
	{
		float& target = cell(channel, row, col);
		target = std::min(1.0f, target + value);
	};
	for (const Plant* plant : state->mPlants)
	{
		if (plant->mDead || plant->mSquished || app->EnvironmentStormHidden() ||
			plant->mRow < 0 || plant->mRow >= 6 || plant->mPlantCol < 0 || plant->mPlantCol >= 9) continue;
		const int row = plant->mRow, col = plant->mPlantCol;
		bump(0, row, col, Clamp01(static_cast<double>(plant->mState) / 32.0));
		bump(1, row, col, Clamp01(static_cast<double>(std::max(0, plant->mStateCountdown)) / 4000.0));
		bump(2, row, col, Clamp01(static_cast<double>(std::max(0, plant->mDoSpecialCountdown)) / 100.0));
		const double launch = plant->mLaunchRate > 0
			? 1.0 - static_cast<double>(std::max(0, plant->mLaunchCounter)) / plant->mLaunchRate : 0.0;
		bump(3, row, col, Clamp01(launch));
		bump(4, row, col, Clamp01(static_cast<double>(std::max(0, plant->mLaunchRate)) / 3000.0));
		bump(5, row, col, Clamp01(static_cast<double>(std::max(0, plant->mWakeUpCounter)) / 1200.0));
		bump(6, row, col, plant->mIsAsleep ? 1.0f : 0.0f);
		add(7, row, col, Clamp01(static_cast<double>(std::max(0, plant->mPlantHealth)) / 1000.0));
		bump(8, row, col, plant->mTargetX > 0 ? Clamp01(static_cast<double>(plant->mTargetX) / 800.0) : 0.0f);
		bump(9, row, col, plant->mTargetY > 0 ? Clamp01(static_cast<double>(plant->mTargetY) / 600.0) : 0.0f);
		bump(10, row, col, Clamp01(static_cast<double>(std::max(0, plant->mDisappearCountdown)) / 600.0));
		bump(11, row, col, Clamp01(static_cast<double>(std::max(0, plant->mSubclass)) / 4.0));
	}
	int zombies = 0, plants = 0;
	for (const Plant* plant : state->mPlants)
		if (!plant->mDead && !plant->mSquished) ++plants;
	for (Zombie* zombie : state->mZombies)
	{
		if (zombie->mDead || !zombie->IsOnBoard() || zombie->mRow < 0 || zombie->mRow >= 6 ||
			!app->EnvironmentZombieVisible(zombie)) continue;
		double x = zombie->mPosX;
		const double boundary = 40.0 + 80.0 * std::round((x - 40.0) / 80.0);
		if (std::abs(x - boundary) < 0.001) x = EnvironmentWaitEvents::RenderedPublicCoordinate(x);
		const int row = zombie->mRow;
		const int col = std::clamp(static_cast<int>(std::floor((x - 40.0) / 80.0)), 0, 9);
		++zombies;
		bump(12, row, col, Clamp01(static_cast<double>(std::max(0, static_cast<int>(zombie->mZombiePhase))) / 32.0));
		bump(13, row, col, Clamp01(static_cast<double>(std::max(0, zombie->mPhaseCounter)) / 3000.0));
		bump(14, row, col, static_cast<float>(std::clamp(static_cast<double>(zombie->mVelX) / 2.0, -1.0, 1.0)));
		bump(15, row, col, static_cast<float>(std::clamp(static_cast<double>(zombie->mVelZ) / 2.0, -1.0, 1.0)));
		bump(16, row, col, Clamp01(static_cast<double>(std::max(0.0f, zombie->mAltitude)) / 600.0));
		add(17, row, col, Clamp01(static_cast<double>(std::max(0, zombie->mFlyingHealth)) / 600.0));
		add(18, row, col, zombie->mMindControlled ? 0.25f : 0.0f);
		add(19, row, col, zombie->mMindControlled ? 0.0f : 0.25f);
		bump(20, row, col, zombie->mUseLadderCol >= 0 ? Clamp01((zombie->mUseLadderCol + 1) / 10.0) : 0.0f);
		bump(21, row, col, zombie->mTargetCol >= 0 ? Clamp01((zombie->mTargetCol + 1) / 10.0) : 0.0f);
		bump(22, row, col, Clamp01(static_cast<double>(std::max(0, zombie->mSummonCounter)) / 3000.0));
		add(23, row, col, zombie->mFlatTires ? 0.25f : 0.0f);
		add(24, row, col, Clamp01(static_cast<double>(std::max(0, zombie->mBodyHealth)) / 200.0));
		add(25, row, col, Clamp01(static_cast<double>(std::max(0, zombie->mHelmHealth)) / 200.0));
		add(26, row, col, Clamp01(static_cast<double>(std::max(0, zombie->mShieldHealth)) / 200.0));
		const int parts = (zombie->mHasHead ? 1 : 0) + (zombie->mHasArm ? 1 : 0) + (zombie->mHasObject ? 1 : 0);
		bump(27, row, col, parts / 3.0f);
		bump(28, row, col, Clamp01(static_cast<double>(std::max(0, zombie->mChilledCounter)) / 1200.0));
		bump(29, row, col, Clamp01(static_cast<double>(std::max(0, zombie->mButteredCounter)) / 1200.0));
		bump(30, row, col, Clamp01(static_cast<double>(std::max(0, zombie->mIceTrapCounter)) / 1200.0));
	}
	for (const Projectile* projectile : state->mProjectiles)
	{
		if (projectile->mDead) continue;
		const int row = projectile->mRow;
		const int col = std::clamp(static_cast<int>(std::floor((projectile->mPosX - 40.0) / 80.0)), 0, 9);
		if (row < 0 || row >= 6) continue;
		add(31, row, col, 0.125f);
	}
	for (const GridItem* item : state->mGridItems)
	{
		if (item->mDead || item->mGridX < 0 || item->mGridX >= 9 || item->mGridY < 0 || item->mGridY >= 6) continue;
		const int row = item->mGridY, col = item->mGridX;
		bump(32, row, col, Clamp01(static_cast<double>(static_cast<int>(item->mGridItemType)) / 12.0));
		bump(33, row, col, Clamp01(static_cast<double>(static_cast<int>(item->mGridItemState)) / 10.0));
		bump(34, row, col, Clamp01(static_cast<double>(std::max(0, item->mGridItemCounter)) / 1200.0));
	}
}

int ExtraChannels(uint32_t version)
{
	if (version >= 6) return EnvironmentValueNet::V5ExtraChannels + EnvironmentValueNet::V6ExtraChannels;
	if (version == 5) return EnvironmentValueNet::V5ExtraChannels;
	if (version == 4) return 8;
	if (version == 3) return 6;
	return 0;
}
}

void EnvironmentValueNet::Load(const std::string& path)
{
	std::ifstream input(path, std::ios::binary);
	char magic[8]{};
	uint32_t size = 0, version = 0, contextSize = 0;
	input.read(magic, 8);
	input.read(reinterpret_cast<char*>(&size), 4);
	input.read(reinterpret_cast<char*>(&version), 4);
	input.read(reinterpret_cast<char*>(&contextSize), 4);
	if (!input || std::string(magic, 8) != "PVZVN001" || std::endian::native != std::endian::little ||
		size < 1 || size > 512 || (version < 1 || version > 6) || (contextSize != 0 && contextSize != TaskContext))
		throw std::runtime_error("invalid value network header");
	const size_t w = size, s = Scalars + (version >= 2 ? 6 : 0) + (version >= 6 ? V6ExtraScalars : 0) + contextSize;
	const std::vector<size_t> counts = {w * (Channels + ExtraChannels(version)) * 9, w, w * w * 9, w, w * w * 3, w,
		w * s, w, w * w * 8, w, w, 1};
	std::vector<std::vector<float>> loaded;
	for (size_t count : counts)
	{
		auto& values = loaded.emplace_back(count);
		input.read(reinterpret_cast<char*>(values.data()), count * sizeof(float));
		if (!input || !std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }))
			throw std::runtime_error("invalid value network weights");
	}
	if (input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("unexpected value network data");
	width = size; encoder = version; context = contextSize;
	weights = std::move(loaded);
	first.resize(width * Cells); second.resize(width * Cells);
	joined.resize(width * 8); scalarHidden.resize(width); hidden.resize(width);
	columns.reserve(std::max(Channels, width) * 9 * Cells);
	kernelBuffer.reserve(width * Channels * 9);
	activeChannels.reserve(Channels);
}

void EnvironmentValueNet::Features(LawnApp* app, double income, std::vector<float>& b, std::vector<float>& s) const
{
	Board* state = app->mBoard;
	b.assign(ChannelCount() * Cells, 0.0f); s.assign(ScalarCount(), 0.0f);
	if (encoder >= 3) EnvironmentRollout::BungeeFeatures(app, b.data() + Channels * Cells);
	if (encoder >= 4) EnvironmentRollout::SleepFeatures(app, b.data() + (Channels + 6) * Cells);
	if (encoder >= 5) EnvironmentRollout::TimerFeatures(app, b.data() + (Channels + 8) * Cells);
	if (encoder >= 6) InternalFeatures(app, b.data() + (Channels + V5ExtraChannels) * Cells);
	// numpy's cell += scalar casts each term to float32 first, then accumulates in float32;
	// rounding the term before the add is what reproduces that bit-exactly.
	for (const Plant* plant : state->mPlants)
		if (!plant->mDead && !plant->mSquished && !app->EnvironmentStormHidden() && plant->mRow >= 0 && plant->mRow < 6 && plant->mPlantCol >= 0 && plant->mPlantCol < 9)
			b[std::min(static_cast<int>(plant->mSeedType), 53) * Cells + plant->mRow * 10 + plant->mPlantCol] +=
				static_cast<float>(static_cast<double>(plant->mPlantHealth) / std::max(1, plant->mPlantMaxHealth));
	for (Zombie* zombie : state->mZombies)
	{
		if (zombie->mDead || !zombie->IsOnBoard() || zombie->mRow < 0 || zombie->mRow >= 6 ||
			!app->EnvironmentZombieVisible(zombie)) continue;
		// Public JSON rounds coordinates to six significant digits. Preserve boundary bins.
		double x = zombie->mPosX;
		const double boundary = 40.0 + 80.0 * std::round((x - 40.0) / 80.0);
		if (std::abs(x - boundary) < 0.001) x = EnvironmentWaitEvents::RenderedPublicCoordinate(x);
		const int cell = zombie->mRow * 10 + std::clamp(static_cast<int>(std::floor((x - 40.0) / 80.0)), 0, 9);
		b[(54 + std::min(static_cast<int>(zombie->mZombieType), 33)) * Cells + cell] +=
			static_cast<float>((zombie->mBodyHealth + zombie->mHelmHealth + zombie->mShieldHealth) / 200.0);
		b[88 * Cells + cell] += zombie->mIsEating;
		b[89 * Cells + cell] += zombie->mChilledCounter > 0;
	}
	for (const LawnMower* mower : state->mLawnMowers)
		if (!mower->mDead && !app->EnvironmentStormHidden() && mower->mRow >= 0 && mower->mRow < 6)
			b[90 * Cells + mower->mRow * 10] = mower->mMowerState == LawnMowerState::MOWER_READY;
	for (int row = 0; row < 6; ++row)
	{
		const auto terrain = state->mGridSquareType[0][row];
		bool uniform = true;
		for (int col = 1; col < 9; ++col) uniform &= state->mGridSquareType[col][row] == terrain;
		if (uniform && (terrain == GridSquareType::GRIDSQUARE_GRASS || terrain == GridSquareType::GRIDSQUARE_POOL))
			std::fill_n(b.data() + 91 * Cells + row * 10, 10, 1.0f);
	}
	const int waves = std::max(1, state->mNumWaves);
	s[0] = std::min(state->mSunMoney, 2000) / 1000.0;
	s[1] = static_cast<double>(state->mCurrentWave) / waves;
	s[2] = state->mMainCounter / 60000.0;
	s[3] = income / 100.0;
	s[4] = std::log1p(waves) / std::log1p(50);
	const bool conveyor = app->mEnvironmentTaskSpec.originalConveyorStart;
	for (int index = 0; index < state->mSeedBank->mNumPackets; ++index)
	{
		const SeedPacket& p = state->mSeedBank->mSeedPackets[index];
		// The packet's behaviour is its target plant, not the Imitater card identity.
		const SeedType packetType = p.mPacketType == SeedType::SEED_IMITATER &&
			p.mImitaterType != SeedType::SEED_NONE ? p.mImitaterType : p.mPacketType;
		const int kind = std::min(static_cast<int>(packetType), 53);
		if (encoder >= 2 && kind < 0) continue;
		const int cost = p.mPacketType == SeedType::SEED_NONE ? 0 : Plant::GetCost(p.mPacketType, p.mImitaterType);
		const float ready = p.mActive && ((encoder >= 2 && conveyor) || state->mSunMoney >= cost);
		const float progress = std::min(1.0, p.mActive || !p.mRefreshTime ? 1.0 : static_cast<double>(p.mRefreshCounter) / p.mRefreshTime);
		// Encoding 1 deliberately preserves the historical empty-slot scalar alias.
		s[5 + kind] = encoder >= 2 ? std::max(s[5 + kind], ready) : ready;
		s[59 + kind] = encoder >= 2 ? std::max(s[59 + kind], progress) : progress;
	}
	if (encoder >= 2)
	{
		s[113] = state->StageIsNight(); s[114] = state->StageHasPool();
		s[115] = state->StageHasFog(); s[116] = state->StageHasRoof(); s[117] = conveyor;
		int graves = 0;
		for (const GridItem* item : state->mGridItems) if (!item->mDead) ++graves;
		s[118] = graves / 45.0;
	}
	if (encoder >= 6)
	{
		// Value-net v6 wave/timing scalars, appended after the six v5 scene scalars.
		const int start = Scalars + (encoder >= 2 ? 6 : 0);
		s[start + 0] = Clamp01(static_cast<double>(std::max(0, state->mZombieCountDown)) / 3000.0);
		s[start + 1] = Clamp01(static_cast<double>(std::max(0, state->mHugeWaveCountDown)) / 1000.0);
		const int waveHealth = state->TotalZombiesHealthInWave(state->mCurrentWave - 1);
		s[start + 2] = Clamp01(static_cast<double>(waveHealth) / std::max(1, state->mZombieHealthWaveStart));
		s[start + 3] = Clamp01(static_cast<double>(std::max(0, state->mSunCountDown)) / 1200.0);
		int onBoard = 0;
		for (Zombie* zombie : state->mZombies)
			if (!zombie->mDead && zombie->IsOnBoard()) ++onBoard;
		s[start + 4] = std::min(1.0, onBoard / 50.0);
		int living = 0;
		for (const Plant* plant : state->mPlants)
			if (!plant->mDead && !plant->mSquished) ++living;
		s[start + 5] = std::min(1.0, living / 45.0);
	}
	if (context)
	{
		const int start = Scalars + (encoder >= 2 ? 6 : 0) + (encoder >= 6 ? V6ExtraScalars : 0);
		s[start] = app->mEnvironmentTaskSpec.zombieCountMultiplier / 3.0;
		for (int type = 0; type < NUM_ZOMBIE_TYPES && type < 100; ++type)
			if (state->mZombieAllowed[type]) s[start + 1 + std::min(type, 33)] = 1.0f;
	}
}

void EnvironmentValueNet::Convolve(const std::vector<float>& input, int channels, int kernelRows, int layer,
	std::vector<float>& output)
{
	const auto started = std::chrono::steady_clock::now();
	const float* kernel = weights[layer].data();
	activeChannels.clear();
	for (int ch = 0; ch < channels; ++ch)
		if (layer != 0 || std::any_of(input.data() + ch * Cells, input.data() + (ch + 1) * Cells,
			[](float x) { return x != 0.0f; })) activeChannels.push_back(ch);
	const int count = activeChannels.size() * kernelRows * 3;
	if (layer == 0)
	{
		kernelBuffer.resize(width * count);
		for (int out = 0; out < width; ++out)
			for (size_t index = 0; index < activeChannels.size(); ++index)
				std::copy_n(kernel + (out * channels + activeChannels[index]) * kernelRows * 3, kernelRows * 3,
					kernelBuffer.data() + out * count + index * kernelRows * 3);
		kernel = kernelBuffer.data();
	}
	columns.resize(count * Cells);
	for (size_t index = 0; index < activeChannels.size(); ++index)
		for (int kr = 0; kr < kernelRows; ++kr)
			for (int kc = 0; kc < 3; ++kc)
				for (int row = 0; row < 6; ++row)
					for (int col = 0; col < 10; ++col)
					{
						const int r = row + kr - kernelRows / 2, c = col + kc - 1;
						columns[((index * kernelRows + kr) * 3 + kc) * Cells + row * 10 + col] =
							r >= 0 && r < 6 && c >= 0 && c < 10 ? input[activeChannels[index] * Cells + r * 10 + c] : 0.0f;
					}
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, width, Cells, count, 1.0f,
		kernel, std::max(1, count), columns.data(), Cells, 0.0f, output.data(), Cells);
	for (int ch = 0; ch < width; ++ch)
		for (int cell = 0; cell < Cells; ++cell)
			output[ch * Cells + cell] = std::max(0.0f, output[ch * Cells + cell] + weights[layer + 1][ch]);
	convolutionSeconds[layer / 2] += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

float EnvironmentValueNet::Logit(const std::vector<float>& b, const std::vector<float>& s)
{
	const auto started = std::chrono::steady_clock::now();
	if (!Loaded() || b.size() != ChannelCount() * Cells || s.size() != static_cast<size_t>(ScalarCount()))
		throw std::runtime_error("invalid value network input");
	Convolve(b, ChannelCount(), 3, 0, first); Convolve(first, width, 3, 2, second); Convolve(second, width, 1, 4, first);
	for (int ch = 0; ch < width; ++ch)
	{
		float sum = 0;
		for (int cell = 0; cell < Cells; ++cell) sum += first[ch * Cells + cell];
		joined[ch] = sum / Cells;
		for (int row = 0; row < 6; ++row)
			joined[width + ch * 6 + row] = *std::max_element(first.data() + ch * Cells + row * 10, first.data() + ch * Cells + row * 10 + 10);
	}
	cblas_sgemv(CblasRowMajor, CblasNoTrans, width, ScalarCount(), 1.0f, weights[6].data(), ScalarCount(), s.data(), 1, 0.0f, scalarHidden.data(), 1);
	for (int ch = 0; ch < width; ++ch) joined[width * 7 + ch] = std::max(0.0f, scalarHidden[ch] + weights[7][ch]);
	cblas_sgemv(CblasRowMajor, CblasNoTrans, width, width * 8, 1.0f, weights[8].data(), width * 8, joined.data(), 1, 0.0f, hidden.data(), 1);
	for (int ch = 0; ch < width; ++ch) hidden[ch] = std::max(0.0f, hidden[ch] + weights[9][ch]);
	const float logit = cblas_sdot(width, weights[10].data(), 1, hidden.data(), 1) + weights[11][0];
	++calls;
	forwardSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
	return logit;
}

float EnvironmentValueNet::Probability(float logit)
{
	return logit >= 0 ? 1.0f / (1.0f + std::exp(-logit)) : std::exp(logit) / (1.0f + std::exp(logit));
}

float EnvironmentValueNet::Predict(LawnApp* app, double income)
{
	const auto started = std::chrono::steady_clock::now();
	Features(app, income, board, scalars);
	featureSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
	return Probability(Logit(board, scalars));
}
