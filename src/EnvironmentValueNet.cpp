#include "EnvironmentValueNet.h"
#include "EnvironmentRollout.h"
#include "LawnApp.h"
#include "Lawn/Board.h"
#include "Lawn/Plant.h"
#include "Lawn/Zombie.h"
#include "Lawn/LawnMower.h"
#include "Lawn/SeedPacket.h"
#include "Lawn/GridItem.h"
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
		size < 1 || size > 512 || (version < 1 || version > 4) || (contextSize != 0 && contextSize != TaskContext))
		throw std::runtime_error("invalid value network header");
	const size_t w = size, s = Scalars + (version >= 2 ? 6 : 0) + contextSize;
	const std::vector<size_t> counts = {w * (Channels + (version == 4 ? 8 : version == 3 ? 6 : 0)) * 9, w, w * w * 9, w, w * w * 3, w,
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
	if (encoder == 4) EnvironmentRollout::SleepFeatures(app, b.data() + (Channels + 6) * Cells);
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
		const int kind = std::min(static_cast<int>(p.mPacketType), 53);
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
	if (context)
	{
		const int start = Scalars + (encoder >= 2 ? 6 : 0);
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
