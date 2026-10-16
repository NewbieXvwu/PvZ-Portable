#include "EnvironmentRollout.h"
#include "LawnApp.h"
#include "Lawn/Board.h"
#include "Lawn/Plant.h"
#include "Lawn/Zombie.h"
#include "Lawn/LawnMower.h"
#include "Lawn/SeedPacket.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

#ifdef PVZ_ROLLOUT_BLAS
extern "C" void sgemm_(const char*, const char*, const int*, const int*, const int*, const float*,
	const float*, const int*, const float*, const int*, const float*, float*, const int*);
#endif

void EnvironmentRollout::Load(const std::string& path)
{
	std::ifstream input(path, std::ios::binary);
	char magic[8];
	uint32_t size = 0;
	input.read(magic, 8);
	input.read(reinterpret_cast<char*>(&size), 4);
	if (!input || std::string(magic, 8) != "PVZRN001" || std::endian::native != std::endian::little || size < 1 || size > 512)
		throw std::runtime_error("invalid rollout network header");
	const size_t nextWidth = size;
	const std::vector<size_t> counts = {
		nextWidth * Channels * 9, nextWidth, nextWidth * nextWidth * 9, nextWidth,
		nextWidth * nextWidth * 5, nextWidth, nextWidth * Scalars, nextWidth,
		2 * nextWidth, 1, 20 * nextWidth, 10, 10 * nextWidth,
		4 * nextWidth * nextWidth, nextWidth, nextWidth, 1};
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
	weights = std::move(loaded);
	hidden.resize(Cells * width);
	scratch.resize(Cells * width);
	context.resize(2 * width);
}

void EnvironmentRollout::Features(LawnApp* app, double income, int previousTick, bool previousPlant,
	std::vector<float>& board, std::vector<float>& scalars)
{
	Board* state = app->mBoard;
	board.assign(Channels * Cells, 0.0f);
	scalars.assign(Scalars, 0.0f);
	for (const Plant* plant : state->mPlants)
		if (!plant->mDead && !plant->mSquished && plant->mRow >= 0 && plant->mRow < 6 && plant->mPlantCol >= 0 && plant->mPlantCol < 9)
			board[std::min(static_cast<int>(plant->mSeedType), 53) * Cells + plant->mRow * 10 + plant->mPlantCol] +=
				static_cast<double>(plant->mPlantHealth) / std::max(1, plant->mPlantMaxHealth);
	for (Zombie* zombie : state->mZombies)
	{
		if (zombie->mDead || !zombie->IsOnBoard() || zombie->mRow < 0 || zombie->mRow >= 6) continue;
		double position = zombie->mPosX;
		const double boundary = 40.0 + 80.0 * std::round((position - 40.0) / 80.0);
		if (std::abs(position - boundary) < 0.001)
			position = EnvironmentWaitEvents::RenderedPublicCoordinate(position);
		const int cell = zombie->mRow * 10 + std::clamp(static_cast<int>(std::floor((position - 40.0) / 80.0)), 0, 9);
		board[(54 + std::min(static_cast<int>(zombie->mZombieType), 33)) * Cells + cell] +=
			(zombie->mBodyHealth + zombie->mHelmHealth + zombie->mShieldHealth) / 200.0;
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
	const int waves = std::max(1, state->mNumWaves);
	scalars[0] = std::min(state->mSunMoney, 2000) / 1000.0;
	scalars[1] = static_cast<double>(state->mCurrentWave) / waves;
	scalars[2] = state->mMainCounter / 60000.0;
	scalars[3] = income / 100.0;
	scalars[4] = std::log1p(waves) / std::log1p(50);
	for (int index = 0; index < state->mSeedBank->mNumPackets; ++index)
	{
		const SeedPacket& packet = state->mSeedBank->mSeedPackets[index];
		const int kind = std::min(static_cast<int>(packet.mPacketType), 53);
		const int cost = packet.mPacketType == SeedType::SEED_NONE ? 0 : Plant::GetCost(packet.mPacketType, packet.mImitaterType);
		if (kind < 0) continue;
		scalars[5 + kind] = packet.mActive && state->mSunMoney >= cost;
		scalars[59 + kind] = std::min(1.0, packet.mActive || !packet.mRefreshTime ? 1.0 :
			static_cast<double>(packet.mRefreshCounter) / packet.mRefreshTime);
	}
	if (previousTick >= 0)
	{
		scalars[113] = previousPlant;
		scalars[114] = (state->mMainCounter - previousTick) / 300.0;
	}
}

void EnvironmentRollout::Convolve(const std::vector<float>& input, int channels, int kernelRows, int kernelCols,
	int layer, std::vector<float>& output, bool sparse)
{
	const auto& kernel = weights[layer];
	const auto& bias = weights[layer + 1];
	std::vector<int> active;
	for (int channel = 0; channel < channels; ++channel)
		if (!sparse || std::any_of(input.data() + channel * Cells, input.data() + (channel + 1) * Cells,
			[](float value) { return value != 0.0f; })) active.push_back(channel);
	const int count = active.size() * kernelRows * kernelCols;
	columns.assign(count * Cells, 0.0f);
	kernelBuffer.resize(width * count);
	int feature = 0;
	for (int channel : active)
		for (int kernelRow = 0; kernelRow < kernelRows; ++kernelRow)
			for (int kernelCol = 0; kernelCol < kernelCols; ++kernelCol, ++feature)
			{
				const int deltaRow = kernelRow - kernelRows / 2, deltaCol = kernelCol - kernelCols / 2;
				for (int row = std::max(0, -deltaRow); row < std::min(6, 6 - deltaRow); ++row)
					for (int col = std::max(0, -deltaCol); col < std::min(10, 10 - deltaCol); ++col)
						columns[feature * Cells + row * 10 + col] = input[channel * Cells + (row + deltaRow) * 10 + col + deltaCol];
				if (sparse)
					for (int target = 0; target < width; ++target)
						kernelBuffer[target * count + feature] = kernel[((target * channels + channel) * kernelRows + kernelRow) * kernelCols + kernelCol];
			}
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
	for (int target = 0; target < width; ++target)
	{
		float* dest = output.data() + target * Cells;
		for (int index = 0; index < count; ++index)
		{
			const float factor = matrix[target * count + index];
			const float* source = columns.data() + index * Cells;
			for (int cell = 0; cell < Cells; ++cell) dest[cell] += factor * source[cell];
		}
	}
#endif
	for (float& value : output) value = std::max(0.0f, value);
}

static float Linear(const float* input, const float* weight, int count, float bias)
{
	for (int index = 0; index < count; ++index) bias += input[index] * weight[index];
	return bias;
}

void EnvironmentRollout::Encode(const std::vector<float>& board, const std::vector<float>& scalars)
{
	Convolve(board, Channels, 3, 3, 0, hidden, true);
	Convolve(hidden, width, 3, 3, 2, scratch, false);
	Convolve(scratch, width, 1, 5, 4, hidden, false);
	for (int channel = 0; channel < width; ++channel)
	{
		float total = 0;
		for (int cell = 0; cell < Cells; ++cell) total += hidden[channel * Cells + cell];
		context[channel] = total / Cells;
		context[width + channel] = std::max(0.0f, Linear(scalars.data(), weights[6].data() + channel * Scalars, Scalars, weights[7][channel]));
	}
}

EnvironmentRollout::Prediction EnvironmentRollout::Heads(int packet)
{
	Prediction result{};
	result.cellPacket = packet;
	result.plant = Linear(context.data(), weights[8].data(), 2 * width, weights[9][0]);
	for (int index = 0; index < 10; ++index)
		result.packets[index] = Linear(context.data(), weights[10].data() + index * 2 * width, 2 * width, weights[11][index]);
	if (packet < 0) return result;
	result.cells.fill(weights[16][0]);
	for (int channel = 0; channel < width; ++channel)
	{
		const float* kernel = weights[13].data() + channel * 4 * width;
		float base = Linear(weights[12].data() + packet * width, kernel + width, width, weights[14][channel]);
		base = Linear(context.data(), kernel + 2 * width, 2 * width, base);
		std::fill_n(scratch.data() + channel * Cells, Cells, base);
	}
#ifdef PVZ_ROLLOUT_BLAS
	const char normal = 'N';
	const float one = 1.0f;
	const int cells = Cells, stride = 4 * width;
	sgemm_(&normal, &normal, &cells, &width, &width, &one, hidden.data(), &cells, weights[13].data(), &stride,
		&one, scratch.data(), &cells);
#else
	for (int channel = 0; channel < width; ++channel)
		for (int index = 0; index < width; ++index)
		{
			const float factor = weights[13][channel * 4 * width + index];
			for (int cell = 0; cell < Cells; ++cell)
				scratch[channel * Cells + cell] += hidden[index * Cells + cell] * factor;
		}
#endif
	for (int channel = 0; channel < width; ++channel)
		for (int cell = 0; cell < 54; ++cell)
			result.cells[cell] += std::max(0.0f, scratch[channel * Cells + cell / 9 * 10 + cell % 9]) * weights[15][channel];
	return result;
}

EnvironmentRollout::Prediction EnvironmentRollout::Predict(LawnApp* app, double income, int previousTick, bool previousPlant, bool diagnostics)
{
	Prediction result{};
	if (!diagnostics && !app->EnvironmentAnyLegalPlant()) return result;
	std::vector<float> board, scalars;
	Features(app, income, previousTick, previousPlant, board, scalars);
	Encode(board, scalars);
	result = Heads(-1);
	if (result.plant <= 0 && !diagnostics) return result;
	float bestPacket = -std::numeric_limits<float>::infinity();
	std::vector<Action> legal;
	for (int index = 0; index < app->mBoard->mSeedBank->mNumPackets; ++index)
	{
		SeedPacket& packet = app->mBoard->mSeedBank->mSeedPackets[index];
		if (!packet.CanPickUp() || result.packets[index] <= bestPacket) continue;
		const SeedType seed = packet.mPacketType == SeedType::SEED_IMITATER ? packet.mImitaterType : packet.mPacketType;
		std::vector<Action> cells;
		for (int row = 0; row < 6; ++row)
			for (int col = 0; col < 9; ++col)
				if (app->mBoard->CanPlantAt(col, row, seed) == PlantingReason::PLANTING_OK)
					cells.push_back({index, col, row});
		if (!cells.empty()) { bestPacket = result.packets[index]; legal = std::move(cells); }
	}
	if (legal.empty()) return result;
	result = Heads(legal[0].packet);
	if (result.plant > 0)
		result.action = *std::max_element(legal.begin(), legal.end(), [&result](const Action& left, const Action& right) {
			return result.cells[left.row * 9 + left.col] < result.cells[right.row * 9 + right.col]; });
	return result;
}
