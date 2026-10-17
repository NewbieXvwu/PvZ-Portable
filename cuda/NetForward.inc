// The student forward and the decision it produces, from one source compiled by nvcc (thread = lane)
// and by g++ (single lane).
//
// This is a transcription of src/EnvironmentRollout.cpp -- Encode (the three trunk convolutions plus
// the global context) and Heads (the plant / packet / cell heads) -- with the same loop order, so the
// float accumulation sequence matches the CPU path tap for tap.  The point is not bitwise identity
// with BLAS, which is not achievable anyway; it is that the *semantics* of a decision are the
// original's:
//
//   * no legal plant anywhere   -> the network is not run at all (EnvironmentRollout.cpp:269),
//   * plant head <= 0           -> wait,
//   * the packet must be pickable, affordable, and have at least one legal cell,
//   * ties go to the lower slot (`packets[index] <= bestPacket` continues),
//   * ties between cells go to the first in row-major order,
//   * the cell head is a second half-forward that only runs for the chosen packet.
//
// The weights arrive as the PVZRN dump of python/pvz_rollout_policy.py::export_student; the element
// counts below are the ones EnvironmentRollout::Load reads, so a dump that loads on the CPU loads
// here too.
#ifndef PVZ_NET_FORWARD_INC
#define PVZ_NET_FORWARD_INC

#include "NetCore.h"
#include "SimCore.h"

namespace pvzc
{

// Element of a [channel][cell] plane for this lane.  `stride` and `self` are the two locals every
// routine below carries: stride is the batch's lane count (1 in the single-lane host build), self is
// the lane index, so a warp touching one channel of one cell reads 32 consecutive floats.
#define PVZ_PLANE(plane, channel, cell) ((plane)[((channel) * kCells + (cell)) * stride + self])
#define PVZ_VECTOR(vector, index) ((vector)[(index) * stride + self])

PVZ_SIM float Relu(float value) { return value > 0.0f ? value : 0.0f; }

// A 3x3 convolution over the 6x10 feature grid with zero padding, which is what nn.Conv2d(padding=1)
// computes.  trunk[0] runs it with 92 input channels, trunk[1] with `width`; the original's sparse
// first layer skips input channels whose whole plane is zero, and an all-zero plane adds exactly 0 to
// every accumulator, so computing it costs time and changes nothing.
PVZ_SIM void Conv3x3(Lane lane, const float* input, int32_t channels, int32_t outChannels,
	const float* kernel, const float* bias, float* output)
{
	const int32_t stride = lane.stride;
	const int32_t self = lane.lane;
	for (int32_t oc = 0; oc < outChannels; ++oc)
	{
		const float* weight = kernel + oc * (channels * 9);
		for (int32_t cell = 0; cell < kCells; ++cell)
		{
			const int32_t row = cell / kCols, col = cell % kCols;
			float acc = bias[oc];
			for (int32_t ic = 0; ic < channels; ++ic)
			{
				const float* plane = input + ic * kCells * stride + self;
				const float* tap = weight + ic * 9;
				for (int32_t kr = 0; kr < 3; ++kr)
				{
					const int32_t sourceRow = row + kr - 1;
					if (sourceRow < 0 || sourceRow >= kRows) continue;
					for (int32_t kc = 0; kc < 3; ++kc)
					{
						const int32_t sourceColumn = col + kc - 1;
						if (sourceColumn < 0 || sourceColumn >= kCols) continue;
						acc += tap[kr * 3 + kc] * plane[sourceRow * kCols + sourceColumn];
					}
				}
			}
			output[(oc * kCells + cell) * stride + self] = Relu(acc);
		}
	}
}

// trunk[2]: kernel (1, 5), padding (0, 2) -- rows never move, so only the column needs a bounds test.
PVZ_SIM void Conv1x5(Lane lane, const float* input, int32_t channels, int32_t outChannels,
	const float* kernel, const float* bias, float* output)
{
	const int32_t stride = lane.stride;
	const int32_t self = lane.lane;
	for (int32_t oc = 0; oc < outChannels; ++oc)
	{
		const float* weight = kernel + oc * (channels * 5);
		for (int32_t cell = 0; cell < kCells; ++cell)
		{
			const int32_t row = cell / kCols, col = cell % kCols;
			float acc = bias[oc];
			for (int32_t ic = 0; ic < channels; ++ic)
			{
				const float* plane = input + ic * kCells * stride + self;
				const float* tap = weight + ic * 5;
				for (int32_t kc = 0; kc < 5; ++kc)
				{
					const int32_t sourceColumn = col + kc - 2;
					if (sourceColumn < 0 || sourceColumn >= kCols) continue;
					acc += tap[kc] * plane[row * kCols + sourceColumn];
				}
			}
			output[(oc * kCells + cell) * stride + self] = Relu(acc);
		}
	}
}

// Encode: the three convolutions and the global context.  `hidden` and `scratch` ping-pong, so two
// [width][kCells] planes are live at a time -- the same buffers EnvironmentRollout reuses.
PVZ_SIM void NetEncode(Lane lane, const NetParams& net, const NetState& state)
{
	const int32_t stride = lane.stride;
	const int32_t self = lane.lane;
	const int32_t width = net.width;
	const int32_t context = net.contextSize;

	Conv3x3(lane, state.board, kBoardChannels, width, net.conv1, net.bias1, state.hidden);
	Conv3x3(lane, state.hidden, width, width, net.conv2, net.bias2, state.scratch);
	Conv1x5(lane, state.scratch, width, width, net.conv3, net.bias3, state.hidden);

	for (int32_t oc = 0; oc < width; ++oc)
	{
		// The plane is post-ReLU, so a 0 start for the max matches std::max_element over non-negative
		// values without needing a first-element special case.
		float total = 0.0f;
		float largest = 0.0f;
		for (int32_t cell = 0; cell < kCells; ++cell)
		{
			const float value = PVZ_PLANE(state.hidden, oc, cell);
			total += value;
			if (value > largest) largest = value;
		}
		PVZ_VECTOR(state.context, oc) = total / static_cast<float>(kCells);
		if (net.meanmax) PVZ_VECTOR(state.context, width + oc) = largest;
	}
	for (int32_t oc = 0; oc < width; ++oc)
	{
		const float* weight = net.scalarWeight + oc * net.scalarCount;
		float acc = net.scalarBias[oc];
		for (int32_t index = 0; index < net.scalarCount; ++index)
			acc += PVZ_VECTOR(state.scalars, index) * weight[index];
		PVZ_VECTOR(state.context, context - width + oc) = Relu(acc);
	}
}

PVZ_SIM float PlantLogit(Lane lane, const NetParams& net, const NetState& state)
{
	const int32_t stride = lane.stride;
	const int32_t self = lane.lane;
	float acc = net.plantBias[0];
	for (int32_t index = 0; index < net.contextSize; ++index)
		acc += PVZ_VECTOR(state.context, index) * net.plantWeight[index];
	return acc;
}

PVZ_SIM float PacketLogit(Lane lane, const NetParams& net, const NetState& state, int32_t packet)
{
	const int32_t stride = lane.stride;
	const int32_t self = lane.lane;
	const float* weight = net.packetWeight + packet * net.contextSize;
	float acc = net.packetBias[packet];
	for (int32_t index = 0; index < net.contextSize; ++index)
		acc += PVZ_VECTOR(state.context, index) * weight[index];
	return acc;
}

// Heads(packet).  `scratch` holds the 1x1 convolution's pre-activation (its conv2 result is spent by
// now), which is what the CPU path does too, so the final combine reads it instead of keeping kCells
// accumulators live across the channel loop.
PVZ_SIM void CellLogits(Lane lane, const NetParams& net, const NetState& state, int32_t packet,
	float* logits)
{
	const int32_t stride = lane.stride;
	const int32_t self = lane.lane;
	const int32_t width = net.width;
	const int32_t context = net.contextSize;
	const int32_t step = 2 * width + context;
	const float* embedding = net.packetEmbedding + packet * width;

	for (int32_t oc = 0; oc < width; ++oc)
	{
		const float* kernel = net.cellWeight + oc * step;
		float base = net.cellBias[oc];
		for (int32_t index = 0; index < width; ++index) base += embedding[index] * kernel[width + index];
		for (int32_t index = 0; index < context; ++index)
			base += PVZ_VECTOR(state.context, index) * kernel[2 * width + index];
		for (int32_t cell = 0; cell < kCells; ++cell)
		{
			// One accumulator in a register per (oc, cell), one store per pair: the alternative of
			// filling the plane with `base` and then read-modify-writing it per input channel would put
			// every one of these FMAs behind a global load-store pair.
			float acc = base;
			for (int32_t ic = 0; ic < width; ++ic)
				acc += kernel[ic] * PVZ_PLANE(state.hidden, ic, cell);
			PVZ_PLANE(state.scratch, oc, cell) = acc;
		}
	}
	for (int32_t cell = 0; cell < kRows * kGridCols; ++cell)
	{
		// The lawn is 5x9 of the 6x10 feature grid and column 9 (beyond the lawn) is never an option:
		// EnvironmentRollout.cpp:341 reads cell / 9 * 10 + cell % 9.
		const int32_t gridCell = (cell / kGridCols) * kCols + cell % kGridCols;
		float acc = net.cellOutBias[0];
		for (int32_t oc = 0; oc < width; ++oc)
			acc += Relu(PVZ_PLANE(state.scratch, oc, gridCell)) * net.cellOut[oc];
		logits[cell] = acc;
	}
}

struct Choice
{
	int32_t packet;               // -1 = wait
	int32_t row;
	int32_t col;
	int32_t forwarded;            // 1 = the network ran; net_calls counts these, not decisions
};

// The whole decision, in Predict's order.  `logits` is the lane's own 54-float row-major cell array.
PVZ_SIM Choice NetDecide(Lane lane, const LevelInfo& level, const NetParams& net, const NetState& state,
	const float* waveLog, float* logits)
{
	Choice choice = {-1, 0, 0, 0};
	// The short circuit has to come first: Predict returns before ++networkCalls when nothing can be
	// planted anywhere, so net_calls <= decisions holds on both sides only if the device skips too.
	if (!AnyLegalPlant(lane, level)) return choice;
	choice.forwarded = 1;

	BuildFeatures(lane, level, state, waveLog, lane.s(S_PRIOR_TICK) >= 0);
	NetEncode(lane, net, state);
	if (!(PlantLogit(lane, net, state) > 0.0f)) return choice;

	const int32_t rows = Maximum(level.playableRows, 1);
	float bestPacket = 0.0f;
	int32_t havePacket = 0;
	for (int32_t packet = 0; packet < kPacketSlots; ++packet)
	{
		if (lane.i(F_PACKET_TYPE, packet) < 0) continue;
		if (!lane.i(F_PACKET_ACTIVE, packet)) continue;
		if (lane.s(S_SUN) < lane.i(F_PACKET_COST, packet)) continue;
		const float logit = PacketLogit(lane, net, state, packet);
		// `<= best` skips, so an equal logit keeps the slot already chosen: the lowest index wins.
		if (havePacket && logit <= bestPacket) continue;
		int32_t legal = 0;
		for (int32_t row = 0; row < rows && !legal; ++row)
			for (int32_t col = 0; col < kGridCols; ++col)
				if (lane.i(F_PLANT_TYPE, row * kCols + col) < 0) { legal = 1; break; }
		if (!legal) continue;
		bestPacket = logit;
		havePacket = 1;
		choice.packet = packet;
	}
	if (!havePacket) return choice;

	CellLogits(lane, net, state, choice.packet, logits);
	float best = 0.0f;
	int32_t haveCell = 0;
	for (int32_t row = 0; row < rows; ++row)
		for (int32_t col = 0; col < kGridCols; ++col)
		{
			if (lane.i(F_PLANT_TYPE, row * kCols + col) >= 0) continue;
			const float value = logits[row * kGridCols + col];
			if (haveCell && !(value > best)) continue;
			best = value;
			haveCell = 1;
			choice.row = row;
			choice.col = col;
		}
	return choice;
}

#undef PVZ_PLANE
#undef PVZ_VECTOR

}  // namespace pvzc

#endif
