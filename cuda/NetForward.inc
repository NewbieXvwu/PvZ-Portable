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
//
// Two ways to run the trunk read this file.  The device driver replaces NetTrunk with strided-batched
// GEMMs over one chunk of lanes (cuda/RolloutCuda.cu), which is why the planes are padded and chunked
// (cuda/CudaRollout.h) and why every access goes through PlaneIndex/VectorIndex: the GEMMs and these
// loops then address the same float.  NetContext and the heads stay per-lane in both builds except for
// one piece: the cell head's 1x1 convolution is 93% of the head's arithmetic and does not depend on the
// chosen packet, so the device batches it too (CellPre) while the host build runs the loop below.
#ifndef PVZ_NET_FORWARD_INC
#define PVZ_NET_FORWARD_INC

#include "NetCore.h"
#include "SimCore.h"

namespace pvzc
{

// Element of a [channel][cell] plane and of a per-lane vector, for this lane's *activation* slot.
// `channels` / `width` are the buffer's declared capacity and have to be the same at every access to
// that buffer, because the chunk a lane lives in is derived from the capacity.
#define PVZ_PLANE(plane, channels, channel, cell) \
	((plane)[PlaneIndex((channels), lane.netSlot, (channel), PaddedCell(cell))])
#define PVZ_POS(plane, channels, channel, pos) \
	((plane)[PlaneIndex((channels), lane.netSlot, (channel), (pos))])
#define PVZ_VECTOR(vector, width, index) ((vector)[VectorIndex((width), lane.netSlot, (index))])

PVZ_SIM float Relu(float value) { return value > 0.0f ? value : 0.0f; }

// A 3x3 convolution over the 6x10 feature grid with zero padding, which is what nn.Conv2d(padding=1)
// computes.  trunk[0] runs it with 92 input channels, trunk[1] with `width`; the original's sparse
// first layer skips input channels whose whole plane is zero, and an all-zero plane adds exactly 0 to
// every accumulator, so computing it costs time and changes nothing.
//
// The padded grid is what removes the bounds tests: a source cell off the lawn is a halo cell, the halo
// is never written by anything, and a tap shift of -1..+1 rows and -2..+2 columns from a valid cell
// provably stays inside the kRowsP x kColsP plane.
PVZ_SIM void Conv3x3(Lane lane, const float* input, int32_t channels, int32_t outChannels,
	const float* kernel, const float* bias, float* output)
{
	for (int32_t oc = 0; oc < outChannels; ++oc)
	{
		const float* weight = kernel + oc * (channels * 9);
		for (int32_t cell = 0; cell < kCells; ++cell)
		{
			const int32_t origin = PaddedCell(cell);
			float acc = bias[oc];
			for (int32_t ic = 0; ic < channels; ++ic)
			{
				const float* tap = weight + ic * 9;
				for (int32_t kr = 0; kr < 3; ++kr)
					for (int32_t kc = 0; kc < 3; ++kc)
						acc += tap[kr * 3 + kc]
							* PVZ_POS(input, channels, ic, origin + (kr - 1) * kColsP + (kc - 1));
			}
			PVZ_PLANE(output, outChannels, oc, cell) = Relu(acc);
		}
	}
}

// trunk[2]: kernel (1, 5), padding (0, 2) -- rows never move.
PVZ_SIM void Conv1x5(Lane lane, const float* input, int32_t channels, int32_t outChannels,
	const float* kernel, const float* bias, float* output)
{
	for (int32_t oc = 0; oc < outChannels; ++oc)
	{
		const float* weight = kernel + oc * (channels * 5);
		for (int32_t cell = 0; cell < kCells; ++cell)
		{
			const int32_t origin = PaddedCell(cell);
			float acc = bias[oc];
			for (int32_t ic = 0; ic < channels; ++ic)
			{
				const float* tap = weight + ic * 5;
				for (int32_t kc = 0; kc < 5; ++kc)
					acc += tap[kc] * PVZ_POS(input, channels, ic, origin + (kc - 2));
			}
			PVZ_PLANE(output, outChannels, oc, cell) = Relu(acc);
		}
	}
}

// The trunk by itself: the part the device runs as batched GEMMs instead of as the loops above.
// `hidden` and `scratch` ping-pong, so two width-plane buffers are live at a time -- the same buffers
// EnvironmentRollout reuses.
PVZ_SIM void NetTrunk(Lane lane, const NetParams& net, const NetState& state)
{
	const int32_t width = net.width;
	Conv3x3(lane, state.board, net.boardChannels, width, net.conv1, net.bias1, state.hidden);
	Conv3x3(lane, state.hidden, width, width, net.conv2, net.bias2, state.scratch);
	Conv1x5(lane, state.scratch, width, width, net.conv3, net.bias3, state.hidden);
}

// The global context: mean and max over the grid plus the scalar branch.  It reads the trunk's last
// result out of `hidden`, so it runs after NetTrunk (host) or after the batched GEMMs (device).
PVZ_SIM void NetContextChannel(Lane lane, const NetParams& net, const NetState& state, int32_t oc)
{
	const int32_t width = net.width;
	const int32_t context = net.contextSize;

	// The plane is post-ReLU, so a 0 start for the max matches std::max_element over non-negative
	// values without needing a first-element special case.
	float total = 0.0f;
	float largest = 0.0f;
	for (int32_t cell = 0; cell < kCells; ++cell)
	{
		const float value = PVZ_PLANE(state.hidden, width, oc, cell);
		total += value;
		if (value > largest) largest = value;
	}
	PVZ_VECTOR(state.context, context, oc) = total / static_cast<float>(kCells);
	if (net.meanmax) PVZ_VECTOR(state.context, context, width + oc) = largest;

	const float* weight = net.scalarWeight + oc * net.scalarCount;
	float acc = net.scalarBias[oc];
	for (int32_t index = 0; index < net.scalarCount; ++index)
		acc += PVZ_VECTOR(state.scalars, kScalarFeatures, index) * weight[index];
	PVZ_VECTOR(state.context, context, context - width + oc) = Relu(acc);
}

PVZ_SIM void NetContext(Lane lane, const NetParams& net, const NetState& state)
{
	for (int32_t oc = 0; oc < net.width; ++oc) NetContextChannel(lane, net, state, oc);
}

PVZ_SIM void NetEncode(Lane lane, const NetParams& net, const NetState& state)
{
	NetTrunk(lane, net, state);
	NetContext(lane, net, state);
}

PVZ_SIM float PlantLogit(Lane lane, const NetParams& net, const NetState& state)
{
	float acc = net.plantBias[0];
	for (int32_t index = 0; index < net.contextSize; ++index)
		acc += PVZ_VECTOR(state.context, net.contextSize, index) * net.plantWeight[index];
	return acc;
}

PVZ_SIM float PacketLogit(Lane lane, const NetParams& net, const NetState& state, int32_t packet)
{
	const float* weight = net.packetWeight + packet * net.contextSize;
	float acc = net.packetBias[packet];
	for (int32_t index = 0; index < net.contextSize; ++index)
		acc += PVZ_VECTOR(state.context, net.contextSize, index) * weight[index];
	return acc;
}

// The cell head's 1x1 convolution, which does not depend on the chosen packet: the same 64x64 kernel
// applies whichever seed card the plant head picks.  That is what lets the device run it as a batched
// GEMM over the whole forward (cuda/RolloutCuda.cu's RunCellPre, the same m=10240 shape as a trunk tap)
// instead of here -- measured on its own, this loop was 1954 ns per lane-forward against the trunk's
// 1024 ns for 24 times the arithmetic, because one thread per lane cannot hide a dependent chain of
// 64 FMAs.  The host build runs it, so the reference path is complete.
PVZ_SIM void CellPre(Lane lane, const NetParams& net, const NetState& state)
{
	const int32_t width = net.width;
	const int32_t step = 2 * width + net.contextSize;
	for (int32_t oc = 0; oc < width; ++oc)
	{
		const float* kernel = net.cellWeight + oc * step;
		for (int32_t cell = 0; cell < kCells; ++cell)
		{
			float acc = 0.0f;
			for (int32_t ic = 0; ic < width; ++ic)
				acc += kernel[ic] * PVZ_PLANE(state.hidden, width, ic, cell);
			PVZ_PLANE(state.scratch, width, oc, cell) = acc;
		}
	}
}

// The rest of Heads(packet): add the packet-dependent base (bias + embedding + context, EnvironmentRollout
// pre-fills its scratch with exactly this before accumulating the convolution into it), take the ReLU in
// place, then the 1x1 output over the lawn.  `scratch` holds the pre-activation from CellPre or from the
// batched GEMM; base is folded in at the end of the sum rather than the start, which is the one place the
// device and host orders differ by rounding -- and the CPU original gets it from a BLAS accumulation
// anyway, so neither side is bitwise against the other here.
PVZ_SIM void CellLogitsChannel(Lane lane, const NetParams& net, const NetState& state, int32_t packet, int32_t oc)
{
	const int32_t width = net.width;
	const int32_t context = net.contextSize;
	const int32_t step = 2 * width + context;
	const float* embedding = net.packetEmbedding + packet * width;

	const float* kernel = net.cellWeight + oc * step;
	float base = net.cellBias[oc];
	for (int32_t index = 0; index < width; ++index) base += embedding[index] * kernel[width + index];
	for (int32_t index = 0; index < context; ++index)
		base += PVZ_VECTOR(state.context, context, index) * kernel[2 * width + index];
	for (int32_t cell = 0; cell < kCells; ++cell)
	{
		float& value = PVZ_PLANE(state.scratch, width, oc, cell);
		value = Relu(value + base);
	}
}

PVZ_SIM void CellLogitCell(Lane lane, const NetParams& net, const NetState& state, int32_t cell)
{
	const int32_t width = net.width;

	// The lawn is 5x9 of the 6x10 feature grid and column 9 (beyond the lawn) is never an option:
	// EnvironmentRollout.cpp:341 reads cell / 9 * 10 + cell % 9.
	const int32_t gridCell = (cell / kGridCols) * kCols + cell % kGridCols;
	float acc = net.cellOutBias[0];
	for (int32_t oc = 0; oc < width; ++oc)
		acc += PVZ_PLANE(state.scratch, width, oc, gridCell) * net.cellOut[oc];
	// Strided by the activation slot like every other per-lane array: a flat `logits[cell]` would put
	// every lane of the batch on top of the same 54 floats.
	state.logits[VectorIndex(kLogitCells, lane.netSlot, cell)] = acc;
}

PVZ_SIM void CellLogits(Lane lane, const NetParams& net, const NetState& state, int32_t packet)
{
	for (int32_t oc = 0; oc < net.width; ++oc) CellLogitsChannel(lane, net, state, packet, oc);
	for (int32_t cell = 0; cell < kLogitCells; ++cell) CellLogitCell(lane, net, state, cell);
}

// The plant gate and the packet pick -- everything up to the cell head.  Split out of NetChoose because
// the device runs the cell head's convolution over the whole batch between the two halves.
PVZ_SIM Choice NetPickPacket(Lane lane, const LevelInfo& level, const NetParams& net,
	const NetState& state)
{
	Choice choice = {-1, 0, 0, 1};
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
	return choice;
}

// The cell argmax of Heads(packet)'s logits, Predict's last step.  Ties go to the first cell in
// row-major order because a strictly-greater value is what replaces the incumbent.
PVZ_SIM Choice NetPickCell(Lane lane, const LevelInfo& level, const NetState& state, Choice choice)
{
	const int32_t rows = Maximum(level.playableRows, 1);
	float best = 0.0f;
	int32_t haveCell = 0;
	for (int32_t row = 0; row < rows; ++row)
		for (int32_t col = 0; col < kGridCols; ++col)
		{
			if (lane.i(F_PLANT_TYPE, row * kCols + col) >= 0) continue;
			const float value = state.logits[VectorIndex(kLogitCells, lane.netSlot,
				row * kGridCols + col)];
			if (haveCell && !(value > best)) continue;
			best = value;
			haveCell = 1;
			choice.row = row;
			choice.col = col;
		}
	return choice;
}

// The decision once the trunk has run, in Predict's order: the plant gate, the packet pick, the cell
// head, the cell argmax.  The device's RoundPost runs the same pieces with CellPre taken out and
// batched, and NetDecide / RunRollout keep this one for the host build.
PVZ_SIM Choice NetChoose(Lane lane, const LevelInfo& level, const NetParams& net, const NetState& state)
{
	const Choice choice = NetPickPacket(lane, level, net, state);
	if (choice.packet < 0) return choice;
	CellPre(lane, net, state);
	CellLogits(lane, net, state, choice.packet);
	return NetPickCell(lane, level, state, choice);
}

// The whole decision for one lane, features included: what the host rollout calls, and the same steps
// RoundPre and RoundPost perform on the device with the trunk taken out and batched.
PVZ_SIM Choice NetDecide(Lane lane, const LevelInfo& level, const NetParams& net, const NetState& state,
	const float* waveLog)
{
	Choice choice = {-1, 0, 0, 0};
	// The short circuit has to come first: Predict returns before ++networkCalls when nothing can be
	// planted anywhere, so net_calls <= decisions holds on both sides only if the device skips too.
	if (net.observation > 1) UpdateStudentHistory(lane);
	if (!AnyLegalPlant(lane, level)) return choice;
	BuildFeatures(lane, level, state, waveLog, lane.s(S_PRIOR_TICK) >= 0, net.boardChannels,
		net.scalarCount, net.observation);
	NetEncode(lane, net, state);
	return NetChoose(lane, level, net, state);
}

#undef PVZ_PLANE
#undef PVZ_POS
#undef PVZ_VECTOR

}  // namespace pvzc

#endif
