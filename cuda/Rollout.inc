// One rollout: the ROLLOUT request's loop, from the same source the device kernel compiles.
//
// This is src/main.cpp:422-481 transcribed step for step (cuda/spec/protocol.md §1 lists the line
// numbers it depends on).  The order is the semantics: observe the clock, ask the student, apply the
// action, then bank the step's sun production into the income window and only then test the horizon.
// Planting advances no clock at all, which is why a continuation of 2500 ticks lands on 2520 with 47
// decisions rather than on a multiple of any fixed cadence.
//
// The loop is cut into a head and a tail so that the trunk of the student network can run over the
// whole batch at once.  RoundPre is everything up to the point where the batched GEMMs take over,
// RoundApply is the tail, and RoundPost is the per-lane half of the network plus the tail; RunRollout
// strings the same pieces together in one thread, which is what makes the host build the reference the
// device is checked against rather than a second implementation of the same idea.
#ifndef PVZ_ROLLOUT_INC
#define PVZ_ROLLOUT_INC

#include "NetCore.h"
#include "SimCore.h"

namespace pvzc
{

// EnvironmentWait(60): the rejected-action path.  Unlike the student's wait it has no early exit --
// main.cpp:469 calls the plain fixed wait, so a rejected action costs exactly 60 ticks (or the rest of
// the game, whichever ends first).
PVZ_SIM void WaitFixed(Lane lane, const LevelInfo& level, const WaveTable& waves, Rand& rand, int32_t ticks)
{
	for (int32_t step = 0; step < ticks; ++step)
	{
		if (lane.s(S_RESULT) != 0) return;
		UpdateTick(lane, level, waves, rand);
	}
}

// The prelude: one independent future per rollout, and the counters the leaf reports.
PVZ_SIM void RolloutBegin(Lane lane, int32_t futureSeed, int32_t maxActions)
{
	Rand rand;
	// RESEED: an independent future per rollout, seeded from the request's counter.  The original's own
	// stream is not reproducible here (the brief does not ask for it); what matters is that every draw
	// comes from the same distribution the original draws from.
	rand.Seed(lane.s(S_TICK), futureSeed);
	StoreRandState(lane, rand);
	lane.s(S_DECISIONS) = 0;
	lane.s(S_NET_CALLS) = 0;
	lane.s(S_ROUNDS) = 0;
	lane.s(S_PRIOR_TICK) = -1;
	lane.s(S_PRIOR_PLANT) = 0;
	lane.s(S_MAX_ACTIONS) = maxActions;
	lane.s(S_STATE) = LANE_RUNNING;
	// main.cpp:425: an `until` the position has already passed means "play to the end".  Idempotent, so
	// the device can leave the value alone for the rounds that follow.
	if (lane.s(S_UNTIL_TICK) <= lane.s(S_TICK)) lane.s(S_UNTIL_TICK) = 0x7FFFFFFF;
}

// The loop top: the two tests of the original's while condition, then the observation the student is
// asked about -- the tick it is asked at and the sun banked so far, which the tail reads back after
// the batched trunk.  Returns 0 when the lane is over, having marked it LANE_DONE.
PVZ_SIM int32_t RoundTop(Lane lane)
{
	if (lane.s(S_RESULT) != 0 || lane.s(S_DECISIONS) >= lane.s(S_MAX_ACTIONS))
	{
		lane.s(S_STATE) = LANE_DONE;
		return 0;
	}
	lane.s(S_ROUND_TICK) = lane.s(S_TICK);
	lane.s(S_ROUND_PRODUCED) = lane.s(S_SUN_PRODUCED);
	return 1;
}

// The tail of a round, shared by both paths: apply, bank the step's production, count, test the
// horizon.  `S_PRIOR_TICK` is the clock the action was *observed* at (main.cpp:471), not the one it
// landed on, which is why RoundTop stashed it before the wait advanced anything.
PVZ_SIM void RoundApply(Lane lane, const LevelInfo& level, const WaveTable& waves, Rand& rand,
	const Choice& choice)
{
	const int32_t tick = lane.s(S_ROUND_TICK);
	const int32_t producedBefore = lane.s(S_ROUND_PRODUCED);

	int32_t accepted = 1;
	int32_t priorPlant = 0;
	if (choice.packet >= 0)
	{
		priorPlant = ActionPlant(lane, level, choice.packet, choice.row, choice.col, rand);
		accepted = priorPlant;
	}
	else WaitRound(lane, level, waves, rand, 60);
	if (!accepted)
	{
		WaitFixed(lane, level, waves, rand, 60);
		priorPlant = 0;
	}

	lane.s(S_PRIOR_PLANT) = priorPlant;
	lane.s(S_PRIOR_TICK) = tick;
	const int32_t now = lane.s(S_TICK);
	PushIncomeEvent(lane, now, lane.s(S_SUN_PRODUCED) - producedBefore);
	lane.s(S_DECISIONS) = lane.s(S_DECISIONS) + 1;
	if (choice.forwarded) lane.s(S_NET_CALLS) = lane.s(S_NET_CALLS) + 1;
	lane.s(S_ROUNDS) = lane.s(S_ROUNDS) + 1;
	if (now >= lane.s(S_UNTIL_TICK)) lane.s(S_STATE) = LANE_DONE;
}

enum { PRE_DONE = 0, PRE_WAITED = 1, PRE_NEEDS_NET = 2 };

// Everything before the batched trunk takes over.  A lane with no legal plant anywhere does not consult
// the network at all (EnvironmentRollout.cpp:269), so its entire round -- the wait -- runs here and the
// GEMMs never see it; otherwise the caller compacts the lane into an activation slot, builds its
// features and comes back through RoundPost.
PVZ_SIM int32_t RoundPre(Lane lane, const LevelInfo& level, const WaveTable& waves, Rand& rand)
{
	if (!RoundTop(lane)) return PRE_DONE;
	if (AnyLegalPlant(lane, level)) return PRE_NEEDS_NET;
	const Choice wait = {-1, 0, 0, 0};
	RoundApply(lane, level, waves, rand, wait);
	return PRE_WAITED;
}

// The other half of the same round: the per-lane part of the network, then the tail.  Called with the
// activation slot the trunk wrote this lane's context into.  The cell head's 1x1 convolution is not here
// -- the caller ran it over the whole batch between the trunk and this kernel, into `scratch`, which is
// why this path splits NetChoose instead of calling it.
PVZ_SIM void RoundPost(Lane lane, const LevelInfo& level, const WaveTable& waves, const NetParams& net,
	const NetState& state, Rand& rand)
{
	NetContext(lane, net, state);
	Choice choice = NetPickPacket(lane, level, net, state);
	if (choice.packet >= 0)
	{
		CellLogits(lane, net, state, choice.packet);
		choice = NetPickCell(lane, level, state, choice);
	}
	RoundApply(lane, level, waves, rand, choice);
}

PVZ_SIM void RunRollout(Lane lane, const LevelInfo& level, const WaveTable& waves, const NetParams& net,
	const NetState& state, const float* waveLog, int32_t futureSeed, int32_t maxActions, int32_t useNet)
{
	RolloutBegin(lane, futureSeed, maxActions);
	Rand rand = LoadRandState(lane);
	while (lane.s(S_STATE) != LANE_DONE)
	{
		if (!RoundTop(lane)) break;
		// useNet = 0 measures the tick loop alone: the same wait rounds, the same decision count, no
		// forward at all.  Benchmark scaffolding, not a mode the search ever runs.
		const Choice choice = useNet ? NetDecide(lane, level, net, state, waveLog)
			: Choice{-1, 0, 0, 0};
		RoundApply(lane, level, waves, rand, choice);
	}
	StoreRandState(lane, rand);
}

}  // namespace pvzc

#endif
