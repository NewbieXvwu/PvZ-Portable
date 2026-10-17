// One rollout: the ROLLOUT request's loop, from the same source the device kernel compiles.
//
// This is src/main.cpp:422-481 transcribed step for step (cuda/spec/protocol.md §1 lists the line
// numbers it depends on).  The order is the semantics: observe the clock, ask the student, apply the
// action, then bank the step's sun production into the income window and only then test the horizon.
// Planting advances no clock at all, which is why a continuation of 2500 ticks lands on 2520 with 47
// decisions rather than on a multiple of any fixed cadence.
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

PVZ_SIM void RunRollout(Lane lane, const LevelInfo& level, const WaveTable& waves, const NetParams& net,
	const NetState& state, const float* waveLog, int32_t futureSeed, int32_t maxActions, int32_t useNet)
{
	Rand rand;
	// RESEED: an independent future per rollout, seeded from the request's counter.  The original's own
	// stream is not reproducible here (the brief does not ask for it); what matters is that every draw
	// comes from the same distribution the original draws from.
	rand.Seed(lane.s(S_TICK), futureSeed);
	lane.s(S_DECISIONS) = 0;
	lane.s(S_NET_CALLS) = 0;
	lane.s(S_PRIOR_TICK) = -1;
	lane.s(S_PRIOR_PLANT) = 0;

	int32_t until = lane.s(S_UNTIL_TICK);
	if (until <= lane.s(S_TICK)) until = 0x7FFFFFFF;   // main.cpp:425 -- "until 0 = play to the end"

	while (lane.s(S_RESULT) == 0 && lane.s(S_DECISIONS) < maxActions)
	{
		const int32_t tick = lane.s(S_TICK);
		const int32_t producedBefore = lane.s(S_SUN_PRODUCED);
		// useNet = 0 measures the tick loop alone: the same wait rounds, the same decision count, no
		// forward at all.  Benchmark scaffolding, not a mode the search ever runs.
		const Choice choice = useNet ? NetDecide(lane, level, net, state, waveLog, state.logits)
			: Choice{-1, 0, 0, 0};

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
		lane.s(S_PRIOR_TICK) = tick;                   // the clock the action was *observed* at (471)
		const int32_t now = lane.s(S_TICK);
		PushIncomeEvent(lane, now, lane.s(S_SUN_PRODUCED) - producedBefore);
		lane.s(S_DECISIONS) = lane.s(S_DECISIONS) + 1;
		if (choice.forwarded) lane.s(S_NET_CALLS) = lane.s(S_NET_CALLS) + 1;
		if (now >= until) break;
	}
	StoreRandState(lane, rand);
}

}  // namespace pvzc

#endif
