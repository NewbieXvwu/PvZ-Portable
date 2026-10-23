#include "EnvironmentWaitEvents.h"
#include <cassert>
#include <iostream>

using namespace EnvironmentWaitEvents;

int main()
{
	PublicState state;
	SortedInsert(state.readyDefenses, std::make_pair(0, 0));
	state.plantCount = 1;
	int calls = 0;
	auto result = Run(3, 0, [&](PublicState& out) { out = state; }, [&] { ++calls; ++state.tick; });
	assert(calls == 3 && result.logicSteps == 3 && result.actualTicks == 3 && result.reason == "max_ticks");

	state.tick = 0;
	calls = 0;
	result = Run(1, 1, [&](PublicState& out) { out = state; }, [&] {
		++calls; ++state.tick; ++state.wave; state.terminal = true;
		state.readyDefenses.clear(); SortedInsert(state.leftZoneZombies, -2147483647);
	});
	assert(calls == 1 && result.actualTicks == 1 && result.reason == "terminal");
	assert((result.triggered == std::vector<std::string>{"terminal", "defense_lost", "zombie_entered_left_zone", "condition", "max_ticks"}));

	state.terminal = false;
	calls = 0;
	result = Run(300, 4, [&](PublicState& out) { out = state; }, [&] { ++calls; });
	assert(calls == 0 && result.initialConditionSatisfied && result.actualTicks == 0 && result.reason == "condition");

	state.leftZoneZombies.clear();
	result = Run(3, 0, [&](PublicState& out) { out = state; }, [] {});
	assert(result.logicSteps == 3 && result.actualTicks == 0 && result.stalledClockSteps == 3);

	SortedInsert(state.readyPackets, std::make_tuple(0, 1, -1));
	result = Run(3, 2, [&](PublicState& out) { out = state; }, [&] { ++state.tick; });
	assert(result.reason == "max_ticks");
	result = Run(3, 2, [&](PublicState& out) { out = state; }, [&] { ++state.tick; SortedInsert(state.readyPackets, std::make_tuple(1, 19, -1)); });
	assert(result.reason == "condition" && result.logicSteps == 1);

	state.wave = 0;
	result = Run(3, 1, [&](PublicState& out) { out = state; }, [&] { ++state.tick; state.wave = 1; });
	assert(result.logicSteps == 1 && result.reason == "condition");
	result = Run(1, 1, [&](PublicState& out) { out = state; }, [&] { ++state.tick; state.wave = 2; });
	assert((result.triggered == std::vector<std::string>{"condition", "max_ticks"}));

	assert(RenderedPublicCoordinate(160.0004) == 160.0);
	assert(RenderedPublicCoordinate(160.0006) > LeftZoneMaxX);
	bool rejected = false;
	try { Run(-1, 0, [&](PublicState& out) { out = state; }, [] {}); }
	catch (const std::invalid_argument&) { rejected = true; }
	assert(rejected);
	rejected = false;
	try { Run(1, 0, [&](PublicState& out) { out = state; }, [&] { --state.tick; }); }
	catch (const std::runtime_error&) { rejected = true; }
	assert(rejected);
	// Mirror EnvironmentPublicWaitState's in-place refill, including reuse after
	// two reads. Assignment-only readers above conceal stale scalar counters.
	int tick = 0;
	result = Run(5, 5, [&](PublicState& out) {
		out.clear();
		out.tick = tick;
		for (int plant = 0; plant < (tick < 2 ? 3 : 2); ++plant) ++out.plantCount;
	}, [&] { ++tick; });
	assert(result.reason == "condition" && result.logicSteps == 2);
	std::cout << "native wait reducer: 12 contract scenarios passed\n";
}
