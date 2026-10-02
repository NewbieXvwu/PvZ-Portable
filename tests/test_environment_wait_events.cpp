#include "EnvironmentWaitEvents.h"
#include <cassert>
#include <iostream>

using namespace EnvironmentWaitEvents;

int main()
{
	PublicState state;
	state.readyDefenses.emplace(0, 0);
	state.plantCount = 1;
	int calls = 0;
	auto result = Run(3, 0, [&] { return state; }, [&] { ++calls; ++state.tick; });
	assert(calls == 3 && result.logicSteps == 3 && result.actualTicks == 3 && result.reason == "max_ticks");

	state.tick = 0;
	calls = 0;
	result = Run(1, 1, [&] { return state; }, [&] {
		++calls; ++state.tick; ++state.wave; state.terminal = true;
		state.readyDefenses.clear(); state.leftZoneZombies.insert(-2147483647);
	});
	assert(calls == 1 && result.actualTicks == 1 && result.reason == "terminal");
	assert((result.triggered == std::vector<std::string>{"terminal", "defense_lost", "zombie_entered_left_zone", "condition", "max_ticks"}));

	state.terminal = false;
	calls = 0;
	result = Run(300, 4, [&] { return state; }, [&] { ++calls; });
	assert(calls == 0 && result.initialConditionSatisfied && result.actualTicks == 0 && result.reason == "condition");

	state.leftZoneZombies.clear();
	result = Run(3, 0, [&] { return state; }, [] {});
	assert(result.logicSteps == 3 && result.actualTicks == 0 && result.stalledClockSteps == 3);

	state.readyPackets.emplace(0, 1, -1);
	result = Run(3, 2, [&] { return state; }, [&] { ++state.tick; });
	assert(result.reason == "max_ticks");
	result = Run(3, 2, [&] { return state; }, [&] { ++state.tick; state.readyPackets.emplace(1, 19, -1); });
	assert(result.reason == "condition" && result.logicSteps == 1);

	state.wave = 0;
	result = Run(3, 1, [&] { return state; }, [&] { ++state.tick; state.wave = 1; });
	assert(result.logicSteps == 1 && result.reason == "condition");
	result = Run(1, 1, [&] { return state; }, [&] { ++state.tick; state.wave = 2; });
	assert((result.triggered == std::vector<std::string>{"condition", "max_ticks"}));

	assert(RenderedPublicCoordinate(160.0004) == 160.0);
	assert(RenderedPublicCoordinate(160.0006) > LeftZoneMaxX);
	bool rejected = false;
	try { Run(-1, 0, [&] { return state; }, [] {}); }
	catch (const std::invalid_argument&) { rejected = true; }
	assert(rejected);
	rejected = false;
	try { Run(1, 0, [&] { return state; }, [&] { --state.tick; }); }
	catch (const std::runtime_error&) { rejected = true; }
	assert(rejected);
	std::cout << "native wait reducer: 11 contract scenarios passed\n";
}
