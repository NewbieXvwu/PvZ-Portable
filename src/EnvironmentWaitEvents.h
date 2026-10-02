#ifndef PVZ_ENVIRONMENT_WAIT_EVENTS_H
#define PVZ_ENVIRONMENT_WAIT_EVENTS_H

#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// Every member is either an existing public observation or command bookkeeping.
// This controller never chooses a plant, cell, shovel, or tactical goal.
namespace EnvironmentWaitEvents
{
inline constexpr int Version = 1;
inline constexpr double LeftZoneMaxX = 160.0;
inline constexpr const char* Conditions[] = {
	"timeout", "wave_changed", "packet_became_ready", "sun_increased",
	"left_zone_occupied", "plant_count_decreased"
};

struct PublicState
{
	int tick = 0;
	int wave = 0;
	int sun = 0;
	int plantCount = 0;
	bool terminal = false;
	std::set<std::tuple<int, int, int>> readyPackets; // public index/type/imitater
	std::set<std::pair<int, int>> readyDefenses; // public row/type, state == READY(1)
	std::set<int> leftZoneZombies; // signed public IDs; on_board and rendered x <= 160
};

struct Result
{
	int requestedTicks = 0;
	int actualTicks = 0;
	int logicSteps = 0;
	int stalledClockSteps = 0;
	int condition = 0;
	bool initialConditionSatisfied = false;
	std::string reason;
	std::vector<std::string> triggered;
};

inline double RenderedPublicCoordinate(double value)
{
	// EnvironmentObservation uses a fresh stream's default six significant digits.
	// Comparing an unrounded private coordinate could disagree at the boundary.
	std::ostringstream text;
	text << value;
	return std::stod(text.str());
}

template<class Set> bool HasNew(const Set& before, const Set& after)
{
	for (const auto& item : after)
		if (!before.contains(item)) return true;
	return false;
}

inline std::vector<std::string> Triggered(const PublicState& before,
	const PublicState& after, int condition, bool deadline)
{
	std::vector<std::string> result;
	if (after.terminal) result.emplace_back("terminal");
	if (HasNew(after.readyDefenses, before.readyDefenses)) result.emplace_back("defense_lost");
	if (HasNew(before.leftZoneZombies, after.leftZoneZombies)) result.emplace_back("zombie_entered_left_zone");
	const bool selected =
		(condition == 1 && after.wave != before.wave) ||
		(condition == 2 && HasNew(before.readyPackets, after.readyPackets)) ||
		(condition == 3 && after.sun > before.sun) ||
		(condition == 4 && !after.leftZoneZombies.empty()) ||
		(condition == 5 && after.plantCount < before.plantCount);
	if (selected) result.emplace_back("condition");
	if (deadline) result.emplace_back("max_ticks");
	return result; // simultaneous precedence is the order above; retain all reasons
}

template<class ReadPublic, class Advance>
Result Run(int requestedTicks, int condition, ReadPublic readPublic, Advance advance)
{
	if (requestedTicks < 0 || requestedTicks > 1000000 || condition < 0 || condition >= 6)
		throw std::invalid_argument("invalid event wait request");
	Result result;
	result.requestedTicks = requestedTicks;
	result.condition = condition;
	const PublicState initial = readPublic();
	PublicState previous = initial;
	result.triggered = Triggered(initial, initial, condition, requestedTicks == 0);
	result.initialConditionSatisfied = condition == 4 && !initial.leftZoneZombies.empty();
	while (result.triggered.empty() && result.logicSteps < requestedTicks)
	{
		advance(); // exactly the same one-tick operation as fixed WAIT
		++result.logicSteps;
		PublicState current = readPublic();
		if (current.tick < previous.tick)
			throw std::runtime_error("event wait public clock moved backwards");
		result.stalledClockSteps += current.tick == previous.tick;
		result.actualTicks = current.tick - initial.tick;
		result.triggered = Triggered(previous, current, condition, result.logicSteps == requestedTicks);
		previous = std::move(current);
	}
	result.reason = result.triggered.front();
	return result;
}
}
#endif
