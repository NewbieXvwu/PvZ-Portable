#ifndef PVZ_ENVIRONMENT_WAIT_EVENTS_H
#define PVZ_ENVIRONMENT_WAIT_EVENTS_H

#include <algorithm>
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

// Sorted, duplicate-free sequences.  The public state is rebuilt on every tick of an event
// wait, and `WAIT_UNTIL_LEGAL` runs up to sixty of those per decision, so a std::set paid a
// malloc per element plus a red-black rebalance for a handful of entries.  A sorted vector
// keeps the same iteration order, membership tests and duplicate suppression, and the
// buffers are reused across ticks (2026-10-23: 7% of the rollout CPU sat in this path and
// the allocator behind it).
template <class T> void SortedInsert(std::vector<T>& theItems, const T& theItem)
{
	auto anIterator = std::lower_bound(theItems.begin(), theItems.end(), theItem);
	if (anIterator == theItems.end() || *anIterator != theItem)
		theItems.insert(anIterator, theItem);
}

template <class T> bool Contains(const std::vector<T>& theItems, const T& theItem)
{
	return std::binary_search(theItems.begin(), theItems.end(), theItem);
}

struct PublicState
{
	int tick = 0;
	int wave = 0;
	int sun = 0;
	int plantCount = 0;
	bool terminal = false;
	std::vector<std::tuple<int, int, int>> readyPackets; // public index/type/imitater
	std::vector<std::pair<int, int>> readyDefenses; // public row/type, state == READY(1)
	std::vector<int> leftZoneZombies; // signed public IDs; on_board and rendered x <= 160

	void clear()
	{
		tick = wave = sun = plantCount = 0;
		terminal = false;
		readyPackets.clear();
		readyDefenses.clear();
		leftZoneZombies.clear();
	}
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

template<class Items> bool HasNew(const Items& before, const Items& after)
{
	for (const auto& item : after)
		if (!Contains(before, item)) return true;
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
	// Two buffers, filled in place and swapped: the caller clears and refills the state it
	// is handed, so the containers keep their capacity and a wait costs no allocation.
	PublicState aStates[2];
	readPublic(aStates[0]);
	const int initialTick = aStates[0].tick;
	int previousIndex = 0, currentIndex = 1;
	result.triggered = Triggered(aStates[0], aStates[0], condition, requestedTicks == 0);
	result.initialConditionSatisfied = condition == 4 && !aStates[0].leftZoneZombies.empty();
	while (result.triggered.empty() && result.logicSteps < requestedTicks)
	{
		advance(); // exactly the same one-tick operation as fixed WAIT
		++result.logicSteps;
		PublicState& current = aStates[currentIndex];
		const PublicState& previous = aStates[previousIndex];
		readPublic(current);
		if (current.tick < previous.tick)
			throw std::runtime_error("event wait public clock moved backwards");
		result.stalledClockSteps += current.tick == previous.tick;
		result.actualTicks = current.tick - initialTick;
		result.triggered = Triggered(previous, current, condition, result.logicSteps == requestedTicks);
		std::swap(previousIndex, currentIndex);
	}
	result.reason = result.triggered.front();
	return result;
}
}
#endif
