// Isolated task35 experiment: default six-digit public coordinate round-trip.
#include "EnvironmentWaitEvents.h"
#include <bit>
#include <charconv>
#include <ctime>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

static double Reference(double value)
{
	std::ostringstream text;
	text << value;
	return std::stod(text.str());
}

int main()
{
	std::vector<double> values{0.0, -0.0, 160.0004, 160.0006};
	// Gameplay coordinates and velocities are float, promoted to double by the
	// public encoder. Cover arbitrary finite float patterns and cell boundaries.
	std::mt19937 random(35);
	for (int i = 0; i < 1000000; ++i)
	{
		const float value = std::bit_cast<float>(static_cast<uint32_t>(random()));
		if (std::isfinite(value)) values.push_back(value);
	}
	for (int boundary = -10; boundary <= 1000; ++boundary)
		for (int offset = -10; offset <= 10; ++offset)
			values.push_back(static_cast<float>(boundary + offset * 0.0001));
	for (const double value : values)
		if (std::bit_cast<uint64_t>(EnvironmentWaitEvents::RenderedPublicCoordinate(value)) !=
			std::bit_cast<uint64_t>(Reference(value)))
		{
			std::cerr << "mismatch at " << std::hexfloat << value << '\n';
			return 1;
		}
	std::cout << "identical finite float round-trips: " << values.size() << '\n';
	volatile double sink = 0;
	// Time ordinary lawn coordinates, rather than extreme exponents.
	for (double& value : values) value = static_cast<float>((random() % 20000000) * 0.0001 - 100);
	for (int round = 0; round < 3; ++round)
		for (int arm = 0; arm < 2; ++arm)
		{
			const bool candidate = (arm + round) % 2;
			const auto start = std::clock();
			double total = 0;
			for (const double value : values) total += candidate ?
				EnvironmentWaitEvents::RenderedPublicCoordinate(value) : Reference(value);
			sink = total;
			const double seconds = static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;
			std::cout << (candidate ? "chars" : "stream") << " round=" << round << " seconds=" << seconds << '\n';
		}
}
