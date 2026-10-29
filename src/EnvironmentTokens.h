#pragma once

#include <array>
#include <iosfwd>
#include <vector>

class LawnApp;

// obs v3: fixed global/profile/cells[54]/packets[10], then variable public entities.
// Category/variant are type+1 (0 = absent); row/col/target use -1 for unknown.
struct EnvironmentToken
{
	int kind, category, variant, row, col;
	int target = -1;
	std::array<float, 32> features{};
};

std::vector<EnvironmentToken> PublicObservationTokens(LawnApp* app);
void WriteObservationTokens(std::ostream& out, LawnApp* app);
