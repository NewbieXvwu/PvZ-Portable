#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>
#include <vector>

class LawnApp;

inline constexpr int ObservationTokenFeatures = 32;
inline constexpr int ObservationTokenFields = 6 + ObservationTokenFeatures;
inline constexpr int ObservationCellStart = 2;
inline constexpr int ObservationCardStart = 56;
inline constexpr int ObservationTokenPrefix = 66;

// obs v3: fixed global/profile/cells[54]/packets[10], then variable public entities.
// Category/variant are type+1 (0 = absent); row/col/target use -1 for unknown.
struct EnvironmentToken
{
	int kind, category, variant, row, col;
	int target = -1;
	std::array<float, ObservationTokenFeatures> features{};
};

std::vector<EnvironmentToken> PublicObservationTokens(LawnApp* app);
void PublicObservationTokens(LawnApp* app, std::vector<EnvironmentToken>& tokens);
// capacity=0 keeps the exact count. A fixed-shape consumer must never truncate entities.
std::vector<float> ObservationTokenArray(const std::vector<EnvironmentToken>& tokens, size_t capacity = 0);
void FillObservationTokenArray(const std::vector<EnvironmentToken>& tokens, float* values, size_t capacity);
void WriteObservationTokens(std::ostream& out, LawnApp* app);
