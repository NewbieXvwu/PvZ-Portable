#ifndef PVZ_ENVIRONMENT_VALUE_NET_H
#define PVZ_ENVIRONMENT_VALUE_NET_H

#include <string>
#include <array>
#include <cstdint>
#include <vector>
class LawnApp;

// PVZVN001: little-endian uint32 width, encoder version (1/2/3/4/5/6), context dimension (0/35),
// then contiguous float32 parameters in PyTorch state_dict order. No JSON on the Board path.
// Encoder 6 is value-net v6: the full v5 board/scalars stay a prefix, then 35 internal-state
// channels and 6 wave/timing scalars are appended (see InternalFeatures below).
class EnvironmentValueNet
{
public:
	static constexpr int Channels = 92, Cells = 60, Scalars = 113, TaskContext = 35;
	static constexpr int V5ExtraChannels = 12, V6ExtraChannels = 35, V6ExtraScalars = 6;
	void Load(const std::string& path);
	bool Loaded() const { return width != 0; }
	int ScalarCount() const { return Scalars + (encoder >= 2 ? 6 : 0) + (encoder >= 6 ? V6ExtraScalars : 0) + context; }
	int ChannelCount() const
	{
		if (encoder >= 6) return Channels + V5ExtraChannels + V6ExtraChannels;
		if (encoder == 5) return Channels + V5ExtraChannels;
		if (encoder == 4) return Channels + 8;
		if (encoder == 3) return Channels + 6;
		return Channels;
	}
	void Features(LawnApp* app, double income, std::vector<float>& board, std::vector<float>& scalars) const;
	float Logit(const std::vector<float>& board, const std::vector<float>& scalars);
	float Predict(LawnApp* app, double income);
	static float Probability(float logit);
	double featureSeconds = 0, forwardSeconds = 0;
	std::array<double, 3> convolutionSeconds{};
	uint64_t calls = 0;
private:
	int width = 0, encoder = 0, context = 0;
	std::vector<std::vector<float>> weights;
	std::vector<float> board, scalars, first, second, columns, joined, scalarHidden, hidden, kernelBuffer;
	std::vector<int> activeChannels;
	void Convolve(const std::vector<float>& input, int channels, int kernelRows, int layer,
		std::vector<float>& output);
};
#endif
