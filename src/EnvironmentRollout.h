#ifndef PVZ_ENVIRONMENT_ROLLOUT_H
#define PVZ_ENVIRONMENT_ROLLOUT_H

#include <array>
#include <string>
#include <vector>

class LawnApp;

class EnvironmentRollout
{
public:
	static constexpr int Channels = 92, Cells = 60, Scalars = 115;
	struct Action { int packet = -1, col = 0, row = 0; };
	struct Prediction
	{
		float plant;
		std::array<float, 10> packets;
		std::array<float, 54> cells;
		int cellPacket = -1;
		Action action;
	};
	void Load(const std::string& path);
	bool Loaded() const { return width != 0; }
	void Features(LawnApp* app, double income, int previousTick, bool previousPlant,
		std::vector<float>& board, std::vector<float>& scalars);
	Prediction Predict(LawnApp* app, double income, int previousTick, bool previousPlant, bool diagnostics = false);

private:
	int width = 0;
	bool meanmax = false;
	std::vector<std::vector<float>> weights;
	std::vector<float> hidden, context, scratch, columns, kernelBuffer;
	void Encode(const std::vector<float>& board, const std::vector<float>& scalars);
	Prediction Heads(int packet);
	void Convolve(const std::vector<float>& input, int channels, int kernelRows, int kernelCols,
		int layer, std::vector<float>& output, bool sparse);
};

#endif
