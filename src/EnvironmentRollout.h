#ifndef PVZ_ENVIRONMENT_ROLLOUT_H
#define PVZ_ENVIRONMENT_ROLLOUT_H

#include <array>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include "GenericForward.h"
#include "EnvironmentTokens.h"

class LawnApp;

class EnvironmentRollout
{
public:
	static constexpr int Channels = 92, Cells = 60, Scalars = 115, ZombieTypes = 34, TaskContext = 1 + ZombieTypes;
	struct Action { int packet = -1, col = 0, row = 0; };
	struct Prediction
	{
		float plant;
		std::array<float, 10> packets;
		std::array<float, 54> cells;
		int cellPacket = -1;
		Action action;
	};
	// v2lite's student-visible history: the first tick each (row, col, kind) was observed at a
	// decision, and the tick the current wave started.  It is advanced by Features, so the object
	// must live for the whole continuation, exactly like the Python StudentHistory it mirrors.
	struct History
	{
		std::map<std::tuple<int, int, int>, int> plants;
		int wave = -1, waveTick = 0;
		void Update(LawnApp* app);
	};
	void Load(const std::string& path);
	static void BungeeFeatures(LawnApp* app, float* board);
	static void SleepFeatures(LawnApp* app, float* board);
	static void TimerFeatures(LawnApp* app, float* board);
	bool Loaded() const { return width != 0; }
	bool HasHistory() const { return v2lite || v2; }
	void Features(LawnApp* app, double income, int previousTick, bool previousPlant,
		std::vector<float>& board, std::vector<float>& scalars, History& history);
	Prediction Predict(LawnApp* app, double income, int previousTick, bool previousPlant,
		History& history, bool diagnostics = false);
	// Number of Predict calls that actually ran the network (the ones that return early because
	// nothing is plantable cost nothing).  Reset by the caller around a batch of rollouts.
	int NetworkCalls() const { return networkCalls; }
	void ResetNetworkCalls() { networkCalls = 0; }
private:
	pvz::net::Graph graph;
	bool generic = false;
	int graphBoard = -1, graphScalars = -1, graphInfo = -1, graphSelected = -1, graphTerrain = -1;
	int graphPlant = -1, graphPacket = -1, graphCell = -1;
	int graphTokens = -1;
	int graphMeta = -1, graphValid = -1, graphPool = -1, graphCoords = -1, graphCardIndex = -1, graphCellIndex = -1;
	std::vector<EnvironmentToken> publicTokens;
	void PrepareInputs(LawnApp* app, double income, int previousTick, bool previousPlant, History& history);
	void LoadGraph(const std::string& path);
	void SetObservation(int observation, bool withContext);
	int width = 0;
	int plantEmbedding = 0;
	bool identityPacket = false;
	std::array<std::array<float, 4>, 10> packetInfo{};
	bool meanmax = false;
	bool rowPool = false;
	bool taskContext = false;
	bool v2lite = false;
	bool v2 = false;
	bool fogChannel = false;
	bool sizeChannel = false;
	bool bossChannel = false;
	bool bungeeChannel = false;
	bool sleepChannel = false;
	// task147: v2timer (observation 9) = v2sleep + visible action timers and the seed belt.
	bool timerChannel = false;
	int observationChannels = 0;
	int networkCalls = 0;
	std::vector<std::vector<float>> weights;
	// Every buffer the forward pass needs is allocated once in Load(): a rollout runs the
	// network on every tick, so a per-call std::vector allocation or a full-buffer zero fill is
	// a measurable share of the whole simulation (2026-10-23: 424 KB of kernel transposition was
	// being rebuilt per forward).
	std::vector<float> hidden, context, scratch, columns, kernelBuffer, rowPoolColumns;
	std::vector<float> featureBoard, featureScalars, packetInput, packetActivation;
	std::vector<double> packetWeightDouble, packetMatrix, packetOutput;
	// The sparse first layer only changes its active-channel set when a plant or zombie appears
	// or disappears; while it stays the same, the transposed kernel and the zero border of the
	// im2col buffer are still valid, so they are not rebuilt.
	std::vector<int> activeChannels, kernelChannels;
	int columnsCount = -1;
	// Predict() calls Heads(-1) and then Heads(packet) on the same context; the packet logits do
	// not depend on the cell argument, so the second call reuses them.
	std::array<float, 10> packetLogits{};
	bool packetLogitsValid = false;
	void Encode(const std::vector<float>& board, const std::vector<float>& scalars);
	Prediction Heads(int packet);
	void Convolve(const std::vector<float>& input, int channels, int kernelRows, int kernelCols,
		int layer, std::vector<float>& output, bool sparse);
};

#endif
