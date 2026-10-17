// Single-lane host build of the re-implemented core: one lane, stride 1, the SoA in a heap block.
//
// Its purpose is diagnosis.  The same SimImpl.inc / SimTick.inc the device runs is compiled here by
// plain g++, so an out-of-bounds slot index, a wrong loop bound or a bad constant shows up under
// -fsanitize=address with a source line, without a GPU round trip and without nvcc's inlining making
// the attribution ambiguous.  It is also the reference the phase-1 agreement test compares the GPU
// against when a disagreement must be split into "GPU bug" vs "mechanics bug".
//
// build: g++ -fsanitize=address -g -O1 -std=c++17 -I. -o build/sim_host cuda/SimHost.cpp
#include "cuda/SimTick.inc"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace pvzhost
{

struct Corpus
{
	pvzc::LevelInfo level{};
	std::vector<pvzc::RootRecord> roots;
};

Corpus Read(const char* path)
{
	FILE* file = std::fopen(path, "rb");
	if (!file) { std::perror("open"); std::exit(2); }
	std::fseek(file, 0, SEEK_END);
	const long size = std::ftell(file);
	std::fseek(file, 0, SEEK_SET);
	std::vector<char> raw(static_cast<size_t>(size));
	if (std::fread(raw.data(), 1, raw.size(), file) != raw.size()) { std::fprintf(stderr, "short\n"); std::exit(2); }
	std::fclose(file);
	Corpus corpus;
	size_t off = 0;
	while (off + 12 <= raw.size())
	{
		int32_t magic = 0, version = 0, kind = 0;
		std::memcpy(&magic, raw.data() + off, 4);
		std::memcpy(&version, raw.data() + off + 4, 4);
		std::memcpy(&kind, raw.data() + off + 8, 4);
		if (magic != pvzc::kMagic || version != pvzc::kVersion) { std::fprintf(stderr, "bad tag\n"); std::exit(2); }
		off += 12;
		if (kind == 1) { std::memcpy(&corpus.level, raw.data() + off, sizeof(pvzc::LevelInfo)); off += sizeof(pvzc::LevelInfo); continue; }
		if (kind != 2) { std::fprintf(stderr, "unknown kind %d\n", kind); std::exit(2); }
		pvzc::RootRecord root;
		std::memcpy(&root, raw.data() + off, sizeof(pvzc::RootRecord));
		off += sizeof(pvzc::RootRecord);
		if (root.supported) corpus.roots.push_back(root);
	}
	return corpus;
}

pvzc::WaveTable WaveOf(const pvzc::RootRecord& root)
{
	pvzc::WaveTable table{};
	table.waveCount = root.waveCount;
	for (int32_t wave = 0; wave < pvzc::kMaxWaves; ++wave)
	{
		table.zombieCount[wave] = root.waveZombieCount[wave];
		for (int32_t slot = 0; slot < pvzc::kWaveZombieSlots; ++slot)
			table.type[wave][slot] = root.waveTable[wave][slot];
	}
	return table;
}

}  // namespace pvzhost

static int32_t Live(pvzc::Lane lane, pvzc::IntField typeField, int32_t slots)
{
	int32_t count = 0;
	for (int32_t slot = 0; slot < slots; ++slot)
		if (lane.i(typeField, slot) >= 0) ++count;
	return count;
}

int main(int argc, char** argv)
{
	const char* path = argc > 1 ? argv[1] : "artifacts/cuda/corpus_smoke/roots.cudaroot";
	const int32_t ticks = argc > 2 ? std::atoi(argv[2]) : 300;
	pvzhost::Corpus corpus = pvzhost::Read(path);
	std::printf("host core: %zu supported roots, IntSlotsTotal %d, FloatSlotsTotal %d\n",
		corpus.roots.size(), pvzc::IntSlotsTotal(), pvzc::FloatSlotsTotal());

	// ASan needs red zones around the SoA, so the arrays are heap allocated at exactly their declared
	// size rather than padded: an out-of-range slot index then lands in a red zone.
	int32_t* ints = static_cast<int32_t*>(std::calloc(pvzc::IntSlotsTotal(), sizeof(int32_t)));
	float* floats = static_cast<float*>(std::calloc(pvzc::FloatSlotsTotal(), sizeof(float)));

	for (size_t index = 0; index < corpus.roots.size(); ++index)
	{
		const pvzc::RootRecord& root = corpus.roots[index];
		const pvzc::WaveTable waves = pvzhost::WaveOf(root);
		std::memset(ints, 0, static_cast<size_t>(pvzc::IntSlotsTotal()) * sizeof(int32_t));
		std::memset(floats, 0, static_cast<size_t>(pvzc::FloatSlotsTotal()) * sizeof(float));
		pvzc::Lane lane{ints, floats, 0, 1};
		if (!pvzc::LoadRoot(lane, root, corpus.level))
		{
			std::printf("root %zu rejected by LoadRoot\n", index);
			continue;
		}
		lane.s(pvzc::S_ROOT_INDEX) = static_cast<int32_t>(index);
		pvzc::Rand rand;
		rand.Seed(1600000 + static_cast<int32_t>(index), 7);
		for (int32_t step = 0; step < ticks && lane.s(pvzc::S_RESULT) == 0; ++step)
			pvzc::UpdateTick(lane, corpus.level, waves, rand);
		std::printf("root %zu tick %d -> %d plants %d zombies %d result %d errors %d\n", index,
			root.tick, lane.s(pvzc::S_TICK), Live(lane, pvzc::F_PLANT_TYPE, pvzc::kPlantSlots),
			Live(lane, pvzc::F_ZOMBIE_TYPE, pvzc::kZombieSlots),
			lane.s(pvzc::S_RESULT), lane.s(pvzc::S_ERROR));
	}
	std::free(ints);
	std::free(floats);
	return 0;
}
