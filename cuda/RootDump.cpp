// Dump the fields a lane reads when importing a root, so a device-side out-of-bounds write can be
// traced to the record that causes it without a GPU round trip.
#include "cuda/CudaRollout.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace pvzc;

int main(int argc, char** argv)
{
	if (argc < 2) { std::fprintf(stderr, "usage: %s FILE.cudaroot\n", argv[0]); return 2; }
	FILE* file = std::fopen(argv[1], "rb");
	if (!file) { std::perror("open"); return 2; }
	std::fseek(file, 0, SEEK_END);
	const long size = std::ftell(file);
	std::fseek(file, 0, SEEK_SET);
	std::vector<char> raw(static_cast<size_t>(size));
	if (std::fread(raw.data(), 1, static_cast<size_t>(size), file) != static_cast<size_t>(size))
		{ std::fprintf(stderr, "short read\n"); return 2; }
	std::fclose(file);

	std::printf("sizeof(RootRecord)=%zu sizeof(LevelInfo)=%zu sizeof(WaveTable)=%zu IntSlotsTotal=%d "
		"FloatSlotsTotal=%d scalarCount=%d\n", sizeof(RootRecord), sizeof(LevelInfo), sizeof(WaveTable),
		IntSlotsTotal(), FloatSlotsTotal(), static_cast<int>(F_SCALAR));
	size_t off = 0;
	int kind = 0;
	while (off + 12 <= raw.size())
	{
		int32_t magic = 0, version = 0, tag = 0;
		std::memcpy(&magic, raw.data() + off, 4);
		std::memcpy(&version, raw.data() + off + 4, 4);
		std::memcpy(&tag, raw.data() + off + 8, 4);
		if (magic != kMagic || version != kVersion) { std::fprintf(stderr, "bad tag\n"); return 2; }
		off += 12;
		if (tag == 1) { off += sizeof(LevelInfo); ++kind; continue; }
		if (tag != 2) { std::fprintf(stderr, "unknown kind %d\n", tag); return 2; }
		RootRecord root;
		std::memcpy(&root, raw.data() + off, sizeof(RootRecord));
		off += sizeof(RootRecord);
		std::printf("root %2d tick %6d sun %4d incomeCount %4d head %4d plants %3d zombies %3d proj %3d "
			"coins %3d mowers %2d packets %2d waveCount %3d supported %d reason %d\n",
			kind - 1, root.tick, root.sun, root.incomeCount, root.incomeHead, root.plantCount,
			root.zombieCount, root.projectileCount, root.coinCount, root.mowerCount, root.packetCount,
			root.waveCount, root.supported, root.unsupportedReason);
		for (int32_t index = 0; index < root.plantCount; ++index)
			if (root.plants[index].cell < 0 || root.plants[index].cell >= kPlantSlots)
				std::printf("    plant %d cell %d type %d OUT OF RANGE\n", index, root.plants[index].cell,
					root.plants[index].type);
		for (int32_t wave = 0; wave < root.waveCount && wave < kMaxWaves; ++wave)
			if (root.waveZombieCount[wave] > kWaveZombieSlots)
				std::printf("    wave %d count %d OVER TABLE\n", wave, root.waveZombieCount[wave]);
		++kind;
	}
	return 0;
}
