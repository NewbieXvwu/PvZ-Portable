// Small deterministic mechanism checks, using the same sources as the host and CUDA builds.
#include <cstddef>
#include <cassert>
#include <cstdio>
#include <vector>
#include "SimTick.inc"

using namespace pvzc;

struct State
{
	std::vector<int32_t> ints = std::vector<int32_t>(IntSlotsTotal());
	std::vector<float> floats = std::vector<float>(FloatSlotsTotal());
	Lane lane() { return Lane{ints.data(), floats.data(), 0, 1, 0}; }
};

LevelInfo Level()
{
	LevelInfo level{};
	level.level = 7;
	level.numWaves = 30;
	level.playableRows = 5;
	level.zombieCountMultiplier = 1.5f;
	for (int type = 0; type < 4; ++type)
	{
		auto& def = level.zombies[type];
		def.type = type;
		def.value = type < 2 ? 1 : 2;
		def.bodyHealth = type == 3 ? 500 : 270;
		def.helmHealth = type == 2 ? 370 : 0;
		def.firstAllowedWave = type == 3 ? 5 : 1;
		def.pickWeight = type == 0 ? 4000 : (type == 1 ? 0 : (type == 2 ? 4000 : 2000));
		def.velMin = type == 3 ? 0.66f : (type == 1 ? 0.45f : 0.23f);
		def.velMax = type == 3 ? 0.68f : (type == 1 ? 0.45f : 0.37f);
		level.allowed[type] = type != 1;
	}
	return level;
}

RootRecord Root()
{
	RootRecord root{};
	root.magic = kMagic;
	root.version = kVersion;
	root.supported = 1;
	root.numWaves = root.waveCount = 30;
	root.wave = 9;
	root.zombieCountDown = root.zombieCountDownStart = 1;
	for (int row = 0; row < kRows; ++row) root.rowGotMowered[row] = -100;
	return root;
}

int main()
{
	for (int root = 0; root < 425; ++root)
		for (int future = 0; future < 64; ++future)
		{
			const int global = (future / 8) * 425 * 8 + root * 8 + future % 8;
			assert(RolloutFutureSeed(global, 425, 8, 1600000000) == 1600000000 + future);
		}
	auto level = Level();
	// A real root's stored future table is privileged information.  Mutating it must not affect a
	// newly resampled future, even when the next tick emits a flag wave.
	auto root = Root();
	WaveTable first{}, second{};
	first.waveCount = second.waveCount = 30;
	for (int wave = 0; wave < 30; ++wave)
	{
		first.zombieCount[wave] = 1;
		second.zombieCount[wave] = kWaveZombieSlots;
		for (int slot = 0; slot < kWaveZombieSlots; ++slot) second.type[wave][slot] = 3;
	}
	State a, b;
	assert(LoadRoot(a.lane(), root, level) == 1);
	assert(LoadRoot(b.lane(), root, level) == 1);
	a.lane().s(S_FUTURE_SEED) = b.lane().s(S_FUTURE_SEED) = 12345;
	Rand ra, rb;
	ra.Seed(0, 12345);
	rb.Seed(0, 12345);
	UpdateWaves(a.lane(), level, first, ra);
	UpdateWaves(b.lane(), level, second, rb);
	assert(a.ints == b.ints && a.floats == b.floats && ra.state == rb.state);
	int flags = 0;
	for (int slot = 0; slot < kZombieSlots; ++slot)
		if (a.lane().i(F_ZOMBIE_TYPE, slot) == 1)
		{
			++flags;
			assert(a.lane().i(F_ZOMBIE_BODY, slot) == 270);
			assert(a.lane().f(F_ZOMBIE_VELX, slot) == 0.45f);
		}
	assert(flags >= 1);

	bool changes = false;
	for (int wave = 0; wave < 30; ++wave)
	{
		int reference[kWaveZombieSlots]{};
		const int nr = DrawFutureWave(level, wave, 1, reference);
		for (int seed = 1; seed <= 256; ++seed)
		{
			int types[kWaveZombieSlots]{};
			const int n = DrawFutureWave(level, wave, seed, types);
			int budget = wave * 2 / 5 + 1;
			if (IsFlagWave(wave)) budget = static_cast<int>(budget * 2.5f);
			int spent = 0, counts[4]{};
			for (int index = 0; index < n; ++index)
			{
				const int type = types[index];
				assert(type >= 0 && type < 4);
				assert(type != 3 || wave >= 4);
				spent += level.zombies[type].value;
				++counts[type];
				if (n != nr || types[index] != reference[index]) changes = true;
			}
			assert(spent == budget && n > 0 && n <= kWaveZombieSlots);
			assert(counts[1] == (IsFlagWave(wave) ? 1 : 0));
			if (wave == 29) assert(counts[0] && counts[2] && counts[3]);
		}
	}
	assert(changes);

	// SeedPacket::Update activates strictly after the refresh time and clears its counter.
	State cooldown;
	assert(LoadRoot(cooldown.lane(), root, level) == 1);
	auto lane = cooldown.lane();
	lane.i(F_PACKET_TYPE, 0) = 0;
	lane.i(F_PACKET_REFRESH, 0) = 750;
	lane.i(F_PACKET_COUNTER, 0) = 749;
	UpdatePackets(lane);
	assert(!lane.i(F_PACKET_ACTIVE, 0) && lane.i(F_PACKET_COUNTER, 0) == 750);
	UpdatePackets(lane);
	assert(lane.i(F_PACKET_ACTIVE, 0) && lane.i(F_PACKET_COUNTER, 0) == 0);
	lane.i(F_PACKET_ACTIVE, 0) = lane.i(F_PACKET_REFRESH, 0) = 0;
	UpdatePackets(lane);
	assert(!lane.i(F_PACKET_ACTIVE, 0) && lane.i(F_PACKET_COUNTER, 0) == 0);
	std::puts("mechanics OK: resampled future independence, wave budgets/flags/roster, cooldown boundaries");
}
