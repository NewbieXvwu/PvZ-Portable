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

	// This real failing root has a normal zombie at x=550 and a potato at x=520. The constructor's
	// [50,70] mouth misses it, but LoadPlainZombieReanim's [20,70] mouth overlaps by exactly 20 px.
	State mouth;
	assert(LoadRoot(mouth.lane(), root, level) == 1);
	mouth.lane().i(F_PLANT_TYPE, 36) = 4;
	mouth.lane().f(F_PLANT_X, 36) = 520.0f;
	assert(FindEatTarget(mouth.lane(), level, 3, 550, 0) == 36);
	assert(FindEatTarget(mouth.lane(), level, 3, 551, 0) == -1);
	// Post-vault mouth is [50,70], unlike the pre-vault [-29,41] and plain [20,70].
	assert(FindEatTarget(mouth.lane(), level, 3, 550, 2) == -1);
	assert(FindEatTarget(mouth.lane(), level, 3, 520, 2) == 36);

	State vaulted;
	auto pole = vaulted.lane();
	assert(LoadRoot(pole, root, level) == 1);
	pole.i(F_ZOMBIE_TYPE, 0) = 3;
	pole.i(F_ZOMBIE_FLAGS, 0) = ZB_HAS_HEAD | ZB_HAS_ARM | ZB_ON_BOARD;
	pole.i(F_ZOMBIE_PHASE, 0) = 11;
	pole.i(F_ZOMBIE_BODY, 0) = pole.i(F_ZOMBIE_BODY_MAX, 0) = 500;
	pole.f(F_ZOMBIE_X, 0) = 100.0f;
	pole.i(F_PLANT_TYPE, 0) = 0;
	pole.i(F_PLANT_HP, 0) = 300;
	pole.f(F_PLANT_X, 0) = 40.0f;
	Rand vaultRand;
	vaultRand.Seed(0, 123);
	RedrawZombieSpeed(pole, level.zombies[3], 0, vaultRand);
	assert(pole.i(F_ZOMBIE_ANIM_START, 0) == 13);
	UpdateZombie(pole, level, 0, vaultRand);
	assert(pole.i(F_ZOMBIE_PHASE, 0) == 12 && pole.i(F_ZOMBIE_ANIM_START, 0) == 50);
	assert(pole.i(F_PLANT_HP, 0) == 300 && !(pole.i(F_ZOMBIE_FLAGS, 0) & ZB_EATING));
	for (int i = 0; i < 179; ++i)
	{
		UpdateZombieAnimation(pole, 0);
		UpdateZombie(pole, level, 0, vaultRand);
		assert(pole.i(F_ZOMBIE_PHASE, 0) == 12);
	}
	const float beforeLanding = pole.f(F_ZOMBIE_X, 0);
	UpdateZombieAnimation(pole, 0);
	UpdateZombie(pole, level, 0, vaultRand);
	assert(pole.i(F_ZOMBIE_PHASE, 0) == 13 && pole.i(F_ZOMBIE_ANIM_START, 0) == 93);
	assert(pole.f(F_ZOMBIE_X, 0) < beforeLanding - 150.0f);
	assert(pole.f(F_ZOMBIE_VELX, 0) >= 0.23f && pole.f(F_ZOMBIE_VELX, 0) <= 0.37f);

	// Imported corpses do not restart a guessed duration or disappear at phaseCounter==0.
	State fading;
	auto corpse = fading.lane();
	assert(LoadRoot(corpse, root, level) == 1);
	corpse.i(F_ZOMBIE_TYPE, 0) = 0;
	corpse.i(F_ZOMBIE_FLAGS, 0) = ZB_DYING | ZB_ON_BOARD;
	corpse.i(F_ZOMBIE_FADE, 0) = -1;
	PlayZombieAnimation(corpse, 0, 178, 39, 24.0f, 3);
	UpdateZombie(corpse, level, 0, vaultRand);
	assert(corpse.i(F_ZOMBIE_TYPE, 0) == 0 && corpse.i(F_ZOMBIE_FADE, 0) == -1);
	corpse.i(F_ZOMBIE_ANIM_LOOPS, 0) = 1;
	UpdateZombie(corpse, level, 0, vaultRand);
	assert(corpse.i(F_ZOMBIE_FADE, 0) == 99);
	for (int i = 0; i < 99; ++i) UpdateZombie(corpse, level, 0, vaultRand);
	assert(corpse.i(F_ZOMBIE_TYPE, 0) == -1 && corpse.s(S_ZOMBIES_KILLED) == 1);

	State shot;
	assert(LoadRoot(shot.lane(), root, level) == 1);
	Lane pea = shot.lane();
	pea.i(F_ZOMBIE_TYPE, 0) = 0;
	pea.i(F_ZOMBIE_FLAGS, 0) = ZB_HAS_HEAD | ZB_HAS_ARM | ZB_ON_BOARD;
	pea.i(F_ZOMBIE_BODY, 0) = 230;
	pea.i(F_ZOMBIE_BODY_MAX, 0) = 270;
	pea.f(F_ZOMBIE_X, 0) = 391.53726f;
	pea.i(F_PROJ_TYPE, 0) = 0;
	pea.i(F_PROJ_DAMAGE, 0) = 20;
	pea.f(F_PROJ_X, 0) = 383.87952f;
	pea.f(F_PROJ_VELX, 0) = 3.33f;
	Rand projectileRand;
	projectileRand.Seed(0, 123);
	UpdateProjectiles(pea, level, projectileRand);
	assert(pea.i(F_ZOMBIE_BODY, 0) == 230 && pea.i(F_PROJ_TYPE, 0) == 0);
	UpdateProjectiles(pea, level, projectileRand);
	assert(pea.i(F_ZOMBIE_BODY, 0) == 210 && pea.i(F_PROJ_TYPE, 0) == -1);
	pea.i(F_ZOMBIE_FLAGS, 0) |= ZB_DYING;
	assert(PeaTarget(pea, 0, 400) == -1);
	pea.i(F_ZOMBIE_FLAGS, 0) &= ~ZB_DYING;
	pea.i(F_ZOMBIE_PHASE, 0) = 12;
	assert(PeaTarget(pea, 0, 400) == -1);
	pea.i(F_PROJ_TYPE, 0) = 0;
	pea.f(F_PROJ_X, 0) = 799.0f;
	UpdateProjectiles(pea, level, projectileRand);
	assert(pea.i(F_PROJ_TYPE, 0) == -1);

	// Plant.cpp uses NOTREADY -> RISING -> ARMED. The 19-frame rise at 18 fps needs 106 updates,
	// and cannot explode until the armed branch runs on the following update. Its blast hits all
	// zombies in range, not merely the one that caused the trigger.
	State potato;
	auto mine = potato.lane();
	assert(LoadRoot(mine, root, level) == 1);
	level.plantCount = 1;
	level.plants[0].seedType = 4;
	level.plants[0].behavior = PB_POTATO;
	mine.i(F_PLANT_TYPE, 0) = 4;
	mine.i(F_PLANT_STATE_COUNTDOWN, 0) = 1;
	mine.i(F_PLANT_HP, 0) = 300;
	mine.f(F_PLANT_X, 0) = 40.0f;
	for (int slot = 0; slot < 2; ++slot)
	{
		mine.i(F_ZOMBIE_TYPE, slot) = 0;
		mine.i(F_ZOMBIE_FLAGS, slot) = ZB_HAS_HEAD | ZB_HAS_ARM | ZB_ON_BOARD;
		mine.i(F_ZOMBIE_BODY, slot) = mine.i(F_ZOMBIE_BODY_MAX, slot) = 270;
		mine.f(F_ZOMBIE_X, slot) = 20.0f + slot * 10.0f;
	}
	UpdatePlant(mine, level, 0, projectileRand);
	assert(mine.i(F_PLANT_STATE, 0) == 15 && mine.i(F_PLANT_STATE_COUNTDOWN, 0) == 0);
	for (int tick = 0; tick < 105; ++tick)
	{
		UpdatePlantAnimation(mine, 0);
		UpdatePlant(mine, level, 0, projectileRand);
		assert(mine.i(F_PLANT_STATE, 0) == 15 && mine.i(F_PLANT_TYPE, 0) == 4);
	}
	UpdatePlantAnimation(mine, 0);
	UpdatePlant(mine, level, 0, projectileRand);
	assert(mine.i(F_PLANT_STATE, 0) == 16 && mine.i(F_PLANT_TYPE, 0) == 4);
	UpdatePlant(mine, level, 0, projectileRand);
	assert(mine.i(F_PLANT_TYPE, 0) == -1);
	assert(mine.i(F_ZOMBIE_BODY, 0) == 0 && mine.i(F_ZOMBIE_BODY, 1) == 0);

	// A sky sun on the last tick contributes to the income feature immediately, even though its
	// money will only be collected at the next tick's head. Imported pending coins are not produced twice.
	State sunlight;
	auto sun = sunlight.lane();
	assert(LoadRoot(sun, root, level) == 1);
	sun.s(S_SUN_COUNT_DOWN) = 1;
	SpawnSkySun(sun, projectileRand);
	assert(sun.s(S_SUN_PRODUCED) == 25 && sun.s(S_SUN) == 0);
	BankCoins(sun);
	assert(sun.s(S_SUN_PRODUCED) == 25 && sun.s(S_SUN) == 25);
	sun.s(S_SUN) = 9980;
	AddCoin(sun, 25);
	BankCoins(sun);
	assert(sun.s(S_SUN) == 9990 && sun.s(S_SUN_PRODUCED) == 50);

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
	std::puts("mechanics OK: future independence, waves, cooldown, bite reach, delayed pea collision");
}
