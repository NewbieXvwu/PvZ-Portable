// Print the byte layout of every cuda/CudaRollout.h record, for python/pvz_cuda_root.py to assert on.
//
// The Python reader mirrors these structs by hand (it has to: it reads files on a machine with no
// compiler), and a hand-mirrored layout drifts the moment someone adds a field.  This tool is the
// authority: `python3 scripts/cuda_layout_check.py` compiles it, parses the output and fails on any
// mismatch.  Built by g++ alone -- no CUDA, no link step, it never runs the simulator.
#include "../cuda/CudaRollout.h"

#include <cstdio>
#include <cstddef>

#define PVZ_FIELD(struct_name, field) \
	std::printf("%s.%s %zu\n", #struct_name, #field, offsetof(struct_name, field))
#define PVZ_STRUCT(struct_name) \
	std::printf("%s.size %zu\n", #struct_name, sizeof(struct_name))

int main()
{
	using namespace pvzc;
	PVZ_STRUCT(PlantDef);
	PVZ_FIELD(PlantDef, seedType); PVZ_FIELD(PlantDef, cost); PVZ_FIELD(PlantDef, refreshTime);
	PVZ_FIELD(PlantDef, health); PVZ_FIELD(PlantDef, launchRate); PVZ_FIELD(PlantDef, damage);
	PVZ_FIELD(PlantDef, behavior); PVZ_FIELD(PlantDef, shots); PVZ_FIELD(PlantDef, prodMin);
	PVZ_FIELD(PlantDef, prodMax); PVZ_FIELD(PlantDef, prodValue); PVZ_FIELD(PlantDef, chills);
	PVZ_FIELD(PlantDef, shootCounter);

	PVZ_STRUCT(ZombieDef);
	PVZ_FIELD(ZombieDef, type); PVZ_FIELD(ZombieDef, value); PVZ_FIELD(ZombieDef, firstAllowedWave);
	PVZ_FIELD(ZombieDef, pickWeight); PVZ_FIELD(ZombieDef, bodyHealth); PVZ_FIELD(ZombieDef, helmHealth);
	PVZ_FIELD(ZombieDef, shieldHealth); PVZ_FIELD(ZombieDef, variantCount); PVZ_FIELD(ZombieDef, velX);
	PVZ_FIELD(ZombieDef, velMin); PVZ_FIELD(ZombieDef, velMax);

	PVZ_STRUCT(RootPlant);
	PVZ_FIELD(RootPlant, cell); PVZ_FIELD(RootPlant, type); PVZ_FIELD(RootPlant, health);
	PVZ_FIELD(RootPlant, maxHealth); PVZ_FIELD(RootPlant, state); PVZ_FIELD(RootPlant, stateCountdown);
	PVZ_FIELD(RootPlant, launchCounter); PVZ_FIELD(RootPlant, launchRate); PVZ_FIELD(RootPlant, special);
	PVZ_FIELD(RootPlant, extraA); PVZ_FIELD(RootPlant, extraB); PVZ_FIELD(RootPlant, onBoard);
	PVZ_FIELD(RootPlant, posX); PVZ_FIELD(RootPlant, posY);
	PVZ_FIELD(RootPlant, animTime); PVZ_FIELD(RootPlant, animRate);
	PVZ_FIELD(RootPlant, animFrames); PVZ_FIELD(RootPlant, animLoops);

	PVZ_STRUCT(RootZombie);
	PVZ_FIELD(RootZombie, type); PVZ_FIELD(RootZombie, row); PVZ_FIELD(RootZombie, body);
	PVZ_FIELD(RootZombie, bodyMax); PVZ_FIELD(RootZombie, helm); PVZ_FIELD(RootZombie, helmMax);
	PVZ_FIELD(RootZombie, shield); PVZ_FIELD(RootZombie, shieldMax); PVZ_FIELD(RootZombie, flying);
	PVZ_FIELD(RootZombie, phase); PVZ_FIELD(RootZombie, phaseCounter); PVZ_FIELD(RootZombie, stateBits);
	PVZ_FIELD(RootZombie, chilled); PVZ_FIELD(RootZombie, age); PVZ_FIELD(RootZombie, fromWave);
	PVZ_FIELD(RootZombie, targetCell); PVZ_FIELD(RootZombie, posX); PVZ_FIELD(RootZombie, posY);
	PVZ_FIELD(RootZombie, velX);
	PVZ_FIELD(RootZombie, animTime); PVZ_FIELD(RootZombie, animRate);
	PVZ_FIELD(RootZombie, animStart); PVZ_FIELD(RootZombie, animFrames);
	PVZ_FIELD(RootZombie, animLoopType); PVZ_FIELD(RootZombie, animLoops); PVZ_FIELD(RootZombie, fade);

	PVZ_STRUCT(RootProjectile);
	PVZ_FIELD(RootProjectile, type); PVZ_FIELD(RootProjectile, row); PVZ_FIELD(RootProjectile, damage);
	PVZ_FIELD(RootProjectile, age); PVZ_FIELD(RootProjectile, chill); PVZ_FIELD(RootProjectile, posX);
	PVZ_FIELD(RootProjectile, posY); PVZ_FIELD(RootProjectile, velX); PVZ_FIELD(RootProjectile, velY);

	PVZ_STRUCT(RootCoin); PVZ_FIELD(RootCoin, value); PVZ_FIELD(RootCoin, ticksToCollect);
	PVZ_STRUCT(RootMower); PVZ_FIELD(RootMower, row); PVZ_FIELD(RootMower, state);
	PVZ_FIELD(RootMower, timer); PVZ_FIELD(RootMower, flags); PVZ_FIELD(RootMower, chompCounter);
	PVZ_FIELD(RootMower, posX);
	PVZ_STRUCT(RootPacket); PVZ_FIELD(RootPacket, type); PVZ_FIELD(RootPacket, active);
	PVZ_FIELD(RootPacket, refreshCounter); PVZ_FIELD(RootPacket, refreshTime);
	PVZ_FIELD(RootPacket, cost);

	PVZ_STRUCT(RootRecord);
	PVZ_FIELD(RootRecord, magic); PVZ_FIELD(RootRecord, version); PVZ_FIELD(RootRecord, tick);
	PVZ_FIELD(RootRecord, sun); PVZ_FIELD(RootRecord, wave); PVZ_FIELD(RootRecord, numWaves);
	PVZ_FIELD(RootRecord, zombieCountDown); PVZ_FIELD(RootRecord, zombieCountDownStart);
	PVZ_FIELD(RootRecord, zombieHealthWaveStart); PVZ_FIELD(RootRecord, zombieHealthToNextWave);
	PVZ_FIELD(RootRecord, hugeWaveCountDown); PVZ_FIELD(RootRecord, sunCountDown);
	PVZ_FIELD(RootRecord, numSunsFallen); PVZ_FIELD(RootRecord, sunMoneyProduced);
	PVZ_FIELD(RootRecord, zombiesKilled); PVZ_FIELD(RootRecord, plantsEaten);
	PVZ_FIELD(RootRecord, plantsPlanted); PVZ_FIELD(RootRecord, triggeredLawnMowers);
	PVZ_FIELD(RootRecord, result); PVZ_FIELD(RootRecord, rowGotMowered);
	PVZ_FIELD(RootRecord, rowLastPicked); PVZ_FIELD(RootRecord, rowSecondLastPicked);
	PVZ_FIELD(RootRecord, incomeWindowStart); PVZ_FIELD(RootRecord, incomeCount);
	PVZ_FIELD(RootRecord, incomeHead); PVZ_FIELD(RootRecord, incomeTick);
	PVZ_FIELD(RootRecord, incomeValue); PVZ_FIELD(RootRecord, plantCount);
	PVZ_FIELD(RootRecord, zombieCount); PVZ_FIELD(RootRecord, projectileCount);
	PVZ_FIELD(RootRecord, coinCount); PVZ_FIELD(RootRecord, mowerCount);
	PVZ_FIELD(RootRecord, packetCount); PVZ_FIELD(RootRecord, plants);
	PVZ_FIELD(RootRecord, zombies); PVZ_FIELD(RootRecord, projectiles); PVZ_FIELD(RootRecord, coins);
	PVZ_FIELD(RootRecord, mowers); PVZ_FIELD(RootRecord, packets); PVZ_FIELD(RootRecord, waveCount);
	PVZ_FIELD(RootRecord, waveZombieCount); PVZ_FIELD(RootRecord, waveTable);
	PVZ_FIELD(RootRecord, supported); PVZ_FIELD(RootRecord, unsupportedReason);

	PVZ_STRUCT(WaveTable);
	PVZ_FIELD(WaveTable, waveCount); PVZ_FIELD(WaveTable, zombieCount); PVZ_FIELD(WaveTable, type);

	PVZ_STRUCT(LevelInfo);
	PVZ_FIELD(LevelInfo, level); PVZ_FIELD(LevelInfo, numWaves); PVZ_FIELD(LevelInfo, playableRows);
	PVZ_FIELD(LevelInfo, allowed); PVZ_FIELD(LevelInfo, zombies); PVZ_FIELD(LevelInfo, plantCount);
	PVZ_FIELD(LevelInfo, plants); PVZ_FIELD(LevelInfo, zombieCountMultiplier);

	// Constants the Python reader needs, so a capacity change cannot pass unnoticed there.
	std::printf("kMagic %d\n", kMagic);
	std::printf("kVersion %d\n", kVersion);
	std::printf("kRows %d\n", kRows);
	std::printf("kPlantSlots %d\n", kPlantSlots);
	std::printf("kZombieSlots %d\n", kZombieSlots);
	std::printf("kProjectileSlots %d\n", kProjectileSlots);
	std::printf("kCoinSlots %d\n", kCoinSlots);
	std::printf("kMowerSlots %d\n", kMowerSlots);
	std::printf("kPacketSlots %d\n", kPacketSlots);
	std::printf("kIncomeEvents %d\n", kIncomeEvents);
	std::printf("kMaxWaves %d\n", kMaxWaves);
	std::printf("kWaveZombieSlots %d\n", kWaveZombieSlots);
	std::printf("kZombieTypes %d\n", kZombieTypes);
	std::printf("kPlantTypes %d\n", kPlantTypes);
	return 0;
}
