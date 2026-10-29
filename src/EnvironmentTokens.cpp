#include "EnvironmentTokens.h"
#include "LawnApp.h"
#include "Lawn/Board.h"
#include "Lawn/Plant.h"
#include "Lawn/Zombie.h"
#include "Lawn/Projectile.h"
#include "Lawn/SeedPacket.h"
#include "Lawn/LawnMower.h"
#include "Lawn/GridItem.h"
#include "PvzpLib/Reanimator.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <stdexcept>

std::vector<EnvironmentToken> PublicObservationTokens(LawnApp* app)
{
	std::vector<EnvironmentToken> tokens;
	PublicObservationTokens(app,tokens);
	return tokens;
}

void PublicObservationTokens(LawnApp* app, std::vector<EnvironmentToken>& tokens)
{
	Board* b = app->mBoard;
	const bool dark = app->EnvironmentStormHidden();
	tokens.clear();
	auto add = [&](int kind, int category, int variant, int row, int col,
		std::initializer_list<float> values) -> EnvironmentToken& {
		tokens.push_back({kind, category + 1, variant + 1, row, col});
		std::copy(values.begin(), values.end(), tokens.back().features.begin());
		return tokens.back();
	};
	auto column = [](float x) { return std::clamp(static_cast<int>(std::floor((x - 40) / 80)), 0, 8); };
	add(0, static_cast<int>(b->mBackground), static_cast<int>(app->mGameMode), -1, -1,
		{b->mSunMoney / 1000.f, b->mCurrentWave / static_cast<float>(std::max(1, b->mNumWaves)),
		 b->mMainCounter / 60000.f, b->mNumWaves / 30.f,
		 static_cast<float>(app->mEnvironmentTaskSpec.zombieCountMultiplier / 10),
		 float(b->StageIsNight()), float(b->StageHasPool()), float(b->StageHasFog()), float(b->StageHasRoof()),
		 float(app->mEnvironmentTaskSpec.originalConveyorStart), float(app->IsWallnutBowlingLevel()), float(dark),
		 b->mLevel / 50.f});
	const auto& profile = app->mEnvironmentTaskSpec;
	add(1, -1, -1, -1, -1, {profile.playthrough / 2.f, profile.seedSlotCount / 10.f,
		float(profile.firstAidOwned), float(profile.imitaterOwned), float(profile.poolCleanerOwned),
		float(profile.roofCleanerOwned)});
	for (int row = 0; row < 6; ++row)
		for (int col = 0; col < 9; ++col)
			add(2, static_cast<int>(b->mGridSquareType[col][row]), static_cast<int>(b->mPlantRow[row]), row, col,
				{row / 5.f, col / 8.f, app->EnvironmentFogAmount(row, col) / 255.f,
				 float(b->mPlantRow[row] != PlantRowType::PLANTROW_DIRT)});
	for (int i = 0; i < 10; ++i)
	{
		if (i >= b->mSeedBank->mNumPackets || b->mSeedBank->mSeedPackets[i].mPacketType == SeedType::SEED_NONE)
		{
			add(8, -1, -1, -1, -1, {});
			continue;
		}
		const auto& p = b->mSeedBank->mSeedPackets[i];
		const int cost = Plant::GetCost(p.mPacketType, p.mImitaterType);
		add(8, static_cast<int>(p.mPacketType), static_cast<int>(p.mImitaterType), -1, -1,
			{cost / 500.f, p.mRefreshCounter / 3000.f, p.mRefreshTime / 3000.f, float(p.mActive),
			 float(profile.originalConveyorStart || cost <= b->mSunMoney)});
	}
	for (const Plant* p : b->mPlants)
	{
		if (p->mDead || dark) continue;
		const bool hidden = app->EnvironmentPlantFogHidden(p);
		const bool rolling = app->IsWallnutBowlingLevel();
		// A rolling object's position cannot be recovered from its original planting cell under fog.
		if (hidden && rolling) continue;
		float vx = 0, vy = 0;
		if (rolling && !hidden)
		{
			Reanimation* anim = app->ReanimationTryToGet(p->mBodyReanimID);
			if (anim && anim->TrackExists("_ground"))
				vx = -anim->GetTrackVelocity("_ground") * (p->mSeedType == SeedType::SEED_GIANT_WALLNUT ? 2 : 1);
			vy = p->mState == PlantState::STATE_BOWLING_UP ? -2 : p->mState == PlantState::STATE_BOWLING_DOWN ? 2 : 0;
		}
		auto visible = [hidden](float x) { return hidden ? -1.f : x; };
		add(3, static_cast<int>(p->mSeedType), static_cast<int>(p->mImitaterType), p->mRow,
			rolling ? column(p->mX) : p->mPlantCol,
			{p->mRow / 5.f, p->mPlantCol / 8.f, visible(p->mPlantHealth / 3000.f),
			 visible(p->mPlantMaxHealth / 3000.f), visible(p->mX / 900.f), visible(p->mY / 700.f),
			 visible(static_cast<int>(p->mState) / 128.f), visible(p->mStateCountdown / 1200.f),
			 visible(p->mDoSpecialCountdown / 1200.f), -1.f, p->mLaunchRate / 1200.f,
			 visible(p->mShootingCounter / 1200.f), visible(p->mWakeUpCounter / 1200.f),
			 visible(float(p->mIsAsleep)), visible(float(p->mSquished)), visible(static_cast<int>(p->mOnBungeeState) / 3.f),
			 float(!hidden), float(rolling), vx / 20.f, vy / 20.f});
	}
	for (Zombie* z : b->mZombies)
	{
		if (z->mDead || !app->EnvironmentZombieVisible(z) || z->mZombieType == ZombieType::ZOMBIE_BUNGEE) continue;
		// Phase is the visible action. Random phase/summon timers and internal target choices are omitted.
		add(4, static_cast<int>(z->mZombieType), -1, z->mRow, column(z->mPosX),
			{z->mRow / 5.f, z->mPosX / 900.f, z->mPosY / 700.f, z->mVelX / 5.f,
			 z->mBodyHealth / 2000.f, z->mBodyMaxHealth / 2000.f, z->mHelmHealth / 2000.f,
			 z->mHelmMaxHealth / 2000.f, z->mShieldHealth / 2000.f, z->mShieldMaxHealth / 2000.f,
			 static_cast<int>(z->mZombiePhase) / 128.f, z->mChilledCounter / 1200.f, z->mButteredCounter / 1200.f,
			 z->mIceTrapCounter / 1200.f, float(z->mHasHead), float(z->mHasArm), float(z->mHasObject),
			 float(z->mIsEating), float(z->IsOnBoard()), z->mScaleZombie});
	}
	for (const auto& z : app->EnvironmentBungees().zombies)
	{
		const int col = z[2] ? z[1] : -1;
		auto& t = add(4, static_cast<int>(ZombieType::ZOMBIE_BUNGEE), -1, z[0], col,
			{z[0] / 5.f, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
			 float(z[2]), z[3] / 3.f, z[4] / 700.f});
		if (col >= 0 && col < 9) t.target = 2 + z[0] * 9 + col;
	}
	for (Projectile* p : b->mProjectiles)
	{
		if (p->mDead || !app->EnvironmentProjectileVisible(p)) continue;
		add(5, static_cast<int>(p->mProjectileType), static_cast<int>(p->mMotionType), p->mRow, column(p->mPosX),
			{p->mRow / 5.f, p->mPosX / 900.f, p->mPosY / 700.f, p->mPosZ / 500.f,
			 p->mVelX / 20.f, p->mVelY / 20.f, p->mVelZ / 20.f, p->GetProjectileDef().mDamage / 500.f});
	}
	for (const LawnMower* m : b->mLawnMowers)
		if (!m->mDead && !dark)
			add(6, static_cast<int>(m->mMowerType), static_cast<int>(m->mMowerState), m->mRow, column(m->mPosX),
				{m->mRow / 5.f, m->mPosX / 900.f, m->mPosY / 700.f});
	for (const GridItem* g : b->mGridItems)
		if (!g->mDead && !dark)
			// Contents of unopened vases/graves and their internal timers are hidden.
			add(7, static_cast<int>(g->mGridItemType), -1, g->mGridY, g->mGridX,
				{g->mGridY / 5.f, g->mGridX / 8.f, g->mPosX / 900.f, g->mPosY / 700.f});
	for (int type = 0; type < NUM_ZOMBIE_TYPES && type < 100; ++type)
		if (b->mZombieAllowed[type]) add(9, type, -1, -1, -1, {1});
	// Simple visible counts, without hand-coded economic/firepower or trajectory estimates.
	for (int row = 0; row < 6; ++row)
	{
		float plants = 0, zombies = 0, projectiles = 0;
		for (const auto& t : tokens)
			if (t.row == row) { plants += t.kind == 3; zombies += t.kind == 4; projectiles += t.kind == 5; }
		add(10, -1, -1, row, -1, {plants / 20.f, zombies / 40.f, projectiles / 50.f});
	}
}

std::vector<float> ObservationTokenArray(const std::vector<EnvironmentToken>& tokens, size_t capacity)
{
	if (capacity == 0) capacity = tokens.size();
	if (capacity < tokens.size()) throw std::length_error("obs-v3 token count exceeds graph capacity");
	std::vector<float> values(capacity * ObservationTokenFields);
	FillObservationTokenArray(tokens, values.data(), capacity);
	return values;
}

void FillObservationTokenArray(const std::vector<EnvironmentToken>& tokens, float* values, size_t capacity)
{
	if (capacity < tokens.size()) throw std::length_error("obs-v3 token count exceeds graph capacity");
	std::fill_n(values,capacity*ObservationTokenFields,0.f);
	for (size_t i = 0; i < capacity; ++i)
	{
		float* row = values + i * ObservationTokenFields;
		row[3] = row[4] = row[5] = -1;
		if (i >= tokens.size()) continue;
		const auto& t = tokens[i];
		row[0] = t.kind; row[1] = t.category; row[2] = t.variant;
		row[3] = t.row; row[4] = t.col; row[5] = t.target;
		std::copy(t.features.begin(), t.features.end(), row + 6);
	}
}

void WriteObservationTokens(std::ostream& out, LawnApp* app)
{
	const auto tokens = PublicObservationTokens(app);
	const auto values = ObservationTokenArray(tokens);
	const auto precision = out.precision();
	out << ",\"v3\":{\"version\":3,\"tokens\":[" << std::setprecision(9);
	for (size_t i = 0; i < tokens.size(); ++i)
	{
		if (i) out << ',';
		out << '[';
		for (int j = 0; j < ObservationTokenFields; ++j)
			out << (j ? "," : "") << values[i * ObservationTokenFields + j];
		out << ']';
	}
	out << "]}";
	out.precision(precision);
}
