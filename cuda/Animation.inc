// Logic-relevant reanimation state, without rendering or attachments.
#ifndef PVZ_SIM_ANIMATION_INC
#define PVZ_SIM_ANIMATION_INC
#include "SimCore.h"
#include "AnimationTables.h"
namespace pvzc {

PVZ_SIM void StepAnimation(float& time, float rate, int32_t frames, int32_t loopType, int32_t& loops)
{
	if (frames <= 0) return;
	// SECONDS_PER_UPDATE is the float32 0.01 widened to double in the original (Reanimator.h:52).
	time = static_cast<float>(static_cast<double>(time)
		+ static_cast<double>(0.01f) * static_cast<double>(rate) / static_cast<double>(frames));
	if (loopType == 0 || loopType == 1)
	{
		while (time >= 1.0f) { time -= 1.0f; ++loops; }
	}
	else if (time >= 1.0f) { time = 1.0f; loops = 1; }
}

PVZ_SIM void UpdatePlantAnimation(Lane lane, int32_t slot)
{
	if (lane.i(F_PLANT_TYPE, slot) >= 0 && lane.i(F_PLANT_STATE, slot) == 15)
		StepAnimation(lane.f(F_PLANT_ANIM_TIME, slot), lane.f(F_PLANT_ANIM_RATE, slot),
			lane.i(F_PLANT_ANIM_FRAMES, slot), 3, lane.i(F_PLANT_ANIM_LOOPS, slot));
}

PVZ_SIM void UpdateZombieAnimation(Lane lane, int32_t slot)
{
	if (lane.i(F_ZOMBIE_TYPE, slot) < 0 || (lane.i(F_ZOMBIE_FLAGS, slot) & ZB_DEAD)) return;
	StepAnimation(lane.f(F_ZOMBIE_ANIM_TIME, slot), lane.f(F_ZOMBIE_ANIM_RATE, slot),
		lane.i(F_ZOMBIE_ANIM_FRAMES, slot), lane.i(F_ZOMBIE_ANIM_LOOP_TYPE, slot),
		lane.i(F_ZOMBIE_ANIM_LOOPS, slot));
}

// Body animations are attachments and advance at each entity's UpdateReanim tail. The separate
// mower animation is not an attachment: it advances here AND at Zombie::UpdateReanim (5365).
PVZ_SIM void UpdateMowedAnimations(Lane lane)
{
	for (int32_t slot = 0; slot < kZombieSlots; ++slot)
		if (lane.i(F_ZOMBIE_FLAGS, slot) & ZB_MOWERED) UpdateZombieAnimation(lane, slot);
}

PVZ_SIM void PlayZombieAnimation(Lane lane, int32_t slot, int32_t start, int32_t frames,
	float rate, int32_t loopType)
{
	lane.i(F_ZOMBIE_ANIM_START, slot) = start;
	lane.i(F_ZOMBIE_ANIM_FRAMES, slot) = frames;
	lane.i(F_ZOMBIE_ANIM_LOOP_TYPE, slot) = loopType;
	lane.i(F_ZOMBIE_ANIM_LOOPS, slot) = 0;
	lane.f(F_ZOMBIE_ANIM_TIME, slot) = 0.0f;
	lane.f(F_ZOMBIE_ANIM_RATE, slot) = lane.i(F_ZOMBIE_CHILLED, slot) > 0 ? rate * 0.5f : rate;
}

PVZ_SIM void PlayGroundAnimation(Lane lane, int32_t slot, int32_t start, int32_t frames)
{
	const int32_t type = lane.i(F_ZOMBIE_TYPE, slot);
	const float distance = GroundCoordinate(type, start, frames - 1) - GroundCoordinate(type, start, 0);
	const float invSpeed = static_cast<float>(frames) / distance;
	const float rate = lane.f(F_ZOMBIE_VELX, slot) * invSpeed * 47.0f;
	PlayZombieAnimation(lane, slot, start, frames, rate, 0);
}

PVZ_SIM float GroundAdvance(Lane lane, int32_t slot)
{
	const int32_t type = lane.i(F_ZOMBIE_TYPE, slot);
	const int32_t start = lane.i(F_ZOMBIE_ANIM_START, slot);
	const int32_t frames = lane.i(F_ZOMBIE_ANIM_FRAMES, slot);
	if (frames <= 0) return 0.0f;
	const int32_t loop = lane.i(F_ZOMBIE_ANIM_LOOP_TYPE, slot);
	const int32_t span = (loop == 1 || loop == 4 || loop == 5) ? frames : frames - 1;
#ifdef __CUDA_ARCH__
	const float position = __fadd_rn(static_cast<float>(start),
		__fmul_rn(lane.f(F_ZOMBIE_ANIM_TIME, slot), static_cast<float>(span)));
#else
	volatile float scaled = lane.f(F_ZOMBIE_ANIM_TIME, slot) * static_cast<float>(span);
	const float position = static_cast<float>(start) + scaled;
#endif
	const int32_t before = ClampInt(static_cast<int32_t>(position) - start, 0, frames - 1);
	const int32_t after = Minimum(before + 1, frames - 1);
	const float distance = GroundCoordinate(type, start, after) - GroundCoordinate(type, start, before);
	return static_cast<float>(static_cast<double>(distance) * static_cast<double>(0.01f)
		* static_cast<double>(lane.f(F_ZOMBIE_ANIM_RATE, slot)));
}

} // namespace pvzc
#endif
