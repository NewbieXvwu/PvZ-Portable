"""Task70 B5: a human cannot see their own plants inside the fog wall.

The pre-B5 human_v2 observation reported every own-plant's health and state timers even
when its cell sits under render fog no human could see through.  The revision keeps
row/col/type (memory plus the card-click test recover those) and reports fixed -1
("unknown", the target_zombie_id convention) for health, max_health, mState,
mStateCountdown, mDoSpecialCountdown and mLaunchCounter whenever the plant's grid anchor
is under fog opacity >= 0.925.  Four categories, mirroring task66's fog tests:
threshold exactness, bit-identical human/xray against the pre-B5 engine, non-leakage,
and masking-is-presentation (same simulation, same hash).
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_env import PvZEnv, TaskSpec
from pvz_tasks import deck_for_level, profile_for_deck

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', str(Path.home() / '.cache/pvz-research-resources')))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', ROOT / 'build/pvz-portable'))
OLD_BINARY = ROOT / 'artifacts/task70/deploy4_pre_b5/pvz-portable'
THRESHOLD = 925
MASKED_FIELDS = ('health', 'max_health', 'state', 'state_countdown', 'launch_counter')

engine = pytest.mark.skipif(not BINARY.is_file() or not (RESOURCE / 'main.pak').is_file(),
                            reason='requires simulator and resources')
old_engine = pytest.mark.skipif(not OLD_BINARY.is_file(),
                                reason='requires the archived pre-B5 engine')


def reset(env, level=31, seed=660031, fog=None, preplanted=()):
    deck = deck_for_level(level)
    return env.reset(deck=deck, task=TaskSpec(level=level, seed=seed,
                                              profile=profile_for_deck(deck),
                                              fog_observation=fog, preplanted=preplanted))[0]


# Peashooters on both sides of the fog wall: cols 2 (clear) and 6/7/8 (fogged once the
# fog blows in, which takes the first ~1000 ticks).  Ground rows only: rows 2/3 are pool.
PREPLANTED = ((0, 0, 2), (0, 1, 6), (0, 4, 7), (0, 5, 8))


def plant_opacities(state):
    return {(p['row'], p['col'], p['type']): p['opacity'] for p in state['fog_plants']}


def plant_keys(plants):
    return {(p['row'], p['col'], p['type']) for p in plants}


@engine
def test_human_v2_masks_exactly_the_opaque_plant_cells():
    """Every plant is listed with row/col/type; exactly the >= 0.925 ones carry -1."""
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        reset(env, fog='human_v2', preplanted=PREPLANTED)
        env.step(dict(type='wait', ticks=1000))
        shown_ticks = hidden_ticks = 0
        for _ in range(24):
            observation, *_ = env.step(dict(type='wait', ticks=500))
            state = env.privileged_state()
            opacity = plant_opacities(state)
            # Plants are never dropped from the public list, unlike zombies.
            public = {(p['row'], p['col'], p['type']): p for p in observation['plants']}
            assert set(public) == plant_keys(state['plants'])
            for key, real in {(p['row'], p['col'], p['type']): p for p in state['plants']}.items():
                masked = opacity[key] >= THRESHOLD
                for field in MASKED_FIELDS:
                    assert (public[key][field] == -1) == masked, (key, field, public[key][field])
                    assert real[field] != -1, (key, field, real[field])
                if masked:
                    hidden_ticks += 1
                else:
                    shown_ticks += 1
        assert shown_ticks and hidden_ticks


@engine
def test_public_plants_match_privileged_except_masked_fields():
    """Below the threshold the public values are the real ones."""
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        reset(env, fog='human_v2', preplanted=PREPLANTED)
        env.step(dict(type='wait', ticks=2500))
        observation = env.observe()
        state = env.privileged_state()
        opacity = plant_opacities(state)
        by_key = {(p['row'], p['col'], p['type']): p for p in state['plants']}
        checked = 0
        for plant in observation['plants']:
            key = (plant['row'], plant['col'], plant['type'])
            real = by_key[key]
            if opacity[key] < THRESHOLD:
                for field in MASKED_FIELDS:
                    assert plant[field] == real[field], (key, field)
                checked += 1
        assert checked, 'at least one clear-side plant must be exercised'


@engine
def test_masked_plant_fields_are_the_unknown_sentinel_in_every_form():
    """Full, lean and v2 all carry the same -1 pattern for the same cells."""
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        reset(env, fog='human_v2', preplanted=PREPLANTED)
        env.step(dict(type='wait', ticks=2500))
        full = env.observe()
        state = env.privileged_state()
        opacity = plant_opacities(state)
        masked_keys = {key for key, value in opacity.items() if value >= THRESHOLD}
        assert masked_keys and len(masked_keys) < len(opacity)
        for plant in full['plants']:
            key = (plant['row'], plant['col'], plant['type'])
            if key in masked_keys:
                assert all(plant[field] == -1 for field in MASKED_FIELDS)
        env.set_lean_observation(True)
        lean = env.observe()
        env.set_lean_observation(False)
        for row, col, kind, health, max_health, squished, *_ in lean['plants']:
            if (row, col, kind) in masked_keys:
                assert health == -1 and max_health == -1
        v2 = {(row, col, kind): (state_field, countdown, special, launch)
              for row, col, kind, state_field, countdown, special, launch, rate in full['v2']['plants']}
        assert set(v2) <= plant_keys(state['plants'])
        for key, (state_field, countdown, special, launch) in v2.items():
            if key in masked_keys:
                assert state_field == -1 and countdown == -1 and special == -1 and launch == -1
            else:
                assert state_field != -1 and countdown != -1 and special != -1 and launch != -1


def canonical(observation):
    return json.dumps(observation, separators=(',', ':'), sort_keys=False)


def collect(binary, fog):
    """Reset, settle the fog, then harvest public observations."""
    frames = []
    with PvZEnv(RESOURCE, executable=binary) as env:
        reset(env, fog=fog, preplanted=PREPLANTED)
        env.step(dict(type='wait', ticks=1000))
        for _ in range(8):
            observation, *_ = env.step(dict(type='wait', ticks=500))
            frames.append(canonical(observation))
    return frames


@engine
@old_engine
def test_human_and_xray_observations_are_bit_identical_to_pre_b5():
    """The task66 dui-pai: human / xray / None must not change one byte."""
    for fog in ('human', 'xray', None):
        old_frames = collect(OLD_BINARY, fog)
        new_frames = collect(BINARY, fog)
        assert new_frames == old_frames, fog


@engine
@old_engine
def test_human_v2_changes_only_the_masked_plant_fields():
    """Against the pre-B5 engine, the only allowed diffs are the five fields -> -1."""
    old_frames = collect(OLD_BINARY, 'human_v2')
    new_frames = collect(BINARY, 'human_v2')
    assert len(old_frames) == len(new_frames)
    for old_raw, new_raw in zip(old_frames, new_frames):
        if old_raw == new_raw:
            continue
        old, new = json.loads(old_raw), json.loads(new_raw)
        old_plants, new_plants = old.pop('plants'), new.pop('plants')
        old_v2_plants, new_v2_plants = old['v2'].pop('plants'), new['v2'].pop('plants')
        assert canonical(old) == canonical(new), 'everything outside plants must be identical'
        by_key = {(p['row'], p['col'], p['type']): p for p in new_plants}
        assert len(by_key) == len(old_plants) == len(new_plants)
        for plant in old_plants:
            fresh = by_key[(plant['row'], plant['col'], plant['type'])]
            for field in MASKED_FIELDS:
                assert fresh[field] == plant[field] or fresh[field] == -1, (field, plant, fresh)
        old_v2 = {(p[0], p[1], p[2]): p for p in old_v2_plants}
        for plant in new_v2_plants:
            key = (plant[0], plant[1], plant[2])
            was = old_v2[key]
            for index, field in zip(range(3, 7), ('state', 'state_countdown',
                                                  'do_special_countdown', 'launch_counter')):
                assert plant[index] == was[index] or plant[index] == -1, (field, was, plant)
            assert plant[7] == was[7], 'launch_rate is type-constant and must stay real'


def privileged_logic(binary, fog):
    """The full privileged state minus the new-binary-only fog_plants diagnostics."""
    with PvZEnv(RESOURCE, executable=binary) as env:
        reset(env, fog=fog, preplanted=PREPLANTED)
        env.step(dict(type='wait', ticks=1000))
        env.step(dict(type='wait', ticks=500))
        state = env.privileged_state()
        state.pop('fog_plants', None)
        return canonical(state)


@engine
@old_engine
def test_masking_is_presentation_same_privileged_state():
    """Same actions drive both engines to identical logic states, in every fog mode;
    within one engine the privileged state is fog-mode independent (masking is
    presentation only)."""
    for fog in ('human', 'human_v2', 'xray'):
        old_state = privileged_logic(OLD_BINARY, fog)
        new_state = privileged_logic(BINARY, fog)
        assert new_state == old_state, fog
    new_states = {privileged_logic(BINARY, fog) for fog in ('human', 'human_v2', 'xray')}
    assert len(new_states) == 1
