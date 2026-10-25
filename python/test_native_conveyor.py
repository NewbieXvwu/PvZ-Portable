"""task75 native conveyor family contracts."""
import sys
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

from make_native_conveyor_panel import build_tasks
from pvz_tasks import conveyor_belt_level, deck_for_seed
from pvz_env import PvZEnv, TaskSpec, profile_for_deck
from pvz_rollout_policy import RolloutNet, export_student
from pvz_student_observation import StudentHistory
from render_episode import _frame

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', '/home/newbiexvwu/.cache/pvz-research-resources'))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', ROOT / 'build/pvz-portable'))
engine = pytest.mark.skipif(not BINARY.is_file() or not (RESOURCE / 'main.pak').is_file(),
                            reason='requires simulator and resources')


def _public_entities(entities):
    """Public frames hide the target id; every other field must match privileged."""
    return [{key: value for key, value in entity.items() if key != 'target_zombie_id'}
            for entity in entities]


def reset_native(env, level, mode=None, preplanted=()):
    return env.reset(deck=(), task=TaskSpec(level=level, seed=61_000_000 + (25, 30, 40).index(level) * 100_000,
                     conveyor_start='original', profile=profile_for_deck(()),
                     fog_observation=mode, preplanted=preplanted))[0]


def test_native_conveyor_manifest_preserves_engine_dealing():
    tasks = build_tasks()
    assert len(tasks) == 6
    seeds = set()
    for task in tasks:
        assert conveyor_belt_level(task['level'])
        assert task['deck'] == [] and task['conveyor_start'] == 'original'
        assert task['wave_cap'] is None and task['zombie_count_multiplier'] == 1
        assert task['playthrough'] == 2 and task['preplanted'] == []
        assert not seeds.intersection(task['seeds'])
        seeds.update(task['seeds'])
        assert deck_for_seed(task, task['seeds'][0]) == ()
        assert len(task['seeds']) == (32 if task['task_id'].endswith('_eval') else 8192)
        assert task.get('fog_observation') == ('storm' if task['level'] == 40 else None)


@engine
@pytest.mark.parametrize('level', [25, 30, 40])
def test_native_reset_keeps_ten_empty_slots_and_engine_deals_duplicates(level):
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        obs = reset_native(env, level, 'storm' if level == 40 else None)
        assert len(obs['packets']) == 10
        assert all(packet['type'] == -1 for packet in obs['packets'])
        obs, *_ = env.step(dict(type='wait', ticks=8000))
        kinds = [packet['type'] for packet in obs['packets'] if packet['type'] >= 0]
        assert len(kinds) == 10
        assert len(set(kinds)) < len(kinds)
        assert obs['legal_actions']['plants']


@engine
@pytest.mark.parametrize('level', [25, 30, 40])
def test_native_family_plays_a_full_game_to_terminal(level):
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        obs = reset_native(env, level, 'storm' if level == 40 else None)
        dealt = set()
        for _ in range(80):
            dealt.update(packet['type'] for packet in obs['packets'] if packet['type'] >= 0)
            if obs['terminal']:
                break
            obs, *_ = env.step(dict(type='wait', ticks=500))
        assert obs['terminal'], obs['tick']
        assert obs['result'] in (1, 2)  # BOARDRESULT_WON / BOARDRESULT_LOST
        assert obs['wave'] > 1
        assert len(dealt) >= 2


@engine
def test_little_zombie_size_survives_full_lean_v2_and_native_student():
    torch.set_num_threads(1)
    torch.manual_seed(75)
    with tempfile.TemporaryDirectory() as folder, PvZEnv(RESOURCE, executable=BINARY) as env:
        path = Path(folder) / 'size.bin'
        model = RolloutNet(width=8, obs='v2size').eval()
        export_student(model, path)
        env.load_rollout_net(path)
        reset_native(env, 25)
        obs, *_ = env.step(dict(type='wait', ticks=8000))
        assert obs['zombies']
        assert all(zombie['scale'] == 0.5 for zombie in obs['zombies'])
        assert all(zombie[3] == 0.5 for zombie in obs['v2']['zombies'])
        frame = _frame(obs)
        env.set_lean_observation(True)
        assert _frame(env.observe()) == frame
        board, scalars = StudentHistory('v2size').features(frame, None)
        assert board[-1].sum() > 0
        trace = Path(folder) / 'features.jsonl'
        env.rollout_predict(student_history=StudentHistory('v2size'), trace_path=trace)
        env.observe()
        native = json.loads(trace.read_text().splitlines()[0])
        np.testing.assert_array_equal(board.numpy().ravel(), np.asarray(native['board'], np.float32))
        np.testing.assert_array_equal(scalars.numpy(), np.asarray(native['scalars'], np.float32))


@engine
def test_storm_masks_scene_but_preserves_raw_state_and_lean_alignment():
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        reset_native(env, 40, 'storm', ((0, 1, 0), (3, 1, 1)))
        obs, *_ = env.step(dict(type='wait', ticks=8000))
        seen = set()
        lit_projectile_ticks = 0
        for _ in range(180):
            raw = env.privileged_state()
            dark = obs['fog_amount'][0][0] == 255
            if raw['zombies']:
                seen.add(dark)
            if dark:
                for key in ('zombies', 'plants', 'projectiles', 'defenses', 'grid_items', 'coins'):
                    assert obs[key] == []
                assert obs['v2']['plants'] == obs['v2']['zombies'] == obs['v2']['projectiles'] == []
                assert not obs['enemy_zombies_on_screen']
                assert all(cell['plant_types'] == cell['grid_item_types'] == [] for cell in obs['cells'])
            else:
                for key in ('zombies', 'plants', 'projectiles', 'defenses', 'grid_items', 'coins'):
                    assert _public_entities(obs[key]) == _public_entities(raw[key]), key
                if raw['projectiles']:
                    lit_projectile_ticks += 1
            assert obs['v2']['wave_health'] == obs['v2']['wave_start_health'] == 0
            env.set_lean_observation(True)
            assert _frame(env.observe()) == _frame(obs)
            env.set_lean_observation(False)
            if seen == {False, True}:
                break
            obs, *_ = env.step(dict(type='wait', ticks=30))
        assert seen == {False, True}
        assert lit_projectile_ticks, 'no lit tick carried a projectile; the parity check would be empty'
        storm_state, storm_hash = env.privileged_state(), env.state_hash()
        tick = obs['tick']
        reset_native(env, 40, 'xray', ((0, 1, 0), (3, 1, 1)))
        env.step(dict(type='wait', ticks=tick))
        assert env.state_hash() == storm_hash
        assert env.privileged_state() == storm_state


@engine
@pytest.mark.parametrize('version', ['v2', 'v2fog'])
def test_storm_python_native_features_match_during_blackout_and_flash(version):
    torch.set_num_threads(1)
    torch.manual_seed(75)
    with tempfile.TemporaryDirectory() as folder, PvZEnv(RESOURCE, executable=BINARY) as env:
        path = Path(folder) / 'storm.bin'
        export_student(RolloutNet(width=8, obs=version).eval(), path)
        env.load_rollout_net(path)
        obs = reset_native(env, 40, 'storm', ((0, 1, 0),))
        obs, *_ = env.step(dict(type='wait', ticks=8000))
        seen = set()
        trace = Path(folder) / 'features.jsonl'
        for _ in range(180):
            dark = obs['fog_amount'][0][0] == 255
            if dark not in seen and obs['legal_actions']['plants']:
                board, scalars = StudentHistory(version).features(_frame(obs), None)
                env.rollout_predict(student_history=StudentHistory(version), trace_path=trace)
                env.observe()
                native = json.loads(trace.read_text().splitlines()[-1])
                np.testing.assert_array_equal(board.numpy().ravel(), np.asarray(native['board'], np.float32))
                np.testing.assert_array_equal(scalars.numpy(), np.asarray(native['scalars'], np.float32))
                seen.add(dark)
            if seen == {False, True}:
                break
            obs, *_ = env.step(dict(type='wait', ticks=30))
        assert seen == {False, True}
