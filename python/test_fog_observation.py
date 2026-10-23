"""Task45: public fog filtering, opening fog, and Python/native student parity."""
import os
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_env import PvZEnv, TaskSpec
from pvz_rollout_policy import RolloutNet, export_student
from pvz_student_observation import StudentHistory, fog_channel
from pvz_tasks import deck_for_level, profile_for_deck
from render_episode import _frame, training_task_kwargs

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', '/home/newbiexvwu/.cache/pvz-research-resources'))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', ROOT / 'build/pvz-portable'))
engine = pytest.mark.skipif(not BINARY.is_file() or not (RESOURCE / 'main.pak').is_file(),
                            reason='requires simulator and resources')


def reset(env, mode, preplanted=()):
    deck = deck_for_level(31)
    return env.reset(deck=deck, task=TaskSpec(level=31, seed=450031,
                     profile=profile_for_deck(deck), fog_observation=mode,
                     preplanted=preplanted))[0]


@engine
def test_mask_matches_privileged_grid_and_lean_alignment():
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        reset(env, 'human')
        obs, *_ = env.step(dict(type='wait', ticks=8000))
        raw = env.privileged_state()
        expected = [z for z in raw['zombies'] if obs['fog_amount'][z['row']][
            max(0, min(8, int(np.floor((z['x'] - 40) / 80))))] == 0]
        assert raw['zombies'] and len(expected) < len(raw['zombies'])
        assert obs['zombies'] == expected
        assert len(obs['v2']['zombies']) == len(expected)
        assert obs['v2']['wave_health'] == obs['v2']['wave_start_health'] == 0
        full = _frame(obs)
        env.set_lean_observation(True)
        lean = env.observe()
        assert _frame(lean) == full
        assert np.any(fog_channel(full))
        assert training_task_kwargs(dict(level=31, deck=list(deck_for_level(31)),
                   fog_observation='human', seeds=[450031]), 450031)['task_extra']['fog_observation'] == 'human'


@engine
def test_xray_changes_only_observation_and_plantern_clears_engine_fog():
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        states = []
        for mode in ('human', 'xray'):
            reset(env, mode)
            obs, *_ = env.step(dict(type='wait', ticks=8000))
            states.append(env.privileged_state())
            if mode == 'xray':
                assert obs['zombies'] == states[-1]['zombies']
        assert states[0] == states[1]
        reset(env, 'human', ((25, 1, 7),))  # engine Plantern, no handwritten playing policy
        lit, *_ = env.step(dict(type='wait', ticks=500))
        reset(env, 'human')
        dark, *_ = env.step(dict(type='wait', ticks=500))
        assert np.asarray(lit['fog_amount']).sum() < np.asarray(dark['fog_amount']).sum()
        assert lit['fog_amount'][1][7] == 0 < dark['fog_amount'][1][7]


@engine
@pytest.mark.parametrize('mode', ['human', 'xray'])
@pytest.mark.parametrize('version', ['v2', 'v2fog'])
def test_native_student_features_use_same_filtered_observation(mode, version):
    torch.set_num_threads(1)
    torch.manual_seed(45)
    net = RolloutNet(width=13, obs=version).eval()
    with tempfile.TemporaryDirectory() as tmp, PvZEnv(RESOURCE, executable=BINARY) as env:
        path = Path(tmp) / 'net.bin'
        export_student(net, path)
        env.load_rollout_net(path)
        reset(env, mode)
        obs, *_ = env.step(dict(type='wait', ticks=8000))
        history = StudentHistory(version)
        board, scalars = history.features(_frame(obs), None)
        trace = Path(tmp) / 'features.jsonl'
        native = env.rollout_predict(student_history=StudentHistory(version), trace_path=trace)
        encoded = json.loads(trace.read_text().splitlines()[0])
        np.testing.assert_array_equal(board.numpy().ravel(), np.asarray(encoded['board'], dtype=np.float32))
        np.testing.assert_array_equal(scalars.numpy(), np.asarray(encoded['scalars'], dtype=np.float32))
        with torch.inference_mode():
            _h, g = net.encode(board[None], scalars[None])
            assert np.isclose(float(net.plant(g)[0, 0]), native['plant'], atol=1e-5, rtol=0)
            assert np.allclose(net.packet(g).detach().numpy()[0], native['packet'], atol=1e-5, rtol=0)
        env.rollout(None, 11000, student_history=StudentHistory(version), trace_path=trace)
        replay = StudentHistory(version)
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        assert len(records) > 1
        for record in records:
            frame = _frame(record['observation'])
            frame['sun_income_rate'] = record['income']
            frame['action'] = dict(type='plant' if record['previous_plant'] else 'wait')
            previous = dict(tick=record['previous_tick']) if record['previous_tick'] >= 0 else None
            b, s = replay.features(frame, previous)
            for expected, values in ((b.flatten(), record['board']), (s, record['scalars'])):
                np.testing.assert_array_equal(expected.numpy().view(np.uint32),
                                              np.asarray(values, np.float32).view(np.uint32))


def test_fog_channel_zero_for_old_frames():
    assert not fog_channel({}).any()
    with pytest.raises(ValueError, match='6 by 9'):
        fog_channel({'fog_amount': [[0]]})


@engine
def test_new_channel_preserves_all_v2_feature_bits_and_has_distinct_cache_signature():
    from pvz_frame_store import encoder_signature
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        reset(env, 'human')
        obs, *_ = env.step(dict(type='wait', ticks=8000))
        frame = _frame(obs)
        old = StudentHistory('v2').features(frame, None)
        new = StudentHistory('v2fog').features(frame, None)
        assert torch.equal(old[0], new[0][:-1])
        assert torch.equal(old[1], new[1])
        assert encoder_signature('v2') != encoder_signature('v2fog')


def opening_fixture(env, plant):
    deck = deck_for_level(31) + (plant,)
    # A funded native fixture; each test still uses PLANT and engine cooldown/terrain legality.
    economy = tuple((1, row, col) for row in (0, 1, 4, 5) for col in (0, 1))
    env.reset(deck=deck, task=TaskSpec(level=31, seed=450031,
              profile=profile_for_deck(deck), fog_observation='human', preplanted=economy))
    obs, *_ = env.step(dict(type='wait', ticks=8000))
    raw = env.privileged_state()
    hidden = {z['id'] for z in raw['zombies']} - {z['id'] for z in obs['zombies']}
    assert hidden
    return obs, raw, hidden


def check_lamp(plant):
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        obs, raw, hidden = opening_fixture(env, plant)
        target = next(z for z in raw['zombies'] if z['id'] in hidden and z['row'] in (0, 1, 4, 5))
        row = target['row']
        action = dict(type='plant', packet=8, row=row, col=7)
        assert {k: v for k, v in action.items() if k != 'type'} in obs['legal_actions']['plants']
        env.step(action)
        lit, *_ = env.step(dict(type='wait', ticks=150))
        assert lit['fog_amount'][row][7] < obs['fog_amount'][row][7]
        assert hidden & {z['id'] for z in lit['zombies']}


@engine
def test_plantern_reveals_previously_masked_zombie():
    check_lamp(25)


@engine
def test_torchwood_reveals_previously_masked_zombie():
    check_lamp(22)


@engine
def test_blover_reveals_zombies_then_fog_returns_and_masks_again():
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        obs, raw, hidden = opening_fixture(env, 27)
        env.step(dict(type='plant', packet=8, row=0, col=2))
        clear, *_ = env.step(dict(type='wait', ticks=300))
        assert np.asarray(clear['fog_amount']).sum() < np.asarray(obs['fog_amount']).sum()
        assert hidden & {z['id'] for z in clear['zombies']}
        back, *_ = env.step(dict(type='wait', ticks=5000))
        actual = env.privileged_state()
        assert np.asarray(back['fog_amount']).sum() > np.asarray(clear['fog_amount']).sum()
        assert {z['id'] for z in actual['zombies']} - {z['id'] for z in back['zombies']}


@engine
def test_half_multiplier_is_accepted_and_scales_actual_engine_spawns():
    import math
    deck = deck_for_level(31)
    with PvZEnv(RESOURCE, executable=BINARY) as env:
        obs, _ = env.reset(deck=deck, task=TaskSpec(level=31, seed=450031,
              profile=profile_for_deck(deck), zombie_count_multiplier=.5,
              fog_observation='xray'))
        assert obs['zombie_count_multiplier'] == .5
        original = [len(env.critic_inputs(wave)['wave_zombies']) for wave in range(6)]
        previous_ids = set()
        actual = []
        while obs['wave'] < 6 and not obs['terminal']:
            before = obs['wave']
            obs, *_ = env.step(dict(type='wait', ticks=1))
            identities = {z['id'] for z in obs['zombies']}
            if obs['wave'] > before:
                actual.append(len(identities - previous_ids))
            previous_ids = identities
        assert len(actual) == 6
        assert actual == [math.floor(n * .5 + .5) for n in original]
        assert sum(actual) < sum(original)
