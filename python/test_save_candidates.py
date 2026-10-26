"""Save candidates: the teacher can price a card it cannot afford yet (task120).

The engine only lists plant actions the player can pay for, so a card above the sun the
policy ever holds never reaches the candidate set. These tests pin the opt-in probe: off by
default, the decision becomes a wait only when the future plant beats the best candidate by
MARGIN, and the probe never enters the export operation stream.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

import search_play as sp


class FakeEnv:
    def __init__(self, sun=70, future_sun=130, cost=125, legal_at_future=True, terminal=False):
        self.episode = {'operations': []}
        self._sun = sun
        self._future_sun = future_sun
        self._cost = cost
        self._legal_at_future = legal_at_future
        self._terminal = terminal
        self._tick = 0
        self.restores = 0
        self.episode_during_step = []
        self.snapshots = 0
        self.released = []

    def restore_fast(self, snapshot):
        self.restores += 1
        self._sun = 70
        self._tick = 0

    def step(self, action):
        assert action['type'] == 'wait'
        self.episode_during_step.append(self.episode)
        advanced = int(action['ticks'])
        self._tick += advanced
        self._sun = self._future_sun
        return self._obs(), 0.0, self._terminal, False, {'ticks_advanced': advanced}

    def _obs(self):
        plants = [{'type': 'plant', 'packet': 0, 'col': 0, 'row': 0}] if self._legal_at_future else []
        return {'tick': self._tick, 'sun': self._sun, 'terminal': self._terminal,
                'legal_actions': {'plants': plants}, 'packets': [{'type': 26, 'cost': self._cost}],
                'sun_income_rate': 0.1}

    def snapshot_fast(self):
        self.snapshots += 1
        return self.snapshots

    def release_snapshot(self, snapshot):
        self.released.append(snapshot)

    def static_scores(self, snapshot, actions, income):
        return [0.5 for _ in actions]


def _args(**overrides):
    base = dict(samples=3, verify_samples=9, save_candidates=1, save_share=0.5,
                save_max_wait=1800, save_chunk=300, save_interval=0, horizon=2500,
                leaf_probe=None, min_gap=0, task_id='t')
    base.update(overrides)
    return SimpleNamespace(**base)


def _obs():
    return {'tick': 0, 'sun': 70, 'terminal': False, 'sun_income_rate': 0.1,
            'packets': [{'type': 26, 'cost': 125}],
            'legal_actions': {'plants': [{'type': 'plant', 'packet': 1, 'col': 0, 'row': 0}]}}


def _install(monkeypatch, value):
    monkeypatch.setattr(sp, '_rollout_policy', lambda *a, **k: {'leaf': True})
    monkeypatch.setattr(sp, '_score_terminal_or_value', lambda leaf: None)
    monkeypatch.setattr(sp, '_leaf_values', lambda leaves: [value for _ in leaves])


def test_off_by_default(monkeypatch):
    env = FakeEnv()
    called = []
    monkeypatch.setattr(sp, '_save_probe', lambda *a, **k: called.append(1) or None)
    stats = {}
    assert sp._save_plan(env, _obs(), 1, [{'type': 'plant', 'packet': 1}], [0.1], _args(save_candidates=0), 1, stats) is None
    assert not called and not stats


def test_plan_wins_and_returns_wait(monkeypatch):
    env = FakeEnv()
    _install(monkeypatch, 0.40)
    plan = sp._save_plan(env, _obs(), 1, [{'type': 'plant', 'packet': 1}], [0.10], _args(), 7, {})
    assert plan and plan['packet'] == 0 and plan['cell'] == [0, 0] and plan['ticks'] == 300
    stats = {}
    wait = sp._verified_choice(env, _obs(), 1, [{'type': 'plant', 'packet': 1}], [0.10], [3], 0, 0,
                               {'type': 'plant', 'packet': 1, 'col': 0, 'row': 0}, _args(), 7, stats)
    assert wait == {'type': 'wait', 'ticks': 300, 'until': 'timeout'}
    assert stats['save_overrides'] == 1 and stats['save_wait_ticks'] == 300


def test_plan_loses_keeps_student(monkeypatch):
    env = FakeEnv()
    _install(monkeypatch, 0.10)
    stats = {}
    policy = {'type': 'plant', 'packet': 1, 'col': 0, 'row': 0}
    chosen = sp._verified_choice(env, _obs(), 1, [policy], [0.10], [3], 0, 0, policy, _args(), 7, stats)
    assert chosen == policy and stats.get('save_overrides') is None


def test_probe_keeps_export_stream_clean(monkeypatch):
    env = FakeEnv()
    _install(monkeypatch, 0.40)
    sp._save_plan(env, _obs(), 1, [{'type': 'plant', 'packet': 1}], [0.10], _args(), 7, {})
    assert env.episode_during_step and all(item is None for item in env.episode_during_step)
    assert env.episode is not None and env.episode['operations'] == []


def test_unreachable_card_is_not_a_plan(monkeypatch):
    env = FakeEnv(future_sun=60)
    _install(monkeypatch, 0.90)
    assert sp._save_plan(env, _obs(), 1, [{'type': 'plant', 'packet': 1}], [0.10], _args(), 7, {}) is None
    assert env.released == []


def test_no_legal_cell_after_wait(monkeypatch):
    env = FakeEnv(legal_at_future=False)
    _install(monkeypatch, 0.90)
    assert sp._save_plan(env, _obs(), 1, [{'type': 'plant', 'packet': 1}], [0.10], _args(), 7, {}) is None
    assert env.snapshots == 0 and env.released == []
