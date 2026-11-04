"""task237: scenario placement commands (SPAWN / SET_SUN / SET_COOLDOWN / SET_WAVE).

These are additive engine actions.  The tests cover the wire encoding without an engine,
the type-specific placement/behaviour on a real headless board, archive replay, and the
``Game.from_archive(..., decision=N)`` continuation used by the GPT drill package.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts'), str(ROOT / 'scripts' / 'llmplay')]

# Keep llmplay session/out state out of the repository (same convention as test_llmplay).
_SESSION_DIR = tempfile.mkdtemp(prefix='task237-test-')
os.environ.setdefault('PVZ_LLMPLAY_SESSION', str(Path(_SESSION_DIR) / 'budget.json'))
os.environ.setdefault('LLMPLAY_OUT', str(Path(_SESSION_DIR) / 'out'))

from pvz_env import PvZEnv, TaskSpec  # noqa: E402
from pvz_tasks import deck_for_level, profile_for_deck  # noqa: E402

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', str(Path.home() / '.cache/pvz-research-resources')))
BINARY = Path(os.environ.get('PVZ_SCENARIO_EXECUTABLE', os.environ.get(
    'PVZ_EXECUTABLE', str(Path.home() / 'pvz-mac-collect/build-scenario/pvz-portable'))))


def can_run():
    return BINARY.is_file() and (RESOURCE / 'main.pak').is_file()


def _observation():
    return {"tick": 0, "wave": 0, "wave_count": 20, "wave_timer": 3000, "sun": 50,
            "plants": [], "zombies": [], "packets": [], "terminal": False, "result": 0}


class EncodingTest(unittest.TestCase):
    """No child process: the wrapper must send exactly the protocol the C++ parser reads."""

    def setUp(self):
        self.env = PvZEnv(resource_dir='/nonexistent')
        self.env._reset_done = True
        self.env._tick = 0
        self.commands = []

        def command(text):
            self.commands.append(text)
            return {"ok": True, "observation": _observation(), "events": {}}

        self.env._command = command  # type: ignore[method-assign]

    def test_spawn_set_sun_and_cooldown_wire_format(self):
        self.env.spawn(18, 2, 300.5)
        self.env.spawn(20, 2, 640.0, target_col=7)
        self.env.set_sun(777)
        self.env.set_cooldown(3, 120)
        self.env.set_wave(5, 60)
        self.assertEqual(self.commands, [
            'SPAWN 18 2 300.5',
            'SPAWN 20 2 640 7',
            'SET_SUN 777',
            'SET_COOLDOWN 3 120',
            'SET_WAVE 5 60',
        ])

    def test_invalid_arguments_never_reach_the_child(self):
        for action, message in (
                ({"type": "spawn", "zombie_type": 18, "row": 2, "x": float("nan")}, 'finite'),
                ({"type": "spawn", "zombie_type": 18, "row": 2, "x": 2000}, 'finite'),
                ({"type": "spawn", "zombie_type": 18, "row": 2, "x": 1, "target_col": 1.5}, 'integer'),
                ({"type": "spawn", "zombie_type": 33, "row": 2, "x": 1}, '0..32'),
                ({"type": "set_sun", "sun": 1.5}, 'integer'),
                ({"type": "set_cooldown", "slot": 0, "ticks": -1}, 'ticks'),
                ({"type": "set_wave", "wave": 2, "countdown": 1.5}, 'integer')):
            with self.subTest(action=action):
                with self.assertRaisesRegex(ValueError, message):
                    self.env.step(action)
        self.assertEqual(self.commands, [])


@unittest.skipUnless(can_run(), 'requires the scenario engine and game resources')
class EngineBehaviourTest(unittest.TestCase):
    def _env(self, level):
        deck = deck_for_level(level)
        env = PvZEnv(RESOURCE, executable=BINARY)
        self.addCleanup(env.close)
        observation, _ = env.reset(deck=deck, task=TaskSpec(
            level=level, seed=30000, profile=profile_for_deck(deck)))
        return env, observation

    def _spawn_and_wait(self, level, zombie_type, row, x, wait):
        env, _ = self._env(level)
        observation = env.spawn(zombie_type, row, x)
        initial = [z for z in observation['zombies'] if z['type'] == zombie_type]
        self.assertTrue(initial, f'zombie {zombie_type} was not placed')
        observation = env.step({'type': 'wait', 'ticks': wait})[0]
        after = [z for z in observation['zombies'] if z['type'] == zombie_type]
        return initial, after

    def test_listed_counter_zombies_are_placed_and_move(self):
        cases = (
            (18, 1, 2, 300.0, 120, 20),       # pogo: bouncing phase
            (16, 1, 2, 300.0, 120, 73),       # balloon: flying phase
            (17, 1, 2, 120.0, 180, 32),       # digger: tunneling phase, then rises
            (14, 23, 2, 600.0, 120, 53),      # dolphin: riding in the pool
            (11, 23, 3, 600.0, 120, 59),      # snorkel: swimming in the pool
            (7, 1, 2, 600.0, 120, 0),         # football
            (13, 1, 2, 600.0, 180, 17),       # bobsled team
            (22, 1, 2, 600.0, 120, 0),        # catapult
            (15, 1, 2, 600.0, 120, 15),       # jack-in-the-box
        )
        for zombie_type, level, row, x, wait, phase in cases:
            with self.subTest(zombie_type=zombie_type):
                initial, after = self._spawn_and_wait(level, zombie_type, row, x, wait)
                self.assertEqual(initial[0]['row'], row)
                self.assertAlmostEqual(initial[0]['x'], x, places=3)
                self.assertEqual(initial[0]['phase'], phase)
                self.assertTrue(after, f'zombie {zombie_type} disappeared')
                if zombie_type == 13:
                    # The sled is four zombies and must keep its runners on the generated ice.
                    self.assertEqual(len(initial), 4)
                    self.assertGreater(after[0]['helm_health'], 0)
                if zombie_type == 17:
                    self.assertNotEqual(after[0]['phase'], 32)
                else:
                    self.assertLess(after[0]['x'], initial[0]['x'])

    def test_bungee_targets_the_requested_cell(self):
        env, _ = self._env(1)
        observation = env.spawn(20, 2, 640.0)
        bungee = [z for z in observation['zombies'] if z['type'] == 20][0]
        self.assertEqual((bungee['target_col'], bungee['row'], bungee['phase']), (7, 2, 4))
        observation = env.step({'type': 'wait', 'ticks': 300})[0]
        bungee = [z for z in observation['zombies'] if z['type'] == 20][0]
        self.assertIn(bungee['phase'], (4, 5, 6))
        self.assertAlmostEqual(bungee['x'], 600.0, places=3)

    def test_illegal_terrain_combinations_are_rejected(self):
        env, _ = self._env(45)  # roof
        cases = ((14, 2, 400.0), (11, 2, 400.0), (13, 2, 400.0), (12, 2, 400.0))
        for zombie_type, row, x in cases:
            with self.subTest(zombie_type=zombie_type):
                observation, _, _, _, info = env.step(
                    {'type': 'spawn', 'zombie_type': zombie_type, 'row': row, 'x': x})
                self.assertFalse(info['ok'])
                self.assertFalse([z for z in observation['zombies'] if z['type'] == zombie_type])

    def test_set_sun_and_cooldown_are_exact(self):
        env, observation = self._env(1)
        self.assertEqual(env.set_sun(1234)['sun'], 1234)
        self.assertEqual(env.set_sun(9990)['sun'], 9990)
        with self.assertRaisesRegex(ValueError, '0..9990'):
            env.set_sun(9991)

        observation = env.set_cooldown(0, 60)
        self.assertFalse(observation['packets'][0]['active'])
        observation = env.step({'type': 'wait', 'ticks': 59})[0]
        self.assertFalse(observation['packets'][0]['active'])
        observation = env.step({'type': 'wait', 'ticks': 1})[0]
        self.assertTrue(observation['packets'][0]['active'])

        self.assertTrue(env.set_cooldown(1, 0)['packets'][1]['active'])
        _, _, _, _, info = env.step({'type': 'set_cooldown', 'slot': 99, 'ticks': 10})
        self.assertFalse(info['ok'])

    def test_set_wave_moves_forward_and_starts_the_next_wave(self):
        env, observation = self._env(1)
        observation = env.set_wave(3, 60)
        self.assertEqual((observation['wave'], observation['wave_timer']), (3, 60))
        observation = env.step({'type': 'wait', 'ticks': 60})[0]
        self.assertGreaterEqual(observation['wave'], 4)
        _, _, _, _, info = env.step({'type': 'set_wave', 'wave': 1, 'countdown': 10})
        self.assertFalse(info['ok'])


@unittest.skipUnless(can_run(), 'requires the scenario engine and game resources')
class ArchiveReplayTest(unittest.TestCase):
    def test_new_commands_replay_to_the_same_state_hash(self):
        from human_demo import Recorder, replay

        deck = deck_for_level(1)
        spec = TaskSpec(level=1, seed=30000, profile=profile_for_deck(deck))
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'archive'
            env = PvZEnv(RESOURCE, executable=BINARY)
            try:
                recorder = Recorder(env, archive, 'task237_archive', deck, spec)
                recorder.step({'type': 'spawn', 'zombie_type': 18, 'row': 2, 'x': 300.0})
                recorder.step({'type': 'set_sun', 'sun': 777})
                recorder.step({'type': 'set_cooldown', 'slot': 0, 'ticks': 60})
                recorder.step({'type': 'wait', 'ticks': 90})
                expected = env.state_hash()
                recorder.close()
            finally:
                env.close()
            with PvZEnv(RESOURCE, executable=BINARY) as replay_env:
                observation, _, _ = replay(replay_env, archive)
                self.assertEqual(observation['sun'], 777)
                self.assertEqual(replay_env.state_hash(), expected)

    def test_each_special_zombie_archive_round_trips(self):
        from human_demo import Recorder, replay

        cases = ((18, 1, 2, 300.0, 120), (16, 1, 2, 300.0, 120),
                 (17, 1, 2, 120.0, 180), (14, 23, 2, 600.0, 120),
                 (11, 23, 3, 600.0, 120), (7, 1, 2, 600.0, 120),
                 (13, 1, 2, 600.0, 180), (22, 1, 2, 600.0, 120),
                 (20, 1, 2, 640.0, 300), (15, 1, 2, 600.0, 120))
        for zombie_type, level, row, x, wait in cases:
            with self.subTest(zombie_type=zombie_type):
                deck = deck_for_level(level)
                spec = TaskSpec(level=level, seed=30000, profile=profile_for_deck(deck))
                with tempfile.TemporaryDirectory() as directory:
                    archive = Path(directory) / 'archive'
                    env = PvZEnv(RESOURCE, executable=BINARY)
                    try:
                        recorder = Recorder(env, archive, f'task237_z{zombie_type}', deck, spec)
                        recorder.step({'type': 'spawn', 'zombie_type': zombie_type,
                                       'row': row, 'x': x})
                        recorder.step({'type': 'wait', 'ticks': wait})
                        expected = env.state_hash()
                        recorder.close()
                    finally:
                        env.close()
                    with PvZEnv(RESOURCE, executable=BINARY) as replay_env:
                        observation, _, _ = replay(replay_env, archive)
                        self.assertEqual(replay_env.state_hash(), expected)
                        self.assertTrue([z for z in observation['zombies']
                                         if z['type'] == zombie_type])

    def test_game_from_archive_decision_can_continue(self):
        from human_demo import Recorder
        from llmplay.game import Game

        deck = deck_for_level(1)
        spec = TaskSpec(level=1, seed=30000, profile=profile_for_deck(deck))
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'archive'
            env = PvZEnv(RESOURCE, executable=BINARY)
            try:
                recorder = Recorder(env, archive, 'task237_game', deck, spec)
                recorder.step({'type': 'spawn', 'zombie_type': 18, 'row': 2, 'x': 300.0})
                recorder.step({'type': 'wait', 'ticks': 60})
                recorder.close()
            finally:
                env.close()
            game = Game.from_archive(archive, decision=2, executable=BINARY,
                                     resource_dir=RESOURCE, _gate=False)
            try:
                self.assertTrue(game.spawn(16, 1, 500.0))
                self.assertTrue(game.set_sun(900))
                self.assertEqual(game.obs['sun'], 900)
                self.assertTrue(game.set_cooldown(1, 30))
                types = {z['type'] if isinstance(z, dict) else int(z[1])
                         for z in game.obs['zombies']}
                self.assertEqual(types, {18, 16})
            finally:
                game.close()


PRODUCTION = Path(os.environ.get('PVZ_PRODUCTION_EXECUTABLE', ''))


@unittest.skipUnless(can_run() and PRODUCTION.is_file(),
                     'set PVZ_PRODUCTION_EXECUTABLE to compare against production')
class ProductionParityTest(unittest.TestCase):
    """Sampled no-new-command games must be bit-identical with the production engine."""

    def _run(self, binary, level, seed, preplanted):
        deck = deck_for_level(level)
        actions = [
            {'type': 'wait', 'ticks': 1200},
            {'type': 'plant', 'packet': 0, 'col': 0, 'row': 2},
            {'type': 'wait', 'ticks': 300},
            {'type': 'plant', 'packet': 1, 'col': 1, 'row': 2},
            {'type': 'wait', 'ticks': 300},
            {'type': 'plant', 'packet': 0, 'col': 0, 'row': 1},
            {'type': 'wait', 'ticks': 300},
            {'type': 'plant', 'packet': 1, 'col': 1, 'row': 1},
            {'type': 'wait', 'ticks': 600},
        ]
        with PvZEnv(RESOURCE, executable=binary) as env:
            observation, _ = env.reset(deck=deck, task=TaskSpec(
                level=level, seed=seed, profile=profile_for_deck(deck),
                preplanted=tuple(preplanted)))
            trace = [(observation['tick'], observation['sun'], observation['wave'], env.state_hash())]
            for action in actions:
                observation, _, _, _, info = env.step(action)
                trace.append((bool(info['ok']), observation['tick'], observation['sun'],
                              observation['wave'], env.state_hash()))
            return trace

    def test_sampled_games_match_production(self):
        cases = ((1, 30000, ()), (23, 424242, ((16, 2, 0),)), (45, 700001, ()))
        for level, seed, preplanted in cases:
            with self.subTest(level=level, seed=seed):
                self.assertEqual(self._run(BINARY, level, seed, preplanted),
                                 self._run(PRODUCTION, level, seed, preplanted))
