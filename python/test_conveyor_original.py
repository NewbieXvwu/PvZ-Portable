"""Original conveyor start (task20): empty engine belt, no task deck, old tasks untouched."""
import json
import os
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_env import TaskSpec, profile_for_deck
from pvz_event_env import EventWaitEnv
from pvz_rollout_policy import packet_features
from render_episode import _frame

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', str(Path.home() / '.cache/pvz-research-resources')))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', str(ROOT / 'build/pvz-portable')))
DECK = (0, 1, 2, 3, 4, 5, 7)


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / 'main.pak').is_file(), 'needs local CPU build and resources')
class ConveyorOriginalTest(unittest.TestCase):
    def setUp(self):
        self.env = EventWaitEnv(RESOURCE, executable=BINARY, headless=True)

    def tearDown(self):
        self.env.close()

    def test_original_start_is_an_empty_ten_packet_engine_belt(self):
        task = TaskSpec(level=10, seed=30000, conveyor_start='original', profile=profile_for_deck(()))
        obs, _ = self.env.reset(deck=(), task=task)
        frame = _frame(obs)
        self.assertEqual(frame['conveyor_start'], 'original')
        self.assertEqual(len(frame['packets']), 10)
        self.assertTrue(all(kind < 0 for _slot, kind, *_rest in frame['packets']))
        self.assertEqual(obs['legal_actions']['plants'], [])
        self.assertEqual(frame['sun'], 50)
        # Challenge::UpdateConveyorBelt deals the first card at 200 ticks plus the belt delay;
        # it is already legal in the frame it first appears (sliding does not gate CanPickUp).
        first = None
        for _ in range(20):
            obs, _, done, _, _ = self.env.step({'type': 'wait', 'ticks': 60})
            frame = _frame(obs)
            if any(kind >= 0 for _slot, kind, *_rest in frame['packets']):
                first = (frame['tick'], obs['legal_actions']['plants'])
                break
        self.assertIsNotNone(first, 'the engine never dealt a card')
        self.assertTrue(first[1], 'a dealt card must be legal in the same frame')
        self.assertLess(first[0], 600)
        peak = 0
        for _ in range(80):
            obs, _, done, _, _ = self.env.step({'type': 'wait', 'ticks': 60})
            frame = _frame(obs)
            peak = max(peak, sum(1 for _slot, kind, *_rest in frame['packets'] if kind >= 0))
            if done:
                break
        self.assertEqual(peak, 10, 'the belt holds ten cards')

    def test_task_deck_tasks_keep_the_old_frame_and_reset(self):
        task = TaskSpec(level=10, seed=30000, profile=profile_for_deck(DECK))
        obs, _ = self.env.reset(deck=DECK, task=task)
        frame = _frame(obs)
        self.assertNotIn('conveyor_start', frame)
        self.assertEqual([kind for _slot, kind, *_rest in frame['packets']], list(DECK))
        self.assertEqual(self.env.episode['task']['conveyor_start'], None)

    def test_non_conveyor_levels_reject_the_option(self):
        with self.assertRaises(ValueError):
            self.env.reset(deck=(), task=TaskSpec(level=7, seed=30000, conveyor_start='original',
                                                  profile=profile_for_deck(())))

    def test_empty_deck_requires_the_option(self):
        with self.assertRaises(ValueError):
            self.env.reset(deck=(), task=TaskSpec(level=10, seed=30000, profile=profile_for_deck(())))

    def test_packet_features_follow_the_engine_flag_not_a_python_rule(self):
        task = TaskSpec(level=10, seed=30000, conveyor_start='original', profile=profile_for_deck(()))
        obs, _ = self.env.reset(deck=(), task=task)
        for _ in range(20):
            obs, _, done, _, _ = self.env.step({'type': 'wait', 'ticks': 60})
            if any(packet['type'] >= 0 for packet in obs['packets']):
                break
        frame = _frame(obs)
        features = packet_features(frame)
        for slot, kind, *_rest in frame['packets']:
            if kind >= 0:
                self.assertEqual(float(features[slot][3]), 1.0)
        # Same packet, no engine flag: the nominal cost decides again.
        synthetic = {'sun': 0, 'conveyor_start': 'original', 'packets': [[0, 1, True, 0, 0, 100]]}
        self.assertEqual(float(packet_features(synthetic)[0][3]), 1.0)
        legacy = dict(synthetic)
        legacy.pop('conveyor_start')
        self.assertEqual(float(packet_features(legacy)[0][3]), 0.0)

    def test_episode_records_the_start_mode_for_replay(self):
        task = TaskSpec(level=10, seed=30000, conveyor_start='original', profile=profile_for_deck(()))
        self.env.reset(deck=(), task=task)
        self.assertEqual(self.env.episode['task']['conveyor_start'], 'original')
        self.assertEqual(self.env.episode['deck'], [])


if __name__ == '__main__':
    unittest.main()
