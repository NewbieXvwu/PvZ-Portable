"""Preserve seed identity and distinguish idle action budgets from losses."""
import unittest
from unittest.mock import patch

import donothing_baseline as baseline


class FakeEnv:
    def __init__(self, terminal):
        self.terminal = terminal
        self.actions = []

    def reset(self, **kwargs):
        return {}, {}

    def step(self, action):
        self.actions.append(action)
        return dict(terminal=self.terminal, result=1 if self.terminal else 0,
                    wave=10, tick=300 * len(self.actions)), 0, self.terminal, False, {}


class IdleRecordingTests(unittest.TestCase):
    def test_normal_win_keeps_exact_seed_and_wait_action(self):
        env = FakeEnv(True)
        with patch.object(baseline, '_get_env', return_value=env):
            result = baseline._one_episode(('/resources', 7, [0, 1], 1.0, 10, [], 2, 31415, 2))
        self.assertEqual(result['seed'], 31415)
        self.assertTrue(result['won'])
        self.assertTrue(result['terminated'])
        self.assertFalse(result['truncated'])
        self.assertEqual(env.actions, [baseline.WAIT_ACTION])

    def test_exhausted_wait_budget_is_retained_as_truncated(self):
        env = FakeEnv(False)
        with patch.object(baseline, '_get_env', return_value=env):
            result = baseline._one_episode(('/resources', 7, [0, 1], 1.0, None, [], 2, 2718, 2))
        self.assertEqual(result['seed'], 2718)
        self.assertEqual(result['decisions'], 2)
        self.assertFalse(result['won'])
        self.assertFalse(result['terminated'])
        self.assertTrue(result['truncated'])


if __name__ == '__main__':
    unittest.main()
