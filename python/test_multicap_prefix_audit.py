"""Distinguish intentional task horizons from real prefix/loss mismatches."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import research_multicap_prefix_audit as prefix


class MulticapPrefixAuditTests(unittest.TestCase):
    def observation(self):
        return dict(tick=300, wave=1, terminal=False, result=0, plants=[], projectiles=[],
                    sun=50, coins=[], grid_items=[], defenses=[], packets=[],
                    zombies=[{'id': 1, 'on_board': False}, {'id': 2, 'on_board': True}])

    def test_street_preview_excluded_but_battle_entity_preserved(self):
        full, capped = self.observation(), self.observation()
        capped['zombies'][0]['id'] = 9
        self.assertIsNone(prefix.compare(full, capped, 3))
        capped['zombies'][1]['id'] = 9
        with self.assertRaises(ValueError):
            prefix.compare(full, capped, 3)

    def test_public_battle_or_tick_mismatch_rejected(self):
        for field, value in [('sun', 75), ('tick', 301), ('packets', [{'cooldown': 1}])]:
            full, capped = self.observation(), self.observation()
            capped[field] = value
            with self.assertRaises(ValueError):
                prefix.compare(full, capped, 3)

    def test_intentional_cap_win_and_next_full_wave_are_boundaries(self):
        full, capped = self.observation(), self.observation()
        capped.update(terminal=True, result=1, sun=75)
        self.assertEqual(prefix.compare(full, capped, 1), 'cap_win')
        full['wave'] = 2
        self.assertEqual(prefix.compare(full, capped, 1), 'next_full_wave')

    def test_normal_loss_must_agree_and_remains_a_valid_failed_episode(self):
        full, capped = self.observation(), self.observation()
        capped.update(terminal=True, result=2)
        with self.assertRaises(ValueError):
            prefix.compare(full, capped, 3)
        full.update(terminal=True, result=2)
        self.assertEqual(prefix.compare(full, capped, 3), 'normal_terminal')
        full['sun'] = 75
        with self.assertRaises(ValueError):
            prefix.compare(full, capped, 3)


if __name__ == '__main__':
    unittest.main()
