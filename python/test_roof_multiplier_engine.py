"""Actual reset range and fractional wave spawning, across the Python/C++ boundary."""
import math
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
from pvz_env import PvZEnv, TaskSpec
from pvz_tasks import deck_for_level, profile_for_deck
from smoke_zombie_multiplier import count_waves

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', str(Path.home() / '.cache/pvz-research-resources')))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', str(ROOT / 'build/pvz-portable')))


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / 'main.pak').is_file(), 'needs engine and resources')
class RoofMultiplierEngineTest(unittest.TestCase):
    def test_fractional_reset_and_wave_counts_match_the_task(self):
        deck = deck_for_level(43, family='roof_r7')
        with PvZEnv(RESOURCE, executable=BINARY) as env:
            for multiplier in (0.1, 0.5):
                obs, _ = env.reset(deck=deck, task=TaskSpec(level=43, seed=45000000,
                    profile=profile_for_deck(deck), zombie_count_multiplier=multiplier))
                self.assertEqual(obs['zombie_count_multiplier'], multiplier)
            # Test the engine's guard itself; Python validation would reject before sending.
            response = env._command(f'RESET_V2 43 45000000 2 {len(deck)} 0 0 0 0 0 - - '
                                    f'{",".join(map(str, deck))} 0.05 0 -')
            self.assertFalse(response['ok'])
            base, base_dirty = count_waves(env, 43, 45000000, deck, 1.0, 2)
            half, half_dirty = count_waves(env, 43, 45000000, deck, 0.5, 2)
            self.assertEqual(set(base), {1, 2})
            self.assertEqual(set(half), set(base))
            self.assertFalse(base_dirty | half_dirty)
            for wave in base:
                self.assertEqual(half[wave], math.floor(base[wave] * 0.5 + 0.5))


if __name__ == '__main__':
    unittest.main()
