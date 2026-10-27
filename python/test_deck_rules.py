import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

from deck_rules import (DAY_MUSHROOMS, DEPENDENCIES, banned_for_level, generate_deck,
                        region_for_level, requirements)
from pvz_env import PvZEnv, TaskSpec
from pvz_tasks import profile_for_deck

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', '/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN'))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', ROOT / 'build/pvz-portable'))


class DeckRulesTest(unittest.TestCase):
    def test_generated_decks_follow_v5(self):
        unrestricted = False
        for level in range(1, 50):
            if level % 5 == 0:
                continue
            for length in range(6, 11):
                deck = generate_deck(level, length, level * 100 + length)
                cards = set(deck)
                self.assertEqual(len(cards), length)
                self.assertFalse(cards & banned_for_level(level))
                for choices in requirements(level).values():
                    self.assertTrue(cards & choices, (level, deck, choices))
                for card, choices in DEPENDENCIES.items():
                    if card in cards:
                        self.assertTrue(cards & choices)
                if region_for_level(level) in ('day', 'pool', 'roof'):
                    self.assertEqual(35 in cards, bool(cards & DAY_MUSHROOMS))
                unrestricted |= len(cards & {6, 26, 28, 29, 36}) > 1
        self.assertTrue(unrestricted)
        self.assertFalse({22, 26} & banned_for_level(42))

    def test_reproducible_and_invalid_inputs(self):
        self.assertEqual(generate_deck(7, 8, 129), generate_deck(7, 8, 129))
        for level in (0, 5, 10, 15, 25, 35, 45, 50, True):
            with self.assertRaises(ValueError):
                generate_deck(level, 8)
        for length in (5, 11, 7.5, True):
            with self.assertRaises(ValueError):
                generate_deck(7, length)


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / 'main.pak').is_file(), 'requires simulator and resources')
class DeckRulesEngineTest(unittest.TestCase):
    def test_small_reset_sample(self):
        with PvZEnv(RESOURCE, executable=BINARY) as env:
            upgrades_seen = set()
            for level, length in ((7, 7), (14, 8), (24, 9), (33, 10), (42, 10)):
                deck = generate_deck(level, length, 129 + level)
                profile = profile_for_deck(deck)
                self.assertEqual(profile.seed_slot_count, length)
                self.assertEqual(set(profile.owned_upgrade_plants), set(deck) & set(range(40, 48)))
                upgrades_seen.update(profile.owned_upgrade_plants)
                observation, _ = env.reset(deck=deck, task=TaskSpec(level=level, seed=30000, profile=profile))
                self.assertEqual(tuple(packet['type'] for packet in observation['packets']), deck)
            self.assertTrue(upgrades_seen)


if __name__ == '__main__':
    unittest.main()
