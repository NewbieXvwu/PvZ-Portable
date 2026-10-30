"""task28：整组选卡检查（SeedChooserScreen::OnStartButton）的引擎查询。"""
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_env import TaskSpec, profile_for_deck
from pvz_event_env import EventWaitEnv

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', str(Path.home() / '.cache/pvz-research-resources')))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', str(ROOT / 'build/pvz-portable')))


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / 'main.pak').is_file(), 'needs local CPU build and resources')
class DeckWarningsTest(unittest.TestCase):
    def setUp(self):
        self.env = EventWaitEnv(RESOURCE, executable=BINARY, headless=True)

    def tearDown(self):
        self.env.close()

    def warnings(self, level: int, deck) -> list[str]:
        self.env.reset(deck=tuple(deck), task=TaskSpec(level=level, seed=1,
                                                      profile=profile_for_deck(tuple(deck)), playthrough=2))
        return self.env.deck_warnings(tuple(deck))['warnings']

    def test_day_deck_without_sun_producer_warns(self):
        self.assertEqual(self.warnings(7, (0, 2, 3, 4, 5, 7)), ['sun'])
        self.assertEqual(self.warnings(7, (1, 0, 2, 3, 4, 5)), [])

    def test_pool_deck_without_water_plant_warns(self):
        self.assertEqual(self.warnings(21, (0, 1, 2, 3, 4, 5)), ['pool'])
        self.assertEqual(self.warnings(21, (0, 1, 2, 3, 4, 16)), [])

    def test_roof_deck_without_flower_pot_warns(self):
        self.assertEqual(self.warnings(41, (0, 1, 2, 3, 4, 5)), ['roof'])
        self.assertEqual(self.warnings(41, (0, 1, 2, 3, 4, 33)), [])

    def test_night_deck_without_sunshroom_warns(self):
        self.assertEqual(self.warnings(12, (0, 2, 3, 4, 5, 7, 8, 11, 10, 13)), ['sun'])

    def test_normal_deck_has_no_warnings(self):
        self.assertEqual(self.warnings(12, (0, 1, 2, 3, 4, 5, 7, 8, 9, 11)), [])
        self.assertEqual(self.warnings(7, (0, 1, 2, 3, 4, 5, 7)), [])


if __name__ == '__main__':
    unittest.main()
