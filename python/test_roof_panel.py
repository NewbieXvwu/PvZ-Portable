"""Roof registry contract: terrain, independent seeds, frozen legacy tasks, special level scope."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
from make_roof_panel import EVAL_COUNT, EVAL_START, ROOF, TRAIN_LEVELS
from pvz_tasks import deck_for_level, deck_for_seed, profile_for_deck, terrain_covered
from zone_panel import build_document


class RoofPanelTest(unittest.TestCase):
    def setUp(self):
        args = SimpleNamespace(deck='', suffix='', combined=True, skip=[], include=[],
                               seed_start=EVAL_START, seeds=EVAL_COUNT, train_seeds=8192)
        self.document, _, _ = build_document(ROOF, args)

    def test_original_and_selected_families_have_disjoint_training_seeds(self):
        tasks = self.document['tasks']
        self.assertEqual(len(tasks), 26)
        self.assertEqual({t['level'] for t in tasks if '_train' in t['task_id']}, set(TRAIN_LEVELS))
        training = set()
        for task in tasks:
            self.assertNotEqual(task['level'], 50)
            deck = deck_for_seed(task, task['seeds'][0])
            self.assertTrue(terrain_covered(task['level'], deck))
            self.assertEqual(profile_for_deck(deck).seed_slot_count, len(deck))
            if '_eval' in task['task_id']:
                self.assertEqual(task['seeds'], list(range(EVAL_START, EVAL_START + EVAL_COUNT)))
            else:
                self.assertFalse(training.intersection(task['seeds']))
                training.update(task['seeds'])
        self.assertEqual(len(training), 8 * 8192)

    def test_new_deck_does_not_change_legacy_or_archived_decks(self):
        for level in range(41, 50):
            self.assertEqual(deck_for_level(level), (0, 1, 2, 3, 4, 5, 33))
            self.assertEqual(deck_for_seed(dict(level=level, deck=[0, 1, 2, 3, 4, 5, 33]), 30000),
                             (0, 1, 2, 3, 4, 5, 33))
            self.assertNotEqual(deck_for_level(level, family='roof_r7'), deck_for_level(level))
        for level in (1, 21, 40, 50):
            with self.assertRaises(ValueError):
                deck_for_level(level, family='roof_r7')


if __name__ == '__main__':
    unittest.main()
