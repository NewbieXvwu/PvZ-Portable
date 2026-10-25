"""Unit tests for the descriptive reversal / seed-cost breakdown."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'replace_action_reversal', ROOT / 'scripts/replace_action_reversal.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def event(tick, row, col, removed, added, sun=0):
    return dict(tick=tick, row=row, col=col, removed=[removed], added=[added],
                planted_type=added, sun_before=sun, sun_after=sun, sun_spent=sun)


class ReversalSummaryTest(unittest.TestCase):
    def setUp(self):
        self.costs, self.ids, self.source = MODULE.load_seed_tables(ROOT)

    def test_engine_seed_table_is_read_from_source(self):
        self.assertEqual(self.costs['PEASHOOTER']['cost'], 100)
        self.assertEqual(self.costs['PUFFSHROOM']['cost'], 0)
        self.assertEqual(self.ids[0], 'PEASHOOTER')

    def test_reversal_counts_only_the_same_cell_and_the_exact_previous_type(self):
        rows = [
            dict(replace_events=[event(10, 1, 2, 9, 8), event(20, 1, 2, 8, 9)]),
            dict(replace_events=[event(10, 1, 2, 9, 8), event(20, 1, 3, 8, 9)]),
            dict(replace_events=[event(10, 1, 2, 9, 8), event(20, 1, 2, 8, 3)]),
        ]
        summary = MODULE.summarize(rows, self.costs, self.ids)
        self.assertEqual(summary['events'], 6)
        self.assertEqual(summary['same_cell_reversals'], 1)
        self.assertEqual(summary['games_with_any_reversal'], 1)

    def test_cost_buckets_and_free_share_use_the_engine_table(self):
        rows = [dict(replace_events=[
            event(10, 0, 0, 1, 0, sun=0),   # sunflower 50 -> peashooter 100
            event(20, 0, 1, 0, 1, sun=50),  # peashooter 100 -> sunflower 50
            event(30, 0, 2, 1, 3, sun=50),  # sunflower 50 -> wall-nut 50
        ])]
        summary = MODULE.summarize(rows, self.costs, self.ids)
        self.assertEqual(summary['seed_cost_delta'],
                         dict(upgrade=1, equal=1, downgrade=1, unknown=0))
        self.assertEqual(summary['free_events'], 1)
        self.assertAlmostEqual(summary['free_share'], 1 / 3)


if __name__ == '__main__':
    unittest.main()
