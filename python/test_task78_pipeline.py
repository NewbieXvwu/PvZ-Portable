"""Pure-function tests for the task78 export/training/evaluation plumbing."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
import task78_ab as ab
import task78_export as export


class StopSetTest(unittest.TestCase):
    def test_region_floor_adds_hash_smallest_archive(self):
        archives = [dict(group='roof_l41', world=dict(seed=s, future_seed=None)) for s in range(10)]
        stops, added = export.stop_set(archives, old=False)
        roof = [i for i, a in enumerate(archives)]
        self.assertGreaterEqual(sum(stops[i] for i in roof), 2)
        self.assertTrue(all(isinstance(v, bool) for v in stops.values()))
        # Every archive the floor added must be flagged as a stopping archive.
        for index in added.get('roof', []):
            self.assertTrue(stops[index])

    def test_old_manifest_uses_the_v5_seed_split(self):
        archives = [dict(group='day_d7', world=dict(seed=s, future_seed=None)) for s in (1, 2, 3)]
        stops, added = export.stop_set(archives, old=True)
        self.assertEqual(added, {})
        self.assertEqual(stops, {i: export.v5_stop(a['world']['seed']) for i, a in enumerate(archives)})


class StatisticsTest(unittest.TestCase):
    def test_clopper_bounds_hit_the_extremes(self):
        self.assertEqual(ab.clopper_lower(0, 32), 0.0)
        self.assertEqual(ab.clopper_upper(32, 32), 1.0)
        self.assertGreater(ab.clopper_lower(30, 32), ab.clopper_lower(20, 32))
        self.assertLess(ab.clopper_upper(2, 32), ab.clopper_upper(12, 32))

    def test_mcnemar_is_symmetric_and_one_sided_evidence_is_small(self):
        self.assertAlmostEqual(ab.mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(ab.mcnemar(3, 4), ab.mcnemar(4, 3))
        self.assertLess(ab.mcnemar(0, 8), 0.05)
        self.assertGreater(ab.mcnemar(5, 5), 0.5)


class CellsTest(unittest.TestCase):
    def test_cells_use_disjoint_reserved_seeds(self):
        cells = ab.cells()
        seen = set()
        for cell in cells:
            self.assertEqual(len(cell['seeds']), cell['pairs'])
            self.assertFalse(seen & set(cell['seeds']))
            seen |= set(cell['seeds'])
        self.assertLessEqual(max(seen), ab.SEED_START + 1999)
        self.assertGreaterEqual(min(seen), ab.SEED_START)
        self.assertEqual(sum(c['pairs'] for c in cells), 416)


if __name__ == '__main__':
    unittest.main()
