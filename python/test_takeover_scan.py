#!/usr/bin/env python3
"""task91：接手扫描运行器的纯决策逻辑（不跑引擎）。

覆盖 build_ladder 的阶梯规则与 AdaptiveScan 的取点顺序（4 等距 → 二分 → 补档）。
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from takeover_scan import AdaptiveScan, build_ladder, update_manifest, update_test_expectations


def grid12() -> list:
    return [1, 91, 182, 272, 363, 453, 544, 634, 725, 815, 906, 996]


def drive(scanner: AdaptiveScan, rate_fn) -> int:
    """用合成胜率驱动老师臂循环，返回实测点数。"""
    count = 0
    while True:
        decision = scanner.next_teacher_point()
        if decision is None:
            return count
        wins, games = rate_fn(decision)
        scanner.observe(decision, 'teacher', wins, games)
        count += 1


class BuildLadderTest(unittest.TestCase):
    def test_ladder_uses_full_line_as_top(self):
        ladder = build_ladder({100: (4, 8), 200: (6, 8), 300: (8, 8), 400: (5, 8), 500: (8, 8)})
        self.assertEqual(ladder['ladder'], [500, 400, 300, 200, 100])
        self.assertEqual(ladder['top'], 500)
        self.assertEqual(ladder['l50'], 100)
        self.assertTrue(ladder['top_is_full'])

    def test_passing_above_top_is_excluded(self):
        # 400 的 5/8 在最晚 8/8（300）之上：不如稳定点，不入阶梯；4/8 恰好过线。
        ladder = build_ladder({100: (4, 8), 200: (6, 8), 300: (8, 8), 400: (5, 8)})
        self.assertEqual(ladder['ladder'], [300, 200, 100])
        self.assertEqual(ladder['top'], 300)

    def test_no_full_line_takes_latest_passing(self):
        ladder = build_ladder({100: (6, 8), 200: (7, 8)})
        self.assertEqual(ladder['ladder'], [200, 100])
        self.assertEqual(ladder['top'], 200)
        self.assertFalse(ladder['top_is_full'])

    def test_all_below_half_returns_none(self):
        self.assertIsNone(build_ladder({100: (3, 8), 200: (3, 8)}))

    def test_failed_rungs_are_skipped(self):
        # 唯一 8/8 在 100；300 的 6/8 在顶档之上，不入阶梯（顶档取最晚稳定点）。
        ladder = build_ladder({100: (8, 8), 200: (3, 8), 300: (6, 8)})
        self.assertEqual(ladder['ladder'], [100])


class AdaptiveScanTest(unittest.TestCase):
    def test_equidistant_indices(self):
        self.assertEqual(AdaptiveScan(grid12()).equidistant, [0, 4, 7, 11])

    def test_monotonic_step_finds_l50_by_bisection(self):
        grid = grid12()
        scanner = AdaptiveScan(grid)
        tested = []

        def rate(decision):
            tested.append(decision)
            index = grid.index(decision)
            return (8, 8) if index >= 5 else (0, 8)

        count = drive(scanner, rate)
        state = scanner.finish()
        self.assertEqual(state['l50'], grid[5])
        self.assertEqual(state['top'], grid[11])
        self.assertEqual(state['ladder']['l50'], grid[5])
        # 等距 4 点 + 二分 2 点 + 补档 3 点（idx5..11 里未测的），远少于 12 点全跑。
        self.assertLess(count, 12)
        self.assertEqual(len(state['ladder']['ladder']), 7)
        self.assertEqual(state['student_points'], [grid[5], grid[11]])

    def test_all_fail_reports_no_ladder(self):
        scanner = AdaptiveScan(grid12())
        count = drive(scanner, lambda d: (0, 8))
        state = scanner.finish()
        self.assertEqual(count, 4)
        self.assertIsNone(state['ladder'])
        self.assertIsNone(state['l50'])
        self.assertEqual(state['student_points'], [])

    def test_all_pass_starts_at_first_rung(self):
        grid = grid12()
        scanner = AdaptiveScan(grid)
        count = drive(scanner, lambda d: (8, 8))
        state = scanner.finish()
        self.assertEqual(count, 12)     # 补档把整段阶梯测满
        self.assertEqual(state['l50'], grid[0])
        self.assertEqual(state['top'], grid[11])

    def test_non_monotonic_l48_shape(self):
        grid = grid12()
        scanner = AdaptiveScan(grid)

        def rate(decision):
            index = grid.index(decision)
            if index < 4:
                return (0, 8)
            if index < 8:
                return (6, 8)
            if index < 11:
                return (3, 8)   # L48 型：中段跌回
            return (8, 8)

        drive(scanner, rate)
        state = scanner.finish()
        self.assertEqual(state['l50'], grid[4])
        self.assertEqual(state['top'], grid[11])
        # 中段跌回的档（3/8）不入阶梯。
        ladder = state['ladder']['ladder']
        self.assertNotIn(grid[8], ladder)
        self.assertIn(grid[4], ladder)


class RegisterHelpersTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_update_manifest_writes_and_is_idempotent(self):
        path = self.tmp / 'manifest.json'
        path.write_text(json.dumps({'tasks': [{'task_id': 't1', 'seeds': [1]},
                                              {'task_id': 't2', 'seeds': [2]}]}), encoding='utf-8')
        spec = dict(archive='a', decisions=[300, 200], index=0,
                    future_seed_start=86000000, future_seed_count=256)
        self.assertTrue(update_manifest(path, 't1', spec))
        self.assertFalse(update_manifest(path, 't1', spec))
        document = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(document['tasks'][0]['demo_start'], spec)
        self.assertNotIn('demo_start', document['tasks'][1])

    def test_update_manifest_rejects_unknown_task(self):
        path = self.tmp / 'manifest.json'
        path.write_text(json.dumps({'tasks': []}), encoding='utf-8')
        with self.assertRaises(KeyError):
            update_manifest(path, 'nope', dict(decisions=[1]))

    def test_update_test_expectations_inserts_missing_ids(self):
        path = self.tmp / 'test_x.py'
        path.write_text("self.assertEqual(set(cells), {\n                                      'a',\n                                      'b',\n                                      })\n", encoding='utf-8')
        self.assertTrue(update_test_expectations(path, ['c']))
        self.assertIn("'c',", path.read_text(encoding='utf-8'))
        self.assertFalse(update_test_expectations(path, ['c']))


if __name__ == '__main__':
    unittest.main()
