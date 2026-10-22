"""task29 第 3 步：压力面板的纯函数单测（不跑引擎、不读真实清单）。"""
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

from unified_panel import PRESSURE_LEVELS, PRESSURE_MULTIPLIERS, generate_pressure, pressure_ceiling


def source_tasks():
    day, night, pool = [], [], []
    for zone, level, source_id in PRESSURE_LEVELS:
        task = dict(task_id=source_id, terrain=zone, level=level, deck=[0, 1, 2, 3, 4, 5, 7],
                    zombie_count_multiplier=1.0, wave_cap=None, sun_start=50, preplanted=[],
                    playthrough=2, seeds=list(range(30000, 30128)))
        if source_id == 'day_l10_orig_eval':
            task.update(deck=[], conveyor_start='original')
        if source_id.startswith('day'):
            day.append(task)
        elif source_id.startswith('night'):
            night.append(task)
        else:
            pool.append(task)
    return dict(tasks=day), dict(tasks=night), dict(tasks=pool)


class PressurePanelTest(unittest.TestCase):
    def test_generate_copies_definitions_and_sets_multiplier(self):
        day, night, pool = source_tasks()
        day_orig = dict(tasks=[copy.deepcopy(t) for t in day['tasks'] if t['task_id'] == 'day_l10_orig_eval'])
        document = generate_pressure(day, day_orig, night, pool, seeds=64)
        self.assertEqual(len(document['tasks']), len(PRESSURE_LEVELS) * len(PRESSURE_MULTIPLIERS))
        ids = [t['task_id'] for t in document['tasks']]
        self.assertEqual(len(ids), len(set(ids)))
        for task in document['tasks']:
            self.assertIn(task['zombie_count_multiplier'], PRESSURE_MULTIPLIERS)
            self.assertEqual(len(task['seeds']), 64)
            self.assertEqual(task['seeds'][0], 30000)
            self.assertEqual(task['split'], 'pressure')
            self.assertIn(task['level'], {level for _zone, level, _id in PRESSURE_LEVELS})
        conveyor = [t for t in document['tasks'] if t['task_id'].startswith('press_day_l10')]
        self.assertTrue(all(t.get('conveyor_start') == 'original' and t['deck'] == [] for t in conveyor))

    def test_ceiling_interpolates_between_points(self):
        result = pressure_ceiling({1.0: (90, 100), 1.5: (60, 100), 2.0: (30, 100)})
        self.assertFalse(result['censored'])
        self.assertFalse(result['below_floor'])
        self.assertAlmostEqual(result['ceiling'], 1.5 + 0.1 / 0.3 * 0.5)

    def test_ceiling_crosses_first_interval(self):
        result = pressure_ceiling({1.0: (70, 100), 1.5: (40, 100), 2.0: (10, 100)})
        self.assertAlmostEqual(result['ceiling'], 1.0 + 0.2 / 0.3 * 0.5)

    def test_ceiling_censored_when_top_still_wins(self):
        result = pressure_ceiling({1.0: (100, 100), 1.5: (90, 100), 2.0: (80, 100)})
        self.assertTrue(result['censored'])
        self.assertEqual(result['ceiling'], 2.0)

    def test_ceiling_none_below_floor(self):
        result = pressure_ceiling({1.0: (40, 100), 1.5: (20, 100), 2.0: (5, 100)})
        self.assertIsNone(result['ceiling'])
        self.assertTrue(result['below_floor'])

    def test_ceiling_skips_empty_points(self):
        result = pressure_ceiling({1.0: (0, 0), 1.5: (60, 100), 2.0: (30, 100)})
        self.assertAlmostEqual(result['ceiling'], 1.5 + 0.1 / 0.3 * 0.5)


if __name__ == '__main__':
    unittest.main()
