import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))

_spec = importlib.util.spec_from_file_location('task104_verdict', ROOT / 'scripts/task104_verdict.py')
verdict = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verdict)


def panel(rows):
    """rows: {(arm, seed): [(task_id, terrain, won, games), ...]}"""
    models = {}
    for (arm, seed), tasks in rows.items():
        models[f'{arm}_s{seed}'] = {
            'won': sum(t[2] for t in tasks), 'games': sum(t[3] for t in tasks),
            'tasks': [{'task_id': t[0], 'terrain': t[1], 'won': t[2], 'games': t[3],
                       'seeds': list(range(t[3])),
                       'outcomes': {str(index): index < t[2] for index in range(t[3])}} for t in tasks],
            'terrains': {}}
        for task in models[f'{arm}_s{seed}']['tasks']:
            bucket = models[f'{arm}_s{seed}']['terrains'].setdefault(task['terrain'], {'won': 0, 'games': 0})
            bucket['won'] += task['won']
            bucket['games'] += task['games']
    return {'models': models}


def roof_rows(a_won, b_won, levels=('l42', 'l43', 'l44'), games=32):
    rows = {}
    for seed in ('1', '2'):
        for arm, base in (('a', a_won), ('b', b_won)):
            rows[(arm, seed)] = [(level, 'roof', base, games) for level in levels]
    return rows


def regression_rows(a_won, b_won):
    rows = {}
    for seed in ('1', '2'):
        for arm, base in (('a', a_won), ('b', b_won)):
            rows[(arm, seed)] = [('day', 'day', base, 8), ('night', 'night', base, 8)]
    return rows


class VerdictTest(unittest.TestCase):
    def test_clear_gain_passes(self):
        result = verdict.compare(panel(roof_rows(2, 10)), panel(regression_rows(4, 4)))
        self.assertEqual(result['failures'], [])
        self.assertEqual(result['verdict'], 'demonstration rows help')
        self.assertAlmostEqual(result['roof'][0]['gain_pp'], 25.0)

    def test_one_seed_flat_fails(self):
        rows = roof_rows(2, 10)
        for arm in ('a', 'b'):
            rows[(arm, '2')] = [('l42', 'roof', 5, 32), ('l43', 'roof', 5, 32), ('l44', 'roof', 5, 32)]
        result = verdict.compare(panel(rows), panel(regression_rows(4, 4)))
        self.assertTrue(any('训练种子 2' in failure for failure in result['failures']))
        self.assertEqual(result['verdict'], 'not proven; do not deploy')

    def test_small_gain_fails_threshold(self):
        result = verdict.compare(panel(roof_rows(2, 4)), panel(regression_rows(4, 4)))
        self.assertTrue(any('低于 10' in failure for failure in result['failures']))

    def test_level_harm_fails(self):
        rows = roof_rows(2, 10)
        for seed in ('1', '2'):
            for arm, won in (('a', 10), ('b', 0)):
                rows[(arm, seed)] = [('l42', 'roof', won, 32), ('l43', 'roof', 10, 32), ('l44', 'roof', 10, 32)]
        result = verdict.compare(panel(rows), panel(regression_rows(4, 4)))
        self.assertTrue(any('l42' in failure for failure in result['failures']))

    def test_regression_drop_fails(self):
        result = verdict.compare(panel(roof_rows(2, 10)), panel(regression_rows(8, 4)))
        self.assertTrue(any('非屋顶回归' in failure for failure in result['failures']))

    def test_missing_model_raises(self):
        with self.assertRaises(KeyError):
            verdict.compare({'models': {}}, panel(regression_rows(4, 4)))

    def test_paired_gain_and_harm_are_kept_separate(self):
        roof = panel(roof_rows(2, 2))
        roof['models']['b_s1']['tasks'][0]['outcomes'].update({'0': False, '2': True})
        result = verdict.compare(roof, panel(regression_rows(4, 4)))
        row = next(row for row in result['roof_paired']
                   if row['training_seed'] == 1 and row['task_id'] == 'l42')
        self.assertEqual((row['pairs'], row['gain'], row['harm'], row['delta_wins']), (32, 1, 1, 0))
        self.assertEqual(result['roof_training_seed_ranges_pp']['gain'], [0, 0])
        self.assertEqual(len(result['regression']['paired']), 4)

    def test_unpaired_outcomes_raise(self):
        roof = panel(roof_rows(2, 10))
        del roof['models']['b_s1']['tasks'][0]['outcomes']['0']
        with self.assertRaises(ValueError):
            verdict.compare(roof, panel(regression_rows(4, 4)))


if __name__ == '__main__':
    unittest.main()
