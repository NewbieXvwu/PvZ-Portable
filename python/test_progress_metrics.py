"""Check task/seed pairing when passive mower wins would hide policy harm."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from pvz_common import canonical_digest
from pvz_progress_metrics import idle_baseline, paired_idle_summary, progress_summary


def task(name, cap, seeds, aided=False):
    return dict(task_id=name, wave_cap=cap, seeds=seeds, preplanted=[[0, 0, 0]] if aided else [],
                sun_start=50, zombie_count_multiplier=1)


def row(seed, won, truncated=False):
    return dict(seed=seed, won=won, truncated=truncated, terminal_wave=8,
                terminal_tick=24000, peak_offense=0, economy_curve=[])


class ProgressMetricTests(unittest.TestCase):
    def test_seed_order_cannot_hide_passive_wins_or_harm(self):
        tasks = [task("short", 3, [1, 2, 3]), task("long", 10, [1, 2, 3])]
        idle = {"short": [row(1, True), row(2, True), row(3, True)],
                "long": [row(1, False), row(2, True), row(3, False)]}
        policy = {"short": [row(3, True), row(1, False), row(2, True)],
                  "long": [row(3, True), row(1, False), row(2, True)]}
        result = paired_idle_summary(tasks, policy, idle)
        self.assertEqual(result['per_task']['short']['degraded_pairs'], 1)
        self.assertEqual(result['per_task']['long']['improved_pairs'], 1)
        self.assertAlmostEqual(result['by_group']['ordinary_long']['net_win_gain'], 1/3)
        self.assertAlmostEqual(result['by_group']['short_regression']['net_win_gain'], -1/3)

    def test_reused_seed_in_different_tasks_remains_two_independent_pairs(self):
        tasks = [task("a", 10, [7]), task("b", 15, [7])]
        policy = {"a": [row(7, True)], "b": [row(7, False)]}
        idle = {"a": [row(7, False)], "b": [row(7, True)]}
        result = paired_idle_summary(tasks, policy, idle)['by_group']['ordinary_long']
        self.assertEqual((result['cases'], result['improved_pairs'], result['degraded_pairs']), (2, 1, 1))
        self.assertEqual(result['net_win_gain'], 0)

    def test_missing_duplicate_or_replaced_seed_fails_without_dropping_rows(self):
        tasks = [task("a", 10, [1, 2])]
        idle = {"a": [row(1, True), row(2, False)]}
        for rows in ([row(1, True)], [row(1, True), row(1, False)], [row(1, True), row(3, False)]):
            with self.assertRaises(ValueError):
                paired_idle_summary(tasks, {"a": rows}, idle)

    def test_aid_full_and_short_views_stay_separate_and_truncation_is_retained(self):
        tasks = [task("a", None, [1], True), task("b", None, [2]), task("c", 5, [3])]
        records = {"a": [row(1, True)], "b": [row(2, False, True)], "c": [row(3, True)]}
        result = progress_summary(tasks, records)
        self.assertEqual(set(result), {'aided_full', 'ordinary_full', 'short_regression'})
        self.assertEqual(result['ordinary_full']['truncated'], 1)
        self.assertEqual(result['ordinary_full']['pass_rate'], 0)

    def test_control_reuse_requires_the_same_task_seed_and_experiment_identity(self):
        tasks = [task('a', 10, [1])]
        identity = canonical_digest(dict(schema_version=1, experiment_identity='stage', tasks=tasks,
                                         max_actions=4000, fixed_wait_ticks=300))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'evaluations').mkdir()
            path = output/'evaluations/idle_control.json.gz'
            with gzip.open(path, 'wt') as stream:
                json.dump(dict(identity=identity, seed_results={'a': [row(1, False)]}), stream)
            before = path.read_bytes()
            cached = idle_baseline(tasks, Path('/unused_resources'), output, 'stage', 4000)
            self.assertEqual(cached['a'][0]['seed'], 1)
            with self.assertRaises(ValueError):
                idle_baseline(tasks, Path('/unused_resources'), output, 'otherstage', 4000)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
