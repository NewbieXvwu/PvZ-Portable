"""记分板差异比较的行为测试。"""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from adventure_scoreboard import arm_stats, compare_scoreboards, markdown


def board(wins, games=32, arm='student'):
    return dict(levels=[dict(level=7, task_id='task101_l7_day_d7', deck=[0, 1, 2, 3, 4, 5, 7],
                            **{arm: dict(wins=wins, games=games)})])


class ScoreboardTest(unittest.TestCase):
    def test_increase_and_decrease_have_separated_intervals(self):
        for before, after, direction in ((0, 32, '上升'), (32, 0, '下降')):
            with self.subTest(direction=direction):
                change = compare_scoreboards(board(after), board(before))[0]
                self.assertEqual(change['delta_rate'], (after - before) / 32)
                self.assertTrue(change['wilson_separated'])
                self.assertEqual(change['direction'], direction)

    def test_different_sample_sizes_use_counts_not_rounded_rates(self):
        before, after = board(8, 16), board(20, 32)
        before['levels'][0]['student']['rate'] = 0
        change = compare_scoreboards(after, before)[0]
        self.assertEqual(change['delta_rate'], .125)
        self.assertFalse(change['wilson_separated'])

    def test_touching_intervals_are_not_separated(self):
        with patch('adventure_scoreboard.wilson', side_effect=[[.1, .5], [.5, .9]]):
            self.assertFalse(compare_scoreboards(board(24), board(8))[0]['wilson_separated'])

    def test_missing_arms_and_skipped_levels_are_not_reused(self):
        before = board(8)
        before['levels'][0]['teacher'] = dict(wins=32, games=32)
        before['levels'].append(dict(level=5, student=None, teacher=None))
        current = board(16)
        current['levels'].append(dict(level=5, student=None, teacher=None))
        self.assertEqual(len(compare_scoreboards(current, before)), 1)
        self.assertEqual(compare_scoreboards(board(32, arm='teacher'), before)[0]['arm'], 'teacher')

    def test_different_task_contract_is_not_a_rate_change(self):
        before = board(16)
        for key, value in (('deck', [0, 1, 2, 3, 4, 5]), ('task_id', 'other'),
                           ('task', dict(fog_observation='human'))):
            with self.subTest(key=key):
                current = copy.deepcopy(before)
                before_with_contract = copy.deepcopy(before)
                if key == 'task':
                    before_with_contract['levels'][0][key] = value
                    current['levels'][0][key] = dict(fog_observation='human_v2')
                else:
                    current['levels'][0][key] = value
                change = compare_scoreboards(current, before_with_contract)[0]
                self.assertFalse(change['comparable'])
                self.assertIsNone(change['delta_rate'])
                self.assertIsNone(change['wilson_separated'])

    def test_stats_keep_task101_loss_summary(self):
        stats = arm_stats([
            dict(won=False, final_wave=12, final_zombies=[dict(type=0), dict(type=2)]),
            dict(won=False, final_wave=20, final_zombies=[dict(type=0)]),
            dict(won=True, final_wave=30, final_zombies=[])])
        self.assertEqual((stats['wins'], stats['games']), (1, 3))
        self.assertEqual(stats['loss_median_final_wave'], 16)
        self.assertEqual(stats['loss_breach_zombies_top3'][0]['count'], 2)

    def test_student_only_report_does_not_claim_teacher_measurement(self):
        stats = arm_stats([dict(won=False, final_wave=20, final_zombies=[])])
        report = dict(seeds_per_arm=1, arms=['student'], levels=[dict(level=7, deck=[0],
            family='d7', student=stats, teacher=None, old_status='在练', category='学生<80%，老师未测')])
        self.assertIn('0/1 (0.0%) | — |', markdown(report))


if __name__ == '__main__':
    unittest.main()
