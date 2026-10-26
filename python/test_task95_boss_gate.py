from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from task95_boss_l50_gate import compare_game


def fixture():
    record = dict(seed=85000000, won=False, terminal=True, final_tick=100, final_wave=0)
    frame = dict(tick=0, action=None)
    boss = dict(active=True, stage=0, head=False, action='idle', target_row=-1,
                target_col=-1, projectile=None)
    return record, [frame], [{**frame, 'boss': boss}]


class BossGateTest(unittest.TestCase):
    def test_old_engine_need_not_expose_new_boss_fields(self):
        record, old, new = fixture()
        result = compare_game(record, record, old, new)
        self.assertTrue(result['new_boss_fields_present'])
        self.assertFalse(result['old_boss_block_present'])
        self.assertTrue(result['action_stream_match'])
        self.assertFalse(result['logical_differences'])

    def test_missing_candidate_fields_fail_the_schema_check(self):
        record, old, _ = fixture()
        self.assertFalse(compare_game(record, record, old, old)['new_boss_fields_present'])

    def test_terminal_logic_difference_is_reported(self):
        record, old, new = fixture()
        result = compare_game(record, {**record, 'final_tick': 101}, old, new)
        self.assertIn('final_tick', result['logical_differences'])

    def test_action_prefix_difference_fails_without_an_index_error(self):
        record, old, new = fixture()
        result = compare_game(record, record, old, new + [dict(tick=100, action={'type': 'wait'})])
        self.assertFalse(result['action_stream_match'])


if __name__ == '__main__':
    unittest.main()
