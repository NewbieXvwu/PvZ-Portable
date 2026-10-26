"""task99: 上线前核对脚本的单测（不读真实 state / 账本）。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

from check_start_seed_state import (check_config, check_inventory, check_ledger, check_state)
from pvz_promotion import START_SEED_BLOCK, START_SEED_POOL


def state_cell(seeds, blocks=None, archive='/a', task_id='demo_x'):
    return dict(archive=archive, task_id=task_id, seeds=list(seeds),
                seed_blocks=blocks if blocks is not None else [])


class StartSeedPreflightTest(unittest.TestCase):
    def test_clean_state_has_no_problems(self):
        state = dict(cells={'demo_x': state_cell(range(256))}, offsets={})
        problems, notes = check_state(state, [])
        self.assertEqual(problems, [])
        self.assertEqual(notes, [])
        self.assertEqual(check_inventory({'occupied': [[0, 256]]}), [])
        problems, notes = check_config({'tasks': [dict(
            task_id='demo_x', demo_start=dict(future_seed_start=44000000, future_seed_count=256))]})
        self.assertEqual((problems, notes), ([], []))

    def test_seeds_must_equal_block_concatenation(self):
        cell = state_cell(range(256), blocks=[[0, 128]])
        problems, _ = check_state(dict(cells={'demo_x': cell}, offsets={}), [])
        self.assertTrue(any('seed_blocks concatenation' in line for line in problems))

    def test_renewed_segment_must_be_in_the_ledger(self):
        cell = state_cell(range(512), blocks=[[0, 256], [START_SEED_POOL[0], START_SEED_POOL[0] + 256]])
        problems, _ = check_state(dict(cells={'demo_x': cell}, offsets={}), [])
        self.assertTrue(any('is not in the ledger' in line for line in problems))

    def test_offset_past_the_table_fails(self):
        cell = state_cell(range(256))
        problems, _ = check_state(dict(cells={'demo_x': cell}, offsets={'demo_x': 300}), [])
        self.assertTrue(any('exceeds' in line for line in problems))

    def test_two_cells_sharing_unused_seeds_fail(self):
        cells = {'a': state_cell(range(256), task_id='a'), 'b': state_cell(range(256), task_id='b')}
        problems, _ = check_state(dict(cells=cells, offsets={}), [])
        self.assertTrue(any('share 256 unused seeds' in line for line in problems))
        # 各自用完自己的前半段也仍共享后半段
        problems, _ = check_state(dict(cells=cells, offsets={'a': 128, 'b': 128}), [])
        self.assertTrue(any('share 128 unused seeds' in line for line in problems))

    def test_inventory_pool_overlap_fails(self):
        problems = check_inventory({'occupied': [[START_SEED_POOL[0] - 10, START_SEED_POOL[0] + 10]]})
        self.assertTrue(any('overlaps' in line for line in problems))
        # 池自己出现在占用表里不算重叠
        self.assertEqual(check_inventory({'occupied': [list(START_SEED_POOL)]}), [])

    def test_config_block_inside_the_pool_fails(self):
        problems, _ = check_config({'tasks': [dict(
            task_id='demo_x', demo_start=dict(future_seed_start=START_SEED_POOL[0],
                                              future_seed_count=256))]})
        self.assertTrue(any('inside the renewal pool' in line for line in problems))

    def test_shared_config_block_is_only_a_note(self):
        start = dict(future_seed_start=44000000, future_seed_count=256)
        problems, notes = check_config({'tasks': [dict(task_id='a', demo_start=dict(start)),
                                                  dict(task_id='b', demo_start=dict(start))]})
        self.assertEqual(problems, [])
        self.assertTrue(any('shared by 2 tasks' in line for line in notes))

    def test_ledger_shape(self):
        good = dict(pool=list(START_SEED_POOL), pools=[list(START_SEED_POOL)],
                    allocated=[[START_SEED_POOL[0], START_SEED_POOL[0] + START_SEED_BLOCK]])
        self.assertEqual(check_ledger(good, True), [])
        bad_size = dict(good, allocated=[[START_SEED_POOL[0], START_SEED_POOL[0] + 128]])
        self.assertTrue(any('expected 256' in line for line in check_ledger(bad_size, True)))
        outside = dict(good, allocated=[[1, 257]])
        self.assertTrue(any('outside the renewal pool' in line for line in check_ledger(outside, True)))
        self.assertEqual(check_ledger(None, False), [])       # 账本还不存在是正常的
        self.assertTrue(check_ledger(None, True))


if __name__ == '__main__':
    unittest.main()
