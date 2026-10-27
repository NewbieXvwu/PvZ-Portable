"""task137：v5 随机卡组族（登记表 / 生成器 / 面板）的静态检查，不跑引擎。

覆盖：v5 规则可复现、训练与评估卡组互不重叠、面板格子数与倍率阶梯、
种子段唯一且不与现有清单重叠、登记表里的卡组能由 draw_seed 重放。
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

import make_random_deck_cells as cells
from deck_rules import RULE_VERSION, deck_violations, generate_deck, region_for_level


class DeckRulesTest(unittest.TestCase):
    def test_regions(self):
        self.assertEqual(region_for_level(7), 'day')
        self.assertEqual(region_for_level(14), 'night')
        self.assertEqual(region_for_level(24), 'pool')
        self.assertEqual(region_for_level(37), 'fog')
        self.assertEqual(region_for_level(41), 'roof')
        for bad in (0, 50, 5, 10, 35, 45, 7.0, True):
            with self.assertRaises(ValueError):
                region_for_level(bad)

    def test_generated_decks_satisfy_v5(self):
        for level in cells.LEVELS:
            for length in (6, 8, 10):
                for seed in (1, 2, 3):
                    deck = generate_deck(level, length, seed)
                    self.assertEqual(deck_violations(level, deck), [], (level, length, seed, deck))
                    self.assertEqual(deck, generate_deck(level, length, seed))

    def test_violations_are_detected(self):
        # 白天带蘑菇不带咖啡豆；白天无蘑菇带咖啡豆；泳池缺睡莲；升级缺底座；全局禁卡。
        self.assertTrue(deck_violations(7, (0, 1, 2, 3, 4, 8)))
        self.assertTrue(deck_violations(7, (0, 1, 2, 3, 4, 35)))
        self.assertTrue(deck_violations(24, (0, 1, 2, 3, 4, 5)))
        self.assertTrue(deck_violations(7, (0, 2, 3, 4, 5, 41)))
        self.assertTrue(deck_violations(7, (0, 1, 2, 3, 4, 38)))
        # 合法反例：带蘑菇 + 咖啡豆 + 起手 + 经济。
        self.assertEqual(deck_violations(7, (1, 35, 8, 0, 2, 3)), [])


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.registry = json.loads((ROOT / 'configs/random_deck_cells_v5.json').read_text())

    def test_registry_is_reproducible_and_legal(self):
        self.assertEqual(self.registry['rule_version'], RULE_VERSION)
        self.assertEqual(self.registry, cells.generate_registry())
        cells.check(self.registry)

    def test_split_and_shape(self):
        tasks = self.registry['tasks']
        self.assertEqual(len(tasks), len(cells.LEVELS) * 3)
        train = [task for task in tasks if task['split'] == 'train']
        evaluation = [task for task in tasks if task['split'] == 'eval']
        self.assertEqual((len(train), len(evaluation)), (16, 8))
        for level in cells.LEVELS:
            self.assertEqual(len([t for t in train if t['level'] == level]), 2)
            self.assertEqual(len([t for t in evaluation if t['level'] == level]), 1)
        for task in tasks:
            self.assertEqual(task['terrain'], region_for_level(task['level']))
            self.assertIn(len(task['deck']), range(6, 11))
            self.assertEqual(len(set(task['deck'])), len(task['deck']))

    def test_eval_decks_are_not_training_decks(self):
        train = {tuple(sorted(task['deck'])) for task in self.registry['tasks'] if task['split'] == 'train'}
        evaluation = {tuple(sorted(task['deck'])) for task in self.registry['tasks']
                      if task['split'] == 'eval'}
        self.assertFalse(train & evaluation)
        self.assertEqual(len(train), 16)
        self.assertEqual(len(evaluation), 8)


class PanelTest(unittest.TestCase):
    def setUp(self):
        self.registry = json.loads((ROOT / 'configs/random_deck_cells_v5.json').read_text())
        self.panels = cells.build_panels(self.registry)

    def test_training_panel_cells_and_ladder(self):
        tasks = self.panels['training_panel']['tasks']
        self.assertEqual(len(tasks), 16 * len(cells.LADDER))
        by_cell = {}
        for task in tasks:
            base = task['task_id'].split('_m')[0]
            by_cell.setdefault(base, []).append(task)
            self.assertTrue(task['random_deck_cell'])
            self.assertEqual(task['curriculum_ladder'], list(cells.LADDER))
        self.assertEqual(len(by_cell), 16)
        for base, entries in by_cell.items():
            self.assertEqual(sorted(entry['zombie_count_multiplier'] for entry in entries),
                             sorted(cells.LADDER))
            self.assertEqual([entry for entry in entries if entry['task_id'] == base][0]
                             ['zombie_count_multiplier'], 1.0)

    def test_training_seeds_are_unique_and_inside_the_reserved_block(self):
        tasks = self.panels['training_panel']['tasks']
        seeds = [seed for task in tasks for seed in task['seeds']]
        self.assertEqual(len(seeds), len(set(seeds)))
        low, high = self.registry['training_seed_block']
        self.assertGreaterEqual(min(seeds), low)
        self.assertLess(max(seeds), high)
        for task in tasks:
            self.assertEqual(len(task['seeds']), cells.SEEDS_PER_TASK)
            self.assertEqual(task['seeds'], list(range(task['seeds'][0],
                                                       task['seeds'][0] + cells.SEEDS_PER_TASK)))

    def test_low_rungs_exist_and_are_first(self):
        tasks = self.panels['training_panel']['tasks']
        self.assertIn('rd137_l7_train_d01_m050', {task['task_id'] for task in tasks})
        self.assertIn('rd137_l7_train_d01_m075', {task['task_id'] for task in tasks})
        self.assertEqual(cells.LADDER[0], 0.5)

    def test_eval_panel_is_seedless_and_flagged(self):
        tasks = self.panels['eval_panel']['tasks']
        self.assertEqual(len(tasks), 8)
        for task in tasks:
            self.assertEqual(task['seeds'], [])
            self.assertEqual(task['zombie_count_multiplier'], 1.0)
            self.assertTrue(task['random_deck_cell'])
            self.assertTrue(task['task_id'].endswith('_eval_d01'))

    def test_fog_tasks_use_human_v2_observation(self):
        for document in self.panels.values():
            for task in document['tasks']:
                if task['terrain'] == 'fog':
                    self.assertEqual(task['fog_observation'], 'human_v2')


if __name__ == '__main__':
    unittest.main()
