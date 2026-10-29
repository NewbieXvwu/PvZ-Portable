"""task137：v5 随机卡组族（登记表 / 生成器 / 面板）的静态检查，不跑引擎。

覆盖：v5 规则可复现、训练与评估卡组互不重叠、面板格子数与倍率阶梯、
种子段唯一且不与现有清单重叠、登记表里的卡组能由 draw_seed 重放。
"""
import json
import hashlib
import sys
import unittest
from pathlib import Path

import pytest

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


def test_data_levers_disabled_course_is_bitwise_unchanged():
    registry = json.loads(cells.REGISTRY.read_text())
    assert cells.load_deck_pool(registry) == []
    payload = json.dumps(cells.build_panels(registry, []), sort_keys=True).encode()
    # task137 panels before task159, including every seed and all task fields.
    assert hashlib.sha256(payload).hexdigest() == 'c474925d0502aa31bf677dd6c592ad85b446d2078b961900663f9947d456d50c'


def test_data_levers_pool_deduplicates_wins_filters_rules_and_holdouts(tmp_path):
    from pvz_tasks import profile_for_deck
    registry = json.loads(cells.REGISTRY.read_text())
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path
    # v6's admitted roof decks must stay held out even at a different level.
    reserved = [1, 2, 14, 15, 17, 30, 32, 33, 35, 41]
    heldout = write('gate_v6.json', dict(tasks=[dict(level=42, deck=reserved)]))
    deck = [0, 1, 2, 3, 4, 7, 40]
    rows = []
    cases = [(7, deck, '通过', 1), (7, list(reversed(deck)), '通过', 1),
             (8, deck, '通过', 1), (7, [0, 1, 2, 3, 4, 38], '通过', 1),
             (24, [0, 1, 2, 3, 4, 7], '通过', 1), (47, reserved, '通过', 1),
             (7, generate_deck(7, 8, 159), '失败', 1),
             (7, generate_deck(7, 8, 160), '通过', 2),
             (7, [0, 1, 2, 3, 4, 40], '通过', 1)]  # 升级缺底座。
    for index, (level, cards, verify, result) in enumerate(cases):
        archive = f'archives/{index}'
        # Rejected/lost ledger entries must never cause archive reads.
        if verify == '通过' and result == 1:
            write(f'{archive}/meta.json', dict(task=dict(level=level, deck=list(cards)),
                                              outcome=dict(result=1)))
        rows.append(dict(verify=verify, result=result, grade='half' if result == 1 else 'loss', archive=archive))
    ledger = write('merged_progress.json', dict(archives=rows))
    roof_deck = [1, 2, 3, 4, 6, 20, 29, 30, 32, 33]
    roof = write('roof.json', dict(candidates=[dict(task=dict(level=45, deck=roof_deck)),
                                              dict(task=dict(level=48, deck=list(reversed(reserved))))]))
    pool = cells.load_deck_pool(registry, gpt_ledger=ledger, archive_root=tmp_path,
                               roof_candidates=roof, heldout_panels=[heldout])
    assert [(task['level'], task['deck']) for task in pool] == [(7, deck), (45, roof_deck)]
    assert {task['source'] for task in pool} == {'task105', 'task148'}
    assert pool[0]['profile']['owned_upgrade_plants'] == [40]
    assert profile_for_deck(tuple(pool[0]['deck'])).owned_upgrade_plants == (40,)
    assert all(task['profile']['seed_slot_count'] == len(task['deck']) for task in pool)
    assert cells.load_deck_pool(registry, gpt_ledger=ledger, archive_root=tmp_path,
                                roof_candidates=roof, heldout_panels=[heldout]) == pool
    panels = cells.build_panels(registry, pool)
    baseline = cells.build_panels(registry)
    tasks = panels['training_panel']['tasks']
    assert panels['eval_panel'] == baseline['eval_panel']
    assert tasks[:len(baseline['training_panel']['tasks'])] == baseline['training_panel']['tasks']
    added = tasks[len(baseline['training_panel']['tasks']):]
    assert len(added) == 2 * len(cells.LADDER)
    holdouts = cells.heldout_signatures(registry, [heldout])
    assert not {tuple(sorted(task['deck'])) for task in tasks} & holdouts
    assert len({tuple(task['deck']) for task in pool}) == len(pool)
    seeds = [seed for task in added for seed in task['seeds']]
    assert len(set(seeds)) == len(seeds)
    assert min(seeds) == cells.POOL_SEED_BASE
    assert max(seeds) < 2**31
    assert all(task['profile']['seed_slot_count'] == len(task['deck']) for task in added)
    only_heldout = write('heldout_candidates.json', dict(candidates=[dict(task=dict(level=47, deck=reserved))]))
    with pytest.raises(ValueError, match='no eligible external decks'):
        cells.load_deck_pool(registry, roof_candidates=only_heldout, heldout_panels=[heldout])


if __name__ == '__main__':
    unittest.main()
