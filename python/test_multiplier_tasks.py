"""task29 第 1 步：倍率变体生成器的静态检查（不跑引擎）。

覆盖：命名 / 种子段 / 字段复制 / 扰动族保留 deck_perturb / 每种子卡组合法 /
同一面板内种子互不重叠。引擎侧“倍率真的放大每一波”由 scripts/smoke_zombie_multiplier.py 冒烟。
"""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

from make_multiplier_tasks import (FAMILIES, MULT_STRIDE, MULTIPLIERS, SEEDS_PER_TASK,
                                   SLOTS_PER_FAMILY, build_variants, seed_start, variant_id)
from pvz_tasks import deck_for_seed
import make_multiplier_tasks as generator


def source_task(level, deck, perturb=False, conveyor=False, tag=''):
    task = dict(task_id=f'test{tag}_l{level}_train', terrain='night' if level >= 11 else 'day',
                level=level, deck=list(deck), zombie_count_multiplier=1.0, wave_cap=None,
                sun_start=50, preplanted=[], playthrough=2,
                seeds=list(range(1000, 1000 + SEEDS_PER_TASK)))
    if conveyor:
        task.update(deck=[], conveyor_start='original')
    if perturb:
        task['deck_perturb'] = dict(swaps=[1], pool=[10, 12], excluded_decks=[],
                                    engine_options=dict(level=level, playthrough=2,
                                                        available_plants=list(range(40)),
                                                        not_recommended=[[p, 0] for p in range(40)]))
    return task


class MultiplierTasksTest(unittest.TestCase):
    def setUp(self):
        self.sources = {}
        for _family, key, pattern, levels, _terrain in FAMILIES:
            self.sources.setdefault(key, {})
            for level in levels:
                name = pattern.format(level=level)
                perturb = key in ('day_d7q', 'night_n10p', 'night_n10q')
                self.sources[key][name] = source_task(
                    level, () if key.endswith('_orig') else (0, 1, 2, 3, 4, 5, 7),
                    perturb=perturb, conveyor=key.endswith('_orig'), tag='_' + key)

    def test_variant_ids_and_multipliers(self):
        tasks = build_variants(self.sources, seeds=64)
        ids = [t['task_id'] for t in tasks]
        self.assertEqual(len(ids), len(set(ids)))
        for task in tasks:
            suffix = task['task_id'].rsplit('_', 1)[1]
            expected = dict((s, m) for m, s in MULTIPLIERS)[suffix]
            self.assertEqual(task['zombie_count_multiplier'], expected)

    def test_seed_blocks_do_not_overlap(self):
        tasks = build_variants(self.sources, seeds=SEEDS_PER_TASK)
        owner = {}
        for task in tasks:
            for seed in task['seeds']:
                self.assertNotIn(seed, owner)
                owner[seed] = task['task_id']
        # 每档倍率的块宽 = 族数 * 32 槽 * 每任务种子数，必须小于档间距。
        block = len(FAMILIES) * SLOTS_PER_FAMILY * SEEDS_PER_TASK
        self.assertLess(block, MULT_STRIDE)
        for family_index, (_name, _key, _pattern, levels, _terrain) in enumerate(FAMILIES):
            self.assertGreater(min(seed_start(1, family_index, level) for level in levels),
                               max(seed_start(0, family_index, level) + SEEDS_PER_TASK - 1 for level in levels))

    def test_roof_allocation_preserves_every_existing_seed_block(self):
        old = {(m, f, level): seed_start(m, f, level)
               for m in range(len(MULTIPLIERS))
               for f, entry in enumerate(FAMILIES) for level in entry[3]}
        roof = ('roof', 'roof', 'roof_l{level}_train_r7', (41, 42, 43, 44, 46, 47, 48, 49), 'roof')
        families = tuple(f for f in FAMILIES if f[0] != 'roof') + (roof,)
        with patch.object(generator, 'FAMILIES', families):
            for key, value in old.items():
                if FAMILIES[key[1]][0] != 'roof':
                    self.assertEqual(seed_start(*key), value)
            starts = [seed_start(m, len(families) - 1, level)
                      for m in range(len(MULTIPLIERS)) for level in roof[3]]
            self.assertEqual(min(starts), 48000000)
            for left, right in zip(sorted(starts), sorted(starts)[1:]):
                self.assertGreaterEqual(right - left, SEEDS_PER_TASK)
            self.assertGreater(min(starts), max(v for k, v in old.items()
                                                if FAMILIES[k[1]][0] != 'roof') + SEEDS_PER_TASK)

    def test_fields_copied_and_perturbation_kept(self):
        tasks = build_variants(self.sources, seeds=32)
        by_id = {t['task_id']: t for t in tasks}
        for key, table in self.sources.items():
            for base in table.values():
                for _, suffix in MULTIPLIERS:
                    variant = by_id[variant_id(base['task_id'], suffix)]
                    for field, value in base.items():
                        if field in ('task_id', 'seeds', 'zombie_count_multiplier'):
                            continue
                        self.assertEqual(variant.get(field), value)
                    self.assertEqual(variant['seeds'], list(range(variant['seeds'][0],
                                                                   variant['seeds'][0] + 32)))

    def test_every_seed_yields_a_legal_deck(self):
        tasks = build_variants(self.sources, seeds=64)
        for task in tasks:
            for seed in task['seeds']:
                deck = deck_for_seed(task, seed)
                self.assertEqual(len(deck), len(task['deck']))
                self.assertEqual(len(set(deck)), len(deck))
                if 'deck_perturb' not in task:
                    self.assertEqual(tuple(deck), tuple(task['deck']))


if __name__ == '__main__':
    unittest.main()
