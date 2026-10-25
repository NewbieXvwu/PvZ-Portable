"""task29 第 1 步：倍率变体生成器的静态检查（不跑引擎）。

覆盖：命名 / 种子段 / 字段复制 / 扰动族保留 deck_perturb / 每种子卡组合法 /
同一面板内种子互不重叠。引擎侧“倍率真的放大每一波”由 scripts/smoke_zombie_multiplier.py 冒烟。
"""
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

from make_multiplier_tasks import (FAMILIES, INDEPENDENT_BLOCKS, MULT_STRIDE, MULTIPLIERS,
                                   SEEDS_PER_TASK, SLOTS_PER_FAMILY, build_variants, seed_start,
                                   variant_id)
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
        for _family, key, pattern, levels, terrain in FAMILIES:
            self.sources.setdefault(key, {})
            for level in levels:
                name = pattern.format(level=level)
                perturb = key in ('day_d7q', 'night_n10p', 'night_n10q')
                deck = () if key.endswith('_orig') else (
                    (0, 1, 2, 3, 4, 5, 7, 16) if terrain == 'fog' else (0, 1, 2, 3, 4, 5, 7))
                self.sources[key][name] = source_task(
                    level, deck, perturb=perturb, conveyor=key.endswith('_orig'), tag='_' + key)
                # Multiple families may share one source manifest (fog f8/f8p).
                self.sources[key][name]['task_id'] = name
                self.sources[key][name]['terrain'] = terrain

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
        # Roof and the task68 fog_user family use their own blocks; only the legacy
        # families share the 700k stride.
        block = (sum(f[0] not in INDEPENDENT_BLOCKS for f in FAMILIES)
                 * SLOTS_PER_FAMILY * SEEDS_PER_TASK)
        self.assertLess(block, MULT_STRIDE)
        for family_index, (_name, _key, _pattern, levels, _terrain) in enumerate(FAMILIES):
            self.assertGreater(min(seed_start(1, family_index, level) for level in levels),
                               max(seed_start(0, family_index, level) + SEEDS_PER_TASK - 1 for level in levels))

    def test_roof_allocation_preserves_every_existing_seed_block(self):
        """Re-adding roof last must not move any other family's preregistered seeds."""
        # task70: fog_user_v2 was appended after roof, so it is not part of the seed set
        # roof had to preserve either.
        old = {(m, entry[0], level): seed_start(m, f, level)
               for f, entry in enumerate(FAMILIES) if entry[0] not in ('roof', 'fog_user_v2')
               for m in range(len(MULTIPLIERS)) for level in entry[3]}
        roof = ('roof', 'roof', 'roof_l{level}_train_r7', (41, 42, 43, 44, 46, 47, 48, 49), 'roof')
        families = tuple(f for f in FAMILIES if f[0] not in ('roof', 'fog_user_v2')) + (roof,)
        with patch.object(generator, 'FAMILIES', families):
            index = {entry[0]: i for i, entry in enumerate(families)}
            for (m, name, level), value in old.items():
                self.assertEqual(seed_start(m, index[name], level), value)
            starts = [seed_start(m, len(families) - 1, level)
                      for m in range(len(MULTIPLIERS)) for level in roof[3]]
            self.assertEqual(min(starts), 48000000)
            for left, right in zip(sorted(starts), sorted(starts)[1:]):
                self.assertGreaterEqual(right - left, SEEDS_PER_TASK)
            self.assertGreater(min(starts), max(old.values()) + SEEDS_PER_TASK)

    def test_merged_families_keep_preregistered_seed_starts(self):
        legacy = ('day_d7', 'day_orig', 'night_n10', 'night_orig', 'pool_p9k',
                  'day_d7q', 'night_n10p', 'night_n10q', 'fog_f8', 'fog_f8p')
        for index, (family, _key, _pattern, levels, _terrain) in enumerate(FAMILIES):
            for mult in range(6):
                for level in levels:
                    if family == 'roof':
                        expected = 48000000 + mult * 100000 + (level - 41) * 2048
                    elif family == 'fog_user':
                        # task68: new family, new block; the base task keeps 43_000_000.
                        expected = 43100000 + mult * 100000 + (level - 31) * 2048
                    elif family == 'fog_user_v2':
                        # task70: another independent block above roof (48.52M-50M free window).
                        expected = 49200000 + mult * 100000 + (level - 31) * 2048
                    else:
                        expected = (5000000 + mult * 700000 +
                                    (legacy.index(family) * 32 + level - 1) * 2048)
                    self.assertEqual(seed_start(mult, index, level), expected)

    def test_fog_user_cell_preserves_every_existing_cell(self):
        """task68: appending the fog_user family leaves every preregistered cell untouched.

        The panel is regenerated as a whole on the deployment machine, so the proof is
        structural: drop the family, rebuild, and require the surviving task list to be
        byte-identical (ids, seeds, offsets, every copied field) and the new seeds disjoint.
        """
        # task70: fog_user_v2 now also exists; the original task68 proof (appending fog_user
        # preserved everything) must drop both fog families from the "before" build.
        old_families = tuple(f for f in FAMILIES if f[0] not in ('fog_user', 'fog_user_v2'))
        old_sources = {key: table for key, table in self.sources.items()
                       if key not in ('fog_user', 'fog_user_v2')}
        with patch.object(generator, 'FAMILIES', old_families):
            before = build_variants(old_sources, seeds=SEEDS_PER_TASK)
        families_with_user = old_families + tuple(
            f for f in FAMILIES if f[0] == 'fog_user')
        with patch.object(generator, 'FAMILIES', families_with_user):
            with_user = build_variants({**old_sources, 'fog_user': self.sources['fog_user']},
                                       seeds=SEEDS_PER_TASK)
        after = build_variants(self.sources, seeds=SEEDS_PER_TASK)
        for old, new in zip(before, with_user):
            self.assertEqual(json.dumps(old, sort_keys=True), json.dumps(new, sort_keys=True))
        for old, new in zip(with_user, after):
            self.assertEqual(json.dumps(old, sort_keys=True), json.dumps(new, sort_keys=True))
        added = with_user[len(before):]
        self.assertEqual([task['task_id'] for task in added],
                         [variant_id('fog_user_l31_snow', suffix) for _m, suffix in MULTIPLIERS])
        old_seeds = {seed for task in before for seed in task['seeds']}
        new_seeds = {seed for task in added for seed in task['seeds']}
        self.assertFalse(old_seeds & new_seeds)
        self.assertEqual(min(new_seeds), 43100000)
        self.assertEqual(max(new_seeds), 43100000 + 5 * 100000 + SEEDS_PER_TASK - 1)
        import make_fog_user_panel as panel
        base_seeds = set(range(panel.TRAIN_SEED_BASE, panel.TRAIN_SEED_BASE + panel.TRAIN_SEEDS))
        eval_seeds = set(range(panel.EVAL_SEED_BASE, panel.EVAL_SEED_BASE + panel.EVAL_SEEDS))
        self.assertFalse(base_seeds & (old_seeds | new_seeds))
        self.assertFalse(eval_seeds & (old_seeds | new_seeds))
        self.assertLess(max(base_seeds), min(new_seeds))
        self.assertGreater(min(eval_seeds), max(new_seeds))
        # Demo seeds (task63/64) and the task67 cold-start panel are separate blocks too.
        external = set(range(20261031, 20261035)) | set(range(20261101, 20261133))
        self.assertFalse(external & (old_seeds | new_seeds | base_seeds | eval_seeds))

    def test_fog_user_panel_task_definition(self):
        """The source manifest carries the task67 deck, human fog observation, fresh seeds."""
        import make_fog_user_panel as panel
        document = panel.document()
        train, evaluation = document['tasks']
        self.assertEqual(train['task_id'], 'fog_user_l31_snow')
        self.assertEqual(train['deck'], [9, 8, 10, 16, 23, 17, 19, 2, 3, 5])
        self.assertEqual(train['level'], 31)
        self.assertEqual(train['terrain'], 'fog')
        self.assertEqual(train['fog_observation'], 'human')
        self.assertEqual(train['zombie_count_multiplier'], 1.0)
        self.assertEqual(train['playthrough'], 2)
        self.assertEqual(len(train['seeds']), panel.TRAIN_SEEDS)
        self.assertEqual(evaluation['seeds'][0], panel.EVAL_SEED_BASE)
        self.assertFalse(set(train['seeds']) & set(evaluation['seeds']))
        # AGENTS section 8: the deck covers the pool terrain and needs no upgrade ownership.
        self.assertIn(16, train['deck'])
        self.assertNotIn(43, train['deck'])

    def test_fog_user_v2_cell_preserves_every_existing_cell(self):
        """task70: appending fog_user_v2 (and superseding fog_user) leaves every cell intact.

        Same structural proof as task68: drop fog_user_v2, rebuild, and require the
        surviving task list to be byte-identical (ids, seeds, offsets, every copied field)
        while the new seeds stay disjoint from all preregistered blocks.
        """
        old_families = tuple(f for f in FAMILIES if f[0] != 'fog_user_v2')
        old_sources = {key: table for key, table in self.sources.items() if key != 'fog_user_v2'}
        with patch.object(generator, 'FAMILIES', old_families):
            before = build_variants(old_sources, seeds=SEEDS_PER_TASK)
        after = build_variants(self.sources, seeds=SEEDS_PER_TASK)
        self.assertEqual(len(after), len(before) + len(MULTIPLIERS))
        for old, new in zip(before, after):
            self.assertEqual(json.dumps(old, sort_keys=True), json.dumps(new, sort_keys=True))
        added = after[len(before):]
        self.assertEqual([task['task_id'] for task in added],
                         [variant_id('fog_user_v2_l31_snow', suffix) for _m, suffix in MULTIPLIERS])
        old_seeds = {seed for task in before for seed in task['seeds']}
        new_seeds = {seed for task in added for seed in task['seeds']}
        self.assertFalse(old_seeds & new_seeds)
        self.assertEqual(min(new_seeds), 49200000)
        self.assertEqual(max(new_seeds), 49200000 + 5 * 100000 + SEEDS_PER_TASK - 1)
        import make_fog_user_v2_panel as panel
        base_seeds = set(range(panel.TRAIN_SEED_BASE, panel.TRAIN_SEED_BASE + panel.TRAIN_SEEDS))
        eval_seeds = set(range(panel.EVAL_SEED_BASE, panel.EVAL_SEED_BASE + panel.EVAL_SEEDS))
        self.assertFalse(base_seeds & (old_seeds | new_seeds))
        self.assertFalse(eval_seeds & (old_seeds | new_seeds))
        self.assertLess(max(old_seeds), min(base_seeds))
        self.assertLess(max(base_seeds), min(eval_seeds))
        self.assertLess(max(eval_seeds), min(new_seeds))
        # The old fog_user cells move over byte-identically (the family is superseded, not
        # edited): every fog_user task in the new build matches the without-v2 build.
        fog_user_ids = [task['task_id'] for task in before if task['task_id'].startswith('fog_user_l31')]
        self.assertEqual(len(fog_user_ids), len(MULTIPLIERS))
        for task_id in fog_user_ids:
            old_task = next(t for t in before if t['task_id'] == task_id)
            new_task = next(t for t in after if t['task_id'] == task_id)
            self.assertEqual(json.dumps(old_task, sort_keys=True), json.dumps(new_task, sort_keys=True))

    def test_fog_user_v2_panel_task_definition(self):
        """The source manifest carries the task68 deck under human_v2, fresh seeds."""
        import make_fog_user_v2_panel as panel
        document = panel.document()
        train, evaluation = document['tasks']
        self.assertEqual(train['task_id'], 'fog_user_v2_l31_snow')
        self.assertEqual(train['deck'], [9, 8, 10, 16, 23, 17, 19, 2, 3, 5])
        self.assertEqual(train['level'], 31)
        self.assertEqual(train['terrain'], 'fog')
        self.assertEqual(train['fog_observation'], 'human_v2')
        self.assertEqual(train['zombie_count_multiplier'], 1.0)
        self.assertEqual(train['playthrough'], 2)
        self.assertEqual(len(train['seeds']), panel.TRAIN_SEEDS)
        self.assertEqual(evaluation['seeds'][0], panel.EVAL_SEED_BASE)
        self.assertFalse(set(train['seeds']) & set(evaluation['seeds']))
        # Same deck as task68's fog_user panel; only the observation differs.
        import make_fog_user_panel as old_panel
        old_train = old_panel.document()['tasks'][0]
        self.assertEqual(train['deck'], old_train['deck'])
        self.assertEqual(train['seeds'][0], old_train['seeds'][0] + 6_000_000)
        # AGENTS section 8: the deck covers the pool terrain and needs no upgrade ownership.
        self.assertIn(16, train['deck'])
        self.assertNotIn(43, train['deck'])

    def test_fog_user_v2_registered_but_not_admitted(self):
        """task70: fog_user_v2 has a manifest and seeds but waits for the paired cold start."""
        import make_multiplier_tasks as module
        from curriculum_queue import training_base_ids
        self.assertNotIn('fog_user', module.SUPERSEDED)
        self.assertEqual(module.CURRICULUM_LEVELS.get('fog_user_v2'), ())
        self.assertEqual(module.CURRICULUM_LEVELS.get('fog_user'), (31,))
        ids = training_base_ids()
        self.assertIn('fog_user_l31_snow', ids)
        self.assertNotIn('fog_user_v2_l31_snow', ids)

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
