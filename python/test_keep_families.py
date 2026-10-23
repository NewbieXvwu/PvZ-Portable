"""task31：n10q 族生成器的静态检查（不跑引擎、不跑局）。"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

import make_night_n10q as M
from pvz_tasks import deck_for_seed, terrain_covered

DECK = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11]


def source_task(level, seed_base):
    options = dict(level=level, playthrough=2, available_plants=list(range(40)),
                   not_recommended=[[p, 0] for p in range(40)])
    return dict(task_id=f'night_l{level}_train_n10p', level=level, playthrough=2, deck=list(DECK),
                seeds=list(range(seed_base, seed_base + 64)),
                deck_perturb=dict(swaps=[1, 2], pool=[10, 12, 13, 14, 15, 17, 20, 26],
                                  engine_options=options, excluded_decks=[], require_no_warnings=True,
                                  engine_warned_decks=[[0, 1, 2, 3, 4, 5, 7, 8, 9, 17]]))


class KeepFamiliesTest(unittest.TestCase):
    def setUp(self):
        self.levels = (11, 16)
        self.source = {'tasks': [source_task(11, 2400000), source_task(16, 2450000)]}

    def build(self, keep=(8, 9), train_count=512):
        tasks, held = M.build(self.source, self.levels, list(keep), train_count, 2)
        return tasks, held

    def test_keep_cards_never_replaced_and_seeds_in_new_block(self):
        tasks, _ = self.build()
        for level, task in zip(self.levels, tasks):
            start = M.BASE_SEED + M.LEVEL_STRIDE * (level - M.LEVELS[0])
            self.assertEqual(task['seeds'][0], start)
            self.assertEqual(task['deck_perturb']['keep_plants'], [8, 9])
            for seed in task['seeds']:
                deck = deck_for_seed(task, seed)
                self.assertIn(8, deck)
                self.assertIn(9, deck)
                self.assertIn(len(set(deck) - set(task['deck'])), (1, 2))
                self.assertTrue(terrain_covered(task['level'], deck))

    def test_every_holdout_deck_is_excluded_from_every_level(self):
        tasks, held = self.build()
        holdouts = {tuple(sorted(t['deck'])) for t in held}
        self.assertEqual(len(holdouts), 4)
        for task in tasks:
            excluded = {tuple(sorted(d)) for d in task['deck_perturb']['excluded_decks']}
            self.assertTrue(holdouts <= excluded)
            for seed in task['seeds']:
                self.assertNotIn(tuple(sorted(deck_for_seed(task, seed))), holdouts)

    def test_holdout_tasks_carry_the_shared_eval_seeds_and_no_perturbation(self):
        _, held = self.build()
        for task in held:
            self.assertNotIn('deck_perturb', task)
            self.assertEqual(task['seeds'], M.HOLDOUT_EVAL_SEEDS)
            self.assertEqual(task['split'], 'heldout_deck')

    def test_audit_reports_coverage_and_no_seed_overlap(self):
        tasks, held = self.build()
        result = dict(tasks=tasks + held)
        with_dir = Path('/tmp')
        manifest = with_dir / 'task31_n10q_test_scan.json'
        manifest.write_text(json.dumps(dict(tasks=[dict(task_id='other', seeds=[M.BASE_SEED, 1])])))
        try:
            with self.assertRaises(SystemExit):
                M.audit(result, self.levels, [8, 9], [manifest])
            manifest.write_text(json.dumps(dict(tasks=[dict(task_id='other', seeds=[1, 2])])))
            audit = M.audit(result, self.levels, [8, 9], [manifest])
        finally:
            manifest.unlink()
        self.assertEqual(audit['training_seeds'], 1024)
        self.assertEqual(audit['keep_plants'], [8, 9])
        self.assertEqual(audit['seed_overlap'], 0)
        self.assertTrue(all(audit['required_pool_deck_counts'][str(p)] for p in M.REQUIRED_POOL))


if __name__ == '__main__':
    unittest.main()
