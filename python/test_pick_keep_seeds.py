"""task31：条件采样选种子的静态检查（不跑引擎）。

挑出的每个种子都必须真的把目标底卡换掉 —— 这是“条件采样”的定义，
选错极性（挑成“保留这张卡”）会让整批老师局白跑。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

import pick_keep_seeds
from pvz_tasks import deck_for_seed


def manifest(tmp, level=12, deck=(0, 1, 2, 3, 4, 5, 7, 8, 9, 11)):
    options = dict(level=level, playthrough=2, available_plants=list(range(40)),
                   not_recommended=[[p, 0] for p in range(40)])
    task = dict(task_id='t', level=level, playthrough=2, deck=list(deck),
                seeds=list(range(2000, 2000 + 400)),
                deck_perturb=dict(swaps=[1, 2], pool=list(range(40)), engine_options=options))
    path = Path(tmp) / 'panel.json'
    path.write_text(json.dumps(dict(tasks=[task])))
    return path, task


class PickKeepSeedsTest(unittest.TestCase):
    def run_picker(self, argv):
        old = sys.argv
        sys.argv = ['pick_keep_seeds.py'] + argv
        try:
            pick_keep_seeds.main()
        finally:
            sys.argv = old

    def test_every_picked_seed_actually_drops_the_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, task = manifest(tmp)
            out = Path(tmp) / 'seeds.json'
            self.run_picker(['--manifest', str(path), '--task-id', 't', '--card', '8',
                             '--count', '12', '--out', str(out)])
            picked = json.loads(out.read_text())
            self.assertEqual(len(picked['seeds']), 12)
            self.assertEqual(len(set(picked['seeds'])), 12)
            for seed in picked['seeds']:
                self.assertNotIn(8, deck_for_seed(task, seed))
            self.assertGreater(picked['eligible'], 12)
            # determinism
            self.assertEqual(picked['seeds'], json.loads(out.read_text())['seeds'])

    def test_already_played_seeds_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, task = manifest(tmp)
            first = Path(tmp) / 'first.json'
            self.run_picker(['--manifest', str(path), '--task-id', 't', '--card', '9',
                             '--count', '6', '--out', str(first)])
            played = json.loads(first.read_text())['seeds']
            run = Path(tmp) / 'run'
            run.mkdir()
            (run / 'games.jsonl').write_text(''.join(json.dumps(dict(seed=s, won=True)) + chr(10)
                                                     for s in played))
            second = Path(tmp) / 'second.json'
            self.run_picker(['--manifest', str(path), '--task-id', 't', '--card', '9',
                             '--count', '6', '--played', str(run), '--out', str(second)])
            self.assertFalse(set(json.loads(second.read_text())['seeds']) & set(played))

    def test_card_must_be_a_base_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = manifest(tmp)
            with self.assertRaises(SystemExit):
                self.run_picker(['--manifest', str(path), '--task-id', 't', '--card', '13',
                                 '--count', '2'])

    def test_evenly_spaced_is_sorted_and_inclusive(self):
        items = list(range(100))
        picked = pick_keep_seeds.evenly_spaced(items, 5)
        self.assertEqual(picked, [0, 25, 50, 74, 99])
        self.assertEqual(pick_keep_seeds.evenly_spaced(items, 1), [50])
        self.assertEqual(pick_keep_seeds.evenly_spaced(items, 200), items)


if __name__ == '__main__':
    unittest.main()
