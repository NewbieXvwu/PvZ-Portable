"""Task46 report integrity: missing/duplicate games must not become completed evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from roof_report import summarize


class RoofReportTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps(dict(tasks=[dict(task_id='roof_l41_eval', level=41,
            deck=[0, 1, 2, 3, 4, 5, 33], zombie_count_multiplier=1, seeds=[45000000, 45000001])])) )
        self.game = dict(seed=45000000, arm='fast_search', won=False, final_wave=6, wave_count=10,
                         final_tick=9000, terminal=True, final_zombies=[
                             dict(row=4, type=2, x=300, health=300),
                             dict(row=1, type=0, x=-101, health=50)])
        self.results = self.root / 'roof_l41_eval_teacher_2/games.jsonl'
        self.results.parent.mkdir()

    def report(self, rows):
        self.results.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return summarize(self.manifest, self.root, ['teacher'], 2)

    def test_partial_results_do_not_pass_completion(self):
        result = self.report([self.game])
        self.assertFalse(result['complete'])
        loss = result['cells'][0]['losses'][0]
        self.assertEqual(loss['wave'], 6)
        self.assertEqual(loss['leftmost_zombie']['row'], 1)
        self.assertEqual(loss['leftmost_zombie']['type'], 0)

    def test_complete_requires_the_exact_seed_set(self):
        second = dict(self.game, seed=45000001, won=True)
        result = self.report([self.game, second])
        self.assertTrue(result['complete'])
        self.assertEqual(result['cells'][0]['win_rate'], 0.5)

    def test_duplicate_or_foreign_seed_is_rejected(self):
        for rows in ([self.game, self.game], [dict(self.game, seed=30000)]):
            with self.assertRaises(ValueError):
                self.report(rows)


if __name__ == '__main__':
    unittest.main()
