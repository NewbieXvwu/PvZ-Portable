"""Frozen paired-evaluation evidence gates, without launching games."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import value_widen_ab as ab


class ValueWidenABTest(unittest.TestCase):
    def fixture(self,folder,seeds):
        out=Path(folder)
        manifest=out/'manifest.json'
        manifest.write_text(json.dumps({'tasks':[{'task_id':'pool','seeds':[31000,31001]}]}))
        for arm in ('v3','v4'):
            path=out/arm/'pool'/'games.jsonl';path.parent.mkdir(parents=True)
            path.write_text('\n'.join(json.dumps({'seed':s,'won':True}) for s in seeds)+'\n')
        return SimpleNamespace(out=out),dict(tasks=['pool'],games=2,manifest=str(manifest))

    def test_wrong_frozen_seeds_cannot_pass_by_matching_arm_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            args,plan=self.fixture(folder,[1,2])
            (args.out/'paired.json').write_text('{"complete":true}')
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'unexpected seeds'):
                ab.report(args,plan)
            self.assertFalse((args.out/'paired.json').exists())

    def test_duplicate_game_rows_are_rejected_before_dict_collapse(self):
        with tempfile.TemporaryDirectory() as folder:
            args,plan=self.fixture(folder,[31000,31001,31001])
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'duplicate'):
                ab.report(args,plan)

    def test_both_arms_missing_the_same_seed_are_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            args,plan=self.fixture(folder,[31000])
            with contextlib.redirect_stdout(io.StringIO()):ab.report(args,plan)
            result=json.loads((args.out/'paired.json').read_text())
            self.assertFalse(result['complete'])
            self.assertEqual(result['results']['pool']['missing_expected_seeds'],
                             {'v3':[31001],'v4':[31001]})

    def test_complete_matching_plan_retains_paired_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            args,plan=self.fixture(folder,[31000,31001])
            with contextlib.redirect_stdout(io.StringIO()):ab.report(args,plan)
            result=json.loads((args.out/'paired.json').read_text())
            self.assertTrue(result['complete'])
            self.assertEqual(result['results']['pool']['pairs'],2)
            self.assertEqual(result['results']['pool']['delta_wins'],0)
            self.assertEqual(result['results']['pool']['mcnemar_exact_two_sided'],1)


if __name__=='__main__':unittest.main()
