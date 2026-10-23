"""The report's intervention-dose table must come from audited traces, not claims."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
from report_llm_candidate_study import dose


class DoseTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)
        for fi,family in enumerate(('day','roof','fog')):
            for arm in ('A','B_low','B_novel'):
                for offset in range(16):
                    seed=52000000+1000*fi+offset
                    folder=self.out/'games'/family/arm/str(seed);folder.mkdir(parents=True)
                    trace=[dict(tick=100*i,chosen=dict(type='plant',packet=0,row=0,col=0)) for i in range(3)]
                    if arm=='B_novel' and seed==52001001:
                        trace[1]=dict(trace[1],chosen=dict(type='plant',packet=1,row=2,col=3))
                    (folder/'trace.json').write_text(json.dumps(trace))
                    (folder/'result.json').write_text(json.dumps(dict(
                        contract='c',trace_sha256=hashlib.sha256((folder/'trace.json').read_bytes()).hexdigest(),
                        decisions=3,final_tick=200,final_wave=20,won=True)))

    def test_only_the_changed_pair_is_reported(self):
        study=dose(self.out,'c')
        self.assertEqual(study['pairs'],96)
        self.assertEqual(study['diverged'],1)
        row=study['rows'][0]
        self.assertEqual((row['family'],row['arm'],row['seed']),('roof','B_novel',52001001))
        self.assertEqual(row['first_diff_index'],1)
        self.assertEqual(row['first_diff_tick'],100)
        self.assertEqual(row['differing_decisions'],1)

    def test_contract_and_trace_hash_are_enforced(self):
        with self.assertRaises(AssertionError):
            dose(self.out,'other-contract')
        path=self.out/'games'/'day'/'A'/'52000000'/'trace.json'
        path.write_text(json.dumps([dict(tick=0,chosen=dict(type='wait'))]))
        with self.assertRaises(AssertionError):
            dose(self.out,'c')


if __name__=='__main__':unittest.main()
