import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import value_sparse_pipeline as pipeline
from value_teacher_feasibility import write_json


class SparsePipelineTest(unittest.TestCase):
    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'process identity is read from /proc (Linux only)')
    def test_live_process_identity_uses_real_process_handle(self):
        self.assertIsNotNone(pipeline.process_identity(os.getpid()))
        self.assertIsNone(pipeline.process_identity(999999999))

    def test_terminated_run_without_success_cannot_start_next_phase(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);run=r/'run.json';write_json(run,dict(phase='labels',seeds=list(range(8)),pid=999999999))
            with patch.object(pipeline,'process_identity',return_value=None):
                with self.assertRaises(pipeline.PipelineStop):pipeline.wait_prior(run,dict(bootstrap_seeds=list(range(8))),Mock())
                write_json(r/'result.json',dict(status='stopped',reason='budget'))
                with self.assertRaises(pipeline.PipelineStop):pipeline.wait_prior(run,dict(bootstrap_seeds=list(range(8))),Mock())

    def test_failed_eight_seed_gate_does_not_launch_remaining_collection(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);write_json(r/'gate8.json',dict(status='complete',approved_to_continue=False,nonterminal_effective_pairs=124))
            call=Mock();publish=Mock();status=Mock()
            with patch.object(pipeline,'load_plan',return_value=dict(out=d)):
                pipeline.advance(r/'plan.json',call,publish,status)
            call.assert_called_once_with('gate8')
            self.assertEqual(publish.call_args.args[0],'bootstrap_gate_failed')
            self.assertEqual(status.call_args.args[0],'gated_stop')

    def test_publication_refuses_unrelated_staged_files(self):
        with patch.object(pipeline,'git',return_value='python/unrelated.py\n') as git:
            with self.assertRaises(pipeline.PipelineStop):pipeline.publish('stage',{})
            git.assert_called_once_with('diff','--cached','--name-only')


if __name__=='__main__':unittest.main()
