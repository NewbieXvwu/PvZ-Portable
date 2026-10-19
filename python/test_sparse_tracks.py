"""CPU sparse-track activation must be explicit and reject unsupported backends/builds."""
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import search_play
from pvz_event_env import EventWaitEnv
from pvz_env import PvZEnv


class SparseTrackValidationTest(unittest.TestCase):
    def test_unsupported_cli_combinations_fail_before_outputs(self):
        for options in [['--cpu-sparse-tracks','--rollout-backend','cuda'],
                        ['--rollout-sim-skip','317','--cuda-socket','/nonexistent'],
                        ['--cpu-sparse-tracks','--rollout-sim-skip','0'],
                        ['--cpu-sparse-tracks','--no-cpp-rollouts']]:
            message=io.StringIO()
            with self.subTest(options=options),redirect_stderr(message),patch.object(
                    sys,'argv',['search_play.py','--out','/nonexistent/sparse-test',
                                '--rollout-policy','/absent','--value','/absent',
                                '--manifest','/absent','--task-id','test',*options]):
                with self.assertRaises(SystemExit) as raised:search_play.main()
                self.assertEqual(raised.exception.code,2)
                self.assertIn('CPU sparse tracks require',message.getvalue())

    def test_worker_rejects_a_cuda_connection_before_loading_models(self):
        with self.assertRaisesRegex(ValueError,'CPU-only'):
            search_play._init_worker('/absent','/absent',317,True,None,'/absent')

    def test_old_capability_blocks_sparse_commands(self):
        env=PvZEnv('/absent')
        with patch.object(env,'_start'),patch.object(env,'_command') as command:
            with self.assertRaisesRegex(RuntimeError,'rebuilt simulator'):env.set_sim_skip(317)
            command.assert_not_called()
            env._sparse_tracks_version=1
            env.require_sparse_tracks()


RESOURCE=Path(os.environ.get('PVZ_RESOURCE_DIR','/absent'))
BINARY=Path(os.environ.get('PVZ_EXECUTABLE','/absent'))
REFERENCE=Path(os.environ.get('PVZ_REFERENCE_EXECUTABLE',ROOT/'build/pvz-portable'))


@unittest.skipUnless((RESOURCE/'main.pak').is_file() and BINARY.is_file() and REFERENCE.is_file(),
                     'requires candidate, frozen reference and game resources')
class SparseTrackNativeCapabilityTest(unittest.TestCase):
    def test_new_build_confirms_feature_and_old_build_rejects_before_reset(self):
        candidate=EventWaitEnv(RESOURCE,executable=BINARY,headless=True)
        reference=EventWaitEnv(RESOURCE,executable=REFERENCE,headless=True)
        self.addCleanup(candidate.close);self.addCleanup(reference.close)
        candidate.require_sparse_tracks()
        self.assertEqual(candidate._sparse_tracks_version,1)
        process=candidate._process
        candidate.require_sparse_tracks()
        self.assertIs(candidate._process,process)
        with self.assertRaisesRegex(RuntimeError,'sparse_tracks_version=1'):reference.require_sparse_tracks()
        self.assertFalse(candidate._reset_done)
        self.assertFalse(reference._reset_done)


if __name__=='__main__':unittest.main()
