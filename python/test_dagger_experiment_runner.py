"""Actual subprocess checks for the study's process-tree memory pause gate."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import dagger_experiment_runner as runner


class ExperimentRunnerTest(unittest.TestCase):
    def test_low_memory_pauses_and_resumes_only_own_parent_and_child(self):
        calls = []
        kill = os.kill

        def observed_kill(pid, sig):
            calls.append((pid, sig))
            return kill(pid, sig)

        readings = iter([4, 4, 4, 1.5, 1.5, 4])
        available = lambda: int(next(readings, 4) * 1024 * 1024)
        code = ('import json,os,subprocess,sys,time; '
                'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(3)"]); '
                'print(json.dumps({"parent":os.getpid(),"child":p.pid}),flush=True); '
                'time.sleep(4); p.wait()')
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with patch.object(runner, 'OUT', out), patch.object(runner, 'available', available), \
                    patch.object(runner.os, 'kill', observed_kill):
                result = runner.run([sys.executable, '-c', code], 'tree')
            self.assertEqual(result['returncode'], 0)
            identity = json.loads((out / 'tree.txt').read_text())
            own = {identity['parent'], identity['child']}
            self.assertEqual({p for p, s in calls if s == signal.SIGSTOP}, own)
            self.assertEqual({p for p, s in calls if s == signal.SIGCONT}, own)
            self.assertNotIn(os.getpid(), {p for p, _ in calls})
            samples = [json.loads(v) for v in (out / 'tree.resources.jsonl').read_text().splitlines()]
            self.assertTrue(any(v['paused'] for v in samples))
            self.assertFalse(samples[-1]['paused'])
            self.assertEqual(json.loads((out / 'tree.result.json').read_text()), result)

    def test_child_failure_is_preserved_in_result(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(runner, 'OUT', Path(temp)), patch.object(runner, 'available', lambda: 4 * 1024**2):
                result = runner.run([sys.executable, '-c', 'raise SystemExit(7)'], 'failure')
            self.assertEqual(result['returncode'], 7)


if __name__ == '__main__':
    unittest.main()
