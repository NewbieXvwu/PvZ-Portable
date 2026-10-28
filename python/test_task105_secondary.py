"""Third-slot scans preserve the existing queue and share metadata safely."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import task105_secondary as worker
from task105_state import metadata_lock


class SecondaryTest(unittest.TestCase):
    def test_metadata_updates_from_both_workers_are_serialized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'counter.json'
            path.write_text('0')
            barrier = threading.Barrier(2)

            def update():
                barrier.wait(timeout=1)
                for _ in range(3):
                    with metadata_lock(root):
                        count = json.loads(path.read_text())
                        time.sleep(.01)
                        path.write_text(json.dumps(count + 1))

            with ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(update) for _ in range(2)]
                for future in futures:
                    future.result(timeout=2)
            self.assertEqual(json.loads(path.read_text()), 6)

    def test_reverse_worker_uses_one_game_and_never_replays_finished_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'artifacts/task105'
            results = out / 'scan/results'
            results.mkdir(parents=True)
            (root / 'configs').mkdir()
            (root / 'configs/human_demo_tasks.json').write_text(json.dumps({'tasks': []}))
            entries = [dict(task_id=name, seed=1, archive=f'sha/{name}', future_seeds=list(range(8)))
                       for name in ('old', 'already_done', 'new')]
            (out / 'scan_jobs.json').write_text(json.dumps({'archives': entries}))

            def finish(entry):
                (results / f'{entry["task_id"]}_s1.json').write_text(json.dumps(dict(
                    complete=True, archive=entry['archive'], future_seeds=entry['future_seeds'], ladder=None)))

            finish(entries[1])
            played = []

            def run(command, **kwargs):
                if 'adaptive' in command:
                    self.assertEqual(command[command.index('--workers') + 1], '1')
                    self.assertIn('--skip-complete', command)
                    task = command[command.index('--task-id') + 1]
                    played.append(task)
                    finish(next(e for e in entries if e['task_id'] == task))
                return SimpleNamespace(returncode=0)

            with patch.object(worker, 'ROOT', root), patch.object(worker, 'OUT', out), \
                 patch.object(worker.subprocess, 'run', side_effect=run) as commands:
                worker.main()
                self.assertEqual(played, ['new', 'old'])
                self.assertEqual(commands.call_count, 6)
                worker.main()
                self.assertEqual(commands.call_count, 6)
            self.assertEqual(json.loads((out / 'secondary_status.json').read_text())['state'], 'complete')


if __name__ == '__main__':
    unittest.main()
