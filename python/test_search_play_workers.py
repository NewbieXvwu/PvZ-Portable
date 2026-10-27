"""Worker loss must fail promptly while previously yielded games stay usable."""
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import search_play


def initialize():
    pass


def play(job):
    return dict(seed=job, won=True)


def die(job):
    os._exit(9)


@unittest.skipUnless('fork' in mp.get_all_start_methods(), 'Linux fork regression')
class WorkersTest(unittest.TestCase):
    def test_completed_games_are_yielded(self):
        with patch.object(search_play, '_init_worker', initialize), patch.object(search_play, '_play', play):
            with search_play._play_parallel([1, 2, 3], 2, 'fork', ()) as results:
                rows = list(results)
        self.assertEqual(sorted(row['seed'] for row in rows), [1, 2, 3])

    def test_killed_worker_exits_instead_of_replacing_and_hanging(self):
        # Isolate the regression so a broken implementation cannot hang pytest.
        code = '''
import sys
from unittest.mock import patch
from concurrent.futures.process import BrokenProcessPool
sys.path.insert(0, 'python')
from test_search_play_workers import search_play, initialize, die
with patch.object(search_play, '_init_worker', initialize), patch.object(search_play, '_play', die):
    try:
        with search_play._play_parallel([1, 2], 2, 'fork', ()) as results:
            list(results)
    except BrokenProcessPool:
        pass
    else:
        raise AssertionError('worker loss was not reported')
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
