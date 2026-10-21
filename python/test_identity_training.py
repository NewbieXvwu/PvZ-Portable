import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from pvz_frame_store import cache_episode, read_frames, write_frames
from pvz_rollout_policy import load_student, packet_features
from train_rollout_policy import load


class IdentityTrainingTest(unittest.TestCase):
    def fixture(self, root):
        for seed in range(10, 22):
            folder = root / f'seed_{seed}' / 'example'
            folder.mkdir(parents=True)
            frames = []
            for i in range(9):
                order = np.roll(np.arange(6), i)
                frame = dict(sun=1000, wave=0, tick=i * 60, scene={'wave_count': 10},
                             packets=[[p, int(kind), True, 0, 100, 25] for p, kind in enumerate(order)],
                             plants=[], zombies=[], mowers=[[r, 1] for r in range(5)], lanes=list(range(5)))
                if i:
                    frame['action'] = ({'type': 'plant', 'packet': 0, 'row': 0, 'col': 0}
                                       if i % 2 else {'type': 'wait', 'ticks': 60})
                    frame['action_rejected'] = i == 4
                    frame['targets'] = [[0, 0, 0, .8], [0, 1, 0, .5], [-1, -1, -1, .3]]
                frames.append(frame)
            write_frames(folder / 'frames.npz', frames)
        return sorted(root.glob('*/*/frames.npz'))

    def test_cached_uncached_auxiliary_and_packet_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = self.fixture(root / 'data')
            kwargs = dict(wait_keep=.5, obs='v2', packet_head='identity', cells=True, aux_horizon=2)
            direct = load(root / 'data', {'example'}, **kwargs)
            cached = load(root / 'data', {'example'}, cache_dir=root / 'cache', **kwargs)
            self.assertEqual(len(direct), 11)
            for i in list(range(2, 6)) + list(range(7, 11)):
                torch.testing.assert_close(direct[i], cached[i], rtol=0, atol=0, equal_nan=True)
            self.assertEqual(direct[6], cached[6])
            for i in (0, 1):
                torch.testing.assert_close(direct[i].batch(np.arange(direct[0].shape[0]), 'cpu'),
                                           cached[i].batch(np.arange(cached[0].shape[0]), 'cpu'), rtol=0, atol=0)
            frames = read_frames(paths[0])
            torch.testing.assert_close(direct[-1][0], packet_features(frames[0]), rtol=0, atol=0)
            # The first row is labelled at frame1 but reads the deck at frame0, not frame1.
            self.assertNotEqual(int(direct[-1][0, 0, 0]), frames[1]['packets'][0][1])

    def test_read_cache_is_never_modified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = self.fixture(root / 'data')
            for path in paths:
                cache_episode(path, root / 'readonly', 'v2')
            before = {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                      for p in (root / 'readonly').rglob('*') if p.is_file()}
            load(root / 'data', {'example'}, cache_dir=root / 'private', read_cache_dir=root / 'readonly',
                 obs='v2', packet_head='identity')
            after = {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                     for p in (root / 'readonly').rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(len(list((root / 'private' / 'packets').glob('*.npz'))), 12)

    def test_cpu_training_checkpoint_is_loadable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root / 'data')
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/train_rollout_policy.py'),
                                     '--data', str(root / 'data'), '--labels', 'example', '--out', str(root / 'model'),
                                     '--device', 'cpu', '--width', '8', '--epochs', '2', '--seed', '1', '--obs', 'v2',
                                     '--pool', 'meanmax', '--row-pool', '--packet-head', 'identity',
                                     '--packet-balance', '.5', '--cell-soft-weight', '3', '--spread-weight', '8'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            model = load_student(root / 'model/model.pt')
            self.assertEqual(model.packet_head, 'identity')
            self.assertTrue(model.row_pool)
            checkpoint = torch.load(root / 'model/model.pt', map_location='cpu', weights_only=True)
            self.assertTrue(all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values()))
            self.assertEqual(len(checkpoint['history']), 2)


if __name__ == '__main__':
    unittest.main()
