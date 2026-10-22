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
from pvz_frame_store import cache_episode, read_frames, write_frames, identity_scene_scalars
from pvz_rollout_policy import load_student, packet_features
from train_rollout_policy import CandidateStore, PacketStore, ValueStore, load


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
                frame['scene'] = {'wave_count': 10 + i}
                if i == 2:
                    frame['packets'][5][1] = -1
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
                left, right = direct[i], cached[i]
                if isinstance(left, (PacketStore, CandidateStore, ValueStore)):   # column-split / sparse stores
                    rows = torch.arange(len(left.counts) if isinstance(left, (CandidateStore, ValueStore))
                                        else left.columns[0].shape[0])
                    left, right = left[rows], right[rows]
                torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
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

    def test_scene_partial_decode_matches_full_json_for_ambiguous_or_nested_input(self):
        import math
        frames = [dict(scene={'wave_count': 20}),
                  dict(scene={'wave_count': 40, 'nested': {'x': 1}}),
                  dict(scene={'wave_count': 10}, verification={'scene': {'wave_count': 99}}),
                  dict(scene={'wave_count': None}, note='中文'), dict(scene={})]
        lines = [json.dumps(f).encode() for f in frames]
        lines.append(b'{"\\u0073cene":{"wave_count":30},"verification":{"scene":{"wave_count":99}}}')
        expected = np.asarray([math.log1p(max(1, json.loads(line)['scene'].get('wave_count') or 1)) / math.log1p(50)
                               for line in lines], np.float32)
        np.testing.assert_array_equal(identity_scene_scalars(b'\n'.join(lines)), expected)

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
            report = root/'source_rows.json'
            command = [sys.executable, str(ROOT/'scripts/train_rollout_policy.py'),
                       '--data', str(root/'data'), '--labels', 'example', '--out', str(root/'reported'),
                       '--device', 'cpu', '--width', '8', '--epochs', '2', '--seed', '1', '--obs', 'v2',
                       '--pool', 'meanmax', '--row-pool', '--packet-head', 'identity',
                       '--packet-balance', '.5', '--cell-soft-weight', '3', '--spread-weight', '8',
                       '--source-row-report', str(report)]
            counted = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(counted.returncode,0,counted.stdout+counted.stderr)
            other = torch.load(root/'reported/model.pt',map_location='cpu',weights_only=True)
            for key,tensor in checkpoint['state_dict'].items():
                torch.testing.assert_close(tensor,other['state_dict'][key],rtol=0,atol=0)
            for before,after in zip(checkpoint['history'],other['history']):
                self.assertEqual({k:v for k,v in before.items() if k!='seconds'},
                                 {k:v for k,v in after.items() if k!='seconds'})
            counts=json.loads(report.read_text())
            self.assertEqual(counts['rows'],84)
            self.assertEqual(counts['sources']['example'],dict(archives=12,rows=84,training_rows=77,validation_rows=7))

    def test_empty_labelled_slot_fails_before_optimization(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = self.fixture(root / 'data')[0]
            frames = read_frames(path)
            frames[3]['action']['packet'] = 5
            write_frames(path, frames)
            for cache in (None, root / 'cache'):
                with self.assertRaisesRegex(ValueError, 'empty packet slot'):
                    load(root / 'data', {'example'}, obs='v2', packet_head='identity', cache_dir=cache)


if __name__ == '__main__':
    unittest.main()
