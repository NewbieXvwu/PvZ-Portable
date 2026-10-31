import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from export_value_net import export_value_net
from pack_training_rows import packed_load
from pvz_env import PvZEnv, TaskSpec
from pvz_tasks import profile_for_deck
from pvz_frame_store import write_frames, encoder_signature
from pvz_recipe import validate, training_row_options
from pvz_rollout_policy import RolloutNet, export_student, features
from pvz_student_observation import OBS_EXTRA, StudentHistory
from pvz_value_net import ValueNet, ENCODER_V3, encode_frame, load_value_net
from render_episode import _frame

RESOURCE = Path(os.environ.get('PVZ_RESOURCE_DIR', str(Path.home() / '.cache/pvz-research-resources')))
BINARY = Path(os.environ.get('PVZ_EXECUTABLE', ROOT / 'build/pvz-portable'))


def frame():
    return dict(tick=100, wave=1, sun=0, scene=dict(wave_count=20, roof=True),
                plants=[], zombies=[], packets=[], mowers=[], lanes=list(range(5)),
                v2=dict(wave_timer=0, huge_wave_countdown=0, wave_health=0, wave_start_health=0,
                        sun_countdown=0, sun_pending=0, plants=[], zombies=[], projectiles=[]),
                bungee=dict(zombies=[[0, 4, 1, -1, 0], [1, 5, 0, 0, 300],
                                     [2, 6, 1, 1, 0], [3, 7, 0, 2, 150]], plants=[[3, 7]]))


class BungeeInputsTest(unittest.TestCase):
    def test_visible_channels_and_frozen_versions(self):
        current = frame()
        old = {key: value for key, value in current.items() if key != 'bungee'}
        for version in OBS_EXTRA:
            if version in ('v2bungee', 'v2sleep', 'v2timer'):
                continue
            encode = features if version == 'v1' else StudentHistory(version).features
            left = encode(old, None)
            encode = features if version == 'v1' else StudentHistory(version).features
            for before, after in zip(left, encode(current, None)):
                np.testing.assert_array_equal(before.numpy().view(np.uint32), after.numpy().view(np.uint32))
        for version in (1, 2):
            for before, after in zip(encode_frame(old, version), encode_frame(current, version)):
                np.testing.assert_array_equal(before.numpy().view(np.uint32), after.numpy().view(np.uint32))
        board, scalars = StudentHistory('v2bungee').features(current, None)
        base, base_scalars = StudentHistory('v2boss').features(current, None)
        torch.testing.assert_close(board[:-6], base, rtol=0, atol=0)
        torch.testing.assert_close(scalars, base_scalars, rtol=0, atol=0)
        extra = board[-6:]
        self.assertEqual(tuple(board.shape), (111, 6, 10))
        self.assertEqual(extra[0, 0, 4], 1)
        self.assertEqual(extra[1:, 0, 4].sum(), 0)
        self.assertEqual(extra[1, 1, 5], .5)
        self.assertEqual(extra[2, 1, 5], 1)
        self.assertEqual(extra[3, 2, 6], 1)
        self.assertEqual(extra[4, 3, 7], 1)
        self.assertEqual(extra[5, 3, 7], 1)
        value, _ = encode_frame(current, 3)
        torch.testing.assert_close(value[-6:], extra, rtol=0, atol=0)
        self.assertFalse(encode_frame(old, 3)[0][-6:].any())
        self.assertEqual(encoder_signature('v2boss'), 'b5d77ca232300a2d7331ae572ace07ef1c58757f3ef5e2818c3fb5a691df3bc4')

    def test_recipe_packs_new_observation_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / 'data/seed_30000/example/frames.npz'
            first = frame()
            second = copy.deepcopy(first)
            second.update(tick=160, action=dict(type='wait', ticks=60))
            write_frames(archive, [first, second])
            recipe = validate(dict(schema_version=1, kind='training', data=str(root / 'data'),
                                   labels=['example'], out=str(root / 'out'), obs='v2bungee'))
            options = training_row_options(recipe)
            options.pop('source_weight', None)
            loaded = packed_load(root / 'data', cache_dir=root / 'cache', rebuild=True, **options)
            boards, scalars = loaded[:2]
            expected_board, expected_scalars = StudentHistory('v2bungee').features(first, None)
            torch.testing.assert_close(boards.batch(torch.tensor([0]), 'cpu')[0], expected_board.half().float(), rtol=0, atol=0)
            torch.testing.assert_close(scalars.batch(torch.tensor([0]), 'cpu')[0], expected_scalars, rtol=0, atol=0)
            net = RolloutNet(8, obs='v2bungee').eval()
            self.assertEqual(net(expected_board[None], expected_scalars[None])['plant'].shape, (1,))
            value = ValueNet(8, encoder_version=3)
            checkpoint = root / 'value.pt'
            torch.save(dict(state_dict=value.state_dict(), width=8, encoder=ENCODER_V3), checkpoint)
            self.assertEqual(load_value_net(checkpoint).encoder_version, 3)


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / 'main.pak').is_file(), 'requires simulator and resources')
class BungeeEngineTest(unittest.TestCase):
    def test_l45_public_features_and_native_parity(self):
        torch.set_num_threads(1)
        torch.manual_seed(116)
        with tempfile.TemporaryDirectory() as folder, PvZEnv(RESOURCE, executable=BINARY) as env:
            root = Path(folder)
            net = RolloutNet(8, obs='v2bungee').eval()
            export_student(net, root / 'student.bin')
            env.load_rollout_net(root / 'student.bin')
            value = ValueNet(8, encoder_version=3).eval()
            export_value_net(value, root / 'value.bin')
            env.load_value_net(root / 'value.bin')
            plants = tuple((33, row, col) for row in range(5) for col in range(3, 9))
            plants += tuple((3, row, col) for row in range(5) for col in range(3, 9))
            plants += tuple((39, row, col) for row in range(5) for col in range(3))
            observation, _ = env.reset(deck=(), task=TaskSpec(level=45, seed=30000,
                profile=profile_for_deck(()), conveyor_start='original', preplanted=plants))
            seen = np.zeros(6, dtype=bool)
            samples = 0
            maximum_error = 0.0
            while observation['tick'] < 30000 and not observation['terminal']:
                public = _frame(observation)
                extra = encode_frame(public, 3)[0][-6:]
                active = extra.flatten(1).any(1).numpy()
                if (active & ~seen).any():
                    env.set_lean_observation(True)
                    self.assertEqual(_frame(env.observe()), public)
                    env.set_lean_observation(False)
                    history = StudentHistory('v2bungee')
                    board, scalars = history.features(public, None)
                    native = env.rollout_predict(student_history=StudentHistory('v2bungee'), trace_path=root / 'student.json')
                    encoded = json.loads((root / 'student.json').read_text())
                    for expected, actual in ((board.flatten(), encoded['board']), (scalars, encoded['scalars'])):
                        np.testing.assert_array_equal(expected.numpy().view(np.uint32), np.asarray(actual, np.float32).view(np.uint32))
                    with torch.inference_mode():
                        hidden, context = net.encode(board[None], scalars[None])
                        maximum_error = max(maximum_error, abs(float(net.plant(context)[0, 0]) - native['plant']))
                    value_board, value_scalars = encode_frame(public, 3)
                    probability = env.leaf_values(trace_path=root / 'value.bin')[0]
                    np.testing.assert_array_equal(np.fromfile(root / 'value.bin', dtype='<f4'),
                        np.concatenate([value_board.flatten().numpy(), value_scalars.numpy()]))
                    self.assertAlmostEqual(env.leaf_feature_values(root / 'value.bin', 1)[0], probability, places=6)
                    with torch.inference_mode():
                        maximum_error = max(maximum_error,
                            abs(float(torch.sigmoid(value(value_board[None], value_scalars[None]))[0]) - probability))
                    samples += 1
                    seen |= active
                if seen.all():
                    break
                observation, *_ = env.step(dict(type='wait', ticks=40))
            self.assertTrue(seen.all(), (seen.tolist(), observation['tick']))
            self.assertLess(maximum_error, 1e-5)
            print(json.dumps(dict(level=45, samples=samples, channels=seen.tolist(), max_abs_error=maximum_error)), flush=True)


if __name__ == '__main__':
    unittest.main()
