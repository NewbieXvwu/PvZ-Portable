import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from test_bungee_inputs import BINARY, RESOURCE, frame
from export_value_net import export_value_net
from pack_training_rows import packed_load
from pvz_env import PvZEnv, TaskSpec
from pvz_tasks import profile_for_deck
from pvz_frame_store import encoder_signature, write_frames
from pvz_recipe import training_row_options, validate
from pvz_rollout_policy import RolloutNet, export_student, features, packet_features
from pvz_student_observation import OBS_EXTRA, StudentHistory
from pvz_value_net import ENCODER_V4, ValueNet, encode_frame, load_value_net
from render_episode import _frame


class SleepInputsTest(unittest.TestCase):
    def test_channels_and_frozen_versions(self):
        current = frame()
        current['sleep'] = [[0, 1, 1, 0], [1, 2, 1, 40], [1, 2, 0, 0],
                            [2, 3, 0, 0], [3, 4, -1, -1], [9, 1, 1, 50]]
        old = {key: value for key, value in current.items() if key != 'sleep'}
        for version in OBS_EXTRA:
            if version in ('v2sleep', 'v2timer'):
                continue
            before = features(old, None) if version == 'v1' else StudentHistory(version).features(old, None)
            after = features(current, None) if version == 'v1' else StudentHistory(version).features(current, None)
            for left, right in zip(before, after):
                np.testing.assert_array_equal(left.numpy().view(np.uint32), right.numpy().view(np.uint32))
        for version in (1, 2, 3):
            for left, right in zip(encode_frame(old, version), encode_frame(current, version)):
                np.testing.assert_array_equal(left.numpy().view(np.uint32), right.numpy().view(np.uint32))
        board, scalars = StudentHistory('v2sleep').features(current, None)
        base, base_scalars = StudentHistory('v2bungee').features(current, None)
        self.assertEqual(tuple(board.shape), (113, 6, 10))
        torch.testing.assert_close(board[:-2], base, rtol=0, atol=0)
        torch.testing.assert_close(scalars, base_scalars, rtol=0, atol=0)
        self.assertEqual(board[-2, 0, 1], 1)
        self.assertEqual(board[-1, 0, 1], 0)
        self.assertEqual(board[-2:, 1, 2].tolist(), [1, 1])
        self.assertEqual(board[-2:].sum(), 3)
        value, value_scalars = encode_frame(current, 4)
        self.assertEqual(tuple(value.shape), (100, 6, 10))
        self.assertEqual(len(value_scalars), 119)
        torch.testing.assert_close(value[-2:], board[-2:], rtol=0, atol=0)
        self.assertFalse(encode_frame(old, 4)[0][-2:].any())
        self.assertEqual(encoder_signature('v2bungee'), '82609c46ed14655cead5910457ee9c0a627d77948b896c8afef09b012a2bdb6f')

    def test_recipe_pack_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = frame()
            first['sleep'] = [[0, 1, 1, 10]]
            second = copy.deepcopy(first)
            second.update(tick=160, action=dict(type='wait', ticks=60))
            write_frames(root / 'data/seed_30000/example/frames.npz', [first, second])
            recipe = validate(dict(schema_version=1, kind='training', data=str(root / 'data'),
                                   labels=['example'], out=str(root / 'out'), obs='v2sleep'))
            options = training_row_options(recipe)
            options.pop('source_weight', None)
            boards, scalars, *_ = packed_load(root / 'data', cache_dir=root / 'cache', rebuild=True, **options)
            expected_board, expected_scalars = StudentHistory('v2sleep').features(first, None)
            torch.testing.assert_close(boards.batch(torch.tensor([0]), 'cpu')[0], expected_board.half().float(), rtol=0, atol=0)
            torch.testing.assert_close(scalars.batch(torch.tensor([0]), 'cpu')[0], expected_scalars, rtol=0, atol=0)
            for head in ('slot', 'identity'):
                net = RolloutNet(8, obs='v2sleep', packet_head=head).eval()
                self.assertEqual(net(expected_board[None], expected_scalars[None],
                                     packet_info=packet_features(first)[None])['plant'].shape, (1,))
            value = ValueNet(8, encoder_version=4)
            checkpoint = root / 'value.pt'
            torch.save(dict(state_dict=value.state_dict(), width=8, encoder=ENCODER_V4), checkpoint)
            self.assertEqual(load_value_net(checkpoint).encoder_version, 4)


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / 'main.pak').is_file(), 'requires simulator and resources')
class SleepEngineTest(unittest.TestCase):
    def assert_native_features(self, env, net, root):
        public = _frame(env.observe())
        env.set_lean_observation(True)
        self.assertEqual(_frame(env.observe()), public)
        env.set_lean_observation(False)
        board, scalars = StudentHistory('v2sleep').features(public, None)
        native = env.rollout_predict(student_history=StudentHistory('v2sleep'), trace_path=root / 'student.json')
        encoded = json.loads((root / 'student.json').read_text())
        for expected, actual in ((board.flatten(), encoded['board']), (scalars, encoded['scalars'])):
            np.testing.assert_array_equal(expected.numpy().view(np.uint32), np.asarray(actual, np.float32).view(np.uint32))
        value_board, value_scalars = encode_frame(public, 4)
        probability = env.leaf_values(trace_path=root / 'features.bin')[0]
        np.testing.assert_array_equal(np.fromfile(root / 'features.bin', dtype='<f4').view(np.uint32),
            np.concatenate([value_board.flatten().numpy(), value_scalars.numpy()]).view(np.uint32))
        with torch.inference_mode():
            _, context = net.encode(board[None], scalars[None], packet_info=packet_features(public)[None])
            self.assertAlmostEqual(float(net.plant(context)[0, 0]), native['plant'], places=6)
        return public

    def test_day_mushroom_coffee_native_parity(self):
        torch.set_num_threads(1)
        torch.manual_seed(130)
        deck = (0, 1, 2, 3, 8, 35)
        with tempfile.TemporaryDirectory() as folder, PvZEnv(RESOURCE, executable=BINARY) as env:
            root = Path(folder)
            net = RolloutNet(8, obs='v2sleep', packet_head='identity').eval()
            export_student(net, root / 'student.bin')
            env.load_rollout_net(root / 'student.bin')
            value = ValueNet(8, encoder_version=4).eval()
            export_value_net(value, root / 'value.bin')
            env.load_value_net(root / 'value.bin')
            env.reset(deck=deck, task=TaskSpec(level=7, seed=30000, profile=profile_for_deck(deck)))
            env.step(dict(type='plant', packet=4, row=0, col=2))
            sleeping = self.assert_native_features(env, net, root)
            self.assertIn([0, 2, 1, 0], sleeping['sleep'])
            while env.observe()['sun'] < 75:
                self.assertLess(env.observe()['tick'], 3000)
                env.step(dict(type='wait', ticks=40))
            env.step(dict(type='plant', packet=5, row=0, col=2))
            waking = None
            for _ in range(30):
                public = _frame(env.observe())
                if any(row == 0 and col == 2 and counter > 0 for row, col, asleep, counter in public['sleep']):
                    waking = self.assert_native_features(env, net, root)
                    break
                env.step(dict(type='wait', ticks=10))
            self.assertIsNotNone(waking)
            env.step(dict(type='wait', ticks=150))
            awake = self.assert_native_features(env, net, root)
            self.assertIn([0, 2, 0, 0], awake['sleep'])
            value_board, value_scalars = encode_frame(awake, 4)
            with torch.inference_mode():
                expected = float(torch.sigmoid(value(value_board[None], value_scalars[None]))[0])
            self.assertAlmostEqual(env.leaf_values()[0], expected, places=6)
            print(json.dumps(dict(level=7, samples=3, student_float32_bit_equal=True,
                                  value_float32_bit_equal=True)), flush=True)

    def test_fog_masks_sleep_fields(self):
        deck = (0, 1, 2, 3, 8, 16)
        with tempfile.TemporaryDirectory() as folder, PvZEnv(RESOURCE, executable=BINARY) as env:
            root = Path(folder)
            net = RolloutNet(8, obs='v2sleep').eval()
            export_student(net, root / 'student.bin')
            env.load_rollout_net(root / 'student.bin')
            export_value_net(ValueNet(8, encoder_version=4), root / 'value.bin')
            env.load_value_net(root / 'value.bin')
            env.reset(deck=deck, task=TaskSpec(level=31, seed=30000, profile=profile_for_deck(deck),
                fog_observation='human_v2', preplanted=((8, 0, 2), (8, 1, 7))))
            env.step(dict(type='wait', ticks=2500))
            observation = env.observe()
            hidden = [plant for plant in observation['plants'] if plant['health'] < 0]
            self.assertTrue(hidden)
            for plant in hidden:
                self.assertFalse(plant['asleep'])
                self.assertEqual(plant['wake_up_counter'], -1)
            public = _frame(observation)
            env.set_lean_observation(True)
            self.assertEqual(_frame(env.observe()), public)
            env.set_lean_observation(False)
            board, scalars = StudentHistory('v2sleep').features(public, None)
            env.rollout_predict(student_history=StudentHistory('v2sleep'), trace_path=root / 'student.json')
            encoded = json.loads((root / 'student.json').read_text())
            np.testing.assert_array_equal(board[-2:].numpy().flatten().view(np.uint32),
                np.asarray(encoded['board'], np.float32)[-120:].view(np.uint32))
            env.leaf_values(trace_path=root / 'features.bin')
            native_board = np.fromfile(root / 'features.bin', dtype='<f4')[:6000].reshape(100, 6, 10)
            np.testing.assert_array_equal(encode_frame(public, 4)[0][-2:].numpy().view(np.uint32),
                                          native_board[-2:].view(np.uint32))


if __name__ == '__main__':
    unittest.main()
