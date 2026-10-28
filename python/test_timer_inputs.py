"""task145：可见计时与传送带卡序进学生输入（v2timer，默认不启用）。

只测新版本本身与"旧版本一位都没动"，不跑引擎（新版本还没有 C++ 输入号）。
"""
import copy
import unittest

import numpy as np
import torch

from test_bungee_inputs import frame
from pvz_frame_store import encoder_signature
from pvz_recipe import validate
from pvz_rollout_policy import RolloutNet, export_student, warm_start_student
from pvz_student_observation import OBS_EXTRA, StudentHistory
from pvz_value_net import encode_frame


def timer_frame():
    """一帧构造观测：毁灭菇引爆中、大嘴花咀嚼、窝瓜起跳前、玉米炮装填、土豆雷、雾里一格。"""
    current = frame()
    current['v2'] = dict(current['v2'], plants=[
        [0, 0, 15, 2, 0, 40, 0, 0],        # 毁灭菇：special 40 → 0.60
        [0, 1, 6, 13, 3000, 0, 0, 0],      # 大嘴花咀嚼：1 - 3000/4000 = 0.25
        [0, 2, 17, 4, 20, 0, 0, 0],        # 窝瓜起跳前：1 - 20/45
        [1, 0, 47, 35, 250, 0, 0, 0],      # 玉米炮装填：1 - 250/500 = 0.50
        [1, 1, 4, 0, 750, 0, 0, 0],        # 土豆雷：武装走基础通道 6，不进新通道
        [2, 0, 15, -1, -1, -1, -1, 0],     # 雾里被遮住：state / countdown 都是 -1
    ])
    current['packets'] = [[0, 15, True, 0, 0, 125],     # 第 0 位：毁灭菇，就绪
                          [1, -1, False, 0, 0, 0],      # 第 1 位：空槽
                          [2, 2, True, 30, 5000, 150]]  # 第 2 位：樱桃，还在冷却
    return current


class TimerInputsTest(unittest.TestCase):
    def test_new_version_is_v2sleep_plus_four_channels(self):
        current = timer_frame()
        board, scalars = StudentHistory('v2timer').features(current, None)
        base, base_scalars = StudentHistory('v2sleep').features(current, None)
        self.assertEqual(OBS_EXTRA['v2timer'], (25, 9))
        self.assertEqual(tuple(board.shape), (117, 6, 10))
        torch.testing.assert_close(board[:-4], base, rtol=0, atol=0)
        torch.testing.assert_close(scalars, base_scalars, rtol=0, atol=0)

    def test_special_and_state_progress_channels(self):
        board, _ = StudentHistory('v2timer').features(timer_frame(), None)
        extra = board[-4:]
        self.assertAlmostEqual(float(extra[0, 0, 0]), 0.60, places=6)       # 毁灭菇引爆
        self.assertAlmostEqual(float(extra[1, 0, 1]), 0.25, places=6)       # 大嘴花咀嚼
        self.assertAlmostEqual(float(extra[1, 0, 2]), 1 - 20 / 45, places=6)  # 窝瓜起跳前
        self.assertAlmostEqual(float(extra[1, 1, 0]), 0.50, places=6)       # 玉米炮装填
        self.assertEqual(float(extra[1, 1, 1]), 0.0)                        # 土豆雷不进这里
        self.assertEqual(float(extra[0, 1, 1]), 0.0)

    def test_fog_masked_plant_stays_zero(self):
        board, _ = StudentHistory('v2timer').features(timer_frame(), None)
        self.assertEqual(float(board[-4:, 2, 0].abs().sum()), 0.0)

    def test_belt_order_and_readiness_channels(self):
        board, _ = StudentHistory('v2timer').features(timer_frame(), None)
        belt_type, belt_ready = board[-2], board[-1]
        self.assertAlmostEqual(float(belt_type[0, 0]), 16 / 48, places=6)   # 毁灭菇 = 15
        self.assertEqual(float(belt_type[0, 1]), 0.0)                       # 空槽
        self.assertAlmostEqual(float(belt_type[0, 2]), 3 / 48, places=6)    # 樱桃 = 2
        self.assertEqual(belt_type[0, 3:].abs().sum(), 0.0)
        self.assertEqual([float(value) for value in belt_ready[0, :3]], [1.0, 0.0, 0.0])
        # 卡序是屏幕上的位置，不是棋盘行：其它行必须全零。
        self.assertEqual(float(belt_type[1:].abs().sum()), 0.0)

    def test_belt_shift_moves_the_same_card_to_a_new_slot(self):
        before = timer_frame()
        after = copy.deepcopy(before)
        after['packets'] = [[0, 2, True, 0, 5000, 150], [1, 15, True, 0, 0, 125]]
        first, _ = StudentHistory('v2timer').features(before, None)
        second, _ = StudentHistory('v2timer').features(after, None)
        self.assertAlmostEqual(float(first[-2, 0, 0]), 16 / 48, places=6)
        self.assertAlmostEqual(float(second[-2, 0, 1]), 16 / 48, places=6)
        self.assertEqual(float(second[-2, 0, 0]), float(2 + 1) / 48)

    def test_value_encoder_five_extends_four(self):
        current = timer_frame()
        five_board, five_scalars = encode_frame(current, 5)
        four_board, four_scalars = encode_frame(current, 4)
        self.assertEqual(tuple(five_board.shape), (104, 6, 10))
        torch.testing.assert_close(five_board[:-4], four_board, rtol=0, atol=0)
        torch.testing.assert_close(five_scalars, four_scalars, rtol=0, atol=0)
        student, _ = StudentHistory('v2timer').features(current, None)
        torch.testing.assert_close(five_board[-4:], student[-4:], rtol=0, atol=0)

    def test_old_versions_keep_their_signatures(self):
        # v2sleep 在 task145 之前就存在；新增 v2timer 不能改它的缓存签名。
        self.assertEqual(encoder_signature('v2sleep'),
                         '309266015e12322c291071ddbf1c2e952fb1e7bc2f2248f4cac1ca9b38f41029')
        self.assertEqual(encoder_signature('v2'), '5031d93614db9ecff271dcaa6aa8645d7a799452f170e5ebf699cc2fb8c1df91')
        self.assertNotEqual(encoder_signature('v2timer'), encoder_signature('v2sleep'))

    def test_recipe_accepts_the_new_version(self):
        recipe = validate(dict(schema_version=1, kind='training', data='/tmp/data',
                               labels=['example'], out='/tmp/out', obs='v2timer'))
        self.assertEqual(recipe['obs'], 'v2timer')
        with self.assertRaises(ValueError):
            validate(dict(schema_version=1, kind='training', data='/tmp/data',
                          labels=['example'], out='/tmp/out', obs='v2timer2'))

    def test_warm_start_from_v2sleep_keeps_the_parent_function(self):
        torch.set_num_threads(1)
        torch.manual_seed(145)
        old = RolloutNet(8, 'meanmax', False, 'v2sleep', row_pool=True, aux=True,
                         packet_head='identity').eval()
        new = RolloutNet(8, 'meanmax', False, 'v2timer', row_pool=True, aux=True,
                         packet_head='identity').eval()
        warm_start_student(new, old)
        self.assertFalse(new.trunk[0].weight[:, old.trunk[0].in_channels:].any())
        board = torch.randn(2, old.trunk[0].in_channels, 6, 10)
        scalars = torch.randn(2, old.scalar[0].in_features)
        wider = torch.zeros(2, new.trunk[0].in_channels, 6, 10)
        wider[:, :board.shape[1]] = board
        info = torch.zeros(2, 10, 4)
        packet = torch.tensor([1, 3])
        with torch.inference_mode():
            expected = old(board, scalars, packet, auxiliary=True, packet_info=info)
            actual = new(wider, scalars, packet, auxiliary=True, packet_info=info)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-6, atol=1e-6)

    def test_native_export_refuses_until_the_cpp_channel_exists(self):
        net = RolloutNet(8, obs='v2timer').eval()
        with self.assertRaises(ValueError):
            export_student(net, '/tmp/task145-should-not-exist.bin')


if __name__ == '__main__':
    unittest.main()
