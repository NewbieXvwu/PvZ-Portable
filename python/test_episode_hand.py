"""episode_query 按当时手牌换算卡槽 ↔ 植物（传送带关 deck 为空、换卡组时槽号不对应同一种植物）。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from episode_query import _action_cli, _action_text, _hand_at, _parse_action  # noqa: E402


def _frame(hand):
    # [index, type, active, cooldown, refresh_time, cost]，与 render_episode._frame 一致
    return {"packets": [[slot, kind, True, 0, 0, 0] for slot, kind in enumerate(hand)]}


class HandAtTest(unittest.TestCase):
    def setUp(self):
        # 传送带：第 1 步手里 [樱桃 2, 豌豆 0]，第 2 步变成 [豌豆 0, 坚果 3]
        self.frames = [_frame([2, 0]), _frame([0, 3]), _frame([3])]

    def test_hand_comes_from_previous_frame(self):
        self.assertEqual(_hand_at(self.frames, 1, []), [2, 0])
        self.assertEqual(_hand_at(self.frames, 2, []), [0, 3])

    def test_slot_text_uses_the_step_hand_not_the_deck(self):
        act = {"type": "plant", "packet": 0, "row": 2, "col": 1}
        self.assertIn("豌豆射手", _action_text(act, _hand_at(self.frames, 2, [])))
        self.assertIn("樱桃炸弹", _action_text(act, _hand_at(self.frames, 1, [])))

    def test_parse_and_cli_round_trip_on_shifted_slots(self):
        hand = _hand_at(self.frames, 2, [])
        act = _parse_action("plant:3:1:4", hand)
        self.assertEqual(act["packet"], 1)
        self.assertEqual(_action_cli(act, hand), "plant:3:1:4")
        with self.assertRaises(ValueError):
            _parse_action("plant:2:1:4", hand)   # 这一步手里已经没有樱桃

    def test_old_archive_without_packets_falls_back_to_deck(self):
        frames = [{}, {}]
        self.assertEqual(_hand_at(frames, 1, [0, 1, 7]), [0, 1, 7])


if __name__ == "__main__":
    unittest.main()
