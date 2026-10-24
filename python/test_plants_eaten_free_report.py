"""task65：付费被吃口径与零费显示的纯 Python 单测（不启动引擎）。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python")]

from episode_query import eaten_free_note, paid_eaten
from pvz_env import EVENT_KEYS


class PaidEatenTest(unittest.TestCase):
    def test_paid_eaten_subtracts_free(self):
        self.assertEqual(paid_eaten({"plants_eaten": 10, "plants_eaten_free": 8}), 2)
        self.assertEqual(paid_eaten({"plants_eaten": 10, "plants_eaten_free": 0}), 10)

    def test_old_archive_without_free_column_keeps_old_semantics(self):
        # task65 之前的存档没有零费列：缺省 0，付费被吃 == 全部被吃，口径不变。
        self.assertEqual(paid_eaten({"plants_eaten": 5}), 5)

    def test_free_filler_plants_rank_better_than_paid_ones(self):
        # 拿零费小喷菇当垫材的局（被吃 10、其中零费 8）应排在被吃 5 全付费的前面。
        filler = {"plants_eaten": 10, "plants_eaten_free": 8}
        careful = {"plants_eaten": 5, "plants_eaten_free": 0}
        self.assertLess(paid_eaten(filler), paid_eaten(careful))

    def test_free_note_shown_only_for_new_archives(self):
        self.assertEqual(eaten_free_note({"plants_eaten": 10, "plants_eaten_free": 8}),
                         "（其中零费 8）")
        self.assertEqual(eaten_free_note({"plants_eaten": 10}), "")
        self.assertEqual(eaten_free_note({}), "")

    def test_event_keys_declare_the_new_column(self):
        self.assertIn("plants_eaten_free", set(EVENT_KEYS.values()))


if __name__ == "__main__":
    unittest.main()
