"""Archive replays preserve the saved hand and task semantics independently of current deck rules."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_value_board_replay import archived_task


class ValueBoardReplayTaskTest(unittest.TestCase):
    def test_old_archive_uses_saved_hand_and_upgrade_ownership(self):
        deck = [0, 1, 2, 3, 4, 16, 43]
        meta = dict(task=dict(seed=37100), context=dict(level=21, multiplier=1.5))
        actual, task = archived_task(meta, dict(packets=[[i, kind] for i, kind in enumerate(deck)]))
        self.assertEqual(actual, tuple(deck))
        self.assertEqual(task.profile.seed_slot_count, 7)
        self.assertEqual(task.profile.owned_upgrade_plants, (43,))
        self.assertEqual((task.level, task.seed, task.zombie_count_multiplier), (21, 37100, 1.5))

    def test_original_conveyor_preserves_empty_deck_after_cards_arrive(self):
        meta = dict(task=dict(seed=37110, conveyor_start="original", deck=[]),
                    context=dict(level=10, multiplier=1.0))
        deck, task = archived_task(meta, dict(packets=[[0, 2], [1, 4]]))
        self.assertEqual(deck, ())
        self.assertEqual(task.conveyor_start, "original")
        self.assertEqual(task.profile.seed_slot_count, 6)

    def test_explicit_task_deck_and_replay_fields_are_preserved(self):
        saved = [0, 1, 2, 3, 4, 5]
        meta = dict(task=dict(seed=37100, deck=saved, first_aid_owned=True,
                              wave_cap=5, preplanted=[[1, 0, 0]]),
                    context=dict(level=7, multiplier=1.3))
        deck, task = archived_task(meta, dict(packets=[[0, 7]]))
        self.assertEqual(deck, tuple(saved))
        self.assertTrue(task.profile.first_aid_owned)
        self.assertEqual(task.wave_cap, 5)
        self.assertEqual(task.preplanted, ((1, 0, 0),))


if __name__ == "__main__":
    unittest.main()
