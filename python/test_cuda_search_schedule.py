"""Changing the continuation backend must preserve the requested search schedule."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import search_play


class SearchScheduleTest(unittest.TestCase):
    def test_halving_keeps_the_same_rounds_with_a_cuda_client(self):
        policy = {"type": "plant", "packet": 0, "row": 0, "col": 0}
        legal = [{"packet": packet, "row": 0, "col": 0} for packet in range(7)]
        obs = {"tick": 1000, "legal_actions": {"plants": legal}}
        args = argparse.Namespace(samples=3, horizon=2500, min_gap=0, per_card=1,
                                  margin=0.02, halving=True)

        class Root:
            def snapshot(self):
                return 1

            def branch_snapshot(self, root, chunk, retain_snapshots):
                return [{"ok": True, "observation": {"packet": a["packet"]}} for a in chunk]

            def restore(self, root):
                pass

            def release_snapshot(self, root):
                pass

        schedules, chosen = [], []
        for client in (None, object()):
            rounds = []

            def continue_jobs(env, frame, root, jobs, config, stats):
                rounds.append(jobs.copy())
                return [(action.get("packet", -1) + 1) / 10 for action, seed in jobs]

            with patch.dict(search_play._W, {"cuda": client}, clear=True), \
                 patch("render_episode._frame", side_effect=lambda frame: frame), \
                 patch.object(search_play, "_values", side_effect=lambda frames, **kw: [0.5] * len(frames)), \
                 patch.object(search_play, "_continuations", side_effect=continue_jobs), \
                 patch.object(search_play, "_leaf_values", side_effect=lambda values: values):
                stats = {"searched": 0, "overrides": 0, "override_kinds": Counter()}
                chosen.append(search_play._search(Root(), obs, policy, args, 123, stats))
            schedules.append(rounds)
        self.assertEqual(schedules[0], schedules[1])
        self.assertEqual(chosen[0], chosen[1])
        self.assertEqual(len(schedules[0]), 3, "must retain the 1, 2, 3-future halving rounds")
        self.assertLess(sum(map(len, schedules[0])), 8 * 3, "must actually prune candidates")


if __name__ == "__main__":
    unittest.main()
