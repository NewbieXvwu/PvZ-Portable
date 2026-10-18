"""Speculative execution must retain the original halving labels, including pruned candidates."""
from collections import Counter
import argparse
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import search_play


class SpeculativeHalvingTest(unittest.TestCase):
    def compare(self, count, samples, policy_index, pattern):
        candidates = [{"type": "plant", "packet": i, "row": 0, "col": 0}
                      for i in range(count - 1)] + [search_play.NOOP]
        keys = [json.dumps(c, sort_keys=True) for c in candidates]
        rng = random.Random(count * 100 + samples)
        table = {(i, k): (None if pattern == "missing" and (i + k) % 3 == 0 else
                          0.5 if pattern == "ties" else rng.random())
                 for i in range(count) for k in range(samples)}
        results = []
        for speculate in (False, True):
            calls, batches = [], []
            args = argparse.Namespace(samples=samples, margin=0.02, cuda_speculative_halving=speculate)
            stats = {"overrides": 0, "override_kinds": Counter()}

            def continue_jobs(env, obs, root, jobs, config, counters):
                pairs = [(keys.index(json.dumps(c, sort_keys=True)), seed - 123000) for c, seed in jobs]
                calls.append(pairs)
                return [table[pair] for pair in pairs]

            def score(handles):
                batches.append(list(handles))
                return handles

            with patch.dict(search_play._W, {"cuda": object()}, clear=True), \
                    patch.object(search_play, "_continuations", side_effect=continue_jobs), \
                    patch.object(search_play, "_leaf_values", side_effect=score):
                action = search_play._halving_choice(None, {}, 1, candidates, candidates[policy_index],
                    keys[policy_index], args, 123, stats)
            results.append((action, stats, calls, batches))
        old, new = results
        self.assertEqual(old[0], new[0])
        self.assertEqual(old[3], new[3], "value inference must retain stage batch sizes and order")
        for name in ("last_targets", "overrides", "override_kinds", "halving_rollouts", "halving_selected_lanes"):
            self.assertEqual(old[1][name], new[1][name], name)
        self.assertEqual(len(new[2]), 1)
        self.assertEqual(new[2][0], [(i, k) for i in range(count) for k in range(samples)])
        selected = sum(map(len, old[2]))
        self.assertEqual(new[1]["halving_discarded_lanes"], count * samples - selected)
        self.assertEqual(new[1]["halving_speculative_lanes"], count * samples)

    def test_same_labels_and_stage_values_across_search_shapes(self):
        for count in (2, 3, 4, 5, 7, 9):
            for samples in (1, 2, 3, 4, 5, 8):
                for policy in (0, count - 1):
                    for pattern in ("random", "ties", "missing"):
                        with self.subTest(candidates=count, samples=samples, policy=policy, pattern=pattern):
                            self.compare(count, samples, policy, pattern)


if __name__ == "__main__":
    unittest.main()
