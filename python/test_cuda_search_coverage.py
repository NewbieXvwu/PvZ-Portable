"""GPU usage evidence must distinguish successful requests, fallback and rejected lanes."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import search_play


class CoverageTest(unittest.TestCase):
    def test_agreement_caller_without_backend_setting_keeps_cpu_behavior(self):
        # sim_skip_agreement builds the search namespace itself, without a backend setting.
        with patch.dict(search_play._W, {}, clear=True), \
                patch.object(search_play, "_rollout_policy", return_value="CPU"):
            stats = {}
            result = search_play._continuations(None, {"tick": 0}, 1,
                [({"type": "wait"}, 10)], SimpleNamespace(horizon=2500), stats)
            self.assertEqual(result, ["CPU"])
            self.assertEqual(stats["rollout_lanes"], 1)
            self.assertNotIn("cuda_fallback_lanes", stats)

    def test_gpu_fallback_and_lane_errors_are_counted_per_game(self):
        jobs = [({"type": "wait"}, 1), ({"type": "wait"}, 2)]
        args = SimpleNamespace(rollout_backend="cuda", horizon=2500)
        handles = [({"errorFlags": 0}, None), ({"errorFlags": 8}, None)]
        with patch.dict(search_play._W, {"cuda": object()}, clear=True), \
                patch.object(search_play, "_cuda_continuations", return_value=handles) as gpu, \
                patch.object(search_play, "_rollout_policy", return_value="CPU") as cpu:
            stats = {}
            self.assertEqual(search_play._continuations(None, {"tick": 10}, 0, jobs, args, stats), handles)
            self.assertEqual(stats["cuda_error_lanes"], 1)
            gpu.return_value = None
            self.assertEqual(search_play._continuations(None, {"tick": 10}, 0, jobs, args, stats), ["CPU"] * 2)
            search_play._W["cuda"] = None
            search_play._continuations(None, {"tick": 10}, 0, jobs, args, stats)
            self.assertEqual((stats["rollout_batches"], stats["rollout_lanes"]), (3, 6))
            self.assertEqual((stats["cuda_batches"], stats["cuda_lanes"]), (1, 2))
            self.assertEqual((stats["cuda_fallback_batches"], stats["cuda_fallback_lanes"]), (2, 4))
            self.assertEqual(cpu.call_count, 4)
            self.assertEqual(gpu.call_count, 2)


if __name__ == "__main__":
    unittest.main()
