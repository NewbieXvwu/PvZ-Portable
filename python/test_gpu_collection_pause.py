"""A training pause waits at game boundaries without stopping the CUDA service."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import search_play


class GpuCollectionPauseTest(unittest.TestCase):
    def test_cpu_and_unconfigured_gpu_do_not_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.pause"
            path.touch()
            with patch.dict(os.environ, {"PVZ_GPU_PAUSE_FILE": str(path)}), \
                    patch.object(search_play.time, "sleep", side_effect=AssertionError("CPU was paused")):
                search_play._wait_for_gpu_pause(SimpleNamespace(rollout_backend="cpp"), 30000)
            with patch.dict(os.environ, {}, clear=True), \
                    patch.object(search_play.time, "sleep", side_effect=AssertionError("default GPU was paused")):
                search_play._wait_for_gpu_pause(SimpleNamespace(rollout_backend="cuda"), 30000)

    def test_pause_before_first_game_connects_once_and_checks_every_ten_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training pause"
            path.touch()
            client = Mock()
            sleeps = []
            def sleep(seconds):
                sleeps.append(seconds)
                if len(sleeps) == 2:
                    path.unlink()
            with patch.dict(search_play._W, {"cuda": client}, clear=True), \
                    patch.dict(os.environ, {"PVZ_GPU_PAUSE_FILE": str(path)}), \
                    patch.object(search_play.time, "sleep", side_effect=sleep):
                search_play._wait_for_gpu_pause(SimpleNamespace(rollout_backend="cuda", out=directory), 30000)
            self.assertEqual(sleeps, [10, 10])
            events = [json.loads(line) for line in (Path(directory) / "gpu_pause.jsonl").read_text().splitlines()]
            self.assertEqual([(e["event"], e["seed"]) for e in events], [("paused", 30000), ("resumed", 30000)])
            client.connect.assert_called_once_with()
            client.close.assert_not_called()

    def test_removed_file_allows_next_game_without_waiting_or_reconnecting(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Mock()
            args = SimpleNamespace(rollout_backend="cuda", gpu_pause_file=str(Path(directory) / "absent"))
            with patch.dict(search_play._W, {"cuda": client}, clear=True), \
                    patch.object(search_play.time, "sleep", side_effect=AssertionError("missing file blocked a game")):
                search_play._wait_for_gpu_pause(args, 30001)
            client.connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
