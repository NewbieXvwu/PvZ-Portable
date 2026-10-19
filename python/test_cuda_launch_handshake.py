"""A stale service must not silently discard the queue's CPU-saving wait setting."""
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvz_cuda_rollout import CudaServerError, launch_server


class LaunchHandshakeTest(unittest.TestCase):
    def launch(self, ready, env):
        process = Mock(stdout=io.StringIO(ready), stderr=io.StringIO())
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
                patch("pvz_cuda_rollout.subprocess.Popen", return_value=process) as spawn:
            result = launch_server("service", "net", Path(directory) / "s", env=env)
        self.assertIs(result, process)
        process.terminate.assert_not_called()
        return spawn.call_args.kwargs["env"]

    def test_default_and_explicit_zero_allow_legacy_binary(self):
        for env in ({}, {"PVZ_CUDA_POLL_US": "0"}):
            with self.subTest(env=env):
                self.launch("READY /tmp/s 128\n", env)

    def test_requested_wait_requires_matching_capability(self):
        for ready in ("READY /tmp/s 128\n", "READY /tmp/s 128 host_wait_us=0\n"):
            with self.subTest(ready=ready), tempfile.TemporaryDirectory() as directory:
                process = Mock(stdout=io.StringIO(ready), stderr=io.StringIO())
                with patch.dict(os.environ, {}, clear=True), \
                        patch("pvz_cuda_rollout.subprocess.Popen", return_value=process), \
                        self.assertRaisesRegex(CudaServerError, "rebuild rollout_cuda_small"):
                    launch_server("service", "net", Path(directory) / "s", env={"PVZ_CUDA_POLL_US": "100"})
                process.terminate.assert_called_once_with()
                process.wait.assert_called_once_with(timeout=5)
                self.assertTrue(process.stdout.closed)

    def test_matching_capability_accepts_requested_wait(self):
        env = {"PVZ_CUDA_POLL_US": "100"}
        self.assertEqual(self.launch("READY /tmp/s 128 host_wait_us=100\n", env), env)

    def test_invalid_setting_fails_before_spawning(self):
        for text in ("fast", "", "-1", "10001"):
            with self.subTest(value=text), patch.dict(os.environ, {}, clear=True), \
                    patch("pvz_cuda_rollout.subprocess.Popen") as spawn, tempfile.TemporaryDirectory() as directory, \
                    self.assertRaisesRegex(CudaServerError, "PVZ_CUDA_POLL_US"):
                launch_server("service", "net", Path(directory) / "s", env={"PVZ_CUDA_POLL_US": text})
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
