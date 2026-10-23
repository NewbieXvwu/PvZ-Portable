"""task34: the pressure test must refuse by default and find the processes it is told to avoid.

None of these tests allocate: the script's --i-know path is never entered here.
"""
import json
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python")]

import oom_stress_test as stress  # noqa: E402


class StressPreflightTest(unittest.TestCase):
    def test_refuses_without_i_know(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/oom_stress_test.py")],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertTrue(json.loads(result.stdout)["refused"])

    def test_finds_a_process_by_cmdline(self):
        fake = subprocess.Popen(["bash", "-c", "exec -a fake_train_rollout_policy.py sleep 30"])
        try:
            deadline = time.monotonic() + 5
            found = []
            while time.monotonic() < deadline:
                found = stress.matching_processes(("train_rollout_policy.py",))
                if any(pid == fake.pid for pid, _ in found):
                    break
                time.sleep(0.05)
            self.assertIn(fake.pid, [pid for pid, _ in found])
        finally:
            fake.kill()
            fake.wait()

    def test_meminfo_and_timing(self):
        total, available = stress.meminfo()
        self.assertGreater(total, 0)
        self.assertLessEqual(available, total)
        self.assertGreaterEqual(stress.timed(["bash", "-c", "true"])["seconds"], 0.0)
        self.assertEqual(stress.timed(["bash", "-c", "true"])["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
