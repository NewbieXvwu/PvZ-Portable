"""CPU accounting must include a waited simulator grandchild, without wall-time substitution."""
import os
from pathlib import Path
import resource
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pvz_cpu_usage import difference, process_usage, usage, wait_with_cpu_usage


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux /proc accounting")
class CpuUsageTest(unittest.TestCase):
    def test_live_process_ticks_match_getrusage(self):
        proc = process_usage(os.getpid())
        own = usage()
        for key in ("user_s", "system_s"):
            self.assertLess(abs(proc[key] - own[key]), 0.05)

    def test_wait4_includes_reaped_descendant_cpu(self):
        burn = "import time; t=time.process_time();\nwhile time.process_time()-t<0.12: pass"
        child = "import subprocess,sys; subprocess.run([sys.executable,'-c',sys.argv[1]],check=True)"
        before = usage(resource.RUSAGE_CHILDREN)
        process = subprocess.Popen([sys.executable, "-c", child, burn])
        try:
            measured = wait_with_cpu_usage(process, timeout=15)
        finally:
            if process.returncode is None:
                process.terminate()
                process.wait()
        after = difference(usage(resource.RUSAGE_CHILDREN), before)
        self.assertEqual(measured["returncode"], 0)
        self.assertGreater(measured["total_s"], 0.11)
        self.assertAlmostEqual(measured["total_s"], after["total_s"], places=5)
        with self.assertRaisesRegex(RuntimeError, "already reaped"):
            wait_with_cpu_usage(process)


if __name__ == "__main__":
    unittest.main()
