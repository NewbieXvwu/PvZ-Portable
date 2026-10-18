"""Exercise the shared simulator with ASan, without requiring a GPU or local artifacts."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SharedMechanicsTest(unittest.TestCase):
    def test_shared_mechanics(self):
        with tempfile.TemporaryDirectory(prefix="pvz-cuda-mechanics-") as directory:
            binary = str(Path(directory) / "mechanics")
            subprocess.run(["g++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
                            "-I", str(ROOT), str(ROOT / "cuda/SimMechanicsTest.cpp"), "-o", binary],
                           check=True, capture_output=True, text=True)
            run = subprocess.run([binary], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("mechanics OK", run.stdout)


if __name__ == "__main__":
    unittest.main()
