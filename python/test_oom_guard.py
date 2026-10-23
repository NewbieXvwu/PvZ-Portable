"""task34: the OOM badness each role asks for actually reaches /proc in the child that asks."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

from pvz_common import (OOM_SCORE_CONTROLLER, OOM_SCORE_TRAINING, OOM_SCORE_WORKER,  # noqa: E402
                        set_oom_score_adj)

READ_SCORE = "print(open('/proc/self/oom_score_adj').read().strip())"


@unittest.skipUnless(sys.platform.startswith("linux"), "oom_score_adj is a Linux-only interface")
class OomScoreTest(unittest.TestCase):
    def test_child_writes_its_score(self):
        code = ("from pvz_common import set_oom_score_adj; "
                "assert set_oom_score_adj(1000); " + READ_SCORE)
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "python",
                                check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), str(OOM_SCORE_WORKER))

    def test_engine_preexec_sets_the_score(self):
        from pvz_env import _oom_score_engine
        result = subprocess.run([sys.executable, "-c", READ_SCORE], preexec_fn=_oom_score_engine,
                                check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), str(OOM_SCORE_WORKER))

    def test_roles_are_ordered_above_the_default(self):
        self.assertGreater(OOM_SCORE_WORKER, OOM_SCORE_CONTROLLER)
        self.assertGreater(OOM_SCORE_CONTROLLER, OOM_SCORE_TRAINING)
        self.assertGreater(OOM_SCORE_TRAINING, 0)


if __name__ == "__main__":
    unittest.main()
