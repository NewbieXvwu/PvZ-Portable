"""An incomplete or duplicated evaluation must never pass the 128-pair gate."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cuda_phase2_paired import load_games, mcnemar_exact


class PairedGateTest(unittest.TestCase):
    def test_exact_probability(self):
        self.assertAlmostEqual(mcnemar_exact(5, 3), 0.7265625)
        self.assertEqual(mcnemar_exact(0, 0), 1.0)

    def test_missing_wrong_and_complete_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            for seeds, passed in (([30000], False), ([1, 2], False), ([30000, 30001], True)):
                path.write_text("".join(json.dumps({"seed": seed, "won": True}) + "\n" for seed in seeds))
                run = subprocess.run([sys.executable, str(ROOT / "scripts/cuda_phase2_paired.py"),
                                      str(path), str(path), "--seeds", "2"],
                                     check=True, capture_output=True, text=True)
                self.assertEqual(json.loads(run.stdout)["gate_passed"], passed)

    def test_duplicate_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            path.write_text('{"seed":30000,"won":true}\n' * 2)
            with self.assertRaisesRegex(ValueError, "duplicate seed"):
                load_games(path)


if __name__ == "__main__":
    unittest.main()
