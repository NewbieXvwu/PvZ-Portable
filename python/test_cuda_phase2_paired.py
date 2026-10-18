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
    def make_runs(self, directory, seeds=(30000, 30001)):
        root = Path(directory)
        inputs = {}
        for name in ("rollout_policy", "value", "static_value", "manifest", "executable"):
            path = root / f"{name}.bin"
            path.write_bytes(name.encode())
            inputs[name] = str(path)
        settings = {**inputs, "task_id": "day_l7_z15_eval", "samples": "3", "per_card": "1",
                    "halving": "True", "horizon": "2500", "margin": "0.02", "min_gap": "300",
                    "rollout_sim_skip": "29", "cpp_rollouts": "True", "max_actions": "3000",
                    "dagger_beta": "1.0", "arms": "fast_search", "workers": "4"}
        paths = []
        for name, backend in (("cpu", "cpp"), ("gpu", "cuda")):
            path = root / name
            path.mkdir(exist_ok=True)
            (path / "config.json").write_text(json.dumps({**settings, "rollout_backend": backend}))
            (path / "games.jsonl").write_text("".join(json.dumps({
                "seed": seed, "won": True, "arm": "fast_search",
                "cuda_batches": int(name == "gpu"), "cuda_lanes": 3 * int(name == "gpu")}) + "\n"
                for seed in seeds))
            paths.append(path)
        return paths

    def summarize(self, cpu, gpu):
        run = subprocess.run([sys.executable, str(ROOT / "scripts/cuda_phase2_paired.py"),
                              str(cpu), str(gpu), "--seeds", "2"],
                             check=True, capture_output=True, text=True)
        return json.loads(run.stdout)

    def test_exact_probability(self):
        self.assertAlmostEqual(mcnemar_exact(5, 3), 0.7265625)
        self.assertEqual(mcnemar_exact(0, 0), 1.0)

    def test_missing_wrong_and_complete_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            for seeds, passed in (([30000], False), ([1, 2], False), ([30000, 30001], True)):
                paths = self.make_runs(directory, seeds)
                self.assertEqual(self.summarize(*paths)["gate_passed"], passed)

    def test_cpu_arm_cannot_pose_as_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            cpu, gpu = self.make_runs(directory)
            (gpu / "games.jsonl").write_bytes((cpu / "games.jsonl").read_bytes())
            summary = self.summarize(cpu, gpu)
            self.assertFalse(summary["gate_passed"])
            self.assertEqual(summary["gpu_uncovered_seeds"], [30000, 30001])

    def test_configuration_or_model_mismatch_cannot_pass(self):
        for field, value in (("horizon", "1000"), ("dagger_beta", "0.5"),
                             ("task_id", "day_l7_z13_eval"), ("rollout_backend", "cpp"),
                             ("rollout_policy", "different")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                cpu, gpu = self.make_runs(directory)
                config = json.loads((gpu / "config.json").read_text())
                if field == "rollout_policy":
                    model = Path(directory) / "another_student.bin"
                    model.write_bytes(b"different weights")
                    value = str(model)
                config[field] = value
                (gpu / "config.json").write_text(json.dumps(config))
                self.assertFalse(self.summarize(cpu, gpu)["gate_passed"])

    def test_missing_config_does_not_establish_comparability(self):
        with tempfile.TemporaryDirectory() as directory:
            cpu, gpu = self.make_runs(directory)
            (gpu / "config.json").unlink()
            summary = self.summarize(cpu, gpu)
            self.assertFalse(summary["gate_passed"])
            self.assertFalse(summary["gate_matching_configuration"])

    def test_worker_count_is_reported_without_changing_strength_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            cpu, gpu = self.make_runs(directory)
            config = json.loads((cpu / "config.json").read_text())
            config["workers"] = "2"
            (cpu / "config.json").write_text(json.dumps(config))
            summary = self.summarize(cpu, gpu)
            self.assertTrue(summary["gate_passed"])
            self.assertEqual((summary["workers_cpu"], summary["workers_gpu"]), ("2", "4"))

    def test_duplicate_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.jsonl"
            path.write_text('{"seed":30000,"won":true}\n' * 2)
            with self.assertRaisesRegex(ValueError, "duplicate seed"):
                load_games(path)


if __name__ == "__main__":
    unittest.main()
