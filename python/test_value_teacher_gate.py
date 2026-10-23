"""Acceptance evidence must pair identical roots and exclude extra noise-search work."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_value_teacher import TASKS, compare


class ValueTeacherGateTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.out = Path(directory.name)
        self.plan = dict(contract={"settings": "same"}, implementation={"binary": "test"})
        for phase in ("before", "after"):
            self.write(self.out / f"{phase}-plan.json", self.plan)
            for task, _ in TASKS:
                for index in range(32):
                    dest = self.out / phase / task / f"seed_index_{index}"
                    self.write(dest / "result.json", dict(
                        seed=index, decisions=1, won=True, noise_points=1, noise_agreement=1,
                        # Raw before CPU deliberately includes substantial extra noise work.
                        host_cpu={"worker": {"total_s": 100}, "simulator": {"total_s": 100}},
                        host_cpu_excluding_noise_s=dict(worker=2, simulator=8),
                        memory={who: {"VmRSS": "100 kB"} for who in ("worker", "engine")}))
                    self.write(dest / "trace.json", [dict(state_sha256="root", future_seed=10,
                                                          searched=True, chosen=dict(type="wait", ticks=60))])

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def run_compare(self):
        with redirect_stdout(io.StringIO()):
            compare(self.out)

    def first(self, phase, name):
        return self.out / phase / TASKS[0][0] / "seed_index_0" / name

    def test_identical_paired_games_pass_using_cpu_without_noise(self):
        self.run_compare()
        result = json.loads((self.out / "comparison.json").read_text())
        self.assertTrue(all(row["passed"] for row in result.values()))
        self.assertEqual(result[TASKS[0][0]]["cpu_s_per_game"]["before"], 10)

    def test_different_root_requires_complete_shadow_evidence(self):
        path = self.first("after", "trace.json")
        trace = json.loads(path.read_text())
        trace[0]["state_sha256"] = "diverged"
        self.write(path, trace)
        with self.assertRaises(FileNotFoundError):
            self.run_compare()
        self.write(self.out / "shadow-plan.json", self.plan)
        self.write(self.first("shadow", "result.json"), dict(decisions=1))
        self.write(self.first("shadow", "trace.json"), json.loads(self.first("before", "trace.json").read_text()))
        self.run_compare()

    def test_cpu_regression_cannot_hide_behind_raw_baseline_noise_cost(self):
        path = self.first("after", "result.json")
        result = json.loads(path.read_text())
        result["host_cpu_excluding_noise_s"]["simulator"] = 108
        self.write(path, result)
        with self.assertRaises(SystemExit):
            self.run_compare()

    def test_truncated_trace_is_rejected(self):
        self.write(self.first("after", "trace.json"), [])
        with self.assertRaisesRegex(ValueError, "incomplete trace"):
            self.run_compare()

    def test_changed_recipe_is_rejected(self):
        plan = deepcopy(self.plan)
        plan["contract"]["settings"] = "changed"
        self.write(self.out / "after-plan.json", plan)
        with self.assertRaisesRegex(ValueError, "inputs or settings differ"):
            self.run_compare()


if __name__ == "__main__":
    unittest.main()
