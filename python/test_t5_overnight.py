"""Regression checks for the T5 overnight order's executable safeguards."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest import mock

import torch

import train_pvz_ppo_task_family as task_family
import t5_stage0_gate


ROOT = Path(__file__).resolve().parent.parent


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tasks = json.loads((ROOT / "artifacts/task_family/train.json").read_text())["tasks"]

    def test_all_curriculum_preserves_the_manifest_task_list(self) -> None:
        self.assertIs(task_family._curriculum_tasks(self.tasks, "all"), self.tasks)

    def test_cap1_is_selected_from_manifest_properties(self) -> None:
        selected = task_family._curriculum_tasks(self.tasks, "cap1")
        expected = [task for task in self.tasks
                    if task["wave_cap"] == 1 and task["zombie_count_multiplier"] == 1.0]

        self.assertEqual(selected, expected)
        self.assertEqual(len(selected), 5)
        self.assertTrue(all(task["wave_cap"] == 1
                            and task["zombie_count_multiplier"] == 1.0 for task in selected))

    def test_assignment_sampling_uses_only_selected_tasks(self) -> None:
        selected = task_family._curriculum_tasks(self.tasks, "cap1")
        old_recent = task_family.TASK_RECENT.copy()
        try:
            task_family.TASK_RECENT.clear()
            task_family.TASK_RECENT.update({task["task_id"]: deque(maxlen=64) for task in selected})
            assignments = task_family._assignments(selected, list(range(256)), random.Random(1701))
        finally:
            task_family.TASK_RECENT.clear()
            task_family.TASK_RECENT.update(old_recent)

        selected_ids = {task["task_id"] for task in selected}
        self.assertEqual({item["task"]["task_id"] for item in assignments.values()}, selected_ids)

    def test_empty_or_unknown_curriculum_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "selected no training tasks"):
            task_family._curriculum_tasks([], "cap1")
        with self.assertRaisesRegex(ValueError, "unsupported curriculum"):
            task_family._curriculum_tasks(self.tasks, "unknown")

    def test_learning_curve_records_the_sampled_task_ids(self) -> None:
        selected = task_family._curriculum_tasks(self.tasks, "cap1")
        old_recent = task_family.TASK_RECENT.copy()
        try:
            for item in selected:
                task_family.TASK_RECENT.setdefault(item["task_id"], deque(maxlen=64))
            row = task_family._curve_row(0, selected, 0.0, 0.0, 0.0, "evaluation.json.gz")
        finally:
            task_family.TASK_RECENT.clear()
            task_family.TASK_RECENT.update(old_recent)
        self.assertEqual(row["curriculum_task_ids"], [item["task_id"] for item in selected])


class Stage0EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_tasks = json.loads((ROOT / "artifacts/task_family/train.json").read_text())["tasks"]
        cls.stage0_tasks = task_family._curriculum_tasks(cls.train_tasks, "cap1")

    def test_t4_cap1_baseline_covers_all_320_frozen_seeds(self) -> None:
        train, heldout = task_family._task_family()
        gate_tasks, reference_tasks = task_family._heldout_tasks(heldout)
        baseline = task_family._baseline(train["tasks"], gate_tasks, reference_tasks)
        self.assertEqual(baseline["stage0_set"]["sample_count"], 320)
        self.assertEqual(baseline["stage0_set"]["pass_rate"], 0.0)
        self.assertEqual(baseline["stage0_set"]["task_count"], 5)

    def test_stage0_zero_signal_needs_all_five_full_zero_windows(self) -> None:
        ids = [task["task_id"] for task in self.stage0_tasks]
        old_recent = task_family.TASK_RECENT.copy()
        try:
            task_family.TASK_RECENT.clear()
            task_family.TASK_RECENT.update({task_id: deque([False] * 64, maxlen=64) for task_id in ids})
            self.assertTrue(task_family._stage0_has_no_signal(0.0, self.stage0_tasks))
            task_family.TASK_RECENT[ids[0]][-1] = True
            self.assertFalse(task_family._stage0_has_no_signal(0.0, self.stage0_tasks))
            self.assertFalse(task_family._stage0_has_no_signal(0.01, self.stage0_tasks))
        finally:
            task_family.TASK_RECENT.clear()
            task_family.TASK_RECENT.update(old_recent)

    def test_gate_requires_both_rate_and_320_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "training_state.json"
            gate_path = root / "stage0_gate.json"
            state_path.write_text(json.dumps({"cumulative_episodes": 5000, "evaluations": [{
                "cumulative_episodes": 5000,
                "stage0_set": {"pass_rate": 0.5, "sample_count": 320},
            }]}))

            passed = t5_stage0_gate.evaluate(state_path, gate_path)
            self.assertEqual(passed["result"], "pass")
            self.assertEqual(json.loads(gate_path.read_text())["commit"], passed["commit"])
            state_path.write_text(json.dumps({"evaluations": [{
                "cumulative_episodes": 5000,
                "stage0_set": {"pass_rate": 0.5, "sample_count": 319},
            }]}))
            self.assertEqual(t5_stage0_gate.evaluate(state_path, gate_path)["result"], "fail")

    def test_full_task_curriculum_is_blocked_without_stage0_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gate_path = Path(temporary) / "missing.json"
            with self.assertRaisesRegex(RuntimeError, "阶段 0 未通过，禁止进入阶段 1"):
                task_family._check_stage0_gate(self.train_tasks, False, None, gate_path)
            override = task_family._check_stage0_gate(
                self.train_tasks, True, "documented stage0 exception", gate_path)
            self.assertEqual(override, {"used": True, "reason": "documented stage0 exception"})
            with self.assertRaisesRegex(ValueError, "requires --motivation"):
                task_family._check_stage0_gate(self.train_tasks, True, None, gate_path)
            gate_path.write_text(json.dumps({"result": "pass"}))
            self.assertIsNone(task_family._check_stage0_gate(self.train_tasks, False, None, gate_path))

    def test_changed_seed0_hash_requires_note_and_records_both_hashes(self) -> None:
        with mock.patch.object(task_family, "MODEL_ARCHITECTURE_VERSION", 5):
            with self.assertRaisesRegex(RuntimeError, "网络结构已变更，seed-0 初始化不再与 T4 基线一致"):
                task_family._seed0_initialization_baseline("new-hash", "t4-hash", None)
            record = task_family._seed0_initialization_baseline(
                "new-hash", "t4-hash", "new lane-token architecture supersedes T4")
        self.assertEqual(record["actual"], "new-hash")
        self.assertEqual(record["t4"], "t4-hash")
        self.assertEqual(record["status"], "superseded")


class EvaluationTests(unittest.TestCase):
    """Evaluation runs on a process pool and the reference set is final-eval only."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tasks = json.loads((ROOT / "artifacts/task_family/train.json").read_text())["tasks"]
        cls.stage0_tasks = [task for task in cls.tasks
                            if task["wave_cap"] == 1 and task["zombie_count_multiplier"] == 1.0]
        cls.gate_task = {"task_id": "heldout_cap3_x1", "seeds": [42, 43]}
        cls.reference_only_task = {"task_id": "heldout_cap3_x15", "seeds": [44]}

    class _StubModel:
        def eval(self):
            return self

        def load_state_dict(self, _state):
            pass

        def state_dict(self):
            return {}

    @staticmethod
    def _runner(calls):
        """A stand-in for the pool that honours the job-id ordering contract."""
        def runner(jobs, _model_state, _resource_dir, _workers, _threads, _device):
            records = []
            for job_id in sorted(jobs):
                job = jobs[job_id]
                calls.append((job["task"]["task_id"], job["seed"]))
                records.append({
                    "seed": job["seed"], "won": False, "result": 0,
                    "terminal_wave": 1, "wave_count": 1, "terminal_tick": 100,
                    "peak_offense": 0, "economy_curve": [],
                })
            return records
        return runner

    def _evaluate(self, temporary, calls, include_reference):
        with mock.patch.object(task_family, "GameplayModelV1", self._StubModel):
            return task_family._evaluate(
                self._StubModel(), Path("unused"),
                [self.gate_task, self.reference_only_task], [self.gate_task],
                self.stage0_tasks, 0, Path(temporary),
                workers=2, worker_threads=1, worker_device="cpu",
                include_reference=include_reference, runner=self._runner(calls))

    def test_final_evaluation_covers_gate_reference_and_stage0(self) -> None:
        calls: list[tuple[str, int]] = []
        with tempfile.TemporaryDirectory() as temporary:
            evaluation = self._evaluate(temporary, calls, include_reference=True)
            written = sorted(path.name for path in (Path(temporary) / "evaluations").iterdir())
        self.assertEqual(written, ["heldout_0000000.json.gz"])
        self.assertEqual(evaluation["gate_set"]["sample_count"], 2)
        self.assertEqual(evaluation["gate_set"]["task_count"], 1)
        self.assertEqual(evaluation["reference_set"]["sample_count"], 3)
        self.assertEqual(evaluation["reference_set"]["task_count"], 2)
        self.assertFalse(evaluation["reference_set"]["skipped"])
        self.assertEqual(evaluation["stage0_set"]["sample_count"], 320)
        self.assertEqual(evaluation["stage0_set"]["task_count"], 5)
        self.assertEqual(len(evaluation["stage0_set"]["per_task"]), 5)
        self.assertEqual(len(calls), 323)
        self.assertEqual({task_id for task_id, _ in calls},
                         {self.gate_task["task_id"], self.reference_only_task["task_id"]}
                         | {task["task_id"] for task in self.stage0_tasks})

    def test_intermediate_evaluation_skips_the_reference_set(self) -> None:
        calls: list[tuple[str, int]] = []
        with tempfile.TemporaryDirectory() as temporary:
            evaluation = self._evaluate(temporary, calls, include_reference=False)
            written = sorted(path.name for path in (Path(temporary) / "evaluations").iterdir())
        self.assertEqual(evaluation["gate_set"]["sample_count"], 2)
        self.assertEqual(evaluation["reference_set"]["skipped"], True)
        self.assertIsNone(evaluation["reference_set"]["pass_rate"])
        self.assertEqual(evaluation["reference_set"]["sample_count"], 0)
        self.assertEqual(evaluation["stage0_set"]["sample_count"], 320)
        self.assertEqual(len(calls), 322)
        self.assertNotIn(self.reference_only_task["task_id"], {task_id for task_id, _ in calls})
        self.assertEqual(written, ["heldout_0000000_gate_only.json.gz"])

    def test_job_layout_is_ordered_and_labels_every_bucket(self) -> None:
        reference, jobs, layout = task_family._evaluation_jobs(
            [self.gate_task, self.reference_only_task], [self.gate_task],
            self.stage0_tasks, include_reference=True)
        self.assertEqual([task["task_id"] for task in reference],
                         ["heldout_cap3_x1", "heldout_cap3_x15"])
        self.assertEqual(sorted(jobs), list(range(len(jobs))))
        self.assertEqual(len(jobs), 2 + 1 + 320)
        self.assertEqual(layout[:3], [("reference", "heldout_cap3_x1"),
                                      ("reference", "heldout_cap3_x1"),
                                      ("reference", "heldout_cap3_x15")])
        self.assertEqual(layout[3], ("stage0", self.stage0_tasks[0]["task_id"]))
        self.assertEqual(layout[-1], ("stage0", self.stage0_tasks[-1]["task_id"]))

        reference, jobs, layout = task_family._evaluation_jobs(
            [self.gate_task, self.reference_only_task], [self.gate_task],
            self.stage0_tasks, include_reference=False)
        self.assertEqual([task["task_id"] for task in reference], ["heldout_cap3_x1"])
        self.assertEqual(len(jobs), 2 + 320)

    def test_every_job_runs_exactly_once(self) -> None:
        calls: list[tuple[str, int]] = []
        with tempfile.TemporaryDirectory() as temporary:
            self._evaluate(temporary, calls, include_reference=True)
        self.assertEqual(len(calls), len(set(calls)),
                         "an evaluation episode ran more than once")

    def test_a_short_result_list_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(task_family, "GameplayModelV1", self._StubModel):
                with self.assertRaisesRegex(RuntimeError, "evaluation returned 0 records"):
                    task_family._evaluate(
                        self._StubModel(), Path("unused"),
                        [self.gate_task, self.reference_only_task], [self.gate_task],
                        self.stage0_tasks, 0, Path(temporary),
                        workers=2, worker_threads=1, worker_device="cpu",
                        runner=lambda *_args: [])


class UpdateDeviceGuardTests(unittest.TestCase):
    """M5: a formal run must not silently PPO-update on the CPU.

    ``resolve_device("auto")`` returns the CPU without complaint when CUDA drops off
    the bus (the known WSL long-uptime failure).  The first update then never lands
    while the run still looks alive, which is exactly what burned a stage 0 for 75+
    minutes.
    """

    def _benchmark(self, temporary: str, device: str) -> Path:
        path = Path(temporary) / "ppo_update_2000_flex_saved.json"
        path.write_text(json.dumps({"device": device, "device_name": "NVIDIA GeForce RTX 5080"}))
        return path

    def test_cpu_update_is_blocked_when_the_benchmark_was_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._benchmark(temporary, "cuda")
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                task_family._check_update_device(torch.device("cpu"), path)

    def test_cpu_override_requires_motivation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._benchmark(temporary, "cuda")
            with self.assertRaisesRegex(ValueError, "requires --motivation"):
                task_family._check_update_device(
                    torch.device("cpu"), path, allow_cpu_update=True)
            record = task_family._check_update_device(
                torch.device("cpu"), path, allow_cpu_update=True,
                motivation="CPU-only box; rerunning the tuned hyperparameters there")
        self.assertEqual(record["override"],
                         {"used": True,
                          "reason": "CPU-only box; rerunning the tuned hyperparameters there"})
        self.assertEqual(record["benchmark_device"], "cuda")

    def test_cpu_benchmark_and_missing_benchmark_do_not_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cpu_path = self._benchmark(temporary, "cpu")
            record = task_family._check_update_device(torch.device("cpu"), cpu_path)
            self.assertIsNone(record["override"])
            missing = Path(temporary) / "absent.json"
            record = task_family._check_update_device(torch.device("cpu"), missing)
            self.assertIsNone(record["benchmark_device"])
            self.assertIsNone(record["override"])

    def test_cuda_device_always_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._benchmark(temporary, "cuda")
            with mock.patch.object(torch.cuda, "get_device_name",
                                   return_value="NVIDIA GeForce RTX 5080"):
                record = task_family._check_update_device(torch.device("cuda"), path)
        self.assertEqual(record["resolved_device"], "cuda")
        self.assertEqual(record["device_name"], "NVIDIA GeForce RTX 5080")
        self.assertIsNone(record["override"])

    def test_repo_benchmark_records_cuda(self) -> None:
        """The guard is only meaningful while the recorded evidence is CUDA."""
        measured = json.loads(
            (ROOT / "artifacts/t5/perf/ppo_update_2000_flex_saved.json").read_text())
        self.assertEqual(measured["device"], "cuda")


if __name__ == "__main__":
    unittest.main()
