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

    def test_evaluation_runs_heldout_and_full_cap1_sets_in_one_env(self) -> None:
        class StubModel:
            def eval(self):
                return self

            def load_state_dict(self, _state):
                pass

            def state_dict(self):
                return {}

        class StubEnv:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        task_dir = ROOT / "artifacts/t5/test_stage0_eval"
        shutil.rmtree(task_dir, ignore_errors=True)
        calls = []

        def run_episode(_env, task, seed, _strategy, _model):
            calls.append((task["task_id"], seed))
            return {
                "seed": seed, "won": False, "result": 0,
                "terminal_wave": 1, "wave_count": 1, "terminal_tick": 100,
                "peak_offense": 0, "economy_curve": [],
            }

        heldout_task = {"task_id": "heldout_cap3_x1", "seeds": [42]}
        try:
            with mock.patch.object(task_family, "GameplayModelV1", StubModel), \
                    mock.patch.object(task_family, "PvZEnv", return_value=StubEnv()), \
                    mock.patch.object(task_family.t4_capability_profile, "run_episode", side_effect=run_episode):
                evaluation = task_family._evaluate(
                    StubModel(), Path("unused"), [heldout_task], [heldout_task],
                    self.stage0_tasks, 0, task_dir)

            self.assertEqual(evaluation["gate_set"]["sample_count"], 1)
            self.assertEqual(evaluation["stage0_set"]["sample_count"], 320)
            self.assertEqual(evaluation["stage0_set"]["task_count"], 5)
            self.assertEqual(len(evaluation["stage0_set"]["per_task"]), 5)
            self.assertEqual(len(calls), 321)
            self.assertEqual({task_id for task_id, _ in calls},
                             {heldout_task["task_id"]} | {task["task_id"] for task in self.stage0_tasks})
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)

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


if __name__ == "__main__":
    unittest.main()
