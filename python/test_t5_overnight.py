"""Regression checks for the T5 overnight order's executable safeguards."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
import random
import unittest

import train_pvz_ppo_task_family as task_family


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
            row = task_family._curve_row(0, selected, 0.0, 0.0, "evaluation.json.gz")
        finally:
            task_family.TASK_RECENT.clear()
            task_family.TASK_RECENT.update(old_recent)
        self.assertEqual(row["curriculum_task_ids"], [item["task_id"] for item in selected])


if __name__ == "__main__":
    unittest.main()
