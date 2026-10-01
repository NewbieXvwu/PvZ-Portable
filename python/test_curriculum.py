"""Coverage, failure handling and deterministic continuation of curriculum draws."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pvz_curriculum import initial_state, observe, probabilities, validate_settings, validate_state
from pvz_research import ROOT, _assign, load_config
from pvz_common import sha256_file

TASKS = [dict(task_id=name, terrain=terrain, seeds=list(range(10)))
         for terrain, names in (("day", ("fail", "progress", "mastered")), ("roof", ("roof",)))
         for name in names]
SETTINGS = dict(window_episodes=16, minimum_window_episodes=8, uniform_fraction=.25)


def outcomes(key, values):
    return [dict(task_id=key, terminated=True, truncated=False, won=value) for value in values]


class CurriculumTests(unittest.TestCase):
    def test_zero_successes_do_not_overweight_unlearnable_tasks(self):
        state = initial_state(TASKS)
        observe(state, outcomes("fail", [False] * 80), SETTINGS)
        balanced, _ = probabilities(TASKS, state, SETTINGS, "terrain_balanced")
        progress, _ = probabilities(TASKS, state, SETTINGS, "learning_progress")
        self.assertEqual(balanced, progress)

    def test_terrain_coverage_survives_unequal_task_counts_and_progress(self):
        state = initial_state(TASKS)
        observe(state, outcomes("progress", [False] * 16 + [True] * 16), SETTINGS)
        observe(state, outcomes("mastered", [True] * 32), SETTINGS)
        weights, details = probabilities(TASKS, state, SETTINGS, "learning_progress")
        self.assertAlmostEqual(sum(weights[:3]), weights[3])
        self.assertGreater(weights[1], weights[0])
        self.assertEqual(weights[0], weights[2])
        for task, probability in zip(TASKS, weights):
            count = sum(t["terrain"] == task["terrain"] for t in TASKS)
            self.assertGreaterEqual(probability, .25 / 2 / count)
        self.assertEqual(details["tasks"]["progress"]["previous_pass_rate"], 0)
        self.assertEqual(details["tasks"]["progress"]["recent_pass_rate"], 1)

    def test_truncation_is_reported_without_inventing_a_failure(self):
        state = initial_state(TASKS)
        observe(state, [dict(task_id="fail", terminated=False, truncated=True, won=False)], SETTINGS)
        self.assertEqual(state["ignored_truncations"]["fail"], 1)
        self.assertEqual(state["completed"]["fail"], 0)
        self.assertEqual(state["history"]["fail"], [])
        with self.assertRaisesRegex(ValueError, "normal terminal or a win"):
            observe(state, [dict(task_id="fail", terminated=False, truncated=True, won=True)], SETTINGS)

    def test_serialized_history_and_assignment_rng_restore_the_exact_next_batch(self):
        state = initial_state(TASKS)
        observe(state, outcomes("progress", [False] * 20 + [True] * 12), SETTINGS)
        rng = random.Random(111)
        rng.random()
        saved_rng, saved_state = rng.getstate(), json.dumps(state)
        weights, expected_details = probabilities(TASKS, state, SETTINGS, "learning_progress")
        expected = _assign(TASKS, list(range(40)), rng, {}, "learning_progress", weights)
        restored_state = json.loads(saved_state)
        restored_rng = random.Random()
        restored_rng.setstate(saved_rng)
        weights, actual_details = probabilities(TASKS, restored_state, SETTINGS, "learning_progress")
        actual = _assign(TASKS, list(range(40)), restored_rng, {}, "learning_progress", weights)
        self.assertEqual(actual, expected)
        self.assertEqual(actual_details, expected_details)
        self.assertEqual(restored_state, state)

    def test_unknown_tasks_and_changed_pool_cannot_contaminate_course(self):
        state = initial_state(TASKS)
        with self.assertRaisesRegex(ValueError, "outside its training pool"):
            observe(state, outcomes("validation", [True]), SETTINGS)
        with self.assertRaisesRegex(ValueError, "task pool differs"):
            validate_state(state, TASKS[:2], SETTINGS)

    def test_invalid_windows_and_zero_coverage_are_rejected(self):
        for changes in (dict(uniform_fraction=0), dict(uniform_fraction=float("nan")),
                        dict(minimum_window_episodes=17), dict(window_episodes=True),
                        dict(uniform_fraction=True)):
            with self.assertRaises(ValueError):
                validate_settings({**SETTINGS, **changes})

    def test_probability_query_does_not_mutate_resume_state(self):
        state = initial_state(TASKS)
        original = copy.deepcopy(state)
        probabilities(TASKS, state, SETTINGS, "learning_progress")
        self.assertEqual(state, original)

    def test_explicit_config_requires_course_parameters_without_changing_legacy_schema(self):
        config = json.loads((ROOT / "experiments/t5/reward_comparison_v2/reward_r0_seed0_v2.json").read_text())
        config["prerequisites"] = []  # Parsing only, not an execution permission.
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(config))
            load_config(path)
            config["sampling"]["method"] = "learning_progress"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "sampling settings"):
                load_config(path)
            config["sampling"]["curriculum"] = SETTINGS
            path.write_text(json.dumps(config))
            _, tasks, _ = load_config(path)
            self.assertEqual(len(tasks), 5)

    def test_old_training_gate_rejects_this_changed_core(self):
        binary = ROOT / "build/pvz-portable"
        gate = json.loads((ROOT / "gates/T5-A-research-v4.json").read_text())
        def hash_with_existing_binary(path):
            return gate["simulator_sha256"] if path == binary else sha256_file(path)
        with patch("pvz_research.sha256_file", side_effect=hash_with_existing_binary):
            with self.assertRaisesRegex(RuntimeError, "source is stale"):
                load_config(ROOT / "experiments/t5/reward_comparison_v2/reward_r0_seed0_v2.json")


if __name__ == "__main__":
    unittest.main()
