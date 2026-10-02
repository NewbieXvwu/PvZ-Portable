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
SETTINGS = dict(window_episodes=16, minimum_window_episodes=8, uniform_fraction=.25, coverage="terrain")
LENGTH_TASKS = [dict(task_id=f"{terrain}_{cap}_{index}", terrain=terrain, wave_cap=cap, seeds=list(range(10)))
                for terrain in ("day", "roof") for cap in (1, 3, 5)
                for index in range(2 if (terrain, cap) == ("day", 1) else 1)]
LENGTH_SETTINGS = {**SETTINGS, "coverage": "terrain_wave_cap"}


def explicit_fixture():
    config = json.loads((ROOT / "experiments/t5/reward_comparison_v2/reward_r0_seed0_v2.json").read_text())
    config["model"]["input_flags"] = 0
    config["runtime"].update(deterministic_algorithms=True, cublas_workspace_config=":4096:8")
    return config


def outcomes(key, values):
    return [dict(task_id=key, terminated=True, truncated=False, won=value) for value in values]


class CurriculumTests(unittest.TestCase):
    def test_singleton_coverage_groups_make_progress_sampling_identical(self):
        tasks = [task for task in LENGTH_TASKS if task["task_id"].endswith("_0")]
        state = initial_state(tasks)
        observe(state, outcomes(tasks[0]["task_id"], [False] * 16 + [True] * 16), LENGTH_SETTINGS)
        uniform, _ = probabilities(tasks, state, LENGTH_SETTINGS, "terrain_balanced")
        progress, details = probabilities(tasks, state, LENGTH_SETTINGS, "learning_progress")
        self.assertGreater(details["tasks"][tasks[0]["task_id"]]["positive_progress"], 0)
        self.assertEqual(uniform, progress)

    def test_probe_gate_cannot_authorize_another_experiment(self):
        config = explicit_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            gate = Path(temporary) / "gate.json"
            gate.write_text(json.dumps({"gate_result": "pass", "allowed_experiment_ids": ["probe"]}))
            config["prerequisites"] = [str(gate)]
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(RuntimeError, "does not authorize"):
                load_config(path)

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

    def test_short_task_progress_cannot_take_mass_from_long_tasks(self):
        state = initial_state(LENGTH_TASKS)
        observe(state, outcomes("day_1_0", [False] * 16 + [True] * 16), LENGTH_SETTINGS)
        weights, details = probabilities(LENGTH_TASKS, state, LENGTH_SETTINGS, "learning_progress")
        for terrain in ("day", "roof"):
            for cap in (1, 3, 5):
                mass = sum(w for t, w in zip(LENGTH_TASKS, weights) if (t["terrain"], t["wave_cap"]) == (terrain, cap))
                self.assertAlmostEqual(mass, 1 / 6)
                self.assertAlmostEqual(details["coverage_probabilities"][terrain][f"cap{cap}"], 1 / 6)
        self.assertGreater(weights[0], weights[1])
        self.assertGreaterEqual(weights[1], .25 / 6 / 2)

    def test_unequal_length_group_counts_still_keep_equal_terrain_mass(self):
        tasks = [t for t in LENGTH_TASKS if t["terrain"] == "day" or t["wave_cap"] == 5]
        weights, _ = probabilities(tasks, initial_state(tasks), LENGTH_SETTINGS, "terrain_balanced")
        self.assertAlmostEqual(sum(w for t, w in zip(tasks, weights) if t["terrain"] == "day"), .5)
        self.assertAlmostEqual(sum(w for t, w in zip(tasks, weights) if t["terrain"] == "roof"), .5)

    def test_wave_cap_coverage_rejects_missing_invalid_and_changed_metadata(self):
        with self.assertRaisesRegex(ValueError, "unique task IDs"):
            probabilities(LENGTH_TASKS + LENGTH_TASKS[:1], initial_state(LENGTH_TASKS),
                          LENGTH_SETTINGS, "terrain_balanced")
        with self.assertRaisesRegex(ValueError, "explicit wave_cap"):
            probabilities(TASKS, initial_state(TASKS), LENGTH_SETTINGS, "terrain_balanced")
        for value in (True, 0, 51, "5"):
            tasks = copy.deepcopy(LENGTH_TASKS)
            tasks[0]["wave_cap"] = value
            with self.assertRaisesRegex(ValueError, "explicit wave_cap"):
                probabilities(tasks, initial_state(tasks), LENGTH_SETTINGS, "learning_progress")
        state = initial_state(LENGTH_TASKS)
        tasks = copy.deepcopy(LENGTH_TASKS)
        tasks[0]["wave_cap"] = 3
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            validate_state(state, tasks, LENGTH_SETTINGS)

    def test_length_group_history_and_rng_resume_exactly(self):
        state = initial_state(LENGTH_TASKS)
        observe(state, outcomes("day_1_0", [False] * 16 + [True] * 16), LENGTH_SETTINGS)
        rng = random.Random(88)
        rng.random()
        saved = rng.getstate()
        weights, details = probabilities(LENGTH_TASKS, state, LENGTH_SETTINGS, "learning_progress")
        expected = _assign(LENGTH_TASKS, list(range(80)), rng, {}, "learning_progress", weights)
        restored = json.loads(json.dumps(state))
        next_rng = random.Random()
        next_rng.setstate(saved)
        next_weights, next_details = probabilities(LENGTH_TASKS, restored, LENGTH_SETTINGS, "learning_progress")
        actual = _assign(LENGTH_TASKS, list(range(80)), next_rng, {}, "learning_progress", next_weights)
        self.assertEqual(actual, expected)
        self.assertEqual(next_details, details)
        self.assertEqual(next_rng.getstate(), rng.getstate())

    def test_explicit_config_requires_course_parameters_without_changing_legacy_schema(self):
        config = explicit_fixture()
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
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "config.json"
                path.write_text(json.dumps(explicit_fixture()))
                with self.assertRaisesRegex(RuntimeError, "source is stale"):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
