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
    def test_initial_focus_retains_all_tasks_and_ignores_truncated_episodes(self):
        settings = {**SETTINGS, "frontier_pass_range": [.2, .8],
                    "initial_focus_task_ids": ["progress"], "initial_focus_completed_episodes": 4}
        state = initial_state(TASKS)
        observe(state, [dict(task_id="progress", terminated=False, truncated=True, won=False)], settings)
        weights, detail = probabilities(TASKS, state, settings, "frontier_v1")
        self.assertEqual(detail["initial_focus"]["completed"], 0)
        self.assertTrue(detail["initial_focus"]["active"])
        self.assertAlmostEqual(weights[1], .75 + .25 / 6)
        for weight, row in zip(weights, detail["tasks"].values()):
            self.assertGreaterEqual(weight, .25 * row["coverage_base_probability"])
        self.assertEqual(set(state["history"]), {t["task_id"] for t in TASKS})

    def test_initial_focus_exit_and_assignment_rng_resume_at_completed_boundary(self):
        settings = {**SETTINGS, "frontier_pass_range": [.2, .8],
                    "initial_focus_task_ids": ["progress"], "initial_focus_completed_episodes": 4}
        state = initial_state(TASKS)
        observe(state, outcomes("progress", [True, False, True]), settings)
        restored = json.loads(json.dumps(state));rng = random.Random(71);restored_rng = random.Random(71)
        for current, next_rng in ((state, rng), (restored, restored_rng)):
            weights, _ = probabilities(TASKS, current, settings, "frontier_v1")
            drawn = _assign(TASKS, list(range(30)), next_rng, {}, "frontier_v1", weights)
            if current is state:expected = drawn
            else:self.assertEqual(drawn, expected)
            observe(current, outcomes("progress", [False]), settings)
        weights, detail = probabilities(TASKS, restored, settings, "frontier_v1")
        self.assertFalse(detail["initial_focus"]["active"])
        ordinary = {k:v for k,v in settings.items() if not k.startswith("initial_focus")}
        reference, _ = probabilities(TASKS, state, ordinary, "frontier_v1")
        self.assertEqual(weights, reference)
        self.assertEqual(rng.getstate(), restored_rng.getstate())

    def test_initial_focus_cannot_name_validation_or_use_implicit_limits(self):
        settings = {**SETTINGS, "frontier_pass_range": [.2, .8],
                    "initial_focus_task_ids": ["validation"], "initial_focus_completed_episodes": 4}
        with self.assertRaisesRegex(ValueError, "outside the frozen training pool"):
            probabilities(TASKS, initial_state(TASKS), settings, "frontier_v1")
        for changes in ({"initial_focus_completed_episodes": 0}, {"initial_focus_task_ids": []},
                        {"initial_focus_task_ids": ["progress", "progress"]}):
            with self.assertRaises(ValueError):validate_settings({**settings, **changes}, "frontier_v1")

    def test_frontier_prioritizes_stable_edge_across_singleton_groups(self):
        tasks = [task for task in LENGTH_TASKS if task["task_id"].endswith("_0")]
        settings = {**LENGTH_SETTINGS, "frontier_pass_range": [.2, .8]}
        state = initial_state(tasks)
        for index, task in enumerate(tasks):
            values = [False, True] * 16 if index == 0 else [bool(index % 2)] * 32
            observe(state, outcomes(task["task_id"], values), settings)
        weights, details = probabilities(tasks, state, settings, "frontier_v1")
        self.assertGreater(weights[0], .75)
        self.assertEqual(details["tasks"][tasks[0]["task_id"]]["positive_progress"], 0)
        self.assertEqual(details["tasks"][tasks[0]["task_id"]]["frontier_pass_rate"], .5)
        for task, weight in zip(tasks, weights):
            row = details["tasks"][task["task_id"]]
            self.assertGreaterEqual(weight, .25 * row["coverage_base_probability"])
        self.assertAlmostEqual(sum(weights), 1)
        self.assertGreater(details["terrain_probabilities"]["day"], .5)

    def test_frontier_empty_or_zero_win_pool_falls_back_to_balanced_coverage(self):
        settings = {**LENGTH_SETTINGS, "frontier_pass_range": [.2, .8]}
        state = initial_state(LENGTH_TASKS)
        for filled in (False, True):
            if filled:
                for task in LENGTH_TASKS:
                    observe(state, outcomes(task["task_id"], [False] * 32), settings)
            weights, _ = probabilities(LENGTH_TASKS, state, settings, "frontier_v1")
            expected, _ = probabilities(LENGTH_TASKS, state, LENGTH_SETTINGS, "terrain_balanced")
            self.assertEqual(weights, expected)

    def test_frontier_band_is_inclusive_and_never_uses_truncations(self):
        tasks = [dict(task_id=str(i), terrain="day", seeds=[0]) for i in range(4)]
        settings = dict(window_episodes=5, minimum_window_episodes=5,
                        uniform_fraction=.25, coverage="terrain", frontier_pass_range=[.2, .8])
        state = initial_state(tasks)
        for i, wins in enumerate((1, 4, 0, 5)):
            observe(state, outcomes(str(i), [True] * wins + [False] * (5-wins)), settings)
        observe(state, [dict(task_id="2", terminated=False, truncated=True, won=False)], settings)
        weights, details = probabilities(tasks, state, settings, "frontier_v1")
        self.assertEqual(weights, [.4375, .4375, .0625, .0625])
        self.assertEqual(details["tasks"]["2"]["ignored_truncations"], 1)
        self.assertEqual(state["completed"]["2"], 5)

    def test_frontier_resume_keeps_next_assignments_and_original_seeds(self):
        settings = {**SETTINGS, "frontier_pass_range": [.2, .8]}
        state = initial_state(TASKS)
        observe(state, outcomes("progress", [True, False] * 16), settings)
        weights, details = probabilities(TASKS, state, settings, "frontier_v1")
        rng = random.Random(17); saved_rng = rng.getstate()
        expected = _assign(TASKS, list(range(40)), rng, {}, "frontier_v1", weights)
        restored = json.loads(json.dumps(state)); next_rng = random.Random(); next_rng.setstate(saved_rng)
        next_weights, next_details = probabilities(TASKS, restored, settings, "frontier_v1")
        actual = _assign(TASKS, list(range(40)), next_rng, {}, "frontier_v1", next_weights)
        self.assertEqual(actual, expected)
        self.assertEqual(next_details, details)
        self.assertEqual(next_rng.getstate(), rng.getstate())

    def test_frontier_settings_cannot_be_implicit_or_attached_to_old_methods(self):
        with self.assertRaises(ValueError):
            validate_settings(SETTINGS, "frontier_v1")
        settings = {**SETTINGS, "frontier_pass_range": [.2, .8]}
        with self.assertRaises(ValueError):
            validate_settings(settings, "learning_progress")
        for band in ([0, .8], [.8, .2], [True, .8], [.2, float("nan")]):
            with self.assertRaises(ValueError):
                validate_settings({**settings, "frontier_pass_range": band}, "frontier_v1")

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
