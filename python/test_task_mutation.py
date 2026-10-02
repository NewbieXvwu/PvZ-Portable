"""Task identity, coverage and interruption continuation for append-only mutation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import pvz_curriculum as course
import pvz_task_mutation as mutation
from pvz_research import ROOT, _assign, load_config


class TaskMutationTests(unittest.TestCase):
    def setUp(self):
        self.config, self.base, self.evaluation = load_config(
            ROOT / "experiments/t5/periodic_mutation_v1/seed0.json")
        self.settings = self.config["sampling"]["task_mutation"]
        self.course_settings = self.config["sampling"]["curriculum"]
        self.state = mutation.initial_state(self.base)
        self.history = course.initial_state(self.base)

    def advance(self, decisions):
        additions = mutation.advance(self.state, self.settings, self.base, self.evaluation,
                                     self.history, self.course_settings, decisions)
        if additions:
            course.append_tasks(self.history, additions)
        mutation.validate_state(self.state, self.settings, self.base, self.evaluation)
        course.validate_state(self.history, self.state["pool"], self.course_settings)
        return additions

    def test_schedule_preserves_originals_and_caps_without_repeating_rounds(self):
        before = copy.deepcopy(self.base)
        self.assertEqual(self.advance(24999), [])
        additions = self.advance(25000)
        self.assertEqual(len(additions), 5)
        self.assertEqual(self.advance(25000), [])
        self.assertEqual(self.base, before)
        self.assertEqual(self.state["pool"][:30], before)
        for row in self.state["events"][0]["additions"]:
            parent = next(task for task in self.base if task["task_id"] == row["parent_task_id"])
            child = next(task for task in additions if task["task_id"] == row["task_id"])
            changed = [key for key in parent if parent[key] != child[key]]
            self.assertEqual(set(changed), {"task_id", "seeds", row["axis"]})

    def test_rounds_stop_at_declared_limit_and_keep_failed_tasks(self):
        failed = self.base[0]["task_id"]
        course.observe(self.history, [dict(task_id=failed, terminated=True, truncated=False, won=False)] * 8,
                       self.course_settings)
        self.advance(500000)
        self.assertEqual(self.state["rounds"], 10)
        self.assertLessEqual(len(self.state["pool"]), 80)
        self.assertEqual(self.history["history"][failed], [False] * 8)
        self.assertEqual(self.advance(1000000), [])
        weights, _ = course.probabilities(self.state["pool"], self.history, self.course_settings, "frontier_v1")
        self.assertTrue(all(x > 0 for x in weights))

    def test_generated_seeds_and_definitions_never_overlap_frozen_evaluation(self):
        self.advance(250000)
        originals = {seed for task in self.base + self.evaluation for seed in task["seeds"]}
        generated = [seed for task in self.state["pool"][30:] for seed in task["seeds"]]
        self.assertEqual(len(generated), len(set(generated)))
        self.assertFalse(originals & set(generated))
        definitions = {mutation.signature(task) for task in self.evaluation}
        self.assertFalse(definitions & {mutation.signature(task) for task in self.state["pool"][30:]})

    def test_existing_history_and_coverage_survive_extension(self):
        task = self.base[0]
        course.observe(self.history, [dict(task_id=task["task_id"], terminated=True, truncated=False, won=x)
                                      for x in [False, True] * 4], self.course_settings)
        previous = copy.deepcopy(self.history)
        additions = self.advance(25000)
        for field in ("history", "completed", "ignored_truncations", "task_metadata"):
            for key, value in previous[field].items():
                self.assertEqual(self.history[field][key], value)
        weights, details = course.probabilities(self.state["pool"], self.history, self.course_settings, "frontier_v1")
        for child in additions:
            self.assertEqual(self.history["completed"][child["task_id"]], 0)
        for task, weight in zip(self.state["pool"], weights):
            self.assertGreaterEqual(weight, .25 * details["tasks"][task["task_id"]]["coverage_base_probability"])
        self.assertEqual(self.state["events"][0]["additions"][0]["parent_task_id"], self.base[0]["task_id"])

    def test_disk_restore_recreates_next_round_and_collection_assignments(self):
        self.advance(25000)
        rng = random.Random(71)
        weights, _ = course.probabilities(self.state["pool"], self.history, self.course_settings, "frontier_v1")
        _assign(self.state["pool"], [0, 1], rng, {}, "frontier_v1", weights)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mutation.json"
            path.write_text(json.dumps({"mutation": self.state, "curriculum": self.history}))
            saved_rng = rng.getstate()
            self.advance(50000)
            expected_pool, expected_course = copy.deepcopy(self.state), copy.deepcopy(self.history)
            weights, details = course.probabilities(self.state["pool"], self.history, self.course_settings, "frontier_v1")
            expected = _assign(self.state["pool"], list(range(2, 102)), rng, {}, "frontier_v1", weights)
            restored = json.loads(path.read_text())
            self.state, self.history = restored["mutation"], restored["curriculum"]
            next_rng = random.Random(); next_rng.setstate(saved_rng)
            self.advance(50000)
            self.assertEqual(self.state, expected_pool)
            self.assertEqual(self.history, expected_course)
            weights, actual_details = course.probabilities(self.state["pool"], self.history, self.course_settings, "frontier_v1")
            actual = _assign(self.state["pool"], list(range(2, 102)), next_rng, {}, "frontier_v1", weights)
            self.assertEqual(actual, expected)
            self.assertEqual(actual_details, details)
            self.assertEqual(next_rng.getstate(), rng.getstate())

    def test_state_cannot_change_old_tasks_or_extra_child_axes(self):
        self.advance(25000)
        for index in (0, 30):
            corrupt = copy.deepcopy(self.state)
            corrupt["pool"][index]["sun_start"] = 500
            with self.assertRaises(ValueError):
                mutation.validate_state(corrupt, self.settings, self.base, self.evaluation)

    def test_invalid_seed_blocks_or_implicit_bounds_rejected(self):
        for patch in ({"seed_start": 60000}, {"multipliers": [.5]}, {"interval_decisions": 0},
                      {"seeds_per_task": 1}, {"levels": {"day": [7]}}, {"max_rounds": True}):
            with self.assertRaises(ValueError):
                mutation.validate_settings({**self.settings, **patch}, self.base, self.evaluation)
        settings = copy.deepcopy(self.settings); del settings["generator_seed"]
        with self.assertRaises(ValueError):
            mutation.validate_settings(settings, self.base, self.evaluation)

    def test_config_requires_declared_frontier_and_preserves_old_config(self):
        original, tasks, _ = load_config(ROOT / "experiments/t5/frontier_v1/seed0.json")
        self.assertNotIn("task_mutation", original["sampling"])
        self.assertEqual(tasks, self.base)
        config = copy.deepcopy(self.config)
        config["sampling"]["method"] = "learning_progress"
        del config["sampling"]["curriculum"]["frontier_pass_range"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"; path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "requires frontier_v1"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
