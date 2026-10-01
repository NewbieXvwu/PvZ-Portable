"""Evidence-integrity checks for the corrected-native reevaluation protocol."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from research_packet_cost_reevaluation import paired_changes, validate_curve


class ReevaluationIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [{"task_id": name, "seeds": [101, 102], "evaluation_role": "validation",
                       "terrain": "day", "wave_cap": cap, "zombie_count_multiplier": 1.}
                      for name, cap in (("normal", 1), ("conveyor", 3))]
        self.modes = ["greedy", "sampled"]
        self.before = {"seed_results": {mode: {t["task_id"]: [
            {"seed": seed, "won": False, "result": -1, "truncated": False,
             "terminal_wave": 1, "terminal_tick": 10, "actions": 2}
            for seed in t["seeds"]] for t in self.tasks} for mode in self.modes}}

    def changes(self, after):
        return paired_changes(self.before, after, self.tasks, self.modes, ["conveyor"])

    def test_full_normal_record_change_is_unexpected_even_without_win_change(self):
        after = copy.deepcopy(self.before)
        after["seed_results"]["sampled"]["normal"][0]["actions"] += 1
        summary, details = self.changes(after)
        self.assertEqual(summary["unexpected_changed_rows"], 1)
        self.assertEqual(summary["win_label_changes"], 0)
        self.assertEqual(details[0]["seed"], 101)

    def test_missing_key_is_detected_even_when_new_value_is_none(self):
        after = copy.deepcopy(self.before)
        after["seed_results"]["greedy"]["normal"][0]["new_field"] = None
        self.assertEqual(self.changes(after)[0]["unexpected_changed_rows"], 1)

    def test_conveyor_win_change_keeps_both_original_and_corrected_rows(self):
        after = copy.deepcopy(self.before)
        after["seed_results"]["greedy"]["conveyor"][0].update(won=True, result=1)
        summary, details = self.changes(after)
        self.assertEqual(summary["conveyor_changed_rows"], 1)
        self.assertEqual(summary["unexpected_changed_rows"], 0)
        self.assertEqual(summary["win_label_changes"], 1)
        self.assertFalse(details[0]["before"]["won"])
        self.assertTrue(details[0]["after"]["won"])

    def test_cannot_drop_failed_seed(self):
        after = copy.deepcopy(self.before)
        after["seed_results"]["greedy"]["normal"].pop()
        with self.assertRaises(ValueError):
            self.changes(after)

    def test_cannot_substitute_or_duplicate_seed(self):
        for replacement in (101, 999):
            after = copy.deepcopy(self.before)
            after["seed_results"]["sampled"]["conveyor"][1]["seed"] = replacement
            with self.assertRaises(ValueError):
                self.changes(after)

    def test_cannot_drop_whole_mode_or_task(self):
        for drop_mode in (True, False):
            after = copy.deepcopy(self.before)
            if drop_mode:
                del after["seed_results"]["sampled"]
            else:
                del after["seed_results"]["greedy"]["normal"]
            with self.assertRaises(ValueError):
                self.changes(after)

    def test_inconsistent_or_truncated_win_is_not_counted(self):
        for mutation in ({"won": True}, {"won": True, "result": 1, "truncated": True}):
            after = copy.deepcopy(self.before)
            after["seed_results"]["sampled"]["normal"][0].update(mutation)
            with self.assertRaises(ValueError):
                self.changes(after)

    def curve_fixture(self):
        points = [{"counters": {"decisions": n}, "updates": i}
                  for i, n in enumerate((0, 125010, 250020, 375030, 500040))]
        return {"status": "budget_complete", "counters": points[-1]["counters"], "updates": 4,
                "learning_curve": points}, {"budget": {"decisions": 500000},
                "evaluation": {"decision_nodes": [125000, 250000, 375000, 500000]}}

    def test_actual_threshold_overshoot_is_allowed(self):
        state, config = self.curve_fixture()
        self.assertEqual(len(validate_curve(state, config)), 5)

    def test_missing_zero_baseline_or_middle_node_is_rejected(self):
        for index in (0, 2):
            state, config = self.curve_fixture()
            state["learning_curve"].pop(index)
            with self.assertRaises(ValueError):
                validate_curve(state, config)

    def test_budget_status_alone_cannot_hide_early_stop_or_reordered_updates(self):
        for mutation in ("early", "reorder"):
            state, config = self.curve_fixture()
            if mutation == "early":
                state["counters"]["decisions"] = 400000
            else:
                state["learning_curve"][2]["updates"] = 1
            with self.assertRaises(ValueError):
                validate_curve(state, config)


if __name__ == "__main__":
    unittest.main()
