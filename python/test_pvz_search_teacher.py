"""Focused regression tests for search-teacher ordering, horizons, and value semantics."""

from __future__ import annotations

import math
import unittest

from pvz_agent import SearchTeacher, VALUE_GAMMA, discounted_terminal_value


class _FastProtocolEnv:
    def __init__(self) -> None:
        self._reset_done = True
        self.episode = {"operations": [{"kind": "real"}], "final_state": {"tick": 10}}
        self.commands: list[str] = []
        self.next_snapshot = 1

    def _command(self, command: str) -> dict:
        self.commands.append(command)
        if command == "SNAPSHOT_FAST":
            snapshot = self.next_snapshot
            self.next_snapshot += 1
            return {"ok": True, "snapshot_id": snapshot}
        if command.startswith("RESTORE_FAST ") or command.startswith("DROP_SNAPSHOT_FAST "):
            return {"ok": True}
        raise AssertionError(command)


class SearchTeacherTests(unittest.TestCase):
    def test_search_horizon_cannot_change_root_wait_semantics(self) -> None:
        with self.assertRaises(ValueError):
            SearchTeacher(_FastProtocolEnv(), horizon_ticks=899)

    def test_terminal_outcome_has_lexical_priority(self) -> None:
        results = [
            ({"type": "wait", "ticks": 60}, 9.0, None, 4, 900),
            ({"type": "wait", "ticks": 150}, -3.0, 1, 5, 700),
            ({"type": "wait", "ticks": 300}, 100.0, 0, 2, 400),
        ]
        ordered = sorted(results, key=lambda item: SearchTeacher._result_key(item[2], item[1]), reverse=True)
        self.assertEqual(ordered[0][2], 1)
        self.assertIsNone(ordered[1][2])
        self.assertEqual(ordered[2][2], 0)

    def test_policy_only_assigns_mass_to_best_outcome_class(self) -> None:
        results = [
            ({"type": "wait", "ticks": 60}, -2.0, 1, 3, 500),
            ({"type": "wait", "ticks": 150}, -3.0, 1, 3, 500),
            ({"type": "wait", "ticks": 300}, 50.0, None, 3, 900),
            ({"type": "wait_decision", "max_ticks": 900}, 80.0, 0, 3, 900),
        ]
        policy = SearchTeacher._policy_from_results(results)
        self.assertAlmostEqual(sum(policy), 1.0)
        self.assertGreater(policy[0], policy[1])
        self.assertEqual(policy[2], 0.0)
        self.assertEqual(policy[3], 0.0)

    def test_temporal_actions_are_capped_to_remaining_horizon(self) -> None:
        self.assertEqual(
            SearchTeacher._fit_action_to_remaining({"type": "wait", "ticks": 300}, 90),
            {"type": "wait", "ticks": 90},
        )
        self.assertEqual(
            SearchTeacher._fit_action_to_remaining({"type": "wait_decision", "max_ticks": 900}, 47),
            {"type": "wait_decision", "max_ticks": 47},
        )
        self.assertIsNone(SearchTeacher._fit_action_to_remaining({"type": "wait", "ticks": 60}, 0))

    def test_discounted_terminal_value_matches_tick_semantics(self) -> None:
        self.assertEqual(discounted_terminal_value(True, 0), 1.0)
        self.assertEqual(discounted_terminal_value(False, 0), -1.0)
        self.assertAlmostEqual(discounted_terminal_value(True, 300), VALUE_GAMMA)
        self.assertAlmostEqual(discounted_terminal_value(False, 600), -(VALUE_GAMMA ** 2))
        self.assertLess(abs(discounted_terminal_value(True, 900)), 1.0)

    def test_fast_speculation_restores_episode_bookkeeping(self) -> None:
        env = _FastProtocolEnv()
        teacher = SearchTeacher(env)
        original_final = env.episode["final_state"]
        with teacher._speculative_fast() as snapshot:
            self.assertEqual(snapshot, 1)
            env.episode["operations"].append({"kind": "speculative"})
            env.episode["final_state"] = {"tick": 999}
        self.assertEqual(env.episode["operations"], [{"kind": "real"}])
        self.assertIs(env.episode["final_state"], original_final)
        self.assertEqual(env.commands, ["SNAPSHOT_FAST", "RESTORE_FAST 1", "DROP_SNAPSHOT_FAST 1"])


if __name__ == "__main__":
    unittest.main()
