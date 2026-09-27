"""Focused regression tests for simulator-search ordering, coverage, budgets, batching, and value semantics."""

from __future__ import annotations

import inspect
import unittest

from pvz_search import SearchTeacher, _SearchNode
from pvz_search_candidates import fit_action_to_remaining
from pvz_value import VALUE_GAMMA, discounted_terminal_value


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
        if command.startswith("BRANCH_SNAPSHOT_FAST "):
            fields = command.split()
            count = int(fields[2])
            self.assert_command_shape = len(fields[3:]) == count
            branches = []
            for token in fields[3:]:
                snapshot = self.next_snapshot
                self.next_snapshot += 1
                ticks = int(token.split(":", 1)[1]) if token.startswith("W:") else 0
                observation = _candidate_observation()
                observation["tick"] = ticks
                branches.append({"ok": True, "observation": observation, "events": {},
                                 "ticks_advanced": ticks, "snapshot_id": snapshot})
            return {"ok": True, "branches": branches}
        raise AssertionError(command)


def _candidate_observation() -> dict:
    cells = [{"row": row, "row_type": 1} for row in range(5)]
    placements = [{"packet": packet, "row": row, "col": col}
                  for packet in range(3) for row, col in ((0, 1), (1, 3), (2, 5), (3, 7))]
    return {
        "terminal": False, "result": 0, "tick": 0, "wave": 1, "wave_count": 10, "sun": 100,
        "cells": cells, "plants": [],
        "zombies": [{"row": 2, "x": 420.0, "body_health": 200, "helm_health": 0, "shield_health": 0}],
        "defenses": [{"row": row, "state": 1} for row in range(5)],
        "legal_actions": {"plants": placements, "shovels": [], "wait": True},
    }


class SearchTeacherTests(unittest.TestCase):
    def test_search_horizon_cannot_change_root_wait_semantics(self) -> None:
        with self.assertRaises(ValueError):
            SearchTeacher(_FastProtocolEnv(), horizon_ticks=899)

    def test_search_api_separates_time_and_compute_budgets(self) -> None:
        parameters = inspect.signature(SearchTeacher).parameters
        self.assertIn("horizon_ticks", parameters)
        self.assertIn("simulation_budget", parameters)
        self.assertIn("max_same_tick_actions", parameters)
        self.assertNotIn("max_decisions", parameters)

    def test_terminal_outcome_has_lexical_priority(self) -> None:
        results = [({"type": "wait", "ticks": 60}, 9.0, None, 900),
                   ({"type": "wait", "ticks": 150}, -3.0, 1, 700),
                   ({"type": "wait", "ticks": 300}, 100.0, 0, 400)]
        ordered = sorted(results, key=lambda item: SearchTeacher._result_key(item[2], item[1]), reverse=True)
        self.assertEqual(ordered[0][2], 1)
        self.assertIsNone(ordered[1][2])
        self.assertEqual(ordered[2][2], 0)

    def test_nonterminal_partial_beats_known_loss(self) -> None:
        self.assertGreater(SearchTeacher._result_key(None, -100.0), SearchTeacher._result_key(0, 100.0))

    def test_policy_only_assigns_mass_to_best_outcome_class(self) -> None:
        results = [({"type": "wait", "ticks": 60}, -2.0, 1, 500),
                   ({"type": "wait", "ticks": 150}, -3.0, 1, 500),
                   ({"type": "wait", "ticks": 300}, 50.0, None, 900),
                   ({"type": "wait_decision", "max_ticks": 900}, 80.0, 0, 900)]
        policy = SearchTeacher._policy_from_results(results)
        self.assertAlmostEqual(sum(policy), 1.0)
        self.assertGreater(policy[0], policy[1])
        self.assertEqual(policy[2], 0.0)
        self.assertEqual(policy[3], 0.0)

    def test_temporal_actions_are_capped_to_remaining_horizon(self) -> None:
        self.assertEqual(fit_action_to_remaining({"type": "wait", "ticks": 300}, 90), {"type": "wait", "ticks": 90})
        self.assertEqual(fit_action_to_remaining({"type": "wait_decision", "max_ticks": 900}, 47),
                         {"type": "wait_decision", "max_ticks": 47})
        self.assertIsNone(fit_action_to_remaining({"type": "wait", "ticks": 60}, 0))

    def test_structural_root_candidates_cover_every_legal_packet(self) -> None:
        searcher = SearchTeacher(_FastProtocolEnv())
        actions = searcher.candidate_generator.actions(_candidate_observation(), None, 32, True, 900)
        self.assertEqual({action["packet"] for action in actions if action["type"] == "plant"}, {0, 1, 2})

    def test_internal_candidates_reserve_space_for_model_proposals(self) -> None:
        searcher = SearchTeacher(_FastProtocolEnv(), candidate_limit=8)
        model_action = {"type": "plant", "packet": 2, "row": 4, "col": 8}
        searcher.candidate_generator.model_proposals = lambda observation, output, limit: [model_action]
        actions = searcher.candidate_generator.actions(_candidate_observation(), {}, 8, False, 900)
        self.assertIn(model_action, actions)

    def test_discounted_terminal_value_matches_tick_semantics(self) -> None:
        self.assertEqual(discounted_terminal_value(True, 0), 1.0)
        self.assertEqual(discounted_terminal_value(False, 0), -1.0)
        self.assertAlmostEqual(discounted_terminal_value(True, 300), VALUE_GAMMA)
        self.assertAlmostEqual(discounted_terminal_value(False, 600), -(VALUE_GAMMA ** 2))
        self.assertLess(abs(discounted_terminal_value(True, 900)), 1.0)

    def test_fast_speculation_restores_episode_bookkeeping(self) -> None:
        env = _FastProtocolEnv()
        searcher = SearchTeacher(env)
        original_final = env.episode["final_state"]
        with searcher._speculative_fast() as snapshot:
            self.assertEqual(snapshot, 1)
            env.episode["operations"].append({"kind": "speculative"})
            env.episode["final_state"] = {"tick": 999}
        self.assertEqual(env.episode["operations"], [{"kind": "real"}])
        self.assertIs(env.episode["final_state"], original_final)
        self.assertEqual(env.commands, ["SNAPSHOT_FAST", "RESTORE_FAST 1", "DROP_SNAPSHOT_FAST 1"])

    def test_branch_protocol_batches_sibling_actions(self) -> None:
        env = _FastProtocolEnv()
        searcher = SearchTeacher(env)
        branches = searcher._branch_snapshot_fast(7, [{"type": "wait", "ticks": 60},
                                                       {"type": "wait", "ticks": 150},
                                                       {"type": "shovel", "col": 2, "row": 3}])
        self.assertEqual(len(branches), 3)
        self.assertEqual(env.commands, ["BRANCH_SNAPSHOT_FAST 7 3 W:60 W:150 S:2:3"])
        self.assertTrue(env.assert_command_shape)

    def test_simulation_budget_bounds_transition_expansion(self) -> None:
        env = _FastProtocolEnv()
        searcher = SearchTeacher(env, simulation_budget=5, candidate_limit=2)
        searcher.candidate_generator.actions = lambda *args, **kwargs: [
            {"type": "wait", "ticks": 60}, {"type": "wait", "ticks": 150}]
        observation = _candidate_observation()
        root = _SearchNode(observation, 99, None, 0, 0, 0.0, 1.0, searcher._leaf_value(observation), ("root",))
        [(child, leaf)] = searcher._expand_actions(root, [{"type": "wait", "ticks": 60}])
        result = searcher._search_branch(child, leaf)
        self.assertEqual(result[2], 5)

    def test_prune_preserves_distinct_branch_groups_before_filling(self) -> None:
        env = _FastProtocolEnv()
        searcher = SearchTeacher(env, beam_width=3)
        searcher._snapshots = {1, 2, 3, 4}
        observation = _candidate_observation()
        nodes = [_SearchNode(observation, 1, None, 0, 300, 0.0, 1.0, 1.0, ("plant", 0, 2)),
                 _SearchNode(observation, 2, None, 0, 300, 0.0, 1.0, 0.9, ("plant", 0, 2)),
                 _SearchNode(observation, 3, None, 0, 300, 0.0, 1.0, 0.8, ("wait", 1)),
                 _SearchNode(observation, 4, None, 0, 300, 0.0, 1.0, 0.7, ("shovel", 3))]
        kept = searcher._prune(nodes)
        self.assertEqual({node.diversity_key for node in kept}, {("plant", 0, 2), ("wait", 1), ("shovel", 3)})
        self.assertIn("DROP_SNAPSHOT_FAST 2", env.commands)


if __name__ == "__main__":
    unittest.main()
