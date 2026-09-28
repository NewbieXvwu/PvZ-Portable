"""Focused regression tests for simulator-search ordering, budgets, transpositions, and value semantics."""

from __future__ import annotations

import inspect
import unittest

from pvz_env import ENV_PROTOCOL_VERSION, REPLAY_FORMAT_VERSION
from pvz_search import SearchTeacher, _RootState, _SearchNode
from pvz_search_candidates import CandidateGenerator, fit_action_to_remaining
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
            branches = []
            for index, token in enumerate(fields[3:]):
                snapshot = self.next_snapshot
                self.next_snapshot += 1
                ticks = int(token.split(":", 1)[1]) if token.startswith("W:") else 0
                observation = _candidate_observation()
                observation["tick"] = ticks
                branches.append({
                    "ok": True,
                    "observation": observation,
                    "events": {},
                    "ticks_advanced": ticks,
                    "snapshot_id": snapshot,
                    "state_hash": f"state-{snapshot}-{ticks}-{index}",
                })
            return {"ok": True, "branches": branches}
        raise AssertionError(command)


def _candidate_observation() -> dict:
    cells = [{"row": row, "row_type": 1} for row in range(5)]
    placements = [{"packet": packet, "row": row, "col": col}
                  for packet in range(3) for row, col in ((0, 1), (1, 3), (2, 5), (3, 7))]
    return {
        "terminal": False,
        "result": 0,
        "tick": 0,
        "wave": 1,
        "wave_count": 10,
        "sun": 100,
        "cells": cells,
        "plants": [],
        "projectiles": [],
        "packets": [],
        "zombie_count_multiplier": 1.0,
        "night": False,
        "pool": False,
        "fog": False,
        "roof": False,
        "zombies": [{"row": 2, "x": 420.0, "body_health": 200, "helm_health": 0,
                     "shield_health": 0, "is_eating": False}],
        "defenses": [{"row": row, "state": 1} for row in range(5)],
        "legal_actions": {"plants": placements, "shovels": [], "wait": True},
    }


class SearchTeacherTests(unittest.TestCase):
    def test_protocol_and_replay_versions_are_breaking(self) -> None:
        self.assertEqual(ENV_PROTOCOL_VERSION, 2)
        self.assertEqual(REPLAY_FORMAT_VERSION, 4)

    def test_search_teacher_is_student_independent(self) -> None:
        parameters = inspect.signature(SearchTeacher).parameters
        self.assertIn("value_model", parameters)
        self.assertNotIn("model", parameters)
        self.assertNotIn("max_decisions", parameters)
        self.assertNotIn("hidden", inspect.signature(SearchTeacher.advice).parameters)

    def test_candidate_generator_has_no_model_proposal_api(self) -> None:
        self.assertFalse(hasattr(CandidateGenerator(), "model_proposals"))

    def test_terminal_outcome_has_lexical_priority(self) -> None:
        results = [({"type": "wait", "ticks": 60}, 9.0, None, 900),
                   ({"type": "wait", "ticks": 150}, -3.0, 1, 700),
                   ({"type": "wait", "ticks": 300}, 100.0, 0, 400)]
        ordered = sorted(results, key=lambda item: SearchTeacher._result_key(item[2], item[1]), reverse=True)
        self.assertEqual([row[2] for row in ordered], [1, None, 0])

    def test_policy_only_assigns_mass_to_best_outcome_class(self) -> None:
        results = [({"type": "wait", "ticks": 60}, -2.0, 1, 500),
                   ({"type": "wait", "ticks": 150}, -3.0, 1, 500),
                   ({"type": "wait", "ticks": 300}, 50.0, None, 900)]
        policy = SearchTeacher._policy_from_results(results)
        self.assertAlmostEqual(sum(policy), 1.0)
        self.assertGreater(policy[0], policy[1])
        self.assertEqual(policy[2], 0.0)

    def test_temporal_actions_are_capped_to_remaining_horizon(self) -> None:
        self.assertEqual(fit_action_to_remaining({"type": "wait", "ticks": 300}, 90),
                         {"type": "wait", "ticks": 90})
        self.assertEqual(fit_action_to_remaining({"type": "wait_decision", "max_ticks": 900}, 47),
                         {"type": "wait_decision", "max_ticks": 47})

    def test_structural_root_candidates_cover_every_legal_packet(self) -> None:
        actions = CandidateGenerator().actions(_candidate_observation(), 32, True, 900)
        self.assertEqual({action["packet"] for action in actions if action["type"] == "plant"}, {0, 1, 2})

    def test_discounted_terminal_value_matches_tick_semantics(self) -> None:
        self.assertEqual(discounted_terminal_value(True, 0), 1.0)
        self.assertEqual(discounted_terminal_value(False, 0), -1.0)
        self.assertAlmostEqual(discounted_terminal_value(True, 300), VALUE_GAMMA)

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

    def test_branch_protocol_requires_state_hash(self) -> None:
        env = _FastProtocolEnv()
        searcher = SearchTeacher(env)
        root_obs = _candidate_observation()
        root = _SearchNode(root_obs, 99, "root", 0, 0, 0.0, 1.0, 0.0, ("root",))
        [(child, _)] = searcher._expand_actions(root, [{"type": "wait", "ticks": 60}])
        self.assertIsNotNone(child)
        assert child is not None
        self.assertTrue(child.state_hash.startswith("state-"))

    def test_transposition_table_rejects_worse_duplicate(self) -> None:
        env = _FastProtocolEnv()
        searcher = SearchTeacher(env)
        searcher._snapshots = {1, 2}
        observation = _candidate_observation()
        root = _RootState(action={"type": "wait", "ticks": 60})
        first = _SearchNode(observation, 1, "same", 0, 60, 0.0, 1.0, 0.8, ("wait", 1))
        second = _SearchNode(observation, 2, "same", 0, 60, 0.0, 1.0, 0.5, ("wait", 1))
        self.assertTrue(searcher._accept_node(root, first))
        self.assertFalse(searcher._accept_node(root, second))
        self.assertIn("DROP_SNAPSHOT_FAST 2", env.commands)

    def test_global_simulation_budget_bounds_entire_decision(self) -> None:
        env = _FastProtocolEnv()
        searcher = SearchTeacher(env, simulation_budget=24, candidate_limit=2)
        advice = searcher.advice(_candidate_observation())
        self.assertLessEqual(advice.simulation_count, 24)
        self.assertGreater(advice.simulation_count, 0)

    def test_root_candidate_limit_reserves_budget_for_depth(self) -> None:
        searcher = SearchTeacher(_FastProtocolEnv(), simulation_budget=64, candidate_limit=8)
        self.assertLessEqual(searcher.root_candidate_limit, 16)


if __name__ == "__main__":
    unittest.main()
