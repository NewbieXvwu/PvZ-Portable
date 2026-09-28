"""Regression tests for simulator-search ordering, budgets, transpositions, and value semantics.

The fake simulator in this module is deliberately *not* a stub that echoes whatever
the search asks for: it reproduces the three protocol details that the search's
correctness depends on (absolute ticks, content-derived state hashes, and terminal
branches), so a broken search cannot make these tests pass by construction.
"""

from __future__ import annotations

import unittest
from typing import Any, cast

from pvz_agent_model import WAIT_TICKS
from pvz_search import BRANCH_BATCH_LIMIT, SearchTeacher, _RootState, _SearchNode
from pvz_search_candidates import CandidateGenerator, fit_action_to_remaining, shovel_proposals
from pvz_search_value import SearchValueModel
from pvz_value import VALUE_GAMMA, discounted_terminal_value

ROOT_TICK = 120
PLANT_COST = 25
PACKETS = (0, 1, 2)
PLACEMENTS = ((0, 1), (1, 3), (2, 5), (3, 7))


def _board(tick: int = ROOT_TICK, sun: int = 200, plants: tuple[tuple[int, int, int], ...] = ()) -> dict:
    return {"tick": tick, "sun": sun, "plants": tuple(sorted(plants))}


def _state_hash(board: dict) -> str:
    """Hash of the visible board, so different action orders can collide on purpose."""
    return f"t{board['tick']}|s{board['sun']}|p{board['plants']}"


def _observation(board: dict, terminal: bool = False, result: int = 0) -> dict:
    occupied = {(row, col) for row, col, _ in board["plants"]}
    return {
        "terminal": terminal,
        "result": result,
        "tick": board["tick"],
        "wave": 1,
        "wave_count": 10,
        "sun": board["sun"],
        "cells": [{"row": row, "row_type": 1} for row in range(5)],
        "plants": [
            {"type": packet, "imitater_type": -1, "row": row, "col": col,
             "health": 100, "max_health": 100, "squished": False}
            for row, col, packet in board["plants"]
        ],
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
        "legal_actions": {
            "plants": [
                {"packet": packet, "row": row, "col": col}
                for packet in PACKETS for row, col in PLACEMENTS
                if (row, col) not in occupied
            ],
            "shovels": [[col, row] for row, col, _ in board["plants"]],
            "wait": True,
        },
    }


class _FakeSimulator:
    """Deterministic stand-in for the C++ ``*_FAST`` snapshot protocol.

    Reproduces the contract details the search relies on:

    * ``observation["tick"]`` is the **absolute** tick after the action, so a 60-tick
      wait issued at tick 120 reports tick 180. ``SearchTeacher`` derives
      ``elapsed_ticks`` from exactly this difference, so feeding the wait *duration*
      back as the tick (as a naive stub does) silently collapses every branch to a
      zero-tick advance.
    * ``state_hash`` is derived from the **visible board**, so two different action
      orders that reach the same board collide and the transposition table is really
      exercised. A monotonically numbered hash can never collide.
    * A branch that ends the level reports ``terminal=True`` and carries no
      ``snapshot_id``/``state_hash``, matching the non-terminal guard in
      ``SearchTeacher._branch_result``.
    """

    def __init__(self, terminal_tick: int | None = None, winning: bool = True) -> None:
        self._reset_done = True
        self.episode: dict = {"operations": [{"kind": "real"}], "final_state": {"tick": 10}}
        self.commands: list[str] = []
        self.issued: list[int] = []
        self.dropped: set[int] = set()
        self._next_snapshot = 1
        self._states: dict[int, dict] = {}
        self._terminal_tick = terminal_tick
        self._winning = winning

    def root_snapshot(self, **board_fields: object) -> int:
        """Register a board and return the snapshot id the real ``SNAPSHOT_FAST`` would."""
        return self._take_snapshot(_board(**board_fields))  # type: ignore[arg-type]

    def _take_snapshot(self, board: dict) -> int:
        snapshot_id = self._next_snapshot
        self._next_snapshot += 1
        self._states[snapshot_id] = board
        self.issued.append(snapshot_id)
        return snapshot_id

    def _command(self, command: str) -> dict:
        self.commands.append(command)
        if command == "SNAPSHOT_FAST":
            return {"ok": True, "snapshot_id": self._take_snapshot(_board())}
        if command.startswith("RESTORE_FAST "):
            return {"ok": True}
        if command.startswith("DROP_SNAPSHOT_FAST "):
            self.dropped.add(int(command.split()[1]))
            return {"ok": True}
        if command.startswith("BRANCH_SNAPSHOT_FAST "):
            fields = command.split()
            parent, count, tokens = int(fields[1]), int(fields[2]), fields[3:]
            if len(tokens) != count:
                raise AssertionError(f"branch count disagrees with the token list: {command}")
            return {"ok": True, "branches": [self._branch(parent, token) for token in tokens]}
        raise AssertionError(f"unexpected command: {command}")

    def _branch(self, parent: int, token: str) -> dict:
        board = dict(self._states[parent])
        fields = token.split(":")
        kind, events = fields[0], {}
        if kind == "W":
            board["tick"] += int(fields[1])
        elif kind == "P":
            _, packet, col, row = fields
            board["plants"] = tuple(sorted({*board["plants"], (int(row), int(col), int(packet))}))
            board["sun"] -= PLANT_COST
            events = {"sun_spent": PLANT_COST}
        elif kind == "S":
            _, col, row = fields
            board["plants"] = tuple(p for p in board["plants"] if (p[0], p[1]) != (int(row), int(col)))
        else:
            raise AssertionError(f"unsupported branch token: {token}")

        terminal = self._terminal_tick is not None and board["tick"] >= self._terminal_tick
        branch = {
            "ok": True,
            "observation": _observation(board, terminal=terminal, result=1 if self._winning else 0),
            "events": events,
        }
        if not terminal:
            branch["snapshot_id"] = self._take_snapshot(board)
            branch["state_hash"] = _state_hash(board)
        return branch


def _node(observation: dict, snapshot_id: int, state_hash: str = "root", *,
          same_tick_actions: int = 0, elapsed_ticks: int = 0, decision_count: int = 0,
          path_return: float = 0.0, discount: float = 1.0, score: float = 0.0,
          diversity_key: tuple[Any, ...] = ("root",)) -> _SearchNode:
    """Build a ``_SearchNode`` by keyword.

    ``_SearchNode`` has ten same-shaped fields; positional construction silently changes
    meaning if two of them are ever reordered, so every node in these tests is named.
    """
    return _SearchNode(
        observation=observation,
        snapshot_id=snapshot_id,
        state_hash=state_hash,
        same_tick_actions=same_tick_actions,
        elapsed_ticks=elapsed_ticks,
        decision_count=decision_count,
        path_return=path_return,
        discount=discount,
        score=score,
        diversity_key=diversity_key,
    )


def _root_node(env: _FakeSimulator, board: dict | None = None) -> _SearchNode:
    board = _board() if board is None else board
    return _node(_observation(board), env.root_snapshot())


class SearchTeacherProtocolTests(unittest.TestCase):
    """The search must read the simulator's tick/hash/terminal contract literally."""

    def test_branch_tick_is_absolute_and_not_the_wait_duration(self) -> None:
        env = _FakeSimulator()
        searcher = SearchTeacher(env, horizon_ticks=900)
        root = _root_node(env)

        [(child, leaf)] = searcher._expand_actions(root, [{"type": "wait", "ticks": 60}])
        self.assertIsNotNone(child, "a 60-tick wait must produce a child node")
        child = cast(_SearchNode, child)
        # 180, not 60: the observation reports the absolute tick, and elapsed is the delta.
        self.assertEqual(child.observation["tick"], ROOT_TICK + 60)
        self.assertEqual(child.elapsed_ticks, 60)
        self.assertEqual(leaf[2], 60)

        [(deeper, deeper_leaf)] = searcher._expand_actions(child, [{"type": "wait", "ticks": 150}])
        self.assertIsNotNone(deeper, "a second wait must produce a deeper child node")
        deeper = cast(_SearchNode, deeper)
        self.assertEqual(deeper.observation["tick"], ROOT_TICK + 60 + 150)
        self.assertEqual(deeper.elapsed_ticks, 210)
        self.assertEqual(deeper_leaf[2], 210)

    def test_zero_tick_actions_do_not_advance_the_horizon(self) -> None:
        env = _FakeSimulator()
        searcher = SearchTeacher(env, horizon_ticks=900)
        root = _root_node(env)

        [(planted, _)] = searcher._expand_actions(root, [{"type": "plant", "packet": 0, "row": 0, "col": 1}])
        self.assertIsNotNone(planted, "an instant plant must still produce a child node")
        planted = cast(_SearchNode, planted)
        self.assertEqual(planted.observation["tick"], ROOT_TICK)
        self.assertEqual(planted.elapsed_ticks, 0)
        self.assertEqual(planted.same_tick_actions, 1)

    def test_actions_beyond_the_remaining_horizon_are_never_branched(self) -> None:
        env = _FakeSimulator()
        searcher = SearchTeacher(env, horizon_ticks=900)
        root = _root_node(env)
        root.elapsed_ticks = 750  # 150 ticks of horizon remain

        searcher._expand_actions(root, [
            {"type": "wait", "ticks": 300},
            {"type": "wait", "ticks": 150},
            {"type": "wait", "ticks": 60},
        ])
        branch = next(command for command in env.commands if command.startswith("BRANCH_SNAPSHOT_FAST"))
        self.assertEqual(branch.split()[3:], ["W:150", "W:60"])

    def test_different_action_orders_that_reach_the_same_board_transpose(self) -> None:
        env = _FakeSimulator()
        searcher = SearchTeacher(env, horizon_ticks=900)
        root = _root_node(env)
        first = {"type": "plant", "packet": 0, "row": 0, "col": 1}
        second = {"type": "plant", "packet": 1, "row": 1, "col": 3}
        wait = {"type": "wait", "ticks": 60}

        def expand(actions: list[dict]) -> _SearchNode:
            node = root
            for action in actions:
                [(child, _)] = searcher._expand_actions(node, [action])
                self.assertIsNotNone(child, f"{action} produced a leaf, not a child node")
                node = cast(_SearchNode, child)
            return node

        node_a = expand([first, second, wait])
        node_b = expand([second, first, wait])

        # Two distinct snapshot ids, one shared board: the hash must come from the board,
        # not from the snapshot counter, or nothing can ever be deduplicated.
        self.assertNotEqual(node_a.snapshot_id, node_b.snapshot_id)
        self.assertEqual(node_a.state_hash, node_b.state_hash)
        self.assertEqual(searcher._transposition_key(node_a), searcher._transposition_key(node_b))

        # Same elapsed ticks, same decision count, same same-tick budget -- only the board
        # differs, so only state_hash can tell these two apart.
        other = expand([{"type": "plant", "packet": 2, "row": 2, "col": 5},
                        {"type": "plant", "packet": 0, "row": 3, "col": 7},
                        wait])
        self.assertEqual((other.elapsed_ticks, other.decision_count, other.same_tick_actions),
                         (node_a.elapsed_ticks, node_a.decision_count, node_a.same_tick_actions))
        self.assertNotEqual(other.state_hash, node_a.state_hash)
        self.assertNotEqual(searcher._transposition_key(other), searcher._transposition_key(node_a))

    def test_terminal_branch_reports_its_outcome_and_has_no_snapshot(self) -> None:
        env = _FakeSimulator(terminal_tick=ROOT_TICK + 60)
        searcher = SearchTeacher(env, horizon_ticks=900)
        root = _root_node(env)
        issued_before = len(env.issued)

        [(child, leaf)] = searcher._expand_actions(root, [{"type": "wait", "ticks": 60}])
        self.assertIsNone(child)
        self.assertEqual(leaf[1], 1)
        self.assertEqual(leaf[2], 60)
        self.assertEqual(len(env.issued), issued_before)

    def test_terminal_branch_can_report_a_loss(self) -> None:
        env = _FakeSimulator(terminal_tick=ROOT_TICK, winning=False)
        searcher = SearchTeacher(env, horizon_ticks=900)
        root = _root_node(env)

        [(child, leaf)] = searcher._expand_actions(root, [{"type": "wait", "ticks": 60}])
        self.assertIsNone(child)
        self.assertEqual(leaf[1], 0)

    def test_nonterminal_branch_without_snapshot_is_rejected(self) -> None:
        env = _FakeSimulator()
        searcher = SearchTeacher(env, horizon_ticks=900)
        root = _root_node(env)
        original = env._command

        def strip_snapshot(command: str) -> dict:
            response = original(command)
            if "branches" in response:
                for branch in response["branches"]:
                    branch.pop("snapshot_id", None)
            return response

        env._command = strip_snapshot  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "snapshot and state_hash"):
            searcher._expand_actions(root, [{"type": "wait", "ticks": 60}])

    def test_search_requires_a_reset_environment(self) -> None:
        env = _FakeSimulator()
        env._reset_done = False
        with self.assertRaisesRegex(RuntimeError, "reset"):
            SearchTeacher(env).advice(_observation(_board()))


class SearchTeacherBookkeepingTests(unittest.TestCase):
    def test_fast_speculation_restores_episode_bookkeeping(self) -> None:
        env = _FakeSimulator()
        searcher = SearchTeacher(env)
        original_final = env.episode["final_state"]

        with searcher._speculative_fast() as snapshot:
            self.assertEqual(snapshot, 1)
            env.episode["operations"].append({"kind": "speculative"})
            env.episode["final_state"] = {"tick": 999}

        self.assertEqual(env.episode["operations"], [{"kind": "real"}])
        self.assertIs(env.episode["final_state"], original_final)
        self.assertIn("RESTORE_FAST 1", env.commands)
        self.assertIn("DROP_SNAPSHOT_FAST 1", env.commands)

    def test_search_releases_every_snapshot_it_creates(self) -> None:
        env = _FakeSimulator()
        searcher = SearchTeacher(env, simulation_budget=64, candidate_limit=4)

        searcher.advice(_observation(_board()))

        self.assertTrue(env.issued, "the search never asked for a snapshot")
        self.assertEqual(set(env.issued) - env.dropped, set())

    def test_transposition_table_rejects_worse_duplicate(self) -> None:
        env = _FakeSimulator()
        searcher = SearchTeacher(env, horizon_ticks=900)
        root = _root_node(env)
        observation = root.observation
        state = _RootState(action={"type": "wait", "ticks": 60})
        better = _node(observation, env.root_snapshot(), "same", elapsed_ticks=60,
                       decision_count=1, score=0.8, diversity_key=("wait", 1))
        worse = _node(observation, env.root_snapshot(), "same", elapsed_ticks=60,
                      decision_count=1, score=0.5, diversity_key=("wait", 1))
        searcher._snapshots.update({better.snapshot_id, worse.snapshot_id})

        self.assertTrue(searcher._accept_node(state, better))
        self.assertFalse(searcher._accept_node(state, worse))
        self.assertIn(f"DROP_SNAPSHOT_FAST {worse.snapshot_id}", env.commands)
        self.assertNotIn(f"DROP_SNAPSHOT_FAST {better.snapshot_id}", env.commands)

    def test_partial_leaves_prefer_the_deeper_line(self) -> None:
        searcher = SearchTeacher(_FakeSimulator(), horizon_ticks=900)
        observation = _observation(_board())
        shallow = _node(observation, 1, "h", elapsed_ticks=100, decision_count=1,
                        score=0.5, diversity_key=("a",))
        deep = _node(observation, 2, "h", elapsed_ticks=800, decision_count=1,
                     score=0.5, diversity_key=("a",))

        self.assertGreater(searcher._partial_leaf(deep)[0], searcher._partial_leaf(shallow)[0])

    def test_search_parameters_must_be_positive(self) -> None:
        for field in ("beam_width", "candidate_limit", "horizon_ticks",
                      "simulation_budget", "max_same_tick_actions", "max_decisions"):
            with self.assertRaises(ValueError, msg=field):
                SearchTeacher(_FakeSimulator(), **{field: 0})

    def test_unsupported_search_actions_are_rejected(self) -> None:
        searcher = SearchTeacher(_FakeSimulator())
        with self.assertRaisesRegex(ValueError, "unsupported search action"):
            searcher._branch_action_token({"type": "sun"})
        with self.assertRaisesRegex(ValueError, "unsupported search action"):
            searcher._diversity_key({"type": "sun"})


class SearchTeacherScoringTests(unittest.TestCase):
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

    def test_transition_reward_caps_penalties_and_pays_terminal_outcomes(self) -> None:
        reward = SearchTeacher._transition_reward
        self.assertEqual(reward({}, False, False), 0.0)
        self.assertAlmostEqual(reward({}, True, True), 1.0)
        self.assertAlmostEqual(reward({}, True, False), -1.0)
        self.assertAlmostEqual(reward({"plants_eaten": 100}, False, False), -0.20)
        self.assertAlmostEqual(reward({"mower_triggered": 4}, False, False), -0.25)

    def test_leaf_value_delegates_to_the_configured_value_model(self) -> None:
        model = SearchValueModel()
        searcher = SearchTeacher(_FakeSimulator(), value_model=model)
        self.assertIs(searcher.value_model, model)
        self.assertEqual(searcher._leaf_value(_observation(_board(), terminal=True, result=1)), 1.0)
        self.assertEqual(searcher._leaf_value(_observation(_board(), terminal=True, result=0)), -1.0)

        nonterminal = _observation(_board())
        self.assertAlmostEqual(
            searcher._leaf_value(nonterminal),
            max(-1.0, min(1.0, float(model.predict(nonterminal)))),
        )

    def test_bootstrap_evaluator_is_used_without_a_value_model(self) -> None:
        searcher = SearchTeacher(_FakeSimulator())
        self.assertIsNone(searcher.value_model)
        observation = _observation(_board())

        self.assertEqual(searcher._leaf_value(observation),
                         SearchTeacher._bootstrap_leaf_value(observation))

    def test_fallback_action_covers_every_legal_action_kind(self) -> None:
        fallback = SearchTeacher._fallback_action
        self.assertEqual(fallback({"legal_actions": {"plants": [], "shovels": [], "wait": True}}),
                         {"type": "wait", "ticks": WAIT_TICKS[0]})
        self.assertEqual(
            fallback({"legal_actions": {"plants": [{"packet": 2, "row": 1, "col": 3}],
                                        "shovels": [], "wait": False}}),
            {"type": "plant", "packet": 2, "row": 1, "col": 3},
        )
        self.assertEqual(fallback({"legal_actions": {"plants": [], "shovels": [[4, 2]], "wait": False}}),
                         {"type": "shovel", "col": 4, "row": 2})
        with self.assertRaisesRegex(RuntimeError, "no legal action"):
            fallback({"legal_actions": {"plants": [], "shovels": [], "wait": False}})


class OneStepChildrenTests(unittest.TestCase):
    """``one_step_children`` is the counterfactual-leaf view the audit tools consume."""

    def test_every_action_comes_back_with_its_own_child(self) -> None:
        env = _FakeSimulator()
        searcher = SearchTeacher(env, horizon_ticks=900)
        plant = {"type": "plant", "packet": 0, "row": 0, "col": 1}

        children = searcher.one_step_children(_observation(_board()), [{"type": "wait", "ticks": 60}, plant])

        self.assertEqual([action for action, _ in children], [{"type": "wait", "ticks": 60}, plant])
        self.assertEqual(children[0][1]["tick"], ROOT_TICK + 60)
        self.assertEqual(children[1][1]["tick"], ROOT_TICK)
        self.assertEqual([(item["row"], item["col"]) for item in children[1][1]["plants"]], [(0, 1)])

    def test_actions_that_do_not_fit_the_horizon_are_dropped_with_their_child(self) -> None:
        env = _FakeSimulator()
        searcher = SearchTeacher(env, horizon_ticks=150)
        plant = {"type": "plant", "packet": 0, "row": 0, "col": 1}

        children = searcher.one_step_children(_observation(_board()),
                                              [{"type": "wait", "ticks": 300}, plant])

        self.assertEqual([action for action, _ in children], [plant])

    def test_an_action_that_ends_the_level_has_no_child(self) -> None:
        env = _FakeSimulator(terminal_tick=ROOT_TICK + 60)
        searcher = SearchTeacher(env, horizon_ticks=900)

        children = searcher.one_step_children(_observation(_board()), [{"type": "wait", "ticks": 60}])

        self.assertEqual(len(children), 1)
        self.assertIsNone(children[0][1])

    def test_a_wider_action_set_than_the_protocol_allows_is_chunked(self) -> None:
        """The protocol refuses more than ``BRANCH_BATCH_LIMIT`` specs in one command."""
        env = _FakeSimulator()
        searcher = SearchTeacher(env, horizon_ticks=900)
        actions = [{"type": "plant", "packet": packet, "row": row, "col": col}
                   for packet in range(5) for row in range(5) for col in range(9)]
        self.assertGreater(len(actions), BRANCH_BATCH_LIMIT)

        children = searcher.one_step_children(_observation(_board()), actions)

        self.assertEqual(len(children), len(actions))
        self.assertTrue(all(child is not None for _, child in children))

    def test_an_oversized_branch_command_is_rejected_rather_than_truncated(self) -> None:
        """Silently sending 129 specs would let the C++ side answer `ok:false` mid-search."""
        env = _FakeSimulator()
        searcher = SearchTeacher(env, horizon_ticks=900)
        actions = [{"type": "wait", "ticks": 60}] * (BRANCH_BATCH_LIMIT + 1)

        with self.assertRaisesRegex(ValueError, "at most"):
            searcher._branch_snapshot_fast(1, actions)

    def test_children_are_released_even_though_only_the_observations_escape(self) -> None:
        env = _FakeSimulator()
        searcher = SearchTeacher(env, horizon_ticks=900)

        searcher.one_step_children(_observation(_board()),
                                   [{"type": "plant", "packet": 0, "row": 0, "col": 1}])

        self.assertTrue(env.issued, "the audit never asked for a snapshot")
        self.assertEqual(set(env.issued) - env.dropped, set(), "a child snapshot leaked")

    def test_unsupported_search_actions_are_rejected(self) -> None:
        env = _FakeSimulator()
        searcher = SearchTeacher(env, horizon_ticks=900)

        with self.assertRaises(ValueError):
            searcher.one_step_children(_observation(_board()), [{"type": "sun"}])


class SearchTeacherDecisionTests(unittest.TestCase):
    def test_global_simulation_budget_bounds_entire_decision(self) -> None:
        searcher = SearchTeacher(_FakeSimulator(), simulation_budget=24, candidate_limit=2)

        advice = searcher.advice(_observation(_board()))

        self.assertGreater(advice.simulation_count, 0)
        self.assertLessEqual(advice.simulation_count, 24)

    def test_root_candidate_limit_reserves_budget_for_depth(self) -> None:
        for budget, candidates, expected in ((64, 8, 16), (256, 8, 24), (32, 4, 8)):
            with self.subTest(budget=budget, candidates=candidates):
                searcher = SearchTeacher(_FakeSimulator(), simulation_budget=budget,
                                         candidate_limit=candidates)
                self.assertEqual(searcher.root_candidate_limit, expected)
                self.assertLessEqual(searcher.root_candidate_limit * 2, searcher.simulation_budget)

    def test_root_screening_limit_never_consumes_the_whole_budget(self) -> None:
        """``advice`` must reserve depth budget when it asks for raw root actions.

        The cap handed to the candidate generator is
        ``min(max(1, budget // 2), max(32, root_candidate_limit * 4))``.  The
        second term is slack for diversity; the first one is the budget guard.
        Without it a small budget is spent *entirely* on screening -- with
        ``budget=64, candidate_limit=8`` the generator would be asked for 64
        actions out of a budget of 64, leaving ``_successive_halving`` nothing
        to expand with.  That is the same invariant
        ``test_root_candidate_limit_reserves_budget_for_depth`` pins one level
        up, so it is asserted here through the actual call site.
        """
        for budget, candidates in ((8, 8), (32, 8), (64, 8), (128, 8), (256, 8), (512, 8)):
            with self.subTest(budget=budget):
                searcher = SearchTeacher(_FakeSimulator(), simulation_budget=budget,
                                         candidate_limit=candidates)
                limits: list[int] = []
                inner = searcher.candidate_generator

                class _RecordingGenerator:
                    def actions(self, observation, limit, root, remaining_ticks, allow_instant=True):
                        if root:
                            limits.append(limit)
                        return inner.actions(observation, limit, root, remaining_ticks, allow_instant)

                searcher.candidate_generator = _RecordingGenerator()
                advice = searcher.advice(_observation(_board()))

                self.assertEqual(len(limits), 1, "root screening must issue exactly one candidate request")
                self.assertLessEqual(limits[0], max(1, searcher.simulation_budget // 2))
                self.assertEqual(limits[0], min(max(1, budget // 2),
                                                max(32, searcher.root_candidate_limit * 4)))
                self.assertLessEqual(advice.simulation_count, searcher.simulation_budget)

    def test_root_screening_keeps_best_diverse_candidates_within_budget(self) -> None:
        searcher = SearchTeacher(_FakeSimulator(), simulation_budget=64, candidate_limit=2)

        advice = searcher.advice(_observation(_board()))

        self.assertEqual(len(advice.candidates), searcher.root_candidate_limit)
        self.assertLessEqual(advice.simulation_count, searcher.simulation_budget)
        self.assertGreater(advice.simulation_count, len(advice.candidates))
        self.assertAlmostEqual(sum(advice.search_policy), 1.0)
        self.assertIn(advice.action, [action for action, _ in advice.candidates])

    def test_root_initialisation_pairs_each_action_with_its_own_branch(self) -> None:
        """A dropped action must not shift every later action onto its neighbour's result.

        ``_branch_snapshot_fast`` is only ever handed the actions that fit the
        remaining horizon, so pairing its results back against the caller's
        *original* action list misaligns them the moment one action is filtered
        out.  A 300-tick wait does not fit a 150-tick horizon; a plant always
        does.  If the two lists drift, the surviving root state reports the wait
        as its action while carrying the plant's branch -- and the search would
        then credit the plant's outcome to an action it never took.
        """
        env = _FakeSimulator()
        root = _root_node(env)
        plant = {"type": "plant", "packet": 0, "row": 0, "col": 1}
        searcher = SearchTeacher(env, horizon_ticks=150, candidate_limit=2)

        states = searcher._initialize_roots(root, [{"type": "wait", "ticks": 300}, plant])

        self.assertEqual([state.action for state in states], [plant])
        self.assertEqual(searcher._fit_to_remaining(root, [{"type": "wait", "ticks": 300}]), [])
        self.assertEqual(searcher._fit_to_remaining(root, [plant]), [plant])

    def test_root_initialisation_keeps_every_action_when_all_of_them_fit(self) -> None:
        env = _FakeSimulator()
        root = _root_node(env)
        actions = [
            {"type": "wait", "ticks": 60},
            {"type": "plant", "packet": 0, "row": 0, "col": 1},
            {"type": "plant", "packet": 1, "row": 1, "col": 3},
        ]
        searcher = SearchTeacher(env, horizon_ticks=900, candidate_limit=2)

        states = searcher._initialize_roots(root, actions)

        self.assertEqual([state.action for state in states], actions)

    def test_advice_reports_the_screening_and_depth_split(self) -> None:
        """``simulation_count`` alone cannot be read as search depth.

        Every root candidate costs one simulation before any line is extended, so
        the depth search only gets ``simulation_budget - screening_simulations``.
        Without this split a budget sweep looks like "128 / 256 / 512 of depth"
        when it is really three different screening-to-depth ratios.
        """
        for budget in (64, 128, 256, 512):
            with self.subTest(budget=budget):
                searcher = SearchTeacher(_FakeSimulator(), simulation_budget=budget, candidate_limit=8)

                advice = searcher.advice(_observation(_board()))

                self.assertEqual(advice.screening_simulations + advice.depth_simulations,
                                 advice.simulation_count)
                self.assertEqual(advice.screening_simulations, advice.root_actions_generated)
                self.assertEqual(advice.effective_depth_budget,
                                 max(0, budget - advice.screening_simulations))
                self.assertEqual(advice.root_candidates_screened,
                                 min(advice.root_actions_generated, searcher.root_candidate_limit))
                self.assertLessEqual(advice.simulation_count, budget)

    def test_a_small_budget_spends_most_of_itself_on_root_screening(self) -> None:
        """Pin the actual numbers a budget ablation has to be read against.

        The fake board above only offers a handful of legal actions, so screening
        never binds there.  This generator hands back ``limit`` distinct actions
        instead, which is what the budget guard in ``advice`` is protecting
        against: at ``budget=128`` half of the budget is screening and the depth
        search is left with 64, whereas at 256 and above the request saturates at
        ``max(32, root_candidate_limit * 4)`` and the depth budget stops growing
        in step with ``simulation_budget``.
        """
        pool = [{"type": "plant", "packet": packet, "row": row, "col": col}
                for packet in (0, 1, 2) for row in range(5) for col in range(9)]

        class _WideGenerator:
            def actions(self, observation, limit, root, remaining_ticks, allow_instant=True):
                return [dict(action) for action in pool[:limit]]

        expected = {64: (32, 32), 128: (64, 64), 256: (96, 160), 512: (96, 416)}
        for budget, (screening, depth_budget) in expected.items():
            with self.subTest(budget=budget):
                searcher = SearchTeacher(_FakeSimulator(), simulation_budget=budget, candidate_limit=8)
                searcher.candidate_generator = _WideGenerator()

                advice = searcher.advice(_observation(_board()))

                self.assertEqual(advice.screening_simulations, screening)
                self.assertEqual(advice.effective_depth_budget, depth_budget)
                self.assertLessEqual(advice.simulation_count, budget)

    def test_screening_accounting_counts_branches_not_requests(self) -> None:
        """A root action that cannot fit the horizon is never simulated, so it costs nothing.

        The candidate generator normally pre-fits at ``horizon_ticks`` and the
        request and the spend agree, but the accounting must not *depend* on that:
        reserving the requested count instead of the branched count would
        under-fund the depth search by exactly the number of dropped actions.
        """
        searcher = SearchTeacher(_FakeSimulator(), horizon_ticks=150, simulation_budget=64,
                                 candidate_limit=8)

        class _StaleGenerator:
            def actions(self, observation, limit, root, remaining_ticks, allow_instant=True):
                return [{"type": "wait", "ticks": 300},
                        {"type": "plant", "packet": 0, "row": 0, "col": 1}]

        searcher.candidate_generator = _StaleGenerator()

        advice = searcher.advice(_observation(_board()))

        self.assertEqual(advice.root_actions_generated, 2)
        self.assertEqual(advice.screening_simulations, 1)
        self.assertEqual(advice.effective_depth_budget, 63)
        self.assertEqual(advice.screening_simulations + advice.depth_simulations,
                         advice.simulation_count)

    def test_max_decisions_stops_search_even_when_tick_horizon_remains(self) -> None:
        env = _FakeSimulator()
        root = _root_node(env)
        action = [{"type": "wait", "ticks": 60}]

        stopped = SearchTeacher(env, horizon_ticks=900, max_decisions=1)
        [(child, leaf)] = stopped._expand_actions(root, action)
        self.assertIsNone(child)
        self.assertEqual(leaf[2], 60)

        continued = SearchTeacher(env, horizon_ticks=900, max_decisions=2)
        [(child, _)] = continued._expand_actions(root, action)
        self.assertIsNotNone(child)

    def test_search_prefers_a_terminal_win_over_partial_lines(self) -> None:
        searcher = SearchTeacher(_FakeSimulator(terminal_tick=ROOT_TICK + 150),
                                 simulation_budget=64, candidate_limit=4)

        advice = searcher.advice(_observation(_board()))

        self.assertEqual(advice.terminal_outcome, 1)
        self.assertIsNotNone(advice.best_second_margin)

    def test_search_never_latches_onto_a_terminal_loss(self) -> None:
        searcher = SearchTeacher(_FakeSimulator(terminal_tick=ROOT_TICK + 150, winning=False),
                                 simulation_budget=64, candidate_limit=4)

        advice = searcher.advice(_observation(_board()))

        # With no winnable line the search keeps searching rather than reporting a loss,
        # because _outcome_rank orders win > unfinished > loss.
        self.assertIsNone(advice.terminal_outcome)
        self.assertGreater(SearchTeacher._result_key(None, 0.0), SearchTeacher._result_key(0, 100.0))
        self.assertGreater(SearchTeacher._result_key(1, -100.0), SearchTeacher._result_key(None, 100.0))

    def test_advice_only_proposes_actions_legal_in_the_root_observation(self) -> None:
        observation = _observation(_board())
        searcher = SearchTeacher(_FakeSimulator(), simulation_budget=64, candidate_limit=4)

        advice = searcher.advice(observation)

        legal = observation["legal_actions"]
        placements = {(item["packet"], item["row"], item["col"]) for item in legal["plants"]}
        shovels = {(col, row) for col, row in legal["shovels"]}
        for action, _ in advice.candidates:
            if action["type"] == "plant":
                self.assertIn((action["packet"], action["row"], action["col"]), placements)
            elif action["type"] == "shovel":
                self.assertIn((action["col"], action["row"]), shovels)
            else:
                self.assertIn(action["ticks"], WAIT_TICKS)

    def test_search_is_reproducible_from_the_simulator_alone(self) -> None:
        observation = _observation(_board())
        first = SearchTeacher(_FakeSimulator(), simulation_budget=64, candidate_limit=4)
        second = SearchTeacher(_FakeSimulator(), simulation_budget=64, candidate_limit=4)

        left = first.advice(observation)
        right = second.advice(observation)

        self.assertEqual(left.action, right.action)
        self.assertEqual(left.candidates, right.candidates)
        self.assertEqual(left.search_policy, right.search_policy)


class CandidateGeneratorTests(unittest.TestCase):
    def test_temporal_actions_are_fitted_to_the_remaining_horizon(self) -> None:
        self.assertIsNone(fit_action_to_remaining({"type": "wait", "ticks": 300}, 90))
        self.assertIsNone(fit_action_to_remaining({"type": "wait", "ticks": 60}, 0))
        self.assertIsNone(fit_action_to_remaining({"type": "wait", "ticks": 90}, 300))
        self.assertEqual(fit_action_to_remaining({"type": "wait", "ticks": 60}, 60),
                         {"type": "wait", "ticks": 60})
        self.assertEqual(fit_action_to_remaining({"type": "wait", "ticks": 60}, 61),
                         {"type": "wait", "ticks": 60})
        self.assertEqual(
            fit_action_to_remaining({"type": "plant", "packet": 1, "row": 2, "col": 3}, 1),
            {"type": "plant", "packet": 1, "row": 2, "col": 3},
        )
        with self.assertRaisesRegex(ValueError, "unsupported search action"):
            fit_action_to_remaining({"type": "sun"}, 300)

    def test_internal_candidates_round_robin_wait_and_plant_actions(self) -> None:
        actions = CandidateGenerator().actions(_observation(_board()), 8, False, 900)
        waits = [action for action in actions if action["type"] == "wait"]

        self.assertEqual(len(waits), 1)
        self.assertIn("plant", {action["type"] for action in actions})
        self.assertTrue(all(action["ticks"] in WAIT_TICKS for action in waits))

    def test_structural_root_candidates_cover_every_legal_packet(self) -> None:
        actions = CandidateGenerator().actions(_observation(_board()), 32, True, 900)

        self.assertEqual({action["packet"] for action in actions if action["type"] == "plant"}, {0, 1, 2})

    def test_instant_actions_are_skipped_when_same_tick_budget_is_spent(self) -> None:
        actions = CandidateGenerator().actions(_observation(_board()), 8, False, 900, allow_instant=False)

        self.assertEqual({action["type"] for action in actions}, {"wait"})

    def test_candidates_never_place_a_plant_on_an_occupied_cell(self) -> None:
        observation = _observation(_board(plants=((0, 1, 0),)))
        placements = {(item["row"], item["col"]) for item in observation["legal_actions"]["plants"]}

        actions = CandidateGenerator().actions(observation, 32, True, 900)

        self.assertNotIn((0, 1), placements)
        for action in actions:
            if action["type"] == "plant":
                self.assertIn((action["row"], action["col"]), placements)

    def test_shovel_proposals_accept_json_coordinate_lists(self) -> None:
        observation = _observation(_board(plants=((1, 2, 0),)))

        self.assertEqual(shovel_proposals(observation, 1), [{"type": "shovel", "col": 2, "row": 1}])

    def test_shovel_proposals_prefer_the_weakest_plant(self) -> None:
        observation = _observation(_board(plants=((0, 1, 0), (2, 4, 1))))
        observation["plants"][0]["health"] = 100
        observation["plants"][1]["health"] = 10

        self.assertEqual(shovel_proposals(observation, 1), [{"type": "shovel", "col": 4, "row": 2}])


class ValueSemanticsTests(unittest.TestCase):
    def test_discounted_terminal_value_matches_tick_semantics(self) -> None:
        self.assertEqual(discounted_terminal_value(True, 0), 1.0)
        self.assertEqual(discounted_terminal_value(False, 0), -1.0)
        self.assertAlmostEqual(discounted_terminal_value(True, 300), VALUE_GAMMA)
        self.assertAlmostEqual(discounted_terminal_value(True, 600), VALUE_GAMMA ** 2)
        # Remaining ticks are clamped at zero, so an over-run never amplifies the value.
        self.assertEqual(discounted_terminal_value(False, -50), -1.0)


if __name__ == "__main__":
    unittest.main()
