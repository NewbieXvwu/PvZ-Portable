"""Simulator beam search used to generate imitation targets."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import Tensor

from pvz_agent_model import GameplayModelV1, WAIT_DECISION_TICKS
from pvz_search_candidates import CandidateGenerator, fit_action_to_remaining, lane_pressure
from pvz_value import DISCOUNT_REFERENCE_TICKS, VALUE_GAMMA

SEARCH_POLICY_MIN_SCALE = 0.05
SEARCH_PARTIAL_SHORTFALL_PENALTY = 0.15


@dataclass(frozen=True)
class SearchAdvice:
    action: dict[str, Any]
    candidates: list[tuple[dict[str, Any], float]]
    search_policy: list[float]
    best_second_margin: float | None
    simulation_count: int
    terminal_outcome: int | None
    search_elapsed_ticks: int


@dataclass
class _SearchNode:
    observation: dict[str, Any]
    snapshot_id: int
    output: dict[str, Any] | None
    same_tick_actions: int
    elapsed_ticks: int
    path_return: float
    discount: float
    score: float
    diversity_key: tuple[Any, ...]


class SearchTeacher:
    def __init__(self, env: Any, model: GameplayModelV1 | None = None, beam_width: int = 3,
                 candidate_limit: int = 8, horizon_ticks: int = 900,
                 simulation_budget: int = 64, max_same_tick_actions: int = 2) -> None:
        if min(beam_width, candidate_limit, horizon_ticks, simulation_budget, max_same_tick_actions) < 1:
            raise ValueError("search parameters must be positive")
        if horizon_ticks < WAIT_DECISION_TICKS:
            raise ValueError(f"search horizon must be at least {WAIT_DECISION_TICKS} ticks")
        self.env = env
        self.model = model
        self.beam_width = beam_width
        self.candidate_limit = candidate_limit
        self.root_candidate_limit = max(32, candidate_limit * 5)
        self.horizon_ticks = horizon_ticks
        self.simulation_budget = simulation_budget
        self.max_same_tick_actions = max_same_tick_actions
        self.candidate_generator = CandidateGenerator(model)
        self._snapshots: set[int] = set()
        self._root_snapshot = 0

    @staticmethod
    def _outcome_rank(outcome: int | None) -> int:
        return 2 if outcome == 1 else 1 if outcome is None else 0

    @classmethod
    def _result_key(cls, outcome: int | None, score: float) -> tuple[int, float]:
        return cls._outcome_rank(outcome), score

    @classmethod
    def _policy_from_results(cls, results: list[tuple[dict[str, Any], float, int | None, int]]) -> list[float]:
        if not results:
            return []
        rank = max(cls._outcome_rank(item[2]) for item in results)
        scores = [item[1] for item in results if cls._outcome_rank(item[2]) == rank]
        if len(scores) == 1:
            return [1.0 if cls._outcome_rank(item[2]) == rank else 0.0 for item in results]
        mean = sum(scores) / len(scores)
        scale = max(SEARCH_POLICY_MIN_SCALE, math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores)))
        top = max(scores)
        weights = [math.exp((item[1] - top) / scale) if cls._outcome_rank(item[2]) == rank else 0.0 for item in results]
        total = sum(weights)
        return [weight / total for weight in weights]

    @staticmethod
    def _leaf_value(observation: dict[str, Any]) -> float:
        if observation["terminal"]:
            return 1.0 if observation["result"] == 1 else -1.0
        rows = {cell["row"] for cell in observation["cells"] if cell["row_type"] > 0}
        row_count = max(1, len(rows))
        plants = [p for p in observation["plants"] if not p.get("squished") and p.get("health", 0) > 0]
        coverage = len({p["row"] for p in plants if p["row"] in rows}) / row_count
        health = min(1.0, sum(max(0.0, min(1.0, p.get("health", 0) / max(1, p.get("max_health", 1))))
                              for p in plants) / (row_count * 3.0))
        density = min(1.0, len(plants) / (row_count * 4.0))
        defense = min(1.0, sum(d.get("state") == 1 for d in observation.get("defenses", [])) / row_count)
        progress = min(max(observation["wave"] / max(1, observation["wave_count"]), 0.0), 1.0)
        sun = min(max(observation["sun"] / 1000.0, 0.0), 1.0)
        threat = min(1.0, sum(lane_pressure(observation).values()) / (row_count * 5.0))
        nearest = min((zombie["x"] for zombie in observation["zombies"]), default=900.0)
        breach = max(0.0, min(1.0, (420.0 - nearest) / 300.0))
        value = (-0.16 + 0.18 * progress + 0.30 * coverage + 0.14 * health + 0.08 * density
                 + 0.08 * sun + 0.10 * defense - 0.46 * threat - 0.20 * breach)
        return max(-1.0, min(1.0, value))

    @staticmethod
    def _transition_reward(events: dict[str, Any], terminal: bool, won: bool) -> float:
        reward = min(0.05, max(0, events.get("zombies_killed", 0)) * 0.01)
        reward -= min(0.20, max(0, events.get("plants_eaten", 0)) * 0.05)
        reward -= min(0.25, max(0, events.get("mower_triggered", 0)) * 0.25)
        if terminal:
            reward += 1.0 if won else -1.0
        return reward

    def _model_output(self, observation: dict[str, Any], hidden: Tensor | None,
                      previous_action: dict[str, Any] | None, delta_ticks: int,
                      events: dict[str, Any] | None) -> dict[str, Any] | None:
        return None if self.model is None else self.model.step(observation, hidden, previous_action, delta_ticks, events)

    def _snapshot_fast(self) -> int:
        response = self.env._command("SNAPSHOT_FAST")
        if not response.get("ok") or "snapshot_id" not in response:
            raise RuntimeError(f"snapshot failed: {response}")
        return int(response["snapshot_id"])

    def _restore_fast(self, snapshot_id: int) -> None:
        response = self.env._command(f"RESTORE_FAST {snapshot_id}")
        if not response.get("ok"):
            raise RuntimeError(f"restore failed: {response}")

    def _release_fast(self, snapshot_id: int) -> None:
        response = self.env._command(f"DROP_SNAPSHOT_FAST {snapshot_id}")
        if not response.get("ok"):
            raise RuntimeError(f"drop snapshot failed: {response}")

    @contextmanager
    def _speculative_fast(self) -> Iterator[int]:
        if not self.env._reset_done or self.env.episode is None:
            raise RuntimeError("call reset() before search")
        count = len(self.env.episode["operations"])
        final_state = self.env.episode["final_state"]
        snapshot_id = self._snapshot_fast()
        try:
            yield snapshot_id
        finally:
            self._restore_fast(snapshot_id)
            del self.env.episode["operations"][count:]
            self.env.episode["final_state"] = final_state
            self._release_fast(snapshot_id)

    def _release(self, snapshot_id: int) -> None:
        if snapshot_id != self._root_snapshot and snapshot_id in self._snapshots:
            self._release_fast(snapshot_id)
            self._snapshots.remove(snapshot_id)

    @staticmethod
    def _branch_action_token(action: dict[str, Any]) -> str:
        kind = action.get("type")
        if kind == "plant":
            return f"P:{int(action['packet'])}:{int(action['col'])}:{int(action['row'])}"
        if kind == "shovel":
            return f"S:{int(action['col'])}:{int(action['row'])}"
        if kind == "wait":
            return f"W:{int(action['ticks'])}"
        if kind == "wait_decision":
            return f"D:{int(action.get('max_ticks', WAIT_DECISION_TICKS))}"
        raise ValueError(f"unsupported search action: {action}")

    def _branch_snapshot_fast(self, snapshot_id: int, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not actions:
            return []
        tokens = " ".join(self._branch_action_token(action) for action in actions)
        response = self.env._command(f"BRANCH_SNAPSHOT_FAST {snapshot_id} {len(actions)} {tokens}")
        branches = response.get("branches")
        if not response.get("ok") or not isinstance(branches, list) or len(branches) != len(actions):
            raise RuntimeError(f"branch evaluation failed: {response}")
        return branches

    @staticmethod
    def _diversity_key(action: dict[str, Any]) -> tuple[Any, ...]:
        kind = action.get("type")
        if kind == "plant":
            return "plant", int(action["packet"]), int(action["row"])
        if kind == "shovel":
            return "shovel", int(action["row"])
        if kind == "wait":
            return "wait", int(action["ticks"]) // 60
        return ("wait_decision",) if kind == "wait_decision" else (str(kind),)

    def _branch_result(self, node: _SearchNode, action: dict[str, Any], branch: dict[str, Any]) -> tuple[_SearchNode | None, tuple[float, int | None, int]]:
        if not branch.get("ok") or branch.get("observation") is None:
            return None, (float("-inf"), None, node.elapsed_ticks)
        obs = branch["observation"]
        done = bool(obs.get("terminal"))
        advanced = max(int(branch.get("ticks_advanced", action.get("ticks", 0) if action["type"] == "wait" else 0)),
                       int(obs["tick"]) - int(node.observation["tick"]), 0)
        remaining = self.horizon_ticks - node.elapsed_ticks
        if advanced > remaining:
            raise RuntimeError(f"search action exceeded horizon: {action}")
        elapsed = node.elapsed_ticks + advanced
        snapshot_id = int(branch["snapshot_id"]) if branch.get("snapshot_id") is not None else None
        if not done and snapshot_id is None:
            raise RuntimeError(f"nonterminal branch returned no snapshot: {branch}")
        if snapshot_id is not None:
            self._snapshots.add(snapshot_id)
        try:
            events = branch.get("events", {})
            discount = node.discount * VALUE_GAMMA ** (advanced / DISCOUNT_REFERENCE_TICKS)
            path_return = node.path_return + discount * self._transition_reward(events, done, obs["result"] == 1)
            output = None if done else self._model_output(obs, node.output["hidden"] if node.output is not None else None,
                                                         action, advanced, events)
            score = path_return + discount * (0.0 if done else self._leaf_value(obs))
            outcome = int(obs["result"] == 1) if done else None
            leaf = score, outcome, elapsed
            if done or elapsed >= self.horizon_ticks:
                if snapshot_id is not None:
                    self._release(snapshot_id)
                return None, leaf
            assert snapshot_id is not None
            same_tick = node.same_tick_actions + 1 if advanced == 0 and action["type"] in ("plant", "shovel") else 0
            return _SearchNode(obs, snapshot_id, output, same_tick, elapsed, path_return, discount, score,
                               self._diversity_key(action)), leaf
        except Exception:
            if snapshot_id is not None:
                self._release(snapshot_id)
            raise

    def _expand_actions(self, node: _SearchNode, actions: list[dict[str, Any]]) -> list[tuple[_SearchNode | None, tuple[float, int | None, int]]]:
        remaining = self.horizon_ticks - node.elapsed_ticks
        fitted = [item for action in actions if (item := fit_action_to_remaining(action, remaining)) is not None]
        branches = self._branch_snapshot_fast(node.snapshot_id, fitted)
        return [self._branch_result(node, action, branch) for action, branch in zip(fitted, branches)]

    def _prune(self, nodes: list[_SearchNode]) -> list[_SearchNode]:
        ranked = sorted(nodes, key=lambda node: node.score + 0.05 * min(1.0, node.elapsed_ticks / self.horizon_ticks), reverse=True)
        selected: list[_SearchNode] = []
        groups: set[tuple[Any, ...]] = set()
        for node in ranked:
            if node.diversity_key not in groups:
                selected.append(node)
                groups.add(node.diversity_key)
                if len(selected) == self.beam_width:
                    break
        ids = {id(node) for node in selected}
        for node in ranked:
            if len(selected) == self.beam_width:
                break
            if id(node) not in ids:
                selected.append(node)
                ids.add(id(node))
        for node in nodes:
            if id(node) not in ids:
                self._release(node.snapshot_id)
        return selected

    def _partial_leaf(self, node: _SearchNode) -> tuple[float, int | None, int]:
        shortfall = max(0.0, 1.0 - node.elapsed_ticks / self.horizon_ticks)
        return node.score - SEARCH_PARTIAL_SHORTFALL_PENALTY * shortfall, None, node.elapsed_ticks

    def _search_branch(self, child: _SearchNode | None, leaf: tuple[float, int | None, int]) -> tuple[float, int | None, int, int]:
        simulations = 1
        leaves: list[tuple[float, int | None, int]] = []
        if child is None:
            return leaf[0], leaf[1], simulations, leaf[2]
        beam = [child]
        while beam and simulations < self.simulation_budget:
            expanded: list[_SearchNode] = []
            processed = 0
            for index, node in enumerate(beam):
                if simulations >= self.simulation_budget:
                    break
                processed = index + 1
                candidates = self.candidate_generator.actions(
                    node.observation, node.output, self.candidate_limit, False,
                    self.horizon_ticks - node.elapsed_ticks,
                    node.same_tick_actions < self.max_same_tick_actions,
                )
                available = self.simulation_budget - simulations
                limited = len(candidates) > available
                candidates = candidates[:available]
                if not candidates:
                    leaves.append(self._partial_leaf(node))
                    self._release(node.snapshot_id)
                    continue
                branch_results = self._expand_actions(node, candidates)
                simulations += len(candidates)
                for next_node, next_leaf in branch_results:
                    (leaves if next_node is None else expanded).append(next_leaf if next_node is None else next_node)
                if limited:
                    leaves.append(self._partial_leaf(node))
                self._release(node.snapshot_id)
            for node in beam[processed:]:
                leaves.append(self._partial_leaf(node))
                self._release(node.snapshot_id)
            if simulations >= self.simulation_budget:
                if expanded:
                    for node in self._prune(expanded):
                        leaves.append(self._partial_leaf(node))
                        self._release(node.snapshot_id)
                beam = []
            elif expanded:
                beam = self._prune(expanded)
            else:
                beam = []
        for node in beam:
            leaves.append(self._partial_leaf(node))
            self._release(node.snapshot_id)
        if not leaves:
            return float("-inf"), None, simulations, 0
        best = max(leaves, key=lambda item: self._result_key(item[1], item[0]))
        return best[0], best[1], simulations, best[2]

    @staticmethod
    def _fallback_action(observation: dict[str, Any]) -> dict[str, Any]:
        legal = observation["legal_actions"]
        if legal.get("wait", True):
            return {"type": "wait_decision", "max_ticks": WAIT_DECISION_TICKS}
        if legal["plants"]:
            return {"type": "plant", **legal["plants"][0]}
        if legal["shovels"]:
            col, row = legal["shovels"][0]
            return {"type": "shovel", "col": col, "row": row}
        raise RuntimeError("environment exposed no legal action")

    @torch.inference_mode()
    def advice(self, observation: dict[str, Any], hidden: Tensor | None = None,
               previous_action: dict[str, Any] | None = None, delta_ticks: int = 0,
               events: dict[str, Any] | None = None) -> SearchAdvice:
        with self._speculative_fast() as root_snapshot:
            self._root_snapshot = root_snapshot
            self._snapshots = set()
            try:
                root_output = self._model_output(observation, hidden, previous_action, delta_ticks, events)
                root_actions = self.candidate_generator.actions(
                    observation, root_output, self.root_candidate_limit, True, self.horizon_ticks)
                root = _SearchNode(observation, root_snapshot, root_output, 0, 0, 0.0, 1.0,
                                   self._leaf_value(observation), ("root",))
                initial = self._expand_actions(root, root_actions)
                results: list[tuple[dict[str, Any], float, int | None, int]] = []
                total = 0
                for action, (child, leaf) in zip(root_actions, initial):
                    value, outcome, count, elapsed = self._search_branch(child, leaf)
                    total += count
                    if math.isfinite(value):
                        results.append((action, value, outcome, elapsed))
                if not results:
                    action = self._fallback_action(observation)
                    return SearchAdvice(action, [(action, self._leaf_value(observation))], [1.0], None, total, None, 0)
                results.sort(key=lambda item: self._result_key(item[2], item[1]), reverse=True)
                policy = self._policy_from_results(results)
                best = results[0]
                if len(results) == 1:
                    margin = None
                elif self._outcome_rank(best[2]) != self._outcome_rank(results[1][2]):
                    margin = float(self._outcome_rank(best[2]) - self._outcome_rank(results[1][2]))
                else:
                    margin = best[1] - results[1][1]
                return SearchAdvice(best[0], [(item[0], item[1]) for item in results], policy, margin,
                                    total, best[2], best[3])
            finally:
                for snapshot_id in tuple(self._snapshots):
                    self._release(snapshot_id)
