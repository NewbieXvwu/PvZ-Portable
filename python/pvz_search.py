"""Simulator beam search used to generate imitation targets."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from pvz_agent_model import WAIT_TICKS
from pvz_search_candidates import CandidateGenerator, fit_action_to_remaining, lane_pressure
from pvz_search_value import SearchValueModel
from pvz_value import DISCOUNT_REFERENCE_TICKS, VALUE_GAMMA

SEARCH_POLICY_MIN_SCALE = 0.05
SEARCH_PARTIAL_SHORTFALL_PENALTY = 0.15

# ``BRANCH_SNAPSHOT_FAST`` refuses more than this many specs in one command
# (``src/main.cpp`` bounds ``branchCount`` at 128), so callers that want to expand
# a wider action set have to chunk.  The search itself never gets near this --
# the root request saturates at 96 and internal nodes use ``candidate_limit`` --
# but the audit tools deliberately enumerate every legal placement.
BRANCH_BATCH_LIMIT = 128


@dataclass(frozen=True)
class SearchAdvice:
    action: dict[str, Any]
    candidates: list[tuple[dict[str, Any], float]]
    search_policy: list[float]
    best_second_margin: float | None
    simulation_count: int
    terminal_outcome: int | None
    search_elapsed_ticks: int
    # ``simulation_budget`` is not the depth of the search.  Every root candidate
    # costs one simulation before any line is extended, so the amount actually
    # available to ``_successive_halving`` is ``simulation_budget - screening_simulations``.
    # Recording the split here is what makes a budget ablation interpretable: a run
    # at ``simulation_budget=128`` spends half of it on screening and is therefore
    # not "half of 256", it is a different regime.
    root_actions_generated: int = 0
    root_candidates_screened: int = 0
    screening_simulations: int = 0
    depth_simulations: int = 0
    effective_depth_budget: int = 0


@dataclass
class _SearchNode:
    observation: dict[str, Any]
    snapshot_id: int
    state_hash: str
    same_tick_actions: int
    elapsed_ticks: int
    decision_count: int
    path_return: float
    discount: float
    score: float
    diversity_key: tuple[Any, ...]


@dataclass
class _RootState:
    action: dict[str, Any]
    beam: list[_SearchNode] = field(default_factory=list)
    leaves: list[tuple[float, int | None, int]] = field(default_factory=list)
    simulations: int = 0
    transpositions: dict[tuple[str, int, int, int], float] = field(default_factory=dict)


class SearchTeacher:
    """State-only simulator search. It never reads the student policy/value model."""

    def __init__(
        self,
        env: Any,
        value_model: SearchValueModel | None = None,
        beam_width: int = 3,
        candidate_limit: int = 8,
        horizon_ticks: int = 900,
        simulation_budget: int = 256,
        max_same_tick_actions: int = 2,
        max_decisions: int = 64,
    ) -> None:
        if min(beam_width, candidate_limit, horizon_ticks, simulation_budget, max_same_tick_actions, max_decisions) < 1:
            raise ValueError("search parameters must be positive")
        self.env = env
        self.value_model = value_model
        self.beam_width = beam_width
        self.candidate_limit = candidate_limit
        self.root_candidate_limit = max(4, min(max(12, candidate_limit * 3), max(4, simulation_budget // 4)))
        self.horizon_ticks = horizon_ticks
        self.simulation_budget = simulation_budget
        self.max_same_tick_actions = max_same_tick_actions
        self.max_decisions = max_decisions
        self.candidate_generator = CandidateGenerator()
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
        scale = max(SEARCH_POLICY_MIN_SCALE, math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores)))
        top = max(scores)
        weights = [
            math.exp((item[1] - top) / scale) if cls._outcome_rank(item[2]) == rank else 0.0
            for item in results
        ]
        total = sum(weights)
        return [weight / total for weight in weights]

    @staticmethod
    def _bootstrap_leaf_value(observation: dict[str, Any]) -> float:
        """Generic cold-start evaluator used only before SearchValueModel exists."""
        if observation["terminal"]:
            return 1.0 if observation["result"] == 1 else -1.0
        rows = {cell["row"] for cell in observation["cells"] if cell["row_type"] > 0}
        row_count = max(1, len(rows))
        plants = [plant for plant in observation["plants"]
                  if not plant.get("squished") and plant.get("health", 0) > 0]
        coverage = len({plant["row"] for plant in plants if plant["row"] in rows}) / row_count
        health = min(1.0, sum(
            max(0.0, min(1.0, plant.get("health", 0) / max(1, plant.get("max_health", 1))))
            for plant in plants
        ) / (row_count * 3.0))
        defense = min(1.0, sum(item.get("state") == 1 for item in observation.get("defenses", [])) / row_count)
        progress = min(max(observation["wave"] / max(1, observation["wave_count"]), 0.0), 1.0)
        sun = min(max(observation["sun"] / 1000.0, 0.0), 1.0)
        threat = min(1.0, sum(lane_pressure(observation).values()) / (row_count * 5.0))
        nearest = min((zombie["x"] for zombie in observation["zombies"]), default=900.0)
        breach = max(0.0, min(1.0, (420.0 - nearest) / 300.0))
        value = -0.12 + 0.18 * progress + 0.28 * coverage + 0.16 * health + 0.10 * sun + 0.10 * defense
        value -= 0.48 * threat + 0.22 * breach
        return max(-1.0, min(1.0, value))

    def _leaf_value(self, observation: dict[str, Any]) -> float:
        if observation["terminal"]:
            return 1.0 if observation["result"] == 1 else -1.0
        if self.value_model is not None:
            return max(-1.0, min(1.0, float(self.value_model.predict(observation))))
        return self._bootstrap_leaf_value(observation)

    @staticmethod
    def _transition_reward(events: dict[str, Any], terminal: bool, won: bool) -> float:
        reward = -min(0.20, max(0, events.get("plants_eaten", 0)) * 0.05)
        reward -= min(0.25, max(0, events.get("mower_triggered", 0)) * 0.25)
        if terminal:
            reward += 1.0 if won else -1.0
        return reward

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
        raise ValueError(f"unsupported search action: {action}")

    def _branch_snapshot_fast(self, snapshot_id: int, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not actions:
            return []
        if len(actions) > BRANCH_BATCH_LIMIT:
            raise ValueError(
                f"BRANCH_SNAPSHOT_FAST accepts at most {BRANCH_BATCH_LIMIT} specs per command, "
                f"got {len(actions)}"
            )
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
        raise ValueError(f"unsupported search action: {action}")

    def _branch_result(
        self,
        node: _SearchNode,
        action: dict[str, Any],
        branch: dict[str, Any],
    ) -> tuple[_SearchNode | None, tuple[float, int | None, int]]:
        if not branch.get("ok") or branch.get("observation") is None:
            return None, (float("-inf"), None, node.elapsed_ticks)
        observation = branch["observation"]
        done = bool(observation.get("terminal"))
        advanced = max(int(observation["tick"]) - int(node.observation["tick"]), 0)
        remaining = self.horizon_ticks - node.elapsed_ticks
        if advanced > remaining:
            raise RuntimeError(f"search action exceeded horizon: {action}")
        elapsed = node.elapsed_ticks + advanced
        decision_count = node.decision_count + 1
        snapshot_id = int(branch["snapshot_id"]) if branch.get("snapshot_id") is not None else None
        state_hash = str(branch.get("state_hash", ""))
        if not done and (snapshot_id is None or not state_hash):
            raise RuntimeError(f"nonterminal branch must return snapshot and state_hash: {branch}")
        if snapshot_id is not None:
            self._snapshots.add(snapshot_id)
        try:
            events = branch.get("events", {})
            discount = node.discount * VALUE_GAMMA ** (advanced / DISCOUNT_REFERENCE_TICKS)
            path_return = node.path_return + discount * self._transition_reward(
                events, done, observation["result"] == 1
            )
            score = path_return + discount * (0.0 if done else self._leaf_value(observation))
            outcome = int(observation["result"] == 1) if done else None
            leaf = score, outcome, elapsed
            if done or elapsed >= self.horizon_ticks or decision_count >= self.max_decisions:
                if snapshot_id is not None:
                    self._release(snapshot_id)
                return None, leaf
            assert snapshot_id is not None
            same_tick = (
                node.same_tick_actions + 1
                if advanced == 0 and action["type"] in ("plant", "shovel")
                else 0
            )
            return _SearchNode(
                observation,
                snapshot_id,
                state_hash,
                same_tick,
                elapsed,
                decision_count,
                path_return,
                discount,
                score,
                self._diversity_key(action),
            ), leaf
        except Exception:
            if snapshot_id is not None:
                self._release(snapshot_id)
            raise

    def _fit_to_remaining(self, node: _SearchNode, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop the actions that cannot fit in *node*'s remaining horizon.

        Callers that need to keep action and result aligned must expand through
        :meth:`_expand_paired`; re-deriving this list next to a separate expansion
        is what allowed the two to drift apart.
        """
        remaining = self.horizon_ticks - node.elapsed_ticks
        return [item for action in actions if (item := fit_action_to_remaining(action, remaining)) is not None]

    def _expand_paired(
        self,
        node: _SearchNode,
        actions: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[tuple[_SearchNode | None, tuple[float, int | None, int]]]]:
        """Branch *actions* and return ``(fitted_actions, results)`` positionally aligned.

        ``_branch_snapshot_fast`` only sees the *fitted* actions, so pairing its
        results back against the caller's original list silently misaligns every
        action with its neighbour's outcome as soon as one action is dropped for
        not fitting the remaining horizon.  Returning both lists from one place
        makes that impossible.
        """
        fitted = self._fit_to_remaining(node, actions)
        branches = self._branch_snapshot_fast(node.snapshot_id, fitted)
        return fitted, [self._branch_result(node, action, branch) for action, branch in zip(fitted, branches)]

    def _expand_actions(
        self,
        node: _SearchNode,
        actions: list[dict[str, Any]],
    ) -> list[tuple[_SearchNode | None, tuple[float, int | None, int]]]:
        return self._expand_paired(node, actions)[1]

    @staticmethod
    def _transposition_key(node: _SearchNode) -> tuple[str, int, int, int]:
        return node.state_hash, node.elapsed_ticks, node.decision_count, node.same_tick_actions

    def _prune(self, nodes: list[_SearchNode]) -> list[_SearchNode]:
        deduplicated: dict[tuple[str, int, int, int], _SearchNode] = {}
        for node in nodes:
            key = self._transposition_key(node)
            previous = deduplicated.get(key)
            if previous is None or node.score > previous.score:
                if previous is not None:
                    self._release(previous.snapshot_id)
                deduplicated[key] = node
            else:
                self._release(node.snapshot_id)
        ranked = sorted(
            deduplicated.values(),
            key=lambda node: node.score + 0.05 * min(1.0, node.elapsed_ticks / self.horizon_ticks),
            reverse=True,
        )
        selected: list[_SearchNode] = []
        groups: set[tuple[Any, ...]] = set()
        for node in ranked:
            if node.diversity_key not in groups:
                selected.append(node)
                groups.add(node.diversity_key)
                if len(selected) == self.beam_width:
                    break
        selected_ids = {id(node) for node in selected}
        for node in ranked:
            if len(selected) == self.beam_width:
                break
            if id(node) not in selected_ids:
                selected.append(node)
                selected_ids.add(id(node))
        for node in nodes:
            if id(node) not in selected_ids:
                self._release(node.snapshot_id)
        return selected

    def _partial_leaf(self, node: _SearchNode) -> tuple[float, int | None, int]:
        shortfall = max(0.0, 1.0 - node.elapsed_ticks / self.horizon_ticks)
        return node.score - SEARCH_PARTIAL_SHORTFALL_PENALTY * shortfall, None, node.elapsed_ticks

    def _accept_node(self, root: _RootState, node: _SearchNode) -> bool:
        key = self._transposition_key(node)
        previous = root.transpositions.get(key)
        if previous is not None and previous >= node.score:
            self._release(node.snapshot_id)
            return False
        root.transpositions[key] = node.score
        return True

    def _advance_root(self, root: _RootState, budget: int) -> int:
        spent = 0
        while root.beam and spent < budget:
            expanded: list[_SearchNode] = []
            processed = 0
            for index, node in enumerate(root.beam):
                available = budget - spent
                if available <= 0:
                    break
                processed = index + 1
                candidates = self.candidate_generator.actions(
                    node.observation,
                    self.candidate_limit,
                    False,
                    self.horizon_ticks - node.elapsed_ticks,
                    node.same_tick_actions < self.max_same_tick_actions,
                )[:available]
                if not candidates:
                    root.leaves.append(self._partial_leaf(node))
                    self._release(node.snapshot_id)
                    continue
                for next_node, next_leaf in self._expand_actions(node, candidates):
                    spent += 1
                    root.simulations += 1
                    if next_node is None:
                        root.leaves.append(next_leaf)
                    elif self._accept_node(root, next_node):
                        expanded.append(next_node)
                self._release(node.snapshot_id)
                if spent >= budget:
                    break
            expanded.extend(root.beam[processed:])
            root.beam = self._prune(expanded) if expanded else []
        return spent

    def _estimate_root(self, root: _RootState) -> tuple[float, int | None, int]:
        candidates = list(root.leaves)
        candidates.extend(self._partial_leaf(node) for node in root.beam)
        if not candidates:
            return float("-inf"), None, 0
        return max(candidates, key=lambda item: self._result_key(item[1], item[0]))

    def _finish_root(self, root: _RootState) -> tuple[float, int | None, int]:
        for node in root.beam:
            root.leaves.append(self._partial_leaf(node))
            self._release(node.snapshot_id)
        root.beam = []
        return self._estimate_root(root)

    def _initialize_roots(self, root: _SearchNode, actions: list[dict[str, Any]]) -> list[_RootState]:
        states: list[_RootState] = []
        fitted, initial = self._expand_paired(root, actions)
        for action, (child, leaf) in zip(fitted, initial):
            state = _RootState(action=action, simulations=1)
            if child is None:
                state.leaves.append(leaf)
            else:
                state.beam.append(child)
                state.transpositions[self._transposition_key(child)] = child.score
            states.append(state)
        return states

    @staticmethod
    def _root_diversity_key(action: dict[str, Any]) -> tuple[Any, ...]:
        kind = action["type"]
        if kind == "plant":
            return kind, action["packet"], action["row"], action["col"] // 3
        if kind == "shovel":
            return kind, action["row"], action["col"] // 3
        return kind, action["ticks"]

    def _root_rank(self, root: _RootState) -> tuple[int, float]:
        """Rank a root state by its best estimate.

        Extracted so sorting evaluates ``_estimate_root`` exactly once per element.
        """
        score, outcome, _ = self._estimate_root(root)
        return self._result_key(outcome, score)

    def _screen_root_states(self, roots: list[_RootState]) -> list[_RootState]:
        ranked = sorted(roots, key=self._root_rank, reverse=True)
        selected: list[_RootState] = []
        groups: set[tuple[Any, ...]] = set()
        for root in ranked:
            key = self._root_diversity_key(root.action)
            if key not in groups:
                selected.append(root)
                groups.add(key)
                if len(selected) == self.root_candidate_limit:
                    break
        selected_ids = {id(root) for root in selected}
        for root in ranked:
            if len(selected) == self.root_candidate_limit:
                break
            if id(root) not in selected_ids:
                selected.append(root)
                selected_ids.add(id(root))
        for root in roots:
            if id(root) not in selected_ids:
                self._finish_root(root)
        for root in selected:
            root.simulations = 0
        return selected

    def _successive_halving(
        self,
        roots: list[_RootState],
        initial_cost: int,
    ) -> tuple[list[tuple[dict[str, Any], float, int | None, int]], int]:
        if not roots:
            return [], initial_cost
        remaining = max(0, self.simulation_budget - initial_cost)
        active = list(roots)
        stage_quota = max(2, min(4, self.candidate_limit))
        while remaining > 0 and any(root.beam for root in active):
            active.sort(key=self._root_rank, reverse=True)
            progressed = False
            for root in active:
                if remaining <= 0:
                    break
                if not root.beam:
                    continue
                allocation = min(stage_quota, remaining)
                spent = self._advance_root(root, allocation)
                remaining -= spent
                progressed = progressed or spent > 0
            if not progressed:
                break
            if len(active) > 1:
                active.sort(key=self._root_rank, reverse=True)
                keep = max(1, (len(active) + 1) // 2)
                for root in active[keep:]:
                    self._finish_root(root)
                active = active[:keep]
            stage_quota = min(stage_quota * 2, self.simulation_budget)

        results: list[tuple[dict[str, Any], float, int | None, int]] = []
        for root in roots:
            value, outcome, elapsed = self._finish_root(root)
            if math.isfinite(value):
                results.append((root.action, value, outcome, elapsed))
        return results, sum(root.simulations for root in roots)

    @staticmethod
    def _fallback_action(observation: dict[str, Any]) -> dict[str, Any]:
        legal = observation["legal_actions"]
        if legal.get("wait", True):
            return {"type": "wait", "ticks": WAIT_TICKS[0]}
        if legal["plants"]:
            return {"type": "plant", **legal["plants"][0]}
        if legal["shovels"]:
            col, row = legal["shovels"][0]
            return {"type": "shovel", "col": col, "row": row}
        raise RuntimeError("environment exposed no legal action")

    def one_step_children(
        self,
        observation: dict[str, Any],
        actions: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
        """Return the observation each action leads to, without searching past it.

        These are the counterfactual leaves the value model is asked to score most
        often: the search branches *every* root candidate before it extends a single
        line, so one-step children dominate ``_leaf_value`` calls.  The audit tools
        need to look at them directly, and re-deriving the snapshot protocol outside
        this class is how the two would drift apart.

        Returns ``(action, child_observation)`` pairs, with ``None`` for an action
        that ended the level.  Actions come back alongside their child so a caller
        cannot misalign them: actions that do not fit ``horizon_ticks`` are dropped
        here, and a caller that paired the results against its own list would
        silently shift every remaining action onto its neighbour's outcome.

        Expands in ``BRANCH_BATCH_LIMIT`` chunks, because the protocol refuses a
        wider command than that and the audit tools deliberately enumerate every
        legal placement.
        """
        children: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        with self._speculative_fast() as root_snapshot:
            self._root_snapshot = root_snapshot
            self._snapshots = set()
            try:
                root = _SearchNode(observation, root_snapshot, "root", 0, 0, 0, 0.0, 1.0, 0.0, ("root",))
                fitted = self._fit_to_remaining(root, actions)
                for start in range(0, len(fitted), BRANCH_BATCH_LIMIT):
                    chunk = fitted[start:start + BRANCH_BATCH_LIMIT]
                    for action, branch in zip(chunk, self._branch_snapshot_fast(root_snapshot, chunk)):
                        child, _ = self._branch_result(root, action, branch)
                        children.append((action, None if child is None else child.observation))
                return children
            finally:
                for snapshot_id in tuple(self._snapshots):
                    self._release(snapshot_id)

    @property
    def root_request_limit(self) -> int:
        """How many raw root actions ``advice`` asks the generator for.

        The second term is slack so diversity screening has something to choose
        from; the first one is the budget guard.  Without it a small budget is
        spent *entirely* on screening -- at ``budget=64, candidate_limit=8`` the
        generator would be asked for 64 actions out of a budget of 64, leaving
        ``_successive_halving`` nothing to expand with.  Exposed because the audit
        tools have to reproduce the search's own candidate request exactly.
        """
        return min(max(1, self.simulation_budget // 2), max(32, self.root_candidate_limit * 4))

    def advice(self, observation: dict[str, Any]) -> SearchAdvice:
        with self._speculative_fast() as root_snapshot:
            self._root_snapshot = root_snapshot
            self._snapshots = set()
            try:
                root_actions = self.candidate_generator.actions(
                    observation,
                    self.root_request_limit,
                    True,
                    self.horizon_ticks,
                )
                root = _SearchNode(
                    observation,
                    root_snapshot,
                    "root",
                    0,
                    0,
                    0,
                    0.0,
                    1.0,
                    # The root node is only ever used as an expansion origin
                    # (snapshot/observation/elapsed/discount/decision_count/same_tick_actions);
                    # its own score is never read, so evaluating the leaf value here would be
                    # a wasted value-model forward pass on every decision.
                    0.0,
                    ("root",),
                )
                initial_states = self._initialize_roots(root, root_actions)
                root_states = self._screen_root_states(initial_states)
                # Reserve exactly what screening spent.  ``_initialize_roots`` drops the
                # actions that do not fit the horizon, so ``len(initial_states)`` can be
                # smaller than ``len(root_actions)``; reserving the request rather than the
                # spend would quietly hand the depth search less budget than it has.
                screening_cost = len(initial_states)
                results, depth_cost = self._successive_halving(root_states, screening_cost)
                total = screening_cost + depth_cost
                accounting = {
                    "root_actions_generated": len(root_actions),
                    "root_candidates_screened": len(root_states),
                    "screening_simulations": screening_cost,
                    "depth_simulations": depth_cost,
                    "effective_depth_budget": max(0, self.simulation_budget - screening_cost),
                }
                if not results:
                    action = self._fallback_action(observation)
                    return SearchAdvice(
                        action, [(action, self._leaf_value(observation))], [1.0], None, total, None, 0,
                        **accounting,
                    )
                results.sort(key=lambda item: self._result_key(item[2], item[1]), reverse=True)
                policy = self._policy_from_results(results)
                best = results[0]
                if len(results) == 1:
                    margin = None
                elif self._outcome_rank(best[2]) != self._outcome_rank(results[1][2]):
                    margin = float(self._outcome_rank(best[2]) - self._outcome_rank(results[1][2]))
                else:
                    margin = best[1] - results[1][1]
                return SearchAdvice(
                    best[0],
                    [(item[0], item[1]) for item in results],
                    policy,
                    margin,
                    total,
                    best[2],
                    best[3],
                    **accounting,
                )
            finally:
                for snapshot_id in tuple(self._snapshots):
                    self._release(snapshot_id)
