"""PvZ gameplay policy, search teacher, and search-policy distillation."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import Tensor

from pvz_agent_model import (
    GameplayModelV1,
    MODEL_ARCHITECTURE_VERSION,
    MODEL_CONFIG,
    WAIT_DECISION_TICKS,
    WAIT_TICKS,
    hard_behavior_cloning_loss,
    predict_action,
    resolve_device,
    select_action,
)
from pvz_teacher import TeacherAdvice, TeacherPolicy, teacher_advice


DISCOUNT_REFERENCE_TICKS = 300
VALUE_GAMMA = 0.99
VALUE_SEMANTICS = "discounted_terminal_v1"
SEARCH_GAMMA = VALUE_GAMMA
SEARCH_POLICY_TEMPERATURE = 0.20


def discounted_terminal_value(won: bool, remaining_ticks: int) -> float:
    sign = 1.0 if won else -1.0
    return sign * VALUE_GAMMA ** (max(0, int(remaining_ticks)) / DISCOUNT_REFERENCE_TICKS)


def behavior_cloning_loss(
    model: GameplayModelV1,
    output: dict[str, Any],
    observation: dict[str, Any],
    action: dict[str, Any],
    plant_weight: float = 1.0,
    candidate_actions: list[dict[str, Any]] | None = None,
    search_policy: list[float] | None = None,
) -> Tensor:
    """Distill search probabilities exactly; use weighted hard BC only without search labels."""
    if candidate_actions and search_policy and len(candidate_actions) == len(search_policy):
        log_probs = torch.stack([
            select_action(model, output, observation, action=candidate)[1]
            for candidate in candidate_actions
        ])
        probabilities = torch.tensor(search_policy, dtype=log_probs.dtype, device=log_probs.device).clamp_min(0.0)
        total = probabilities.sum()
        if float(total.item()) > 0.0:
            return -((probabilities / total) * log_probs).sum()
    return hard_behavior_cloning_loss(model, output, observation, action, plant_weight)


@dataclass
class _SearchNode:
    observation: dict[str, Any]
    snapshot_id: int
    output: dict[str, Any] | None
    sunflower_placements: int
    same_tick_actions: int
    elapsed_ticks: int
    decisions: int
    path_return: float
    discount: float
    score: float


class SearchTeacher:
    """Per-root beam search with tick-based horizons and cumulative transition rewards."""

    def __init__(self, env: Any, model: GameplayModelV1 | None = None, beam_width: int = 3,
                 max_decisions: int = 6, candidate_limit: int = 6, horizon_ticks: int = 900,
                 learned_value_weight: float = 0.15, max_same_tick_actions: int = 2) -> None:
        if min(beam_width, max_decisions, candidate_limit, horizon_ticks, max_same_tick_actions) < 1:
            raise ValueError("search width, decisions, candidates, horizon, and same-tick limit must be positive")
        if horizon_ticks < WAIT_DECISION_TICKS:
            raise ValueError(f"search horizon must be at least {WAIT_DECISION_TICKS} ticks")
        if not 0.0 <= learned_value_weight <= 1.0:
            raise ValueError("learned value weight must be from 0 to 1")
        self.env = env
        self.model = model
        self.beam_width = beam_width
        self.max_decisions = max_decisions
        self.candidate_limit = candidate_limit
        self.root_candidate_limit = max(16, candidate_limit * 3)
        self.horizon_ticks = horizon_ticks
        self.learned_value_weight = learned_value_weight
        self.max_same_tick_actions = max_same_tick_actions
        self.policy = TeacherPolicy()
        self._snapshots: set[int] = set()
        self._root_snapshot = 0

    def record_action(self, observation: dict[str, Any], action: dict[str, Any]) -> None:
        self.policy.record_action(observation, action)

    @staticmethod
    def _action_key(action: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        return tuple(sorted(action.items()))

    @staticmethod
    def _outcome_rank(outcome: int | None) -> int:
        if outcome == 1:
            return 2
        if outcome is None:
            return 1
        return 0

    @classmethod
    def _result_key(cls, outcome: int | None, score: float) -> tuple[int, float]:
        return cls._outcome_rank(outcome), score

    @classmethod
    def _policy_from_results(cls, results: list[tuple[dict[str, Any], float, int | None, int, int]]) -> list[float]:
        if not results:
            return []
        best_rank = max(cls._outcome_rank(item[2]) for item in results)
        eligible = [cls._outcome_rank(item[2]) == best_rank for item in results]
        top = max(item[1] for item, keep in zip(results, eligible) if keep)
        weights = [math.exp((item[1] - top) / SEARCH_POLICY_TEMPERATURE) if keep else 0.0
                   for item, keep in zip(results, eligible)]
        total = sum(weights)
        return [weight / total for weight in weights]

    @staticmethod
    def _state_heuristic(observation: dict[str, Any]) -> float:
        if observation["terminal"]:
            return 1.0 if observation["result"] == 1 else -1.0
        rows = {cell["row"] for cell in observation["cells"] if cell["row_type"] > 0}
        attackers: set[int] = set()
        sunflowers = 0
        blockers = 0
        for plant in observation["plants"]:
            kind = plant["imitater_type"] if plant["type"] == 48 else plant["type"]
            if kind in (0, 5) and not plant["squished"] and plant["health"] > 0:
                attackers.add(plant["row"])
            elif kind == 1:
                sunflowers += 1
            elif kind == 3:
                blockers += 1
        coverage = len(attackers) / max(1, len(rows))
        economy = min(sunflowers / 6.0, 1.0)
        defense = min(blockers / max(1, len(rows)), 1.0)
        progress = min(max(observation["wave"] / max(1, observation["wave_count"]), 0.0), 1.0)
        sun = min(max(observation["sun"] / 1000.0, 0.0), 1.0)
        threat = 0.0
        for zombie in observation["zombies"]:
            urgency = max(0.0, min(1.0, (700.0 - zombie["x"]) / 600.0))
            health = sum(max(0.0, zombie.get(key, 0.0))
                         for key in ("body_health", "helm_health", "shield_health"))
            threat += urgency * min(1.0, health / 1200.0)
        threat = min(1.0, threat / max(1, len(rows)))
        return max(-1.0, min(1.0,
            -0.2 + 0.22 * progress + 0.14 * economy + 0.07 * sun
            + 0.38 * coverage + 0.08 * defense - 0.48 * threat))

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
        if self.model is None:
            return None
        return self.model.step(observation, hidden, previous_action, delta_ticks, events)

    def _leaf_value(self, observation: dict[str, Any], output: dict[str, Any] | None) -> float:
        heuristic = self._state_heuristic(observation)
        if output is None or self.learned_value_weight <= 0.0:
            return heuristic
        learned = max(-1.0, min(1.0, float(output["value"].item())))
        w = self.learned_value_weight
        return (1.0 - w) * heuristic + w * learned

    def _snapshot_fast(self) -> int:
        response = self.env._command("SNAPSHOT_FAST")
        if not response.get("ok") or "snapshot_id" not in response:
            raise RuntimeError(f"environment could not save a fast snapshot: {response}")
        return int(response["snapshot_id"])

    def _restore_fast(self, snapshot_id: int) -> None:
        response = self.env._command(f"RESTORE_FAST {snapshot_id}")
        if not response.get("ok"):
            raise RuntimeError(f"environment could not restore fast snapshot {snapshot_id}: {response}")

    def _release_fast(self, snapshot_id: int) -> None:
        response = self.env._command(f"DROP_SNAPSHOT_FAST {snapshot_id}")
        if not response.get("ok"):
            raise RuntimeError(f"environment could not release fast snapshot {snapshot_id}: {response}")

    @contextmanager
    def _speculative_fast(self) -> Iterator[int]:
        if not self.env._reset_done or self.env.episode is None:
            raise RuntimeError("call reset() before search")
        operation_count = len(self.env.episode["operations"])
        final_state = self.env.episode["final_state"]
        snapshot_id = self._snapshot_fast()
        try:
            yield snapshot_id
        finally:
            self._restore_fast(snapshot_id)
            del self.env.episode["operations"][operation_count:]
            self.env.episode["final_state"] = final_state
            self._release_fast(snapshot_id)

    def _release(self, snapshot_id: int) -> None:
        if snapshot_id != self._root_snapshot and snapshot_id in self._snapshots:
            self._release_fast(snapshot_id)
            self._snapshots.remove(snapshot_id)

    @staticmethod
    def _packet_type(observation: dict[str, Any], packet_index: int) -> int:
        packet = next(item for item in observation["packets"] if item["index"] == packet_index)
        return packet["imitater_type"] if packet["type"] == 48 else packet["type"]

    @classmethod
    def _is_sunflower_action(cls, observation: dict[str, Any], action: dict[str, Any]) -> bool:
        return action.get("type") == "plant" and cls._packet_type(observation, action["packet"]) == 1

    @staticmethod
    def _lane_pressure(observation: dict[str, Any]) -> dict[int, float]:
        pressure = {cell["row"]: 0.0 for cell in observation["cells"] if cell["row_type"] > 0}
        for zombie in observation["zombies"]:
            urgency = max(0.0, min(1.5, (760.0 - zombie["x"]) / 520.0))
            health = sum(max(0.0, zombie.get(key, 0.0))
                         for key in ("body_health", "helm_health", "shield_health"))
            pressure[zombie["row"]] = pressure.get(zombie["row"], 0.0) + urgency * (1.0 + min(4.0, health / 400.0))
        return pressure

    def _coverage_plants(self, observation: dict[str, Any], placements: list[dict[str, Any]], count: int = 1) -> list[dict[str, Any]]:
        seed_type = self._packet_type(observation, placements[0]["packet"])
        pressure = self._lane_pressure(observation)
        zombies = observation["zombies"]

        def score(item: dict[str, Any]) -> float:
            row, col = item["row"], item["col"]
            lane = pressure.get(row, 0.0)
            plant_x = 80.0 + 80.0 * col
            if seed_type == 1:
                return -2.0 * lane - abs(col - 1) * 0.4
            if seed_type == 2:
                cluster = sum(1.0 + max(0.0, (520.0 - zombie["x"]) / 300.0)
                              for zombie in zombies
                              if abs(zombie["row"] - row) <= 1 and abs(zombie["x"] - plant_x) <= 150)
                return cluster * 6.0 + lane
            preferred = 4 if seed_type == 3 else 3 if seed_type == 4 else 2 if seed_type in (0, 5) else 3
            multiplier = 2.2 if seed_type == 3 else 1.8 if seed_type == 4 else 2.4 if seed_type in (0, 5) else 1.0
            return multiplier * lane - abs(col - preferred) * 0.25

        ranked = sorted(placements, key=score, reverse=True)
        return [{"type": "plant", **item} for item in ranked[:count]]

    @staticmethod
    def _adaptive_wait(observation: dict[str, Any]) -> dict[str, Any]:
        nearest = min((zombie["x"] for zombie in observation["zombies"]), default=9999.0)
        if nearest < 360:
            ticks = 60
        elif nearest < 560:
            ticks = 150
        else:
            ticks = 300
        return {"type": "wait", "ticks": ticks}

    @staticmethod
    def _fit_action_to_remaining(action: dict[str, Any], remaining_ticks: int) -> dict[str, Any] | None:
        if remaining_ticks <= 0:
            return None
        fitted = dict(action)
        if fitted.get("type") == "wait":
            fitted["ticks"] = min(int(fitted.get("ticks", remaining_ticks)), remaining_ticks)
            if fitted["ticks"] < 1:
                return None
        elif fitted.get("type") == "wait_decision":
            fitted["max_ticks"] = min(int(fitted.get("max_ticks", WAIT_DECISION_TICKS)), remaining_ticks)
            if fitted["max_ticks"] < 1:
                return None
        return fitted

    def _model_proposals(self, observation: dict[str, Any], output: dict[str, Any] | None,
                         limit: int) -> list[dict[str, Any]]:
        if self.model is None or output is None or limit <= 0:
            return []
        legal = observation["legal_actions"]
        type_logits = output["type_logits"].clone()
        if not legal["plants"]:
            type_logits[0] = -1e9
        if not legal["shovels"]:
            type_logits[1] = -1e9
        if not legal.get("wait", True):
            type_logits[2:] = -1e9
        type_logp = torch.log_softmax(type_logits, dim=0)
        scored: list[tuple[float, dict[str, Any]]] = []

        valid_packets = sorted({item["packet"] for item in legal["plants"]})
        if valid_packets:
            packet_logits = output["packet_logits"].clone()
            allowed = set(valid_packets)
            for index, packet_id in enumerate(output["packet_ids"]):
                if packet_id not in allowed:
                    packet_logits[index] = -1e9
            packet_logp = torch.log_softmax(packet_logits, dim=0)
            for packet_index, packet in enumerate(output["packet_ids"]):
                if packet not in allowed:
                    continue
                cell_logits = self.model.plant_cell_scores(output, packet).clone()
                valid_cells = {a["row"] * 9 + a["col"] for a in legal["plants"] if a["packet"] == packet}
                for cell in range(54):
                    if cell not in valid_cells:
                        cell_logits[cell] = -1e9
                cell_logp = torch.log_softmax(cell_logits, dim=0)
                top_cells = sorted(valid_cells, key=lambda cell: float(cell_logp[cell].item()), reverse=True)[:2]
                for cell in top_cells:
                    joint = float((type_logp[0] + packet_logp[packet_index] + cell_logp[cell]).item())
                    scored.append((joint, {"type": "plant", "packet": packet, "col": cell % 9, "row": cell // 9}))

        if legal["shovels"]:
            cell_logits = self.model.shovel_cell_scores(output).clone()
            valid_cells = {row * 9 + col for col, row in legal["shovels"]}
            for cell in range(54):
                if cell not in valid_cells:
                    cell_logits[cell] = -1e9
            cell_logp = torch.log_softmax(cell_logits, dim=0)
            for cell in sorted(valid_cells, key=lambda item: float(cell_logp[item].item()), reverse=True)[:2]:
                scored.append((float((type_logp[1] + cell_logp[cell]).item()),
                               {"type": "shovel", "col": cell % 9, "row": cell // 9}))

        if legal.get("wait", True):
            wait_logp = torch.log_softmax(output["wait_logits"], dim=0)
            for index, ticks in enumerate(WAIT_TICKS):
                scored.append((float((type_logp[2] + wait_logp[index]).item()), {"type": "wait", "ticks": ticks}))
            scored.append((float(type_logp[3].item()), {"type": "wait_decision", "max_ticks": WAIT_DECISION_TICKS}))

        scored.sort(key=lambda item: item[0], reverse=True)
        proposals: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, Any], ...]] = set()
        for _, action in scored:
            key = self._action_key(action)
            if key not in seen:
                seen.add(key)
                proposals.append(action)
            if len(proposals) >= limit:
                break
        return proposals

    def _candidate_actions(self, observation: dict[str, Any], output: dict[str, Any] | None,
                           sunflower_placements: int, limit: int, root: bool,
                           remaining_ticks: int, allow_instant: bool = True) -> list[dict[str, Any]]:
        legal = observation["legal_actions"]
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, Any], ...]] = set()

        def add(action: dict[str, Any]) -> None:
            if len(candidates) >= limit:
                return
            if not allow_instant and action.get("type") in ("plant", "shovel"):
                return
            fitted = self._fit_action_to_remaining(action, remaining_ticks)
            if fitted is None:
                return
            key = self._action_key(fitted)
            if key not in seen:
                seen.add(key)
                candidates.append(fitted)

        heuristic = [action for action, _ in teacher_advice(observation, sunflower_placements).candidates]
        structured: list[dict[str, Any]] = []
        if allow_instant:
            by_packet: dict[int, list[dict[str, Any]]] = {}
            for placement in legal["plants"]:
                by_packet.setdefault(placement["packet"], []).append(placement)
            for packet in sorted(by_packet):
                structured.extend(self._coverage_plants(observation, by_packet[packet], 1))
            if legal["shovels"]:
                health = {(plant["col"], plant["row"]): plant["health"] / max(1, plant["max_health"])
                          for plant in observation["plants"]}
                col, row = min(legal["shovels"], key=lambda cell: health.get(cell, 1.0))
                structured.append({"type": "shovel", "col": col, "row": row})

        if legal.get("wait", True):
            add({"type": "wait_decision", "max_ticks": min(WAIT_DECISION_TICKS, remaining_ticks)})
            if root:
                for ticks in WAIT_TICKS:
                    add({"type": "wait", "ticks": ticks})
            else:
                add(self._adaptive_wait(observation))
                if not allow_instant:
                    for ticks in WAIT_TICKS:
                        add({"type": "wait", "ticks": ticks})

        if root:
            for action in structured:
                add(action)
            for action in heuristic[:4]:
                add(action)
            for action in self._model_proposals(observation, output, 4):
                add(action)
        else:
            heuristic_budget = 2
            structured_budget = 1
            model_budget = 1
            for action in heuristic[:heuristic_budget]:
                add(action)
            for action in structured[:structured_budget]:
                add(action)
            for action in self._model_proposals(observation, output, model_budget):
                add(action)
            for pool in (heuristic[heuristic_budget:], structured[structured_budget:],
                         self._model_proposals(observation, output, limit)):
                for action in pool:
                    add(action)
                    if len(candidates) >= limit:
                        break
                if len(candidates) >= limit:
                    break
        return candidates

    def _step_from_snapshot(self, node: _SearchNode, action: dict[str, Any]) -> tuple[_SearchNode | None, tuple[float, int | None, int, int], int]:
        remaining = self.horizon_ticks - node.elapsed_ticks
        fitted_action = self._fit_action_to_remaining(action, remaining)
        if fitted_action is None:
            return None, (float("-inf"), None, node.elapsed_ticks, node.decisions), 0
        self._restore_fast(node.snapshot_id)
        next_observation, _, done, _, info = self.env.step(fitted_action)
        if not info.get("ok"):
            return None, (float("-inf"), None, node.elapsed_ticks, node.decisions), 0
        reported = info.get("ticks_advanced", fitted_action.get("ticks", 0) if fitted_action["type"] == "wait" else 0)
        advanced = max(int(reported), int(next_observation["tick"]) - int(node.observation["tick"]), 0)
        if advanced > remaining:
            raise RuntimeError(f"search action exceeded horizon by {advanced - remaining} ticks: {fitted_action}")
        events = info["events"]
        next_sunflowers = node.sunflower_placements + int(self._is_sunflower_action(node.observation, fitted_action))
        next_same_tick = node.same_tick_actions + 1 if advanced == 0 and fitted_action["type"] in ("plant", "shovel") else 0
        elapsed = node.elapsed_ticks + advanced
        decisions = node.decisions + 1
        step_discount = SEARCH_GAMMA ** (advanced / DISCOUNT_REFERENCE_TICKS)
        reward = self._transition_reward(events, done, next_observation["result"] == 1)
        discount = node.discount * step_discount
        path_return = node.path_return + discount * reward
        child_output = None if done else self._model_output(
            next_observation, node.output["hidden"] if node.output is not None else None,
            fitted_action, advanced, events,
        )
        leaf = 0.0 if done else self._leaf_value(next_observation, child_output)
        score = path_return + discount * leaf
        outcome = int(next_observation["result"] == 1) if done else None
        complete = done or elapsed >= self.horizon_ticks
        exhausted = decisions >= self.max_decisions
        leaf_info = (score, outcome, elapsed, decisions)
        if complete or exhausted:
            return None, leaf_info, 1
        snapshot_id = self._snapshot_fast()
        self._snapshots.add(snapshot_id)
        return _SearchNode(next_observation, snapshot_id, child_output, next_sunflowers,
                           next_same_tick, elapsed, decisions, path_return, discount, score), leaf_info, 1

    def _prune(self, nodes: list[_SearchNode]) -> list[_SearchNode]:
        def rank(node: _SearchNode) -> float:
            progress = min(1.0, node.elapsed_ticks / self.horizon_ticks)
            return node.score + 0.05 * progress
        nodes.sort(key=rank, reverse=True)
        selected = nodes[:self.beam_width]
        selected_ids = {id(node) for node in selected}
        for node in nodes:
            if id(node) not in selected_ids:
                self._release(node.snapshot_id)
        return selected

    def _search_root(self, root_snapshot: int, observation: dict[str, Any],
                     root_output: dict[str, Any] | None, root_action: dict[str, Any]) -> tuple[float, int | None, int, int, int] | None:
        root_node = _SearchNode(observation, root_snapshot, root_output, self.policy.sunflower_placements,
                                0, 0, 0, 0.0, 1.0, self._leaf_value(observation, root_output))
        child, leaf, simulations = self._step_from_snapshot(root_node, root_action)
        if simulations == 0:
            return None
        completed: list[tuple[float, int | None, int, int]] = []
        partial: list[tuple[float, int | None, int, int]] = [leaf]
        if leaf[1] is not None or leaf[2] >= self.horizon_ticks:
            completed.append(leaf)
        beam = [child] if child is not None else []

        while beam:
            expanded: list[_SearchNode] = []
            for node in beam:
                allow_instant = node.same_tick_actions < self.max_same_tick_actions
                remaining = self.horizon_ticks - node.elapsed_ticks
                candidates = self._candidate_actions(
                    node.observation, node.output, node.sunflower_placements, self.candidate_limit,
                    root=False, remaining_ticks=remaining, allow_instant=allow_instant,
                )
                for action in candidates:
                    next_node, next_leaf, count = self._step_from_snapshot(node, action)
                    simulations += count
                    if count == 0:
                        continue
                    partial.append(next_leaf)
                    if next_leaf[1] is not None or next_leaf[2] >= self.horizon_ticks:
                        completed.append(next_leaf)
                    if next_node is not None:
                        expanded.append(next_node)
                self._release(node.snapshot_id)
            if not expanded:
                break
            beam = self._prune(expanded)

        for node in beam:
            self._release(node.snapshot_id)
        pool = completed if completed else partial
        if not pool:
            return None
        if completed:
            best = max(pool, key=lambda item: self._result_key(item[1], item[0]))
        else:
            best = max(pool, key=lambda item: (item[2], item[0]))
            shortfall = max(0.0, 1.0 - best[2] / self.horizon_ticks)
            best = (best[0] - 0.15 * shortfall, best[1], best[2], best[3])
        return best[0], best[1], best[3], simulations, best[2]

    def advice(self, observation: dict[str, Any], hidden: Tensor | None = None,
               previous_action: dict[str, Any] | None = None, delta_ticks: int = 0,
               events: dict[str, Any] | None = None) -> TeacherAdvice:
        with torch.inference_mode():
            with self._speculative_fast() as root_snapshot:
                self._root_snapshot = root_snapshot
                self._snapshots = set()
                try:
                    root_output = self._model_output(observation, hidden, previous_action, delta_ticks, events)
                    root_actions = self._candidate_actions(
                        observation, root_output, self.policy.sunflower_placements,
                        self.root_candidate_limit, root=True, remaining_ticks=self.horizon_ticks,
                    )
                    results: list[tuple[dict[str, Any], float, int | None, int, int]] = []
                    simulation_count = 0
                    for action in root_actions:
                        result = self._search_root(root_snapshot, observation, root_output, action)
                        if result is None:
                            continue
                        value, outcome, decisions, count, elapsed = result
                        results.append((action, value, outcome, decisions, elapsed))
                        simulation_count += count
                    if not results:
                        return self.policy.advice(observation)
                    results.sort(key=lambda item: self._result_key(item[2], item[1]), reverse=True)
                    actions = [item[0] for item in results]
                    scores = [item[1] for item in results]
                    probabilities = self._policy_from_results(results)
                    best = results[0]
                    if len(results) == 1:
                        margin = 2.0
                    elif self._outcome_rank(best[2]) != self._outcome_rank(results[1][2]):
                        margin = float(self._outcome_rank(best[2]) - self._outcome_rank(results[1][2]))
                    else:
                        margin = best[1] - results[1][1]
                    return TeacherAdvice(
                        action=best[0], candidates=list(zip(actions, scores)), search_policy=probabilities,
                        best_second_margin=margin, search_depth=best[3], simulation_count=simulation_count,
                        terminal_outcome=best[2], search_elapsed_ticks=best[4],
                    )
                finally:
                    for snapshot_id in tuple(self._snapshots):
                        self._release(snapshot_id)
