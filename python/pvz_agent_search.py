"""Search-teacher v2 overrides for the PvZ gameplay policy module."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

import pvz_agent_base as base

TeacherAdvice = base.TeacherAdvice
GameplayModelV1 = base.GameplayModelV1


def behavior_cloning_loss(
    model: GameplayModelV1,
    output: dict[str, Any],
    observation: dict[str, Any],
    action: dict[str, Any],
    plant_weight: float = 1.0,
    candidate_actions: list[dict[str, Any]] | None = None,
    search_policy: list[float] | None = None,
) -> Tensor:
    """Distill search probabilities without class-weighting their relative ratios."""
    if candidate_actions and search_policy and len(candidate_actions) == len(search_policy):
        log_probs = torch.stack([
            base.select_action(model, output, observation, action=candidate)[1]
            for candidate in candidate_actions
        ])
        probabilities = torch.tensor(search_policy, dtype=log_probs.dtype, device=log_probs.device).clamp_min(0.0)
        total = probabilities.sum()
        if float(total.item()) > 0.0:
            return -((probabilities / total) * log_probs).sum()
    return base.behavior_cloning_loss(model, output, observation, action, plant_weight)


@dataclass
class _SearchNode:
    observation: dict[str, Any]
    snapshot_id: int
    output: dict[str, Any] | None
    score: float
    sunflower_placements: int
    same_tick_actions: int


class SearchTeacher(base.SearchTeacher):
    """Rolling-horizon search with a separate beam for every root candidate."""

    def __init__(self, env: Any, model: GameplayModelV1 | None = None, beam_width: int = 4,
                 depth: int = 3, candidate_limit: int = 4, learned_value_weight: float = 0.15,
                 max_same_tick_actions: int = 3) -> None:
        super().__init__(env, model=model, beam_width=beam_width, depth=depth, candidate_limit=candidate_limit)
        if not 0.0 <= learned_value_weight <= 1.0:
            raise ValueError("learned value weight must be from 0 to 1")
        if max_same_tick_actions < 1:
            raise ValueError("max_same_tick_actions must be positive")
        self.root_candidate_limit = max(12, candidate_limit * 3)
        self.learned_value_weight = learned_value_weight
        self.max_same_tick_actions = max_same_tick_actions

    def _value(self, observation: dict[str, Any], output: dict[str, Any] | None,
               events: dict[str, Any] | None) -> float:
        heuristic = self._heuristic_value(observation, events)
        if output is None or self.learned_value_weight <= 0.0:
            return heuristic
        learned = max(0.0, min(1.0, float(output["value"].item()))) * 2.0 - 1.0
        w = self.learned_value_weight
        return (1.0 - w) * heuristic + w * learned

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
            health = sum(max(0.0, zombie.get(key, 0.0)) for key in
                         ("body_health", "helm_health", "shield_health"))
            pressure[zombie["row"]] = pressure.get(zombie["row"], 0.0) + urgency * (1.0 + min(4.0, health / 400.0))
        return pressure

    def _coverage_plant(self, observation: dict[str, Any], placements: list[dict[str, Any]]) -> dict[str, Any]:
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

        return {"type": "plant", **max(placements, key=score)}

    def _model_proposals(self, observation: dict[str, Any], output: dict[str, Any] | None) -> list[dict[str, Any]]:
        if self.model is None or output is None:
            return []
        legal = observation["legal_actions"]
        legal_types = [0] if legal["plants"] else []
        if legal["shovels"]:
            legal_types.append(1)
        if legal.get("wait", True):
            legal_types.extend((2, 3))
        proposals = []
        for type_index in sorted(legal_types, key=lambda i: float(output["type_logits"][i].item()), reverse=True):
            logits = output["type_logits"].clone()
            logits[:] = -1e9
            logits[type_index] = output["type_logits"][type_index]
            try:
                action, _, _ = base.select_action(self.model, dict(output, type_logits=logits), observation,
                                                  deterministic=True)
            except ValueError:
                continue
            proposals.append(action)
        return proposals

    def _candidate_actions_v2(self, observation: dict[str, Any], output: dict[str, Any] | None,
                              sunflower_placements: int, limit: int, root: bool,
                              allow_instant: bool = True) -> list[dict[str, Any]]:
        legal = observation["legal_actions"]
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, Any], ...]] = set()

        def add(action: dict[str, Any]) -> None:
            if not allow_instant and action.get("type") in ("plant", "shovel"):
                return
            key = self._action_key(action)
            if key not in seen:
                seen.add(key)
                candidates.append(action)

        if legal.get("wait", True):
            add({"type": "wait_decision", "max_ticks": base.WAIT_DECISION_TICKS})
            add({"type": "wait", "ticks": 150})

        if allow_instant and root:
            by_packet: dict[int, list[dict[str, Any]]] = {}
            for placement in legal["plants"]:
                by_packet.setdefault(placement["packet"], []).append(placement)
            for packet in sorted(by_packet):
                add(self._coverage_plant(observation, by_packet[packet]))
            if legal["shovels"]:
                health = {(plant["col"], plant["row"]): plant["health"] / max(1, plant["max_health"])
                          for plant in observation["plants"]}
                col, row = min(legal["shovels"], key=lambda cell: health.get(cell, 1.0))
                add({"type": "shovel", "col": col, "row": row})

        for action, _ in base.teacher_advice(observation, sunflower_placements).candidates:
            add(action)
            if not root and len(candidates) >= max(2, limit // 2):
                break
        for action in self._model_proposals(observation, output):
            add(action)

        if allow_instant and not root and len(candidates) < limit:
            by_packet: dict[int, list[dict[str, Any]]] = {}
            for placement in legal["plants"]:
                by_packet.setdefault(placement["packet"], []).append(placement)
            for packet in sorted(by_packet):
                add(self._coverage_plant(observation, by_packet[packet]))
                if len(candidates) >= limit:
                    break
        return candidates[:limit]

    def _step_from_snapshot(self, snapshot_id: int, observation: dict[str, Any],
                            output: dict[str, Any] | None, sunflower_placements: int,
                            same_tick_actions: int, action: dict[str, Any]) -> tuple[_SearchNode | None, float | None, int | None, int]:
        self.env.restore(snapshot_id)
        next_observation, _, done, _, info = self.env.step(action)
        if not info.get("ok"):
            return None, None, None, 0
        advanced = info.get("ticks_advanced", action.get("ticks", 0) if action["type"] == "wait" else 0)
        next_events = info["events"]
        next_sunflowers = sunflower_placements + int(self._is_sunflower_action(observation, action))
        next_same_tick = same_tick_actions + 1 if advanced == 0 and action["type"] in ("plant", "shovel") else 0
        child_output = None if done else self._model_output(
            next_observation, output["hidden"] if output is not None else None, action, advanced, next_events
        )
        score = self._value(next_observation, child_output, next_events)
        if done:
            return None, score, int(next_observation["result"] == 1), 1
        child_snapshot = self.env.snapshot()
        self._snapshots.add(child_snapshot)
        return _SearchNode(next_observation, child_snapshot, child_output, score,
                           next_sunflowers, next_same_tick), score, None, 1

    def _prune_v2(self, nodes: list[_SearchNode]) -> list[_SearchNode]:
        nodes.sort(key=lambda node: node.score, reverse=True)
        selected = nodes[:self.beam_width]
        selected_ids = {id(node) for node in selected}
        for node in nodes:
            if id(node) not in selected_ids:
                self._release(node.snapshot_id)
        return selected

    def _search_root(self, root_snapshot: int, observation: dict[str, Any],
                     root_output: dict[str, Any] | None, root_action: dict[str, Any]) -> tuple[float, int | None, int, int] | None:
        child, score, outcome, simulations = self._step_from_snapshot(
            root_snapshot, observation, root_output, self.policy.sunflower_placements, 0, root_action
        )
        if score is None:
            return None
        value, selected_outcome, reached_depth = score, outcome, 1
        if child is None:
            return value, selected_outcome, reached_depth, simulations
        if self.depth == 1:
            self._release(child.snapshot_id)
            return value, selected_outcome, reached_depth, simulations

        beam = [child]
        for _ in range(1, self.depth):
            expanded: list[_SearchNode] = []
            leaves: list[tuple[float, int | None]] = []
            for node in beam:
                candidates = self._candidate_actions_v2(
                    node.observation, node.output, node.sunflower_placements, self.candidate_limit,
                    root=False, allow_instant=node.same_tick_actions < self.max_same_tick_actions
                )
                for action in candidates:
                    next_node, next_score, next_outcome, count = self._step_from_snapshot(
                        node.snapshot_id, node.observation, node.output, node.sunflower_placements,
                        node.same_tick_actions, action
                    )
                    simulations += count
                    if next_score is None:
                        continue
                    leaves.append((next_score, next_outcome))
                    if next_node is not None:
                        expanded.append(next_node)
                self._release(node.snapshot_id)
            if not leaves:
                break
            reached_depth += 1
            if any(leaf_outcome == 1 for _, leaf_outcome in leaves):
                for next_node in expanded:
                    self._release(next_node.snapshot_id)
                return 1.0, 1, reached_depth, simulations
            value, selected_outcome = max(leaves, key=lambda item: item[0])
            if not expanded:
                beam = []
                break
            beam = self._prune_v2(expanded)
        for node in beam:
            self._release(node.snapshot_id)
        return value, selected_outcome, reached_depth, simulations

    def advice(self, observation: dict[str, Any], hidden: Tensor | None = None,
               previous_action: dict[str, Any] | None = None, delta_ticks: int = 0,
               events: dict[str, Any] | None = None) -> TeacherAdvice:
        with torch.inference_mode():
            with self.env.speculative() as root_snapshot:
                self._root_snapshot = root_snapshot
                self._snapshots = set()
                try:
                    root_output = self._model_output(observation, hidden, previous_action, delta_ticks, events)
                    root_actions = self._candidate_actions_v2(
                        observation, root_output, self.policy.sunflower_placements,
                        self.root_candidate_limit, root=True
                    )
                    results: list[tuple[dict[str, Any], float, int | None]] = []
                    simulation_count = 0
                    reached_depth = 0
                    for action in root_actions:
                        result = self._search_root(root_snapshot, observation, root_output, action)
                        if result is None:
                            continue
                        value, outcome, action_depth, count = result
                        results.append((action, value, outcome))
                        simulation_count += count
                        reached_depth = max(reached_depth, action_depth)
                    if not results:
                        return self.policy.advice(observation)
                    results.sort(key=lambda item: item[1], reverse=True)
                    actions = [item[0] for item in results]
                    scores = [item[1] for item in results]
                    top = scores[0]
                    exponents = [math.exp((value - top) / 0.25) for value in scores]
                    total = sum(exponents)
                    probabilities = [value / total for value in exponents]
                    margin = top - scores[1] if len(scores) > 1 else top - (-1.0)
                    return TeacherAdvice(
                        action=actions[0], candidates=list(zip(actions, scores)), search_policy=probabilities,
                        best_second_margin=margin, search_depth=reached_depth,
                        simulation_count=simulation_count, terminal_outcome=results[0][2],
                        # Search scores remain in search_values; the main value head now learns the
                        # actual episode result through the existing training fallback.
                        search_value=None,
                    )
                finally:
                    for snapshot_id in tuple(self._snapshots):
                        self._release(snapshot_id)
