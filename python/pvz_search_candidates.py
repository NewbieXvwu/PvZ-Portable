"""Generic legal-action coverage and optional model proposals for simulator search."""

from __future__ import annotations

from typing import Any

import torch

from pvz_agent_model import GameplayModelV1, WAIT_DECISION_TICKS, WAIT_TICKS


def action_key(action: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(action.items()))


def lane_pressure(observation: dict[str, Any]) -> dict[int, float]:
    pressure = {cell["row"]: 0.0 for cell in observation["cells"] if cell["row_type"] > 0}
    for zombie in observation["zombies"]:
        urgency = max(0.0, min(1.5, (760.0 - zombie["x"]) / 520.0))
        health = sum(max(0.0, zombie.get(key, 0.0)) for key in ("body_health", "helm_health", "shield_health"))
        pressure[zombie["row"]] = pressure.get(zombie["row"], 0.0) + urgency * (1.0 + min(4.0, health / 400.0))
    return pressure


def round_robin(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    depth = 0
    while True:
        added = False
        for group in groups:
            if depth < len(group):
                result.append(group[depth])
                added = True
        if not added:
            return result
        depth += 1


def fit_action_to_remaining(action: dict[str, Any], remaining_ticks: int) -> dict[str, Any] | None:
    if remaining_ticks <= 0:
        return None
    fitted = dict(action)
    if fitted.get("type") == "wait":
        fitted["ticks"] = min(int(fitted.get("ticks", remaining_ticks)), remaining_ticks)
        return fitted if fitted["ticks"] >= 1 else None
    if fitted.get("type") == "wait_decision":
        fitted["max_ticks"] = min(int(fitted.get("max_ticks", WAIT_DECISION_TICKS)), remaining_ticks)
        return fitted if fitted["max_ticks"] >= 1 else None
    return fitted


def diverse_plant_groups(observation: dict[str, Any], per_packet: int) -> list[list[dict[str, Any]]]:
    by_packet: dict[int, list[dict[str, Any]]] = {}
    for placement in observation["legal_actions"]["plants"]:
        by_packet.setdefault(placement["packet"], []).append(placement)
    pressure = lane_pressure(observation)
    groups: list[list[dict[str, Any]]] = []
    for packet in sorted(by_packet):
        placements = by_packet[packet]
        chosen: list[dict[str, Any]] = []
        cells: set[tuple[int, int]] = set()

        def choose(key: Any) -> None:
            if len(chosen) >= per_packet:
                return
            item = max(placements, key=key)
            cell = (item["row"], item["col"])
            if cell not in cells:
                cells.add(cell)
                chosen.append({"type": "plant", **item})

        choose(lambda item: (pressure.get(item["row"], 0.0), -abs(item["col"] - 3)))
        choose(lambda item: (pressure.get(item["row"], 0.0), item["col"]))
        choose(lambda item: (-pressure.get(item["row"], 0.0), -item["col"]))
        while len(chosen) < min(per_packet, len(placements)):
            remaining = [item for item in placements if (item["row"], item["col"]) not in cells]
            if not remaining:
                break
            if not chosen:
                item = remaining[0]
            else:
                item = max(remaining, key=lambda candidate: min(
                    abs(candidate["row"] - selected["row"]) * 2 + abs(candidate["col"] - selected["col"])
                    for selected in chosen
                ))
            cells.add((item["row"], item["col"]))
            chosen.append({"type": "plant", **item})
        if chosen:
            groups.append(chosen)
    return groups


def shovel_proposals(observation: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    health = {(plant["col"], plant["row"]): plant.get("health", 0) / max(1, plant.get("max_health", 1))
              for plant in observation["plants"]}
    pressure = lane_pressure(observation)
    cells = sorted(observation["legal_actions"]["shovels"],
                   key=lambda cell: (health.get(cell, 1.0), pressure.get(cell[1], 0.0), -cell[0]))
    return [{"type": "shovel", "col": col, "row": row} for col, row in cells[:limit]]


def adaptive_wait(observation: dict[str, Any]) -> dict[str, Any]:
    nearest = min((zombie["x"] for zombie in observation["zombies"]), default=9999.0)
    return {"type": "wait", "ticks": 60 if nearest < 360 else 150 if nearest < 560 else 300}


class CandidateGenerator:
    def __init__(self, model: GameplayModelV1 | None = None) -> None:
        self.model = model

    def model_proposals(self, observation: dict[str, Any], output: dict[str, Any] | None,
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
                valid_cells = {item["row"] * 9 + item["col"] for item in legal["plants"] if item["packet"] == packet}
                for cell in range(54):
                    if cell not in valid_cells:
                        cell_logits[cell] = -1e9
                cell_logp = torch.log_softmax(cell_logits, dim=0)
                for cell in sorted(valid_cells, key=lambda value: float(cell_logp[value].item()), reverse=True)[:3]:
                    joint = float((type_logp[0] + packet_logp[packet_index] + cell_logp[cell]).item())
                    scored.append((joint, {"type": "plant", "packet": packet, "col": cell % 9, "row": cell // 9}))

        if legal["shovels"]:
            cell_logits = self.model.shovel_cell_scores(output).clone()
            valid_cells = {row * 9 + col for col, row in legal["shovels"]}
            for cell in range(54):
                if cell not in valid_cells:
                    cell_logits[cell] = -1e9
            cell_logp = torch.log_softmax(cell_logits, dim=0)
            for cell in sorted(valid_cells, key=lambda value: float(cell_logp[value].item()), reverse=True)[:2]:
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
            key = action_key(action)
            if key not in seen:
                seen.add(key)
                proposals.append(action)
            if len(proposals) >= limit:
                break
        return proposals

    def actions(self, observation: dict[str, Any], output: dict[str, Any] | None, limit: int,
                root: bool, remaining_ticks: int, allow_instant: bool = True) -> list[dict[str, Any]]:
        legal = observation["legal_actions"]
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, Any], ...]] = set()

        def add(action: dict[str, Any]) -> None:
            if len(candidates) >= limit or (not allow_instant and action.get("type") in ("plant", "shovel")):
                return
            fitted = fit_action_to_remaining(action, remaining_ticks)
            if fitted is None:
                return
            key = action_key(fitted)
            if key not in seen:
                seen.add(key)
                candidates.append(fitted)

        temporal: list[dict[str, Any]] = []
        if legal.get("wait", True):
            temporal.append({"type": "wait_decision", "max_ticks": min(WAIT_DECISION_TICKS, remaining_ticks)})
            if root:
                temporal.extend({"type": "wait", "ticks": ticks} for ticks in WAIT_TICKS)
            else:
                temporal.append(adaptive_wait(observation))
                if not allow_instant:
                    temporal.extend({"type": "wait", "ticks": ticks} for ticks in WAIT_TICKS)

        structural = round_robin(diverse_plant_groups(observation, 3 if root else 2)) if allow_instant else []
        shovels = shovel_proposals(observation, 2 if root else 1) if allow_instant else []
        model_actions = self.model_proposals(observation, output, 8 if root else 2)
        first: list[dict[str, Any]] = []
        extras: list[dict[str, Any]] = []
        packets: set[int] = set()
        for action in structural:
            packet = action["packet"]
            (first if packet not in packets else extras).append(action)
            packets.add(packet)
        pools = (first, temporal, model_actions, shovels, extras) if root else [temporal, model_actions, first, shovels, extras]
        for action in ([item for pool in pools for item in pool] if root else round_robin(list(pools))):
            add(action)
            if len(candidates) >= limit:
                break
        return candidates
