"""Joint differentiable goal conditioning, with public-only event scheduling.

The goal is an internal representation, not a stochastic high-level action.
The packed hidden tensor carries fast state, slow state, held goal and public
phase counters. It can cross a replay prefix without changing the old API.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pvz_observation_features import packet_refresh
from pvz_wait_events import public_state

SLOW_FIELDS = {"cell", "layers", "goal_width", "max_ticks", "max_decisions"}
CONTEXT_FIELDS = {"tick", "wave", "sun", "plant_count", "defense_count",
                  "packet_ready_count", "cooldown_min"}


def validate_slow_config(config: dict[str, Any]) -> None:
    slow = config["slow_memory"]
    if (not isinstance(slow, dict) or set(slow) != SLOW_FIELDS
            or slow["cell"] not in ("gru", "lstm")
            or any(type(value) is not int or value < 1
                   for key, value in slow.items() if key != "cell")
            or config["gru_width"] < 16 or slow["goal_width"] > config["gru_width"]):
        raise ValueError("invalid explicit slow_memory configuration")
    if config.get("wait_mask") != "progress_v1" or config.get("input_flags") != 7:
        raise ValueError("slow_memory requires the repaired input7/progress_v1 policy")


def planner_context(observation: dict[str, Any]) -> dict[str, int]:
    state = public_state(observation)
    refresh = [packet_refresh(packet) for packet in observation["packets"]]
    cooldowns = [packet["remaining"] for packet in refresh if packet["scheduled"]]
    return dict(tick=state.tick, wave=state.wave, sun=state.sun,
                plant_count=state.plant_count, defense_count=len(state.ready_defenses),
                packet_ready_count=len(state.ready_packets), cooldown_min=min(cooldowns, default=0))


def _plan(control: list[float], transition: dict, settings: dict) -> tuple[bool, int, list, list]:
    context = transition["planner_context"]
    if (set(context) != CONTEXT_FIELDS
            or any(type(value) is not int for value in context.values())
            or context["wave"] != transition["wave"]):
        raise ValueError("slow_memory requires its exact public planner_context")
    initialized, previous_wave, phase_tick, phase_decision, decision = control[:5]
    if context["tick"] < phase_tick:
        raise ValueError("slow planner public clock moved backwards")
    events = transition.get("events") or {}
    for index, key in enumerate(("zombies_killed", "plants_eaten", "sun_produced",
                                 "sun_spent", "mower_triggered"), 8):
        control[index] += float(events.get(key, 0))
    urgent = set((transition.get("previous_wait_result") or {}).get("triggered", ()))
    update = (not initialized or context["wave"] != previous_wave
              or context["tick"] - phase_tick >= settings["max_ticks"]
              or decision - phase_decision >= settings["max_decisions"]
              or bool(urgent & {"defense_lost", "zombie_entered_left_zone"})
              or events.get("plants_eaten", 0) > 0 or events.get("mower_triggered", 0) > 0)
    features = [(context["tick"] - phase_tick) / settings["max_ticks"],
                (decision - phase_decision) / settings["max_decisions"],
                (context["sun"] - control[5]) / 1000,
                (context["plant_count"] - control[6]) / 54,
                (context["defense_count"] - control[7]) / 6,
                control[8] / 10, control[9] / 6, control[10] / 1000,
                control[11] / 1000, control[12] / 6,
                (context["wave"] - previous_wave) / 5,
                context["packet_ready_count"] / 10, context["cooldown_min"] / 7500,
                context["sun"] / 1000, context["plant_count"] / 54, float(not initialized)]
    if update:
        control[2:4] = [float(context["tick"]), decision]
        control[5:8] = [float(context[k]) for k in ("sun", "plant_count", "defense_count")]
        control[8:13] = [0.] * 5
    origin = int(control[3])
    if "decision_index" in transition and transition["decision_index"] != int(decision):
        raise ValueError("slow planner replay decision index differs from its history")
    control[0], control[1], control[4] = 1., float(context["wave"]), decision + 1
    return bool(update), origin, features, control


class DualMemory(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        validate_slow_config(config)
        self.settings = config["slow_memory"]
        self.width, self.hidden_width = config["width"], config["gru_width"]
        self.fast_layers = config["gru_layers"]
        self.slow_layers = self.settings["layers"]
        self.goal_width = self.settings["goal_width"]
        self.is_lstm = self.settings["cell"] == "lstm"
        self.input_projection = nn.Sequential(nn.Linear(3 * self.width + 16, self.width), nn.SiLU())
        memory = nn.LSTM if self.is_lstm else nn.GRU
        self.memory = memory(self.width, self.hidden_width, self.slow_layers, batch_first=True)
        self.goal = nn.Linear(self.hidden_width, self.goal_width)

    @property
    def packed_layers(self) -> int:
        return self.fast_layers + self.slow_layers * (2 if self.is_lstm else 1) + 2

    def forward_sequences(self, fast_input: Tensor, global_input: Tensor, lane_input: Tensor,
                          packet_input: Tensor, sequences: list[list[dict]],
                          hiddens: list[Tensor | None], fast_memory: nn.GRU,
                          *, verify_records: bool = True) -> tuple[Tensor, Tensor, list[dict]]:
        beliefs, packed_hiddens, records = [], [], []
        offset = 0
        for sequence, hidden in zip(sequences, hiddens, strict=True):
            if hidden is None:
                hidden = fast_input.new_zeros(self.packed_layers, self.hidden_width)
            if hidden.shape != (self.packed_layers, self.hidden_width):
                raise ValueError("dual recurrent hidden shape differs from saved configuration")
            fast_hidden = hidden[:self.fast_layers].unsqueeze(1)
            begin = self.fast_layers
            slow_hidden = hidden[begin:begin + self.slow_layers].unsqueeze(1)
            if self.is_lstm:
                slow_cell = hidden[begin + self.slow_layers:begin + 2 * self.slow_layers].unsqueeze(1)
                slow_hidden = (slow_hidden, slow_cell)
            goal = hidden[-2, :self.goal_width].unsqueeze(0)
            control = hidden[-1].detach().cpu().tolist()
            schedule = []
            for transition in sequence:
                update, origin, features, control = _plan(control, transition, self.settings)
                if verify_records and (type(transition.get("slow_update")) is not bool
                        or transition["slow_update"] != update
                        or type(transition.get("slow_stage_start")) is not int
                        or transition["slow_stage_start"] != origin):
                    raise ValueError("saved slow planner schedule differs from replayed public history")
                schedule.append((update, origin, features))
                records.append(dict(slow_update=update, slow_stage_start=origin))
            position = 0
            while position < len(sequence):
                update, _, features = schedule[position]
                if update:
                    index = offset + position
                    public_features = fast_input.new_tensor(features).unsqueeze(0)
                    slow_input = self.input_projection(torch.cat((global_input[index:index+1],
                        lane_input[index:index+1], packet_input[index:index+1], public_features), dim=-1))
                    slow_output, slow_hidden = self.memory(slow_input.unsqueeze(1), slow_hidden)
                    # This goal remains attached to the producing event. Fast
                    # policy losses can differentiate it across the entire phase.
                    goal = torch.tanh(self.goal(slow_output[:, 0]))
                end = position + 1
                while end < len(sequence) and not schedule[end][0]:
                    end += 1
                phase = torch.cat((fast_input[offset+position:offset+end],
                                   goal.expand(end-position, -1)), dim=-1).unsqueeze(0)
                belief, fast_hidden = fast_memory(phase, fast_hidden)
                beliefs.append(belief[0])
                position = end
            slow_rows = torch.cat(slow_hidden, dim=0) if self.is_lstm else slow_hidden
            goal_row = F.pad(goal, (0, self.hidden_width-self.goal_width)).unsqueeze(0)
            control_row = fast_input.new_tensor(control).view(1, 1, self.hidden_width)
            packed_hiddens.append(torch.cat((fast_hidden, slow_rows, goal_row, control_row), dim=0))
            offset += len(sequence)
        return torch.cat(beliefs), torch.cat(packed_hiddens, dim=1), records


def gradient_start(transitions: list[dict], start: int) -> int:
    """Include the event that produced the held goal, without duplicating loss."""
    origin = transitions[start].get("slow_stage_start")
    if (type(origin) is not int or not 0 <= origin <= start
            or transitions[origin].get("slow_update") is not True):
        raise ValueError("slow gradient window has no recorded producing event")
    return origin
