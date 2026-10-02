"""Public-observation reference semantics for native interruptible waits, version 1.

The native command performs one macro action. This reference is for engineering
checks, not a Python micro-step adapter: annotating intermediate steps changes
the existing environment's economic history.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
from typing import Callable

VERSION = 1
LEFT_ZONE_MAX_X = 160.0
CONDITIONS = ('timeout', 'wave_changed', 'packet_became_ready', 'sun_increased',
              'left_zone_occupied', 'plant_count_decreased')
REASON_PRECEDENCE = ('terminal', 'defense_lost', 'zombie_entered_left_zone', 'condition', 'max_ticks')


def validate_wait_result(action: dict, result: dict | None, actual_ticks: int) -> None:
    """Shared native/trajectory validation; time is the observed clock delta."""
    expected = {'version', 'condition', 'requested_ticks', 'actual_ticks', 'logic_steps',
                'stalled_clock_steps', 'initial_condition_satisfied', 'reason', 'triggered'}
    integers = ('version', 'requested_ticks', 'actual_ticks', 'logic_steps', 'stalled_clock_steps')
    if (not isinstance(result, dict) or set(result) != expected
            or any(type(result[key]) is not int for key in integers)
            or result['version'] != VERSION or result['condition'] != action.get('until')
            or result['condition'] not in CONDITIONS or result['requested_ticks'] != action.get('ticks')
            or type(actual_ticks) is not int):
        raise ValueError('native event wait metadata identity differs')
    reasons = result['triggered']
    if (actual_ticks < 0 or result['actual_ticks'] != actual_ticks
            or not 0 <= result['logic_steps'] <= result['requested_ticks']
            or not 0 <= result['stalled_clock_steps'] <= result['logic_steps']
            or type(result['initial_condition_satisfied']) is not bool
            or not isinstance(reasons, list) or not reasons
            or any(reason not in REASON_PRECEDENCE for reason in reasons)
            or reasons != sorted(set(reasons), key=REASON_PRECEDENCE.index)
            or result['reason'] != reasons[0]):
        raise ValueError('native event wait duration/reason metadata differs')


def summarize_wait_records(records: list[dict]) -> dict:
    """Keep actual/requested time and all simultaneous reasons separate."""
    conditions, reasons, triggered = Counter(), Counter(), Counter()
    for record in records:
        action, result = record['action'], record['wait_result']
        if action.get('until') is not None:
            validate_wait_result(action, result, record['actual_ticks'])
            conditions[action['until']] += 1
            reasons[result['reason']] += 1
            triggered.update(result['triggered'])
        elif result is not None:
            raise ValueError('fixed actions cannot carry event wait metadata')
    return {'wait_actions': len(records), 'condition_counts': dict(conditions),
            'primary_reason_counts': dict(reasons), 'all_trigger_counts': dict(triggered),
            'requested_logic_ticks': sum(r['action']['ticks'] for r in records),
            'actual_ticks': sum(r['actual_ticks'] for r in records),
            'zero_actual_tick_waits': sum(r['actual_ticks'] == 0 for r in records),
            'logic_steps': sum(r['wait_result']['logic_steps'] for r in records if r['wait_result'] is not None),
            'stalled_clock_steps': sum(r['wait_result']['stalled_clock_steps'] for r in records
                                       if r['wait_result'] is not None),
            'initial_condition_satisfied': sum(r['wait_result']['initial_condition_satisfied']
                                               for r in records if r['wait_result'] is not None)}


@dataclass(frozen=True)
class PublicState:
    tick: int
    wave: int
    sun: int
    plant_count: int
    terminal: bool
    ready_packets: frozenset
    ready_defenses: frozenset
    left_zone_zombies: frozenset


def _integer(value):
    if type(value) is not int:
        raise ValueError('event predicates require explicit public integer fields')
    return value


def _boolean(value):
    if type(value) is not bool:
        raise ValueError('event predicates require explicit public boolean fields')
    return value


def public_state(observation: dict) -> PublicState:
    """Read only the specified public fields, with no hidden wave-table access."""
    ready = frozenset((_integer(p['index']), _integer(p['type']), _integer(p['imitater_type']))
                      for p in observation['packets'] if _boolean(p['active']) and _integer(p['type']) != -1)
    defenses = frozenset((_integer(d['row']), _integer(d['type']))
                         for d in observation['defenses'] if _integer(d['state']) == 1)
    zombie_ids, left = set(), set()
    for zombie in observation['zombies']:
        identity = _integer(zombie['id'])
        if identity in (0, -1) or identity in zombie_ids:
            raise ValueError('public zombie IDs must be unique and non-null')
        zombie_ids.add(identity)
        on_board = _boolean(zombie['on_board'])
        x = zombie['x']
        if type(x) not in (float, int) or not math.isfinite(x):
            raise ValueError('public zombie x must be finite')
        if on_board and x <= LEFT_ZONE_MAX_X:
            left.add(identity)
    return PublicState(_integer(observation['tick']), _integer(observation['wave']),
                       _integer(observation['sun']), len(observation['plants']),
                       _boolean(observation['terminal']), ready, defenses, frozenset(left))


def triggers(before: PublicState, after: PublicState, condition: str, deadline=False) -> tuple[str, ...]:
    if condition not in CONDITIONS:
        raise ValueError('unknown public wait condition')
    selected = ((condition == 'wave_changed' and after.wave != before.wave)
                or (condition == 'packet_became_ready' and bool(after.ready_packets-before.ready_packets))
                or (condition == 'sun_increased' and after.sun > before.sun)
                or (condition == 'left_zone_occupied' and bool(after.left_zone_zombies))
                or (condition == 'plant_count_decreased' and after.plant_count < before.plant_count))
    active = (after.terminal, bool(before.ready_defenses-after.ready_defenses),
              bool(after.left_zone_zombies-before.left_zone_zombies), selected, deadline)
    return tuple(reason for reason, enabled in zip(REASON_PRECEDENCE, active, strict=True) if enabled)


def reference_wait(initial: dict, advance: Callable[[], dict], requested_ticks: int, condition: str) -> dict:
    """Bound logic calls like legacy WAIT; separately record actual board ticks."""
    if type(requested_ticks) is not int or not 0 <= requested_ticks <= 1_000_000:
        raise ValueError('requested ticks must be an integer from 0 to 1000000')
    if condition not in CONDITIONS:
        raise ValueError('unknown public wait condition')
    start = previous = public_state(initial)
    current_observation = initial
    reasons = triggers(start, start, condition, requested_ticks == 0)
    calls = stalled = 0
    while not reasons and calls < requested_ticks:
        current_observation = advance()
        current = public_state(current_observation)
        if current.tick < previous.tick:
            raise RuntimeError('event wait public clock moved backwards')
        calls += 1
        stalled += current.tick == previous.tick
        reasons = triggers(previous, current, condition, calls == requested_ticks)
        previous = current
    return {'version': VERSION, 'condition': condition, 'requested_ticks': requested_ticks,
            'actual_ticks': previous.tick-start.tick, 'logic_steps': calls, 'stalled_clock_steps': stalled,
            'initial_condition_satisfied': condition == 'left_zone_occupied' and bool(start.left_zone_zombies),
            'reason': reasons[0], 'triggered': list(reasons), 'observation': current_observation}
