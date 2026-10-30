"""Native interruptible-wait metadata, version 1.

The engine performs the wait as one macro action; this module only validates the
duration/reason record it reports before that record enters the episode.
"""
from __future__ import annotations

VERSION = 1
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
