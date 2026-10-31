"""One native macro action per event wait; fixed actions use the existing path."""
from __future__ import annotations

from pvz_env import PvZEnv
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




class EventWaitEnv(PvZEnv):
    def _read_message(self):
        response = super()._read_message()
        if response.get('ready') and response.get('wait_events_version') != VERSION:
            self._abort_process()
            raise RuntimeError('native simulator lacks wait_events_version=1')
        return response

    def step(self, action):
        if not isinstance(action, dict):
            raise TypeError('action must be a dictionary')
        if 'until' not in action:
            return super().step(action)
        if not self._reset_done:
            raise RuntimeError('call reset() before step()')
        if set(action) != {'type', 'ticks', 'until'} or action['type'] != 'wait':
            raise ValueError('event wait action requires exactly type=wait, ticks and until')
        ticks, condition = action['ticks'], action['until']
        if type(ticks) is not int or not 0 <= ticks <= 1_000_000 or type(condition) is not str or condition not in CONDITIONS:
            raise ValueError('invalid event wait duration or condition')
        response = self._command(f'WAIT_EVENT_V1 {ticks} {CONDITIONS.index(condition)}')
        observation, result = response.get('observation'), response.get('wait_result')
        if not response.get('ok') or not isinstance(observation, dict) or not isinstance(result, dict):
            raise RuntimeError('native event wait rejected or omitted its observation/result')
        if type(observation.get('tick')) is not int or type(self._tick) is not int:
            raise RuntimeError('native event wait metadata identity differs')
        actual = int(observation['tick']) - self._tick
        try:
            validate_wait_result(action, result, actual)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        events = response.get('events', {})
        self._annotate_observation(observation, events)
        self._adopt_tick(observation)
        info = {'ok': True, 'events': events, 'ticks_advanced': actual, 'wait_result': result}
        self._record_operation({'kind': 'action', 'request': dict(action), 'action': dict(action),
                                'ticks_advanced': actual, 'wait_result': dict(result)}, observation, events)
        return observation, 0.0, bool(observation.get('terminal')), False, info


    def wait_until_legal(self, until_tick: int):
        """Repeat ``wait 60 / timeout`` in the simulator until a plant is legal, the level ends, or
        the clock reaches ``until_tick`` (``WAIT_UNTIL_LEGAL``): one round trip instead of one per
        wait. For search rollouts, where a policy without a legal plant can only wait.

        The sun-income history gets every sub-step's (tick, sun produced), exactly as if each wait
        had been its own ``step``, so ``sun_income_rate`` is identical. ``info["previous_tick"]``
        is the tick of the observation the last sub-step started from. Like ``branch_snapshot`` this
        bypasses the episode record (search code restores its root afterwards anyway).
        """
        if not self._reset_done:
            raise RuntimeError('call reset() before wait_until_legal()')
        if type(until_tick) is not int:
            raise ValueError('until_tick must be an integer')
        response = self._command(f'WAIT_UNTIL_LEGAL {until_tick}')
        observation, result = response.get('observation'), response.get('wait_result')
        substeps = response.get('substeps')
        if (not response.get('ok') or not isinstance(observation, dict) or not isinstance(result, dict)
                or not substeps or type(response.get('previous_tick')) is not int):
            raise RuntimeError('native WAIT_UNTIL_LEGAL rejected or omitted its observation/result')
        previous_tick = response['previous_tick']
        try:
            validate_wait_result({'ticks': 60, 'until': 'timeout'}, result, int(observation['tick']) - previous_tick)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        for tick, produced in substeps:
            if produced > 0:
                self._sun_production_history.append((tick, produced))
        events = response.get('events', {})
        self._annotate_observation(observation)       # history already has every sub-step
        actual = int(observation['tick']) - self._tick
        self._adopt_tick(observation)
        info = {'ok': True, 'events': events, 'ticks_advanced': actual, 'wait_result': result,
                'previous_tick': previous_tick, 'substeps': len(substeps)}
        return observation, 0.0, bool(observation.get('terminal')), False, info
