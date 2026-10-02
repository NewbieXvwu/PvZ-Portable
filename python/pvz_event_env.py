"""One native macro action per event wait; fixed actions use the existing path."""
from __future__ import annotations

from pvz_env import PvZEnv
from pvz_wait_events import CONDITIONS, VERSION, validate_wait_result


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


def policy_env(model_config: dict, *args, **kwargs) -> PvZEnv:
    """Training/evaluation select the native interface from saved model config."""
    mode = model_config.get('wait_mode', 'fixed')
    if mode not in ('fixed', 'events'):
        raise ValueError('wait_mode must be fixed or events')
    return (EventWaitEnv if mode == 'events' else PvZEnv)(*args, **kwargs)


def require_policy_env(model_config: dict, env: PvZEnv) -> None:
    if model_config.get('wait_mode') == 'events' and not isinstance(env, EventWaitEnv):
        raise ValueError('event policy requires EventWaitEnv')
