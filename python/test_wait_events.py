import unittest

from pvz_env import PvZEnv, branch_action_token
from pvz_event_env import EventWaitEnv


def observation(tick=0):
    return dict(tick=tick, wave=1, wave_count=3, wave_timer=500, sun=50,
                terminal=False, result=0, plants=[{'type': 1, 'row': 0, 'col': 0}],
                packets=[dict(index=0, type=0, imitater_type=-1, active=False)],
                zombies=[], defenses=[dict(row=0, type=0, state=1, x=0)])








class FakeEventEnv(EventWaitEnv):
    def __init__(self, response):
        super().__init__('/tmp/not-a-resource-run')
        self._reset_done = True
        self._tick = 0
        self._sun_history_start_tick = 0
        self.episode = {'operations': [], 'final_state': {}}
        self.response = response
        self.commands = []
    def _command(self, command):
        self.commands.append(command)
        return self.response


class EventWaitWrapperTests(unittest.TestCase):
    def test_single_macro_income_timestamp_and_actual_duration(self):
        final = observation(30)
        result = dict(version=1, condition='timeout', requested_ticks=60, actual_ticks=30, logic_steps=60,
                      stalled_clock_steps=30, initial_condition_satisfied=False, reason='max_ticks', triggered=['max_ticks'])
        env = FakeEventEnv(dict(ok=True, observation=final, wait_result=result, events={'sun_produced': 25}))
        _, _, _, _, info = env.step({'type': 'wait', 'ticks': 60, 'until': 'timeout'})
        self.assertEqual(env.commands, ['WAIT_EVENT_V1 60 0'])
        self.assertEqual(info['ticks_advanced'], 30)
        self.assertEqual(list(env._sun_production_history), [(30, 25)])
        self.assertEqual(len(env.episode['operations']), 1)
        self.assertEqual(env.episode['operations'][0]['wait_result'], result)

    def test_tampered_metadata_stops_before_history_or_record_mutation(self):
        result = dict(version=1, condition='timeout', requested_ticks=60, actual_ticks=61, logic_steps=60,
                      stalled_clock_steps=0, initial_condition_satisfied=False, reason='max_ticks', triggered=['max_ticks'])
        env = FakeEventEnv(dict(ok=True, observation=observation(30), wait_result=result, events={'sun_produced': 25}))
        with self.assertRaisesRegex(RuntimeError, 'metadata'): env.step({'type': 'wait', 'ticks': 60, 'until': 'timeout'})
        self.assertEqual((list(env._sun_production_history), env.episode['operations']), ([], []))

    def test_fixed_env_cannot_silently_drop_wait_condition(self):
        env = PvZEnv('/tmp/not-a-resource-run')
        env._reset_done = True
        with self.assertRaisesRegex(ValueError, 'EventWaitEnv'):
            env.step({'type': 'wait', 'ticks': 60, 'until': 'wave_changed'})

    def test_branch_batch_encoder_rejects_incomplete_event_representation(self):
        with self.assertRaisesRegex(ValueError, 'cannot drop until'):
            branch_action_token({'type': 'wait', 'ticks': 60, 'until': 'wave_changed'})

    def test_boolean_duration_in_native_metadata_is_not_accepted_as_integer(self):
        result = dict(version=True, condition='timeout', requested_ticks=60, actual_ticks=30, logic_steps=60,
                      stalled_clock_steps=30, initial_condition_satisfied=False, reason='max_ticks', triggered=['max_ticks'])
        env = FakeEventEnv(dict(ok=True, observation=observation(30), wait_result=result, events={}))
        with self.assertRaisesRegex(RuntimeError, 'identity'): env.step({'type': 'wait', 'ticks': 60, 'until': 'timeout'})


if __name__ == '__main__':
    unittest.main()
