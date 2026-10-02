from copy import deepcopy
import unittest

from pvz_env import PvZEnv, REPLAY_FORMAT_VERSION, branch_action_token
from pvz_event_env import EventWaitEnv
from pvz_wait_events import public_state, reference_wait


def observation(tick=0):
    return dict(tick=tick, wave=1, wave_count=3, wave_timer=500, sun=50,
                terminal=False, result=0, plants=[{'type': 1, 'row': 0, 'col': 0}],
                packets=[dict(index=0, type=0, imitater_type=-1, active=False)],
                zombies=[], defenses=[dict(row=0, type=0, state=1, x=0)])


def zombie(identity=65536, x=200., on_board=True):
    return dict(id=identity, x=x, on_board=on_board)


def sequence(rows):
    calls = []
    iterator = iter(rows)
    def advance():
        value = next(iterator)
        calls.append(value['tick'])
        return value
    return advance, calls


class WaitEventSemanticsTests(unittest.TestCase):
    def test_transient_event_returns_at_first_tick(self):
        first, second = observation(1), observation(2)
        first['wave'] = 2
        advance, calls = sequence([first, second])
        result = reference_wait(observation(), advance, 2, 'wave_changed')
        self.assertEqual(calls, [1])
        self.assertEqual((result['actual_ticks'], result['reason']), (1, 'condition'))

    def test_all_simultaneous_reasons_are_retained_in_frozen_precedence(self):
        before, after = observation(), observation(1)
        before['zombies'] = [zombie()]
        after.update(terminal=True, wave=2, sun=75, plants=[], zombies=[zombie(x=160.)], defenses=[])
        after['packets'][0]['active'] = True
        advance, calls = sequence([after])
        result = reference_wait(before, advance, 1, 'wave_changed')
        self.assertEqual(result['triggered'], ['terminal', 'defense_lost', 'zombie_entered_left_zone', 'condition', 'max_ticks'])
        self.assertEqual(result['reason'], 'terminal')
        self.assertEqual(calls, [1])

    def test_emergency_interrupts_an_unmet_selected_condition(self):
        before, after = observation(), observation(1)
        before['zombies'] = [zombie()]
        after.update(zombies=[zombie(x=159.)], defenses=[])
        advance, _ = sequence([after])
        result = reference_wait(before, advance, 300, 'sun_increased')
        self.assertEqual(result['triggered'], ['defense_lost', 'zombie_entered_left_zone'])

    def test_already_ready_card_is_not_a_new_readiness_edge(self):
        before = observation()
        before['packets'][0]['active'] = True
        rows = [deepcopy(before) for _ in range(3)]
        for i, row in enumerate(rows, 1): row['tick'] = i
        advance, calls = sequence(rows)
        result = reference_wait(before, advance, 3, 'packet_became_ready')
        self.assertEqual((result['reason'], result['actual_ticks'], len(calls)), ('max_ticks', 3, 3))

    def test_initial_level_condition_returns_zero_without_forced_tick(self):
        before = observation()
        before['zombies'] = [zombie(identity=-2147483648, x=159.)]
        advance, calls = sequence([])
        result = reference_wait(before, advance, 300, 'left_zone_occupied')
        self.assertEqual((result['reason'], result['actual_ticks'], result['logic_steps']), ('condition', 0, 0))
        self.assertTrue(result['initial_condition_satisfied'])
        self.assertEqual(calls, [])

    def test_selected_condition_precedes_simultaneous_deadline(self):
        rows = [observation(1), observation(2)]
        rows[1]['wave'] = 2
        advance, _ = sequence(rows)
        result = reference_wait(observation(), advance, 2, 'wave_changed')
        self.assertEqual(result['triggered'], ['condition', 'max_ticks'])

    def test_conveyor_arrival_is_new_ready_card_without_reading_cost(self):
        before, after = observation(), observation(1)
        before['packets'][0].update(type=-1, active=True, cost=0)
        after['packets'][0].update(type=19, active=True, cost=999999)
        advance, _ = sequence([after])
        result = reference_wait(before, advance, 10, 'packet_became_ready')
        self.assertEqual(result['reason'], 'condition')

    def test_plant_removal_uses_public_count_not_private_plant_identity(self):
        before, after = observation(), observation(1)
        before['plants'][0]['hidden_id'] = 100
        after['plants'][0]['hidden_id'] = 200
        advance, _ = sequence([after])
        self.assertEqual(reference_wait(before, advance, 1, 'plant_count_decreased')['reason'], 'max_ticks')
        after['plants'] = []
        advance, _ = sequence([after])
        self.assertEqual(reference_wait(before, advance, 1, 'plant_count_decreased')['reason'], 'condition')

    def test_initial_terminal_and_zero_deadline_need_no_advance(self):
        before = observation()
        before['terminal'] = True
        advance, calls = sequence([])
        result = reference_wait(before, advance, 0, 'timeout')
        self.assertEqual(result['triggered'], ['terminal', 'max_ticks'])
        self.assertEqual((result['actual_ticks'], calls), (0, []))

    def test_stalled_board_clock_is_bounded_and_recorded_separately(self):
        advance, calls = sequence([observation()] * 3)
        result = reference_wait(observation(), advance, 3, 'timeout')
        self.assertEqual((result['actual_ticks'], result['logic_steps'], result['stalled_clock_steps']), (0, 3, 3))
        self.assertEqual(len(calls), 3)

    def test_backwards_clock_is_a_failure(self):
        advance, _ = sequence([observation(1)])
        with self.assertRaisesRegex(RuntimeError, 'backwards'):
            reference_wait(observation(2), advance, 3, 'timeout')

    def test_hidden_wave_plan_cannot_change_predicates(self):
        before = observation()
        after = deepcopy(before)
        after.update(wave_zombies=[999], hidden_wave_timer=-9999, spawning_rng='different')
        self.assertEqual(public_state(before), public_state(after))

    def test_off_board_zombies_and_negative_public_ids(self):
        value = observation()
        value['zombies'] = [zombie(-2147483648, 150., False), zombie(65536, 160., True)]
        self.assertEqual(public_state(value).left_zone_zombies, frozenset({65536}))
        value['zombies'][1]['id'] = 0
        with self.assertRaisesRegex(ValueError, 'non-null'): public_state(value)

    def test_invalid_requests_reject_before_any_advance(self):
        for duration, condition in ((True, 'timeout'), (-1, 'timeout'), (1000001, 'timeout'), (1, 'plant_peashooter')):
            advance, calls = sequence([])
            with self.assertRaises(ValueError): reference_wait(observation(), advance, duration, condition)
            self.assertEqual(calls, [])


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

    def test_replay_checks_interruption_when_ticks_and_events_still_match(self):
        class ReplayFixture(FakeEventEnv):
            def _check_manifest(self, *args):
                pass  # This test isolates interruption comparison, not manifest integrity.
            def reset(self, **kwargs):
                self._tick = 0
                self._sun_production_history.clear()
                self._sun_history_start_tick = 0
                self.episode = {'operations': [], 'final_state': {}}
                return observation(), {}

        metadata = dict(version=1, condition='timeout', requested_ticks=60, actual_ticks=30, logic_steps=60,
                        stalled_clock_steps=30, initial_condition_satisfied=False, reason='max_ticks', triggered=['max_ticks'])
        env = ReplayFixture(dict(ok=True, observation=observation(30), wait_result=metadata, events={'sun_produced': 25}))
        env.step({'type': 'wait', 'ticks': 60, 'until': 'timeout'})
        record = dict(format_version=REPLAY_FORMAT_VERSION, manifest={}, task={'profile': {}, 'forced_seeds': []}, deck=[{'seed_type': 0}],
                      initial_state=env._state_record(observation()), operations=deepcopy(env.episode['operations']),
                      final_state=deepcopy(env.episode['final_state']))
        self.assertEqual(env.replay_record(record)['tick'], 30)
        record['operations'][0]['wait_result']['reason'] = 'condition'
        with self.assertRaisesRegex(RuntimeError, 'interruption diverged'): env.replay_record(record)


if __name__ == '__main__':
    unittest.main()
