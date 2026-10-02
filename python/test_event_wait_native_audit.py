"""Independent oracle and released-only launch contracts, with no native process."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_event_wait_native_audit import check_protocol, oracle_macro_wait, saved_replay_record
from pvz_common import canonical_digest, sha256_bytes
from pvz_env import PvZEnv, REPLAY_FORMAT_VERSION
from test_wait_events import observation


def counters(**changes):
    return dict(zombies_killed=0,plants_eaten=0,sun_produced=0,sun_spent=0,
                mower_triggered=0,waves_started=0,level_won=False,level_lost=False,**changes)


class RawOracle:
    def __init__(self, states, events):
        self.states, self.events = states,events
        self.index, self.commands, self.annotations, self.operations = 0,[],[],[]

    def _command(self, command):
        self.commands.append(command)
        if command == 'WAIT 1':
            self.index += 1
        return dict(ok=True,observation=copy.deepcopy(self.states[self.index]),
                    events=copy.deepcopy(self.events[self.index]))

    def _annotate_observation(self, observation, events):
        self.annotations.append((observation['tick'],copy.deepcopy(events)))
        observation['sun_income_rate']=float(events['sun_produced'])

    def _adopt_tick(self, observation):
        self.tick=observation['tick']

    def _record_operation(self,operation,observation,events):
        self.operations.append(operation)


class NativeAuditContracts(unittest.TestCase):
    def test_saved_replay_uses_real_manifest_and_retains_operations_and_debug_state(self):
        env = PvZEnv.__new__(PvZEnv)
        patch = b'fixture source patch\n'
        manifest = dict(source_revision='fixture', source_dirty=True,
                        working_tree_patch_sha256=sha256_bytes(patch), build_config={},
                        resource_sha256='fixture-resource', properties_partner_sha256='fixture-properties',
                        executable_sha256='fixture-executable')
        env._manifest_data = lambda: (manifest, patch)
        env.episode = dict(format_version=REPLAY_FORMAT_VERSION, task={}, deck=[], initial_state={'tick':0},
            operations=[dict(record_type='operation', kind='action', action=dict(type='wait', ticks=60, until='timeout'),
                wait_result={'actual_ticks':60}, state={'tick':60}, debug_state_sha256='retained-debug-state')],
            final_state={'tick':60})
        before = copy.deepcopy(env.episode)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            record = saved_replay_record(env, directory)
            self.assertEqual(env.episode, before)
            self.assertEqual({key:value for key,value in record.items() if key != 'manifest'}, before)
            written = json.loads((directory/'experiment_manifest.json').read_text())
            self.assertEqual(record['manifest'], dict(path='experiment_manifest.json', sha256=canonical_digest(written)))
            self.assertEqual((directory/'working_tree.patch').read_bytes(), patch)
            self.assertTrue((directory/'episode.jsonl.gz').is_file())
            self.assertTrue((directory/'record.json.gz').is_file())
            # These are the real integrity checks used by replay_record, not a mock.
            (directory/'working_tree.patch').write_bytes(b'corrupted patch')
            with self.assertRaisesRegex(ValueError, 'patch is missing or corrupted'):
                env._check_manifest(record['manifest'], directory)
            (directory/'working_tree.patch').write_bytes(patch)
            written['executable_sha256'] = 'changed-executable'
            (directory/'experiment_manifest.json').write_text(json.dumps(written))
            with self.assertRaisesRegex(ValueError, 'manifest digest mismatch'):
                env._check_manifest(record['manifest'], directory)

    def test_draft_gate_precedes_source_binary_resources_and_process_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'draft.json'
            path.write_text(json.dumps(dict(release_status='draft')))
            with self.assertRaisesRegex(RuntimeError,'publish the released protocol'):
                check_protocol(path)

    def test_micro_reference_annotates_once_and_uses_macro_sun_spent(self):
        states=[observation(0),observation(1),observation(2)]
        states[0]['sun']=100; states[1]['sun']=150; states[2]['sun']=120
        events=[counters(),counters(),counters()]
        events[1]['sun_produced']=50; events[2]['sun_spent']=30
        env=RawOracle(states,events)
        trace=[]
        result=oracle_macro_wait(env,states[0],dict(type='wait',ticks=2,until='timeout'),trace.append)
        self.assertEqual(env.commands,['WAIT 0','WAIT 1','WAIT 1'])
        self.assertEqual(len(env.annotations),1)
        self.assertEqual(result[-1]['events']['sun_produced'],50)
        self.assertEqual(result[-1]['events']['sun_spent'],0)
        self.assertEqual(result[0]['sun_income_rate'],50.)
        self.assertEqual(result[-1]['wait_result']['reason'],'max_ticks')
        self.assertEqual(len(trace),3)
        self.assertEqual(env.operations[0]['wait_result'],result[-1]['wait_result'])

    def test_wave_edge_interrupts_before_maximum_and_retains_terminal_priority(self):
        states=[observation(0),observation(1),observation(2)]
        states[2].update(wave=states[0]['wave']+1,terminal=True,result=1)
        events=[counters(),counters(),counters()]
        events[2]['waves_started']=1; events[2]['level_won']=True
        env=RawOracle(states,events)
        result=oracle_macro_wait(env,states[0],dict(type='wait',ticks=60,until='wave_changed'),lambda _:None)
        self.assertEqual(result[-1]['wait_result']['triggered'],['terminal','condition'])
        self.assertEqual(result[-1]['wait_result']['actual_ticks'],2)
        self.assertTrue(result[-1]['events']['level_won'])

    def test_already_satisfied_left_zone_returns_zero_without_forced_advance(self):
        state=observation(12)
        state['zombies']=[dict(id=101,row=0,x=160.,on_board=True)]
        env=RawOracle([state],[counters()])
        result=oracle_macro_wait(env,state,dict(type='wait',ticks=300,until='left_zone_occupied'),lambda _:None)
        self.assertEqual(env.commands,['WAIT 0'])
        self.assertEqual(result[-1]['ticks_advanced'],0)
        self.assertTrue(result[-1]['wait_result']['initial_condition_satisfied'])
        self.assertEqual(len(env.annotations),1)

    def test_stalled_public_clock_has_a_finite_logic_bound(self):
        states=[observation(20) for _ in range(4)]
        env=RawOracle(states,[counters() for _ in states])
        result=oracle_macro_wait(env,states[0],dict(type='wait',ticks=3,until='timeout'),lambda _:None)
        self.assertEqual(result[-1]['ticks_advanced'],0)
        self.assertEqual(result[-1]['wait_result']['logic_steps'],3)
        self.assertEqual(result[-1]['wait_result']['stalled_clock_steps'],3)


if __name__ == '__main__':
    unittest.main()
