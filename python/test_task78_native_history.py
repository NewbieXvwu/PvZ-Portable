from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import numpy as np
from pvz_rollout_policy import SCALARS
from pvz_student_observation import StudentHistory
from render_episode import _frame
from value_teacher_feasibility import history_record
from task78_native_history import POLICY, recover, context_digest, audit_record
from task78_native_labels import pack_trace


class NativeHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.trace = self.directory / 'native_candidate_0.jsonl'
        self.rows = []
        self.initial = dict(frame=None, action=dict(type='wait'), history=StudentHistory('v2'))
        shapes = StudentHistory('v2')
        for tick, wave, previous, plant in ((10, 1, -1, 0), (70, 2, 10, 1)):
            obs = dict(lean=1, tick=tick, wave=wave, sun=500, level=41,
                       terrain=4, roof=True, wave_count=20, grid=[[0]*9 for _ in range(6)],
                       plants=[], zombies=[], defenses=[], packets=[], grid_items=[],
                       terminal=False, result=0)
            frame = _frame(obs); frame['sun_income_rate'] = 0
            board, scalars = shapes.features(frame, None)
            native_scalars = np.zeros(len(scalars), dtype=np.float32)
            native_scalars[SCALARS] = plant
            native_scalars[SCALARS+1] = 0 if previous < 0 else 0.2
            self.rows.append(dict(observation=obs, income=0, previous_tick=previous,
                previous_plant=plant, board=[0]*board.numel(), scalars=native_scalars.tolist()))
        self.root = _frame(self.rows[0]['observation']); self.root['sun_income_rate'] = 0.0
        self.leaf = dict(self.rows[-1]['observation'], tick=130)
        self.response = dict(decisions=2, substeps=[[70, 0], [130, 0]], planted_slots=[1]+[0]*9)

    def recover(self):
        self.trace.write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        return recover(self.trace, self.initial, self.root, self.leaf, self.response, None, 'v2')

    def test_recovers_wave_change_and_last_native_action_context(self):
        state, proof = self.recover()
        self.assertEqual((state['history'].wave, state['history'].wave_tick), (2, 70))
        self.assertEqual(state['frame']['tick'], 70)
        self.assertEqual(state['action'], dict(type='wait'))
        self.assertEqual(proof['compared_float32'], 6)
        self.assertIsNone(self.initial['history'].wave)

    def test_incorrect_history_wave_input_is_rejected(self):
        self.rows[1]['scalars'][-2] = 0.5
        with self.assertRaisesRegex(ValueError, 'input bits differ'):
            self.recover()

    def test_incomplete_native_trace_is_rejected(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, 'decision coverage'):
            self.recover()

    def test_inconsistent_native_plant_count_is_rejected(self):
        self.response['planted_slots'][0] = 3
        with self.assertRaisesRegex(ValueError, 'final action context'):
            self.recover()

    def test_reuse_compares_exact_observations_but_only_consumed_action_context(self):
        state, _ = self.recover()
        first = history_record(state)
        second = deepcopy(first)
        second['previous_action'].update(packet=2, row=3, col=4)
        self.assertEqual(context_digest(first), context_digest(second))
        second['previous_frame']['sun'] += 1
        self.assertNotEqual(context_digest(first), context_digest(second))

    def test_audit_binds_history_and_compressed_native_trace(self):
        state, proof = self.recover()
        proof.update(pack_trace(self.trace))
        record = dict(candidate=0, history_source=POLICY,
                      history_checks=dict(native_history=True), native_history=proof,
                      leaf_frame=dict(tick=130), leaf_history=history_record(state))
        root = dict(history=history_record(self.initial), root_frame=self.root)
        self.assertEqual(len(audit_record(record, root, self.directory)), 1)
        (self.directory / proof['trace_file']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'trace changed'):
            audit_record(record, root, self.directory)


if __name__ == '__main__':
    unittest.main()
