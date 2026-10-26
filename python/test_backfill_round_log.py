import copy
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

from backfill_round_log import backfill
from curriculum_queue import pack_labels, run_round
from pvz_selection_refresh import current_cells


class BackfillRoundLogTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.runs = self.root / 'hard'
        self.qc_out = self.root / 'qc.out'
        self.qc_out.touch()
        self.student = str(self.root / 'round_000011/student/model.pt')
        self.state = dict(round=1, student=self.student, student_round=11,
                          offsets={'day_l7_train_d7_m125': 6},
                          cells={'day_l7_train_d7': dict(windows={
                              '1.25': dict(recent=[1, 0, 0, 0, 1, 1], wins=3, games=6)})},
                          log=[dict(round=1, groups=[], changes=['kept'])])

    def add_group(self, label, offset, minute, outcomes, beta=0, demo=False, rc=0, round_index=1):
        task = 'demo_fog_m200' if demo else 'day_l7_train_d7_m125'
        folder = self.runs / f'{label}_2'
        folder.mkdir(parents=True)
        config = dict(task_id=task, frames_label=label, seeds='2', seed_offset='0' if demo else str(offset),
                      rollout_policy=self.student, dagger_beta=str(beta), workers='2', manifest='/panel',
                      demo_archive='/demo' if demo else 'None', demo_decision='100',
                      future_seeds='42,43' if demo else 'None')
        (folder / 'config.json').write_text(json.dumps(config))
        (folder / 'games.jsonl').write_text(''.join(json.dumps(dict(won=bool(won), terminal=True)) + '\n'
                                                  for won in outcomes))
        stamp = f'2026-10-26T12:{minute:02d}:00'
        with self.qc_out.open('a') as handle:
            handle.write(f'{stamp} round {round_index} {label}: workers=2/2 (MemAvailable n/a)\n')
            handle.write(f'{stamp} round {round_index} {label} {task} mult={100 if demo else 1.25}'
                         + (' start=100' if demo else '')
                         + f' offset={offset} games=2 wins=1 rate=0.50 beta={beta} rc={rc} 10s\n')
        return dict(base=task if demo else 'day_l7_train_d7', multiplier=100 if demo else 1.25,
                    task_id=task, offset=offset, games=2, manifest='/panel', label=label,
                    collection_started=datetime.fromisoformat(stamp).timestamp(),
                    collection_finished=datetime.fromisoformat(stamp).timestamp(),
                    student=self.student, student_round=11, workers=2, beta=beta)

    def test_backfill_restores_order_recipe_and_selection_without_recounting(self):
        first = self.add_group('first', 0, 1, [1, 0])
        self.add_group('missing', 2, 2, [0, 0])
        last = self.add_group('last', 4, 3, [1, 1])
        self.add_group('mixed', 6, 4, [1, 1], beta=0.5)
        self.add_group('demo', 72, 5, [1, 1], demo=True)
        self.state['log'][0]['groups'] = [first, last]
        before = copy.deepcopy(self.state)
        self.assertTrue(any('suffix' in item['reason'] for item in current_cells(
            self.state, self.student, self.runs)[2]))
        added, skipped = backfill(self.state, self.qc_out, self.runs)
        self.assertEqual(skipped, [])
        self.assertEqual([group['label'] for group in self.state['log'][0]['groups']],
                         ['first', 'missing', 'last', 'mixed', 'demo'])
        self.assertEqual(len(added), 3)
        self.assertEqual(added[-1]['offset'], 72)
        self.assertEqual(added[-1]['future_seeds'], [42, 43])
        self.assertEqual(added[-1]['demo_archive'], '/demo')
        self.assertEqual(added[-1]['demo_decision'], 100)
        self.assertEqual(added[0]['student_round'], 11)
        self.assertEqual(self.state['cells'], before['cells'])
        self.assertEqual(self.state['offsets'], before['offsets'])
        self.assertEqual(self.state['log'][0]['changes'], ['kept'])
        cells, _, excluded = current_cells(self.state, self.student, self.runs)
        self.assertEqual(excluded, [])
        self.assertEqual((cells[0]['wins'], cells[0]['games']), (3, 6))
        self.assertEqual(len(pack_labels(self.state)), 5)
        self.assertEqual(backfill(self.state, self.qc_out, self.runs), ([], []))
        self.add_group('pending', 74, 6, [1, 1], demo=True, round_index=2)
        saved = copy.deepcopy(self.state)
        pending, skipped = backfill(self.state, self.qc_out, self.runs)
        self.assertEqual(len(pending), 1)
        self.assertEqual(skipped, [])
        self.assertEqual(self.state['log'], saved['log'])
        self.assertEqual(self.state['round'], 1)
        self.assertEqual(self.state['completed_groups'][0]['label'], 'pending')
        self.assertEqual(backfill(self.state, self.qc_out, self.runs), ([], []))
        args = SimpleNamespace(state=self.root / 'state.json', log=self.root / 'controller.log')
        self.assertTrue(run_round(args, self.state, [], []))
        self.assertEqual(self.state['round'], 2)
        self.assertEqual(self.state['log'][-1]['groups'][0]['label'], 'pending')
        self.assertNotIn('completed_groups', self.state)
        self.assertEqual(len(pack_labels(self.state)), 6)
        self.assertEqual(self.state['cells'], saved['cells'])
        self.assertEqual(self.state['offsets'], saved['offsets'])

    def test_dry_run_and_write_only_accept_complete_matching_successes(self):
        self.add_group('valid', 2, 2, [0, 0])
        self.add_group('failed', 4, 3, [1, 1], rc=1)
        self.add_group('short', 6, 4, [1])
        self.add_group('wrong', 8, 5, [1, 1])
        path = self.runs / 'wrong_2/config.json'
        config = json.loads(path.read_text())
        config['task_id'] = 'different'
        path.write_text(json.dumps(config))
        state_path = self.root / 'state.json'
        state_path.write_text(json.dumps(self.state))
        original = state_path.read_bytes()
        command = [sys.executable, str(ROOT / 'scripts/backfill_round_log.py'), '--state', str(state_path),
                   '--qc-out', str(self.qc_out), '--runs', str(self.runs)]
        preview = json.loads(subprocess.check_output(command, text=True))
        self.assertTrue(preview['dry_run'])
        self.assertEqual([group['label'] for group in preview['added']], ['valid'])
        self.assertEqual({group['label'] for group in preview['skipped']}, {'short', 'wrong'})
        self.assertEqual(state_path.read_bytes(), original)
        subprocess.check_output([*command, '--write'], text=True)
        written = json.loads(state_path.read_text())
        self.assertEqual(written['cells'], self.state['cells'])
        self.assertEqual(written['offsets'], self.state['offsets'])
        self.assertEqual([group['label'] for group in written['log'][0]['groups']], ['valid'])


if __name__ == '__main__':
    unittest.main()
