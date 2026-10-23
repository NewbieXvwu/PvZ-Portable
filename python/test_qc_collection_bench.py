"""Evidence gates must reject false completion without launching games."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qc_collection_bench', ROOT / 'scripts/bench_qc_collection.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class EvidenceGateTest(unittest.TestCase):
    def rows(self, seeds):
        return [dict(seed=s, won=False, arm='fast_search') for s in seeds]

    def test_wrong_matching_count_is_not_frozen_coverage(self):
        with self.assertRaisesRegex(ValueError, 'exactly cover'):
            bench.validate_rows(self.rows([11, 12]), [10, 11])

    def test_duplicate_rows_cannot_replace_a_missing_game(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            bench.validate_rows(self.rows([10, 10]), [10, 11])

    def test_cumulative_previous_games_are_not_this_batch(self):
        with self.assertRaisesRegex(ValueError, 'exactly cover'):
            bench.validate_rows(self.rows(list(range(16))), list(range(8, 16)))

    def test_string_false_is_not_a_valid_game_outcome(self):
        rows=self.rows([10]);rows[0]['won']='False'
        with self.assertRaisesRegex(ValueError, 'invalid'):
            bench.validate_rows(rows,[10])

    def test_valid_unordered_complete_batch(self):
        bench.validate_rows(self.rows([12, 10, 11]), [10, 11, 12])

    def test_changed_frozen_model_fails_before_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model';p.write_bytes(b'old')
            frozen=dict(input_sha256={str(p):bench.sha(p)},source_python_sha256={})
            p.write_bytes(b'new')
            with self.assertRaisesRegex(ValueError, 'frozen input changed'):
                bench.require_inputs(frozen)

    def design(self):
        return dict(workers=2, games_per_arm_per_task=16, stages=[
            dict(id=f'{t}_{a}_{b}', task_id=t, arm=a, block=b, seeds=list(range(b*8, b*8+8)))
            for t in bench.TASKS for a in ('old', 'cq') for b in (0, 1)])

    def test_incomplete_design_cannot_be_reported_as_complete(self):
        design = self.design()
        bench.require_complete_design(design)
        design['stages'].pop()
        with self.assertRaisesRegex(ValueError, '16 distinct stages'):
            bench.require_complete_design(design)

    def test_same_count_with_unpaired_seeds_cannot_be_reported(self):
        design = self.design()
        design['stages'][-1]['seeds'][-1] = 100
        with self.assertRaisesRegex(ValueError, 'identical seeds'):
            bench.require_complete_design(design)

    def test_repeated_block_cannot_replace_the_other_block(self):
        design = self.design()
        design['stages'][-1]['block'] = 0
        with self.assertRaisesRegex(ValueError, 'both blocks'):
            bench.require_complete_design(design)

    def test_stage_resource_evidence_checks_duration_count_and_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'timing.json'; file.write_text('{}')
            timing = dict(stage='stage', wall_seconds=5, cpu_seconds=8, games=1, samples=[])
            evidence = bench.stage_evidence(timing, self.rows([10]), file)
            self.assertIsNone(evidence['mem_available_bytes'])
            self.assertEqual(evidence['wins'], 0)
            timing['games'] = 2
            with self.assertRaisesRegex(ValueError, 'game count'):
                bench.stage_evidence(timing, self.rows([10]), file)
            timing.update(games=1, samples=[dict(engine_processes=3)])
            with self.assertRaisesRegex(ValueError, 'two game processes'):
                bench.stage_evidence(timing, self.rows([10]), file)

    def test_followup_plan_reuses_frozen_tasks_seeds_and_selects_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root/'followup'; out.mkdir()
            reference = self.design()
            manifest = root/'original-manifest.json'
            tasks = [dict(task_id=t, seeds=list(range(100, 116)), deck=[0, 1, 2, 3, 4, 5])
                     for t in bench.TASKS]
            manifest.write_text(json.dumps(dict(tasks=tasks)))
            for s in reference['stages']:
                s['seeds'] = list(range(100+s['block']*8, 108+s['block']*8))
            reference.update(manifest=str(manifest), original_panel_sha256='panel-sha',
                             input_sha256={}, source_python_sha256={})
            ref = root/'reference.json'; ref.write_text(json.dumps(reference))
            (out/'sources.json').write_text(json.dumps({a:dict(source=str(root)) for a in ('old','cq')}))
            def command(source, values, task, offset, label):
                return ['python', '--workers', '2', '--dagger-beta', '0', '--executable', values['executable'],
                        '--out', str(out/'runs'/label)]
            engine = root/'new-engine'
            with patch.object(bench, 'sha', return_value='fixture-sha'), patch.object(bench, 'command_from_source', command):
                result = bench.plan(out, engine, ref)
            self.assertEqual(json.loads(Path(result['manifest']).read_text())['tasks'], tasks)
            self.assertEqual(result['input_sha256'][str(ref)], 'fixture-sha')
            self.assertEqual(result['original_panel_sha256'], 'panel-sha')
            for stage in result['stages']:
                if stage['arm'] == 'cq':
                    self.assertEqual(stage['command'][stage['command'].index('--executable')+1], str(engine))
            bench.require_complete_design(result)
            with self.assertRaisesRegex(ValueError, 'already frozen'):
                bench.plan(out, engine, ref)

    def test_metadata_alone_cannot_prove_a_complete_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'meta.json').write_text(json.dumps(dict(task=dict(seed=10),outcome=dict(won=False,final_wave=3),
                                                          context=dict(level=7,multiplier=1.5))))
            (root/'frames.npz').write_bytes(b'test reader supplies frames')
            row=dict(seed=10,won=False,final_wave=3,decisions=2)
            task=dict(level=7,zombie_count_multiplier=1.5,deck=[0])
            with self.assertRaisesRegex(ValueError,'missing decision frames'):
                bench.audit_archive(root,row,task,reader=lambda p:[dict(action={},label={},packets=[[0,0]])])

    def test_readable_complete_archive_retains_seed_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'meta.json').write_text(json.dumps(dict(task=dict(seed=10),outcome=dict(won=False,final_wave=3),
                                                          context=dict(level=7,multiplier=1.5))))
            (root/'frames.npz').write_bytes(b'test reader supplies frames')
            frames=[dict(action={},label={},packets=[[0,0]]),dict(wave=3)]
            r=bench.audit_archive(root,dict(seed=10,won=False,final_wave=3,decisions=1),
                                 dict(level=7,zombie_count_multiplier=1.5,deck=[0]),reader=lambda p:frames)
            self.assertEqual(r['frames'],2);self.assertEqual(r['seed'],10)
            self.assertEqual(r['frames_sha256'],bench.sha(root/'frames.npz'))


if __name__=='__main__':unittest.main()
