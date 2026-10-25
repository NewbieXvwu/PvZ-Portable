"""Evidence checks for task77: a shortened or relabelled run must not pass identity."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from task77_search_hotpath import abba_baseline, compare_frames, paired_interval, family_interval


class SearchHotpathEvidenceTest(unittest.TestCase):
    def test_experiment_commands_enforce_the_two_engine_budget(self):
        from types import SimpleNamespace
        from task77_search_hotpath import command
        with self.assertRaisesRegex(ValueError, 'limited to two game processes'):
            command(SimpleNamespace(workers=4), 'day', 'test', 4)

    def test_qc_cohorts_exclude_ancestor_matches_and_midblock_shutdown(self):
        from task77_search_hotpath import qc_cohorts
        rows = [dict(cell=cell, index=index,
                     before={'qc_controllers': [9, 88]},
                     after={'qc_controllers': [9] if cell == 'pool' and index == 0 else [9, 88]})
                for cell in ('day', 'pool') for index in range(4)]
        self.assertEqual(qc_cohorts(rows, {88}), {'day': [88], 'pool': []})
        with self.assertRaisesRegex(ValueError, 'incomplete ABBA load'):
            qc_cohorts(rows[:-1], {88})

    def test_length_difference_at_end_is_reported(self):
        self.assertEqual(compare_frames([{'tick': 1}], [])['reason'], 'length')
        self.assertEqual(compare_frames([], [{'tick': 1}])['index'], 0)

    def test_checks_labels_targets_and_observations_even_when_action_matches(self):
        frame = {'tick': 1, 'action': {'type': 'wait'}, 'label': {'type': 'plant'},
                 'targets': [[0, 0, 0, 0.1234]], 'sun': 25}
        for key, replacement in [('label', {'type': 'wait'}),
                                 ('targets', [[0, 0, 0, 0.1235]]), ('sun', 50)]:
            mismatch = compare_frames([frame], [dict(frame, **{key: replacement})])
            self.assertIn(key, mismatch['fields'])

    def test_retains_float_bit_relevant_sign_and_verification_values(self):
        self.assertIsNotNone(compare_frames([{'value': 0.0}], [{'value': -0.0}]))
        self.assertIsNotNone(compare_frames([{'verification': {'gap': 0.1}}],
                                             [{'verification': {'gap': 0.2}}]))

    def test_only_verification_wall_time_and_dict_order_are_excluded(self):
        left = {'tick': 1, 'verification': {'gap': 0.1, 'seconds': 2}}
        right = {'verification': {'seconds': 3, 'gap': 0.1}, 'tick': 1}
        self.assertIsNone(compare_frames([left], [right]))

    def test_abba_a_arm_is_the_same_build_not_the_pgo_production_binary(self):
        import json
        import tempfile
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            production = root / 'production-pgo'
            production.write_bytes(b'production')
            (root / 'freeze.json').write_text(json.dumps({'engine': {'path': str(production)}}))
            explicit = root / 'release-before'
            explicit.write_bytes(b'x')
            self.assertEqual(abba_baseline(SimpleNamespace(out=root, baseline=explicit)), explicit)
            with self.assertRaises(ValueError):
                abba_baseline(SimpleNamespace(out=root, baseline=None))

    def test_sample_reader_rejects_partial_parallel_arrays_and_bad_module_ids(self):
        import struct
        import tempfile
        from prof_symbolize import load
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'samples.bin'
            good = b'PVZPROF 1 1\n/engine\n' + struct.pack('<2Q2I', 123, 456, 0, 0)
            path.write_bytes(good)
            self.assertEqual(load(path), (['/engine'], [123, 456], [0, 0]))
            for data in (good[:-1], good + b'\0\0\0\0',
                         good[:-4] + struct.pack('<I', 4)):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    load(path)

    def test_interval_rejects_invalid_or_empty_evidence(self):
        for ratios in ([], [0], [-1], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                paired_interval(ratios, draws=100)
        result = paired_interval([.8] * 8, draws=100)
        self.assertAlmostEqual(result['geomean'], .8)
        self.assertAlmostEqual(result['bootstrap95'][0], .8)
        with self.assertRaises(ValueError):
            family_interval([])
        interval = family_interval([dict(cell=cell, ratio=.8) for cell in ('day', 'pool')
                                    for _ in range(2)], draws=100)
        self.assertAlmostEqual(interval[0], .8)
        self.assertAlmostEqual(interval[1], .8)

    def test_abba_audit_checks_raw_timings_configuration_and_label_archives(self):
        import json
        import os
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        import task77_search_hotpath as bench
        import pvz_frame_store

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('before', 'after', 'weights'):
                (root / name).write_bytes(name.encode())
            bench.save(root / 'manifest.json', {'tasks': [{'task_id': 'day_task', 'seeds': [1, 2]}]})
            bench.save(root / 'freeze.json', {k: {'path': str(root / 'weights')}
                                            for k in ('student', 'value', 'static')})
            evidence = {'order': ['A', 'B', 'B', 'A'], 'games': [
                dict(cell='day', seed=seed, old=1., new=.8, ratio=.8) for seed in (1, 2)],
                'observables': ['wall', 'engine_cpu'],
                'cpu': [dict(cell='day', index=index, arm=arm, processes=2, wall_seconds=2.,
                             seconds=2. if arm == 'A' else 1.8)
                        for index, arm in enumerate(['A', 'B', 'B', 'A'])],
                'cpu_ratios': [dict(cell='day', old=2., new=1.8, ratio=.9)]}
            for key, name in (('baseline', 'before'), ('candidate', 'after')):
                evidence[key] = {'path': str(root / name), 'sha256': bench.digest(root / name)}
            for index, arm in enumerate(evidence['order']):
                path = root / f'abba_{index}_{arm}' / f'day_abba_{index}_{arm}_2'
                path.mkdir(parents=True)
                config = dict(task_id='day_task', dagger_beta='1.0', verify_samples='9',
                    student_candidates='2', horizon='2500', samples='3', seed_offset='0', seeds='2',
                    halving='True', executable=str(root / ('before' if arm == 'A' else 'after')),
                    manifest=str(root / 'manifest.json'), rollout_policy=str(root / 'weights'),
                    value=str(root / 'weights'), static_value=str(root / 'weights'))
                bench.save(path / 'config.json', config)
                (path / 'games.jsonl').write_text(''.join(json.dumps(dict(seed=seed,
                    decisions=1, won=True, state_hash='same', seconds=1. if arm == 'A' else .8))
                    + '\n' for seed in (1, 2)))
                start = 10_000_000_000 + index * 2_000_000_000
                os.utime(path / 'config.json', ns=(start, start))
                os.utime(path / 'games.jsonl', ns=(start + 1_000_000_000,) * 2)
            args = SimpleNamespace(out=root, count=2)
            frames = [{'tick': 0}, {'tick': 60, 'label': {'type': 'wait'}}]
            with patch.object(bench, 'CELLS', {'day': 'day_task'}), \
                    patch.object(pvz_frame_store, 'read_frames', return_value=frames), \
                    patch.object(bench, 'paired_interval', return_value={}), \
                    patch('builtins.print'):
                bench.save(root / 'abba_evidence.json', evidence)
                bench.audit_abba(args)
                audited = json.loads((root / 'abba_evidence.json').read_text())
                self.assertEqual(audited['identity_audit']['frame_comparisons'], 6)
                self.assertAlmostEqual(audited['cpu_ratios'][0]['ratio'], .9)
                self.assertEqual(audited['identity_audit']['observables'], ['wall', 'engine_cpu'])
                # Engine CPU must be recomputable from the per-arm records, so a hand-edited
                # ratio or a truncated arm list cannot survive the audit.
                evidence['cpu'][1]['seconds'] = 1.0
                bench.save(root / 'abba_evidence.json', evidence)
                with self.assertRaisesRegex(ValueError, 'engine CPU ratio differs'):
                    bench.audit_abba(args)
                evidence['cpu'][1]['seconds'] = 1.8
                dropped = evidence.pop('cpu')
                bench.save(root / 'abba_evidence.json', evidence)
                with self.assertRaisesRegex(ValueError, 'missing engine CPU measurements'):
                    bench.audit_abba(args)
                evidence['cpu'] = dropped
                evidence['games'][0]['ratio'] = .1
                bench.save(root / 'abba_evidence.json', evidence)
                with self.assertRaisesRegex(ValueError, 'timing differs'):
                    bench.audit_abba(args)
                evidence['games'][0]['ratio'] = .8
                bench.save(root / 'abba_evidence.json', evidence)
                config_path = root / 'abba_1_B/day_abba_1_B_2/config.json'
                config = json.loads(config_path.read_text())
                config['halving'] = 'False'
                bench.save(config_path, config)
                with self.assertRaisesRegex(ValueError, 'configuration differs'):
                    bench.audit_abba(args)


    def test_hotspot_categories_are_disjoint_and_keep_unlisted_samples(self):
        import json
        import tempfile
        from types import SimpleNamespace
        import task77_search_hotpath as bench
        self.assertEqual(bench.hotspot_category('pvz-portable:DataArray<Reanimation>::Iterator::operator++'),
                         'data_array')
        self.assertEqual(bench.hotspot_category('pvz-portable:Reanimation::Update'), 'animation')
        self.assertEqual(bench.hotspot_category('libopenblas.so.0:sgemm_kernel_HASWELL'), 'openblas')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bench.save(root / 'profile_summary.json', {'cells': {'day': {
                'samples': 100, 'invalid_sample_files': [],
                'cpp_samples': [['pvz-portable:Reanimation::Update', 30],
                                ['libopenblas.so.0:sgemm_kernel_HASWELL', 20]]}}})
            bench.hotspots(SimpleNamespace(out=root))
            summary = json.loads((root / 'hotspots.json').read_text())
            self.assertEqual(summary['samples'], 100)
            self.assertAlmostEqual(summary['categories']['animation'], .3)
            self.assertAlmostEqual(summary['categories']['unlisted'], .5)
            self.assertAlmostEqual(sum(summary['categories'].values()), 1.0)
            bench.save(root / 'profile_summary.json', {'cells': {'day': {
                'samples': 0, 'invalid_sample_files': [], 'cpp_samples': []}}})
            with self.assertRaisesRegex(ValueError, 'no symbolized samples'):
                bench.hotspots(SimpleNamespace(out=root))


    def test_engine_cpu_sampler_counts_only_the_measured_binary(self):
        import os
        import shutil
        import subprocess
        import sys
        import tempfile
        import time
        from task77_search_hotpath import EngineCpu
        with tempfile.TemporaryDirectory() as folder:
            # A private copy gives the sampler an executable path no other process on the
            # box can match; attribution is by /proc/<pid>/exe, as it is for the two arms.
            engine = Path(folder) / 'pvz-portable'
            shutil.copy2(os.path.realpath(sys.executable), engine)
            burn = 'import time\nt=time.time()\nwhile time.time()-t<0.4: pass\n'
            with EngineCpu([engine], interval=0.05) as sampler:
                subprocess.run([str(engine), '-c', burn], check=True)
                time.sleep(0.15)
            self.assertEqual(sampler.processes(), 1)
            self.assertGreater(sampler.seconds(), 0.1)
            self.assertLess(sampler.seconds(), 2.0)
            with EngineCpu(['/nonexistent/pvz-portable'], interval=0.05) as empty:
                subprocess.run([str(engine), '-c', burn], check=True)
                time.sleep(0.1)
            self.assertEqual(empty.processes(), 0)
            self.assertEqual(empty.seconds(), 0.0)


if __name__ == '__main__':
    unittest.main()
