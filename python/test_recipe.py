import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

from pvz_recipe import (load_recipe, repository_root, training_args, panel_args, materialize_recipe,
                        validate, write_recipe)

ROOT = Path(__file__).resolve().parents[1]


class RecipeTests(unittest.TestCase):
    def base(self):
        return dict(schema_version=1, kind='training', data='/data', labels=['one'], out='/output')

    def test_defaults_and_reject_typos_at_every_level(self):
        self.assertEqual(validate(self.base())['optimizer']['lr'], .002)
        for update in [dict(epohs=1), dict(optimizer=dict(lrr=.1)),
                       dict(validation=dict(fracton=.1)), dict(label_weight=dict(missing=2))]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate({**self.base(), **update})

    def test_reject_bad_types_ranges_versions_and_paths(self):
        for update in [dict(schema_version=2), dict(data=None), dict(out=''),
                       dict(epochs=True), dict(epochs=0), dict(wait_keep=0), dict(wait_keep=2),
                       dict(labels=['one', 'one']), dict(row_pool='false'), dict(packet_head='other'),
                       dict(validation=dict(fraction=1)), dict(optimizer=dict(betas=[1,.9])),
                       dict(optimizer=dict(lr=float('nan'))),dict(cache=False,source_row_report=True)]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate({**self.base(), **update})

    def test_json_duplicates_and_operational_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'r.json'
            write_recipe(path, self.base())
            args = training_args(argparse.ArgumentParser(), ['--recipe', str(path), '--out', '/elsewhere', '--threads', '2'])
            self.assertEqual(args.out, Path('/elsewhere'))
            self.assertEqual(args.resolved_recipe['threads'], 2)
            with self.assertRaises(SystemExit):
                training_args(argparse.ArgumentParser(), ['--recipe', str(path), '--epochs', '2'])
            path.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_recipe(path)

    def test_freeze_variants_and_panel_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'panel.json'
            materialize_recipe(ROOT/'recipes/panel_v2.json', path, tasks='day_l7_eval_d7',
                               task_seed_overrides={'day_l7_eval_d7': 1})
            args = panel_args(argparse.ArgumentParser(), ['--recipe', str(path), '--workers', '1'])
            self.assertEqual(args.workers, 1)
            self.assertEqual(args.pool_seeds, 64)
            self.assertEqual(args.task_seed_overrides, {'day_l7_eval_d7':1})
            with self.assertRaisesRegex(ValueError, 'frozen recipe'):
                materialize_recipe(ROOT/'recipes/panel_v2.json',path,tasks='different')
            with self.assertRaises(ValueError):
                validate({**args.resolved_recipe,'day_seed':1})

    def test_parity_cache_is_a_private_copy_and_never_clobbers(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts'))
        from verify_recipe_migration import seed_private_cache
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base/'original/v2'
            source.mkdir(parents=True)
            (source/'entry.npz').write_bytes(b'original cached bytes')
            recipe = dict(cache_dir=str(base/'private'),
                          provenance=dict(original_cache_dir=str(base/'original')))
            self.assertEqual(seed_private_cache(recipe)['copied_files'], 1)
            destination = base/'private/v2/entry.npz'
            self.assertNotEqual(destination.stat().st_ino, (source/'entry.npz').stat().st_ino)
            (source/'entry.npz').write_bytes(b'changed source')
            self.assertEqual(seed_private_cache(recipe)['copied_files'], 0)
            self.assertEqual(destination.read_bytes(), b'original cached bytes')

    def test_full_200_step_observer_with_source_row_report(self):
        import hashlib
        import json
        import sys
        from unittest.mock import patch
        sys.path.insert(0, str(ROOT / 'scripts'))
        from test_identity_training import IdentityTrainingTest
        import verify_recipe_migration as verifier
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            IdentityTrainingTest().fixture(root / 'data')
            source = root / 'command.json'
            command = ['python', str(ROOT / 'scripts/train_rollout_policy.py'),
                       '--data', str(root / 'data'), '--labels', 'example', '--out', str(root / 'old'),
                       '--cache-dir', str(root / 'cache'), '--precompute-workers', '2',
                       '--epochs', '205', '--width', '8', '--seed', '1', '--obs', 'v2', '--lr-schedule', 'cosine',
                       '--packet-head', 'identity', '--pool', 'meanmax', '--row-pool',
                       '--packet-balance', '.5', '--cell-soft-weight', '3', '--spread-weight', '8',
                       '--source-row-report', str(root / 'old/source_rows.json')]
            source.write_text(json.dumps(dict(command=command)))
            recipe = dict(schema_version=1, kind='training', data=str(root / 'data'),
                          labels=['example'], out=str(root / 'new'), cache_dir=str(root / 'cache'),
                          epochs=205, width=8, seed=1, obs='v2', packet_head='identity', pool='meanmax', lr_schedule='cosine',
                          row_pool=True, packet_balance=.5, cell_soft_weight=3, spread_weight=8,
                          source_row_report=True, precompute_workers=2, provenance=dict(source_path=str(source),
                          source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          original_cache_dir=str(root / 'absent')))
            path = root / 'recipe.json'
            write_recipe(path, recipe)
            out = root / 'parity'
            out.mkdir()
            with patch.object(verifier, 'audit_source', return_value={}), \
                 patch.object(verifier, 'foreign_trainers', return_value=[]), \
                 patch.object(verifier, 'audit_corpus', return_value={'fixture':True}):
                # CPU: this exercises the parity logic, not the device, and deterministic kernels
                # have no MPS implementation.
                verifier.verify(path, out, 'cpu')
            evidence = json.loads((out / 'parity.json').read_text())
            self.assertTrue(evidence['passed'])
            self.assertEqual(evidence['steps'], 200)
            self.assertEqual(evidence['precompute_workers'], 2)
            self.assertEqual(len(evidence['traces']['legacy']['losses']), 200)
            self.assertEqual(len(evidence['traces']['recipe']['losses']), 200)

    def test_corpus_audit_rejects_missing_rows_or_changed_splits(self):
        import json
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        from verify_recipe_migration import audit_corpus
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / 'original'
            original.mkdir()
            (original / 'metrics.json').write_text(json.dumps(dict(data_memory=dict(rows=20))))
            expected = dict(rows=20, sources=dict(one=dict(rows=20, archives=2, validation_rows=2)))
            (original / 'source_rows.json').write_text(json.dumps(expected))
            for phase in ('legacy', 'recipe'):
                (root / phase).mkdir()
                (root / phase / 'source_rows.json').write_text(json.dumps(expected))
            recipe = dict(provenance=dict(original_out=str(original)), source_row_report=True)
            self.assertTrue(audit_corpus(recipe, root, 20)['rows_match'])
            with self.assertRaises(AssertionError):
                audit_corpus(recipe, root, 19)
            (root / 'recipe/source_rows.json').write_text(json.dumps(dict(rows=20, sources={})))
            with self.assertRaises(AssertionError):
                audit_corpus(recipe, root, 20)

    def test_historical_recipes_resolve_all_training_settings(self):
        for name, labels in [('dw4i',170),('dw5i',281),('identity_dw3_s1',82)]:
            recipe = load_recipe(ROOT/f'recipes/{name}.json', 'training')
            # Counts verified against original command records in the migration evidence.
            self.assertEqual(len(recipe['labels']), labels)
            self.assertEqual(recipe['seed'], 1)
            self.assertEqual(recipe['packet_head'], 'identity')
            self.assertEqual(recipe['validation'], dict(seed=0,fraction=.1))
            root = repository_root()
            self.assertTrue(Path(recipe['out']).is_relative_to(root))
            self.assertTrue(Path(recipe['cache_dir']).is_relative_to(root))
            self.assertTrue(Path(recipe['data']).is_relative_to(root))
            # The file itself must stay portable: nothing inside the repository is absolute,
            # so the recipe survives deleting a worktree and moving to another machine.
            document = json.loads((ROOT/f'recipes/{name}.json').read_text())
            for field in ('data', 'out', 'cache_dir'):
                self.assertFalse(Path(document[field]).is_absolute(), f'{name}.{field}')
            self.assertIsNone(recipe['read_cache_dir'])
            if document.get('read_cache_dir') is not None:
                self.assertFalse(Path(document['read_cache_dir']).is_absolute(), f'{name}.read_cache_dir')
            self.assertEqual(validate(copy.deepcopy(recipe)), recipe)


if __name__ == '__main__':
    unittest.main()
