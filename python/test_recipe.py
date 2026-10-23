import argparse
import copy
from pathlib import Path
import tempfile
import unittest

from pvz_recipe import load_recipe, training_args, panel_args, materialize_recipe, validate, write_recipe

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
        for update in [dict(schema_version=2), dict(data=None), dict(out='relative'),
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

    def test_historical_recipes_resolve_all_training_settings(self):
        for name, labels in [('dw4i',170),('dw5i',281),('identity_dw3_s1',82)]:
            recipe = load_recipe(ROOT/f'recipes/{name}.json', 'training')
            # Counts verified against original command records in the migration evidence.
            self.assertEqual(len(recipe['labels']), labels)
            self.assertEqual(recipe['seed'], 1)
            self.assertEqual(recipe['packet_head'], 'identity')
            self.assertEqual(recipe['validation'], dict(seed=0,fraction=.1))
            self.assertTrue(Path(recipe['out']).is_relative_to(ROOT))
            self.assertTrue(Path(recipe['cache_dir']).is_relative_to(ROOT))
            self.assertEqual(validate(copy.deepcopy(recipe)), recipe)


if __name__ == '__main__':
    unittest.main()
