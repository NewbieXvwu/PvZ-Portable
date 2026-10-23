import argparse
import copy
from pathlib import Path
import tempfile
import unittest

from pvz_recipe import load_recipe, training_args, validate, write_recipe

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
