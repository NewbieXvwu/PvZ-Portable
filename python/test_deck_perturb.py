import copy
import random
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
from make_identity_tasks import make
from pvz_tasks import deck_for_seed, terrain_covered
from render_episode import training_task_kwargs


def task(level=7, deck=(0, 1, 2, 3, 4, 5, 7)):
    options = dict(level=level, playthrough=2, available_plants=list(range(40)),
                   not_recommended=[[p, 0] for p in range(40)])
    return dict(level=level, playthrough=2, deck=list(deck), seeds=[1840000],
                deck_perturb=dict(swaps=[1, 2], pool=list(range(40)), engine_options=options))


class DeckPerturbTest(unittest.TestCase):
    def test_seed_determinism_exact_replacements_and_no_global_rng_mutation(self):
        spec = task()
        before = copy.deepcopy(spec)
        state = random.getstate()
        orders = set()
        for seed in range(100):
            deck = deck_for_seed(spec, seed)
            self.assertEqual(deck, deck_for_seed(copy.deepcopy(spec), seed))
            self.assertEqual(len(deck), 7)
            self.assertEqual(len(set(deck)), 7)
            removed = set(spec['deck']) - set(deck)
            added = set(deck) - set(spec['deck'])
            self.assertEqual(len(removed), len(added))
            self.assertIn(len(added), (1, 2))
            orders.add(deck)
        self.assertGreater(len(orders), 90)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(spec, before)

    def test_terrain_required_cards_survive_sampling(self):
        for level, required in ((21, 16), (31, 16), (41, 33)):
            spec = task(level, (0, 1, 2, 3, 4, 5, required))
            for seed in range(100):
                deck = deck_for_seed(spec, seed)
                self.assertIn(required, deck)
                self.assertTrue(terrain_covered(level, deck))

    def test_locked_warned_upgrade_and_impossible_requests_fail(self):
        for bad in (40, 43, 48, 99):
            spec = task()
            spec['deck_perturb']['pool'].append(bad)
            with self.assertRaisesRegex(ValueError, 'pool'):
                deck_for_seed(spec, 1)
        spec = task()
        spec['deck_perturb']['engine_options']['not_recommended'][8][1] = 1
        with self.assertRaisesRegex(ValueError, 'warned'):
            deck_for_seed(spec, 1)
        spec = task()
        spec['deck_perturb']['pool'] = spec['deck']
        with self.assertRaisesRegex(ValueError, 'not enough'):
            deck_for_seed(spec, 1)

    def test_global_heldout_sets_cannot_appear_in_any_training_family(self):
        originals = []
        catalog = {}
        for level in (7, 8, 9, 10):
            original = task(level)
            catalog[level] = original.pop('deck_perturb')['engine_options']
            original['task_id'] = f'day_l{level}_train_d7'
            originals.append(original)
        source = {'tasks': originals}
        before = copy.deepcopy(source)
        result = make(source, catalog, train_count=128)
        held = {tuple(sorted(t['deck'])) for t in result['tasks'] if t.get('split') == 'heldout_deck'}
        self.assertEqual(len(held), 12)
        training = [t for t in result['tasks'] if 'deck_perturb' in t]
        self.assertEqual(len(training), 4)
        for spec in training:
            self.assertGreaterEqual(min(spec['seeds']), 1840000)
            for seed in spec['seeds']:
                self.assertNotIn(tuple(sorted(deck_for_seed(spec, seed))), held)
        self.assertEqual(source, before)

    def test_engine_warned_decks_are_never_sampled(self):
        # task28: the whole-deck OnStartButton warnings arrive as an engine-produced exclusion table.
        spec = task()
        spec['deck_perturb']['require_no_warnings'] = True
        spec['deck_perturb']['engine_warned_decks'] = [[0, 2, 3, 4, 5, 6, 7]]
        banned = (0, 2, 3, 4, 5, 6, 7)
        for seed in range(64):
            self.assertNotEqual(tuple(sorted(deck_for_seed(spec, seed))), banned)
        spec['deck_perturb'].pop('engine_warned_decks')
        with self.assertRaisesRegex(ValueError, 'engine_warned_decks'):
            deck_for_seed(spec, 1)

    def test_legacy_task_kwargs_unchanged_and_perturbed_kwargs_use_actual_deck(self):
        legacy = dict(level=7, deck=[0, 1, 2, 3, 4, 5, 7], seeds=[10])
        expected = dict(level=7, deck=legacy['deck'], seed=10, first_aid_owned=False,
                        task_extra=dict(wave_cap=None, zombie_count_multiplier=None))
        self.assertEqual(training_task_kwargs(legacy), expected)
        spec = task()
        self.assertEqual(training_task_kwargs(spec, 1840007)['deck'], list(deck_for_seed(spec, 1840007)))

    def test_keep_plants_are_never_replacement_targets(self):
        # task31: a measured keep card's slot is never a replacement target; everything else still is.
        spec = task(level=12, deck=(0, 1, 2, 3, 4, 5, 7, 8, 9, 11))
        spec['deck_perturb']['keep_plants'] = [8, 9]
        removed_seen = set()
        for seed in range(300):
            deck = deck_for_seed(spec, seed)
            self.assertIn(8, deck)
            self.assertIn(9, deck)
            removed = set(spec['deck']) - set(deck)
            self.assertFalse(removed & {8, 9})
            removed_seen |= removed
        self.assertEqual(deck_for_seed(spec, 7), deck_for_seed(spec, 7))
        self.assertEqual(len(removed_seen), len(spec['deck']) - 2)

    def test_keep_plants_must_be_base_cards_and_leave_enough_replaceable_slots(self):
        spec = task()
        spec['deck_perturb']['keep_plants'] = [8]
        with self.assertRaisesRegex(ValueError, 'keep_plants'):
            deck_for_seed(spec, 1)
        spec = task(deck=(0, 1, 2, 3, 4, 5))
        spec['deck_perturb']['keep_plants'] = [0, 1, 2, 3, 4]
        with self.assertRaisesRegex(ValueError, 'replaceable'):
            deck_for_seed(spec, 1)

    def test_whatif_uses_archived_deck_instead_of_a_current_manifest(self):
        import episode_query
        archived = list(deck_for_seed(task(), 1840001))
        meta = dict(task=dict(seed=1840001, level=7, policy='student', deck=archived,
                              checkpoint='/frozen/model.pt', wave_cap=None, zombie_count_multiplier=1.0))
        with patch.object(episode_query, 'collect', return_value={'outcome': {}, 'frames': []}) as collect:
            episode_query._run_variant(meta, {}, '/resources')
        self.assertEqual(collect.call_args.args[5], archived)
        self.assertEqual(collect.call_args.kwargs['checkpoint'], '/frozen/model.pt')


if __name__ == '__main__':
    unittest.main()
