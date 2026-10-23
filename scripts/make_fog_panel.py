"""Task45 fog families, paired observation arms and engine-eligible opening-card perturbations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from zone_panel import Family, Zone, build_document
from pvz_env import PvZEnv, TaskSpec
from pvz_tasks import deck_for_level, profile_for_deck, deck_for_seed
from perturb_family import engine_family, warned_decks
import pvz_constants

NORMAL_LEVELS = (31, 32, 33, 34, 36, 37, 38, 39)
OPENING_PLANTS = tuple(p['type'] for p in pvz_constants.plant_defs()
                       if p['seed'] in ('TORCHWOOD', 'PLANTERN', 'BLOVER'))
FOG = Zone(
    key='fog', task_prefix='fog_l', first_level=31, last_level=40, terrain='fog',
    default_out=ROOT / 'artifacts/task45/fog_panel.json',
    purpose='task45: fog observations and opening-card cold start',
    excluded={35: 'ScaryPotter: vasebreaker, not a normal fog level'},
    special_notes={40: 'StormyNight: conveyor and StageHasFog=false; evaluation only'},
    families=(Family('_f8', fog_observation='human'),
              Family('_f8x', fog_observation='xray'),
              Family('_f8', kind='train', exclude_levels=(40,), fog_observation='human'),
              Family('_f8x', kind='train', exclude_levels=(40,),
                     train_seed_base=41_000_000, fog_observation='xray')),
    train_seed_base=40_000_000, eval_seed_base=39_000_000, eval_seeds=32,
    eval_seed_stride=100,
    reserved_seed_ranges=((0, 38_999_999),),
)


def document(env, train_seeds=8192):
    args = argparse.Namespace(deck='', include=[], skip=[], combined=True, train_seeds=train_seeds,
                              seed_start=FOG.eval_seed_base, seeds=FOG.eval_seeds)
    result, _checks, _meta = build_document(FOG, args)
    catalog = {}
    base_tasks = {t['task_id']: t for t in result['tasks']}
    for level in NORMAL_LEVELS:
        base = deck_for_level(level)
        options, full_pool = engine_family(env, level, base, 2, 39_000_000,
                                            profile_for_deck, TaskSpec)
        # A targeted intervention: 1 or 2 replacements from the game's opening-card subset.
        # No card strength ranking and no hard-coded playing rule. All additions are engine eligible.
        pool = sorted(set(full_pool) & set(OPENING_PLANTS))
        if set(pool) != set(OPENING_PLANTS):
            raise ValueError(f'L{level}: engine pool does not cover all opening plants: {pool}')
        warned, checked = warned_decks(env, base, pool)
        catalog[str(level)] = dict(engine_options=options, opening_pool=pool,
                                  first_adventure_opening_available=sorted(
                                      set(options['first_adventure_available']) & set(OPENING_PLANTS)),
                                  engine_candidates=checked, warned=len(warned))
        for split, count, start in [('eval', 32, 39_000_000 + (level - 31) * 100),
                                     ('train', train_seeds, 42_000_000 + (level - 31) * 10000)]:
            task = dict(base_tasks[f'fog_l{level}_{split}_f8'])
            task.update(task_id=f'fog_l{level}_{split}_f8p', seeds=list(range(start, start + count)))
            task['deck_perturb'] = dict(swaps=[1, 2], pool=pool, engine_options=options,
                                      excluded_decks=[], require_no_warnings=True,
                                      engine_warned_decks=warned)
            coverage = {p: 0 for p in OPENING_PLANTS}
            for seed in task['seeds']:
                deck = deck_for_seed(task, seed)
                assert set(deck) & set(OPENING_PLANTS)
                for plant in OPENING_PLANTS:
                    coverage[plant] += plant in deck
            if not all(coverage.values()):
                raise ValueError(f'L{level} {split}: missing opening-card coverage {coverage}')
            result['tasks'].append(task)
            catalog[str(level)][split + '_opening_deck_counts'] = coverage
    result['engine_catalog'] = catalog
    result['note'] = ('Task45 new families; legacy decks unchanged. Paired evaluation arms use the same '
                      '32 seeds/level. f8p replaces 1-2 cards using only engine-eligible opening plants; '
                      'whole-deck warnings excluded. Teacher measurements decide collection eligibility.')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=FOG.default_out)
    p.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    p.add_argument('--resource-dir', type=Path, default=Path('/home/newbiexvwu/.cache/pvz-research-resources'))
    args = p.parse_args()
    with PvZEnv(args.resource_dir, executable=args.executable) as env:
        result = document(env)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(dict(path=str(args.out), tasks=len(result['tasks']),
                         opening_plants=OPENING_PLANTS, catalog=result['engine_catalog']), indent=1))


if __name__ == '__main__':
    main()
