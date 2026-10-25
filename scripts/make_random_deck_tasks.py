"""task93: freeze uniformly sampled, engine-validated decks for three new families."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_env import PvZEnv, TaskSpec
from pvz_constants import upgrade_plants
from pvz_tasks import (deck_for_level, deck_for_seed, profile_for_deck,
                       random_deck_covered, random_deck_draws)

FAMILIES = ((7, 'day_l7_train_rand'), (12, 'night_l12_train_rand'), (21, 'pool_l21_train_rand'))


def heldout_decks(manifest: Path) -> list[list[int]]:
    tasks = json.loads(manifest.read_text())['tasks']
    required = next(task for task in tasks if task['task_id'] == 'day_l7_eval_d7p_holdout1')
    excluded = {tuple(sorted(task['deck'])) for task in tasks if task.get('split') == 'heldout_deck'}
    excluded.add(tuple(sorted(required['deck'])))
    return [list(deck) for deck in sorted(excluded)]


def generate(env, *, excluded, base=93_000_000, train_count=48) -> dict:
    tasks, audits = [], {}
    for index, (level, task_id) in enumerate(FAMILIES):
        start = base + index * 10_000
        splits = dict(cold=list(range(start, start + 32)),
                      tiers=list(range(start + 32, start + 48)),
                      train=list(range(start + 100, start + 100 + train_count)))
        seeds = [seed for split in splits.values() for seed in split]
        base_deck = deck_for_level(level)
        capability_profile = replace(profile_for_deck(base_deck),
                                     owned_upgrade_plants=tuple(sorted(upgrade_plants())))
        env.reset(deck=base_deck, task=TaskSpec(level=level, seed=seeds[0], playthrough=2,
                                               profile=capability_profile))
        options = env.deck_options()
        recommended = dict(options['not_recommended'])
        # Candidate pool per perturb_family.engine_family precedent: Crazy Dave candidates,
        # non-upgrade/special cards, and only cards the engine does not flag as
        # not-recommended for this level. Stored as available_plants so the runtime
        # sampler in pvz_tasks.random_deck_draws reads exactly what was validated.
        pool = sorted(p for p in env.deck_pool()['crazy_dave_pool']
                      if 0 <= p < 40 and not recommended.get(p, 1))
        options['available_plants'] = pool
        options['pool_filter'] = 'crazy_dave_pool ∩ [0,40) ∩ not_recommended==0 (perturb_family.engine_family)'
        task = dict(task_id=task_id, level=level, deck=list(base_deck), playthrough=2,
                    zombie_count_multiplier=1.0, seeds=seeds, splits=splits,
                    deck_random=dict(sampler='uniform_length_ordered_sample_v1',
                                     engine_options=options,
                                     excluded_decks=excluded, validated_decks={}))
        rejected = Counter()
        for seed in seeds:
            env.reset(deck=base_deck, task=TaskSpec(level=level, seed=seed, playthrough=2,
                                                   profile=capability_profile))
            for attempt, deck in random_deck_draws(task, seed):
                if not random_deck_covered(task, deck):
                    rejected['terrain_or_heldout'] += 1
                    continue
                warnings = env.deck_warnings(deck)['warnings']
                if warnings:
                    rejected.update(warnings)
                    continue
                try:
                    env.reset(deck=deck, task=TaskSpec(level=level, seed=seed, playthrough=2,
                                                       profile=profile_for_deck(deck)))
                except RuntimeError:
                    rejected['reset_rejected'] += 1
                    continue
                warnings = env.deck_warnings(deck)['warnings']
                if warnings:
                    rejected.update(warnings)
                    continue
                task['deck_random']['validated_decks'][str(seed)] = dict(attempt=attempt, deck=list(deck))
                assert deck_for_seed(task, seed) == deck
                break
            else:
                raise RuntimeError(f'no legal draw for {task_id} seed {seed}')
        counts = Counter(card for seed in seeds for card in deck_for_seed(task, seed))
        audits[task_id] = dict(seeds=len(seeds),
                              length_counts=dict(Counter(len(deck_for_seed(task, seed)) for seed in seeds)),
                              plant_counts={str(card): counts[card] for card in task['deck_random']['engine_options']['available_plants']},
                              rejected=dict(rejected), all_engine_warnings_empty=True,
                              all_reset_profiles_valid=True, heldout_matches=0)
        tasks.append(task)
        print(json.dumps(dict(task_id=task_id, **audits[task_id])), flush=True)
    return dict(schema_version=1, purpose='task93 random deck families; separate from fixed/perturbed decks',
                sampling='Uniform length in 6..10, then uniform ordered samples without replacement; '
                         'length is fixed during engine-warning/terrain rejection sampling.',
                tasks=tasks, generation_audit=audits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heldout-manifest', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--resource-dir', type=Path, default=Path.home()/'.cache/pvz-research-resources')
    parser.add_argument('--out', type=Path, default=ROOT/'configs/random_deck_tasks.json')
    parser.add_argument('--train-count', type=int, default=48)
    args = parser.parse_args()
    if not 1 <= args.train_count <= 8192:
        parser.error('--train-count must be in 1..8192')
    with PvZEnv(resource_dir=args.resource_dir, executable=args.executable, headless=True) as env:
        document = generate(env, excluded=heldout_decks(args.heldout_manifest), train_count=args.train_count)
        document['provenance'] = dict(
            executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
            heldout_manifest_sha256=hashlib.sha256(args.heldout_manifest.read_bytes()).hexdigest(),
            engine_identity=env.episode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=1)+'\n')


if __name__ == '__main__':
    main()
