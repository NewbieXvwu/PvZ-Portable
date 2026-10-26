"""Prepare task93's two recipes and an isolated corpus after all 144 games finish.

Uses task84 A's frozen archives and immutable pack segments as the control corpus.
No production files are changed, and no training or game process is launched.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_recipe import validate
from pvz_tasks import deck_for_seed
from task93_analyze import load_games

LABELS = dict(day_l7_train_rand='rand_d7_fast_search',
              night_l12_train_rand='rand_n12_fast_search',
              pool_l21_train_rand='rand_p21_fast_search')


def link(source, target):
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError(f'conflicting link: {target}')
    elif target.exists():
        raise ValueError(f'refusing to replace: {target}')
    else:
        target.symlink_to(source.resolve(), target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-recipe', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT/'artifacts/task93')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    baseline = json.loads(args.baseline_recipe.read_text())
    if len(baseline['labels']) != 348:
        raise ValueError('expected task84 A control corpus with 348 labels')
    manifest = json.loads(args.manifest.read_text())
    tasks = {t['task_id']: t for t in manifest['tasks']}
    if set(tasks) != set(LABELS):
        raise ValueError('expected all three random deck families')
    corpus = args.root/'training_frames'
    recipes = {}
    for seed in (1, 2):
        recipe = copy.deepcopy(baseline)
        recipe.update(seed=seed, data=str(corpus), labels=baseline['labels']+list(LABELS.values()),
                      out=str(args.root/f'seed_{seed}/student'),
                      cache_dir=str(args.root/'cache'))
        # Explicitly preserve default parity for new teacher supervision.
        recipe['label_weight'].update({label: 1.0 for label in LABELS.values()})
        recipe['provenance'] = dict(source_path=str(args.baseline_recipe),
            source_sha256=hashlib.sha256(args.baseline_recipe.read_bytes()).hexdigest(),
            notes='task93: task84 A corpus plus 144 random-deck teacher games; seeds 1 and 2; no heldout selection')
        validate(recipe, 'training')
        out = args.root/f'seed_{seed}/recipe.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and json.loads(out.read_text()) != recipe:
            raise ValueError(f'recipe already exists with different settings: {out}')
        out.write_text(json.dumps(recipe, indent=2)+'\n')
        recipes[str(seed)] = str(out)
    plan = dict(recipes=recipes, baseline_data=baseline['data'], corpus=str(corpus),
                new_games=144, new_labels=LABELS, prepared=False)
    if not args.plan_only:
        # Validate every finished game and its archive before exposing a training corpus.
        for family, label in LABELS.items():
            task = tasks[family]
            seeds = task['splits']['train']
            if len(seeds) != 48:
                raise ValueError('expected 48 train seeds per family')
            decks = {s: deck_for_seed(task, s) for s in seeds}
            games = args.root/'collect'/family/'games.jsonl'
            load_games(games, seeds, 'fast_search', decks)
            config = json.loads(games.with_name('config.json').read_text())
            if (float(config['dagger_beta']) != 1 or config['frames_label']+'_fast_search' != label
                    or config['task_id'] != family):
                raise ValueError(f'wrong teacher collection contract: {family}')
            for seed in seeds:
                archive = args.root/'frames'/f'seed_{seed}'/label
                meta = json.loads((archive/'meta.json').read_text())
                if (meta['task']['seed'] != seed or meta['task']['deck'] != list(decks[seed])
                        or meta['task']['task_id'] != family or not (archive/'frames.npz').is_file()):
                    raise ValueError(f'invalid random-deck training archive: {archive}')
        corpus.mkdir(parents=True, exist_ok=True)
        for source in sorted(Path(baseline['data']).glob('seed_*')):
            link(source, corpus/source.name)
        for family, label in LABELS.items():
            for seed in tasks[family]['splits']['train']:
                source = args.root/'frames'/f'seed_{seed}'
                link(source, corpus/source.name)
        # Copy directory entries only. Existing immutable segments are reused by name;
        # new segments and corpus indexes are written exclusively under task93.
        segments = corpus/'packs/segments'
        segments.mkdir(parents=True, exist_ok=True)
        reused = 0
        for source in (Path(baseline['data'])/'packs/segments').glob('*.npz'):
            target = segments/source.name
            if target.exists():
                if not os.path.samefile(source, target):
                    raise ValueError(f'conflicting immutable segment: {target}')
            else:
                os.link(source, target)
            reused += 1
        plan.update(prepared=True, linked_segments=reused)
    (args.root/'training_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(plan))


if __name__ == '__main__':
    main()
