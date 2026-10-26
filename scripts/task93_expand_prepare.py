"""Build the 720-game task93 dose corpus (A + 144 + 576) without launching games.

Reuses the frozen task84 A recipe and every immutable segment already on disk; only the
three Mac expand labels are new. Writes seed 1/2 recipes under --root and, unless
--plan-only, packs the corpus and asserts the row totals.

  python3 /abs/repo/scripts/task93_expand_prepare.py --plan-only
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
from pvz_tasks import profile_for_deck, random_deck_covered

FROZEN = ROOT / 'artifacts/task93'
CONTROL = Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/gatepower/artifacts/task84')
FRAMES_V2 = Path('/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2')
PANEL = Path('/home/newbiexvwu/PvZ-search/artifacts/stage4/random_deck_panel.json')
AUDIT = FROZEN / 'mac_import_audit.json'
EXPAND_SEGMENTS = FROZEN / 'expand_pack/frames/packs/segments'
R11_SHA = '1cf82f5e6cf7994ca961d91e878d4ac65a4563eea2be50a1e96a9787c22dfd4c'
EXPECTED = dict(total=17_132_763, old=17_012_427,
                new={'d7dag93m100_fast_search': 46_098,
                     'n12dag93m050_fast_search': 35_347,
                     'p21w93m050_fast_search': 38_891})


def link(source, target):
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError(f'conflicting link: {target}')
    elif target.exists():
        raise ValueError(f'refusing to replace: {target}')
    else:
        target.symlink_to(source.resolve(), target_is_directory=True)


def hardlink(source, target):
    if target.exists():
        if not os.path.samefile(source, target):
            raise ValueError(f'conflicting segment: {target}')
    else:
        os.link(source, target)


def expand_families():
    """Seed ranges and labels of the imported Mac batch, straight from its audit receipt."""
    audit = json.loads(AUDIT.read_text())
    families = []
    for row in audit['families']:
        start, stop = row['seed_range']
        families.append(dict(task_id=row['task_id'], label=row['label'],
                             seeds=list(range(start, stop + 1)), multiplier=row['multiplier'],
                             rows=row['unsampled_usable_rows'], archives=row['archives']))
    if len(families) != 3 or sum(len(f['seeds']) for f in families) != 576:
        raise ValueError('unexpected expand receipt shape')
    return families


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=FROZEN/'expand')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    template = json.loads((FROZEN/'seed_1/recipe.json').read_text())
    if len(template['labels']) != 351:
        raise ValueError('expected the frozen task93 corpus with 351 labels')
    families = expand_families()
    panel = {t['task_id']: t for t in json.loads(PANEL.read_text())['tasks']}
    checks = {}
    for family in families:
        base = panel[family['task_id'].removesuffix('_expand')]
        for seed in family['seeds']:
            folder = FRAMES_V2/f'seed_{seed}/{family["label"]}'
            meta = json.loads((folder/'meta.json').read_text())
            task, outcome = meta['task'], meta['outcome']
            if (task['task_id'] != family['task_id'] or task['seed'] != seed
                    or task['level'] != base['level'] or task['playthrough'] != 2
                    or task['zombie_count_multiplier'] != family['multiplier']
                    or task['checkpoint_sha256'] != R11_SHA
                    or meta['label_source'] != 'teacher' or task['arm'] != 'fast_search'
                    or not (folder/'frames.npz').is_file()
                    or type(outcome['won']) is not bool):
                raise ValueError(f'invalid expand archive: {folder}')
            deck = task['deck']
            if not 6 <= len(deck) <= 10 or len(set(deck)) != len(deck):
                raise ValueError(f'illegal deck length: {folder}')
            if not random_deck_covered(base, deck) or deck in base['deck_random']['excluded_decks']:
                raise ValueError(f'deck misses terrain or hits heldout: {folder}')
            if profile_for_deck(tuple(deck)).seed_slot_count != len(deck):
                raise ValueError(f'profile slot count differs: {folder}')
        checks[family['label']] = len(family['seeds'])

    corpus = args.root/'training_frames'
    plan = dict(root=str(args.root), corpus=str(corpus), new_games=576,
                new_labels={f['task_id']: f['label'] for f in families},
                archives=checks, prepared=False)
    if not args.plan_only:
        corpus.mkdir(parents=True, exist_ok=True)
        baseline = json.loads((CONTROL/'variant_a/recipe.json').read_text())
        for source in sorted(Path(baseline['data']).glob('seed_*')):
            link(source, corpus/source.name)
        for source in sorted((FROZEN/'training_frames').glob('seed_*')):
            link(source, corpus/source.name)
        for family in families:
            for seed in family['seeds']:
                source = FRAMES_V2/f'seed_{seed}'
                link(source, corpus/source.name)
        segments = corpus/'packs/segments'
        segments.mkdir(parents=True, exist_ok=True)
        sources = [Path(baseline['data'])/'packs/segments',
                   FROZEN/'training_frames/packs/segments', EXPAND_SEGMENTS]
        reused = 0
        for folder in sources:
            for source in sorted(folder.glob('*.npz')):
                hardlink(source, segments/source.name)
                reused += 1
        plan.update(prepared=True, linked_segments=reused)

    recipes = {}
    for seed in (1, 2):
        recipe = copy.deepcopy(template)
        labels = template['labels'] + [f['label'] for f in families]
        recipe.update(seed=seed, data=str(corpus), labels=labels,
                      out=str(args.root/f'seed_{seed}/student'),
                      cache_dir=str(args.root/'cache'),
                      provenance=dict(source_path=str(FROZEN/'seed_1/recipe.json'),
                          source_sha256=hashlib.sha256((FROZEN/'seed_1/recipe.json').read_bytes()).hexdigest(),
                          notes='task93 dose step: A + 144 frozen + 576 Mac expand games; same hyperparameters; '
                                'seeds 1 and 2; no heldout selection'))
        recipe['label_weight'].update({f['label']: 1.0 for f in families})
        validate(recipe, 'training')
        out = args.root/f'seed_{seed}/recipe.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and json.loads(out.read_text()) != recipe:
            raise ValueError(f'recipe exists with different settings: {out}')
        out.write_text(json.dumps(recipe, indent=2)+'\n')
        recipes[str(seed)] = str(out)
    plan['recipes'] = recipes

    if not args.plan_only:
        from pack_training_rows import pack
        from pvz_recipe import training_row_options
        from pvz_training_pack import archive_fingerprint, pack_options
        import numpy as np
        recipe = json.loads((args.root/'seed_1/recipe.json').read_text())
        options = training_row_options(recipe)
        options.pop('source_weight', None)
        options = pack_options(**options, sources=archive_fingerprint(corpus, set(recipe['labels'])))
        target = pack(corpus, options, cache_dir=args.root/'cache', precompute_workers=2)
        with np.load(target, allow_pickle=False) as stored:
            rows = int(stored['rows'])
            names = stored['segments'].tolist()
        per = {}
        for name in names:
            with np.load(target.parent/name, allow_pickle=False) as seg:
                meta = json.loads(str(seg['meta']))
                per[meta['labels'][0]] = int(seg['rows'])
        if rows != EXPECTED['total']:
            raise ValueError(f'row total {rows} != {EXPECTED["total"]}')
        for label, expected in EXPECTED['new'].items():
            if per.get(label) != expected:
                raise ValueError(f'{label}: {per.get(label)} != {expected}')
        if sum(per.values()) - sum(EXPECTED['new'].values()) != EXPECTED['old']:
            raise ValueError('old label rows changed')
        plan.update(rows=rows, new_label_rows={k: per[k] for k in EXPECTED['new']},
                    labels=len(per), pack=str(target))
    (args.root/'expand_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
