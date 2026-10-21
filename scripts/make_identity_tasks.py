"""Generate new task19 perturbation families and globally excluded held-out card sets."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_tasks import deck_for_seed, profile_for_deck, terrain_covered


def make(source, catalog, train_count=8192, holdouts_each=3):
    """Pure generator after engine queries; exclude card sets, regardless of slot order, globally."""
    tasks = []
    heldout = []
    signatures = set()
    for level in (7, 8, 9, 10):
        original = next(t for t in source['tasks'] if t['task_id'] == f'day_l{level}_train_d7')
        options = catalog[level]
        warnings = dict(options['not_recommended'])
        pool = sorted(p for p in options['available_plants'] if 0 <= p < 40 and not warnings.get(p, 1))
        task = copy.deepcopy(original)
        task.update(task_id=f'day_l{level}_train_d7p', seeds=list(range(1840000 + (level - 7) * 10000,
                                                                    1840000 + (level - 7) * 10000 + train_count)))
        task['deck_perturb'] = dict(swaps=[1, 2], pool=pool, engine_options=options, excluded_decks=[])
        tasks.append(task)
        attempt = 0
        for i in range(holdouts_each):
            while True:
                deck = deck_for_seed(task, 1919000 + 10000 * level + attempt)
                attempt += 1
                signature = tuple(sorted(deck))
                if signature not in signatures:
                    break
                if attempt > 4096:
                    raise ValueError('not enough distinct held-out decks')
            signatures.add(signature)
            held = copy.deepcopy(original)
            held.update(task_id=f'day_l{level}_eval_d7p_holdout{i + 1}', deck=list(deck),
                        seeds=list(range(30000, 30256)), split='heldout_deck',
                        holdout_semantics='unordered plant set excluded from every perturbation training family')
            heldout.append(held)
    for task in tasks:
        task['deck_perturb']['excluded_decks'] = [list(v) for v in sorted(signatures)]
    return dict(schema_version=1, purpose='task19 identity-head collection; new task families only',
                warning_scope='raw engine per-card recommendation flags; interactive whole-deck OnStartButton checks not called',
                tasks=tasks + heldout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/stage4/day_panel_identity.json')
    parser.add_argument('--resource-dir', type=Path, default=Path.home() / '.cache/pvz-research-resources')
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--train-count', type=int, default=8192)
    args = parser.parse_args()
    if not 1 <= args.train_count <= 8192:
        parser.error('--train-count must be in 1..8192')
    source = json.loads(args.manifest.read_text())
    catalog = {}
    with EventWaitEnv(args.resource_dir, executable=args.executable) as env:
        for level in (7, 8, 9, 10):
            task = next(t for t in source['tasks'] if t['task_id'] == f'day_l{level}_train_d7')
            deck = tuple(task['deck'])
            env.reset(deck=deck, task=TaskSpec(level=level, seed=1840000 + (level - 7) * 10000,
                                              profile=profile_for_deck(deck), playthrough=task.get('playthrough', 2)))
            catalog[level] = env.deck_options()
    result = make(source, catalog, args.train_count)
    # Audit every published training seed, never merely a random sample of possible held-out collisions.
    excluded = {tuple(sorted(t['deck'])) for t in result['tasks'] if t.get('split') == 'heldout_deck'}
    decks = set()
    for task in result['tasks']:
        if 'deck_perturb' not in task:
            continue
        for seed in task['seeds']:
            deck = deck_for_seed(task, seed)
            assert len(set(deck) - set(task['deck'])) in (1, 2)
            assert terrain_covered(task['level'], deck) and tuple(sorted(deck)) not in excluded
            decks.add(tuple(sorted(deck)))
    result['generation_audit'] = dict(training_seeds=4 * args.train_count, distinct_training_card_sets=len(decks),
                                      heldout_card_sets=len(excluded), all_heldout_sets_excluded=True,
                                      source_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                                      executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result['generation_audit']), flush=True)


if __name__ == '__main__':
    main()
