"""task23 step 6.1: the same-seed n9 vs n10 night A/B manifest.

Reads the authoritative night panel (task21 + task25 families), keeps only the requested levels and the
two decks, and checks that the pair differs in nothing but the extra grave buster (11) in the last slot.
The output is a local panel manifest for scripts/unified_panel.py --manifest; it is not committed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

MAIN = Path('/home/newbiexvwu/PvZ-search')
N9_DECK = [0, 1, 2, 3, 4, 5, 7, 8, 9]
N10_DECK = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--panel', type=Path, default=MAIN/'artifacts/stage4/night_panel.json')
    p.add_argument('--levels', type=int, action='append', default=None)
    p.add_argument('--seeds', type=int, default=128)
    p.add_argument('--extra-card', type=int, default=None,
                   help='control: n9 plus one card that is never plantable on night grass (e.g. 43 cattail)')
    p.add_argument('--extra-suffix', default='n9c')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    levels = args.levels or [12, 14, 17]
    document = json.loads(args.panel.read_text())
    tasks = {task['task_id']: task for task in document['tasks']}
    chosen = []
    if args.extra_card is not None:
        deck = N9_DECK + [args.extra_card]
        for level in levels:
            task = copy.deepcopy(tasks[f'night_l{level}_eval_n9'])
            task['task_id'] = f'night_l{level}_eval_{args.extra_suffix}'
            task['deck'] = deck
            task['seeds'] = task['seeds'][:args.seeds]
            chosen.append(task)
        manifest = dict(schema_version=1,
                        purpose=f'task23 step 6.1 control: n9 plus card {args.extra_card} (never plantable on night grass)',
                        groups={task['task_id']: 'night' for task in chosen}, tasks=chosen,
                        semantics='same seeds and task fields as the n9 arm; only the unused tenth slot differs')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=2) + '\n')
        for task in chosen:
            print(task['task_id'], task['deck'], len(task['seeds']))
        return
    for level in levels:
        pair = []
        for suffix, deck in (('n9', N9_DECK), ('n10', N10_DECK)):
            name = f'night_l{level}_eval_{suffix}'
            task = copy.deepcopy(tasks[name])
            assert task['deck'] == deck, (name, task['deck'])
            assert task['level'] == level
            task['seeds'] = task['seeds'][:args.seeds]
            pair.append(task)
            chosen.append(task)
        assert pair[0]['seeds'] == pair[1]['seeds'], 'the pair must use the same evaluation seeds'
        left, right = copy.deepcopy(pair[0]), copy.deepcopy(pair[1])
        left.pop('task_id'); right.pop('task_id'); left.pop('deck'); right.pop('deck')
        assert left == right, f'L{level}: the pair differs outside the deck'
    manifest = dict(schema_version=1,
                    purpose='task23 step 6.1 same-seed n9 vs n10 night A/B (pure student)',
                    groups={task['task_id']: 'night' for task in chosen},
                    tasks=chosen,
                    semantics='each level is played with the same seeds and task fields; only the deck differs, '
                              'n10 = n9 + grave buster (11) in the last slot')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + '\n')
    for task in chosen:
        print(task['task_id'], task['deck'], len(task['seeds']), task['seeds'][:3], task['seeds'][-1])


if __name__ == '__main__':
    main()
