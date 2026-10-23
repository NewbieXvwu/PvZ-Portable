"""Preregistered task45 structural-deck check before lowering difficulty."""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
from fog_cold_start import prepare
from make_pool_panel import P9K_DECK
from pvz_tasks import deck_for_level


def main():
    folder = ROOT / 'artifacts/task45/deck-diagnostic'
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.glob('*/games.jsonl')):
        raise ValueError('diagnostic already started; cannot change its preregistered tasks')
    source = json.loads((ROOT / 'artifacts/task45/fog_panel.json').read_text())
    tasks = []
    for level in range(31, 35):
        base = next(t for t in source['tasks'] if t['task_id'] == f'fog_l{level}_eval_f8')
        # Transfer the controller-approved pool p9k deck, changing its daytime economic
        # card to the existing fog economy. This is a new task family, never a policy rule.
        deck = list(P9K_DECK)
        deck[1] = deck_for_level(level)[1]
        # The engine warns 'flyer' on L33/34 without an aerial answer. The additional
        # engine-eligible Blover is an experimental card, not a required policy action.
        deck.append(27)
        for suffix, cards, multiplier in [('f10b', deck, 1.0), ('f8_m050', base['deck'], .5),
                                           ('f10b_m050', deck, .5)]:
            task = copy.deepcopy(base)
            task.update(task_id=f'fog_l{level}_eval_{suffix}', deck=cards,
                        zombie_count_multiplier=multiplier)
            tasks.append(task)
    manifest = folder / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, tasks=tasks), indent=1) + '\n')
    # Reuse exactly the main comparison's model, engine and command construction.
    reference = ROOT / 'artifacts/task45/fog_panel.json'
    plan = prepare(reference, folder)
    plan.update(manifest=str(manifest), manifest_sha256=__import__('hashlib').sha256(manifest.read_bytes()).hexdigest())
    prototype = [c for c in plan['cells'] if c['suffix'] == 'f8' and c['arm'] == 'fast_search']
    cells = []
    for level in range(31, 35):
        for suffix in ('f10b', 'f8_m050', 'f10b_m050'):
            cell = copy.deepcopy(next(c for c in prototype if c['level'] == level))
            cell.update(task_id=f'fog_l{level}_eval_{suffix}', suffix=suffix)
            command = cell['command']
            command[command.index('--task-id') + 1] = cell['task_id']
            command[command.index('--manifest') + 1] = str(manifest)
            cell['out'] = str(folder / f'{cell["task_id"]}_fast_search_16')
            command[command.index('--out') + 1] = cell['out']
            command[command.index('--frames-label') + 1] = cell['task_id'] + '_fast_search'
            cells.append(cell)
    plan['cells'] = cells
    path = folder / 'plan.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    print(f'prepared {len(cells)} teacher cells / 192 games: {path}')


if __name__ == '__main__':
    main()
