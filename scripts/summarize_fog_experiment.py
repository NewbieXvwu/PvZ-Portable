"""Persist the KB-scale task45 conclusions and per-game loss/opening-card evidence."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts/task45'


def phase(folder):
    plan = json.loads((folder / 'plan.json').read_text())
    cells, games = [], []
    for cell in plan['cells']:
        path = Path(cell['out']) / 'games.jsonl'
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        counts = Counter()
        positions = Counter()
        agreement = Counter()
        for row in rows:
            diagnostic = row['fog_diagnostics']
            agreement.update(diagnostic['teacher_agreement'])
            for plant in diagnostic['opening_plant_positions']:
                counts[plant['type']] += plant['count']
                positions[plant['type'], plant['row'], plant['col']] += plant['count']
            games.append(dict(task_id=cell['task_id'], arm=cell['arm'], seed=row['seed'],
                won=row['won'], wave=row['final_wave'], tick=row['final_tick'], deck=row['deck'],
                loss=diagnostic['loss'], openings=diagnostic['opening_plant_positions']))
        cells.append(dict(task_id=cell['task_id'], arm=cell['arm'], games=len(rows),
                          won=sum(r['won'] for r in rows), teacher_agreement=dict(agreement),
                          opening_plants=dict(counts), positions=[dict(type=k, row=r, col=c, count=n)
                           for (k, r, c), n in sorted(positions.items())]))
    return dict(manifest_sha256=plan['manifest_sha256'], models=plan['models'],
                complete=len(cells) == len(plan['cells']) and len(games) == len(plan['cells']) * 16,
                cells=cells, games=games)


def memory():
    folder = ARTIFACTS / 'memory'
    if not (folder / 'same-data-audit.json').exists():
        return None
    result = dict(same_data=json.loads((folder / 'same-data-audit.json').read_text()),
                   resource_gate=json.loads((folder / 'resource_gate.json').read_text()))
    path = folder / 'evaluation.jsonl'
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        result['games'] = rows
        result['complete'] = len(rows) == 512 and all(r['terminal'] for r in rows)
        result['cells'] = []
        for task_id in sorted({r['task_id'] for r in rows}):
            for arm in ('current', 'stack4'):
                group = [r for r in rows if r['task_id'] == task_id and r['arm'] == arm]
                result['cells'].append(dict(task_id=task_id, arm=arm, games=len(group),
                       won=sum(r['won'] for r in group), return_events=sum(len(r['returns']) for r in group),
                       post_return_plantings=sum(p['count'] for r in group for p in r['planting_after_return']),
                       post_return_seen_row_plantings=sum(p['count'] for r in group
                                for p in r['planting_after_return'] if p['seen_before'])))
    return result


def main():
    result = dict(task='task45', cold=phase(ARTIFACTS / 'cold'),
                  nonfog_replay=json.loads((ARTIFACTS / 'replay/verify.json').read_text()),
                  seed_audit=json.loads((ARTIFACTS / 'seed-audit.json').read_text()),
                  memory=memory())
    if (ARTIFACTS / 'deck-diagnostic/summary.json').exists():
        result['deck_diagnostic'] = phase(ARTIFACTS / 'deck-diagnostic')
    path = ROOT / 'MEMORY/docs/task45-fog-results.json'
    path.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f'wrote {path}: {path.stat().st_size} bytes')
    for suffix in ('f8', 'f8x', 'f8p'):
        cells = [c for c in result['cold']['cells']
                 if c['task_id'].endswith('_' + suffix) and c['arm'] == 'fast_search']
        agreement = sum((Counter(c['teacher_agreement']) for c in cells), Counter())
        print(suffix, json.dumps(dict(agreement), sort_keys=True))


if __name__ == '__main__':
    main()
