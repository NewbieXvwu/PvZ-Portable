"""Frozen paired-evaluation cells for task78 and their pre-registered statistics.

Seeds come from the reserved blocks [78000000, 78002000); the seed audit in
artifacts/task78/seed_audit_summary.json found no collision with any local panel,
conclusion file, archive identity or the v5 checkpoint's trained_on list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
from pvz_tasks import deck_for_level

DAY_DECK = (0, 1, 2, 3, 4, 5)
NIGHT_N10_DECK = (0, 1, 2, 3, 4, 5, 7, 8, 9, 11)
POOL_P9K_DECK = (0, 1, 2, 3, 4, 5, 7, 16, 19)
FOG_DECK = (9, 8, 10, 16, 5, 23, 17, 19, 2, 3)  # configs/human_demo_tasks.json fog_user_l31_snow
ROOF_DECK = (0, 1, 2, 3, 32, 35, 33)

# (task_id, terrain, level, deck, multiplier, pairs)
CELLS = (
    ('day_l7_d7', 'day', 7, DAY_DECK, 1.0, 64),
    ('night_l14_n10', 'night', 14, NIGHT_N10_DECK, 1.0, 32),
    ('night_l17_n10', 'night', 17, NIGHT_N10_DECK, 1.0, 32),
    ('night_l20_n10', 'night', 20, NIGHT_N10_DECK, 1.0, 32),
    ('pool_l21_p9k', 'pool', 21, POOL_P9K_DECK, 1.0, 32),
    ('pool_l22_p9k', 'pool', 22, POOL_P9K_DECK, 1.0, 32),
    ('pool_l23_p9k', 'pool', 23, POOL_P9K_DECK, 1.0, 32),
    # L25 is a native conveyor level (task75) with no p9k task family; the panel's
    # hard pool level L24 stands in, recorded as a deviation in the task document.
    ('pool_l24_p9k', 'pool', 24, POOL_P9K_DECK, 1.0, 32),
    ('fog_l31_snow_m100', 'fog', 31, FOG_DECK, 1.0, 32),
    ('roof_l41_r7', 'roof', 41, ROOF_DECK, 1.0, 32),
    ('roof_l42_r7', 'roof', 42, ROOF_DECK, 1.0, 32),
    ('roof_l43_r7', 'roof', 43, ROOF_DECK, 1.0, 32),
)
SEED_START = 78000000


PANELS = dict(day='artifacts/stage4/day_panel.json',
              night='artifacts/stage4/night_panel_n10.json',
              pool='artifacts/stage4/pool_panel.json')


def panel_deck(terrain, level):
    """Deck of the frozen evaluation task this cell reuses, never a hand-written one."""
    main = Path('/home/newbiexvwu/PvZ-search')
    if terrain == 'fog':
        document = json.loads((ROOT/'configs/human_demo_tasks.json').read_text())
        task = next(t for t in document['tasks'] if t['task_id'] == 'fog_user_l31_snow')
        return tuple(task['deck'])
    if terrain == 'roof':
        return tuple(deck_for_level(level, family='roof_r7'))
    document = json.loads((main/PANELS[terrain]).read_text())
    suffix = '_eval_n10' if terrain == 'night' else '_eval_p9k' if terrain == 'pool' else '_eval'
    task = next(t for t in document['tasks'] if t['task_id'] == f'{terrain}_l{level}{suffix}')
    return tuple(task['deck'])


def cells():
    result, seed = [], SEED_START
    for task_id, terrain, level, deck, multiplier, pairs in CELLS:
        seeds = list(range(seed, seed + pairs))
        seed += pairs
        if level in (21, 22, 23, 24) and not ({16, 43} & set(deck)):
            raise ValueError(f'{task_id}: deck does not cover water')
        if level >= 41 and 33 not in deck:
            raise ValueError(f'{task_id}: deck lacks flower pots')
        if deck != panel_deck(terrain, level):
            raise ValueError(f'{task_id}: deck differs from the frozen task family')
        result.append(dict(cell_id=task_id, task_id=task_id, terrain=terrain, level=level, deck=list(deck),
                           zombie_count_multiplier=multiplier, seeds=seeds, pairs=pairs))
        if terrain == 'fog':
            result[-1]['fog_observation'] = 'human'
    if seed > SEED_START + 2000:
        raise ValueError('cells exceed the reserved seed block')
    document=json.loads((ROOT/'configs/human_demo_tasks.json').read_text())
    demo=next(t for t in document['tasks'] if t['task_id']=='demo_fog_33_cactus_snow_m200')
    future=78100000
    for decision in (839,762,686):
        if decision not in demo['demo_start']['decisions']:
            raise ValueError('pre-registered L33 takeover point missing from its task family')
        result.append(dict(deepcopy(demo),cell_id=f'fog_l33_cactus_d{decision}',terrain='fog',
                           demo_decision=decision,future_seeds=list(range(future,future+32)),pairs=32))
        future+=32
    return result


def tasks():
    entries={}
    for cell in cells():
        entries.setdefault(cell['task_id'],{k:v for k,v in cell.items()
            if k not in ('cell_id','terrain','pairs','demo_decision','future_seeds')})
    document = dict(tasks=list(entries.values()),cells=cells(),
                    reserved=[[SEED_START,SEED_START+2000],[78100000,78102000]])
    out = ROOT/'artifacts/task78/ab'
    out.mkdir(parents=True, exist_ok=True)
    path = out/'tasks.json'
    if path.exists() and list(out.rglob('games.jsonl')):
        if json.loads(path.read_text()) != document:
            raise ValueError('cannot change the frozen panel after paired games start')
    path.write_text(json.dumps(document, indent=1)+'\n')
    summary = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   cells=[dict(cell_id=c['cell_id'],task_id=c['task_id'], terrain=c['terrain'], pairs=c['pairs'],
                               seed_role='future' if 'future_seeds' in c else 'environment',
                               seeds=[(c.get('future_seeds') or c['seeds'])[0],
                                      (c.get('future_seeds') or c['seeds'])[-1]]) for c in cells()],
                   total_pairs=sum(c['pairs'] for c in cells()))
    (out/'tasks_summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    print(json.dumps(summary, indent=1))


def binomial_tail(n, p, first, last):
    return sum(math.comb(n, k)*p**k*(1-p)**(n-k) for k in range(first, last+1))


def clopper_lower(k, n, alpha=0.025):
    """One-sided lower bound: largest p with P(X >= k | n, p) <= alpha."""
    if k == 0:
        return 0.0
    low, high = 0.0, k/n
    for _ in range(80):
        mid = (low+high)/2
        if binomial_tail(n, mid, k, n) < alpha:
            low = mid
        else:
            high = mid
    return (low+high)/2


def clopper_upper(k, n, alpha=0.025):
    if k == n:
        return 1.0
    low, high = k/n, 1.0
    for _ in range(80):
        mid = (low+high)/2
        if binomial_tail(n, mid, 0, k) > alpha:
            low = mid
        else:
            high = mid
    return (low+high)/2


def mcnemar(gain, harm):
    n = gain + harm
    if not n:
        return 1.0
    return min(1.0, 2*sum(math.comb(n, k) for k in range(min(gain, harm)+1))/2**n)


def load_arm(directory, expected, use_future=False):
    path = Path(directory)/'games.jsonl'
    if not path.is_file():
        return None, sorted(expected)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(type(r.get('won')) is not bool or r.get('terminal') is not True for r in rows):
        raise ValueError(f'{path}: only terminal games with boolean wins are valid')
    if use_future and any(type(r.get('future_seed')) is not int for r in rows):
        raise ValueError(f'{path}: demonstration games must carry their future identity')
    field='future_seed' if use_future else 'seed'
    seeds = [int(r[field]) for r in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f'{path}: duplicate seeds')
    if set(seeds) - set(expected):
        raise ValueError(f'{path}: seeds outside the frozen cell')
    return {int(r[field]): r['won'] for r in rows}, sorted(set(expected)-set(seeds))


def game_gates(results, pooled):
    """Only complete cells can support the pre-registered victory gates."""
    gates={}
    for terrain in ('day','night','pool','fog','roof'):
        planned=[c for c in cells() if c['terrain']==terrain]
        complete=all(results.get(c['cell_id'],{}).get('complete') is True for c in planned)
        if not complete:
            gates[terrain]=dict(status='incomplete',passed=None)
            continue
        acc=pooled[terrain]
        if terrain in ('day','night','pool'):
            floor=all(results[c['cell_id']]['net'] >= (-4 if c['pairs']==64 else -2) for c in planned)
            passed=acc['conservative_lower'] > -.10 and floor
            gates[terrain]=dict(status='evaluated',passed=passed,
                                conservative_lower=acc['conservative_lower'],cell_floors_met=floor)
        else:
            direction=(results['fog_l31_snow_m100']['net']>=0 and
                       sum(results[c['cell_id']]['net'] for c in planned if 'future_seeds' in c)>=0
                       if terrain=='fog' else all(results[c['cell_id']]['net']>=0 for c in planned))
            gates[terrain]=dict(status='evaluated',passed=acc['net']>0 and acc['mcnemar_p']<.05 and direction,
                                net=acc['net'],mcnemar_p=acc['mcnemar_p'],direction_met=direction)
    return gates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['tasks', 'report'])
    parser.add_argument('--root', default=str(ROOT/'artifacts/task78/ab'))
    args = parser.parse_args()
    if args.phase == 'tasks':
        return tasks()
    root = Path(args.root)
    results, pooled = {}, {}
    for cell in cells():
        expected = set(cell.get('future_seeds') or cell['seeds'])
        base = root/cell['cell_id']
        old, old_missing = load_arm(base/'v3'/cell['task_id'], expected,'future_seeds' in cell)
        new, new_missing = load_arm(base/'v4'/cell['task_id'], expected,'future_seeds' in cell)
        if old is None or new is None:
            results[cell['cell_id']] = dict(terrain=cell['terrain'], pairs=0,
                                            complete=False, missing_old=old_missing,
                                            missing_new=new_missing)
            continue
        seeds = sorted(set(old) & set(new))
        gain = sum(new[s] and not old[s] for s in seeds)
        harm = sum(old[s] and not new[s] for s in seeds)
        entry = dict(terrain=cell['terrain'], pairs=len(seeds),
                     old_wins=sum(old[s] for s in seeds), new_wins=sum(new[s] for s in seeds),
                     gain=gain, harm=harm, net=gain-harm,
                     mcnemar_p=mcnemar(gain, harm),
                     complete=len(seeds) == cell['pairs'] and not old_missing and not new_missing)
        results[cell['cell_id']] = entry
        acc = pooled.setdefault(cell['terrain'], dict(pairs=0, old_wins=0, new_wins=0, gain=0, harm=0))
        acc['pairs'] += len(seeds); acc['old_wins'] += entry['old_wins']
        acc['new_wins'] += entry['new_wins']; acc['gain'] += gain; acc['harm'] += harm
    for terrain, acc in pooled.items():
        n = acc['pairs']
        acc['net'] = acc['gain']-acc['harm']
        acc['mcnemar_p'] = mcnemar(acc['gain'], acc['harm'])
        acc['conservative_lower'] = (clopper_lower(acc['gain'], n) - clopper_upper(acc['harm'], n)) if n else None
    gates=game_gates(results,pooled)
    document=dict(cells=results,terrain=pooled,gates=gates,
                  complete=all(v['status']=='evaluated' for v in gates.values()),
                  all_game_gates_pass=all(v['passed'] is True for v in gates.values()),
                  includes_terminal_and_stability_gates=False)
    out = root/'paired.json'
    out.write_text(json.dumps(document,indent=1)+'\n')
    print(json.dumps(document,indent=1))


if __name__ == '__main__':
    main()
