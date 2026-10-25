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
import os
import subprocess

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


def tasks(out=None):
    entries={}
    for cell in cells():
        entries.setdefault(cell['task_id'],{k:v for k,v in cell.items()
            if k not in ('cell_id','terrain','pairs','demo_decision','future_seeds')})
    document = dict(tasks=list(entries.values()),cells=cells(),
                    reserved=[[SEED_START,SEED_START+2000],[78100000,78102000]])
    out = ROOT/'artifacts/task78/ab' if out is None else out
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


def game_gates(results, pooled, planned=None):
    """Only complete cells can support the pre-registered victory gates."""
    gates={}
    panel=cells() if planned is None else planned
    for terrain in ('day','night','pool','fog','roof'):
        cohort=[c for c in panel if c['terrain']==terrain]
        complete=all(results.get(c['cell_id'],{}).get('complete') is True for c in cohort)
        if not complete:
            gates[terrain]=dict(status='incomplete',passed=None)
            continue
        acc=pooled[terrain]
        if terrain in ('day','night','pool'):
            floor=all(results[c['cell_id']]['net'] >= (-4 if c['pairs']==64 else -2) for c in cohort)
            passed=acc['conservative_lower'] > -.10 and floor
            gates[terrain]=dict(status='evaluated',passed=passed,
                                conservative_lower=acc['conservative_lower'],cell_floors_met=floor)
        else:
            direction=(results['fog_l31_snow_m100']['net']>=0 and
                       sum(results[c['cell_id']]['net'] for c in cohort if 'future_seeds' in c)>=0
                       if terrain=='fog' else all(results[c['cell_id']]['net']>=0 for c in cohort))
            gates[terrain]=dict(status='evaluated',passed=acc['net']>0 and acc['mcnemar_p']<.05 and direction,
                                net=acc['net'],mcnemar_p=acc['mcnemar_p'],direction_met=direction)
    return gates


def digest(path):
    from task78_data import sha
    return sha(path)


def frozen_panel(root):
    path=root/'tasks.json'
    receipt=json.loads((ROOT/'MEMORY/docs/task78-ab-plan.json').read_text())
    if digest(path)!=receipt['sha256']:
        raise ValueError('paired panel differs from the committed 512-pair preregistration')
    return json.loads(path.read_text())['cells']


def runtime_plan(root):
    """Freeze all model, code, engine and demo inputs before either arm starts."""
    import torch
    from pvz_frame_store import frame_path
    from task78_train import MIX,MIX_DIAGNOSTIC,BASELINE_SHA
    main=Path('/home/newbiexvwu/PvZ-search')
    panel=frozen_panel(root)
    model_paths=dict(v5=main/'artifacts/value_net/v5/model.pt',v6=ROOT/'artifacts/task78/v6/model.pt',
                     student=main/'artifacts/student/dw4i_s1/model.pt',
                     static=main/'artifacts/hard/value_branch1/model.pt')
    models={name:dict(path=str(path),sha256=digest(path)) for name,path in model_paths.items()}
    if models['v5']['sha256']!=BASELINE_SHA:
        raise ValueError('baseline is not the adopted v5')
    training=json.loads((ROOT/'artifacts/task78/v6/train_log.json').read_text())
    if training.get('complete') is not True:
        raise ValueError('candidate training has not finished')
    evaluation={s for c in panel for s in (c.get('future_seeds') or c['seeds'])}
    for arm in ('v5','v6'):
        checkpoint=torch.load(model_paths[arm],map_location='cpu',weights_only=False)
        if evaluation & set(checkpoint['trained_on']):
            raise ValueError(f'{arm}: training overlaps frozen evaluation identities')
        if arm=='v6':
            provenance=checkpoint['input_provenance'].get('original',{})
            if checkpoint['mix'] not in (MIX,MIX_DIAGNOSTIC) or 'original' not in checkpoint['input_provenance']:
                raise ValueError('candidate did not use the complete original-corpus recipe')
            # A diagnostic fallback is allowed to run, but it can never pass as the
            # preregistered candidate: the receipt has to match the mixture exactly.
            if checkpoint['mix']==MIX and provenance.get('preregistered_corpus') is not True:
                raise ValueError('original mixture must carry the preregistered corpus receipt')
            if checkpoint['mix']==MIX_DIAGNOSTIC and provenance.get('preregistered_corpus') is not False:
                raise ValueError('diagnostic mixture must be marked as not preregistered')
    source=json.loads((ROOT/'artifacts/task78/sources.json').read_text())
    demos={}
    for cell in panel:
        if 'future_seeds' not in cell or cell['task_id'] in demos:
            continue
        paths={Path(r['demo_archive']) for r in source['archives']
               if r.get('demo_archive') and Path(r['demo_archive']).name==cell['task_id']}
        if len(paths)!=1:
            raise ValueError('cannot identify the frozen human demonstration source')
        archive=paths.pop()
        meta=json.loads((archive/'meta.json').read_text())
        if meta['task']['id']!=cell['task_id'] or meta['task']['deck']!=cell['deck']:
            raise ValueError('demo identity or deck differs from the frozen family')
        demos[cell['task_id']]=dict(path=str(archive),files={str(p):digest(p)
            for p in (archive/'meta.json',frame_path(archive),archive/'ticks.jsonl')})
    code=['scripts/search_play.py','scripts/pvz_tasks.py','scripts/render_episode.py',
          'python/pvz_env.py','python/pvz_event_env.py','python/pvz_rollout_policy.py',
          'python/pvz_student_observation.py','python/pvz_value_net.py','scripts/task78_ab.py']
    code=sorted(set(code)|{str(p.relative_to(ROOT)) for p in (ROOT/'python').glob('pvz_*.py')})
    result=dict(panel_sha256=digest(root/'tasks.json'),models=models,demos=demos,
                python='/home/newbiexvwu/.venvs/ml/bin/python3',
                software=dict(python=sys.version,torch=str(torch.__version__)),
                executable=dict(path=str(ROOT/'build/pvz-portable'),sha256=digest(ROOT/'build/pvz-portable')),
                code={str(ROOT/p):digest(ROOT/p) for p in code},workers=2,
                control='only leaf value changes; beta1/samples3/verify9/student2/h2500/gap300/skip61')
    path=root/'runtime.json'
    if path.exists() and json.loads(path.read_text())!=result:
        raise ValueError('an input changed after paired runtime inputs were frozen')
    if not path.exists():
        path.write_text(json.dumps(result,indent=1)+'\n')
    return result


def command(cell,arm,plan,root):
    """Explicit production arguments, including takeover and independent future IDs."""
    legacy_arm='v3' if arm=='v5' else 'v4'
    output=root/cell['cell_id']/legacy_arm/cell['task_id']
    args=[plan.get('python',sys.executable),str(ROOT/'scripts/search_play.py'),
          '--manifest',str(root/'tasks.json'),'--task-id',cell['task_id'],
          '--out',str(output),'--value',plan['models'][arm]['path'],
          '--rollout-policy',plan['models']['student']['path'],
          '--static-value',plan['models']['static']['path'],
          '--executable',plan['executable']['path'],'--resource-dir','/home/newbiexvwu/.cache/pvz-research-resources',
          '--workers','2','--dagger-beta','1','--samples','3','--verify-samples','9',
          '--student-candidates','2','--horizon','2500','--min-gap','300','--per-card','1',
          '--halving','--rollout-sim-skip','61','--seed-list',','.join(map(str,cell['seeds']))]
    if 'future_seeds' in cell:
        args+=['--demo-archive',plan['demos'][cell['task_id']]['path'],
               '--demo-decision',str(cell['demo_decision']),
               '--future-seeds',','.join(map(str,cell['future_seeds']))]
    return args,output


def run_arm(root,arm):
    import fcntl
    with (ROOT/'artifacts/task78/games-controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan=runtime_plan(root)
        for cell in frozen_panel(root):
            # No fifth/third engine can be hidden behind another task78 stage.
            for path in Path('/proc').glob('[0-9]*/cmdline'):
                try:
                    first=path.read_bytes().split(b'\0')[0]
                    if first==plan['executable']['path'].encode():
                        raise RuntimeError('another task78 game is live; wait before starting paired workers')
                except (FileNotFoundError,ProcessLookupError,PermissionError):
                    continue
            runtime_plan(root)  # Recheck mutable external model/demo inputs at every cell.
            args,output=command(cell,arm,plan,root)
            output.mkdir(parents=True,exist_ok=True)
            receipt=dict(runtime_sha256=digest(root/'runtime.json'),command=args,
                         cell_id=cell['cell_id'],arm=arm)
            receipt_path=output/'cell_run.json'
            if receipt_path.exists() and json.loads(receipt_path.read_text())!=receipt:
                raise ValueError('cell results belong to another frozen runtime')
            receipt_path.write_text(json.dumps(receipt,indent=1)+'\n')
            print(f'{arm} {cell["cell_id"]}',flush=True)
            with (output/'run.log').open('a') as log:
                subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,
                    env=os.environ | dict(PVZ_NO_POPUP='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['tasks','freeze','v5','v6','report'])
    parser.add_argument('--root', default=str(ROOT/'artifacts/task78/ab'))
    args = parser.parse_args()
    if args.phase == 'tasks':
        if not Path(args.root).is_absolute():
            parser.error('absolute paired output root required')
        return tasks(Path(args.root))
    root = Path(args.root)
    if not root.is_absolute():
        parser.error('absolute paired output root required')
    if args.phase=='freeze':
        return runtime_plan(root)
    if args.phase in ('v5','v6'):
        return run_arm(root,args.phase)
    panel=frozen_panel(root)
    results, pooled = {}, {}
    checked_runtime=None
    for cell in panel:
        expected = set(cell.get('future_seeds') or cell['seeds'])
        base = root/cell['cell_id']
        old, old_missing = load_arm(base/'v3'/cell['task_id'], expected,'future_seeds' in cell)
        new, new_missing = load_arm(base/'v4'/cell['task_id'], expected,'future_seeds' in cell)
        for arm,legacy,rows in [('v5','v3',old),('v6','v4',new)]:
            if rows:
                if checked_runtime is None:
                    checked_runtime=runtime_plan(root)
                plan=checked_runtime
                args_expected,_=command(cell,arm,plan,root)
                record=json.loads((base/legacy/cell['task_id']/'cell_run.json').read_text())
                if record!=dict(runtime_sha256=digest(root/'runtime.json'),command=args_expected,
                                cell_id=cell['cell_id'],arm=arm):
                    raise ValueError('paired rows are not tied to the frozen runtime inputs')
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
    gates=game_gates(results,pooled,panel)
    import torch
    candidate=torch.load(ROOT/'artifacts/task78/v6/model.pt',map_location='cpu',weights_only=False)
    diagnostic=not candidate.get('preregistered_corpus',True)
    document=dict(cells=results,terrain=pooled,gates=gates,
                  complete=all(v['status']=='evaluated' for v in gates.values()),
                  all_game_gates_pass=all(v['passed'] is True for v in gates.values()),
                  includes_terminal_and_stability_gates=False,
                  diagnostic=diagnostic,preregistered_recipe_met=not diagnostic)
    out = root/'paired.json'
    out.write_text(json.dumps(document,indent=1)+'\n')
    print(json.dumps(document,indent=1))


if __name__ == '__main__':
    main()
