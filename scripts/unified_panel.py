"""Generate and run task23's single day, held-out-deck and night student panel."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
from run_identity_validation import descendants
import dagger_experiment_runner as guard


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2)+'\n')


def generate(day, perturb, night, version='v1', pool=None):
    """Copy authoritative task definitions; keep all twelve initial-deck holdouts intact."""
    tasks = []; groups = {}
    def add(task, group):
        tasks.append(copy.deepcopy(task)); groups[task['task_id']] = group
    def get(document, name):
        return next(t for t in document['tasks'] if t['task_id'] == name)
    for level in (1, 3, 4, 6, 7, 8, 9):
        add(get(day, f'day_l{level}_eval_d7'), 'day')
    original = copy.deepcopy(get(day, 'day_l10_eval_d7'))
    original.update(task_id='day_l10_orig_eval', deck=[], conveyor_start='original')
    add(original, 'day')
    for level in (7, 8, 9, 10):
        for held in (1, 2, 3):
            add(get(perturb, f'day_l{level}_eval_d7p_holdout{held}'), 'heldout')
    assert version in ('v1','v2')
    suffix = 'n9' if version == 'v1' else 'n10'
    for level in (11, 12, 13, 14, 16, 17, 18, 19):
        add(get(night, f'night_l{level}_eval_{suffix}'), 'night')
    if version == 'v1':
        original = copy.deepcopy(get(night, 'night_l20_eval_n9'))
        original.update(task_id='night_l20_eval_n9_orig', deck=[], conveyor_start='original')
        add(original, 'night')
    else:
        add(get(night, 'night_l20_eval_n10_orig'), 'night')
        for level in (21,22,23,24,26):
            add(get(pool, f'pool_l{level}_eval_p9k'), 'pool')
    assert len(tasks) == len(groups) == (29 if version == 'v1' else 34)
    for task in tasks:
        assert task['seeds'][:128] == list(range(30000, 30128))
        assert task['zombie_count_multiplier'] == 1 and task['wave_cap'] is None
        assert task['playthrough'] == 2 and task['sun_start'] == 50 and not task['preplanted']
        if task.get('conveyor_start') == 'original':
            assert task['deck'] == []
    result = dict(schema_version=1, purpose='task23 unified student panel', groups=groups, tasks=tasks,
                semantics='L10 holdouts retain archived initial decks and native subsequent conveyor; original L10/L20 start empty')
    if version == 'v2':
        result.update(panel_version='v2', purpose='task27 unified student panel: night n10 and pool p9k')
    return result


def summarize(folder, task, seeds, plant_types=False):
    rows = [json.loads(v) for v in (folder/'games.jsonl').read_text().splitlines()]
    assert len(rows) == len(seeds) and {r['seed'] for r in rows} == set(seeds), task['task_id']
    assert all(r['arm'] == 'fast' and r['searched'] == 0 for r in rows)
    won = sum(r['won'] for r in rows)
    assert read(folder/'summary.json')['fast']['won'] == won
    result = dict(task_id=task['task_id'], level=task['level'], deck=task['deck'],
                conveyor_start=task.get('conveyor_start'), games=len(rows), won=won, win_rate=won/len(rows),
                mean_mowers_used=sum(r['mowers_used'] for r in rows)/len(rows),
                mean_final_wave=sum(r['final_wave'] for r in rows)/len(rows),
                games_sha256=sha(folder/'games.jsonl'))
    if plant_types:
        # Dynamic conveyor slots cannot be translated through the initial deck. Native and
        # Python loops count the actual type before each successful planting instead.
        from collections import Counter
        from pvz_episode_stats import planted_by_type,planted_slots
        totals = Counter()
        unknown=0
        for row in rows:
            counts = planted_by_type(row,task)
            if counts is None:
                unknown+=1;continue
            slots=planted_slots(row)
            if slots is not None:assert sum(counts.values()) == sum(slots.values())
            assert all(0 <= int(k) < 54 and type(v) is int and v >= 0 for k,v in counts.items())
            totals.update(counts)
        result.update(planted_by_type_total=None if unknown else dict(sorted(totals.items(), key=lambda kv:int(kv[0]))),
                      planted_by_type_per_game=None if unknown else {k:v/len(rows) for k,v in sorted(totals.items(), key=lambda kv:int(kv[0]))})
        if unknown:
            result.update(planted_types_unknown_games=unknown,
                          planted_types_note='Legacy conveyor slot totals cannot identify plant types; no translation guessed.')
    return result


def evaluate(args):
    manifest = read(args.manifest)
    model_hash = sha(args.rollout_policy); binary_hash = sha(args.executable); manifest_hash = sha(args.manifest)
    selected = set(args.tasks.split(',')) if args.tasks else set(manifest['groups'])
    assert selected <= set(manifest['groups'])
    guard.OUT = args.out/'resources'; guard.OUT.mkdir(parents=True, exist_ok=True)
    guard.descendants = descendants
    guard.ENV = {**os.environ, 'PYTHONPATH': str(ROOT/'python')+':'+str(ROOT/'scripts'),
                 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MALLOC_ARENA_MAX': '2',
                 'CUDA_VISIBLE_DEVICES': '', 'PVZ_RESOURCE_DIR': str(args.resource_dir),
                 'PVZ_EXECUTABLE': str(args.executable)}
    rows = []
    for task in manifest['tasks']:
        name = task['task_id']
        if name not in selected: continue
        group = manifest['groups'][name]
        count = args.task_seed_overrides.get(name, getattr(args, group+'_seeds'))
        seeds = task['seeds'][:count]
        assert count > 0 and len(seeds) == count
        folder = args.out/name
        command = [sys.executable, str(ROOT/'scripts/search_play.py'), '--rollout-policy', str(args.rollout_policy),
                   '--value', str(args.value), '--manifest', str(args.manifest), '--task-id', name,
                   '--executable', str(args.executable), '--resource-dir', str(args.resource_dir),
                   '--seeds', str(count), '--workers', str(args.workers), '--arms', 'fast', '--out', str(folder)]
        contract = dict(model_sha256=model_hash, executable_sha256=binary_hash, manifest_sha256=manifest_hash,
                        seeds=seeds, command=command)
        registration = guard.OUT/(name+'.contract.json')
        if registration.exists():
            assert read(registration) == contract, 'cannot resume a different panel/model/build'
        else:
            assert not (folder/'games.jsonl').exists(), 'unregistered existing games'
            write(registration, contract)
        result_path = guard.OUT/(name+'.result.json')
        if not result_path.exists() or read(result_path)['returncode'] != 0:
            pidfile = guard.OUT/(name+'.process.json')
            assert not pidfile.exists() or not guard.live(read(pidfile)['pid']), 'panel stage already live'
            result = guard.run(command, name, cwd=ROOT)
            assert result['returncode'] == 0, (name, result)
        config = read(folder/'config.json')
        assert config['rollout_policy'] == str(args.rollout_policy) and config['task_id'] == name
        assert config['arms'] == 'fast' and int(config['workers']) == args.workers
        row = summarize(folder, task, seeds, plant_types=True); row['group'] = group; rows.append(row)
        assert sha(args.rollout_policy) == model_hash and sha(args.executable) == binary_hash and sha(args.manifest) == manifest_hash
        print(f"{name}: {row['won']}/{row['games']}", flush=True)
    groups = {}
    for group in ('day', 'heldout', 'night', 'pool'):
        part = [r for r in rows if r['group'] == group]
        if part:
            groups[group] = dict(tasks=len(part), games=sum(r['games'] for r in part), won=sum(r['won'] for r in part),
                                 mean_win_rate=sum(r['win_rate'] for r in part)/len(part))
            from collections import Counter
            counts = Counter()
            unknown=0
            for task in part:
                if task['planted_by_type_total'] is None:unknown+=task['games']
                else:counts.update(task['planted_by_type_total'])
            groups[group]['planted_by_type_total'] = None if unknown else dict(sorted(counts.items(), key=lambda kv:int(kv[0])))
            groups[group]['planted_by_type_per_game'] = None if unknown else {k:v/groups[group]['games']
                                                        for k,v in groups[group]['planted_by_type_total'].items()}
            if unknown:groups[group]['planted_types_unknown_games']=unknown
    panel = dict(complete=not args.tasks, model=str(args.rollout_policy), model_sha256=model_hash,
                 executable_sha256=binary_hash, manifest_sha256=manifest_hash, tasks=rows, groups=groups)
    write(args.out/'panel.json', panel)
    table = ['| 任务 | 区域 | 胜局 | 胜率 |', '|---|---|---:|---:|']
    table += [f"| {r['task_id']} | {r['group']} | {r['won']}/{r['games']} | {r['win_rate']:.1%} |" for r in rows]
    table += [f"| **{name}均值** | | {r['won']}/{r['games']} | **{r['mean_win_rate']:.1%}** |" for name,r in groups.items()]
    (args.out/'panel.md').write_text('\n'.join(table)+'\n'); print('\n'.join(table), flush=True)


PRESSURE_LEVELS = (
    ('day', 7, 'day_l7_eval_d7'),
    ('day', 10, 'day_l10_orig_eval'),
    ('night', 12, 'night_l12_eval_n10'),
    ('night', 18, 'night_l18_eval_n10'),
    ('night', 20, 'night_l20_eval_n10_orig'),
    ('pool', 21, 'pool_l21_eval_p9k'),
    ('pool', 26, 'pool_l26_eval_p9k'),
)
PRESSURE_MULTIPLIERS = (1.0, 1.5, 2.0)


def generate_pressure(day, day_orig, night, pool, seeds=64):
    """v3 压力面板：每区代表关 × 1.0 / 1.5 / 2.0，各取评估段前 N 个种子。

    任务定义直接从已登记的评估任务复制，只改 task_id / seeds / zombie_count_multiplier；
    卡组、地形、playthrough、sun_start、conveyor_start 全不动（AGENTS §8：这是新任务族）。
    """
    tables = {'day': day, 'day_orig': day_orig, 'night': night, 'pool': pool}
    tasks = []
    for zone, level, source_id in PRESSURE_LEVELS:
        table = tables['day_orig' if source_id.startswith('day_l10_orig') else zone]
        source = next(t for t in table['tasks'] if t['task_id'] == source_id)
        for multiplier in PRESSURE_MULTIPLIERS:
            task = copy.deepcopy(source)
            task.update(task_id=f'press_{zone}_l{level}_m{round(multiplier * 100):03d}',
                        seeds=list(source['seeds'][:seeds]),
                        zombie_count_multiplier=multiplier,
                        split='pressure')
            tasks.append(task)
    assert len(tasks) == len(PRESSURE_LEVELS) * len(PRESSURE_MULTIPLIERS)
    for task in tasks:
        assert task['zombie_count_multiplier'] in PRESSURE_MULTIPLIERS
        assert len(task['seeds']) == seeds
    return dict(schema_version=1, purpose='task29 v3 pressure panel: level x multiplier',
                seeds_per_task=seeds, multipliers=list(PRESSURE_MULTIPLIERS), tasks=tasks)


def pressure_ceiling(points):
    """points: {multiplier: (wins, games)} -> 胜率过 50% 的最大倍率（相邻两点线性插值，§13.7）。

    返回 dict(ceiling, censored, below_floor, points)：ceiling 为 None 表示 1.0 都没过 50%；
    censored=True 表示最高档仍 > 50%，真实上限在测点之外。
    """
    measured = sorted((m, w / g) for m, (w, g) in points.items() if g)
    if not measured:
        return dict(ceiling=None, censored=False, below_floor=True, points={})
    if measured[0][1] <= 0.5:
        return dict(ceiling=None, censored=False, below_floor=True, points=dict(measured))
    ceiling = measured[-1][0]
    censored = True
    for (m1, r1), (m2, r2) in zip(measured, measured[1:]):
        if r1 > 0.5 >= r2:
            ceiling = m1 + (r1 - 0.5) / (r1 - r2) * (m2 - m1)
            censored = False
            break
    return dict(ceiling=ceiling, censored=censored, below_floor=False, points=dict(measured))


def evaluate_pressure(args):
    manifest = read(args.pressure_manifest)
    model_hash = sha(args.rollout_policy)
    binary_hash = sha(args.executable)
    manifest_hash = sha(args.pressure_manifest)
    guard.OUT = args.out / 'resources'
    guard.OUT.mkdir(parents=True, exist_ok=True)
    guard.descendants = descendants
    guard.ENV = {**os.environ, 'PYTHONPATH': str(ROOT / 'python') + ':' + str(ROOT / 'scripts'),
                 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MALLOC_ARENA_MAX': '2',
                 'CUDA_VISIBLE_DEVICES': '', 'PVZ_RESOURCE_DIR': str(args.resource_dir),
                 'PVZ_EXECUTABLE': str(args.executable)}
    selected = set(args.pressure_tasks.split(',')) if args.pressure_tasks else None
    rows = []
    for task in manifest['tasks']:
        name = task['task_id']
        if selected is not None and name not in selected:
            continue
        seeds = task['seeds']
        folder = args.out / name
        command = [sys.executable, str(ROOT / 'scripts/search_play.py'),
                   '--rollout-policy', str(args.rollout_policy), '--value', str(args.value),
                   '--manifest', str(args.pressure_manifest), '--task-id', name,
                   '--executable', str(args.executable), '--resource-dir', str(args.resource_dir),
                   '--seeds', str(len(seeds)), '--workers', str(args.workers),
                   '--arms', 'fast', '--out', str(folder)]
        contract = dict(model_sha256=model_hash, executable_sha256=binary_hash,
                        manifest_sha256=manifest_hash, seeds=seeds, command=command)
        registration = guard.OUT / (name + '.contract.json')
        if registration.exists():
            assert read(registration) == contract, 'cannot resume a different pressure panel/model/build'
        else:
            assert not (folder / 'games.jsonl').exists(), 'unregistered existing pressure games'
            write(registration, contract)
        result_path = guard.OUT / (name + '.result.json')
        if not result_path.exists() or read(result_path)['returncode'] != 0:
            pidfile = guard.OUT / (name + '.process.json')
            assert not pidfile.exists() or not guard.live(read(pidfile)['pid']), 'pressure stage already live'
            result = guard.run(command, name, cwd=ROOT)
            assert result['returncode'] == 0, (name, result)
        row = summarize(folder, task, seeds)
        row['multiplier'] = task['zombie_count_multiplier']
        rows.append(row)
        print(f"{name}: {row['won']}/{row['games']}", flush=True)
    by_level = {}
    for row in rows:
        zone, level = row['task_id'].split('_')[1], row['level']
        by_level.setdefault((zone, level), {})[row['multiplier']] = (row['won'], row['games'])
    summary = []
    for (zone, level), points in sorted(by_level.items()):
        ceiling = pressure_ceiling(points)
        summary.append(dict(zone=zone, level=level,
                            win_rates={f'{m:g}': w / g for m, (w, g) in sorted(points.items()) if g},
                            games={f'{m:g}': g for m, (w, g) in sorted(points.items())},
                            **ceiling))
    panel = dict(complete=selected is None, model=str(args.rollout_policy), model_sha256=model_hash,
                 executable_sha256=binary_hash, manifest_sha256=manifest_hash,
                 seeds_per_task=manifest['seeds_per_task'], levels=summary)
    write(args.out / 'pressure.json', panel)
    table = ['| 区 | 关卡 | 1.0 | 1.5 | 2.0 | 压力上限 |', '|---|---:|---:|---:|---:|---:|']
    for item in summary:
        cells = [f"{item['win_rates'].get(f'{m:g}', float('nan')):.1%}"
                 if f'{m:g}' in item['win_rates'] else '—' for m in PRESSURE_MULTIPLIERS]
        ceiling = '—（1.0 未过 50%）' if item['ceiling'] is None else (
            f"≥{item['ceiling']:.2f}" if item['censored'] else f"{item['ceiling']:.2f}")
        table.append(f"| {item['zone']} | {item['level']} | " + ' | '.join(cells) + f" | {ceiling} |")
    (args.out / 'pressure.md').write_text('\n'.join(table) + '\n')
    print('\n'.join(table), flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generate', action='store_true')
    p.add_argument('--version', choices=('v1','v2'), default='v1')
    p.add_argument('--day-manifest', type=Path, default=MAIN/'artifacts/stage4/day_panel.json')
    p.add_argument('--perturb-manifest', type=Path, default=MAIN/'.claude/worktrees/identity/artifacts/stage4/day_panel_identity.json')
    p.add_argument('--night-manifest', type=Path, default=MAIN/'artifacts/stage4/night_panel.json')
    p.add_argument('--day-orig-manifest', type=Path,
                   default=MAIN/'.claude/worktrees/conveyor/artifacts/stage4/day_panel.json')
    p.add_argument('--pool-manifest', type=Path, default=MAIN/'artifacts/stage4/pool_panel.json')
    p.add_argument('--generate-pressure', action='store_true')
    p.add_argument('--pressure', action='store_true', help='run the v3 pressure panel')
    p.add_argument('--pressure-manifest', type=Path, default=ROOT/'artifacts/task29/pressure_manifest.json')
    p.add_argument('--pressure-seeds', type=int, default=64)
    p.add_argument('--pressure-tasks', default='', help='optional comma-separated smoke subset')
    p.add_argument('--manifest', type=Path, default=ROOT/'artifacts/task23/unified_manifest.json')
    p.add_argument('--rollout-policy', type=Path)
    p.add_argument('--value', type=Path, default=MAIN/'artifacts/value_net/v3/model.pt')
    p.add_argument('--executable', type=Path, default=ROOT/'build/pvz-portable')
    p.add_argument('--resource-dir', type=Path, default=Path('/home/newbiexvwu/.cache/pvz-research-resources'))
    p.add_argument('--out', type=Path, default=ROOT/'artifacts/task23/panel')
    p.add_argument('--day-seeds', type=int, default=128)
    p.add_argument('--heldout-seeds', type=int, default=64)
    p.add_argument('--night-seeds', type=int, default=128)
    p.add_argument('--pool-seeds', type=int, default=64)
    p.add_argument('--task-seeds', action='append', default=[], metavar='TASK=N',
                   help='override any individual task budget; repeat for multiple tasks')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--tasks', default='', help='optional comma-separated smoke subset; marks output incomplete')
    args = p.parse_args()
    if not 1 <= args.workers <= 4: p.error('--workers must be 1..4')
    for name in ('manifest', 'rollout_policy', 'value', 'executable', 'resource_dir', 'out',
                 'day_manifest', 'perturb_manifest', 'night_manifest', 'day_orig_manifest',
                 'pool_manifest', 'pressure_manifest'):
        value = getattr(args, name)
        if value is not None and not value.is_absolute(): p.error('all paths must be absolute: '+name)
    if args.generate:
        document = generate(read(args.day_manifest), read(args.perturb_manifest), read(args.night_manifest),
                            args.version, read(args.pool_manifest) if args.version == 'v2' else None)
        write(args.manifest, document)
        print(f'wrote {args.manifest}: {len(document["tasks"])} tasks ({args.version})')
    if args.generate_pressure:
        assert 1 <= args.pressure_seeds <= 1024
        pressure = generate_pressure(read(args.day_manifest), read(args.day_orig_manifest),
                                     read(args.night_manifest), read(args.pool_manifest),
                                     args.pressure_seeds)
        write(args.pressure_manifest, pressure)
        print(f'wrote {args.pressure_manifest}: {len(pressure["tasks"])} tasks')
        if not args.pressure:
            return
    if args.pressure:
        if not args.rollout_policy:
            p.error('--pressure needs --rollout-policy')
        evaluate_pressure(args)
        return
    if args.rollout_policy:
        tasks = {task['task_id']:task for task in read(args.manifest)['tasks']}
        selected = set(args.tasks.split(',')) if args.tasks else set(tasks)
        args.task_seed_overrides = {}
        for item in args.task_seeds:
            try:
                name, value = item.rsplit('=',1); count = int(value)
            except ValueError:
                p.error('--task-seeds requires TASK=N')
            if name not in tasks or name not in selected or name in args.task_seed_overrides:
                p.error('--task-seeds task must be selected and specified once: '+name)
            if not 1 <= count <= len(tasks[name]['seeds']):
                p.error('--task-seeds exceeds the task seed range: '+name)
            args.task_seed_overrides[name] = count
        evaluate(args)
    elif not args.generate: p.error('pass --generate or --rollout-policy')


if __name__ == '__main__':
    main()
