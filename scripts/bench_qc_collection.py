"""task51: frozen, alternating production collection commands with real frame archives.

This tool never starts the curriculum controller or writes its state/offsets. Commands
come from each historical controller's command_for and use isolated output directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
PYTHON = Path('/home/newbiexvwu/.venvs/ml/bin/python3')
TASKS = ('day_l7_train_d7_m150', 'night_l14_train_n10_m150',
         'pool_l24_train_p9k_m150', 'day_l10_orig_train_m150')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_rows(rows: list[dict], seeds: list[int]) -> None:
    actual = [row['seed'] for row in rows]
    if len(actual) != len(set(actual)) or len(seeds) != len(set(seeds)):
        raise ValueError('duplicate game or planned seeds')
    if len(rows) != len(seeds) or set(actual) != set(seeds):
        raise ValueError('games do not exactly cover the frozen seeds')
    if any(type(row['won']) is not bool or row['arm'] != 'fast_search' for row in rows):
        raise ValueError('invalid game outcome or arm')


def audit_archive(archive: Path, row: dict, task: dict, reader=None) -> dict:
    meta_file=archive/'meta.json';frame_file=archive/'frames.npz'
    meta=json.loads(meta_file.read_text())
    if meta['task']['seed']!=row['seed'] or any(meta['outcome'][k]!=row[k] for k in ('won','final_wave')):
        raise ValueError(f'archive outcome/seed does not match game: {archive}')
    if meta['context']['level']!=task['level'] or meta['context']['multiplier']!=task['zombie_count_multiplier']:
        raise ValueError(f'archive task context does not match frozen task: {archive}')
    if reader is None:
        sys.path.insert(0,str(ROOT/'python'))
        from pvz_frame_store import read_frames
        reader=read_frames
    frames=reader(frame_file)
    # Production archives contain every decision frame plus the final observation.
    if (len(frames)!=row['decisions']+1 or
            sum('action' in f for f in frames)!=row['decisions'] or
            sum('label' in f for f in frames)!=row['decisions']):
        raise ValueError(f'archive is missing decision frames/actions/labels: {archive}')
    if task['deck'] and [p[1] for p in frames[0]['packets']]!=task['deck']:
        raise ValueError(f'archive initial deck differs from frozen task: {archive}')
    if frames[-1]['wave']!=row['final_wave']:
        raise ValueError(f'archive final observation differs from game: {archive}')
    return dict(seed=row['seed'],frames=len(frames),meta_sha256=sha(meta_file),
                frames_sha256=sha(frame_file),archive=str(archive))


def tree_sha(source: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted([*(source / 'scripts').rglob('*.py'), *(source / 'python').rglob('*.py')]):
        digest.update(str(path.relative_to(source)).encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def command_from_source(source: Path, values: dict, task: str, offset: int, label: str) -> list[str]:
    # Separate interpreters prevent old/new helper modules from contaminating each other.
    code = '''import json,sys
from pathlib import Path
from types import SimpleNamespace
d=json.load(sys.stdin)
sys.path.insert(0,str(Path(d['source'])/'scripts'))
import curriculum_queue as c
a=SimpleNamespace(**{k:Path(v) if k!='workers' else v for k,v in d['values'].items()})
print(json.dumps(c.command_for(a,Path(d['manifest']),d['task'],d['offset'],8,d['label'])))
'''
    data = dict(source=str(source), values=values, task=task, offset=offset, label=label,
                manifest=values['manifest'])
    output = subprocess.check_output([str(PYTHON), '-c', code], input=json.dumps(data), text=True,
                                     cwd=ROOT)
    return json.loads(output)


def plan(out: Path, current_executable: Path | None = None, reference_plan: Path | None = None) -> dict:
    target = out / 'collection-plan.json'
    if target.exists():
        existing = json.loads(target.read_text())
        if current_executable is not None or reference_plan is not None:
            raise ValueError('plan already frozen; run/report without planning overrides')
        return existing
    sources = json.loads((out / 'sources.json').read_text())
    panel = Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/artifacts/stage4/multiplier_panel.json')
    reference = None
    if reference_plan is not None:
        reference = json.loads(reference_plan.read_text())
        require_complete_design(reference)
        require_inputs(reference)
        panel_data = json.loads(Path(reference['manifest']).read_text())
    else:
        panel_data = json.loads(panel.read_text())
    tasks = []
    for name in TASKS:
        task = next(t for t in panel_data['tasks'] if t['task_id'] == name)
        seeds = task['seeds'] if reference is not None else task['seeds'][64:80]
        if len(seeds) != 16 or len(set(seeds)) != 16:
            raise ValueError(f'{name}: insufficient distinct seeds')
        tasks.append({**task, 'seeds': seeds})
    manifest = out / 'collection-manifest.json'
    write_json(manifest, dict(tasks=tasks))
    controls = dict(student=MAIN / 'artifacts/student/dw4i_s1/model.pt',
                    static=MAIN / 'artifacts/hard/value_branch1/model.pt',
                    old_value=MAIN / 'artifacts/value_net/v3/model.pt',
                    cq_value=MAIN / 'artifacts/value_net/v4/model.pt',
                    old_engine=ROOT / 'build-old/pvz-portable',
                    cq_engine=current_executable or Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/deploy2/build/pvz-portable'))
    fingerprints = {str(p): sha(p) for p in controls.values()}
    fingerprints[str(manifest)] = sha(manifest)
    if reference_plan is not None:
        fingerprints[str(reference_plan)] = sha(reference_plan)
        fingerprints[reference['manifest']] = sha(Path(reference['manifest']))
    source_shas = {arm: tree_sha(Path(data['source'])) for arm, data in sources.items()}
    stages = []
    for index, task in enumerate(tasks):
        sequence = ('old', 'cq', 'cq', 'old') if index % 2 == 0 else ('cq', 'old', 'old', 'cq')
        blocks = dict(old=0, cq=0)
        for arm in sequence:
            block = blocks[arm]; blocks[arm] += 1
            label = f'q51_{index}_{arm}_{block}'
            values = dict(python=str(PYTHON), rollout_policy=str(controls['student']),
                          static_value=str(controls['static']), value=str(controls[arm + '_value']),
                          executable=str(controls[arm + '_engine']), workers=2,
                          resource_dir='/home/newbiexvwu/.cache/pvz-research-resources',
                          frames=str(out / 'frames' / arm), out=str(out / 'runs'),
                          manifest=str(manifest))
            command = command_from_source(Path(sources[arm]['source']), values, task['task_id'],
                                          block * 8, label)
            assert command[command.index('--workers') + 1] == '2'
            assert command[command.index('--dagger-beta') + 1] == '0'
            stages.append(dict(id=label, arm=arm, task_id=task['task_id'], block=block,
                               seeds=task['seeds'][block * 8:block * 8 + 8], command=command,
                               frames=values['frames'], label=label,
                               output=command[command.index('--out') + 1]))
    result = dict(protocol='task51 formal collection wall time including frame archives',
                  workers=2, games_per_arm_per_task=16, manifest=str(manifest),
                  original_panel_sha256=reference['original_panel_sha256'] if reference else sha(panel),
                  original_seed_offset=64, reference_plan=str(reference_plan) if reference_plan else None,
                  sources=sources, source_python_sha256=source_shas,
                  input_sha256=fingerprints, stages=stages)
    write_json(target, result)
    return result


def sample(root_pid: int) -> dict:
    table = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / 'stat').read_text(); fields = text.rsplit(')', 1)[1].split()
            table[int(entry.name)] = (int(fields[1]), int(fields[21]) * os.sysconf('SC_PAGE_SIZE'),
                                     'pvz-portable' in text.split(')', 1)[0])
        except (OSError, ValueError, IndexError):
            continue
    children = {root_pid}
    while True:
        added = {pid for pid, (parent, _, _) in table.items() if parent in children} - children
        if not added:
            break
        children.update(added)
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    return dict(timestamp=time.time(), mem_available_bytes=available, load1=os.getloadavg()[0],
                tree_rss_bytes=sum(table[p][1] for p in children if p in table),
                engine_processes=sum(table[p][2] for p in children if p in table))


def require_inputs(document: dict) -> None:
    for filename, expected in document['input_sha256'].items():
        if sha(Path(filename)) != expected:
            raise ValueError(f'frozen input changed: {filename}')
    for arm, expected in document['source_python_sha256'].items():
        if tree_sha(Path(document['sources'][arm]['source'])) != expected:
            raise ValueError(f'frozen Python changed: {arm}')


def require_complete_design(document: dict) -> None:
    if document['workers'] != 2 or document['games_per_arm_per_task'] != 16:
        raise ValueError('controlled design requires two workers and 16 games per arm/task')
    stages = document['stages']
    if len(stages) != 16 or len({s['id'] for s in stages}) != 16:
        raise ValueError('controlled design requires 16 distinct stages')
    for task in TASKS:
        paired = {}
        for arm in ('old', 'cq'):
            group = [s for s in stages if s['task_id'] == task and s['arm'] == arm]
            if len(group) != 2 or {s['block'] for s in group} != {0, 1}:
                raise ValueError('controlled design requires both blocks for every arm/task')
            seeds = [seed for s in group for seed in s['seeds']]
            if any(len(s['seeds']) != 8 for s in group) or len(set(seeds)) != 16:
                raise ValueError('controlled design requires 16 distinct seeds per arm/task')
            paired[arm] = {s['block']: s['seeds'] for s in group}
        if paired['old'] != paired['cq']:
            raise ValueError('controlled arms must use identical seeds in each paired block')


def stage_evidence(timing: dict, rows: list[dict], timing_path: Path) -> dict:
    if timing['wall_seconds'] <= 0 or timing['games'] != len(rows):
        raise ValueError('invalid timing duration or game count')
    samples = timing['samples']
    if any(s['engine_processes'] > 2 for s in samples):
        raise ValueError('controlled stage exceeded two game processes')
    bounds = lambda key: (dict(min=min(s[key] for s in samples), max=max(s[key] for s in samples))
                          if samples else None)
    return dict(stage=timing['stage'], wall_seconds=timing['wall_seconds'],
                cpu_seconds=timing['cpu_seconds'], wins=sum(r['won'] for r in rows),
                samples=len(samples), mem_available_bytes=bounds('mem_available_bytes'),
                tree_rss_bytes=bounds('tree_rss_bytes'), load1=bounds('load1'),
                max_sampled_engine_processes=max((s['engine_processes'] for s in samples), default=None),
                timing_sha256=sha(timing_path))


def collect(out: Path, document: dict) -> None:
    for stage in document['stages']:
        require_inputs(document)
        folder = Path(stage['output']); result_path = out / 'timing' / (stage['id'] + '.json')
        if result_path.exists():
            result = json.loads(result_path.read_text())
            rows = [json.loads(s) for s in (folder / 'games.jsonl').read_text().splitlines()]
            validate_rows(rows, stage['seeds'])
            if not result['complete'] or result['returncode'] != 0:
                raise ValueError('prior stage is incomplete')
            continue
        if folder.exists():
            raise ValueError(f'untimed or interrupted output requires inspection, not reuse: {folder}')
        log = out / 'logs' / (stage['id'] + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
        cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started_wall = time.time(); started = time.perf_counter(); samples = []
        env = {**os.environ, 'PVZ_NO_POPUP': '1', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
               'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}
        print('START', stage['id'], stage['task_id'], stage['seeds'], flush=True)
        with log.open('w') as handle:
            process = subprocess.Popen(stage['command'], cwd=ROOT, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            write_json(out / 'active.json', dict(runner_pid=os.getpid(), child_pid=process.pid,
                                                stage=stage['id'], started_at=started_wall))
            while process.poll() is None:
                current = sample(process.pid); samples.append(current)
                if current['engine_processes'] > 2:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    raise RuntimeError('more than two owned game engines; inspect owned process tree')
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            code = process.wait()
        seconds = time.perf_counter() - started
        cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = dict(stage=stage['id'], task_id=stage['task_id'], arm=stage['arm'],
                      started_at=started_wall, ended_at=time.time(), wall_seconds=seconds,
                      cpu_seconds=(cpu_after.ru_utime + cpu_after.ru_stime - cpu_before.ru_utime - cpu_before.ru_stime),
                      returncode=code, complete=False, samples=samples)
        rows = [json.loads(s) for s in (folder / 'games.jsonl').read_text().splitlines()] if (folder / 'games.jsonl').exists() else []
        if code == 0:
            validate_rows(rows, stage['seeds'])
            for row in rows:
                archive = Path(stage['frames']) / f'seed_{row["seed"]}' / (stage['label'] + '_fast_search')
                if not (archive / 'meta.json').is_file():
                    raise ValueError(f'formal frame archive missing: {archive}')
            result['complete'] = True
        result['games'] = len(rows); result['rows_sha256'] = sha(folder / 'games.jsonl') if rows else None
        write_json(result_path, result)
        if not result['complete']:
            raise RuntimeError(f'{stage["id"]} failed with exit {code}; no success timing accepted')
        print('DONE', stage['id'], f'{seconds:.3f}s', flush=True)
    write_json(out / 'active.json', dict(runner_pid=os.getpid(), complete=True, ended_at=time.time()))


def report(out: Path, document: dict) -> dict:
    require_complete_design(document)
    require_inputs(document)
    groups = {};archives=[]
    frozen={t['task_id']:t for t in json.loads(Path(document['manifest']).read_text())['tasks']}
    for task in TASKS:
        groups[task] = {}
        for arm in ('old', 'cq'):
            stages = [s for s in document['stages'] if s['arm'] == arm and s['task_id'] == task]
            total = 0.; rows = []; evidence = []
            for stage in stages:
                timing_path = out / 'timing' / (stage['id'] + '.json')
                timing = json.loads(timing_path.read_text())
                file = Path(stage['output']) / 'games.jsonl'
                if not timing['complete'] or timing['returncode'] != 0 or sha(file) != timing['rows_sha256']:
                    raise ValueError(f'incomplete or changed stage: {stage["id"]}')
                part = [json.loads(s) for s in file.read_text().splitlines()]
                validate_rows(part, stage['seeds'])
                if any(timing[k] != stage[k] for k in ('task_id', 'arm')) or timing['stage'] != stage['id']:
                    raise ValueError('timing identity differs from frozen stage')
                evidence.append(dict(block=stage['block'], **stage_evidence(timing, part, timing_path)))
                for row in part:
                    archive=Path(stage['frames'])/f'seed_{row["seed"]}'/(stage['label']+'_fast_search')
                    archives.append(audit_archive(archive,row,frozen[task]))
                rows.extend(part); total += timing['wall_seconds']
            seeds = [seed for s in stages for seed in s['seeds']]
            validate_rows(rows, seeds)
            groups[task][arm] = dict(games=len(rows), wall_seconds=total, games_per_hour=len(rows) * 3600 / total,
                                     games_per_worker_hour=len(rows) * 3600 / (2 * total), rows=rows,
                                     stages=evidence, wins=sum(r['won'] for r in rows))
        old, new = groups[task]['old'], groups[task]['cq']
        a={r['seed']:r for r in old['rows']}; b={r['seed']:r for r in new['rows']}
        groups[task]['trajectory_differences'] = [s for s in sorted(a) if any(a[s][k] != b[s][k] for k in ('won','final_wave','decisions'))]
        groups[task]['throughput_ratio_cq_over_old'] = old['wall_seconds'] / new['wall_seconds']
        blocks = {arm: {s['block']: s for s in groups[task][arm]['stages']} for arm in ('old', 'cq')}
        groups[task]['paired_block_ratios'] = [dict(block=b,
            throughput_ratio_cq_over_old=blocks['old'][b]['wall_seconds']/blocks['cq'][b]['wall_seconds'])
            for b in (0, 1)]
    overall = {arm:dict(games=sum(g[arm]['games'] for g in groups.values()),
                       wall_seconds=sum(g[arm]['wall_seconds'] for g in groups.values())) for arm in ('old','cq')}
    for d in overall.values():d['games_per_hour']=d['games']*3600/d['wall_seconds']
    ratio=overall['cq']['games_per_hour']/overall['old']['games_per_hour']
    result=dict(complete=True,workers=2,results=groups,overall=overall,archive_audit=archives,
                throughput_ratio_cq_over_old=ratio,
                classification='>=2.0' if ratio>=2 else '<1.5' if ratio<1.5 else '1.5-2.0',
                note='Collection wall time only, including archives; cycle/pack/training results are separate.')
    write_json(out/'collection-results.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('plan','run','report'))
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/task51')
    parser.add_argument('--current-executable',type=Path,help='plan only: engine for the comparison arm')
    parser.add_argument('--reference-plan',type=Path,help='plan only: reuse the exact frozen task/seed set')
    args=parser.parse_args()
    if not all(p.is_absolute() for p in (args.out,args.current_executable,args.reference_plan) if p is not None):
        parser.error('paths must be absolute')
    if args.phase!='plan' and (args.current_executable or args.reference_plan):
        parser.error('planning overrides are only accepted by plan')
    document=plan(args.out,args.current_executable,args.reference_plan)
    if args.phase=='run':collect(args.out,document)
    if args.phase=='report':
        result=report(args.out,document)
        print(json.dumps({k:v for k,v in result.items() if k not in ('results','archive_audit')},indent=2))
    if args.phase=='plan':print(json.dumps(dict(stages=len(document['stages']),workers=document['workers'])))


if __name__=='__main__':main()
