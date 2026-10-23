"""Cold-pack the frozen task51 cohorts with their original controller's pack_command.

Measures selected-cohort overhead, not the full historical production corpus. Each
task has an isolated symlink view and fresh cache; source archives stay unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from bench_qc_collection import (PYTHON, ROOT, TASKS, require_complete_design, require_inputs,
                                 sample, sha, write_json)


def validate_sources(actual: list[str], expected: list[str]) -> None:
    if len(actual) != len(set(actual)) or set(actual) != set(expected) or len(actual) != len(expected):
        raise ValueError('pack must cover exactly the frozen archive paths')


def require_cold(view: Path) -> None:
    if (view / '.feature_cache').exists() or (view / 'packs').exists():
        raise ValueError('untimed cache/pack exists; inspect interrupted cold run instead of reusing it')


def make_plan(out: Path) -> dict:
    target = out / 'cohort-pack/plan.json'
    if target.exists():
        return json.loads(target.read_text())
    collection_file = out / 'collection-results.json'
    collection = json.loads(collection_file.read_text())
    if not collection['complete']:
        raise ValueError('collection experiment must finish before timing pack stages')
    document = json.loads((out / 'collection-plan.json').read_text())
    require_complete_design(document)
    require_inputs(document)
    source = Path(document['sources']['cq']['source'])
    recipe = source / 'recipes/dw4i.json'
    # Separate processes isolate each historical controller's imports.
    code = '''import json,sys
from pathlib import Path
from types import SimpleNamespace
d=json.load(sys.stdin)
sys.path.insert(0,str(Path(d['source'])/'scripts'))
import curriculum_queue as c
if d['arm']=='old':
 assert not hasattr(c,'pack_command') and not hasattr(c,'write_pack')
 print('null')
else:
 a=SimpleNamespace(python=Path(d['python']),frames=Path(d['view']),recipe=Path(d['recipe']))
 print(json.dumps(c.pack_command(a,d['labels'])))
'''
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    def command(arm, view, labels):
        data = dict(source=document['sources'][arm]['source'], arm=arm, python=str(PYTHON),
                    view=str(view), recipe=str(recipe), labels=labels)
        return json.loads(subprocess.check_output([str(PYTHON), '-c', code], input=json.dumps(data),
                                                 text=True, cwd=ROOT, env=env))
    if document['sources']['old']['commit'] != '1b89bfa92554602ff83a44bd969de7ac532894ba':
        raise ValueError('no-pack old baseline is only established for frozen 1b89bfa9')
    assert command('old', out, []) is None
    archive_audits = {x['archive']: x for x in collection['archive_audit']}
    stages = []
    inputs = {str(collection_file): sha(collection_file), str(out / 'collection-plan.json'): sha(out / 'collection-plan.json'),
              str(recipe): sha(recipe)}
    for task in TASKS:
        view = out / 'cohort-pack' / task / 'frames'
        require_cold(view)
        selected = [s for s in document['stages'] if s['task_id'] == task and s['arm'] == 'cq']
        labels = sorted(s['label'] + '_fast_search' for s in selected)
        paths = []; frame_counts = {}
        for stage in selected:
            for seed in stage['seeds']:
                archive = Path(stage['frames']) / f'seed_{seed}' / (stage['label'] + '_fast_search')
                audited = archive_audits[str(archive)]
                for name, key in (('meta.json', 'meta_sha256'), ('frames.npz', 'frames_sha256')):
                    file = archive / name
                    if sha(file) != audited[key]:
                        raise ValueError('collection archive changed after audit')
                    inputs[str(file)] = audited[key]
                link = view / f'seed_{seed}' / (stage['label'] + '_fast_search')
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink():
                    if link.resolve() != archive.resolve():
                        raise ValueError('archive view points at different input')
                else:
                    link.symlink_to(archive, target_is_directory=True)
                paths.append(str((link / 'frames.npz').relative_to(view)))
                frame_counts[paths[-1]] = audited['frames']
        validate_sources([str(p.relative_to(view)) for p in view.glob('*/*/frames.npz')], paths)
        stages.append(dict(task_id=task, view=str(view), labels=labels, expected_sources=sorted(paths),
                           expected_frame_counts=frame_counts,
                           command=command('cq', view, labels)))
    result = dict(scope='selected 16-game cohorts, cold cache, no historical corpus; collection+pack stage sums',
                  old_pack_seconds=0, training='No training stage in the frozen old/cq controller; none launched.',
                  input_sha256=inputs, stages=stages)
    write_json(target, result)
    return result


def audit_pack(file: Path, stage: dict) -> dict:
    import numpy as np
    if not file.resolve().is_relative_to(Path(stage['view']).resolve() / 'packs'):
        raise ValueError('pack output escaped isolated view')
    with np.load(file, allow_pickle=False) as stored:
        paths = stored['source'].tolist()
        validate_sources(paths, stage['expected_sources'])
        meta = json.loads(str(stored['meta']))
        if sorted(meta['labels']) != stage['labels']:
            raise ValueError('pack labels differ from frozen cohort')
        count = int(stored['rows'])
        ids = stored['source_archive']
        source_rows = stored['source_row']
        if count <= 0 or len(ids) != count or len(source_rows) != count or set(ids.tolist()) != set(range(16)):
            raise ValueError('packed rows do not cover all 16 archives')
        for index, path in enumerate(paths):
            archive_meta = json.loads((Path(stage['view']) / path).with_name('meta.json').read_text())
            seed = int(Path(path).parts[0].removeprefix('seed_'))
            selected = source_rows[ids == index]
            if (archive_meta['task']['seed'] != seed or np.any(selected < 0) or
                    np.any(selected >= stage['expected_frame_counts'][path])):
                raise ValueError('invalid packed provenance')
    return dict(path=str(file), sha256=sha(file), bytes=file.stat().st_size, rows=count,
                source_archives=len(paths), options=meta)


def run(out: Path, plan: dict) -> None:
    for stage in plan['stages']:
        require_inputs(dict(input_sha256=plan['input_sha256'], source_python_sha256={}))
        require_inputs(json.loads((out / 'collection-plan.json').read_text()))
        folder = Path(stage['view']).parent
        timing_file = folder / 'timing.json'
        if timing_file.exists():
            prior = json.loads(timing_file.read_text())
            if not prior['complete'] or sha(Path(prior['pack']['path'])) != prior['pack']['sha256']:
                raise ValueError('prior pack attempt is incomplete or changed')
            continue
        require_cold(Path(stage['view']))
        samples = []
        cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.perf_counter()
        env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
               'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}
        print('START', stage['task_id'], flush=True)
        with (folder / 'pack.log').open('w') as log:
            process = subprocess.Popen(stage['command'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_json(out / 'cohort-pack/active.json', dict(pid=process.pid, task_id=stage['task_id']))
            while process.poll() is None:
                samples.append(sample(process.pid))
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            code = process.wait()
        seconds = time.perf_counter() - started
        cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = dict(complete=False, returncode=code, wall_seconds=seconds, samples=samples,
                      cpu_seconds=cpu_after.ru_utime+cpu_after.ru_stime-cpu_before.ru_utime-cpu_before.ru_stime)
        if any(s['engine_processes'] for s in samples):
            raise ValueError('pack-only phase unexpectedly started a game engine')
        if code == 0:
            messages = []
            for line in (folder / 'pack.log').read_text().splitlines():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict) and 'pack' in message:
                    messages.append(message)
            if len(messages) != 1:
                raise ValueError('missing or ambiguous pack completion record')
            result['pack'] = audit_pack(Path(messages[0]['pack']), stage)
            result['complete'] = True
        write_json(timing_file, result)
        if not result['complete']:
            raise RuntimeError(f'cold pack failed: {stage["task_id"]}, rc={code}')
        print('DONE', stage['task_id'], f'{seconds:.3f}s', flush=True)
    write_json(out / 'cohort-pack/active.json', dict(complete=True, ended_at=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('plan', 'run'))
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/task51')
    args = parser.parse_args()
    if not args.out.is_absolute():
        parser.error('--out must be absolute')
    plan = make_plan(args.out)
    if args.phase == 'run':
        run(args.out, plan)
    else:
        print(json.dumps(dict(stages=len(plan['stages']), scope=plan['scope'])))


if __name__ == '__main__':
    main()
