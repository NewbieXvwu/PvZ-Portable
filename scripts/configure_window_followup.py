"""Install seed1-first scheduling, the next-window probe, and lower-priority task163 jobs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from gpu_window_runner import write_json


def checkout(state, revision):
    revision = subprocess.check_output(['git','-C',str(ROOT),'rev-parse',revision+'^{commit}'], text=True).strip()
    path = state/'code'/revision
    if not path.exists():
        subprocess.run(['git','-C',str(ROOT),'worktree','add','--detach',str(path),revision], check=True)
    return revision, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--task163-jobs', type=Path, required=True)
    parser.add_argument('--task163-revision', required=True)
    args = parser.parse_args()
    state = args.state.resolve()
    revision, code = checkout(state, args.revision)
    external_revision, external_code = checkout(state, args.task163_revision)
    manifest = json.loads((code/'configs/decision_r42.json').read_text())
    decision_ids = [f'r42_{arm["name"]}_s{seed}' for seed in (1,2) for arm in manifest['arms']]
    seed1 = [name for name in decision_ids if name.endswith('_s1')]
    probe = state/'decision-data/throughput'
    history = state/'queue-history'/revision
    (history/'jobs').mkdir(parents=True, exist_ok=True)
    (history/'recipes').mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in ['decision_split', 'decision_tokens', *decision_ids]:
        path = state/'queue'/(name+'.json')
        job = json.loads(path.read_text())
        out = Path(job['out'])
        if (out/'resume.pt').exists() or (out/'model.pt').exists():
            raise RuntimeError(f'{name} already has optimizer/model progress; do not silently change its contract')
        old_code = job['code_root']
        shutil.copyfile(path, history/'jobs'/path.name)
        for field in ('command','evaluate'):
            if field in job:
                job[field] = [arg.replace(old_code+'/', str(code)+'/') for arg in job[field]]
        job.update(revision=revision, code_root=str(code), enabled=True)
        job.pop('pending_measured_budget', None)
        if name == 'decision_tokens':
            if '--budget' not in job['command']:
                job['command'] += ['--budget',str(probe/'ready.json')]
            job['depends_on'] = list(dict.fromkeys([*job.get('depends_on',[]),str(probe/'ready.json')]))
        if name.startswith('r42_'):
            arm_name = name[4:-3]
            arm = next(arm for arm in manifest['arms'] if arm['name']==arm_name)
            job['priority'] = arm['priority'] + (100 if job['seed']==2 else 0)
            job['after_jobs'] = seed1 if job['seed']==2 else []
            job['depends_on'] = list(dict.fromkeys([*job['depends_on'], str(probe/'ready.json')]))
            if '--training-budget' not in job['command']:
                job['command'] += ['--training-budget',str(probe/'ready.json'),'--budget-arm',arm_name]
                job['evaluate'] += ['--training-budget',str(probe/'ready.json')]
            recipe = Path(job['recipe'])
            shutil.copyfile(recipe, history/'recipes'/recipe.name)
            document = json.loads(recipe.read_text())
            document['provenance']['source_revision'] = revision
            write_json(recipe, document)
        jobs.append(job)
    base_recipe = json.loads(Path(jobs[2]['recipe']).read_text())['provenance']['source_path']
    jobs.append(dict(id='decision_probe', kind='prepare', priority=1, out=str(probe), recipe=base_recipe,
        revision=revision, code_root=str(code), host_gib=5, gpu_gib=4, slice_seconds=900,
        depends_on=[str(state/'decision-data/split/ready.json')],
        command=[sys.executable,str(code/'scripts/bench_decision_window.py'),'--state',str(state),
                 '--out',str(probe),'--revision',revision]))
    jobs.append(dict(id='decision_cleanup', kind='prepare', priority=300, out=str(state/'decision-data/cleanup'),
        recipe=base_recipe, revision=revision, code_root=str(code), host_gib=.25, gpu_gib=0, slice_seconds=900,
        after_jobs=decision_ids, command=[sys.executable,str(code/'scripts/cleanup_decision_cache.py'),
            '--state',str(state),'--out',str(state/'decision-data/cleanup')]))
    external = []
    for path in sorted(args.task163_jobs.glob('*.json')):
        job = json.loads(path.read_text())
        recipe = json.loads(Path(job['recipe']).read_text())
        if recipe['seed'] not in (1,2):
            continue
        target = state/'task163-recipes'/Path(job['recipe']).name
        target.parent.mkdir(parents=True, exist_ok=True)
        recipe['provenance']['notes'] += ' training_code_revision='+external_revision
        write_json(target, recipe)
        old_root = str(Path(job['command'][1]).parents[1])
        for field in ('command','evaluate'):
            job[field] = [str(target) if arg==job['recipe'] else arg.replace(old_root+'/',str(external_code)+'/')
                          for arg in job[field]]
        job.update(recipe=str(target), revision=external_revision, code_root=str(external_code),
                   seed=recipe['seed'], arch=recipe.get('network',{}).get('architecture','cnn'),
                   priority=1000 + (100 if recipe['seed']==2 else 0), after_jobs=[*decision_ids,'decision_cleanup'])
        external.append(job); jobs.append(job)
    for job in jobs:
        write_json(state/'queue'/(job['id']+'.json'), job)
    study = json.loads((state/'decision-study.json').read_text())
    study.update(revision=revision, code_root=str(code), base_recipe=base_recipe,
        scheduling='all eight decision seed1 complete before any decision seed2; task163 after all sixteen decisions',
        budget_probe=str(probe), task163_revision=external_revision, task163_jobs=[j['id'] for j in external],
        jobs=[{k:j[k] for k in ('id','priority','revision','out')} for j in jobs])
    write_json(state/'decision-study.json', study)
    print(json.dumps(dict(decisions=16, task163=len(external), revision=revision, task163_revision=external_revision)))


if __name__ == '__main__':
    main()
