"""Task20's isolated original-conveyor baseline, two-cohort DAgger pilot and fixed evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
import dagger_experiment_runner as guard
from run_identity_validation import descendants, trainers
from pvz_recipe import materialize_recipe

OUT = ROOT / 'artifacts/task20'
MAIN = Path.home() / 'PvZ-search'
IDENTITY = MAIN / 'artifacts/archive/identity/task19'
guard.OUT = OUT
guard.descendants = descendants
guard.ENV = {**os.environ, 'PYTHONPATH': str(ROOT/'python')+':'+str(ROOT/'scripts'),
    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MALLOC_ARENA_MAX': '2',
    'PVZ_RESOURCE_DIR': str(Path.home()/'.cache/pvz-research-resources'),
    'PVZ_EXECUTABLE': str(ROOT/'build/pvz-portable')}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    path = OUT/'study_plan.json'
    if path.exists():
        return read(path)
    original = read(ROOT/'artifacts/stage4/day_panel.json')
    reference = read(MAIN/'artifacts/stage4/day_panel.json')
    tasks = [t for t in original['tasks'] if t['task_id'] in ('day_l10_orig_train', 'day_l10_orig_eval')]
    tasks += [next(t for t in reference['tasks'] if t['task_id']==f'day_l{level}_eval_d7') for level in (7,8,9)]
    assert len(tasks)==5
    manifest = OUT/'study_manifest.json'
    write(manifest, dict(schema_version=1, tasks=tasks))
    inherited = read(ROOT/'MEMORY/docs/task19-validation-amended-plan.json')
    labels = inherited['labels'] + ['d10orig1_fast_search','d10orig2_fast_search']
    recipe = materialize_recipe(ROOT/'recipes/identity_dw3_s1.json', OUT/'training.recipe.json',
                                data=OUT/'frames_v2', labels=labels, out=OUT/'identity_dagger_s1',
                                cache_dir=OUT/'cache')
    command = [guard.PYTHON, str(ROOT/'scripts/train_rollout_policy.py'), '--recipe', str(recipe)]
    names = ['scripts/run_conveyor_study.py','scripts/dagger_experiment_runner.py',
        'scripts/run_identity_validation.py','scripts/train_rollout_policy.py',
        'python/pvz_recipe.py','recipes/identity_dw3_s1.json',
        'python/pvz_rollout_policy.py','python/pvz_frame_store.py','python/pvz_student_observation.py',
        'python/pvz_value_net.py','python/pvz_env.py','python/pvz_event_env.py',
        'scripts/search_play.py','scripts/render_episode.py','scripts/episode_query.py',
        'scripts/pvz_tasks.py','scripts/pvz_constants.py','src/EnvironmentRollout.cpp',
        'src/EnvironmentRollout.h','src/LawnApp.cpp','src/LawnApp.h','src/main.cpp']
    models = dict(v34=str(MAIN/'artifacts/student/v34/model.pt'),
        identity=str(IDENTITY/'identity_dw3_s1/model.pt'),
        value=str(MAIN/'artifacts/value_net/v3/model.pt'),
        static=str(MAIN/'artifacts/hard/value_branch1/model.pt'))
    plan = dict(registered_at=datetime.now(timezone.utc).isoformat(),
        manifest=str(manifest), manifest_sha256=sha(manifest), models=models,
        model_sha256={n:sha(p) for n,p in models.items()}, sources={n:sha(ROOT/n) for n in names},
        executable_sha256=sha(ROOT/'build/pvz-portable'), workers=4,
        baseline_seeds=list(range(30000,30256)), teacher_seeds=list(range(30000,30064)),
        collection_seeds=[list(range(1880000,1880128)),list(range(1880128,1880256))],
        collection_labels=['d10orig1','d10orig2'], training_seed=1, labels=labels,
        training_command=command, recipe_sha256=sha(recipe), evaluation_tasks=['day_l10_orig_eval']+[f'day_l{l}_eval_d7' for l in (7,8,9)],
        teacher_baseline_beta=1, collection_beta=0,
        teacher_parameters=dict(samples=3,halving=True,verify_samples=9,student_candidates=2,
            candidate_generator='static',per_card=1,horizon=2500,margin=.02,min_gap=300,rollout_sim_skip=61),
        interpretation='one fresh seed1 model after both frozen-identity collection cohorts; no epoch/seed choice; no production change')
    write(path,plan)
    return plan


def verify(plan):
    if 'recipe_sha256' in plan:
        assert sha(OUT/'training.recipe.json') == plan['recipe_sha256']
    assert sha(plan['manifest'])==plan['manifest_sha256']
    assert sha(ROOT/'build/pvz-portable')==plan['executable_sha256']
    for name,digest in plan['sources'].items():
        assert sha(ROOT/name)==digest,name
    for name,path in plan['models'].items():
        assert sha(path)==plan['model_sha256'][name],name


def run(command,tag,cuda=''):
    marker=OUT/(tag+'.result.json')
    if marker.exists() and read(marker)['returncode']==0:
        return read(marker)
    process=OUT/(tag+'.process.json')
    if process.exists() and guard.live(read(process)['pid']):
        raise RuntimeError(f'{tag} already has a live process; reuse its controller')
    while guard.available()<6*1024*1024:
        print(json.dumps(dict(waiting_for_memory=tag,available_kib=guard.available())),flush=True)
        time.sleep(10)
    result=guard.run(command,tag,extra={'CUDA_VISIBLE_DEVICES':cuda})
    assert result['returncode']==0,(tag,result)
    return result


def play(plan,model,task,seeds,dest,*,teacher=False,beta=0,frames=None,label=None,max_actions=3000):
    # search_play resumes only complete (arm,seed) records; an absent process is not a live job.
    verify(plan)
    definition=next(t for t in read(plan['manifest'])['tasks'] if t['task_id']==task)
    offset=definition['seeds'].index(seeds[0])
    assert definition['seeds'][offset:offset+len(seeds)]==seeds
    command=[guard.PYTHON,'scripts/search_play.py','--rollout-policy',str(model),
        '--value',plan['models']['value'],'--manifest',plan['manifest'],'--task-id',task,
        '--executable',str(ROOT/'build/pvz-portable'),'--resource-dir',guard.ENV['PVZ_RESOURCE_DIR'],
        '--workers','4','--seeds',str(len(seeds)),'--seed-offset',str(offset),
        '--arms','fast_search' if teacher else 'fast','--out',str(dest),'--max-actions',str(max_actions)]
    if teacher:
        command += ['--static-value',plan['models']['static'],'--samples','3','--halving',
            '--verify-samples','9','--student-candidates','2',
            '--per-card','1','--horizon','2500','--margin','.02','--min-gap','300',
            '--rollout-sim-skip','61','--dagger-beta',str(beta)]
    if frames:
        command += ['--frames',str(frames),'--frames-label',label,]
    tag='t20_'+dest.name
    result=run(command,tag)
    rows=[json.loads(v) for v in (dest/'games.jsonl').read_text().splitlines()]
    assert len(rows)==len(seeds) and {r['seed'] for r in rows}==set(seeds)
    assert all(r['arm']==('fast_search' if teacher else 'fast') for r in rows)
    config=read(dest/'config.json')
    assert config['rollout_policy']==str(model) and config['task_id']==task and config['workers']=='4'
    if teacher:
        assert float(config['dagger_beta'])==beta and config['verify_samples']=='9'
    else:
        assert all(r['searched']==0 for r in rows)
    wins=sum(r['won'] for r in rows)
    assert read(dest/'summary.json')['fast_search' if teacher else 'fast']['won']==wins
    verify(plan)
    return dict(games=len(rows),won=wins,games_sha256=sha(dest/'games.jsonl'),resources=result)


def preflight(plan):
    marker=OUT/'preflight_complete.json'
    if marker.exists():
        verify(plan)
        return
    smoke=OUT/'review_smoke_identity'
    result=play(plan,plan['models']['identity'],'day_l10_orig_eval',plan['baseline_seeds'][:2],
        smoke,frames=OUT/'review_smoke_frames',label='review_identity')
    from pvz_frame_store import read_frames
    from episode_query import _replay_task_extra
    from render_episode import collect
    for seed in plan['baseline_seeds'][:2]:
        archive=OUT/f'review_smoke_frames/seed_{seed}/review_identity_fast'
        meta=read(archive/'meta.json');task=meta['task']
        assert task['deck']==[] and task['conveyor_start']=='original' and task['policy']=='student'
        assert task['checkpoint_sha256']==plan['model_sha256']['identity']
        frames=read_frames(archive/'frames.npz')
        assert len(frames[0]['packets'])==10 and all(p[1]<0 for p in frames[0]['packets'])
        replay=collect(guard.ENV['PVZ_RESOURCE_DIR'],seed,10,'student',8,deck=(),
            checkpoint=task['checkpoint'],prefix_actions=[f['action'] for f in frames[1:8]],
            task_extra=_replay_task_extra(task))
        assert replay['task']['conveyor_start']=='original' and replay['task']['deck']==[]
        assert [p[1] for p in replay['frames'][0]['packets']]==[-1]*10
    write(marker,dict(plan_sha256=sha(OUT/'study_plan.json'),identity_smoke=result,
        archived_mode_and_checkpoint_verified=True,live_replay_prefixes=2))


def baseline(plan):
    marker=OUT/'baseline_complete.json'
    if marker.exists():
        verify(plan)
        return
    # v34's existing completed student-only games keep the same slot inputs; retain their provenance.
    folder=OUT/'base_v34';rows=[json.loads(v) for v in (folder/'games.jsonl').read_text().splitlines()]
    assert len(rows)==256 and {r['seed'] for r in rows}==set(plan['baseline_seeds'])
    assert all(r['arm']=='fast' and r['searched']==0 for r in rows)
    config=read(folder/'config.json')
    assert config['rollout_policy']==plan['models']['v34'] and config['task_id']=='day_l10_orig_eval'
    v34=dict(games=256,won=sum(r['won'] for r in rows),games_sha256=sha(folder/'games.jsonl'),
             provenance='completed cb slot baseline; unchanged slot inputs; original binary hash unavailable')
    identity=play(plan,plan['models']['identity'],'day_l10_orig_eval',plan['baseline_seeds'],OUT/'base_identity_aligned')
    teacher=play(plan,plan['models']['v34'],'day_l10_orig_eval',plan['teacher_seeds'],
        OUT/'base_teacher_executed',teacher=True,beta=plan['teacher_baseline_beta'])
    write(marker,dict(plan_sha256=sha(OUT/'study_plan.json'),v34=v34,identity=identity,teacher=teacher))


def collect_data(plan):
    marker=OUT/'collection_complete.json'
    if marker.exists():
        verify(plan)
        return
    records=[];mirror=OUT/'frames_v2'
    assert not mirror.is_symlink() and mirror.resolve().is_relative_to(ROOT)
    for index,(seeds,label) in enumerate(zip(plan['collection_seeds'],plan['collection_labels'])):
        result=play(plan,plan['models']['identity'],'day_l10_orig_train',seeds,OUT/f'collect_round{index+1}',
            teacher=True,beta=plan['collection_beta'],frames=OUT/'frames',label=label)
        for seed in seeds:
            source=OUT/f'frames/seed_{seed}/{label}_fast_search'
            task=read(source/'meta.json')['task']
            assert task['seed']==seed and task['deck']==[] and task['conveyor_start']=='original'
            assert task['checkpoint_sha256']==plan['model_sha256']['identity']
            target=mirror/f'seed_{seed}'
            assert not target.is_symlink() and target.resolve().is_relative_to(ROOT)
            target=target/(label+'_fast_search');target.mkdir(parents=True,exist_ok=True)
            for name in ('meta.json','frames.npz'):
                dest=target/name
                if not dest.exists():os.link(source/name,dest)
                assert sha(dest)==sha(source/name)
        records.append(result)
    write(marker,dict(plan_sha256=sha(OUT/'study_plan.json'),rounds=records,archives=256))


def train(plan):
    marker=OUT/'training_complete.json'
    if marker.exists():
        verify(plan)
        assert sha(OUT/'identity_dagger_s1/model.pt')==read(marker)['model_sha256']
        return
    stable=None
    while True:
        busy=trainers()
        if not busy and guard.available()>=14*1024*1024:
            if stable is None:stable=time.monotonic()
            if time.monotonic()-stable>=30:break
        else:stable=None
        print(json.dumps(dict(training_wait=busy,available_kib=guard.available())),flush=True)
        time.sleep(10)
    verify(plan)
    result=run(plan['training_command'],'t20_training',cuda='0')
    import torch
    model=torch.load(OUT/'identity_dagger_s1/model.pt',map_location='cpu',weights_only=True)
    assert model['packet_head']=='identity' and model['row_pool'] and len(model['history'])==12
    assert set(model['labels'].split(','))==set(plan['labels'])
    assert all(torch.isfinite(v).all() for v in model['state_dict'].values())
    verify(plan)
    write(marker,dict(plan_sha256=sha(OUT/'study_plan.json'),model_sha256=sha(OUT/'identity_dagger_s1/model.pt'),resources=result))


def evaluate(plan):
    trained=read(OUT/'training_complete.json')
    model=OUT/'identity_dagger_s1/model.pt'
    assert sha(model)==trained['model_sha256']
    results={task:play(plan,model,task,plan['baseline_seeds'],OUT/('pilot_'+task))
             for task in plan['evaluation_tasks']}
    verify(plan)
    write(OUT/'evaluation_complete.json',dict(plan_sha256=sha(OUT/'study_plan.json'),
        model_sha256=sha(model),tasks=results,games=1024,complete=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['prepare','preflight','baseline','collect','train','evaluate','all'],required=True)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);plan=prepare();verify(plan)
    phases=dict(preflight=preflight,baseline=baseline,collect=collect_data,train=train,evaluate=evaluate)
    if args.phase=='all':
        for phase in phases.values():phase(plan)
    elif args.phase!='prepare':phases[args.phase](plan)
