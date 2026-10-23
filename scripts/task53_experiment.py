"""Frozen task53 sources, cost gates and experiment stages; collection lives in critical_forks."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'python')]
import critical_forks as forks
OUT = ROOT/'artifacts/task53'


def register_training():
    """Freeze the unchanged replay sample and all optimizer/evaluation choices before labels."""
    from pvz_frame_store import frame_path,read_frames
    if list((OUT/'roots').glob('*/screen_[0-9]*.json')):
        raise RuntimeError('register recipe before terminal labels')
    plan=forks.load(OUT/'plan.json');rng=random.Random(5301);replay=[];evaluation=[]
    for index,e in enumerate(plan['episodes']):
        frames=read_frames(frame_path(Path(e['archive'])))
        valid=[i for i in range(1,len(frames)) if not frames[i].get('action_rejected') and
               (frames[i].get('label') or frames[i].get('action',{})).get('type') in ('plant','wait')]
        rows=sorted(rng.sample(valid,min(128,len(valid))))
        replay.append(dict(archive=e['archive'],frames_sha256=e['frames_sha256'],rows=rows,
                           label='replay_'+e['terrain'],seed=e['reset']['seed']))
        for offset in range(8):
            reset=deepcopy(e['reset']);reset['seed']=53000000+index*100+offset
            evaluation.append(dict(id=f"{e['id']}-eval-{offset}",terrain=e['terrain'],
                                   reset=reset,source_episode=e['id']))
    recipe=forks.load(ROOT/'recipes/dw4i.json')
    recipe.update(data=str(OUT/'training/frames'),out=str(OUT/'training/model'),
                  init=plan['teacher']['rollout_policy'],labels=['replay_day','replay_night','replay_pool'],
                  label_weight={},epochs=2,device='cuda',seed=1,threads=1,precompute_workers=1,
                  wait_keep=1.,max_steps=64,batch_size=256,validation_batch_size=512,
                  cache_dir=str(OUT/'training/cache'),read_cache_dir=None,source_row_report=True)
    recipe['optimizer']['lr']=.0001
    recipe['provenance']=dict(task='task53',parent_recipe_sha256=forks.sha(ROOT/'recipes/dw4i.json'),
                              additional_rows='only independently confirmed critical-fork corrections; labels/weights from export manifest')
    forks.save(ROOT/'MEMORY/docs/task53-critical-forks-finetune.json',recipe)
    forks.save(OUT/'training-registration.json',dict(recipe=recipe,replay=replay,evaluation=evaluation,
               comparison='frozen dw4i_s1 vs one fixed finetune; same 192 evaluation resets, 64 per terrain',
               input_plan_sha256=forks.sha(OUT/'plan.json')))
    forks.save(ROOT/'MEMORY/docs/task53-critical-forks-training-panel.json',dict(
               replay_rows=sum(len(r['rows']) for r in replay),replay=replay,evaluation=evaluation,
               registration_sha256=forks.sha(OUT/'training-registration.json')))
    print('REGISTERED',sum(len(r['rows']) for r in replay),'old replay rows;',len(evaluation),'paired evaluation seeds')


def prepare():
    from pvz_frame_store import frame_path, read_frames
    from render_episode import load_training_task, training_task_kwargs
    protocol = forks.load(ROOT/'MEMORY/docs/task53-critical-forks-plan.json')
    if (OUT/'plan.json').exists():
        raise RuntimeError('already frozen; inspect existing plan instead of overwriting')
    state_path = Path('/home/newbiexvwu/pvz-logs/curriculum_state.json')
    state = forks.load(state_path); latest = state['log'][-1]; pools = defaultdict(list)
    sources = []
    for group in latest['groups']:
        run = MAIN/'artifacts/hard'/f"{group['label']}_{group['games']}"
        config = forks.load(run/'config.json')
        task = load_training_task(group['manifest'], group['task_id'])
        terrain = task['terrain']
        if terrain not in ('day', 'night'):
            continue
        seeds = set(task['seeds'][group['offset']:group['offset']+group['games']])
        rows = {r['seed']: r for r in map(json.loads, (run/'games.jsonl').read_text().splitlines())
                if r['seed'] in seeds and r['arm'] == 'fast_search'}
        if len(rows) != group['games']:
            raise ValueError('latest complete qc group lacks exact seeds: '+str(run))
        losses = sorted([r for r in rows.values() if not r['won']], key=lambda r:r['seed'])
        pools[terrain].append((group, config, task, losses, run))
        sources.append(dict(group=group, games_sha256=forks.sha(run/'games.jsonl'),
                            config_sha256=forks.sha(run/'config.json'), losses=len(losses)))
    # No pool appears in the complete qc round. These separate teacher sources remain explicit.
    for level in (21,22,23,24,26,27,28,29):
        run = MAIN/'artifacts/hard'/f'p{level}dag3_48'; config = forks.load(run/'config.json')
        task = load_training_task(config['manifest'], config['task_id'])
        rows = {r['seed']:r for r in map(json.loads,(run/'games.jsonl').read_text().splitlines())}
        losses = sorted([r for r in rows.values() if not r['won']],key=lambda r:r['seed'])
        group = dict(label=config['frames_label'],task_id=config['task_id'],manifest=config['manifest'])
        pools['pool'].append((group,config,task,losses,run))
    episodes = []; jobs = []
    rng = random.Random(53)
    for terrain in ('day','night','pool'):
        available = [list(p) for p in pools[terrain]]
        for p in available:
            rng.shuffle(p[3])
        chosen = []
        while len(chosen) < protocol['games_per_terrain']:
            progress = False
            for group, config, task, losses, run in available:
                if losses and len(chosen) < protocol['games_per_terrain']:
                    chosen.append((group,config,task,losses.pop(),run)); progress = True
            if not progress:
                raise ValueError('insufficient archived losses for '+terrain)
        for group, config, task, row, run in chosen:
            archive = Path(config['frames'])/f"seed_{row['seed']}"/f"{group['label']}_fast_search"
            frames_path = frame_path(archive); frames = read_frames(frames_path)
            meta = forks.load(archive/'meta.json')
            if meta['outcome']['won'] or not frames[-1]['terminal'] or frames[-1]['result'] != 2:
                raise ValueError('selected loss is not a complete archive')
            kw = training_task_kwargs(task,row['seed'])
            deck = [p[1] for p in frames[0]['packets']]
            if deck != kw['deck']:
                raise ValueError('source deck no longer matches archive')
            reset = dict(level=kw['level'],seed=row['seed'],deck=deck,playthrough=2,
                         first_aid_owned=kw['first_aid_owned'],
                         **{k:v for k,v in kw['task_extra'].items() if v is not None})
            episode = dict(id=f"{terrain}-{row['seed']}-{group['label']}",terrain=terrain,
                source='qc_latest_complete_round' if terrain!='pool' else 'independent_p9k_teacher_round3',
                qc_round=latest['round'] if terrain!='pool' else None,
                archive=str(archive),frames_sha256=forks.sha(frames_path),meta_sha256=forks.sha(archive/'meta.json'),
                manifest=config['manifest'],task_id=config['task_id'],reset=reset,
                source_value=config['value'],student_executed=float(config['dagger_beta'])==0,
                original_outcome=row,source_run=str(run))
            points = forks.select_points(frames, protocol['selection'])
            episode['points'] = points; episodes.append(episode)
            for point in points:
                jobs.append(dict(**{k:v for k,v in episode.items() if k not in ('id','points','original_outcome')},
                                 **point,id=f"{episode['id']}-d{point['decision']}"))
    teacher = dict(rollout_policy=protocol['student'],value=protocol['value'],static_value=protocol['static_value'],
        executable=protocol['executable'],resource_dir=protocol['resource_dir'],
        rollout_sim_skip=61,per_card=1,student_candidates=2,samples=3,horizon=2500,verify_samples=9,min_gap=300)
    paths = {k:Path(teacher[k]) for k in ('rollout_policy','value','static_value','executable')}
    paths.update(collector=ROOT/'scripts/critical_forks.py',search=ROOT/'scripts/search_play.py',
                 env=ROOT/'python/pvz_env.py',student_features=ROOT/'python/pvz_student_observation.py')
    plan = dict(schema_version=1,teacher=teacher,inputs={k:dict(path=str(p),sha256=forks.sha(p)) for k,p in paths.items()},
        jobs=jobs,episodes=episodes,workers=4,out=str(OUT),screen_futures=3,confirm_futures=16,
        artifact_budget_bytes=protocol['artifact_budget_bytes'],protocol=protocol,
        qc_snapshot=dict(round=latest['round'],state_sha256=forks.sha(state_path),groups=sources))
    OUT.mkdir(parents=True,exist_ok=True)
    forks.save(OUT/'qc-state-snapshot.json',state)
    forks.save(OUT/'plan.json',plan)
    forks.save(ROOT/'MEMORY/docs/task53-critical-forks-sources.json',dict(
        qc_round=latest['round'],source_state_sha256=plan['qc_snapshot']['state_sha256'],
        episodes=[{k:v for k,v in e.items() if k!='original_outcome'} for e in episodes],
        frozen_inputs=plan['inputs'],collector_plan_sha256=forks.sha(OUT/'plan.json')))
    print('FROZEN',Counter(e['terrain'] for e in episodes),len(jobs),'roots',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','refresh-setup','register-training'])
    args=p.parse_args()
    if args.stage=='prepare':
        prepare()
    elif args.stage=='register-training':
        register_training()
    else:
        if list((OUT/'roots').glob('*/screen_[0-9]*.json')) or list((OUT/'roots').glob('*/confirm_[0-9]*.json')):
            raise RuntimeError('terminal labels exist; frozen inputs cannot be changed')
        plan=forks.load(OUT/'plan.json');changes=[]
        for name,record in plan['inputs'].items():
            current=forks.sha(record['path'])
            if current!=record['sha256']:
                changes.append(dict(input=name,before=record['sha256'],after=current));record['sha256']=current
        plan.setdefault('setup_revisions',[]).append(dict(before_any_terminal_labels=True,changes=changes))
        forks.save(OUT/'plan.json',plan)
        path=ROOT/'MEMORY/docs/task53-critical-forks-sources.json';sources=forks.load(path)
        sources.update(frozen_inputs=plan['inputs'],collector_plan_sha256=forks.sha(OUT/'plan.json'))
        forks.save(path,sources)
        print('REFRESHED SETUP',changes)


if __name__=='__main__':main()
