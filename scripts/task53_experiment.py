"""Frozen task53 sources, cost gates and experiment stages; collection lives in critical_forks."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
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
    recipe['provenance']=dict(source_path=str(ROOT/'recipes/dw4i.json'),
        source_sha256=forks.sha(ROOT/'recipes/dw4i.json'),
        notes='task53: only independently confirmed correction rows added to the fixed replay recipe; labels/weights from export manifest')
    forks.save(ROOT/'MEMORY/docs/task53-critical-forks-finetune.json',recipe)
    forks.save(OUT/'training-registration.json',dict(recipe=recipe,replay=replay,evaluation=evaluation,
               comparison='frozen dw4i_s1 vs one fixed finetune; same 192 evaluation resets, 64 per terrain',
               input_plan_sha256=forks.sha(OUT/'plan.json')))
    forks.save(ROOT/'MEMORY/docs/task53-critical-forks-training-panel.json',dict(
               replay_rows=sum(len(r['rows']) for r in replay),replay=replay,evaluation=evaluation,
               registration_sha256=forks.sha(OUT/'training-registration.json')))
    print('REGISTERED',sum(len(r['rows']) for r in replay),'old replay rows;',len(evaluation),'paired evaluation seeds')


def refine_selection():
    """Separate the two root mechanisms using archive timing alone, before outcome analysis."""
    from pvz_frame_store import frame_path,read_frames
    plan=forks.load(OUT/'plan.json');jobs=[];per_episode=[]
    params=forks.load(ROOT/'MEMORY/docs/task53-critical-forks-plan.json')['selection_v2']
    lookback=params['breach_lookback_decisions']
    if lookback<1 or not 0<=params['flat_quantile']<=1 or params['max_points_per_game']<1:
        raise ValueError('invalid point selection parameters')
    if (OUT/'plan-keypoints-v2.json').exists():
        raise RuntimeError('v2 selection already frozen')
    for episode in plan['episodes']:
        frames=read_frames(frame_path(Path(episode['archive'])))
        # The loss is the final breach. The previous N decisions form its lookback window;
        # take the beginning of that window instead of the already-doomed final move.
        decision=max(1,len(frames)-lookback)
        reasons={decision:[dict(kind='before_loss_breach',lookback_decisions=lookback,
                                breach_decision=len(frames)-1)]}
        flat=[]
        for i in range(1,len(frames)):
            targets=frames[i].get('targets',[])
            vals=[a[3] for a in targets]
            if (len(vals)>=2 and max(vals)-min(vals)<params['flat_spread_threshold'] and
                    frames[-1]['tick']-frames[i-1]['tick']<=params['flat_window_before_loss_ticks']):
                flat.append((i,max(vals)-min(vals)))
        if flat:
            i,spread=flat[min(len(flat)-1,int(len(flat)*params['flat_quantile']))]
            reasons.setdefault(i,[]).append(dict(kind='flat_archived_leaf',spread=spread,
                                                position='median in final 6000 ticks'))
        indices=sorted(reasons)[:params['max_points_per_game']]
        points=[dict(decision=i,tick=frames[i-1]['tick'],reasons=reasons[i]) for i in indices]
        for point in points:
            base={k:v for k,v in episode.items() if k not in ('id','points','original_outcome')}
            jobs.append(dict(**base,**point,id=f"v2-{episode['id']}-d{point['decision']}"))
        per_episode.append(dict(episode=episode['id'],points=points))
    plan['jobs']=jobs
    plan['sampling_refinement']=dict(version=2,parameters=params,source_plan_sha256=forks.sha(OUT/'plan.json'),
        rule='16th decision before loss; median archived-flat search point in final 6000 ticks',
        reason='Initial structural inspection showed adjacent last decisions and no plant options at some final roots.',
        candidate_outcome_analysis_before_refinement=False,off_panel_pilot_labels=543,
        initial_pilot_cpu_access='cpu_seconds only; original pilot labels retained and charged',
        episodes=per_episode)
    forks.save(OUT/'plan-keypoints-v2.json',plan)
    pilot=[next(j for j in jobs if j['terrain']==t) for t in ('day','night','pool')]
    forks.save(OUT/'pilot-v2.json',dict(jobs=pilot))
    forks.save(ROOT/'MEMORY/docs/task53-critical-forks-sampling-v2.json',dict(
        **plan['sampling_refinement'],inputs=plan['inputs'],jobs=jobs,
        plan_sha256=forks.sha(OUT/'plan-keypoints-v2.json')))
    print('FROZEN V2',len(jobs),'roots; pilot',[(j['terrain'],j['decision'],j['tick']) for j in pilot])


def cost_gate():
    """Only timing and candidate counts enter this gate; candidate outcomes are not read."""
    plan=forks.load(OUT/'plan-keypoints-v2.json')
    state=forks.load(OUT/'qc-state-snapshot.json');groups=state['log'][-1]['groups']
    runs=[MAIN/'artifacts/hard'/f"{g['label']}_{g['games']}" for g in groups]
    start=min((r/'config.json').stat().st_mtime for r in runs)
    end=max((r/'games.jsonl').stat().st_mtime for r in runs)
    qc_cpu=0.;metrics=[]
    for path in Path('/home/newbiexvwu/pvz-logs/metrics').glob('curriculum_queue-*.jsonl'):
        measured=0.
        for line in path.read_text().splitlines():
            row=json.loads(line);stop=row['timestamp'];duration=row.get('interval_s',0)
            overlap=max(0,min(end,stop)-max(start,stop-duration))
            measured+=overlap*(row.get('tree_cpu_percent') or 0)/100
        if measured:
            qc_cpu+=measured;metrics.append(dict(path=str(path),sha256=forks.sha(path),cpu_seconds=measured))
    games=sum(g['games'] for g in groups)
    if games!=768 or qc_cpu<=0:raise ValueError('missing 768-game qc CPU reference')
    rates={};pilot=forks.load(OUT/'pilot-v2.json')['jobs']
    for job in pilot:
        folder=OUT/'roots'/job['id'];marker=forks.load(folder/'screen-complete.json')
        times=[forks.load(p)['cpu_seconds'] for p in folder.glob('screen_[0-9]*.json')]
        if len(times)!=marker['labels']:raise ValueError('incomplete timing pilot')
        rates[job['terrain']]=sum(times)/len(times)
    estimates=[]
    for job in plan['jobs']:
        n=len(forks.load(OUT/'roots'/job['id']/'root.json')['candidates'])
        # Reserve confirmation of challenger/student in all 16 new futures, plus two replays.
        estimate=1.5*(3*n+32)*rates[job['terrain']]+12
        estimates.append(dict(id=job['id'],terrain=job['terrain'],candidates=n,forecast_cpu_seconds=estimate))
    budget=qc_cpu*.15;spent=forks.spent_cpu(OUT);forecast=spent+sum(e['forecast_cpu_seconds'] for e in estimates)+250
    value=dict(qc_round=state['round'],qc_games=games,qc_cpu_seconds=qc_cpu,
        qc_reference_window_unix=[start,end],metrics=metrics,
        qc_measurement='read-only curriculum process-tree CPU sample integration over frozen complete round; final exiting children between samples can be missed, making this reference conservative',
        qc_reference_engine='deploy2 (engine used by the frozen qc round); task53 collection uses required deploy3',
        cpu_budget_seconds=budget,cpu_already_spent=spent,label_cpu_seconds_by_terrain=rates,
        per_point=estimates,forecast_total_cpu_seconds=forecast,forecast_share=forecast/qc_cpu,
        approved=forecast<=budget,production_root_quota=math.floor(budget/max(e['forecast_cpu_seconds'] for e in estimates)),
        production_policy='768-game round: select a deterministic loss-root reservoir, cap by root quota and accumulated forecast; stop collection at measured 15% CPU. Do not process every loss root.')
    forks.save(ROOT/'MEMORY/docs/task53-critical-forks-cost.json',value)
    if not value['approved']:
        raise RuntimeError('forecast exceeds 15%; tighten selection before full labels')
    plan['cpu_budget_seconds']=budget;plan['cost_gate']=value
    forks.save(OUT/'plan-budgeted.json',plan)
    print('COST GATE',forecast,'/',budget,'CPU seconds;',round(value['forecast_share']*100,2),'% qc CPU')


def budget_roots(jobs,prices,budget,max_roots,round_index=0):
    """Budget whole roots, stratify reasons/terrain, and rotate priority across rounds."""
    if budget<=0 or max_roots<1 or any(not math.isfinite(prices[j['id']]) or prices[j['id']]<=0 for j in jobs):
        raise ValueError('positive finite root costs, budget and quota required')
    pools=defaultdict(list)
    for job in jobs:
        mechanism='flat' if any(r['kind'].startswith('flat') for r in job['reasons']) else 'breach'
        pools[(job['terrain'],mechanism)].append(job)
    for values in pools.values():values.sort(key=lambda j:(prices[j['id']],j['id']))
    strata=sorted(pools);offset=round_index%len(strata);strata=strata[offset:]+strata[:offset]
    selected=[];spent=0.
    while len(selected)<max_roots:
        added=False
        for stratum in strata:
            if len(selected)>=max_roots:break
            values=pools[stratum]
            while values:
                job=values.pop(0);price=prices[job['id']]
                if spent+price<=budget:
                    selected.append(job);spent+=price;added=True;break
        if not added:break
    if not selected:raise ValueError('no complete legal-action root fits the budget')
    return selected,spent


def production_preview():
    plan=forks.load(OUT/'plan-budgeted.json');raw=forks.load(OUT/'screen-summary.json')
    legacy=forks.load(ROOT/'MEMORY/docs/task53-critical-forks-cost.json')
    current=forks.load(ROOT/'MEMORY/docs/task53-critical-forks-deploy3-cost.json')
    reference=min(legacy['qc_cpu_seconds'],current['projected_768_game_cpu_seconds']);budget=.15*reference
    prices={g['job']['id']:1.5*(sum(c['cpu_seconds'] for c in g['candidates'])+
                 32*max(c['cpu_seconds']/g['samples'] for c in g['candidates']))+12 for g in raw['groups']}
    fixed_overhead=250.
    selected,predicted=budget_roots(plan['jobs'],prices,budget-fixed_overhead,legacy['production_root_quota'],
                                    round_index=legacy['qc_round'])
    preview=deepcopy(plan);preview.update(jobs=selected,out=str(OUT/'production-preview/round_000002'),
                                        cpu_budget_seconds=budget)
    path=OUT/'production-plan.preview.json';forks.save(path,preview)
    value=dict(status='dry_run_only_not_deployed',candidate_model_sha256=plan['inputs']['rollout_policy']['sha256'],
        reference_cpu_seconds=reference,reference_scope='minimum of measured old full round and current deploy3 night-group projection; refresh from a complete deployment-matched mixed round before online use',
        budget_fraction=.15,budget_cpu_seconds=budget,selected_roots=len(selected),
        selected_by_terrain=dict(Counter(j['terrain'] for j in selected)),
        forecast_cpu_seconds=predicted,forecast_share=predicted/reference,
        fixed_overhead_cpu_seconds=fixed_overhead,forecast_total_cpu_seconds=predicted+fixed_overhead,
        forecast_total_share=(predicted+fixed_overhead)/reference,
        scope='label screening and independent confirmation plus fixed overhead; task49 full training and promotion evaluation are accounted separately',
        root_ids=[j['id'] for j in selected],per_root_forecast={j['id']:prices[j['id']] for j in selected},
        plan=str(path),plan_sha256=forks.sha(path),
        tradeoff='Less data per round; cost ordering within rotated terrain/reason strata can favour cheaper/shorter continuations. All legal actions and 16 confirmation futures remain intact. No empty-root budget success.',
        command=[sys.executable,str(ROOT/'scripts/critical_forks.py'),str(path),'screen'])
    forks.save(ROOT/'MEMORY/docs/task53-critical-forks-production-preview.json',value)
    print('PRODUCTION DRY RUN',len(selected),'roots;',predicted,'CPU seconds;',predicted/reference,'share')


def repair_recipe_metadata():
    """Metadata-only compatibility repair, preserving the original registered settings."""
    from pvz_recipe import validate
    registration_path=OUT/'training-registration.json';registration=forks.load(registration_path)
    if (OUT/'training/model/model.pt').exists():raise RuntimeError('repair before training')
    previous_sha=forks.sha(registration_path)
    frozen=OUT/'frozen/training-registration-before-metadata-repair.json'
    if not frozen.exists():frozen.parent.mkdir(parents=True,exist_ok=True);frozen.write_bytes(registration_path.read_bytes())
    provenance=dict(source_path=str(ROOT/'recipes/dw4i.json'),source_sha256=forks.sha(ROOT/'recipes/dw4i.json'),
        notes='task53: fixed pre-registered replay plus independently confirmed corrections; descriptive provenance field names repaired before training, no training settings changed')
    for path in [ROOT/'MEMORY/docs/task53-critical-forks-finetune.json',OUT/'training/recipe.json']:
        recipe=forks.load(path);old=deepcopy(recipe);recipe['provenance']=provenance
        assert {k:v for k,v in recipe.items() if k!='provenance'}=={k:v for k,v in old.items() if k!='provenance'}
        validate(recipe,'training');forks.save(path,recipe)
    registration['recipe']['provenance']=provenance
    registration.setdefault('metadata_revisions',[]).append(dict(previous_registration_sha256=previous_sha,
        training_settings_unchanged=True,reason='existing recipe schema only accepts descriptive source_path/source_sha256/notes'))
    forks.save(registration_path,registration)
    path=ROOT/'MEMORY/docs/task53-critical-forks-training-panel.json';panel=forks.load(path)
    panel['registration_sha256']=forks.sha(registration_path);panel['metadata_revisions']=registration['metadata_revisions']
    forks.save(path,panel)
    print('METADATA REPAIRED; hyperparameters, replay rows, new rows and evaluation resets unchanged')


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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','refresh-setup','register-training','refine-selection','cost-gate','production-preview','repair-recipe-metadata'])
    args=p.parse_args()
    if args.stage=='prepare':
        prepare()
    elif args.stage=='register-training':
        register_training()
    elif args.stage=='refine-selection':
        refine_selection()
    elif args.stage=='cost-gate':
        cost_gate()
    elif args.stage=='production-preview':
        production_preview()
    elif args.stage=='repair-recipe-metadata':
        repair_recipe_metadata()
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
