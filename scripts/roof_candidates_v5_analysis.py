"""Audited task76 analysis and paired terminal probes; isolated from production.

The original terminal collector is retained as historical invalid data. Schema 2
uses the current root's student history, common futures across actions, and scores
v4/v5 at identical v5-teacher roots with explicit native-network loading.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'python')]
import roof_candidates_v5 as base
OUT = base.OUT
TERM = OUT/'terminal_v2'
SCHEMA = 2
PAYLOAD = None


def key(action):
    return base.action_key(action)


def future_seeds(level, seed, decision, stage, n):
    return [int.from_bytes(hashlib.sha256(
        f'task76-schema2:{level}:{seed}:{decision}:{stage}:{k}'.encode()).digest()[:4], 'little')
        for k in range(n)]


def static_keys(row, per_card=1):
    by_packet = defaultdict(list)
    for r in row['static_scores']:
        by_packet[r['action']['packet']].append((r['value'], key(r['action'])))
    return {k for values in by_packet.values() for _, k in sorted(values,key=lambda x:-x[0])[:per_card]}


def decision_measure(row):
    """Exact logged candidates; never infer a missing top-k from a subset."""
    offered = {key(a) for a in row['production_candidates']}
    static = static_keys(row)
    legal = {key(a) for a in row['legal_actions'] if a['type']=='plant'}
    values = base.round_values(row)
    available = legal & values.keys()
    inside = available & offered
    global_complete = available == legal
    best = max((values[k] for k in available),default=None)
    maxima = {k for k in available if abs(values[k]-best)<1e-12} if best is not None else set()
    best_inside = max((values[k] for k in inside),default=None)
    result = dict(decision=row['decision'],tick=row['tick'],wave=row['wave'],
        legal_plants=len(legal),scored_legal_plants=len(available),global_complete=global_complete,
        production_candidates=len(offered),static_selected_plants=len(static),
        missing_legal_plants=len(legal-offered),chosen_outside_production=key(row['executed']) not in offered,
        scored_best=best,global_best_omitted=(not bool(maxima & offered)) if global_complete and maxima else None,
        global_best_static_omitted=(not bool(maxima & static)) if global_complete and maxima else None,
        global_regret=(best-best_inside) if global_complete and best_inside is not None else None)
    if row['evaluated_candidates']==row['production_candidates']:
        assert not result['chosen_outside_production'], 'production executed an unoffered action'
    return result


def read_receipt(level,arm,seed,verify_trace=False):
    path=OUT/'panels'/f'l{level}'/arm/str(seed)/'result.json'
    if not path.exists():return None
    r=base.read(path);plan=base.read(base.PLAN)
    assert r['identity']['level']==level and r['identity']['seed']==seed and r['identity']['diagnostic_arm']==arm
    assert r['identity']['input_sha256']=={k:v['sha256'] for k,v in plan['inputs'].items()}
    assert r['identity']['manifest_sha256']==plan['manifest_sha256'] and r['identity']['teacher']==plan['teacher']
    assert r['outcome']['terminal'] and r['outcome']['won'] in (True,False)
    if verify_trace:assert base.sha(path.with_name('decisions.jsonl.gz'))==r['trace_sha256']
    return r


def mean(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def coverage_summary(rows):
    measures=[decision_measure(r) for r in rows if r['searched']]
    complete=[r for r in measures if r['global_complete']]
    return dict(search_decisions=len(measures),full_legal_scoring_roots=len(complete),
        unscored_legal_plants=sum(r['legal_plants']-r['scored_legal_plants'] for r in measures),
        mean_legal_plants=mean([r['legal_plants'] for r in measures]),
        mean_production_candidates=mean([r['production_candidates'] for r in measures]),
        global_best_omission_rate=mean([r['global_best_omitted'] for r in complete]),
        static_global_best_omission_rate=mean([r['global_best_static_omitted'] for r in complete]),
        global_regret_above_margin=mean([r['global_regret']>.02 for r in complete if r['global_regret'] is not None]),
        chosen_outside_production_rate=mean([r['chosen_outside_production'] for r in measures]))


def panel_report(levels,verify=False):
    plan=base.verify_plan();report=dict(schema_version=SCHEMA,task=76,levels=levels,
        inputs={k:v['sha256'] for k,v in plan['inputs'].items()},teacher=plan['teacher'],per_level={},
        coverage_method='Actual logged production_candidates including both extra student candidates; round-1 values only. Missing legal actions are unscored, not assumed poor. Global omissions only reported when every legal plant was scored. Tied maxima use set overlap.',
        historical_error='The prior reconstructed subset falsely counted recorded extra-student candidates as omitted. Its omission/regret numbers and causal conclusions are superseded.',
        task74={'commit':'0c543b8df5dd21595a96bdee3c705a8085587923','file':'MEMORY/docs/task74-cold-start-sweep.md','roof_levels_42_to_49':'all 0/16; all eight levels require expansion'})
    overall=Counter();all_complete=True
    for level in levels:
        cell=dict(arms={},paired={})
        outcomes={}
        for arm in plan['arms']:
            receipts=[r for seed in range(45000000,45000032) if (r:=read_receipt(level,arm,seed,verify))]
            outcomes[arm]={r['identity']['seed']:r['outcome'] for r in receipts}
            all_complete &= len(receipts)==32
            traces=[];finals=[];econ=defaultdict(Counter);plantings=Counter();means=[]
            for receipt in receipts:
                seed=receipt['identity']['seed'];rows=base.trace_rows(level,arm,seed)
                assert len(rows)==receipt['recorded_decisions']==receipt['outcome']['decisions']
                assert all(r['decision']==i and not r['rejected'] for i,r in enumerate(rows))
                for row in rows:
                    legal_packets={a['packet'] for a in row['legal_actions'] if a['type']=='plant'}
                    for packet in row['packets']:
                        c=econ[packet['type']];c['observations']+=1
                        if packet['index'] in legal_packets:c['legal']+=1
                        elif not packet['active']:c['inactive_or_cooldown']+=1
                        elif level!=45 and packet['cost']>row['sun']:c['unaffordable']+=1
                        else:c['no_legal_cell_or_other_engine_restriction']+=1
                for kind,n in receipt['outcome'].get('planted_types',{}).items():plantings[kind]+=n
                if rows:
                    plants=rows[-1]['frame']['plants']
                    finals.append(dict(records=len(plants),occupied_cells=len({(p[0],p[1]) for p in plants}),
                        non_pot_cells=len({(p[0],p[1]) for p in plants if p[2]!=33})))
                traces.extend(rows)
            cov=coverage_summary(traces)
            n=len(receipts)
            cell['arms'][arm]=dict(games=n,wins=sum(r['outcome']['won'] for r in receipts),
                median_final_wave=statistics.median([r['outcome']['final_wave'] for r in receipts]) if n else None,
                median_final_tick=statistics.median([r['outcome']['final_tick'] for r in receipts]) if n else None,
                mean_planted_types={k:v/n for k,v in plantings.items()} if n else {},coverage=cov,
                economic_availability={str(k):dict(v) for k,v in sorted(econ.items())},
                final_preterminal_occupancy={k:mean([f[k] for f in finals]) for k in ('records','occupied_cells','non_pot_cells')})
        for arm in plan['arms'][1:]:
            shared=outcomes['production'].keys() & outcomes[arm].keys()
            up=sum(not outcomes['production'][s]['won'] and outcomes[arm][s]['won'] for s in shared)
            down=sum(outcomes['production'][s]['won'] and not outcomes[arm][s]['won'] for s in shared)
            cell['paired'][arm]=dict(pairs=len(shared),gained=up,lost=down,
                exact_mcnemar_p=base.binom_two_sided(min(up,down),up+down))
            if arm=='all_plants':overall.update(pairs=len(shared),gained=up,lost=down)
        report['per_level'][str(level)]=cell
    report['complete']=all_complete
    report['pooled_primary']=dict(overall,exact_mcnemar_p=base.binom_two_sided(min(overall['gained'],overall['lost']),overall['gained']+overall['lost']))
    base.write(ROOT/'MEMORY/docs/task76-roof-candidates-v5-report.json',report)
    print(json.dumps({'complete':all_complete,'pooled_primary':report['pooled_primary'],
        'per_level':{k:{a:[v['games'],v['wins']] for a,v in c['arms'].items()} for k,c in report['per_level'].items()}}),flush=True)
    return report


def history_sha(fast_state):
    from critical_forks import history_record
    return hashlib.sha256(base.action_key(history_record(fast_state)).encode()).hexdigest()


class PairedProbe:
    def __init__(self,plan,level,seed,models,pilot=False):
        self.plan,self.level,self.seed,self.models=plan,level,seed,models
        self.rows=base.trace_rows(level,'production',seed)
        self.targets=set(base.probe_targets(level,seed));self.written=[];self.pilot=pilot
        if pilot:self.targets={min(self.targets)}
        self.decision=None;self.fast_state=None

    def before(self,env,obs,fast_state,decision):
        from render_episode import _frame
        self.decision,self.fast_state=decision,fast_state
        assert _frame(obs)==self.rows[decision]['frame'],f'production replay state differs at {decision}'

    def after(self,obs,action,policy,rejected,info):
        expected=self.rows[self.decision]
        assert not rejected and action==expected['executed'] and policy==expected['policy'],f'production replay action differs at {self.decision}'

    def __enter__(self):
        import search_play as teacher
        self.original=teacher._halving_choice
        def wrapper(env,obs,root,candidates,policy,policy_key,args,decision_seed,stats):
            if self.decision in self.targets:
                assert candidates==self.rows[self.decision]['production_candidates']
                self.probe(env,obs,root,candidates,policy)
            return self.original(env,obs,root,candidates,policy,policy_key,args,decision_seed,stats)
        teacher._halving_choice=wrapper
        return self

    def __exit__(self,*error):
        import search_play as teacher
        teacher._halving_choice=self.original

    def probe(self,env,obs,root,candidates,policy):
        import search_play as teacher
        env.restore_fast(root)
        state_hash=env.state_hash();income=(list(env._sun_production_history),env._sun_history_start_tick)
        history=copy.deepcopy(teacher._W.get('student_history'));hs=history_sha(self.fast_state)
        actions=base.all_actions(obs)
        filename=TERM/f'l{self.level}'/str(self.seed)/f'root_{self.decision}.json'
        identity=dict(schema_version=SCHEMA,level=self.level,seed=self.seed,decision=self.decision,
            input_sha256={k:v['sha256'] for k,v in self.plan['inputs'].items()},
            native_model_bin_sha256={k:base.sha(v) for k,v in self.models.items()},
            trace_sha256=base.sha(OUT/'panels'/f'l{self.level}'/'production'/str(self.seed)/'decisions.jsonl.gz'),
            state_hash=state_hash,history_sha256=hs,terminal_futures=8,leaf_futures=9,
            horizons=[2500,4000,6000],protocol='current v5 teacher prefix; frozen student continuation; common futures across actions; actual terminal required')
        if filename.exists():
            old=base.read(filename);assert old['identity']==identity and old['complete']
            self.written.append(str(filename));return
        world=future_seeds(self.level,self.seed,self.decision,'terminal',8)
        leaf_seeds=future_seeds(self.level,self.seed,self.decision,'leaf',9)
        records=[];started=time.monotonic()
        try:
            for index,action in enumerate(actions):
                env.restore_fast(root)
                branch=env.branch_snapshot(root,[action],retain_snapshots=False)[0]
                assert branch['ok'],f'listed legal action rejected: {action}'
                outcomes=[]
                env.load_value_net(self.models['value'])
                for future in world:
                    terminal=env.rollout(action,0,sim_skip=self.plan['teacher']['rollout_sim_skip'],
                        snapshot_id=root,seed=future,student_history=history)
                    assert terminal.get('terminal') and terminal.get('result') in (1,2),'unfinished future cannot be a loss'
                    outcomes.append(dict(future=future,won=terminal['result']==1,wave=terminal['wave'],tick=terminal['tick']))
                predictions={}
                for name,path in self.models.items():
                    env.load_value_net(path)
                    predictions[name]={}
                    for horizon in ([2500,4000,6000] if name=='value' else [2500]):
                        values=[]
                        for future in leaf_seeds:
                            result=env.rollout(action,obs['tick']+horizon,
                                sim_skip=self.plan['teacher']['rollout_sim_skip'],snapshot_id=root,
                                seed=future,student_history=history,score_value=True)
                            value=teacher._score_terminal_or_value(result)
                            if value is None:value=result['value']
                            assert math.isfinite(value)
                            values.append(float(value))
                        predictions[name][str(horizon)]=values
                records.append(dict(action=action,terminal=outcomes,leaf=predictions))
                if index%10==0:print(json.dumps(dict(level=self.level,seed=self.seed,root=self.decision,
                    action=index+1,total_actions=len(actions),elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
        finally:
            env.load_value_net(self.models['value'])
            env.restore_fast(root)
        assert env.state_hash()==state_hash,'probe changed native root'
        assert (list(env._sun_production_history),env._sun_history_start_tick)==income,'probe changed income history'
        assert history_sha(self.fast_state)==hs,'probe changed student history'
        document=dict(identity=identity,complete=True,tick=obs['tick'],wave=obs['wave'],sun=obs['sun'],
            packets=obs['packets'],policy=policy,production_choice=self.rows[self.decision]['executed'],
            production_candidates=candidates,static_selected_keys=sorted(static_keys(self.rows[self.decision])),
            actions=records,terminal_world_seeds=world,leaf_world_seeds=leaf_seeds,
            root_restore_verified=True,income_history_verified=True,student_history_verified=True,
            elapsed_seconds=time.monotonic()-started)
        base.write(filename,document);self.written.append(str(filename))


def init_probe(payload):
    global PAYLOAD
    PAYLOAD=payload
    base.initialize_worker(payload['plan'])


def terminal_job(job):
    import search_play as teacher
    level,seed=job;plan=PAYLOAD['plan'];teacher._close_worker_env()
    args=base.teacher_args(plan,level)
    observer=PairedProbe(plan,level,seed,PAYLOAD['models'],PAYLOAD.get('pilot',False))
    with observer:
        outcome=teacher._play(dict(args=args,seed=seed,arm='fast_search'),observer=observer)
    original=read_receipt(level,'production',seed)['outcome']
    assert base.semantic_game(outcome)==base.semantic_game(original),'terminal probe changed complete production replay'
    result=dict(level=level,seed=seed,roots=len(observer.written),complete_game_parity=True,
        input_sha256={k:v['sha256'] for k,v in plan['inputs'].items()},outcome=base.semantic_game(outcome),
        root_sha256={str(path):base.sha(path) for path in observer.written})
    base.write(TERM/f'l{level}'/str(seed)/'replay-complete.json',result)
    print(json.dumps({'replay_complete':True,'level':level,'seed':seed,'roots':len(observer.written)}),flush=True)
    teacher._close_worker_env()
    return result


def safe_terminal_job(job):
    import search_play as teacher
    try:
        return terminal_job(job)
    except Exception:
        import traceback
        result=dict(level=job[0],seed=job[1],error=traceback.format_exc(),complete_game_parity=False)
        base.write(TERM/f'l{job[0]}'/str(job[1])/'failure.json',result)
        print(json.dumps(result),flush=True)
        return result
    finally:
        teacher._close_worker_env()


def terminal_probe(levels,workers,pilot=False):
    import search_play as teacher
    from export_value_net import export_value_net
    from pvz_value_net import load_value_net
    base.require_game_capacity(workers)
    plan=base.verify_plan()
    with base.controller_lock():
        keep=base.preload(plan)
        v4=OUT/'frozen/v4_control.bin'
        export_value_net(load_value_net(plan['inputs']['v4_control']['path']),str(v4))
        payload=dict(plan=plan,models={'value':teacher._W['value_files'][0],'v4_control':str(v4)},pilot=pilot)
        assert base.sha(payload['models']['value'])!=base.sha(v4),'two controls exported identical native models'
        jobs=[(level,seed) for level in levels for seed in base.probe_seeds(level,2)]
        if pilot:jobs=jobs[:1]
        try:
            with mp.get_context('fork').Pool(min(workers,len(jobs)),initializer=init_probe,initargs=(payload,)) as pool:
                results=list(pool.imap_unordered(safe_terminal_job,jobs,chunksize=1))
            failed=[r for r in results if not r.get('complete_game_parity')]
            base.write(TERM/('pilot-complete.json' if pilot else 'complete.json'),dict(schema_version=SCHEMA,
                levels=levels,jobs=len(jobs),roots=sum(r.get('roots',0) for r in results),
                pilot=pilot,complete=not failed,failed_jobs=len(failed)))
            if failed:raise RuntimeError('terminal replay/probe failures; inspect per-seed failure.json; not loss labels')
        finally:keep.cleanup()


def direction_score(prediction,truth):
    if prediction==0:return .5
    return float((prediction>0)==(truth>0))


def metrics(document,model='value',horizon=2500,representable=True):
    records=[a for a in document['actions'] if not representable or a['action']['type']!='shovel']
    truth={key(a['action']):mean([float(v['won']) for v in a['terminal']]) for a in records}
    pred={key(a['action']):mean(a['leaf'][model][str(horizon)]) for a in records}
    scores=[];ks=sorted(truth)
    for i,a in enumerate(ks):
        for b in ks[i+1:]:
            delta=truth[a]-truth[b]
            if delta:scores.append(direction_score(pred[a]-pred[b],delta))
    max_truth=max(truth.values());min_truth=min(truth.values());max_pred=max(pred.values())
    best_truth={k for k,v in truth.items() if v==max_truth};best_pred={k for k,v in pred.items() if abs(v-max_pred)<1e-12}
    return dict(actions=len(records),pairs=len(scores),pair_score=sum(scores),
        accuracy=mean(scores),discriminating=max_truth>min_truth,
        tie_aware_argmax_overlap=bool(best_truth & best_pred),
        all_actions_zero_wins=max_truth==0,max_terminal_win_fraction=max_truth,
        mae=mean([abs(pred[k]-truth[k]) for k in ks]),
        selected_leaf_prediction=max_pred,
        selected_leaf_action_truth=mean([truth[k] for k in best_pred]),
        mean_prediction=mean(list(pred.values())),mean_truth=mean(list(truth.values())))


def terminal_report(levels):
    report=dict(schema_version=SCHEMA,task=76,complete=False,per_level={},
        method='Shared eight terminal futures for every legal action at identical current-v5-teacher roots; current student history passed after proposal; true terminal checked; nine independent shared leaf futures. v4 loaded explicitly for identical roots/futures; v5 horizons2500/4000/6000.',
        historical_comparison='task46 v4 53.9% (95%47.1-60.9): different roots, models, two futures and search-teacher continuation, so no direct before/after claim. Matched v4/v5 numbers below isolate leaf model under task72b student continuation.',
        invalidated_legacy='artifacts/task76/terminal schema1 used preloaded v5 for v4_control, pre-proposal history and action-specific futures; excluded in full.',
        root_evidence=[],good_action_coverage={})
    complete=True;clusters=defaultdict(list);coverage=Counter()
    for level in levels:
        seeds=base.probe_seeds(level,2);expected=[(seed,d) for seed in seeds for d in base.probe_targets(level,seed)]
        rows=[]
        for seed,d in expected:
            path=TERM/f'l{level}'/str(seed)/f'root_{d}.json';replay=path.with_name('replay-complete.json')
            if not path.exists() or not replay.exists():complete=False;continue
            root=base.read(path);replay_record=base.read(replay)
            if replay_record.get('root_sha256',{}).get(str(path))!=base.sha(path):complete=False;continue
            assert root['complete'] and root['identity']['schema_version']==SCHEMA
            assert len(root['actions'])==len({key(a['action']) for a in root['actions']})
            assert all(len(a['terminal'])==8 and [r['future'] for r in a['terminal']]==root['terminal_world_seeds'] for a in root['actions'])
            summary={}
            for model,horizon in [('value',2500),('value',4000),('value',6000),('v4_control',2500)]:
                name=f'{model}_h{horizon}'
                summary[name]=metrics(root,model,horizon)
                clusters[(model,horizon,level,seed)].append(summary[name])
            rows.append(dict(seed=seed,decision=d,tick=root['tick'],metrics=summary))
            policy=root['policy'] if root['policy']['type']=='plant' else dict(type='wait',ticks=60,until='timeout')
            gold={key(a['action']):mean([v['won'] for v in a['terminal']]) for a in root['actions']}
            anchor=gold[key(policy)];better={k for k,v in gold.items() if v>anchor and json.loads(k)['type']=='plant'}
            static=set(root['static_selected_keys']);production={key(a) for a in root['production_candidates']}
            coverage.update(roots=1,observed_improving_plant_actions=len(better),static_missed=len(better-static),production_missed=len(better-production))
            report['root_evidence'].append(dict(path=str(path),sha256=base.sha(path),level=level,seed=seed,decision=d,
                actions=len(root['actions']),terminal_futures=len(root['actions'])*8))
        report['per_level'][str(level)]=dict(expected_roots=len(expected),completed_roots=len(rows),roots=rows)
    pooled={}
    rng=random.Random(76)
    for model,horizon in [('value',2500),('value',4000),('value',6000),('v4_control',2500)]:
        cohorts=[v for k,v in clusters.items() if k[:2]==(model,horizon)]
        values=[v for cohort in cohorts for v in cohort]
        pairs=sum(v['pairs'] for v in values);score=sum(v['pair_score'] for v in values)
        boot=[]
        for _ in range(2000):
            sample=[v for _ in cohorts for v in rng.choice(cohorts)] if cohorts else []
            n=sum(v['pairs'] for v in sample)
            if n:boot.append(sum(v['pair_score'] for v in sample)/n)
        boot.sort()
        pooled[f'{model}_h{horizon}']=dict(roots=len(values),reset_seed_clusters=len(cohorts),pairs=pairs,
            pairwise_accuracy=score/pairs if pairs else None,
            reset_seed_bootstrap95=[boot[int(.025*(len(boot)-1))],boot[int(.975*(len(boot)-1))]] if boot else None,
            all_actions_zero_wins_roots=sum(v['all_actions_zero_wins'] for v in values),
            discriminating_roots=sum(v['discriminating'] for v in values),
            argmax_overlap_on_discriminating=mean([v['tie_aware_argmax_overlap'] for v in values if v['discriminating']]),
            mean_root_mae=mean([v['mae'] for v in values]),mean_root_prediction=mean([v['mean_prediction'] for v in values]),
            mean_root_truth=mean([v['mean_truth'] for v in values]))
    report['pooled']=pooled;report['complete']=complete;report['good_action_coverage']=dict(coverage)
    if coverage['observed_improving_plant_actions']:
        report['good_action_coverage'].update(static_miss_rate=coverage['static_missed']/coverage['observed_improving_plant_actions'],
            production_miss_rate=coverage['production_missed']/coverage['observed_improving_plant_actions'])
    base.write(ROOT/'MEMORY/docs/task76-roof-terminal-truth.json',report)
    print(json.dumps({'complete':complete,'pooled':pooled,'good_action_coverage':report['good_action_coverage']}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['panel-report','terminal','terminal-report','pilot'])
    p.add_argument('--levels',default='42,43,44,45,46,47,48,49')
    p.add_argument('--workers',type=int,default=base.MAX_GAME_PROCESSES)
    p.add_argument('--verify-traces',action='store_true')
    a=p.parse_args();levels=list(map(int,a.levels.split(',')))
    if a.mode=='panel-report':panel_report(levels,a.verify_traces)
    elif a.mode=='terminal-report':terminal_report(levels)
    else:terminal_probe(levels,a.workers,a.mode=='pilot')


if __name__=='__main__':main()
