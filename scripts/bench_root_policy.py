#!/usr/bin/env python3
"""Full teacher games: independent native roots, one fixed CNN continuation, per-decision timing."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import torch
import numpy as np
import search_play as search
from pvz_rollout_policy import load_student,export_student
from pvz_generic_forward import export_graph,rollout_graph
from pvz_student_v3 import StudentV3
from pvz_tasks import deck_for_level


ARMS=('cnn64','token128_2','token192_4','terrain5')


class Timings:
    def __init__(self,arm,folder,matched=False):
        self.arm=arm;self.rows=[];self.root=0.;self.search=0.;self.folder=folder
        self.matched=matched;self.loaded=False
    def before(self,env,obs,state,index):
        if self.matched:
            if not self.loaded:
                for arm in ARMS:
                    path=str(self.folder/f'{arm}.bin')
                    quoted='"'+path.replace('\\','\\\\').replace('"','\\"')+'"'
                    if not env._command('LOAD_ROOT_BENCH_NET '+quoted).get('ok'):
                        raise RuntimeError('isolated engine requires LOAD_ROOT_BENCH_NET')
                self.loaded=True
            previous=(state.get('frame') or {}).get('tick',-1)
            planted=int((state.get('action') or {}).get('type')=='plant')
            command=f'ROOT_BENCH_COMPARE {previous} {planted} {obs["sun_income_rate"]:.17g}'
            self.comparison=env._command(command+env._student_options(state.get('history'),None))
            if not self.comparison.get('ok'): raise RuntimeError('ROOT_BENCH_COMPARE failed')
        self.started=time.perf_counter();self.tick=obs['tick'];self.root=0.;self.search=0.
        self.rollouts=dict(search._W.get('rollout_metrics',{}))
    def after(self,obs,action,policy,rejected,info):
        elapsed=time.perf_counter()-self.started
        metrics=search._W.get('rollout_metrics',{})
        self.rows.append(dict(decision=len(self.rows),tick=self.tick,end_tick=obs['tick'],
            e2e_us=elapsed*1e6,root_us=self.root*1e6,search_us=self.search*1e6,
            continuation_decisions=metrics.get('decisions',0)-self.rollouts.get('decisions',0),
            network_us=(metrics.get('network_seconds',0)-self.rollouts.get('network_seconds',0))*1e6))
        if self.matched:
            self.rows[-1].update(matched_root_us=self.comparison['us'],tokens=self.comparison['tokens'])
        if len(self.rows)%100==0:
            print(f'{self.arm}: {len(self.rows)} decisions, tick {obs["tick"]}, {sum(r["e2e_us"] for r in self.rows)/1e6:.1f}s',flush=True)
    def save(self):
        with (self.folder/f'{self.arm}-decisions.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(self.rows[0]));writer.writeheader();writer.writerows(self.rows)
    def summary(self):
        values=np.asarray([r['e2e_us'] for r in self.rows])
        root=np.asarray([r['root_us'] for r in self.rows]);seconds=values.sum()/1e6
        summary=dict(decisions=len(values),decision_seconds=seconds,games_per_hour_decisions=3600/seconds,
            e2e_us=dict(mean=float(values.mean()),p50=float(np.median(values)),p95=float(np.percentile(values,95)),max=float(values.max())),
            root_us=dict(mean=float(root.mean()),p50=float(np.median(root)),p95=float(np.percentile(root,95))),
            root_share=float(root.sum()/values.sum()),continuation_decisions=sum(r['continuation_decisions'] for r in self.rows),
            continuation_network_seconds=sum(r['network_us'] for r in self.rows)/1e6)
        if self.matched:
            matched=np.asarray([r['matched_root_us'] for r in self.rows])
            summary['same_trajectory_counterfactual']={}
            for i,arm in enumerate(ARMS):
                delta=(matched[:,i]-matched[:,0]).sum()/1e6
                predicted=seconds+delta
                summary['same_trajectory_counterfactual'][arm]=dict(
                    native_root_mean_us=float(matched[:,i].mean()),root_extra_seconds=float(delta),
                    counterfactual_seconds=predicted,games_per_hour=3600/predicted,
                    games_per_hour_change_percent=(seconds/predicted-1)*100,
                    e2e_mean_us=float((values+matched[:,i]-matched[:,0]).mean()))
            lengths=np.asarray([r['tokens'] for r in self.rows])
            summary['actual_tokens']=dict(min=int(lengths.min()),max=int(lengths.max()),mean=float(lengths.mean()))
        return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cnn',type=Path,required=True)
    p.add_argument('--value',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--level',type=int,default=9)
    p.add_argument('--seed',type=int,default=161006)
    p.add_argument('--horizon',type=int,default=2500)
    p.add_argument('--samples',type=int,default=3)
    p.add_argument('--matched',action='store_true',help='measure all four native roots ABBA on every CNN game frame outside its decision timer; hold CNN search/actions fixed for throughput counterfactual')
    p.add_argument('--arm',action='append',choices=('cnn64','token128_2','token192_4','terrain5'))
    p.add_argument('--executable',type=Path,default=ROOT/'build-cppnet/pvz-portable')
    p.add_argument('--resource-dir',type=Path,default=Path.home()/'.cache/pvz-research-resources')
    args=p.parse_args();torch.set_num_threads(1);torch.manual_seed(161)
    folder=args.output.resolve();folder.mkdir(parents=True,exist_ok=True)
    cnn=load_student(args.cnn)
    if cnn.trunk[0].out_channels!=64: raise ValueError('comparison requires CNN64')
    export_graph(cnn,rollout_graph(cnn),folder/'cnn64.bin')
    manifest=folder/'tasks.json'
    manifest.write_text(json.dumps(dict(tasks=[dict(task_id='task161_bench',level=args.level,seeds=[args.seed],deck=list(deck_for_level(args.level)))])))
    keep,values=search._export_values(str(args.value),None)
    search._load_nets(values[0],str(args.cnn),values[1])
    search._W.update(rollout_sim_skip=61,lean_rollouts=True,batch_waits=True)
    report=dict(level=args.level,seed=args.seed,deck=list(deck_for_level(args.level)),horizon=args.horizon,samples=args.samples,
        halving=True,min_gap=300,per_card=1,sim_skip=61,cnn_checkpoint=str(args.cnn),value_checkpoint=str(args.value),
        token_weights='seed161 initialization, timing fixtures; no trained v3 checkpoint exists locally',
        timing='one real terminal game per arm; observer.before through step response includes root IPC/features, full teacher search, CNN continuations, simulator step and public observation; excludes model load/reset and CSV I/O',
        matched_probes=args.matched,
        matched_definition='counterfactual holds CNN game states/actions/search fixed, adds only paired native root-time difference; shadow probe time excluded from decision timer',
        records=[])
    original_action,original_search=search._fast_action,search._search
    probe=None
    def action(*a,**kw):
        start=time.perf_counter();out=original_action(*a,**kw);probe.root+=time.perf_counter()-start;return out
    def teacher(*a,**kw):
        start=time.perf_counter();out=original_search(*a,**kw);probe.search+=time.perf_counter()-start;return out
    search._fast_action=action;search._search=teacher
    for arm in ARMS[1:]:
        width=192 if arm=='token192_4' else 128
        torch.manual_seed(161)
        net=StudentV3(width,dict(layers=4 if width==192 else 2,heads=6 if width==192 else 4,
            ff_width=256 if arm=='terrain5' else width*4,ffn='terrain' if arm=='terrain5' else 'dense'),token_input='obs_v3').eval()
        export_student(net,folder/f'{arm}.bin',token_capacity=256)
    for arm in args.arm or ARMS:
        config=folder/f'{arm}.conf'
        config.write_text(f'PVZPOL001\nroot "{arm}.bin"\nrollout "cnn64.bin"\n')
        teacher_args=search.parser().parse_args(['--rollout-policy',str(args.cnn),'--value',values[0],
            '--manifest',str(manifest),'--task-id','task161_bench','--out',str(folder),
            '--workers','1','--executable',str(args.executable),'--resource-dir',str(args.resource_dir),
            '--policy-config',str(config),'--horizon',str(args.horizon),'--samples',str(args.samples)])
        probe=Timings(arm,folder,matched=args.matched and arm=='cnn64')
        search._W['rollout_metrics']={}
        start=time.perf_counter()
        result=search._play(dict(args=teacher_args,seed=args.seed,arm='fast_search'),observer=probe)
        wall=time.perf_counter()-start;probe.save()
        summary=dict(arm=arm,**probe.summary(),whole_call_seconds=wall,games_per_hour_with_startup=3600/wall,
            game={k:result[k] for k in ('terminal','won','decisions','searched','final_tick','final_wave','overrides')})
        if probe.matched:
            summary['games_per_hour_with_startup']=None # whole_call contains explicit shadow probes
        if not result['terminal']: raise RuntimeError(f'{arm} did not reach a terminal state')
        report['records'].append(summary)
        (folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
        search._close_worker_env()


if __name__=='__main__': main()
