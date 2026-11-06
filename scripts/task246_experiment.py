"""Run task246's frozen experiment through the existing collectors/trainers/panel.

No production entry point is changed. Completed per-game labels and panel games
resume through their existing drivers.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
from value_teacher_feasibility import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--external-pids',type=int,nargs='*',default=[])
    args=p.parse_args();base=args.data.resolve()
    started=time.time()
    environment=dict(os.environ,TMPDIR=str(base/'work'),PVZ_NO_POPUP='1',OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    conclusion=ROOT/'MEMORY/docs/task246-vn-branch.json'
    report=json.loads(conclusion.read_text())
    report['production_deployed']=False

    def stage(name,script,options):
        report.update(status=name,updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_json(conclusion,report)
        print(name,flush=True)
        with (base/f'{name}.log').open('a') as stream:
            subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,options)],
                env=environment,stdout=stream,stderr=stream,check=True,cwd=ROOT)

    def panel(label,model,relative=0):
        stage(f'panel-{label}','task244_experiment.py',['--data',base,'--phase','panel',
            '--label',label,'--model',model,'--margin-relative',relative,
            '--seed-start','246100000','--workers','4'])

    try:
        plan=json.loads((base/'plan.json').read_text())
        diagnostic=json.loads((base/'l23-plan.json').read_text())
        write_json(base/'generation-plan.json',dict(jobs=plan['jobs']+diagnostic['jobs']))
        stage('generation','task246_collect.py',['--data',base,'--plan',base/'generation-plan.json',
            '--workers','4','--external-pids',*args.external_pids])
        generated=json.loads((base/'generation-plan-report.json').read_text())
        selected={j['game'] for j in plan['jobs']}
        games=[g for g in generated['games'] if g['game'] in selected]
        labels=[]
        for job in plan['jobs']+diagnostic['jobs']:
            for root in (base/'branches'/f"game_{job['game']:04d}").glob('root_*'):
                labels.append(dict(game=job['game'],terrain=job['terrain'],
                    root=json.loads((root/'root.json').read_text()),
                    labels=json.loads((root/'labels.json').read_text())))
        write_json(base/'branch-labels.json',labels)
        report['branch_data']=dict(games=len(games),by_terrain=dict(Counter(g['terrain'] for g in games)),
            roots=sum(len(g['roots']) for g in games),rows=sum(g['rows'] for g in games),
            terminal_labels=sum(r['labels'] for g in games for r in g['roots']),
            generation_seconds=generated['seconds'],games_report=games,
            diagnostic_games=[g for g in generated['games'] if g['game'] not in selected])
        report['episode_data']=json.loads((base/'features/summary.json').read_text())
        if report['episode_data']['failed']:
            raise RuntimeError('episode export failed; inspect its existing summary before training')
        stage('training','task246_train.py',['--data',base])
        offline=json.loads((base/'offline.json').read_text())
        report['offline']=offline
        eligible=[name for name,ok in offline['eligible_for_panel'].items() if ok]
        report['panel']=dict(ran=False,reason='离线排序未达到预先冻结的改善条件')
        report['recommend_production']=False
        if eligible:
            # Choose using validation loss; test ranking is only an entry gate.
            chosen=min(eligible,key=lambda name:offline['training'][name.removesuffix('-t3')]['best_validation'])
            report['selected_model']=chosen
            panel_document=json.loads((base/'panel-original.json').read_text())
            calibration=[]
            from make_multiplier_tasks import MULTIPLIERS
            # MULTIPLIERS holds (value, suffix) pairs above 1.0; the panel also uses the low rungs.
            grid=sorted({0.25,0.5,0.75,1.0}|{float(m) for m,_ in MULTIPLIERS})
            while True:
                panel('production',base/'models/production.pt')
                outcomes=json.loads((base/'panel-production/outcomes.json').read_text())
                changes=[]
                for task in panel_document['tasks']:
                    if task['level'] not in (21,24,27,29):continue
                    rows=[r for r in outcomes if r['level']==task['level']]
                    rate=sum(r['won'] for r in rows)/len(rows)
                    multiplier=task['zombie_count_multiplier']
                    calibration.append(dict(level=task['level'],multiplier=multiplier,wins=sum(r['won'] for r in rows),games=len(rows)))
                    if .2<=rate<=.8:continue
                    index=grid.index(float(multiplier))+(1 if rate>.8 else -1)
                    if not 0<=index<len(grid):
                        raise RuntimeError(f"L{task['level']} cannot reach 20%-80% on the existing multiplier grid")
                    next_multiplier=grid[index]
                    if any(r['level']==task['level'] and r['multiplier']==next_multiplier for r in calibration):
                        raise RuntimeError(f"L{task['level']} calibration straddles 20%-80%; inspect the measured results")
                    changes.append((task,multiplier,next_multiplier))
                write_json(base/'calibration.json',calibration)
                if not changes:break
                for task,old,new in changes:
                    saved=base/'calibration'/f"l{task['level']}-m{old:g}"
                    saved.parent.mkdir(exist_ok=True)
                    shutil.move(base/'panel-production'/f"l{task['level']}",saved)
                    task['zombie_count_multiplier']=new
                write_json(base/'panel-original.json',panel_document)
            panel(chosen,base/'models'/f'{chosen}.pt',.02)
            result=json.loads((base/f'panel-{chosen}'/'report.json').read_text())
            report['panel']=dict(ran=True,model=chosen,temperature=3,margin_relative=.02,
                margin_floor=.001,seed_start=246100000,pairs_per_level=32,
                tasks=panel_document['tasks'],calibration=calibration,result=result)
            report['recommend_production']=result['eligible']
        report['models']={path.name:dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in (base/'models').glob('*.pt')}
        report['status']='完成，未部署'
        report['elapsed_seconds']=time.time()-started
        report['recommendation_reason']=('整局面板全部达到任务243验收条件' if report['recommend_production'] else
            '整局面板未通过' if report['panel']['ran'] else '离线排序未达到整局面板进入条件')
        if report['recommend_production']:
            report['production_steps']=[f"保留生产v5及原门槛作为回滚，备份采集入口的当前配置",
                f"将{report['selected_model']}模型、温度3与相对门槛max(0.001, 0.02×最佳候选值)同时接入采集入口",
                '恢复旧模型与原门槛即可回滚。本任务未执行接入']
        write_json(conclusion,report)
        write_json(base/'final-report.json',report)
        with (ROOT/'MEMORY'/f"{time.strftime('%Y-%m-%d')}.md").open('a') as stream:
            stream.write(f"\n- task246（vnbranch）完成：同根分支{report['branch_data']['games']}局、"
                f"{report['branch_data']['roots']}根、{report['branch_data']['terminal_labels']}条终局标签。"
                f"生产v5热启动与从头混训的离线结果及L23翻转点见 MEMORY/docs/task246-vn-branch.json。"
                f"{report['recommendation_reason']}，未部署。\n")
        for name in ('corpus','features','packed','branches','work'):
            shutil.rmtree(base/name,ignore_errors=True)
        for path in base.glob('*.npz'):path.unlink()
        for path in base.glob('*.log'):path.unlink()
        for path in base.glob('*.tsv'):path.unlink()
        tracked=subprocess.run(['git','ls-files','-z','artifacts','logs'],cwd=ROOT,check=True,capture_output=True).stdout
        ignored=subprocess.run(['git','check-ignore','-z','--stdin','--no-index'],cwd=ROOT,input=tracked,capture_output=True)
        if ignored.stdout:raise RuntimeError('tracked artifacts violate repository ignore rules')
        subprocess.run(['git','add',str(conclusion),str(ROOT/'MEMORY'/f"{time.strftime('%Y-%m-%d')}.md")],check=True,cwd=ROOT)
        subprocess.run(['git','diff','--cached','--numstat'],check=True,cwd=ROOT)
        subprocess.run(['git','diff','--cached','--stat'],check=True,cwd=ROOT)
        subprocess.run(['git','commit','-m','task246: 记录同根分支监督实验结果与离线验收'],check=True,cwd=ROOT)
        subprocess.run(['git','push','origin','vnbranch'],check=True,cwd=ROOT)
        print(json.dumps(dict(status=report['status'],recommend_production=report['recommend_production'])),flush=True)
    except Exception as error:
        report.update(status='实验中断，保留数据待续跑',error=str(error),elapsed_seconds=time.time()-started)
        write_json(conclusion,report)
        raise


if __name__=='__main__':main()
