"""Wait for one verified task53 collector; audit, confirm, export, finetune and pair-evaluate.

Never restarts a collector on timeouts. No qc/session operations or student promotion.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import critical_forks as f
import critical_forks_data as data
OUT=ROOT/'artifacts/task53'


def alive(pid,start):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[0] not in ('Z','X') and fields[19]==start
    except (FileNotFoundError,ProcessLookupError):
        return False


def git(*args):
    return subprocess.run(['git','-C',str(ROOT),*args],capture_output=True,text=True,check=True)


def publish(stage,value,note,extra=()):
    path=ROOT/'MEMORY/docs/task53-critical-forks-progress.json'
    progress=f.load(path) if path.exists() else dict(stages={})
    progress['stages'][stage]=value;f.save(path,progress)
    today=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d');daily=ROOT/f'MEMORY/{today}.md'
    book=ROOT/'MEMORY/docs/task53-critical-forks.md';marker=f'<!-- task53-{stage} -->'
    if marker not in book.read_text():
        book.write_text(book.read_text().replace('## 结果\n',f'## 结果\n\n{marker}\n- {today} {note}\n'))
        with daily.open('a') as stream:stream.write(f'\n- task53 {note}\n')
    paths=['MEMORY/docs/task53-critical-forks-progress.json','MEMORY/docs/task53-critical-forks.md',str(daily.relative_to(ROOT)),*extra]
    if git('diff','--cached','--name-only').stdout.strip():raise RuntimeError('existing staged changes')
    tracked=git('ls-files','-z','artifacts','logs').stdout
    audit=subprocess.run(['git','-C',str(ROOT),'check-ignore','-z','--stdin','--no-index'],input=tracked,capture_output=True,text=True)
    if audit.returncode not in (0,1) or audit.stdout:raise RuntimeError('tracked ignored artifact audit failed')
    if git('branch','--show-current').stdout.strip()!='forks':raise RuntimeError('wrong branch')
    git('add',*paths)
    stat=git('diff','--cached','--numstat').stdout
    if any(line.startswith('-') for line in stat.splitlines()):raise RuntimeError('binary staged')
    if not set(git('diff','--cached','--name-only').stdout.splitlines())<=set(paths):raise RuntimeError('staging changed')
    if stat:print(git('commit','-m',f'task53: {stage} critical forks evidence').stdout,flush=True)
    print(git('push','origin','forks').stderr,flush=True)
    f.save(OUT/f'published-{stage}.json',dict(head=git('rev-parse','HEAD').stdout.strip(),time=time.time()))


def command_metered(command,phase):
    run=OUT/'runs'/f'{int(time.time())}-{phase}';run.mkdir(parents=True)
    started=f.cpu();wall=time.monotonic()
    with (run/f'{phase}.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    f.save(run/'complete.json',dict(cpu_seconds=f.cpu()-started,wall_seconds=time.monotonic()-wall,
                                   exit_code=result.returncode,command=command))
    if result.returncode:raise RuntimeError(f'{phase} failed: {run}')
    return run


def report(plan,by_terrain,confirmation,exported,evaluation):
    from pvz_constants import PLANT_NAME
    cost=f.load(ROOT/'MEMORY/docs/task53-critical-forks-cost.json');spent=f.spent_cpu(OUT)
    raw=f.load(OUT/'screen-summary.json')
    point_costs=[dict(root=g['job']['id'],archive=g['job']['archive'],reasons=g['job']['reasons'],
                     candidates=len(g['candidates']),labels=g['labels'],
                     terminal_and_short_leaf_cpu_seconds=sum(c['cpu_seconds'] for c in g['candidates']))
                 for g in raw['groups']]
    counts={t:dict(roots=by_terrain[t]['roots'],corrections=0,student_corrections=0,action_types={}) for t in by_terrain}
    for c in confirmation['corrections']:
        t=c['job']['terrain'];counts[t]['corrections']+=1;counts[t]['student_corrections']+=c['student_representable']
        label=PLANT_NAME.get(c['plant_type'],c['action']['type'])
        counts[t]['action_types'][label]=counts[t]['action_types'].get(label,0)+1
    results=dict(status='experiment_complete_pending_final_delivery_audit',frozen_inputs=plan['inputs'],
        by_terrain=by_terrain,correction_counts=counts,confirmation=confirmation,
        training=dict(recipe=str(OUT/'training/recipe.json'),recipe_sha256=f.sha(OUT/'training/recipe.json'),
                      model=str(OUT/'training/model/model.pt'),model_sha256=f.sha(OUT/'training/model/model.pt'),
                      new_rows=len(exported['student_corrections']),old_replay_rows=3072),
        paired_evaluation={k:v for k,v in evaluation.items() if k not in ('paired','contract')},
        cost=dict(cpu_seconds_metered=spent,additional_allowance=250,qc_round_cpu_seconds=cost['qc_cpu_seconds'],
                  conservative_share=(spent+250)/cost['qc_cpu_seconds'],reference_engine=cost['qc_reference_engine'],
                  per_point=point_costs,formal_points_per_game=len(plan['jobs'])/len(plan['episodes']),
                  metered_cost_per_formal_point_including_all_pilots_training_and_evaluation=spent/len(plan['jobs']),
                  scope='Per-point branch CPU includes native short-leaf and true-terminal teacher work. Shared replay/startup, off-panel pilot, training and evaluation overhead is retained in the total ledger, not silently omitted.'),
        artifact_bytes=f.artifact_bytes(OUT),value_rows=sum(r['rows'] for r in exported['value_files']),
        value_ranking_pairs=exported['ranking_pairs'],
        raw_evidence=str(OUT),export_sha256=f.sha(OUT/'export.json'))
    if spent+250>plan['cpu_budget_seconds']:raise RuntimeError('final conservative CPU budget exceeded')
    f.save(ROOT/'MEMORY/docs/task53-critical-forks-results.json',results)
    lines=['\n<!-- task53-measured-results -->','\n## 小规模实测结果\n',
       '| 场地 | 局/点 | 候选×未来 | 胜标签 | 确认纠正点/可训点 | 纠正类型 |',
       '|---|---|---|---|---|---|']
    for t,r in by_terrain.items():
        c=counts[t];lines.append(f"| {t} | {r['games']}/{r['roots']} | {r['labels']} | {r['wins']} | {c['corrections']}/{c['student_corrections']} | {c['action_types']} |")
    lines+=['','v4一致率按原档案整局聚类；阈值化全部候选对与真正非终局v4对分开，详细数值见结果JSON。独立确认使用新16未来、配对符号检验及全家族Holm校正。',
       '',f"导出{len(exported['student_corrections'])}条学生纠正、{results['value_rows']}条价值状态行。铲除纠正保留为证据/价值数据，学生没有铲除头，未将它冒充等待。价值kind=0是老师路径的非终局2500tick叶，kind=1是候选执行后的非终局状态；真终局网络旁路状态排除。",
       '','| 面板 | 原学生胜 | 微调胜 | 净差 | 配对95%区间 |', '|---|---|---|---|---|']
    for t,r in evaluation['by_terrain'].items():
        lines.append(f"| {t} | {r['incumbent_wins']}/{r['pairs']} | {r['candidate_wins']}/{r['pairs']} | {r['delta_rate']:+.4f} | {r['ci95']} |")
    lines+=['',f"总CPU计量{spent:.3f}秒，另计250秒余量，占冻结qc 768局参考轮CPU的{results['cost']['conservative_share']:.2%}。参考轮引擎为deploy2；本任务使用deploy3，成本表保留版本区别。产物{results['artifact_bytes']/1e9:.3f}GB。",
       '','## 后台流水线建议\n',
       '每轮768局结束，冻结games/档案、学生与搜索老师SHA，按破防窗口和平坦点选点；对候选数预估费用，用本轮实际采集CPU的15%作硬账本。未知点价时先用最贵点配额，按确定性reservoir选择输局，不能处理所有输局根。',
       '','终局采集在独立4槽后台队列中运行；完整根和独立确认通过后，原子发布export清单。学生纠正档案作为新增不可变label段，confidence进入label_weight；task49 pack→train队列冻结新增清单和权重，继续保留同种子换学生门禁。价值分片按kind分组送task43，保留seed/position/candidate与原始终局证据，不把真终局状态训练成叶值。',
       '','采集、确认或打包中断时保持原学生；预算/5GB门禁停止后不得把未完分叉计败局。正式上线前须补齐统一qc泳池来源与当时部署引擎的混合轮CPU测量。这里的192对是小规模可行性面板，不签发上线证书。主控决定是否上线。',
       '','[完整结果](task53-critical-forks-results.json)、[配对评估](task53-critical-forks-evaluation.json)、[成本登记](task53-critical-forks-cost.json)。\n']
    p=ROOT/'MEMORY/docs/critical-forks.md';old=p.read_text().split('<!-- task53-measured-results -->')[0]
    p.write_text(old+'\n'.join(lines))
    return results


def after_screen(plan):
    groups,by_terrain=data.screen_summary(plan,plan['jobs'])
    publish('screen-complete',by_terrain,
        f"正式47根全合法候选筛查完成并逐条审计，共{sum(r['labels'] for r in by_terrain.values())}标签；继续按预算独立确认，不把筛查最优当纠正。")
    confirmation=data.confirmation_plan(plan,plan['jobs'])
    roots={g['job']['id']:g for g in groups};priced=[]
    for job in confirmation['jobs']:
        group=roots[job['id']]
        # Measure the actual chosen/student branches, not the terrain-average candidate.
        predicted=1.5*job['samples']*sum(c['cpu_seconds']/group['samples'] for c in group['candidates']
                                      if c['index'] in job['indices'])+8
        priced.append((predicted,job))
    available=plan['cpu_budget_seconds']-f.spent_cpu(OUT)-500
    selected=[];excluded=[]
    for predicted,job in sorted(priced,key=lambda item:(item[0],item[1]['id'])):
        if predicted<=available:
            selected.append(job);available-=predicted
        else:
            excluded.append(dict(root=job['id'],forecast_cpu_seconds=predicted))
    confirmation.update(jobs=selected,labels=sum(len(j['indices'])*j['samples'] for j in selected),
                        excluded_by_cpu=excluded,cost_selection='cheapest actual screening challenger/student branch CPU first; preserve 16 independent futures; reserve 500 seconds for exports, training, evaluation and final gates')
    f.save(OUT/'confirmation.json',confirmation)
    f.run(plan,confirmation['jobs'],'confirm')
    confirmed=data.confirm_summary(plan)
    confirmed['selection_budget']=confirmation
    f.save(OUT/'confirmation-results.json',confirmed)
    f.save(ROOT/'MEMORY/docs/task53-critical-forks-confirmation.json',confirmed)
    publish('confirmation-complete',dict(roots=confirmed['root_count'],labels=confirmed['labels'],
            corrections=len(confirmed['corrections'])),
        f"独立16未来确认完成：{confirmed['root_count']}根/{confirmed['labels']}条标签，{len(confirmed['corrections'])}个动作通过全家族统计校正与终局差值门槛。",
        extra=['MEMORY/docs/task53-critical-forks-confirmation.json'])
    exported=data.export(plan,plan['jobs'])
    f.save(ROOT/'MEMORY/docs/task53-critical-forks-export.json',exported)
    publish('export-complete',dict(student_corrections=len(exported['student_corrections']),
            value_rows=sum(r['rows'] for r in exported['value_files'])),
        f"正式导出{len(exported['student_corrections'])}条可训纠正及{sum(r['rows'] for r in exported['value_files'])}条价值状态行；检查点/分片只留ignored产物目录。",
        extra=['MEMORY/docs/task53-critical-forks-export.json'])
    return train_evaluate_report(plan,by_terrain,confirmed,exported)


def train_evaluate_report(plan,by_terrain,confirmed,exported):
    if not exported['student_corrections']:
        f.save(OUT/'pipeline-needs-more-data.json',dict(reason='No independently confirmed student-representable corrections; do not substitute replay-only finetuning.'))
        return
    from pvz_recipe import load_recipe,training_row_options
    from pvz_training_pack import pack_options,pack_flags
    recipe=load_recipe(OUT/'training/recipe.json');options=pack_options(**training_row_options(recipe))
    command_metered([sys.executable,str(ROOT/'scripts/pack_training_rows.py'),'--data',recipe['data'],
        '--labels',','.join(recipe['labels']),'--cache-dir',recipe['cache_dir'],'--precompute-workers','1',*pack_flags(options)],'pack')
    command_metered([sys.executable,str(ROOT/'scripts/train_rollout_policy.py'),'--recipe',str(OUT/'training/recipe.json')],'finetune')
    model=OUT/'training/model/model.pt'
    if not model.exists():raise RuntimeError('finetune checkpoint absent')
    publish('finetune-complete',dict(model_sha256=f.sha(model),new_rows=len(exported['student_corrections'])),
        '固定配方小规模微调完成，模型只留本机ignored目录；继续192种子配对评估。')
    from task53_evaluate import run as evaluate
    evaluation=evaluate(plan,model)
    results=report(plan,by_terrain,confirmed,exported,evaluation)
    publish('experiment-complete',dict(cost=results['cost'],paired=evaluation['overall'],artifact_bytes=results['artifact_bytes']),
        f"小规模可行性实验完成：192对评估净胜率差{evaluation['overall']['delta_rate']:+.4f}，完整分场地结果、区间与接入建议见critical-forks报告。尚待最终门禁/实际复现验收，不上线、不切qc。",
        extra=['MEMORY/docs/task53-critical-forks-results.json','MEMORY/docs/task53-critical-forks-evaluation.json','MEMORY/docs/critical-forks.md'])
    f.save(OUT/'pipeline-complete.json',dict(head=git('rev-parse','HEAD').stdout.strip(),time=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=Path)
    p.add_argument('stage',nargs='?',choices=['run','resume-training'],default='run');a=p.parse_args()
    plan=f.check_plan(a.plan)
    with (OUT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'pipeline-complete.json').exists():raise RuntimeError('pipeline already completed')
        import os
        start=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
        f.save(OUT/'pipeline-controller.json',dict(pid=os.getpid(),start_ticks=start,
               plan=str(a.plan),source_sha256=f.sha(__file__)))
        if a.stage=='resume-training':
            if not (OUT/'published-export-complete.json').exists():raise RuntimeError('complete audited export required')
            _,by_terrain=data.screen_summary(plan,plan['jobs']);confirmed=data.confirm_summary(plan)
            exported=f.load(OUT/'export.json')
            if exported['confirmation_sha256']!=f.sha(OUT/'confirmation-results.json'):
                raise RuntimeError('confirmation/export evidence changed')
            return train_evaluate_report(plan,by_terrain,confirmed,exported)
        watch=f.load(OUT/'watch-screen.json');run=Path(watch['run'])
        while alive(watch['pid'],watch['start_ticks']):
            print('WAIT VERIFIED SCREEN',watch['pid'],f.spent_cpu(OUT),flush=True);time.sleep(20)
        if not (run/'complete.json').exists():raise RuntimeError('screen ended without complete run; never restart on a timeout')
        after_screen(plan)


if __name__=='__main__':main()
