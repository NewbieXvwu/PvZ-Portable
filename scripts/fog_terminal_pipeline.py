"""Resume task45 E after one verified live screening controller, then audit and publish stages.

The watcher never restarts screening. Confirmation requires the full frozen panel audit and
its remaining-CPU gate. Only this task's conclusion files may be staged automatically.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import fog_terminal_diagnostic as diag
import fog_terminal_report as report
OUT=diag.OUT


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if fields[0] in ('Z','X'):return None
        return fields[19]
    except (FileNotFoundError,ProcessLookupError):return None


def process_alive(pid,start):return process_identity(pid)==start


def git(*args,check=True):
    return subprocess.run(['git','-C',str(ROOT),*args],capture_output=True,text=True,check=check)


def publish(stage,value,text,extra=()):
    marker=f'追加E-{stage}'
    today=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')
    daily=ROOT/f'MEMORY/{today}.md'
    progress_path=ROOT/'MEMORY/docs/task45-fog-terminal-progress.json'
    progress=diag.load(progress_path) if progress_path.exists() else {'stages':{}}
    progress['stages'][stage]=value
    diag.save(progress_path,progress)
    book=ROOT/'MEMORY/docs/task45-fog-zone.md'
    if f'<!-- {marker} -->' not in book.read_text():
        book.write_text(book.read_text().replace('## 结果\n',f'## 结果\n\n<!-- {marker} -->\n- {today} {text}\n'))
        with daily.open('a') as f:f.write(f'\n- task45 {text}\n')
    paths=['MEMORY/docs/task45-fog-terminal-progress.json','MEMORY/docs/task45-fog-zone.md',str(daily.relative_to(ROOT)),*extra]
    staged=git('diff','--cached','--name-only').stdout.splitlines()
    if staged:raise RuntimeError('existing staged files; reports saved, automatic commit stopped')
    if git('branch','--show-current').stdout.strip()!='fog':raise RuntimeError('not on fog branch')
    tracked=git('ls-files','-z','artifacts','logs').stdout
    check=subprocess.run(['git','-C',str(ROOT),'check-ignore','-z','--stdin','--no-index'],
                         input=tracked,capture_output=True,text=True)
    if check.returncode not in (0,1):raise RuntimeError('artifact ignore audit failed')
    if check.stdout:raise RuntimeError('tracked ignored artifacts found; automatic commit stopped')
    git('add',*paths)
    if not set(git('diff','--cached','--name-only').stdout.splitlines())<=set(paths):
        raise RuntimeError('staging scope changed')
    numstat=git('diff','--cached','--numstat').stdout
    if any(line.startswith('-') for line in numstat.splitlines()):raise RuntimeError('binary staged unexpectedly')
    if numstat:
        print(git('diff','--cached','--stat').stdout,flush=True)
        print(git('commit','-m',f'task45: {stage} full-candidate terminal evidence').stdout,flush=True)
    print(git('push','origin','fog').stderr,flush=True)
    diag.save(OUT/f'published-{stage}.json',dict(head=git('rev-parse','HEAD').stdout.strip(),time=time.time()))


def cell_table(screen):
    lines=['| 关/观测 | 根 | 候选×未来 | 胜标签 | 末波可分根 | v4末波一致率/对数 | v4终tick一致率/对数 |',
           '|---|---|---|---|---|---|---|']
    for name,data in screen['by_level_observation'].items():
        wave=data['metrics']['wave']['v4'];tick=data['metrics']['tick']['v4']
        fmt=lambda v:'NA' if v is None else f'{v:.3f}'
        lines.append(f"| {name} | {data['roots']} | {data['labels']} | {data['terminal_wins']} | {data['separable_roots']['wave']} | {fmt(wave['accuracy'])}/{wave['pairs']} | {fmt(tick['accuracy'])}/{tick['pairs']} |")
    return '\n'.join(lines)


def conclusions(final):
    from pvz_constants import PLANT_NAME
    overall=final['overall'];sep=overall['separable_roots'];n=overall['roots']
    lines=[f"终局结果在{sep['wave']}/{n}根上能分出平均末波≥1的差异，在{sep['tick']}/{n}根上能分出平均终tick≥600的差异；{sep['won']}/{n}根有胜率差≥1/3。"]
    for metric,label in [('won','胜率'),('wave','末波'),('tick','终tick')]:
        m=overall['metrics'][metric]['v4']
        value='NA（无合格排序对）' if m['accuracy'] is None else f"{m['accuracy']:.3f}（{m['pairs']}对，按reset seed聚类95%区间{m['cluster_bootstrap_95']}）"
        lines.append(f'正式v4叶值对{label}的阈值化一致率：{value}。')
    counts=overall['available_root_types']
    lines.append('根候选覆盖：'+ '、'.join(f"{PLANT_NAME[k]} {counts.get(str(k),0)}根" for k in (0,5,43,3))+ '。没有合法候选的植物，本实验不能判断其当步优劣。')
    confirmed=final['confirmation'].get('significant_better',[])
    if confirmed:
        lines.append('独立未来确认的改善例子（完整家族见JSON）：')
        ordered=sorted(confirmed,key=lambda t:({'wave':0,'won':1,'tick':2}[t['metric']],t['p_holm']))
        for t in ordered[:6]:
            name=PLANT_NAME.get(t['plant_type'],'等待');a=t['action']
            location=f"({a['row']},{a['col']})" if a['type']=='plant' else ''
            lines.append(f"- {t['cell']}，0起决策索引{t['root']}，{name}{location}对学生：{t['metric']}平均差{t['mean_gain']:.3f}，Holm p={t['p_holm']:.5g}。")
    else:
        lines.append('独立确认未证明任何所测候选在Holm校正后显著更好；筛查最优不能作为显著改进。')
    return '\n\n'.join(lines)+'\n\n'


def after_screen():
    # A process ending is not proof of success: require complete.json and exact raw coverage.
    current=diag.load(OUT/'watch-screen.json');run=Path(current['run'])
    if not (run/'complete.json').is_file():raise RuntimeError('screen controller ended without complete raw run')
    screening=report.screen()
    compact={k:v for k,v in screening.items() if k not in ('groups','sampling')}
    compact['cpu_seconds_metered']=diag.spent_cpu()
    compact['source_report_sha256']=diag.sha(OUT/'screen-report.json')
    publish('screen-complete',compact,
        f"追加E筛查完成：32学生轨迹/64冻结根，{screening['overall']['labels']}条全合法候选×3共同未来真实老师终局标签全部核验；每根候选/未来精确覆盖，分叉前物理帧、学生历史、收入历史及完整动作链通过。累计{diag.spent_cpu()/3600:.3f} CPU小时，尚未把筛选最优当显著改进；继续按预算门禁作独立16未来确认。分区数值见终局progress JSON。")
    confirmation=report.confirmation_plan()
    if confirmation['approved_by_predeclared_budget']:
        print('CONFIRM',confirmation['labels'],'labels',flush=True)
        diag.run_jobs('confirm',confirmation['jobs'])
    else:print('CONFIRM NOT RUN: remaining CPU gate',flush=True)
    final=report.final_report()
    screening=diag.load(OUT/'screen-report.json')
    confirmation=final['confirmation']
    significant=confirmation.get('significant_better',[])
    note=(f"独立16未来确认找到{len(significant)}项Holm校正后p≤0.05且均值改善的候选/指标"
          if confirmation['status']=='complete' else '独立确认因剩余CPU门禁未运行，筛查不作显著性结论')
    summary=ROOT/'MEMORY/docs/task45-fog-terminal-diagnosis.md'
    summary.write_text('''# task45追加E：全合法候选接真实老师终局\n\n'''+
        f"筛查32条学生轨迹、64根、{final['overall']['labels']}条候选×未来标签。全部真实终局，根决策范围0–8000tick；每关/观测4个reset seed，初始合法根与最接近8000tick合法根。全候选含等待，无静态/top-k筛除。\n\n"+
        conclusions(final)+cell_table(screening)+'\n\n'+
        f"{note}。同一根的各候选共享世界与搜索噪声种子；跨观测臂只配对reset seed，不把臂间差异解释为严格同未来的因果效应。\n\n"+
        'v4排序对照为正式的2500tick学生续走叶值；真实终局路径从候选立即接冻结搜索老师。另报老师路径2500tick附近的v4值，区分续走分布与估值误差。所有叶中途已终局时的真值评分与真实非终局v4排序对分开报告。\n\n'+
        '成对一致率分胜率差≥1/3、平均末波差≥1、平均终tick差≥600三把尺子；另报全部严格不等对。真值平局不计，v4平局计半分。95%区间按关卡/reset seed整组bootstrap，human/xray和同局各根一起重采样，不能拿候选对数充当独立样本。筛查只有3未来，其最优存在选择偏差；显著改进只认新16未来、配对符号检验及全家族Holm校正。末波提高不等同存活时间变长。\n\n'+
        f"CPU计量{final['cpu_seconds_metered']/3600:.3f}小时，另保守计入480秒未独立计量的准备/测试/报告开销，共{(final['cpu_seconds_metered']+480)/3600:.3f}小时；自有产物{final['artifact_bytes']/1e9:.3f}GB。四槽结构与实际进程采样限额均在run回执。\n\n"+
        '完整分区一致率、聚类区间、各根学生/候选/v4最优以及独立确认的逐未来差异见 [结果JSON](task45-fog-terminal-results.json)。原始逐局目录、输入SHA与冻结方案也在结果中，动作链在ignored的artifacts/task45/phase-e/episodes；本阶段不训练、不换默认、不启动课程或qc。\n')
    publish('complete',dict(status=final['status'],overall=final['overall'],confirmation_status=confirmation['status'],
        significant_confirmed_tests=len(significant),cpu_seconds_metered=final['cpu_seconds_metered'],
        conservative_cpu_hours=(final['cpu_seconds_metered']+480)/3600,artifact_bytes=final['artifact_bytes']),
        f"追加E终局诊断交付：{note}；全量分区排序、逐根最优及确认结果见 [终局诊断](task45-fog-terminal-diagnosis.md) 和 [结果JSON](task45-fog-terminal-results.json)。计量+480秒保守开销共{(final['cpu_seconds_metered']+480)/3600:.3f}CPU小时≤20，自有产物{final['artifact_bytes']/1e9:.3f}GB。原始输入/每局动作链/终局标签可复现，未改v5或生产策略。",
        extra=['MEMORY/docs/task45-fog-terminal-results.json','MEMORY/docs/task45-fog-terminal-diagnosis.md'])
    diag.save(OUT/'pipeline-complete.json',dict(head=git('rev-parse','HEAD').stdout.strip(),time=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pid',type=int,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    watch=OUT/'watch-screen.json'
    if watch.exists():
        info=diag.load(watch)
        if info['pid']!=a.pid or info['run']!=str(a.run):raise ValueError('different screening handle; do not restart')
    else:
        start=process_identity(a.pid)
        argv=Path(f'/proc/{a.pid}/cmdline').read_bytes().split(b'\0')
        if start is None or not any(v.endswith(b'/fog_terminal_diagnostic.py') for v in argv) or b'screen' not in argv:
            raise RuntimeError('screening controller is not verified live')
        if diag.load(a.run/'controller.json')['pid']!=a.pid:raise ValueError('controller receipt/PID mismatch')
        info=dict(pid=a.pid,start=start,run=str(a.run));diag.save(watch,info)
    while process_alive(info['pid'],info['start']):
        print('WAIT LIVE SCREEN',a.pid,round(diag.spent_cpu()/3600,4),'CPUh',flush=True)
        time.sleep(20)
    after_screen()

if __name__=='__main__':main()
