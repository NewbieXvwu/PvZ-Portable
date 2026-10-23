"""task55 offline evidence assembly. Reads completed reports and frozen qc/engine facts only."""
from __future__ import annotations
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
MAIN=Path('/home/newbiexvwu/PvZ-search')
ART=ROOT/'artifacts/task55'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import pvz_constants as constants


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build():
    sources={}
    def source(key,path,date,note):
        path=Path(path)
        sources[key]=dict(path=str(path),sha256=sha(path),date=date,note=note)
        return read(path) if path.suffix=='.json' else path.read_text()
    def metric(w,n,task,model,src,mult=1.,**extra):
        assert 0<=w<=n and n>0
        return dict(wins=w,games=n,task=task,model=model,source=src,date=sources[src]['date'],multiplier=mult,**extra)
    engine=source('E',ART/'engine-inventory.json','2026-10-24','one process; 50 reset + one wait tick; not complete games')
    meta=read(ART/'snapshot-meta.json')
    state=source('Q',ART/'curriculum_state.json',meta['captured_at'],'frozen controller membership/window state')
    log=source('QL',ART/'curriculum_log.log',meta['captured_at'],'beta0 collection execution, not teacher strength')
    panel=source('QP',ART/'curriculum_panel.json',meta['captured_at'],'414 multiplier variants; only state.cells are active')
    for key,name in [('Q','curriculum_state'),('QL','curriculum_log'),('QP','curriculum_panel')]:
        sources[key]['original_path']=meta['sources'][name]['path']
    baseline=source('S27',ROOT/'MEMORY/docs/task27-results.json','2026-10-23','completed task27 v2; dw4i production reference, dw5i comparison')
    done=source('SA0',MAIN/'.claude/worktrees/dw5/artifacts/task36/complete_A.json','2026-10-24 00:41:40+0800','latest completed candidate A; not deployed')
    paired=source('SA',MAIN/'.claude/worktrees/dw5/artifacts/task36/paired_dw4i_vs_A.json',done['finished'],'34 paired tasks, 3264 games; raw files rechecked below')
    assert sha(done['model'])==done['model_sha256']
    prod={t['task_id']:metric(t['won'],t['games'],t['task_id'],'dw4i_s1','S27') for t in baseline['v2']['dw4i_s1']['tasks']}
    latest={};raw_hashes={}
    for t in paired['per_task']:
        tid=t['task_id'];folder=Path(done['panel'])/tid
        games=[json.loads(x) for x in (folder/'games.jsonl').read_text().splitlines()]
        assert len(games)==t['paired'] and len({g['seed'] for g in games})==len(games)
        assert all(g['arm']=='fast' and g['searched']==0 for g in games)
        assert sum(g['won'] for g in games)==t['b_won']==read(folder/'summary.json')['fast']['won']
        assert prod[tid]['wins']==t['a_won'] and prod[tid]['games']==t['paired']
        latest[tid]=metric(t['b_won'],t['paired'],tid,'dw5i_A_s1 (not deployed)','SA')
        raw_hashes[tid+'/games.jsonl']=sha(folder/'games.jsonl')
    assert len(latest)==34 and sum(r['games'] for r in latest.values())==3264
    sources['SA']['raw_directory']=done['panel'];sources['SA']['raw_sha256']=raw_hashes
    fog=source('TF',ROOT/'MEMORY/docs/task45-fog-results.json','2026-10-23','cold-start human f8; separate fast and beta1 fast_search')
    roof=source('TR',ROOT/'MEMORY/docs/task46-roof-cold-start.json','2026-10-23','r7 original difficulty; L45 has overridden initial hand')
    roof_diag=source('TR2',MAIN/'.claude/worktrees/roof/MEMORY/docs/task46-roof-timelines.json','2026-10-24','fixed four-seed teacher diagnostic, not a full evaluation panel; L42-45 overlap old panel seeds')
    half=source('TRh',ROOT/'MEMORY/docs/task46-roof-half-results.json','2026-10-23','diagnostic multiplier .5; not current qc nor original completion')
    v4=source('TV4',ROOT/'MEMORY/docs/value-widen-paired.json','2026-10-23','beta1 v4 paired evaluation; new_wins is v4')
    t50=source('T50',MAIN/'.claude/worktrees/zones/artifacts/task50-a/teacher32-v4/comparison.json','2026-10-24','fresh deployed teacher gate; training-task seeds')
    llm=source('T52',MAIN/'.claude/worktrees/llmcand/artifacts/task52/study-v1/summary.json','2026-10-24','144 complete games; arm A is unaugmented beta1 teacher')
    source('D',ROOT/'MEMORY/docs/day-panel.md','2026-10-21','task17 table archived 2026-10-21 00:51+08; old v32 teacher; task20 original L10')
    source('N',ROOT/'MEMORY/docs/night-panel.md','2026-10-22','old identity n9 teacher; NOT current n10')
    source('P',ROOT/'MEMORY/docs/pool-panel.md','2026-10-22','old identity p9k teacher; no modern same-model comparison')
    for key,path,note in [
        ('D43',MAIN/'.claude/worktrees/v5/MEMORY/docs/task43-value-v5-pool.md','first v5 rejected; round2 teacher-continuation labels in progress'),
        ('D45',MAIN/'.claude/worktrees/fog/MEMORY/docs/task45-fog-terminal-diagnosis.md','complete64-root terminal audit; local failure-delay improvements, no wins'),
        ('D46',MAIN/'.claude/worktrees/roof/MEMORY/docs/teacher-diag-roof-review.md','independent phase2 roof diagnosis, not raw LLM conclusions'),
        ('D46c',MAIN/'.claude/worktrees/roof/MEMORY/docs/task46-roof-training-coverage.json','dw4 frozen 48816 archives; no pool/fog/roof archives'),
        ('D46b',MAIN/'.claude/worktrees/roof/MEMORY/docs/task46-roof-terminal-root-smoke.json','only four complete roots; not the full17-root panel'),
        ('D49',ROOT/'MEMORY/docs/task49-A-evidence.json','candidate promotion quality gate; two controlled demonstrations'),
        ('D53',MAIN/'.claude/worktrees/forks/MEMORY/docs/task53-critical-forks.md','screen47 roots complete; independent16-future confirmation7 actions; training/paired evaluation not yet complete'),
        ('D48',ROOT/'MEMORY/docs/task48-human-demo.md','human demonstration tools; ten L31-34 games planned, recordings not evidenced here'),
        ('PLAN',ROOT/'MEMORY/docs/student-bottleneck-plan.md','controller timeline through04:15; qc only packed before task49 pipeline deployment'),
    ]:source(key,path,'2026-10-24',note)
    teachers={};other={}
    def teacher(level,m):teachers[level]=m
    for level in (1,3,4,6,7,8,9):teacher(level,metric(64,64,f'day_l{level}_eval_d7','v32 + search (old)','D'))
    teacher(10,metric(64,64,'day_l10_orig_eval','v34 + search (old)','D'))
    for level,w in [(11,13),(12,11),(13,14),(14,9),(16,11)]:
        other.setdefault(level,[]).append(metric(w,16,f'night_l{level}_eval_n9','identity + search (old)','N',same_family=False))
    for level,w in [(21,10),(22,5),(23,4),(24,4),(26,2)]:teacher(level,metric(w,16,f'pool_l{level}_eval_p9k','identity + search (old)','P'))
    for tid,t in v4['results'].items():
        level=int(re.search(r'_l(\d+)',tid).group(1))
        if level==7:continue # task is m150; preserve as a different-task observation below.
        teacher(level,metric(t['new_wins'],t['pairs'],tid,'dw4i_s1 + v4','TV4'))
    for tid,t in t50.items():
        level=int(re.search(r'_l(\d+)',tid).group(1))
        m=metric(t['wins']['after'],t['games'],tid,'dw4i_s1 + v4','T50',1.5 if level==7 else 1.,seed_scope='training gate')
        if level==7:other.setdefault(level,[]).append(m)
        else:other.setdefault(level,[]).append(teachers.get(level));teacher(level,m)
    for cell in fog['cold']['cells']:
        tid=cell['task_id']
        if not re.fullmatch(r'fog_l\d+_eval_f8',tid):continue
        level=int(re.search(r'_l(\d+)',tid).group(1));m=metric(cell['won'],cell['games'],tid,'dw4i_s1'+(' + v4' if cell['arm']=='fast_search' else ''),'TF')
        if cell['arm']=='fast_search':teacher(level,m)
        else:prod[tid]=m;latest[tid]=m
    for cell in roof['cells']:
        tid=cell['task_id']
        if not tid.endswith('_r7'):continue
        level=cell['level'];m=metric(cell['wins'],cell['games'],tid,'dw4i_s1'+(' + v4' if cell['arm']=='teacher' else ''),'TR',original_conveyor=False)
        if cell['arm']=='teacher':teacher(level,m)
        else:prod[tid]=m;latest[tid]=m
    for family,level in [('day',7),('roof',41),('fog',31)]:
        x=llm['families'][family];c=x['comparisons']['B_low']
        other.setdefault(level,[]).append(teachers.get(level))
        teacher(level,metric(c['before_wins'],c['games'],x['task']['task_id'],'dw4i_s1 + v4','T52'))
    teacher_panels=dict(teachers)
    for level in range(42,50):
        if level==45 or level in (42,43,44,46,47,48,49):
            games=[g for g in roof_diag['games'] if g['level']==level and g['policy']=='teacher']
            assert len(games)==4 and all(g['outcome']['terminal'] for g in games)
            teacher(level,metric(sum(g['outcome']['won'] for g in games),4,f'roof_l{level}_eval_r7','dw4i_s1 + v4','TR2',diagnostic=True,seed_scope='45000000..45000003; overlaps cold panel'))
    registry={}
    paths=[MAIN/'artifacts/stage4'/n for n in ('day_panel.json','night_panel.json','pool_panel.json')]
    paths += [MAIN/'.claude/worktrees/cq/artifacts/task45/fog_panel.json',MAIN/'.claude/worktrees/cq/artifacts/task46/roof_panel.json']
    paths += [MAIN/'.claude/worktrees/conveyor/artifacts/stage4/day_panel.json']
    for i,path in enumerate(paths):
        key=f'R{i}';doc=source(key,path,meta['captured_at'],'registered tasks; does not imply qc admission or measured wins')
        for t in doc['tasks']:registry[t['task_id']]=(t,key)
    qc_last={}
    pattern=re.compile(r'^(\S+) round \d+ (\S+) (\S+) mult=([\d.]+) offset=(\d+) games=(\d+) wins=(\d+).* rc=(\d+)')
    for line_no,line in enumerate(log.splitlines(),1):
        m=pattern.search(line)
        if not m or int(m[8])!=0:continue
        date,label,tid,mult,offset,n,w,rc=m.groups();base=re.sub(r'_m\d+$','',tid)
        qc_last[base]=dict(date=date,label=label,task=tid,multiplier=float(mult),games=int(n),wins=int(w),source='QL',line=line_no,dagger_beta=0)
    factors=[1.]+panel['generation_audit']['multipliers']
    def primary(level,kind):
        if kind in ('bowling','whack','vasebreaker','boss','little_trouble'):return None
        if level<=10:return 'day_l10_orig_eval' if level==10 else f'day_l{level}_eval_d7'
        if level<=20:return 'night_l20_eval_n10_orig' if level==20 else f'night_l{level}_eval_n10'
        if level<=30:return f'pool_l{level}_eval_p9k'
        if level<=40:return f'fog_l{level}_eval_f8'
        return f'roof_l{level}_eval_r7'
    rows=[]
    for fact in engine['levels']:
        level=fact['level'];kinds=fact['source_predicates']
        kind=next((k for k in ('bowling','whack','vasebreaker','bungee','boss','little_trouble','stormy_night','mini_boss') if k in kinds),'ordinary')
        tid=primary(level,kind);entry=registry.get(tid)
        task={k:v for k,v in entry[0].items() if k in ('task_id','deck','conveyor_start','fog_observation','playthrough')} if entry else None
        if task:task['source']=entry[1]
        qc=[dict(base=name,multiplier=factors[c['mult_index']],window_games=sum(b.get('games',0) for b in c['windows'].values()),latest_execution=qc_last.get(name)) for name,c in state['cells'].items() if c['level']==level]
        current=prod.get(tid);recent=latest.get(tid);t=teachers.get(level);teacher_panel=teacher_panels.get(level)
        passed=bool(current and current['games']>=32 and current['wins']/current['games']>=.95)
        cold_teacher=teacher_panel or t
        same_teacher=bool(cold_teacher and cold_teacher['task']==tid and cold_teacher['multiplier']==1.)
        cold=bool(current and same_teacher and current['wins']==0 and cold_teacher['wins']==0)
        if kind in ('bowling','whack','vasebreaker','boss','little_trouble','bungee','stormy_night'):status='未接入'
        elif passed:status='已过'
        elif cold:status='冷启动全败'
        elif qc or current or t:status='在练'
        else:status='未接入'
        if passed:category=None;certainty='observed panel threshold only'
        elif level<=20:category='工程';certainty='training/evaluation gap; not proof of simulator fault'
        elif level in (24,26):category='叶值压平';certainty='v4 dispersion diagnosis covers these levels'
        elif 31<=level<=34:category='叶值压平';certainty='human/xray controls and candidate-gap diagnosis'
        else:category='冷启动无数据';certainty='current-model training coverage or unmeasured level; not a proof of impossibility'
        if kind!='ordinary':category=None
        evidence=[] if category is None else (['S27','SA','PLAN','D49'] if level<=20 else ['TV4','D43'] if level in (24,26) else ['TF','D45','T52'] if 31<=level<=34 else ['D46c','D46','D46b'] if level>=41 else ['D46c','SA','D43'] if level<=29 else ['TF','D45'])
        note=''
        if level in (30,40,45):note='Registered fixed initial-hand task; native conveyor continues. Not original conveyor-start semantics.'
        if level==2:note='No independent modern panel. Shared engine roster/20-wave setup with L1 does not prove identical difficulty or allow copying its wins.'
        if level in (5,15,25,35,50):note='No registered native task family; probe deck/reset is diagnostic only.'
        rows.append(dict(level=level,adventure=f'{(level-1)//10+1}-{(level-1)%10+1}',kind=kind,
                         conveyor=fact['conveyor'],scene=fact['scene'],background=fact['background'],
                         allowed_roster=fact['allowed_roster'],scheduled_types_probe=fact['scheduled_types'],
                         task_family=task,qc=qc,student_latest=recent,student_current=current,teacher_latest=t,
                         teacher_panel=teacher_panel,
                         teacher_other_tasks=[m for m in other.get(level,[]) if m],status=status,
                         bottleneck=category,bottleneck_certainty=certainty,bottleneck_sources=evidence,notes=note))
    result=dict(status='incomplete',as_of=datetime.now().astimezone().isoformat(),qc_as_of=meta['captured_at'],
                status_method='current dw4i original-task panel observed >=95% and n>=32; not population confidence or whole-adventure proof',
                sources=sources,engine_predicates=engine['predicates'],
                engine=dict(executable=engine['executable'],executable_sha256=engine['executable_sha256'],source_sha256=engine['source_sha256'],zombie_dictionary=engine['zombie_dictionary'],reset_and_one_tick_passed=50),
                runtime=dict(student='dw4i_s1',value='v4',static='branch1',dagger_beta=0),
                summary=dict(levels=50,qc_cells=len(state['cells']),qc_levels=sum(bool(r['qc']) for r in rows),statuses=dict(Counter(r['status'] for r in rows))),
                levels=rows,pending=['special action/observation audit','evidence-linked bottleneck notes and priority proposal','completion verification'])
    return result


def table(d):
    kinds=dict(ordinary='普通',bowling='保龄球',whack='打地鼠',vasebreaker='砸罐子',boss='僵王',bungee='蹦极/传送带',little_trouble='小僵尸/传送带',stormy_night='暴风雨/传送带',mini_boss='传送带')
    scenes={0:'白天',1:'夜草',2:'泳池',3:'雾池',4:'屋顶',5:'Boss屋顶'}
    def score(m):return '—' if not m else f"{m['wins']}/{m['games']} [{m['source']}]"+('诊' if m.get('diagnostic') else '')
    def short(tid):
        return 'orig' if tid.startswith('day_l10_orig_') else re.sub(r'^(day|night|pool|fog|roof)_l\d+_(eval|train)_?','',tid)
    lines=['| 关 | 场景/类型 | 僵尸允许表（ID） | 卡组族 | qc当前倍率 | 最新学生 / 现役dw4 | 最近老师 | 状态 / 排查入口 |',
           '|---|---|---|---|---|---|---|---|']
    for r in d['levels']:
        level=r['level'];family=(r['task_family'] or {}).get('task_id','未注册')
        family=short(family) or '初始卡'
        if level in (30,40,45):family+='†'
        qc='；'.join(short(q['base'])+f"×{q['multiplier']:g}" for q in r['qc']) or '不在qc'
        teacher=score(r['teacher_latest'])
        if r['teacher_latest'] and r['teacher_latest'].get('diagnostic') and r['teacher_panel']:
            teacher+='；面'+score(r['teacher_panel'])
        if r['teacher_latest'] and 'old' in r['teacher_latest']['model']:teacher+=' 旧模型'
        if not r['teacher_latest'] and r['teacher_other_tasks']:
            teacher='同族缺证；'+score(r['teacher_other_tasks'][0])+' '+r['teacher_other_tasks'][0]['task'].split('_eval_')[-1]
        label=r['status']+(' / '+r['bottleneck'] if r['bottleneck'] else '')
        roster=','.join(map(str,r['allowed_roster']))
        scene=scenes[r['scene']['terrain']]+('（无雾）' if level==40 else '')
        lines.append(f"| {r['adventure']}（{level}） | {scene}/{kinds[r['kind']]} | {roster} | {family} | {qc} | {score(r['student_latest'])} / {score(r['student_current'])} | {teacher} | {label} |")
    return '\n'.join(lines)+'\n'


def sources_table(d):
    lines=['| 来源ID | 日期（北京时间；旧表为归档日期） | 证据范围 |','|---|---|---|']
    for key,s in d['sources'].items():
        lines.append(f"| [{key}] | {s['date']} | {s['note']} |")
    lines.append('')
    for key,s in d['sources'].items():lines.append(f"[{key}]: {s['path']}")
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    d=build();(ART/'assembled-gap.json').write_text(json.dumps(d,ensure_ascii=False,indent=1)+'\n')
    (ART/'table.md').write_text(table(d))
    print(json.dumps(d['summary'],ensure_ascii=False))
