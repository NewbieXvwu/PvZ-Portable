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
from day_orig_tasks import day_original_tasks


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build():
    sources={}
    def source(key,path,date,note):
        path=Path(path)
        data=path.read_bytes()
        sources[key]=dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),date=date,note=note,captured_at=datetime.now().astimezone().isoformat())
        if not path.is_relative_to(ART):
            snapshot=ART/'sources'/f'{key}{path.suffix}'
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            snapshot.write_bytes(data)
            sources[key]['snapshot']=str(snapshot)
        return json.loads(data) if path.suffix=='.json' else data.decode()
    def metric(w,n,task,model,src,mult=1.,**extra):
        assert 0<=w<=n and n>0
        return dict(wins=w,games=n,task=task,model=model,source=src,date=sources[src]['date'],multiplier=mult,**extra)
    engine=source('E',ART/'engine-inventory.json','2026-10-24','one process; 50 reset + one wait tick; not complete games')
    meta=read(ART/'snapshot-meta.json')
    state=source('Q',ART/'curriculum_state.json',meta['captured_at'],'frozen controller membership/window state')
    log=source('QL',ART/'curriculum_log.log',meta['captured_at'],'beta0 collection execution, not teacher strength')
    panel=source('QP',ART/'curriculum_panel.json',meta['captured_at'],'414 multiplier variants; only state.cells are active')
    runtime=source('RUN',MAIN/'artifacts/hard/c3d9d7qm100_48/config.json',meta['captured_at'],'configuration of the group running at the qc snapshot; beta0 actual execution')
    assert runtime['dagger_beta']=='0.0' and runtime['rollout_policy'].endswith('/dw4i_s1/model.pt')
    for key,name in [('Q','curriculum_state'),('QL','curriculum_log'),('QP','curriculum_panel')]:
        sources[key]['original_path']=meta['sources'][name]['path']
    baseline=source('S27',ROOT/'MEMORY/docs/task27-results.json','2026-10-23','completed task27 v2; dw4i production reference, dw5i comparison')
    done=source('SA0',MAIN/'artifacts/archive/dw5/task36/complete_A.json','2026-10-24 00:41:40+0800','latest completed candidate A; not deployed')
    paired=source('SA',MAIN/'artifacts/archive/dw5/task36/paired_dw4i_vs_A.json',done['finished'],'34 paired tasks, 3264 games; raw files rechecked below')
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
    roof_diag=source('TR2',MAIN/'artifacts/archive/roof/MEMORY/docs/task46-roof-timelines.json','2026-10-24','fixed four-seed teacher diagnostic, not a full evaluation panel; L42-45 overlap old panel seeds')
    half=source('TRh',ROOT/'MEMORY/docs/task46-roof-half-results.json','2026-10-23','diagnostic multiplier .5; not current qc nor original completion')
    v4=source('TV4',ROOT/'MEMORY/docs/value-widen-paired.json','2026-10-23','beta1 v4 paired evaluation; new_wins is v4')
    dispersion=source('D30',ROOT/'MEMORY/docs/value-widen-dispersion.json','2026-10-23 12:20+0800','L24/L26 each4 games; all480 search decisions per arm, not only sampled200')
    source('D30d',ROOT/'MEMORY/docs/value-widen.md','2026-10-23 12:20+0800','step5-b: eight games per arm all lost; specific L24/L26 dispersion evidence')
    t50=source('T50',MAIN/'artifacts/archive/zones/task50-a/teacher32-v4/comparison.json','2026-10-24','fresh deployed teacher gate; training-task seeds')
    llm=source('T52',MAIN/'artifacts/archive/llmcand/task52/study-v1/summary.json','2026-10-24','144 complete games; arm A is unaugmented beta1 teacher')
    source('D',ROOT/'MEMORY/docs/day-panel.md','2026-10-21','task17 table archived 2026-10-21 00:51+08; old v32 teacher; task20 original L10')
    source('N',ROOT/'MEMORY/docs/night-panel.md','2026-10-22','old identity n9 teacher; NOT current n10')
    source('P',ROOT/'MEMORY/docs/pool-panel.md','2026-10-22','old identity p9k teacher; no modern same-model comparison')
    for key,path,note in [
        ('D43',MAIN/'artifacts/archive/v5/MEMORY/docs/task43-value-v5-pool.md','first v5 rejected; round2 teacher-continuation labels in progress'),
        ('D45',MAIN/'artifacts/archive/fog/MEMORY/docs/task45-fog-terminal-diagnosis.md','complete64-root terminal audit; local failure-delay improvements, no wins'),
        ('D46',MAIN/'artifacts/archive/roof/MEMORY/docs/teacher-diag-roof-review.md','independent phase2 roof diagnosis, not raw LLM conclusions'),
        ('D46c',MAIN/'artifacts/archive/roof/MEMORY/docs/task46-roof-training-coverage.json','dw4 frozen 48816 archives; no pool/fog/roof archives'),
        ('D46b',MAIN/'artifacts/archive/roof/MEMORY/docs/task46-roof-terminal-root-smoke.json','only four complete roots; not the full17-root panel'),
        ('D49',ROOT/'MEMORY/docs/task49-A-evidence.json','candidate promotion quality gate; two controlled demonstrations'),
        ('D53',MAIN/'artifacts/archive/forks/MEMORY/docs/task53-critical-forks.md','screen47 roots complete; independent16-future confirmation7 actions; training/paired evaluation not yet complete'),
        ('D48',ROOT/'MEMORY/docs/task48-human-demo.md','human demonstration tools; ten L31-34 games planned, recordings not evidenced here'),
        ('H48',ART/'human-demo-plan.md','frozen origin/demo 206cc403ec98ef292ecde601303de1ca7bc200a9:MEMORY/docs/human-demo.md; plan, not actual recordings'),
        ('D53c',MAIN/'artifacts/archive/forks/MEMORY/docs/task53-critical-forks-deploy3-cost.json','48-game current-engine night projection only, not a complete mixed-terrain round'),
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
    for level in (24,26):
        tid=f'pool_l{level}_eval_p9k'
        teacher(level,metric(0,dispersion['per_level']['v4'][tid]['games'],tid,'dw4i_s1 + v4','D30d',diagnostic=True,diagnostic_kind='leaf dispersion probe'))
    for level in range(42,50):
        games=[g for g in roof_diag['games'] if g['level']==level and g['policy']=='teacher']
        assert len(games)==4 and all(g['outcome']['terminal'] for g in games)
        teacher(level,metric(sum(g['outcome']['won'] for g in games),4,f'roof_l{level}_eval_r7','dw4i_s1 + v4','TR2',diagnostic=True,seed_scope='45000000..45000003; overlaps cold panel'))
    registry={}
    paths=[MAIN/'artifacts/stage4'/n for n in ('day_panel.json','night_panel.json','pool_panel.json')]
    paths += [MAIN/'.claude/worktrees/cq/artifacts/task45/fog_panel.json',MAIN/'.claude/worktrees/cq/artifacts/task46/roof_panel.json']
    for i,path in enumerate(paths):
        key=f'R{i}';doc=source(key,path,meta['captured_at'],'registered tasks; does not imply qc admission or measured wins')
        for t in doc['tasks']:registry[t['task_id']]=(t,key)
    sources['R5'] = dict(path='scripts/day_orig_tasks.py:day_original_tasks',
                         date=meta['captured_at'], note='generated original day tasks')
    for task_id, task in day_original_tasks().items():
        registry[task_id] = (task, 'R5')
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
        elif level in (24,26):category='叶值压平';certainty='direct task30 L24/L26 dispersion probe; not a proof of sole cause'
        elif 31<=level<=34:category='叶值压平';certainty='human/xray controls and candidate-gap diagnosis'
        else:category='冷启动无数据';certainty='current-model training coverage or unmeasured level; not a proof of impossibility'
        if kind!='ordinary':category=None
        evidence=[] if category is None else (['S27','SA','PLAN','D49'] if level<=20 else ['D30','D30d','TV4','D43'] if level in (24,26) else ['TF','D45','T52'] if 31<=level<=34 else ['D46c','D46','D46b'] if level>=41 else ['D46c','SA','D43'] if level<=29 else ['TF','D45'])
        note=''
        if level in (30,40,45):note='Registered fixed initial-hand task; native conveyor continues. Not original conveyor-start semantics.'
        if level==2:note='No independent modern panel. Shared engine roster/20-wave setup with L1 does not prove identical difficulty or allow copying its wins.'
        if level in (5,15,25,35,50):note='No registered native task family; probe deck/reset is diagnostic only.'
        scripted={35:dict(types=[0,4,7,8,15],note='three Adventure vase stages; Dancer can summon9; Board allowed roster is not the pot roster',code='src/Lawn/Challenge.cpp:3801'),45:dict(types=[20],note='scripted bungee carrier, additional to allowed ground roster',code='src/Lawn/Board.cpp:4800'),50:dict(types=[25],note='scripted Boss, additional to ordinary spawn definitions',code='src/Lawn/Zombie.cpp:10169')}.get(level)
        rows.append(dict(level=level,adventure=f'{(level-1)//10+1}-{(level-1)%10+1}',kind=kind,
                         conveyor=fact['conveyor'],scene=fact['scene'],background=fact['background'],
                         allowed_roster=fact['allowed_roster'],scheduled_types_probe=fact['scheduled_types'],
                         scripted_roster_supplement=scripted,
                         task_family=task,qc=qc,student_latest=recent,student_current=current,teacher_latest=t,
                         teacher_panel=teacher_panel,
                         teacher_other_tasks=[m for m in other.get(level,[]) if m],status=status,
                         bottleneck=category,bottleneck_certainty=certainty,bottleneck_sources=evidence,notes=note))
    result=dict(status='incomplete',as_of=datetime.now().astimezone().isoformat(),qc_as_of=meta['captured_at'],
                status_method='current dw4i original-task panel observed >=95% and n>=32; not population confidence or whole-adventure proof',
                sources=sources,engine_predicates=engine['predicates'],
                engine=dict(executable=engine['executable'],executable_sha256=engine['executable_sha256'],source_sha256=engine['source_sha256'],zombie_dictionary=engine['zombie_dictionary'],reset_and_one_tick_passed=50),
                runtime=dict(student=runtime['rollout_policy'],value=runtime['value'],static=runtime['static_value'],dagger_beta=0,source='RUN'),
                summary=dict(levels=50,qc_cells=len(state['cells']),qc_levels=sum(bool(r['qc']) for r in rows),statuses=dict(Counter(r['status'] for r in rows))),
                levels=rows,pending=['special action/observation audit','evidence-linked bottleneck notes and priority proposal','completion verification'])
    result.update(analysis_sections())
    result['audit_code_sha256']={p:sha(ROOT/p) for p in sorted({loc.rsplit(':',1)[0] for s in result['special_mechanisms'] for loc in s['code']})}
    probes=source('EP',ART/'protocol-probes.json','2026-10-24','two single actions, L5 and L50 accepted; no full games')
    result['protocol_probes']=probes
    return result


def analysis_sections():
    """Evidence-based audit conclusions; no gameplay policy or implementation changes."""
    special=[
        dict(kind='mini_boss',levels=[10,20,30],headless='L10/L20有完整老师结果；L30仅reset+1tick，不证明终局链',actions_sufficient='plant/shovel/wait足以表达出牌、落点、等待；L20学生0/128但v4老师29/32',observation_gaps='现有packets含重复牌/空槽/冷却，需保留原生出牌与槽位语义；L30当前p9k覆盖初始手牌，原生开局族尚缺',effort='小',scope='协议盘点；学习与面板验收另计',sources=['D','TV4','R2','R5'],code=['src/Lawn/Board.cpp:8577','src/LawnApp.cpp:1407','src/Lawn/Challenge.cpp:1649']),
        dict(kind='little_trouble',levels=[25],headless='reset+1tick通过，未跑完整关',actions_sufficient='已有种植可选择传送带卡与水陆格，无证据需要新动作',observation_gaps='核对小僵尸的可见尺寸/碰撞及编码；生命值虽公开，当前编码不显式标识此变体；需原生传送带任务与数据',effort='中',scope='原生牌为睡莲/坚果/豌豆/樱桃；不可套普通p9k手牌',sources=['E'],code=['src/LawnApp.cpp:3122','src/Lawn/Challenge.cpp:1781','src/Lawn/Zombie.cpp:871']),
        dict(kind='stormy_night',levels=[40],headless='reset+1tick通过，未跑完整关',actions_sufficient='种植/等待可表达牌与格，不需要新增点选种类',observation_gaps='StageHasFog=false，human fog屏蔽不覆盖暴风雨黑屏/闪电；缺与可见光照一致的观测及闪电阶段输入，不能声称人类等价可见性',effort='中',scope='原生牌含睡莲/仙人掌/豌豆/小喷菇/樱桃；当前f8覆盖初始手牌，另立原生族才可比较',sources=['E','R3'],code=['src/Lawn/Board.cpp:8710','src/Lawn/Challenge.cpp:1965','src/LawnApp.cpp:1643']),
        dict(kind='bowling',levels=[5],headless='原生reset+1tick通过；packet0,col0,row0种坚果被接受，未完成一局',actions_sufficient='现有plant已经能选行并在合法c0-c2放球；不必先加专用bowling动作',observation_gaps='滚动植物mX/mY及弹跳轨迹未在完整plants公开；只有初始col、当前row/state不足以完整表示连续运动，学生编码需审查',effort='中',scope='动作部分小；轨迹/编码/新族/面板合计中，按用户要求后置',sources=['E','EP'],code=['src/Lawn/Challenge.cpp:2453','src/Lawn/Plant.cpp:2361','src/LawnApp.cpp:2063']),
        dict(kind='whack',levels=[15],headless='reset+1tick通过；当前deck覆盖原生3槽，不是原样原生reset',actions_sufficient='不够：缺锤子点击僵尸；plant不能调用MouseDownWhackAZombie',observation_gaps='需原生锤子/点击目标/节奏与原生三槽语义；当前观测不是已验收的打地鼠输入',effort='中',scope='用户明确不做；只记录缺口，不建议排期',sources=['E'],code=['src/Lawn/Challenge.cpp:1329','src/Lawn/Board.cpp:8590','python/pvz_env.py:536']),
        dict(kind='vasebreaker',levels=[35],headless='reset+1tick通过且有罐；当前reset覆盖原生1槽，完整玩法未验证',actions_sufficient='不够：缺点罐、拾取/使用掉落植物卡；种植只从seedbank取卡',observation_gaps='grid_items直接公开封闭罐的zombie_type/seed_type，存在信息泄漏；需屏蔽隐藏内容并保留合法可见提示。coins只给type/x/y，缺掉落卡mUsableSeedType与使用状态；还需原生bank及点击合法动作',effort='大',scope='新交互/可见性/编码/动作头与数据；按用户要求后置',sources=['E'],code=['src/Lawn/Challenge.cpp:1346','src/Lawn/Challenge.cpp:4057','src/LawnApp.cpp:2099','src/LawnApp.cpp:2133','src/Lawn/Board.cpp:8590']),
        dict(kind='bungee',levels=[45],headless='原生reset+1tick通过；现有完整r7诊断/面板是初始手牌变体',actions_sufficient='基本够：传送带种植与铲子/等待可防守空投，没有必须点击蹦极的新动作',observation_gaps='已有generic phase/target_row/target_col及植物bungee_state，须核对编码是否保留可见空投落点、被抓植物与高度阶段；原生初始语义及数据缺口未补',effort='中',scope='原生牌花盆/大嘴花/南瓜/樱桃；r7面板不能宣称原生开局通过，按用户要求后置',sources=['E','TR','TR2','D46'],code=['src/Lawn/Board.cpp:4800','src/Lawn/Challenge.cpp:1807','src/LawnApp.cpp:2088']),
        dict(kind='boss',levels=[50],headless='原生reset+1tick通过；首张卷心菜种植接受，尚无完整Boss终局证据',actions_sufficient='基础种植/等待可表达原生牌使用；未发现必须新增瞄准动作的证据，仍需完整链验证',observation_gaps='通用僵尸血量/phase不充分代表Boss露头与冰火球；mBossMode/mBossHeadCounter/mBossFireBallReanimID未序列化，需从中提取人类可见状态而非盲目暴露隐藏计时。还缺传送带、终局判定及编码/数据验收',effort='大',scope='原生6种牌没有玉米加农炮；不能把玉米炮瞄准协议缺口当L50必备，按用户要求后置',sources=['E','EP'],code=['src/Lawn/Challenge.cpp:1741','src/Lawn/Zombie.h:150','src/Lawn/Zombie.cpp:10169','src/LawnApp.cpp:2088']),
        dict(kind='cob_cannon',levels=[],headless='非独立冒险关类型；本次未做玉米炮交互探针',actions_sufficient='不够：需选择ready炮后指定像素落点；plant/shovel/wait均不能触发该点击链',observation_gaps='需炮ID/ready/选中状态/落点的可见输入与动作头，并保留30tick误触保护语义；不影响当前无炮任务',effort='中',scope='跨关机制；若含搜索、回放、动作头/编码验收，整体可升大；不作为L50前置',sources=['E'],code=['src/Lawn/Plant.cpp:4207','src/Lawn/Board.cpp:3581','python/pvz_env.py:536']),
    ]
    groups=[
        dict(levels=[1,2,3,7,8,9,12,14,17,18,19],category='工程',confidence='排查入口，非已证实模拟器错误',evidence='现役面板有缺口；L14老师32/32而学生87/128，L17老师31/32而学生65/128；旧qc只打包未训练、流水线刚上线；A改善泳池却伤夜间/heldout，不能直接换。L2仅缺独立现代证据。',sources=['S27','SA','T50','TV4','PLAN','D49'],needed='确认新纠正数据实际进入训练/质量门禁，并补同族评估；不要把缺证当输局或改引擎'),
        dict(levels=[21,22,23,27,28,29],category='冷启动无数据',confidence='现役训练来源缺口有直接证据；27-29正式面板缺证',evidence='dw4冻结48816档案只含白天/夜草；qc历史pool窗口不等于dw4已经学到。A在21-23为42/64、33/64、27/64，说明数据/配比有潜力，但仍非稳定通关。',sources=['D46c','SA','Q','D53'],needed='接task43老师终局信号与task53纠正，完成质量门禁；27-29先取得正式基线'),
        dict(levels=[24,26],category='叶值压平',confidence='task30直接覆盖两关的候选离散度实证，非唯一路径',evidence='现役各0/64；v4 L24正式老师3/32。task30另对L24/L26各4局、每臂480决策做探针，v4候选std约.0007，100% gap落入.02门槛且8局0推翻、两臂均0胜。task43第一轮v5已否决，第二轮真实老师续走标签尚在采集。',sources=['D30','D30d','S27','TV4','P','D43'],needed='保留真实root/续走分布与可排序终局标签，重训后按原三项验收；不堆同样全零学生续走'),
        dict(levels=[31,32,33,34],category='叶值压平',confidence='已证实低分差与局部反序；学习冷启动也是直接缺口',evidence='human/xray双臂均全败；D诊断16局0射手0香蒲、候选gap≤0.011、执行合法且Python/native一致。E完成64根/12372筛查+512确认均无胜，末波/终tick一致率.600/.548，两个土豆雷仅确认延后失败。射手/香蒲在根中不可种，不能评价优劣。',sources=['TF','D45','T52'],needed='在可支付且底座满足的早期状态补老师/人类示范起点数据；先建立胜负可分信号，不能直接调0.02或以降倍率替代诊断'),
        dict(levels=[36,37,38,39],category='冷启动无数据',confidence='缺独立现代面板；不宣称全败',evidence='已登记f8且有地形覆盖，但尚无本关独立结果；新增矿工/跳跳构成对策风险，未有实证证明其为当前主因。',sources=['E','R3','D45'],needed='复核对策覆盖/新族必要性后再预登记小面板；不能从31-34外推胜率'),
        dict(levels=[41,42,43,44,46,47,48,49],category='冷启动无数据',confidence='现役屋顶训练0份；具体搜索问题强弱分根',evidence='r7有32卷心菜，笼统缺投掷已被反驳。L41老师13/16、学生8/16；L46诊断1/4胜，非结构性全败。42-49有4seed老师诊断；候选遗漏/halving/低分差已在少数根发现，完整17根审计尚未完成。',sources=['TR','TR2','T52','D46c','D46','D46b'],needed='先让L41有效标签进训练；L42-49等待全候选终局归因和独立未来后补数据，不自动换卡或降倍率'),
    ]
    priorities=[
        dict(order=1,levels=[1,3,7,8,9,12,14,17,18,19,20],action='先闭合现有数据→训练→质量门禁，保住已过关，吸收普通夜间与原生L20纠正',evidence='L14/L17老师接近全胜而学生约半数，L20老师29/32对学生0/128；旧qc仅打包问题已修。A不是可直接全局替换的净改进。',depends=['task49A流水线与晋升门禁','task36配比/候选B排期','主控批准部署与数据配方'],sources=['T50','TV4','S27','SA','PLAN','D49'],exit='逐来源确认新标签入训；候选通过现有配对质量门禁，再看各关原版独立面板是否稳定',limits='不承诺增加关数；task55不换生产模型/门槛'),
        dict(order=2,levels=[41],action='优先兑现L41准入后的屋顶数据，作为新增区的低成本入口',evidence='老师15/16冷启动、最近13/16，学生8/16；屋顶训练来源为0；已登记qc但冻结时尚0窗口局。',depends=['task46已完成准入','task50已部署deploy3','task49训练门禁','task53可区分纠正标签（若正式验收通过）'],sources=['TR','T52','D46c','Q'],exit='获得实际完整采集/训练/配对评估证据，不把state注册当完成',limits='两组16样本非总体≥95%证明，须按既有准入/晋升程序推进'),
        dict(order=3,levels=[21,22,23,24,26,27,28,29,42,43,44,46,47,48,49],action='合并现有终局标签工作：先完成task43/46B/53再决定数据配比与换值网',evidence='A泳池126/320显示迁移机会；task43第一轮已否决，第二轮真实老师标签未验收；task46B少数根分别暴露候选遗漏、halving与小分差；不能只改某一阈值。',depends=['task43真实老师续走64seed+重训三门','task46B完整17根审计','task53独立确认/微调/192对评估/成本门禁'],sources=['SA','D43','D46b','D53','D53c'],exit='看原版胜负而非单纯末波/存活时间；保留不同来源与任务族后训练',limits='task53已筛查9723标签、确认7动作，但微调/192对仍待验；15%成本基准旧完整轮与新引擎单组投影不同，不宣布预算已在现行混合轮通过'),
        dict(order=4,levels=[31,32,33,34,36,37,38,39],action='雾区先补可赢的合法早期状态/人类示范起点；胜负有信号后才量记忆',evidence='E筛查/确认均无胜，部分延寿不足以教会稳定通关；K4与无记忆都落在0胜地板，尚不能判断记忆收益；对策牌和可支付时机需分开验证。',depends=['task45E固定基线','task48用户十局示范/verified重放与HF交付','task53纠正数据','新卡组族由主控登记'],sources=['TF','D45','D48','H48','T52'],exit='验证示范/老师接管能产生胜负可分的合法状态，再做无记忆/历史输入同数据对照',limits='task55未取得实际十局示范；不写规则打法，不把记忆当已证实根因；task52 LLM候选无胜负收益，暂不生产接入'),
        dict(order=5,levels=[2,25,27,28,29,30,36,37,38,39,40,46,47,48,49],action='分批补未知关证据，并补传送带原生任务；不把传送带整体后置',evidence='L2、27-29及雾后半缺独立面板；屋顶后半只有4seed诊断。L30/40当前为覆盖初始手牌变体，L25原生族未注册；L40还缺闪电可见性审计。',depends=['已有源清单/引擎谓词','task43/45/46的可教信号','任务族登记与主控小规模排期'],sources=['E','R2','R3','TR2'],exit='先核实deck×地形/可见性，再预登记原版同族小面板和独立保留种子',limits='本任务不采批量对局；不静默覆盖现行deck或历史档案'),
        dict(order=6,levels=[5,35,45,50],action='按用户顺序最后处理保龄球/砸罐/蹦极/僵王；打地鼠不做',evidence='保龄球种植选行已有单步证明，蹦极基础动作可表达；砸罐有点击/掉落卡/信息泄漏缺口，Boss有可见状态与终局链缺证。',depends=['本报告特殊机制清单','新任务族/编码/动作协议各自验收','主控排期'],sources=['E','EP','D46'],exit='每类单独通过原生reset、合法动作/公开观测、回放与整局结果门禁后再建学习数据',limits='玉米炮瞄准若另立任务再做；原生L50没有玉米炮，不作为前置'),
    ]
    return dict(special_mechanisms=special,bottleneck_groups=groups,priority_proposal=priorities,
                proposal_authority='建议，主控决定；未更改生产任务、模型、倍率、门槛或qc',
                secondary_risks=dict(deck='现行p9k/f8/r7覆盖地形；未证实其为普通关主因。对策不足需新族对照，先查牌再归因策略。',counter_zombies='潜水/冰车/海豚/小丑/气球/矿工/跳跳/梯子/投石车/巨人均需对策数据；引擎名单仅定位风险，不足以归因输局。',memory='human/xray皆全败、K4全败地板；记忆潜在价值未定，不列当前主因。'))


def table(d):
    kinds=dict(ordinary='普通',bowling='保龄球',whack='打地鼠',vasebreaker='砸罐子',boss='僵王',bungee='蹦极/传送带',little_trouble='小僵尸/传送带',stormy_night='暴风雨/传送带',mini_boss='传送带')
    scenes={0:'白天',1:'夜草',2:'泳池',3:'雾池',4:'屋顶',5:'Boss屋顶'}
    def score(m):return '—' if not m else f"{m['wins']}/{m['games']} [{m['source']}]"+('诊' if m.get('diagnostic') else '')
    def short(tid):
        return 'orig' if tid.startswith('day_l10_orig_') else re.sub(r'^(day|night|pool|fog|roof)_l\d+_(eval|train)_?','',tid)
    lines=['| 关 | 场景/类型 | 僵尸允许表/脚本补充（ID） | 卡组族 | qc当前倍率 | 最新学生 / 现役dw4 | 最近老师 | 状态 / 排查入口 |',
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
        if r['scripted_roster_supplement']:
            roster+='；'+('罐' if level==35 else '脚本')+','.join(map(str,r['scripted_roster_supplement']['types']))
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


def analysis_markdown(d):
    names=dict(mini_boss='原生传送带',little_trouble='LittleTrouble小僵尸',stormy_night='暴风雨传送带',bowling='保龄球',whack='打地鼠（用户排除）',vasebreaker='砸罐子',bungee='蹦极',boss='僵王',cob_cannon='玉米加农炮（跨关机制）')
    lines=['## 特殊机制：现有协议够到哪里','',
           '下表是缺口盘点，未实现任何新动作/观测。规模是相对工程量，小=复用已有协议与新任务验收，中=补可见状态/编码及回放，大=新交互链与训练输入/动作头协同；均不承诺训练能通关所需时间。reset+1tick只证明启动与一次更新；原生保龄球/Boss各一次plant的原始回执保存在[EP]。所有源码链接指本次advgap源码，二进制另存SHA。','',
           '| 类型/关 | 无头证据 | 动作协议 | 观测/任务缺口 | 规模与边界 |','|---|---|---|---|---|']
    for s in d['special_mechanisms']:
        level='/'.join(map(str,s['levels'])) or '非独立关'
        lines.append(f"| {names[s['kind']]} L{level} | {s['headless']} | {s['actions_sufficient']} | {s['observation_gaps']} | **{s['effort']}**；{s['scope']} |")
    lines += ['', '源码定位（对应上表，避免将猜测当作已实现能力）：','']
    for s in d['special_mechanisms']:
        links=[]
        for loc in s['code']:
            path,line=loc.rsplit(':',1);links.append(f'[{Path(path).name}:{line}]({ROOT/path}:{line})')
        lines.append(f"- {names[s['kind']]}："+'、'.join(links)+'；'+' '.join(f'[{key}]' for key in s['sources'])+'。')
    lines += ['', '另外，屋顶探针L41预置25盆（c0–c4），L45/L50为15盆（c0–c2），不能把“所有屋顶预置前五列”当作本次事实。L15/L35原生槽数分别3/1，现有reset会按传入deck重写；两关探针成功不等于保持原生玩法。砸罐的内容屏蔽与Boss可见状态均须同步检查full/lean/v2/原生推演和学生编码，不能只改展示JSON。', '', '## 普通关瓶颈：逐关主类与证据','',
              '总表为每个未过普通关指定一个主类；这里按同一证据分组，**不是声称单一确定根因**。“冷启动无数据”具体指现役模型的来源覆盖不足或尚无独立面板，绝非磁盘上没有任何档案。传送带/小游戏单列，不硬塞进普通关分类。','']
    for g in d['bottleneck_groups']:
        lines += [f"- **L{','.join(map(str,g['levels']))}：{g['category']}**（{g['confidence']}）。{g['evidence']} 下一步：{g['needed']}。"+' '.join(f'[{k}]' for k in g['sources'])]
    lines += ['', '卡组、对策型僵尸与雾记忆暂列次级风险：','']
    for k,v in d['secondary_risks'].items():lines.append(f'- {v}')
    lines += ['', '不能由允许名单直接推断需要换卡；先核对地形/对策覆盖，再用同族分叉或新族配对验证。task52三场地144局已完成，LLM新增提议仅8/510被搜索采用、0/749非法，96配对比较的胜负与末波净差均0，约132万token；当前不建议接生产路径。f8缺开雾/部分对策卡及根的可支付限制，也不能据此否定所有LLM对策先验。若要换卡，按AGENTS §8另开族、DESIGN登记、历史whatif仍用档案deck。[T52] [H48]', '', '## 推进顺序建议（交主控决定）','',
              '以兑现现有可教信号和减少未知为优先，不预测会新增多少关。验收看原版同任务/同种子配对与独立面板，末波/延寿不能代替胜负。以下不修改生产配方、门槛、倍率或qc：','']
    for p in d['priority_proposal']:
        lines += [f"{p['order']}. **{p['action']}**（L{','.join(map(str,p['levels']))}）。{p['evidence']} 依赖：{'；'.join(p['depends'])}。完成证据：{p['exit']}。限制：{p['limits']}。"+' '.join(f'[{k}]' for k in p['sources'])]
    lines += ['', 'task53截至本次来源读取已完成47根9723条筛查及12根384条独立16未来确认，7个动作通过校正；微调/192对评估仍待完成，不能写成学生已进步。task43第二轮仍在采64seed老师续走标签，尚无新v5三门通过证据。task46B只有4/17根442条完整标签的已审计子集。task45E则已完整交付、筛查与确认全无胜局。人类示范工具已交付，十局计划可读；本次未取得真实示范对局/HF证据，不宣称用户已打完。[D53] [D43] [D46b] [D45] [H48]', '']
    return '\n'.join(lines)


if __name__=='__main__':
    d=build();(ART/'assembled-gap.json').write_text(json.dumps(d,ensure_ascii=False,indent=1)+'\n')
    (ART/'table.md').write_text(table(d))
    print(json.dumps(d['summary'],ensure_ascii=False))
