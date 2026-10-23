"""Read-only task55 delivery audit against frozen original evidence; no game processes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'MEMORY/docs/adventure-gap.md'
DATA=ROOT/'MEMORY/docs/adventure-gap.json'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    d=read(DATA);text=REPORT.read_text();sources=d['sources'];rows=d['levels']
    assert [r['level'] for r in rows]==list(range(1,51))
    assert len(re.findall(r'^\| \d+-\d+（',text,re.M))==50
    assert dict(Counter(r['status'] for r in rows))==d['summary']['statuses']
    assert sum(bool(r['qc']) for r in rows)==d['summary']['qc_levels']==26
    assert sum(len(r['qc']) for r in rows)==d['summary']['qc_cells']==38
    changed=[]
    frozen={}
    for key,s in sources.items():
        assert sha(s.get('snapshot',s['path']))==s['sha256'],key
        if sha(s['path'])!=s['sha256']:changed.append(key)
        path=Path(s.get('snapshot',s['path']))
        frozen[key]=read(path) if path.suffix=='.json' else path.read_text()
    for path,h in d['engine']['source_sha256'].items():assert sha(ROOT/path)==h,path
    for path,h in d['audit_code_sha256'].items():assert sha(ROOT/path)==h,path
    e=read(sources['E']['path'])
    assert sha(e['executable'])==e['executable_sha256']
    assert all(r['reset_ok'] and r['one_tick_ok'] and r['observed_level']==r['level'] for r in e['levels'])
    state=read(sources['Q']['path']);panel=read(sources['QP']['path'])
    factors=[1.]+panel['generation_audit']['multipliers']
    def metrics(r):
        return [r[k] for k in ('student_latest','student_current','teacher_latest','teacher_panel') if r[k]]+r['teacher_other_tasks']
    def original_count(m):
        key=m['source'];tid=m['task'];raw=frozen[key]
        if key=='S27':
            t=next(t for t in raw['v2']['dw4i_s1']['tasks'] if t['task_id']==tid);return t['won'],t['games']
        if key=='SA':
            t=next(t for t in raw['per_task'] if t['task_id']==tid);return t['b_won'],t['paired']
        if key=='TV4':
            t=raw['results'][tid];return t['new_wins'],t['pairs']
        if key=='T50':return raw[tid]['wins']['after'],raw[tid]['games']
        if key=='D30d':
            assert '探针 8 局两臂都 0 胜' in raw
            return 0,frozen['D30']['per_level']['v4'][tid]['games']
        if key=='T52':
            t=next(t for t in raw['families'].values() if t['task']['task_id']==tid)['comparisons']['B_low'];return t['before_wins'],t['games']
        if key in ('TF','TR'):
            teacher=' + v4' in m['model'];arm=('fast_search' if teacher else 'fast') if key=='TF' else ('teacher' if teacher else 'student')
            cells=raw['cold']['cells'] if key=='TF' else raw['cells']
            t=next(t for t in cells if t['task_id']==tid and t['arm']==arm);return t['won' if key=='TF' else 'wins'],t['games']
        level=int(re.search(r'_l(\d+)',tid)[1])
        if key=='TR2':
            gs=[g for g in raw['games'] if g['level']==level and g['policy']=='teacher'];return sum(g['outcome']['won'] for g in gs),len(gs)
        if key in ('N','P'):
            # The n9 and p9k teacher is the last column of their old deck-comparison tables.
            chapter=f'{(level-1)//10+1}-{(level-1)%10+1}'
            lines=[line for line in raw.splitlines() if re.match(r'\| '+re.escape(chapter)+r'(?: \(L\d+\))? \|',line) and re.search(r'\d+/16 \|$',line)]
            if key=='N':lines=[line for line in lines if '[' not in line]
            assert len(lines)==1,(key,tid,lines)
            return tuple(map(int,re.findall(r'(\d+/16) \|$',lines[0])[0].split('/')))
        if key=='D':
            if level==10:assert '| v34+搜索老师（beta=1，搜索执行） | 64/64 |' in raw
            else:assert re.search(r'^\| 1-'+str(level)+r' \|.*\| 64/64 \|',raw,re.M)
            return 64,64
        raise AssertionError(('unknown metric source',key))
    for r,fact in zip(rows,e['levels']):
        assert r['scene']==fact['scene'] and r['allowed_roster']==fact['allowed_roster']
        for q in r['qc']:
            cell=state['cells'][q['base']]
            assert cell['level']==r['level'] and q['multiplier']==factors[cell['mult_index']]
        for m in metrics(r):
            assert 0<=m['wins']<=m['games'] and m['games']>0 and m['source'] in sources and m['date']
            assert m['multiplier']==1. or (m['task']=='day_l7_train_d7_m150' and m['multiplier']==1.5)
            assert original_count(m)==(m['wins'],m['games']),(r['level'],m)
        if r['status']=='已过':
            m=r['student_current'];assert m['games']>=32 and m['wins']/m['games']>=.95
        if r['kind']=='ordinary' and r['status']!='已过':
            assert r['bottleneck'] in ('卡组','冷启动无数据','叶值压平','对策型僵尸','记忆（雾）','工程')
            assert r['bottleneck_sources'] and all(k in sources for k in r['bottleneck_sources'])
    grouped=[l for g in d['bottleneck_groups'] for l in g['levels']]
    expected=[r['level'] for r in rows if r['kind']=='ordinary' and r['status']!='已过']
    assert sorted(grouped)==expected and len(set(grouped))==len(grouped)
    for g in d['bottleneck_groups']:
        assert all(rows[l-1]['bottleneck']==g['category'] for l in g['levels'])
    # Missing independent evidence must stay missing, and small diagnostic panels stay labelled.
    assert rows[1]['student_latest'] is None and rows[1]['teacher_latest'] is None
    assert rows[9]['task_family']['deck']==[] and rows[9]['task_family']['conveyor_start']=='original'
    assert rows[34]['scene']['terrain']==1 and not rows[39]['scene']['fog']
    assert rows[45]['teacher_latest']['wins']==1 and rows[45]['teacher_latest']['games']==4
    assert rows[45]['teacher_latest']['diagnostic'] and rows[45]['student_latest'] is None
    assert len(d['special_mechanisms'])==9
    assert {l for s in d['special_mechanisms'] for l in s['levels']}=={5,10,15,20,25,30,35,40,45,50}
    for s in d['special_mechanisms']:
        assert all(s.get(k) for k in ('headless','actions_sufficient','observation_gaps','effort','scope','sources','code'))
        assert s['effort'] in ('小','中','大')
        for loc in s['code']:
            path,line=loc.rsplit(':',1);assert 0<int(line)<=len((ROOT/path).read_text().splitlines())
    for group in (d['special_mechanisms'],d['bottleneck_groups'],d['priority_proposal']):
        assert all(all(k in sources for k in x['sources']) for x in group)
    assert [p['order'] for p in d['priority_proposal']]==list(range(1,7))
    assert all(p.get('depends') and p.get('exit') and p.get('limits') for p in d['priority_proposal'])
    # Check the complete latest student panel against the actual games, not only its summary.
    panel_a=Path(sources['SA']['raw_directory']);games=wins=0
    for tid,h in sources['SA']['raw_sha256'].items():
        path=panel_a/tid;assert sha(path)==h
        rs=[json.loads(line) for line in path.read_text().splitlines()]
        assert len({r['seed'] for r in rs})==len(rs) and all(r['arm']=='fast' and r['searched']==0 for r in rs)
        summary=read(path.parent/'summary.json');assert sum(r['won'] for r in rs)==summary['fast']['won']
        games+=len(rs);wins+=sum(r['won'] for r in rs)
    assert len(sources['SA']['raw_sha256'])==34 and games==3264 and wins==2203
    assert sha(frozen['SA0']['model'])==frozen['SA0']['model_sha256']
    # Both reference-style source links and ordinary local links must resolve.
    for target in re.findall(r'\]\(([^)]+)\)|^\[[^\]]+\]:\s*(\S+)',text,re.M):
        path=next(x for x in target if x).split('#',1)[0]
        if not path or re.match(r'https?://',path):continue
        path=re.sub(r':\d+$','',path);assert (REPORT.parent/Path(path)).is_file(),path
    assert DATA.stat().st_size<200_000
    from build_adventure_gap import table,analysis_markdown
    assert table(d) in text and analysis_markdown(d) in text
    if d['status']=='complete':
        assert not d['pending'] and d['completion_audit']['all_requirements_verified']
        assert sha(REPORT)==d['completion_audit']['report_sha256']
        assert sha(Path(__file__))==d['completion_audit']['verifier_sha256']
    return dict(passed=True,levels=50,ordinary_unpassed_classified=len(expected),special_mechanisms=9,
                qc_cells=38,qc_levels=26,student_a_games=games,student_a_wins=wins,
                source_files=len(sources),live_sources_advanced_after_snapshot=changed,
                scope='evidence/report audit only; not a claim that 50 levels are cleared')


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False))
