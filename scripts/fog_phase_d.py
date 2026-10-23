"""Task45 appendix D: matched original-difficulty timelines, no curriculum selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MAIN=Path('/home/newbiexvwu/PvZ-search')
OUT=ROOT/'artifacts/task45/phase-d'
MANIFEST=ROOT/'artifacts/task45/fog_panel.json'
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def cells():
    for level in (31,32,33,34):
        for mode,suffix in [('human','f8'),('xray','f8x')]:
            for policy in ('teacher','student'):
                yield level,mode,suffix,policy,OUT/f'l{level}-{mode}-{policy}'

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    for level,mode,suffix,policy,out in cells():
        if (out/'meta.json').exists():continue
        out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(ROOT/'scripts/episode_query.py'),'capture','--policy',policy,
             '--checkpoint',str(MAIN/'artifacts/student/dw4i_s1/model.pt'),
             '--value',str(MAIN/'artifacts/value_net/v4/model.pt'),
             '--static-value',str(MAIN/'artifacts/hard/value_branch1/model.pt'),
             '--task-manifest',str(MANIFEST),'--task-id',f'fog_l{level}_eval_{suffix}',
             '--resource-dir','/home/newbiexvwu/.cache/pvz-research-resources',
             '--timeline',str(out/'timeline.jsonl'),'--out',str(out)]
        print('START',out.name,flush=True)
        with (out/'capture.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,
                           env={**os.environ,'PVZ_NO_POPUP':'1'})
        print('DONE',out.name,flush=True)

def summarize():
    from pvz_frame_store import read_frames,frame_path
    summaries=[]
    cold_plan=json.loads((ROOT/'artifacts/task45/cold/plan.json').read_text())
    for level,mode,suffix,policy,out in cells():
        meta=json.loads((out/'meta.json').read_text())
        rows=[json.loads(x) for x in (out/'timeline.jsonl').read_text().splitlines()]
        frames=read_frames(frame_path(out))
        plants=[dict(tick=x['tick'],decision=x['decision'],**x['planting']) for x in rows if x.get('planting')]
        accepted=[p for p in plants if p['accepted']]
        native_mismatches=[dict(decision=x['decision'],tick=x['tick'],python=x['student_action'],native=x['native_action']) for x in rows if not x['native_matches']]
        wait_ticks=sum(x['after_tick']-x['tick'] for x in rows if x['action']['type']=='wait')
        cheap_ticks=sum(x['after_tick']-x['tick'] for x in rows if x['action']['type']=='wait' and x['sun']<100)
        no_economy_ticks=sum(x['after_tick']-x['tick'] for x in rows if x['sunshrooms']==0)
        water=[p for p in accepted if p['terrain']==3]
        shooter_max=[max(sum(row['row_shooters'][r].values()) for row in rows) for r in range(6)]
        summary=dict(level=level,observation=mode,policy=policy,seed=meta['task']['seed'],outcome=meta['outcome'],
            archive=str(out),archive_sha256=sha(out/'frames.jsonl'),timeline_sha256=sha(out/'timeline.jsonl'),
            native_checked=len(rows),native_mismatches=native_mismatches,
            types=dict(Counter(p['type'] for p in accepted)),plantings=plants,
            sun_min=min(x['sun'] for x in rows),sun_max=max(max(x['sun'],x['after_sun']) for x in rows),
            produced=sum(x['events'].get('sun_produced',0) for x in rows),spent=sum(p['cost'] for p in accepted),
            final_sun=rows[-1]['after_sun'],initial_sun=rows[0]['sun'],
            wait_ticks=wait_ticks,wait_below100_ticks=cheap_ticks,no_economy_ticks=no_economy_ticks,
            water_plantings=len(water),water_nonlily_without_base=[p for p in water if p['type']!=16 and 16 not in p['occupants']],
            max_lilypads=max(x['lilypads'] for x in rows),max_cattails=max(x['cattails'] for x in rows),
            max_row_shooters=shooter_max,final_row_shooters=rows[-1]['row_shooters'],
            row_shooter_changes=[dict(tick=x['tick'],rows=x['row_shooters']) for i,x in enumerate(rows) if i==0 or x['row_shooters']!=rows[i-1]['row_shooters']],
            econ_placements=[p for p in accepted if p['type']==9])
        assert summary['initial_sun']+summary['produced']-summary['spent']==summary['final_sun'],summary
        cell=next(c for c in cold_plan['cells'] if c['task_id']==f'fog_l{level}_eval_{suffix}' and
                  c['arm']==('fast_search' if policy=='teacher' else 'fast'))
        source=Path(cell['out'])/'games.jsonl'
        old=next(json.loads(line) for line in source.read_text().splitlines() if json.loads(line)['seed']==summary['seed'])
        assert old['final_wave']==summary['outcome']['final_wave']
        assert old['final_tick']==summary['outcome']['final_tick']
        assert old['decisions']==len(rows)
        assert old['planted_types']=={str(k):v for k,v in summary['types'].items()}
        summary['original_cold_match']=dict(wave_tick_decisions_planted_types=True,source=str(source),sha256=sha(source))
        summary['search_stats']={k:old[k] for k in ('searched','overrides','student_candidate_added','student_candidate_best','student_candidate_override')}
        summary['affordable_pea_decisions']=[dict(tick=x['tick'],sun=x['sun'],action=x['action'],
            legal_cells=sum(p['packet']==0 for p in x['legal_plants'])) for x in rows if x['sun']>=100]
        summaries.append(summary)
    for level in (31,32,33,34):
        for mode in ('human','xray'):
            a,b=[g for g in summaries if g['level']==level and g['observation']==mode]
            af=read_frames(frame_path(Path(a['archive'])));bf=read_frames(frame_path(Path(b['archive'])))
            assert [f.get('action') for f in af]==[f.get('action') for f in bf]
            a['paired_student_all_actions_identical']=True
    result=dict(schema_version=1,scope='four matched games per acting-policy/observation arm, L31–34 first eval seed',
        resolution='decision boundaries indexed by actual engine tick; no interpolated per-tick observations',
        firepower='engine SUBCLASS_SHOOTER live counts by type/row, raw projectiles in timeline; no invented DPS',
        models={str(p):sha(p) for p in [MAIN/'artifacts/student/dw4i_s1/model.pt',MAIN/'artifacts/value_net/v4/model.pt',MAIN/'artifacts/hard/value_branch1/model.pt',ROOT/'build/pvz-portable']},games=summaries)
    (ROOT/'MEMORY/docs/task45-fog-timelines.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([dict(level=x['level'],mode=x['observation'],policy=x['policy'],types=x['types'],sun_max=x['sun_max'],native=len(x['native_mismatches'])) for x in summaries],ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','summary']);a=p.parse_args()
    run() if a.mode=='run' else summarize()
