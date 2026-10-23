"""Reproduce selected Gemini claims and check every pre-branch physical frame."""
from pathlib import Path
import json,sys,os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
r=Path(__file__).resolve().parents[1];sys.path[:0]=[str(r/'scripts'),str(r/'python')]
from episode_query import load,_replay_task_extra,_parse_action,_hand_at
from render_episode import collect
results=[]
for mode,decision,actions in [('xray',32,['plant:sunshroom:4:5','plant:sunshroom:1:3']),('xray',44,['plant:lilypad:3:8','plant:potato:0:8','plant:potato:4:2','plant:potato:4:3','wait:60']),('human',48,['plant:lilypad:2:2','plant:potato:0:2','wait:60']),('human',365,['plant:potato:1:3','plant:potato:4:3','wait:60'])]:
 meta,frames=load(r/f'artifacts/task45/phase-d/l31-{mode}-teacher');t=meta['task'];prefix=[f['action'] for f in frames[1:decision]]
 for action in actions:
  override=_parse_action(action,_hand_at(frames,decision,t['deck']))
  rec=collect('/home/newbiexvwu/.cache/pvz-research-resources',t['seed'],31,'student',4000,t['deck'],
      override={decision:override},prefix_actions=prefix,checkpoint=t['checkpoint'],task_extra=_replay_task_extra(t))
  rec['frames']=json.loads(json.dumps(rec['frames']))
  keys=set(frames[0])&set(rec['frames'][0])-{'action','eaten','targets','label','verification'}
  for i in range(decision):
   assert {k:frames[i][k] for k in keys}=={k:rec['frames'][i][k] for k in keys},(mode,decision,action,i)
  results.append(dict(mode=mode,decision=decision,root_tick=frames[decision-1]['tick'],action=action,
      prefix_frames_equal=decision,fields=sorted(keys),accepted=not rec['frames'][decision].get('action_rejected'),outcome=rec['outcome']))
  print(mode,decision,action,rec['outcome']['final_wave'],flush=True)
(r/'MEMORY/docs/task45-fog-whatif-verified.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
