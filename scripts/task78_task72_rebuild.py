"""Recover task72b's missing takeover mapping without changing its 94 reference states.

plan/bind start no games. run TAKEOVER FUTURE records one production-v5 trajectory
with read-only input signatures. Matching states is only the first step: the original
nine-future probe schedule and rounded v5 values must also be verified before rescore.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
from task78_data import MAIN,sha
from value_teacher_feasibility import history_record,write_json

# The earlier attempt used the desktop static net. Keep its plan and trajectory
# intact; task72/72b used the different Mac branch1 checkpoint.
OUT=ROOT/'artifacts/task78/task72_rebuild_mac_branch1'
ORIGINAL_STATIC=Path('/home/newbiexvwu/pvz-logs/mac_branch1/model.pt')
ORIGINAL_STATIC_SHA256='91e562a051c462f9bf3426b035b7b87187bda3c8b5f3e88094f2478a61a7926c'
REFERENCE=ROOT/'MEMORY/docs/task72b-probe-rows.jsonl'
TASK='demo_fog_33_cactus_snow_m200'
TAKEOVERS=(229,306,382)
FUTURES=tuple(range(101,109))
DEMO=ROOT.parent/'cq/artifacts/human_demo/raw/seed_20261033'/TASK


def reference_rows():
    rows=[json.loads(line) for line in REFERENCE.read_text().splitlines() if line.strip()]
    if len(rows)!=94 or len({identity(row) for row in rows})!=94:
        raise ValueError('exactly 94 unique original state identities required')
    return rows


def identity(row):
    return row['task'],row['seed'],row['decision'],row['tick']


def inputs():
    from pvz_frame_store import frame_path
    if sha(ORIGINAL_STATIC)!=ORIGINAL_STATIC_SHA256:
        raise ValueError('task72b reconstruction requires the original Mac branch1 checkpoint')
    paths=dict(reference=REFERENCE,demo_frames=frame_path(DEMO),demo_meta=DEMO/'meta.json',
        task_config=ROOT/'configs/human_demo_tasks.json',
        value=MAIN/'artifacts/value_net/v5/model.pt',
        student=MAIN/'artifacts/student/dw4i_s1/model.pt',
        static=ORIGINAL_STATIC,executable=ROOT/'build/pvz-portable')
    return {k:dict(path=str(path),sha256=sha(path)) for k,path in paths.items()}


def code():
    paths=list((ROOT/'python').glob('pvz_*.py'))+[ROOT/'scripts'/name for name in
        ('search_play.py','render_episode.py','pvz_tasks.py','task78_task72_rebuild.py')]
    return {str(path):sha(path) for path in sorted(paths)}


def plan():
    from pvz_frame_store import frame_path,read_frames
    reference_rows()
    frames=read_frames(frame_path(DEMO))
    task=next(t for t in json.loads((ROOT/'configs/human_demo_tasks.json').read_text())['tasks']
              if t['task_id']==TASK)
    document=dict(inputs=inputs(),code=code(),task=task,
        takeovers=[dict(decision=n,tick=frames[n-1]['tick'],wave=frames[n-1]['wave']) for n in TAKEOVERS],
        jobs=[dict(takeover=n,future_seed=s) for n in TAKEOVERS for s in FUTURES],
        trajectory_policy=dict(samples=3,verify_samples=9,student_candidates=2,horizon=2500,
            min_gap=300,per_card=1,halving=True,rollout_sim_skip=61,dagger_beta=1),
        matching='exact task/future/continuation decision/tick/wave/sun/cactus legal cells/balloons/static best/chosen action; unique across all 24 trajectories',
        balloon_positions='int(x) truncation, with original full-precision frame retained for auditing',
        original_probe_protocol_verified=False,training_eligible=False,
        required_before_rescore='restore original probe protocol or reproduce all rounded v5 values at all original states, actions and horizons; do not fit a seed schedule to outcomes')
    if document['inputs']['value']['sha256']!='8a636d337c03dcf3037f42ce41f79abc5e0b898eb5a2b39d9a00ad3b9d88c3b4':
        raise ValueError('reconstruction must use adopted v5')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=document:
        raise ValueError('reconstruction plan changed; preserve and explicitly amend it first')
    write_json(path,document)
    write_json(OUT/'tasks.json',dict(tasks=[task]))
    return document


def frozen():
    document=json.loads((OUT/'plan.json').read_text())
    if document['inputs']!=inputs() or document['code']!=code():
        raise ValueError('reconstruction inputs or production code changed')
    if json.loads((OUT/'tasks.json').read_text())!={'tasks':[document['task']]}:
        raise ValueError('reconstruction task manifest changed')
    return document


def resource_check():
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:')))*1024
    if available<10*1024**3:
        raise RuntimeError('yield to qc: reconstruction requires available memory >=10 GiB')


class Recorder:
    def __init__(self,future,path):
        self.future=future
        self.wanted={row['decision'] for row in reference_rows() if row['seed']==future}
        self.path=path
        self.pending=None

    def before(self,env,obs,state,decision):
        resource_check()
        self.pending=None
        if decision not in self.wanted:
            return
        from render_episode import _frame
        frame=_frame(obs)
        cells=sorted([a['row'],a['col']] for a in
                     ((obs.get('legal_actions') or {}).get('plants') or []) if a['packet']==8)
        positions=[z[2] for z in frame['zombies'] if z[1]==16 and z[5]]
        actions=[dict(type='plant',packet=8,row=r,col=c) for r,c in cells]
        static_best=None
        root_hash=env._debug_state_sha256()
        if actions:
            root=env.snapshot_fast()
            try:
                values=env.static_scores(root,actions,obs.get('sun_income_rate') or 0.)
                legal=[(v,a) for v,a in zip(values,actions) if v is not None]
                if legal:
                    best=max(legal,key=lambda pair:pair[0])[1]
                    static_best=[best['row'],best['col']]
            finally:
                env.restore_fast(root);env.release_snapshot(root)
            if env._debug_state_sha256()!=root_hash:
                raise RuntimeError('read-only static signature changed the root state')
        self.pending=dict(task=TASK,seed=self.future,decision=decision,tick=obs['tick'],
            wave=obs['wave'],sun=obs['sun'],balloons=len(positions),
            balloon_x=sorted(int(x) for x in positions),cells=cells,n_cells=len(cells),
            static_best=static_best,frame=frame,history_before_action=history_record(deepcopy(state)),
            engine_state_sha256=root_hash)

    def after(self,obs,action,policy_choice,rejected,info):
        if self.pending is not None:
            row=self.pending|dict(chosen=action,action_rejected=rejected)
            with self.path.open('a') as stream:stream.write(json.dumps(row)+'\n')
        self.pending=None


def run(takeover,future):
    if takeover not in TAKEOVERS or future not in FUTURES:
        raise ValueError('only the original 3 takeover points and 8 future seeds are supported')
    with (ROOT/'artifacts/task78/games-controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        document=frozen()
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                if path.read_bytes().split(b'\0')[0]==document['inputs']['executable']['path'].encode():
                    raise RuntimeError('another task78 game is live')
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        resource_check();os.nice(19)
        directory=OUT/f'n{takeover}/f{future}'
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/'complete.json').exists():
            raise ValueError('completed trajectory already exists; use bind instead of overwriting it')
        signatures=directory/'signatures.jsonl'
        if signatures.exists():
            raise ValueError('partial trajectory exists; preserve it explicitly before restarting')
        import search_play as teacher
        args=teacher.parser().parse_args(['--manifest',str(OUT/'tasks.json'),'--task-id',TASK,
            '--out',str(directory),'--value',document['inputs']['value']['path'],
            '--rollout-policy',document['inputs']['student']['path'],
            '--static-value',document['inputs']['static']['path'],
            '--demo-archive',str(DEMO),'--demo-decision',str(takeover),
            '--future-seeds',str(future),'--workers','1','--verify-samples','9',
            '--student-candidates','2','--frames',str(OUT/'frames'),
            '--frames-label',f'task72b_n{takeover}',
            '--executable',document['inputs']['executable']['path'],
            '--resource-dir','/home/newbiexvwu/.cache/pvz-research-resources'])
        keep,bins=teacher._export_values(args.value,args.static_value)
        teacher._init_worker(bins[0],args.rollout_policy,61,bins[1])
        try:
            job=dict(args=args,seed=20261033,future_seed=future,arm='fast_search')
            result=teacher._play(job,Recorder(future,signatures))
            if result.get('terminal') is not True:
                raise ValueError('reconstruction trajectory did not reach terminal')
            write_json(directory/'complete.json',dict(plan_sha256=sha(OUT/'plan.json'),
                takeover=takeover,future_seed=future,signatures_sha256=sha(signatures),result=result))
        finally:
            teacher._close_worker_env();keep.cleanup()


def matches(reference,candidate):
    for field in ('task','seed','decision','tick','wave','sun','balloons','n_cells','static_best'):
        if reference[field]!=candidate[field]:return False
    if sorted(reference['cells'])!=sorted(candidate['cells']):return False
    if sorted(reference['balloon_x'])!=sorted(candidate['balloon_x']):return False
    action=json.loads(reference['chosen']) if isinstance(reference['chosen'],str) else reference['chosen']
    return action==candidate['chosen'] and candidate.get('action_rejected') is False


def bind():
    document=frozen();missing=[];candidates=[]
    for job in document['jobs']:
        directory=OUT/f"n{job['takeover']}/f{job['future_seed']}"
        complete=directory/'complete.json';path=directory/'signatures.jsonl'
        if not complete.exists():missing.append(job);continue
        receipt=json.loads(complete.read_text())
        if (receipt['plan_sha256']!=sha(OUT/'plan.json') or receipt['takeover']!=job['takeover'] or
                receipt['future_seed']!=job['future_seed'] or receipt['signatures_sha256']!=sha(path) or
                receipt['result'].get('terminal') is not True):
            raise ValueError('completed trajectory has changed identity or incomplete signatures')
        records=[json.loads(line) for line in path.read_text().splitlines()]
        if len({identity(row) for row in records})!=len(records):
            raise ValueError('duplicate input signatures in one reconstruction trajectory')
        candidates.extend(row|dict(takeover=job['takeover'],signatures_path=str(path),
                                   signatures_sha256=sha(path)) for row in records)
    bindings=[]
    for reference in reference_rows():
        found=[row for row in candidates if matches(reference,row)]
        bindings.append(dict(identity=identity(reference),matches=[dict(
            takeover=row['takeover'],signatures_path=row['signatures_path'],
            signatures_sha256=row['signatures_sha256'],engine_state_sha256=row['engine_state_sha256'])
            for row in found]))
    mapped=not missing and all(len(row['matches'])==1 for row in bindings)
    report=dict(reference_sha256=sha(REFERENCE),plan_sha256=sha(OUT/'plan.json'),
        missing_trajectory_jobs=missing,bindings=bindings,mapping_complete=mapped,
        original_probe_protocol_verified=False,rescore_eligible=False,
        task72b_evaluation_complete=False,
        next_required='verify the original nine-future probe protocol and every rounded v5 value before v6 rescore')
    write_json(OUT/'bindings.json',report)
    return report


def main():
    if len(sys.argv)<2 or sys.argv[1] not in ('plan','run','bind'):
        raise SystemExit('usage: task78_task72_rebuild.py plan | run TAKEOVER FUTURE | bind')
    if sys.argv[1]=='run':return run(*map(int,sys.argv[2:]))
    result=plan() if sys.argv[1]=='plan' else bind()
    print(json.dumps({k:v for k,v in result.items() if k not in ('bindings','code','inputs','task')},indent=1))


if __name__=='__main__':main()
