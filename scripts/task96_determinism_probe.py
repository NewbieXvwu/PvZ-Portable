"""Bounded operator diagnosis on one frozen label segment, not a production run.

The full same-pack one-epoch repeats already exist. This diagnostic uses their
unchanged trainer on a small real segment to trace initialisation, shuffle and
deterministic-algorithm errors. All new writes stay in the repro worktree.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/task96/determinism_probe'
SOURCE=Path('/tmp/exp96')


def read(path):return json.loads(path.read_text())


def write(path,document):
    assert path.is_absolute() and path.is_relative_to(ROOT)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(document,ensure_ascii=False,indent=1)+'\n')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def qc_gpu_active():
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,check=True)
    for line in query.stdout.splitlines():
        pid=int(line.strip())
        try:command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
        except FileNotFoundError:continue
        if '/.claude/worktrees/cq/' in command or 'curriculum_state.pipeline/round_' in command:
            return True
    return False


def prepare():
    import numpy as np
    recipe=read(SOURCE/'detexp/recipe_run1.json')
    index=Path(recipe['data'])/'packs/7d15b915a2192805.npz'
    with np.load(index,allow_pickle=False) as stored:
        paths=[index.parent/str(name) for name in stored['segments']]
    selected=None
    for path in sorted(paths,key=lambda p:p.stat().st_size):
        if path.stat().st_size>8*1024**2:break
        with np.load(path,allow_pickle=False) as segment:
            rows=int(segment['rows'])
            if not 2048<=rows<=16384:continue
            seeds=np.unique(segment['seed'])
            if len(seeds)<10:continue
            label=Path(str(segment['source'][0])).parent.name
        selected=path;break
    assert selected is not None, 'no bounded segment found; do not expand into a full training run'
    destination=OUT/'segment.npz';destination.parent.mkdir(parents=True,exist_ok=True)
    expected=sha(selected)
    if destination.exists():assert sha(destination)==expected
    else:shutil.copyfile(selected,destination)
    assert sha(destination)==sha(selected)==expected
    # label_weight must name a subset of the new single-label list; this label was never weighted.
    recipe.update(data=str(OUT),labels=[label],source_row_report=False,batch_size=128,
                  validation_batch_size=128,max_steps=None,epochs=1,cache=False,
                  label_weight={k:v for k,v in recipe.get('label_weight',{}).items() if k==label})
    write(OUT/'recipe.json',recipe)
    write(OUT/'input.json',dict(source_index=str(index),source_index_sha256=sha(index),
        source_segment=str(selected),segment_sha256=expected,rows=rows,seeds=len(seeds),label=label,
        scope='One real label segment; operator/seed diagnosis only, not a reproduction of the full 16,982,931-row experiment'))
    print(json.dumps(read(OUT/'input.json')),flush=True)


def child(case):
    # Set the workspace before importing torch or creating any CUDA context.
    strict=case.startswith('strict')
    if case=='strict_no_workspace' or case.startswith('nows'):os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
    else:os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    # Extra cases: does the cuBLAS workspace size alone change the arithmetic?
    if os.environ.get('TASK96_WORKSPACE'):os.environ['CUBLAS_WORKSPACE_CONFIG']=os.environ['TASK96_WORKSPACE']
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    os.environ['PVZ_NO_POPUP']='1'
    sys.path[:0]=[str(SOURCE/'scripts'),str(SOURCE/'python'),str(ROOT/'scripts')]
    import torch
    import train_rollout_policy as trainer
    assert Path(trainer.__file__).resolve()==SOURCE/'scripts/train_rollout_policy.py'
    expected=read(ROOT/'MEMORY/docs/task96-existing-runs.json')['source_snapshot']
    assert all(sha(SOURCE/entry['path'])==entry['sha256'] for entry in expected), 'frozen trainer source changed'
    from task96_existing_runs import tensor_digest
    uses_gpu=os.environ.get('TASK96_DEVICE',read(OUT/'recipe.json')['device']).startswith('cuda')
    if uses_gpu and qc_gpu_active():raise RuntimeError('qc GPU priority: no task96 training started')
    torch.use_deterministic_algorithms(strict)
    torch.backends.cudnn.deterministic=strict
    torch.backends.cudnn.benchmark=False
    folder=OUT/case;folder.mkdir(parents=True,exist_ok=True)
    recipe=read(OUT/'recipe.json');recipe['out']=str(folder)
    # Smoke-only overrides; the recorded cases never set them.
    if os.environ.get('TASK96_DEVICE'):recipe['device']=os.environ['TASK96_DEVICE']
    if os.environ.get('TASK96_MAX_STEPS'):recipe['max_steps']=int(os.environ['TASK96_MAX_STEPS'])
    if os.environ.get('TASK96_EPOCHS'):recipe['epochs']=int(os.environ['TASK96_EPOCHS'])
    write(folder/'recipe.json',recipe)
    trace=dict(case=case,strict=strict,torch_version=torch.__version__,cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(),workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
               data=read(OUT/'input.json'),steps=[],permutations=[],qc_guard_seconds=0.0,
               gpu=torch.cuda.get_device_name(0),
               cudnn_deterministic=torch.backends.cudnn.deterministic,
               cudnn_benchmark=torch.backends.cudnn.benchmark,
               matmul_precision=torch.get_float32_matmul_precision(),
               fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory)
    original_net=trainer.RolloutNet
    def network(*args,**kwargs):
        net=original_net(*args,**kwargs)
        trace['initial_tensor_sha256']=tensor_digest(net.state_dict())
        return net
    trainer.RolloutNet=network
    # read_segments, not read_pack: a single-file read_pack drops the packet-head store that the
    # identity head needs. The segment carries the same packet columns the merged pack had.
    trainer.load=lambda *args,**kwargs:trainer.read_segments([OUT/'segment.npz'],None)
    original_perm=torch.randperm
    def permutation(*args,**kwargs):
        result=original_perm(*args,**kwargs)
        trace['permutations'].append(hashlib.sha256(result.cpu().numpy().tobytes()).hexdigest())
        return result
    torch.randperm=permutation
    original_metrics=trainer.Metrics
    trainer.Metrics=lambda *args,**kwargs:original_metrics(*args,directory=folder/'telemetry',**kwargs)
    def observe(step,loss):
        guard_started=time.monotonic()
        active=uses_gpu and qc_gpu_active()
        trace['qc_guard_seconds']+=time.monotonic()-guard_started
        if active:raise RuntimeError('qc GPU priority: task96 yields before its next optimiser step')
        trace['steps'].append(dict(step=step,loss=loss))
    sys.argv=[str(SOURCE/'scripts/train_rollout_policy.py'),'--recipe',str(folder/'recipe.json')]
    started=time.monotonic()
    try:
        trainer.main(step_observer=observe)
        model=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        trace.update(status='complete',tensor_sha256=tensor_digest(model['state_dict']),
                     metrics=read(folder/'metrics.json'))
    except Exception:
        trace.update(status='error',error=traceback.format_exc())
    if trace.get('status')=='error' and 'qc GPU priority' in trace.get('error',''):
        trace['status']='yielded_to_qc'
    trace['wall_seconds']=time.monotonic()-started
    write(folder/'trace.json',trace)
    print(json.dumps({k:trace.get(k) for k in ('case','status','initial_tensor_sha256','tensor_sha256','error','wall_seconds')}),flush=True)


def wait_for_qc():
    last=0
    while qc_gpu_active():
        if time.monotonic()-last>=30:
            print(json.dumps(dict(waiting_for_qc_gpu=True,task96_training_active=False)),flush=True)
            last=time.monotonic()
        time.sleep(5)


def run():
    wait_for_qc()
    cases=['strict_no_workspace','default1','strict1','strict2','default2','ws16_a','ws16_b','nows_a','nows_b']
    for case in cases:
        wait_for_qc()
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'child','--case',case]
        extra=dict(os.environ)
        if case.startswith('ws16'):extra['TASK96_WORKSPACE']=':16:8'
        with (OUT/(case+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=150,env=extra)
        assert read(OUT/case/'trace.json')['status']!='yielded_to_qc', 'qc took GPU priority; diagnostic case is incomplete, not an operator error'
    traces=[read(OUT/case/'trace.json') for case in cases]
    summary=dict(task=96,scope=read(OUT/'input.json'),cases=traces,
                 cost_limit_seconds=750,production_changes=False)
    write(ROOT/'MEMORY/docs/task96-determinism-probe.json',summary)
    print(json.dumps(dict(statuses={t['case']:t['status'] for t in traces})),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','run','child']);p.add_argument('--case')
    a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='run':run()
    else:child(a.case)


if __name__=='__main__':main()
