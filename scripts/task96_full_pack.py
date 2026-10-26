"""Task96 full-index control: fixed workspace, real trainer, isolated writable outputs.

Existing full default repeats and small operator probes are never repeated here.
This adds the missing full-pack deterministic control and a bracketing cost pair.
No game or production process is controlled. Only qc and this task affect training admission; other tasks are recorded as timing context. /proc admission works even when WSL
omits CUDA clients from nvidia-smi. All mutations stay in this worktree.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/task96/full_pack'
SOURCE = Path('/tmp/exp96')
CASES = [('strict_a', True), ('default_b', False), ('strict_c', True)]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    assert path.is_absolute() and path.is_relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1)+'\n')


def training_peers():
    found = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            args = directory.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            is_python=bool(args) and Path(args[0]).name.startswith('python')
            is_trainer = is_python and any(a.endswith('/scripts/train_rollout_policy.py') for a in args)
            is_own = is_python and str(Path(__file__).resolve()) in args and 'child' in args
            if is_trainer or is_own:
                found.append(dict(pid=int(directory.name), argv=args,
                                  start_ticks=directory.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def resources():
    memory = {line.split(':')[0]: int(line.split()[1])*1024
              for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    total, used, utilisation = [int(x.strip()) for x in result.stdout.splitlines()[0].split(',')]
    return dict(mem_available_bytes=memory['MemAvailable'], gpu_free_mib=total-used,
                gpu_used_mib=used, gpu_utilization_percent=utilisation)


def admission():
    peers = training_peers()
    usage = resources()
    priority=[peer for peer in peers if any('/.claude/worktrees/cq/scripts/train_rollout_policy.py' in arg
              or 'curriculum_state.pipeline/round_' in arg for arg in peer['argv'])
              or (str(Path(__file__).resolve()) in peer['argv'] and 'child' in peer['argv'])]
    return dict(peers=peers, priority_peers=priority, resources=usage,
                admitted=not priority and usage['mem_available_bytes'] >= 10*1024**3
                and usage['gpu_free_mib'] >= 3072)


def prepare():
    import numpy as np
    audit = read(ROOT/'MEMORY/docs/task96-existing-runs.json')
    recipe = dict(audit['completed_existing_runs'][0]['recipe'])
    index = Path(recipe['data'])/'packs/7d15b915a2192805.npz'
    with np.load(index, allow_pickle=False) as stored:
        paths = [index.parent/str(name) for name in stored['segments']]
        count = int(stored['rows'])
    segments = [dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in paths]
    assert len(segments) == 348 and count == 16982931
    plan = dict(schema=1, task=96, index=str(index), index_sha256=sha(index), segments=segments,
                rows=count, recipe=recipe, cases=CASES, cublas_workspace=':4096:8',
                timeout_seconds_per_case=900,
                purpose='New full-pack fixed-workspace deterministic control; strict/default/strict order for cost. Not a rerun of the existing uncontrolled default pair.')
    path=OUT/'plan.json'
    if path.exists():
        assert read(path)==plan, 'plan already exists with different inputs; preserve prior evidence'
    else:
        write(path, plan)
    print(json.dumps(dict(plan=str(path), rows=count, segments=len(segments), admission=admission())), flush=True)


def verify_inputs(plan):
    assert sha(Path(plan['index'])) == plan['index_sha256']
    for entry in plan['segments']:
        stat = Path(entry['path']).stat()
        assert (stat.st_size,stat.st_mtime_ns) == (entry['size'],entry['mtime_ns']), entry['path']


def child(case):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[name]='1'
    os.environ['PVZ_NO_POPUP']='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    plan=read(OUT/'plan.json'); verify_inputs(plan)
    mode=dict(CASES)[case]
    folder=OUT/case
    assert not (folder/'trace.json').exists(), 'never overwrite a previous control receipt'
    gate=admission()
    assert gate['admitted'], json.dumps(gate)
    sys.path[:0]=[str(SOURCE/'scripts'), str(SOURCE/'python'), str(ROOT/'scripts')]
    import torch
    import train_rollout_policy as trainer
    from task96_existing_runs import tensor_digest
    assert Path(trainer.__file__).resolve()==SOURCE/'scripts/train_rollout_policy.py'
    for entry in read(ROOT/'MEMORY/docs/task96-existing-runs.json')['source_snapshot']:
        assert sha(SOURCE/entry['path'])==entry['sha256']
    torch.use_deterministic_algorithms(mode)
    torch.backends.cudnn.deterministic=mode
    torch.backends.cudnn.benchmark=False
    recipe=dict(plan['recipe']);recipe.update(data=str(folder/'input'),out=str(folder),source_row_report=True)
    write(folder/'recipe.json',recipe)
    trace=dict(task=96,case=case,strict=mode,workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],
               plan_sha256=sha(OUT/'plan.json'),admission=gate,started=time.time(),
               environment=dict(torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                                gpu=torch.cuda.get_device_name(0),matmul_precision=torch.get_float32_matmul_precision()),
               permutations=[],losses=[],resource_samples=[],qc_guard_seconds=0.0)
    paths=[Path(entry['path']) for entry in plan['segments']]
    original_net=trainer.RolloutNet
    def network(*args,**kwargs):
        model=original_net(*args,**kwargs)
        trace['initial_tensor_sha256']=tensor_digest(model.state_dict())
        return model
    trainer.RolloutNet=network
    def load(*args,**kwargs):
        assert not kwargs.get('source_weight') and not recipe.get('row_weights')
        print(json.dumps(dict(case=case,phase='full_index_loading',segments=len(paths))),flush=True)
        value=trainer.read_segments(paths, data=folder/'input', row_observer=kwargs.get('row_observer'))
        print(json.dumps(dict(case=case,phase='loaded',rows=len(value[2]))),flush=True)
        assert len(value[2])==plan['rows']
        return value
    trainer.load=load
    original_randperm=torch.randperm
    def randperm(*args,**kwargs):
        result=original_randperm(*args,**kwargs)
        trace['permutations'].append(dict(rows=result.numel(),sha256=hashlib.sha256(result.numpy().tobytes()).hexdigest()))
        return result
    torch.randperm=randperm
    original_metrics=trainer.Metrics
    trainer.Metrics=lambda *args,**kwargs:original_metrics(*args,directory=folder/'telemetry',**kwargs)
    checked=0.0
    def observe(step,loss):
        nonlocal checked
        trace['losses'].append(loss)
        now=time.monotonic()
        if now-checked<2:
            return
        checked=now;guard_start=time.monotonic()
        gate=admission()
        if gate['priority_peers']:
            trace['priority_yield_peers']=gate['priority_peers']
            raise RuntimeError('task96 yields: qc or another task96 trainer became active; no production process was changed')
        if step%1000<100:
            trace['resource_samples'].append(dict(step=step,peers=gate['peers'],**gate['resources']))
        trace['qc_guard_seconds']+=time.monotonic()-guard_start
        if step%500<50:
            print(json.dumps(dict(case=case,step=step,loss=loss)),flush=True)
    sys.argv=[str(SOURCE/'scripts/train_rollout_policy.py'),'--recipe',str(folder/'recipe.json')]
    began=time.monotonic()
    try:
        trainer.main(step_observer=observe)
        verify_inputs(plan)
        checkpoint=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        trace.update(status='complete',model_file_sha256=sha(folder/'model.pt'),
                     tensor_sha256=tensor_digest(checkpoint['state_dict']),metrics=read(folder/'metrics.json'),
                     source_rows_sha256=sha(folder/'source_rows.json'))
        expected=read(ROOT/'MEMORY/docs/task96-existing-runs.json')['completed_existing_runs'][0]['source_rows_sha256']
        assert trace['source_rows_sha256']==expected, 'full source rows differ from original control'
    except Exception:
        trace.update(status='error',error=traceback.format_exc())
    trace.update(wall_seconds=time.monotonic()-began,finished=time.time(),final_resources=resources())
    trace['runtime_modules']={name:dict(path=str(Path(module.__file__).resolve()),sha256=sha(Path(module.__file__)))
        for name,module in list(sys.modules.items()) if getattr(module,'__file__',None)
        and Path(module.__file__).resolve().is_relative_to(SOURCE)}
    write(folder/'trace.json',trace)
    print(json.dumps({k:trace.get(k) for k in ['case','status','tensor_sha256','error','wall_seconds']}),flush=True)


def run():
    plan=read(OUT/'plan.json')
    for case,_ in CASES:
        path=OUT/case/'trace.json'
        if path.exists():
            assert read(path)['status']=='complete', 'prior case failed: inspect it rather than blindly retry'
            continue
        while True:
            gate=admission()
            if gate['admitted']:
                break
            print(json.dumps(dict(waiting=gate,case=case)),flush=True)
            time.sleep(30)
        log=OUT/(case+'.log')
        assert not log.exists(), 'prior launched case has no receipt: inspect before relaunch'
        with log.open('w') as output:
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'child','--case',case],
                           cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=900)
        assert read(path)['status']=='complete', f'{case} incomplete; preserve evidence and stop'
    traces=[read(OUT/case/'trace.json') for case,_ in CASES]
    for trace in traces:
        trace['losses_sha256']=hashlib.sha256(json.dumps(trace['losses']).encode()).hexdigest()
        trace['steps']=len(trace['losses']); trace['first_losses']=trace['losses'][:8]
        del trace['losses']
    strict=[t for t in traces if t['strict']];default=next(t for t in traces if not t['strict'])
    strict_seconds=[t['metrics']['history'][0]['seconds'] for t in strict]
    summary=dict(task=96,scope='Full 16,982,931-row index, seed1, epoch1; fixed workspace. Strict/default/strict bracketing; observed ratio is conditional on measured shared-host load.',
                 plan={k:v for k,v in plan.items() if k not in ['segments','recipe']},cases=traces,
                 strict_repeats_tensor_equal=strict[0]['tensor_sha256']==strict[1]['tensor_sha256'],
                 all_initial_tensors_equal=len({t['initial_tensor_sha256'] for t in traces})==1,
                 all_permutations_equal=all(t['permutations']==traces[0]['permutations'] for t in traces),
                 strict_default_epoch_time_ratio=sum(strict_seconds)/len(strict_seconds)/default['metrics']['history'][0]['seconds'])
    write(ROOT/'MEMORY/docs/task96-full-pack.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['cases','plan']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run','child']);parser.add_argument('--case',choices=dict(CASES))
    args=parser.parse_args()
    globals()[args.mode](args.case) if args.mode=='child' else globals()[args.mode]()
