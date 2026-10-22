"""Process-tree memory guard for the DAgger study; copied from the validated task8 runner."""
from pathlib import Path
import os, subprocess, signal, time, json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/cuda/task15'
PYTHON=str(Path.home()/'.venvs/ml/bin/python3')
def descendants(pid):
    result=[];todo=[pid]
    while todo:
        p=todo.pop();result.append(p)
        try:todo.extend(map(int,Path(f'/proc/{p}/task/{p}/children').read_text().split()))
        except (FileNotFoundError, ProcessLookupError):pass
    return result
def available():
    return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
def live(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except (FileNotFoundError, ProcessLookupError):return False
ENV={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','TORCHINDUCTOR_COMPILE_THREADS':'1',
     'MAX_JOBS':'1','MALLOC_ARENA_MAX':'2','PYTHONPATH':str(ROOT/'python')+':'+str(ROOT/'scripts'),
     'PVZ_EXECUTABLE':str(ROOT/'build/pvz-portable'),'PVZ_RESOURCE_DIR':str(Path.home()/'.cache/pvz-research-resources'),
     'PVZ_CUDA_BINARY':str(ROOT/'build/rollout_cuda_small'),
     'PVZ_CUDA_NET':'/home/newbiexvwu/pvz-cuda/artifacts/cuda/codex_r66/v28.pvzrn',
     'PVZ_CUDA_CORPUS':'/home/newbiexvwu/pvz-cuda/artifacts/cuda/corpus_l7_z15_v9/roots.cudaroot',
     'PVZ_CUDA_PRIVATE_LIMIT':'1200','PVZ_CUDA_COOPERATIVE_HEADS':'1','PVZ_CUDA_POLL_US':'0'}
ENV.pop('PVZ_GPU_PAUSE_FILE',None)
def run(cmd,tag,cwd=ROOT,extra=None,unset=(),pause_when=None):
    while available()<int(3.25*1024*1024):time.sleep(3)
    env={**ENV,**(extra or {})}
    for key in unset:env.pop(key,None)
    (OUT/(tag+'.command.json')).write_text(json.dumps({'command':cmd,'cwd':str(cwd),'extra':extra or {},'unset':unset},indent=2)+'\n')
    peak=swap=0;minimum=available();paused=set();start=time.monotonic()
    with (OUT/(tag+'.txt')).open('w') as log,(OUT/(tag+'.resources.jsonl')).open('w') as resources:
        p=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (OUT/(tag+'.process.json')).write_text(json.dumps({'pid':p.pid,'group':p.pid})+'\n')
        try:
            while p.poll() is None:
                a=available();minimum=min(minimum,a);owned=descendants(p.pid);rss=sw=0
                for pid in owned:
                    try:
                        lines=Path(f'/proc/{pid}/status').read_text().splitlines()
                        rss+=sum(int(l.split()[1]) for l in lines if l.startswith('VmRSS:'))
                        sw+=sum(int(l.split()[1]) for l in lines if l.startswith('VmSwap:'))
                    except (FileNotFoundError, ProcessLookupError):pass
                peak=max(peak,rss);swap=max(swap,sw)
                busy=bool(pause_when(owned)) if pause_when is not None else False
                if a<2*1024*1024 or busy:
                    for pid in reversed(owned):
                        if pid not in paused:
                            try:os.kill(pid,signal.SIGSTOP);paused.add(pid)
                            except ProcessLookupError:pass
                elif a>=3*1024*1024 and not busy and paused:
                    for pid in paused:
                        try:os.kill(pid,signal.SIGCONT)
                        except ProcessLookupError:pass
                    paused.clear()
                resources.write(json.dumps({'elapsed_s':time.monotonic()-start,'available_kib':a,'rss_kib':rss,'swap_kib':sw,'paused':bool(paused),**({'external_busy':busy} if pause_when is not None else {})})+'\n');resources.flush();time.sleep(1)
        finally:
            if p.poll() is None:
                for pid in reversed(descendants(p.pid)):
                    try:os.kill(pid,signal.SIGCONT);os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                p.wait(timeout=30)
    result={'returncode':p.returncode,'wall_s':time.monotonic()-start,'peak_rss_kib':peak,'peak_swap_kib':swap,'minimum_available_kib':minimum}
    (OUT/(tag+'.result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'stage':tag,**result}),flush=True)
    return result
