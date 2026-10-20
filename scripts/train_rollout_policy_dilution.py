"""One sequential training process; reuse immutable rows, never model/optimizer state."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout, redirect_stderr
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]


def foreign_trainers():
    result=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(Path(v.decode(errors='replace')).name.startswith('train_rollout_policy')
               and v.endswith(b'.py') for v in args[1:] if v):
            result.append(int(entry.name))
    return result


class ImmutableDataCache:
    """Only weight is mutable in the original trainer; return a fresh copy each run."""
    def __init__(self,load):
        self.load=load;self.key=None;self.payload=None;self.observations=[];self.loads=0

    @staticmethod
    def freeze(value):
        if isinstance(value,dict):return tuple(sorted((k,ImmutableDataCache.freeze(v)) for k,v in value.items()))
        if isinstance(value,(set,frozenset)):return tuple(sorted(value))
        if isinstance(value,(list,tuple)):return tuple(ImmutableDataCache.freeze(v) for v in value)
        return value

    def __call__(self,*args,**kwargs):
        observer=kwargs.pop('row_observer',None)
        key=(self.freeze(args),self.freeze(kwargs))
        if key!=self.key:
            self.payload=None;self.observations=[];gc.collect()
            def record(path,rows):self.observations.append((path,rows.copy()))
            self.payload=self.load(*args,**kwargs,row_observer=record)
            assert len(self.payload)==9
            self.key=key;self.loads+=1
        if observer is not None:
            for path,rows in self.observations:observer(path,rows)
        # All features and targets are read-only in batch_loss. Row weighting is
        # the one in-place transformation in main; never share it between runs.
        result=list(self.payload);result[5]=result[5].clone()
        return tuple(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    plan_hash=hashlib.sha256(args.plan.read_bytes()).hexdigest()
    for name,digest in plan.get('source_files',{}).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    import train_rollout_policy as trainer
    import torch
    cache=ImmutableDataCache(trainer.load);trainer.load=cache
    completed=[]
    for job in plan['jobs']:
        dest=Path(job['out']);dest.mkdir(parents=True,exist_ok=True)
        done=dest/'experiment_complete.json'
        if done.exists():
            record=json.loads(done.read_text())
            assert record['run_plan_sha256']==plan_hash, ('changed run plan',job['name'])
            assert hashlib.sha256((dest/'model.pt').read_bytes()).hexdigest()==record['model_sha256']
            completed.append(record);continue
        while foreign_trainers():
            print(json.dumps({'waiting_for_foreign_trainers':foreign_trainers()}),flush=True)
            time.sleep(10)
        print(json.dumps({'starting_job':job['name'],'seed':job['seed']}),flush=True)
        start=time.monotonic()
        command=[str(ROOT/'scripts/train_rollout_policy.py'),*plan['common_args'],
                 '--labels',','.join(plan['cohorts'][job['cohort']]['labels']),
                 '--seed',str(job['seed']),'--out',str(dest),*job.get('extra_args',[])]
        old=sys.argv
        try:
            sys.argv=command
            with (dest/'train.txt').open('w',buffering=1) as log,redirect_stdout(log),redirect_stderr(log):
                trainer.main()
        finally:sys.argv=old
        model=dest/'model.pt';assert model.is_file()
        record={'name':job['name'],'seed':job['seed'],'cohort':job['cohort'],
                'wall_s':time.monotonic()-start,'command':command,
                'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
                'data_loads_so_far':cache.loads,'run_plan_sha256':plan_hash,
                'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
                'torch_num_threads':torch.get_num_threads()}
        done.write_text(json.dumps(record,indent=2)+'\n');completed.append(record)
        print(json.dumps({'completed_job':record}),flush=True)
        gc.collect();torch.cuda.empty_cache()
    (args.plan.parent/'training_complete.json').write_text(json.dumps(completed,indent=2)+'\n')


if __name__=='__main__':main()
