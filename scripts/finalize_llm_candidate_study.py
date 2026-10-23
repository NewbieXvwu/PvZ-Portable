"""Wait for a specific task52 collector, then validate and summarize its completed study.

Read-only with respect to games and frozen inputs: never starts/restarts a collector.
The stage refuses partial results. Raw outputs and status stay in the ignored study folder.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from llm_candidates import atomic_json


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':
            return None
        return fields[19]  # Linux starttime, field 22; a recycled PID is a different process.
    except FileNotFoundError:
        return None


def finalize(out):
    paths=list((out/'games').rglob('result.json'))
    if len(paths)!=144:
        raise ValueError(f'collector exited with {len(paths)}/144 results; inspect it, do not silently restart')
    # The strict auditor checks identities, seeds, lengths, hashes and all proposal events;
    # 144 arbitrary files alone cannot satisfy this gate.
    commands=[
        [sys.executable,str(ROOT/'scripts/audit_llm_candidates.py'),'--out',str(out)],
        [sys.executable,str(ROOT/'scripts/llm_candidate_experiment.py'),'summarize','--out',str(out)],
    ]
    for i,command in enumerate(commands):
        with (out/f'finalize-stage-{i}.log').open('w') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    audit=json.loads((out/'independent-audit.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    assert audit['status']==summary['status']=='complete'
    assert audit['games_verified']==144 and audit['contract']==summary['contract']
    for family,comparisons in audit['paired_comparison'].items():
        for arm,independent in comparisons.items():
            actual=summary['families'][family]['comparisons'][arm]
            for key,value in independent.items():
                assert actual[key]==value,(family,arm,key)
    return dict(status='analysis_ready',games=144,contract=audit['contract'],
                pending=['human-readable report and recommendation','full requirement completion audit'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/task52/study-v1')
    p.add_argument('--wait-for-pid',type=int)
    args=p.parse_args();out=args.out
    with (out/'finalizer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(status='waiting_for_collector',collector_pid=args.wait_for_pid)
        atomic_json(out/'finalizer-status.json',state)
        original=process_identity(args.wait_for_pid) if args.wait_for_pid else None
        while original is not None and process_identity(args.wait_for_pid)==original:
            time.sleep(10)
        try:
            state=finalize(out)
        except Exception as error:
            atomic_json(out/'finalizer-status.json',dict(status='incomplete',reason=str(error)))
            raise
        atomic_json(out/'finalizer-status.json',state)
        print(json.dumps(state),flush=True)


if __name__=='__main__':main()
