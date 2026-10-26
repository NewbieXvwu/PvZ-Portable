"""Wait for a verified existing task76 panel process, then run isolated probes serially.

Does not restart missing or failed panel jobs. A complete 8x32x3 receipt matrix and
zero owned game processes are required before a new stage can begin.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import roof_candidates_v5 as base


def process_token(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,ProcessLookupError):return None


def engines():
    return [game['pid'] for game in base.owned_game_processes()]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait-pid',type=int,required=True)
    args=p.parse_args();token=process_token(args.wait_pid)
    if token is None:raise RuntimeError('specified panel handle not live at handoff; inspect authoritative state first')
    base.write(base.OUT/'finish-run-handoff.json',dict(panel_pid=args.wait_pid,start_ticks=token,controller_pid=os.getpid()))
    last=0
    while process_token(args.wait_pid)==token:
        if time.monotonic()-last>40:
            complete=len(list((base.OUT/'panels').glob('l*/*/*/result.json')))
            print(json.dumps(dict(waiting_for_panel_pid=args.wait_pid,completed_receipts=complete,owned_game_pids=engines())),flush=True)
            last=time.monotonic()
        time.sleep(5)
    for level in range(42,50):
        for arm in ('production','all_plants','all_legal'):
            for seed in range(45000000,45000032):
                path=base.OUT/'panels'/f'l{level}'/arm/str(seed)/'result.json'
                if not path.exists():raise RuntimeError(f'panel stopped incomplete: {path}; not restarting automatically')
                assert base.read(path)['outcome']['terminal']
    # A dead controller is not proof its subprocesses exited. Observe native handles.
    while engines():
        print(json.dumps(dict(waiting_for_owned_game_processes=engines())),flush=True)
        time.sleep(5)
    analysis=ROOT/'scripts/roof_candidates_v5_analysis.py'
    commands=[['panel-report','--verify-traces'],['pilot','--levels','42','--workers','1'],
              ['terminal','--workers','2'],['terminal-report']]
    for options in commands:
        command=[sys.executable,'-B',str(analysis),*options]
        print(json.dumps(dict(starting=command)),flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
        assert not engines(),'owned game subprocess left alive after stage completion'
    base.write(base.OUT/'finish-runs-complete.json',dict(complete=True,commands=commands,all_native_processes_exited=True))


if __name__=='__main__':main()
