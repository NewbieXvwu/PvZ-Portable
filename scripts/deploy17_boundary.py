#!/usr/bin/env python3
"""Prepared deploy17 helper. Called only by the reviewed deployment script, never by qc.

Freeze the scheduler briefly, resume it if a collector is active, and wait for that
group's log line. An idle frozen scheduler cannot race into the next collection group.
All signals are confined to its recorded process tree; no name-based process killing.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import re


def command(pid):
    try: return (Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
    except (OSError,UnicodeError): return []


def identity(pid):
    try: return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError): return None


def descendants(pid):
    result=set()
    pending=[pid]
    while pending:
        parent=pending.pop()
        for path in (Path('/proc')/str(parent)/'task').glob('*/children'):
            try: children=[int(x) for x in path.read_text().split()]
            except OSError: continue
            for child in children:
                if child not in result: result.add(child);pending.append(child)
    return result


def last_group_event(log):
    # Bounded tail only; never reread the whole rolling log.
    with log.open('rb') as stream:
        stream.seek(max(0,log.stat().st_size-65536))
        lines=stream.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        if ': workers=' in line: return 'running',None
        if ' rc=' in line or ': group skipped:' in line:
            match=re.search(r'\bround \d+ (\S+) ',line)
            if match: return 'finished',match[1].rstrip(':')
    return 'unknown',None


def committed_boundary(log,state):
    event,label=last_group_event(log)
    if event!='finished': return False
    # rc is logged BEFORE completed_groups is saved. Only accept its durable label.
    # One small state-file byte scan (~10 MB here), no JSON object graph or archive replay.
    return b'"label": '+json.dumps(label).encode() in state.read_bytes()


def freeze_if_idle(pid, worktree, output=None, ready=None):
    token=identity(pid)
    if token is None: return None
    os.kill(pid,signal.SIGSTOP)
    accepted=False
    try:
        # SIGSTOP delivery is asynchronous. Wait only for this scheduler, never for a game replay.
        for _ in range(100):
            try: status=(Path('/proc')/str(pid)/'status').read_text()
            except FileNotFoundError: return None
            if any(line.startswith('State:') and line.split()[1] in ('T','t') for line in status.splitlines()): break
            time.sleep(.01)
        else: raise RuntimeError('scheduler did not stop')
        children=descendants(pid)
        # Collection subprocess.run is spawned by the scheduler's main thread.
        # Background selection/gate workers belong to its other threads and do not
        # turn this group-boundary wait into a wait for the whole training pipeline.
        main_children=(Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()
        if any(str(worktree/'scripts/search_play.py') in command(int(child)) for child in main_children): return None
        if ready is not None and not ready(): return None
        result=dict(controller=dict(pid=pid,start=token),children=[dict(pid=p,start=identity(p)) for p in children])
        if output is not None: output.write_text(json.dumps(result)+'\n')
        accepted=True
        return result
    finally:
        if not accepted and identity(pid)==token:
            try: os.kill(pid,signal.SIGCONT)
            except ProcessLookupError: pass


def act(record, signum):
    pid=record['pid']
    if record['start'] is not None and identity(pid)==record['start']:
        try: os.kill(pid,signum)
        except ProcessLookupError: pass


def wait_boundary(worktree, log, output):
    needle=str(worktree/'scripts/curriculum_queue.py')
    position=log.stat().st_size if log.exists() else 0
    while True:
        matches=[int(p.name) for p in Path('/proc').glob('[0-9]*') if needle in command(int(p.name))]
        if len(matches)!=1: raise RuntimeError(f'expected one cq scheduler, found {len(matches)}')
        if freeze_if_idle(matches[0],worktree,output,
                          lambda:committed_boundary(log,log.parent/'curriculum_state.json')): return
        if last_group_event(log)[0]=='finished':
            # Let the just-finished group's state write complete; a new workers event
            # or live child sends us back to waiting for that next group's completion.
            time.sleep(.01);continue
        started=time.monotonic();reported=started
        while True:
            time.sleep(.02)
            with log.open() as stream:
                if log.stat().st_size<position: position=0
                stream.seek(position);new=stream.read();position=stream.tell()
            if ' rc=' in new: break
            if time.monotonic()-reported>=30:
                print(f'waiting for collection boundary: {time.monotonic()-started:.0f}s',flush=True)
                reported=time.monotonic()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('wait','resume','stop'))
    p.add_argument('--worktree',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--record',type=Path,required=True)
    args=p.parse_args()
    if args.mode=='wait': wait_boundary(args.worktree,args.log,args.record);return
    record=json.loads(args.record.read_text())
    if args.mode=='resume': act(record['controller'],signal.SIGCONT);return
    controller=record['controller']
    if identity(controller['pid'])==controller['start']:
        # Background builders may have forked since the boundary record was saved.
        for pid in descendants(controller['pid']): act(dict(pid=pid,start=identity(pid)),signal.SIGTERM)
    for child in record['children']: act(child,signal.SIGTERM)
    act(record['controller'],signal.SIGTERM)
    act(record['controller'],signal.SIGCONT)


if __name__=='__main__': main()
