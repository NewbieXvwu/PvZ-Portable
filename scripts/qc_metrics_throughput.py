"""Read-only production-controller telemetry and currently provable pipeline phases.

Uses a verified live production PID, not all metrics with a similar name. GPU samples
are device-wide; progress.games may be cumulative/wrong and is never used as output.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
GIB=2**30


def worker_formula(meta: dict | None, actual: int) -> dict:
    if not meta or meta.get('upper') is None or meta['memory']=='n/a' or meta['per_worker']=='unmeasured':
        return dict(available=False)
    available=float(meta['memory']);rss=float(meta['per_worker']);cap=int(meta['upper'])
    background=float(meta.get('background') or 0)
    background_rounding=.005 if meta.get('background') is not None else 0
    if rss<=.005:return dict(available=False)
    clamp=lambda n:max(1,min(cap,n))
    low=clamp(math.floor((available-.05-6-background-background_rounding)/(rss+.005)))
    high=clamp(math.floor((available+.05-6-background+background_rounding)/(rss-.005)))
    return dict(available=True,actual=actual,predicted_min=low,predicted_max=high,
                consistent=low<=actual<=high and int(meta['workers'])==actual,
                background_headroom_gib=background,
                note='Range accounts for rounded log fields; reserve is 6 GiB plus logged unconsumed background headroom.')


def weighted_window(rows: list[dict], start: float, end: float) -> dict:
    covered=0.;memory_sum=0.;memory_time=0.;cpu_sum=0.;cpu_time=0.
    for previous,current in zip(rows,rows[1:]):
        delta=current['timestamp']-previous['timestamp']
        # A missing sample is unknown time, not continuous observation or zero usage.
        if delta<=0 or delta>61:continue
        seconds=max(0.,min(end,current['timestamp'])-max(start,previous['timestamp']))
        if not seconds:continue
        covered+=seconds
        memory=previous.get('mem_available_bytes')
        if memory is not None:memory_sum+=memory*seconds;memory_time+=seconds
        cpu=current.get('tree_cpu_percent')
        if cpu is not None:cpu_sum+=cpu*seconds;cpu_time+=seconds
    duration=end-start
    inside=[r for r in rows if start<=r['timestamp']<=end]
    values=[r['mem_available_bytes'] for r in inside if r.get('mem_available_bytes') is not None]
    rss_values=[r['rss_bytes']+r['children_rss_bytes'] for r in inside
                if r.get('rss_bytes') is not None and r.get('children_rss_bytes') is not None]
    return dict(window_seconds=duration,observed_seconds=covered,
                coverage=covered/duration if duration>0 else None,
                memory_coverage=memory_time/duration if duration>0 else None,
                cpu_coverage=cpu_time/duration if duration>0 else None,
                mem_available_mean_gib=memory_sum/memory_time/GIB if memory_time else None,
                mem_available_min_gib=min(values)/GIB if values else None,
                mem_available_max_gib=max(values)/GIB if values else None,
                tree_cpu_mean_percent=cpu_sum/cpu_time if cpu_time else None,
                sampled_tree_rss_peak_gib=max(rss_values)/GIB if rss_values else None)


def proc_records() -> dict:
    boot=next(int(l.split()[1]) for l in Path('/proc/stat').read_text().splitlines() if l.startswith('btime '))
    ticks=os.sysconf('SC_CLK_TCK');page=os.sysconf('SC_PAGE_SIZE');result={}
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():continue
        try:
            fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
            args=(folder/'cmdline').read_bytes().split(b'\0');args=[a.decode() for a in args if a]
            result[int(folder.name)]=dict(pid=int(folder.name),ppid=int(fields[1]),argv=args,
                started_at=boot+int(fields[19])/ticks,cpu_seconds=(int(fields[11])+int(fields[12]))/ticks,
                rss_bytes=int(fields[21])*page)
        except (OSError,ValueError,UnicodeError,IndexError):continue
    return result


def verify_production(record: dict) -> dict:
    argv=record['argv'];script=next((a for a in argv if a.endswith('/curriculum_queue.py')),None)
    if not script:raise ValueError('PID is not a curriculum controller')
    option=lambda key,default:argv[argv.index(key)+1] if key in argv else default
    expected=Path('/home/newbiexvwu/pvz-logs')
    if Path(option('--state',str(expected/'curriculum_state.json')))!=expected/'curriculum_state.json':
        raise ValueError('isolated/nonproduction state; do not attribute its metrics to qc')
    if Path(option('--log',str(expected/'curriculum.log')))!=expected/'curriculum.log':
        raise ValueError('isolated/nonproduction log')
    result={**record,'script':script,'source_sha256':hashlib.sha256(Path(script).read_bytes()).hexdigest()}
    frozen=ROOT/'artifacts/task51/cq-source/scripts/curriculum_queue.py'
    result['matches_frozen_cq_source']=frozen.exists() and frozen.read_bytes()==Path(script).read_bytes()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history',type=Path,required=True)
    p.add_argument('--controller-pid',type=int,required=True)
    p.add_argument('--metrics',type=Path,default=Path('/home/newbiexvwu/pvz-logs/metrics'))
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/task51/runtime')
    args=p.parse_args()
    if not all(x.is_absolute() for x in (args.history,args.metrics,args.out)):p.error('paths must be absolute')
    table=proc_records()
    if args.controller_pid not in table:raise ValueError('production PID is no longer live; inspect authoritative state')
    controller=verify_production(table[args.controller_pid]);cutoff=time.time()
    folder=args.out/datetime.now().strftime('%Y%m%dT%H%M%S');folder.mkdir(parents=True,exist_ok=False)
    sources={};rows=[]
    files=sorted(args.metrics.glob(f'curriculum_queue-{args.controller_pid}*.jsonl'))
    if not files:raise ValueError('verified controller has no telemetry files')
    for source in files:
        target=folder/source.name;shutil.copyfile(source,target);data=target.read_bytes()
        sources[source.name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        for line in data.splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:continue # Active file may end with an incomplete write.
            if row.get('event')=='sample' and row.get('pid')==args.controller_pid:
                if controller['started_at']-2<=row['timestamp']<=cutoff:rows.append(row)
    rows=sorted({(r['timestamp'],r['monotonic_s']):r for r in rows}.values(),key=lambda r:r['timestamp'])
    history=json.loads(args.history.read_text());windows=[]
    for run in history['runs']:
        if run['complete'] and run['started_at']>=controller['started_at']:
            windows.append(dict(task_id=run['task_id'],label=run['label'],workers=run['workers'],
                mem_available_before_launch_gib=run['mem_available_gib'],
                worker_formula=worker_formula(run['worker_log'],run['workers']),
                telemetry=weighted_window(rows,run['started_at'],run['ended_at'])))
    owned={args.controller_pid}
    while True:
        added={pid for pid,d in table.items() if d['ppid'] in owned}-owned
        if not added:break
        owned.update(added)
    children=[]
    for pid in sorted(owned-{args.controller_pid}):
        record=table[pid]
        script=next((s for s in record['argv'] if s.endswith(('/pack_training_rows.py','/train_rollout_policy.py','/search_play.py'))),None)
        if script:
            window=weighted_window(rows,record['started_at'],cutoff)
            children.append(dict(pid=pid,ppid=record['ppid'],script=script,started_at=record['started_at'],
                observed_live_at=cutoff,elapsed_seconds=cutoff-record['started_at'],cpu_seconds=record['cpu_seconds'],
                rss_gib=record['rss_bytes']/GIB,telemetry=window,
                telemetry_scope='Entire production controller tree during this child window; overlapping phases are not separable.'))
    active_training=[]
    for pid,record in table.items():
        if pid not in owned and any(s.endswith('/train_rollout_policy.py') for s in record['argv']):
            active_training.append(dict(pid=pid,ppid=record['ppid'],started_at=record['started_at']))
    result=dict(controller=controller,cutoff=cutoff,history_file=str(args.history),
        history_sha256=hashlib.sha256(args.history.read_bytes()).hexdigest(),metrics_sources=sources,
        samples=len(rows),collection_windows=windows,live_descendant_phases=children,
        other_live_training=active_training,
        notes=['GPU telemetry is device-wide, not proof that qc is waiting for training.',
               'progress.games is not used: it may count cumulative reused directories.',
               'Only the verified production controller is joined; smoke/benchmark PIDs are excluded.',
               'Telemetry gaps over 61 seconds remain unobserved.',
               'Interval averages approximate boundary splits; sampled peaks use only timestamps inside the phase.',
               'Descendant phase duration is right-censored while the process is live.'])
    (folder/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.out/'latest.json').write_text(json.dumps(dict(folder=str(folder)))+'\n')
    print(json.dumps(dict(folder=str(folder),samples=len(rows),collection_windows=len(windows),
        formula_checked=sum(w['worker_formula']['available'] for w in windows),
        formula_mismatches=[w['label'] for w in windows if w['worker_formula'].get('consistent') is False],
        live_phases=children,other_live_training=active_training),indent=2))


if __name__=='__main__':main()
