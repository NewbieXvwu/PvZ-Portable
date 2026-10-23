"""Read-only task51 reconstruction of completed qc collection segments and cycles.

Controller games/progress can be cumulative across reused labels. Count this invocation's
completed jobs from its own search_play start/final progress, never its directory total.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ZoneInfo('Asia/Shanghai')
STAMP=r'(?P<time>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:[+-]\d\d:\d\d)?)'
START=re.compile(r'^(\d+) games on (\d+) workers$')
PROGRESS=re.compile(r'^(\d+)/(\d+) \((\d+)s\)$')
DONE=re.compile('^'+STAMP+r' round (?P<round>\d+) (?P<label>\S+) (?P<task>\S+) mult=(?P<mult>\S+) offset=(?P<offset>\d+) games=(?P<games>\d+) wins=\d+ rate=\S+ rc=(?P<rc>-?\d+) (?P<seconds>\d+)s$')
WORKERS=re.compile('^'+STAMP+r' round \d+ (?P<label>\S+): workers=(?P<workers>\d+)/\d+ \(MemAvailable (?P<memory>[\d.]+|n/a)(?: GiB)?, per-worker (?P<per_worker>[\d.]+|unmeasured)(?: GiB)?\)$')
ROUND=re.compile('^'+STAMP+r' round (?P<round>\d+) done: (?P<groups>\d+) groups, (?P<games>\d+) games,')
BUILD=re.compile(r'^INFO: BuildNum: (\d+)$')


def stamp(value: str) -> float:
    date=datetime.fromisoformat(value)
    return (date if date.tzinfo else date.replace(tzinfo=LOCAL)).timestamp()


def parse(lines, cutoff: float) -> dict:
    runs=[];rounds=[];orphans=[];pending=None;workers=None;seen=set()
    for index,text in enumerate(lines):
        line=text.strip()
        if match:=WORKERS.match(line):
            workers=match.groupdict();continue
        if match:=START.match(line):
            if pending is not None:orphans.append(pending)
            pending=dict(jobs=int(match[1]),workers=int(match[2]),progress=0,
                         start_line=index,build_numbers=[],worker_log=workers)
            workers=None;continue
        if pending is not None:
            if match:=BUILD.match(line):
                number=int(match[1])
                if number not in pending['build_numbers']:pending['build_numbers'].append(number)
            if match:=PROGRESS.match(line):
                if int(match[2])==pending['jobs']:pending['progress']=max(pending['progress'],int(match[1]))
        if match:=DONE.match(line):
            raw=match.groupdict();key=tuple(raw.items())
            if key in seen:continue
            seen.add(key);seconds=int(raw['seconds']);end=stamp(raw['time']);rc=int(raw['rc'])
            jobs=None if pending is None else pending['jobs']
            verified=(pending is not None and rc==0 and pending['progress']==jobs
                      and jobs<=int(raw['games']) and seconds>0)
            meta=None if pending is None else pending['worker_log']
            if meta is not None and meta['label']!=raw['label']:meta=None
            runs.append(dict(task_id=raw['task'],label=raw['label'],round=int(raw['round']),
                             multiplier=float(raw['mult']),offset=int(raw['offset']),
                             reported_cumulative_games=int(raw['games']),requested_jobs=jobs,
                             new_valid_games=jobs if verified else None,complete=verified,
                             returncode=rc,wall_seconds=seconds,started_at=end-seconds,ended_at=end,
                             workers=None if pending is None else pending['workers'],
                             mem_available_gib=(None if meta is None or meta['memory']=='n/a' else float(meta['memory'])),
                             worker_log=meta,build_numbers=[] if pending is None else pending['build_numbers'],
                             finish_progress=None if pending is None else pending['progress'],line=index,
                             start_line=index if pending is None else pending['start_line']))
            pending=None;workers=None;continue
        if match:=ROUND.match(line):
            raw=match.groupdict();count=int(raw['groups']);group=runs[-count:]
            valid=(len(group)==count and all(x['complete'] and x['round']==int(raw['round']) for x in group)
                   and sum(x['new_valid_games'] for x in group)==int(raw['games']))
            rounds.append(dict(round=int(raw['round']),line=index,ended_at=stamp(raw['time']),
                               planned_games=int(raw['games']),groups=[x['line'] for x in group],
                               complete_collection=valid))
    if pending is not None:orphans.append(pending)
    for round_ in rounds:
        if not round_['complete_collection']:continue
        group=[r for r in runs if r['line'] in round_['groups']]
        if any(group[0]['start_line']<=x['start_line']<round_['line'] for x in orphans):
            round_['complete_collection']=False
            round_['cycle_invalid']='unmatched invocation inside proposed round';continue
        start=group[0]['started_at'];following=[]
        for r in runs:
            if r['start_line']>round_['line']:following.append((r['start_line'],r['started_at'],'next_completion_start'))
        for item in orphans:
            if item['start_line']>round_['line']:
                meta=item['worker_log']
                following.append((item['start_line'],stamp(meta['time']) if meta else None,'next_worker_log'))
        next_start=min(following,key=lambda x:x[0]) if following else None
        if next_start is not None and next_start[1] is None:
            round_['cycle_invalid']='next invocation has no recoverable start timestamp';continue
        end=next_start[1] if next_start else cutoff
        if end<round_['ended_at'] or end<=start:
            round_['cycle_invalid']='overlapping or invalid chronology';continue
        collection=sum(r['wall_seconds'] for r in group)
        worker_hours=sum(r['workers']*r['wall_seconds'] for r in group)/3600
        round_.update(started_at=start,cycle_end=end,cycle_complete=next_start is not None,
                      cycle_end_source=next_start[2] if next_start else 'right_censor_cutoff',
                      cycle_wall_seconds=end-start,collection_wall_seconds=collection,
                      new_valid_games=sum(r['new_valid_games'] for r in group),worker_hours=worker_hours,
                      collection_games_per_hour=sum(r['new_valid_games'] for r in group)*3600/collection,
                      cycle_games_per_hour_or_upper_bound=sum(r['new_valid_games'] for r in group)*3600/(end-start),
                      collection_games_per_worker_hour=sum(r['new_valid_games'] for r in group)/worker_hours,
                      noncollection_seconds_lower_bound=max(0,end-start-collection))
    return dict(runs=runs,rounds=rounds,orphan_starts=orphans,cutoff=cutoff)


def aggregate(runs: list[dict]) -> list[dict]:
    groups={}
    for row in runs:
        if not row['complete']:continue
        key=(row['task_id'],row['multiplier'],tuple(row['build_numbers']))
        group=groups.setdefault(key,dict(task_id=key[0],multiplier=key[1],build_numbers=list(key[2]),
                                         runs=0,games=0,wall_seconds=0,worker_hours=0,workers=[],mem_available_gib=[]))
        group['runs']+=1;group['games']+=row['new_valid_games'];group['wall_seconds']+=row['wall_seconds']
        group['worker_hours']+=row['workers']*row['wall_seconds']/3600;group['workers'].append(row['workers'])
        if row['mem_available_gib'] is not None:group['mem_available_gib'].append(row['mem_available_gib'])
    for group in groups.values():
        group['games_per_hour']=group['games']*3600/group['wall_seconds']
        group['games_per_worker_hour']=group['games']/group['worker_hours']
    return list(groups.values())


def audit_raw_rows(runs: list[dict], root: Path) -> list[dict]:
    """Corroborate every counted invocation with its own slice of durable unique games.

    Directory/config totals may be newer than the invocation. Current manifest seeds are
    deliberately not used as historical truth: some old manifests have been regenerated.
    """
    cache={};audits=[]
    for run in runs:
        if not run['complete']:continue
        label=run['label']
        if label not in cache:
            candidates=[]
            for folder in root.glob(label+'_*'):
                file=folder/'games.jsonl';config=folder/'config.json'
                if not file.is_file() or not config.is_file():continue
                c=json.loads(config.read_text())
                if c.get('frames_label')!=label or c.get('task_id')!=run['task_id']:continue
                data=file.read_bytes();rows=[json.loads(s) for s in data.splitlines()]
                candidates.append((file,rows,hashlib.sha256(data).hexdigest()))
            if len(candidates)!=1:raise ValueError(f'{label}: ambiguous/missing raw game evidence')
            cache[label]=candidates[0]
        file,rows,digest=cache[label];end=run['reported_cumulative_games'];n=run['new_valid_games']
        part=rows[end-n:end];seeds=[x['seed'] for x in part]
        if len(part)!=n or len(set(seeds))!=n or any(type(x.get('won')) is not bool for x in part):
            raise ValueError(f'{label}: completed log does not match unique durable game rows')
        audits.append(dict(label=label,offset=run['offset'],games=n,raw_file=str(file),
                           raw_file_sha256=digest,seeds=sorted(seeds)))
    return audits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs',type=Path,default=Path('/home/newbiexvwu/pvz-logs'))
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/task51/history')
    parser.add_argument('--runs',type=Path,default=Path('/home/newbiexvwu/PvZ-search/artifacts/hard'))
    args=parser.parse_args()
    if not all(p.is_absolute() for p in (args.logs,args.out,args.runs)):parser.error('paths must be absolute')
    # Every analysis is a new immutable cut: active production logs may keep growing.
    cutoff=datetime.now().timestamp();folder=args.out/datetime.now().strftime('%Y%m%dT%H%M%S')
    folder.mkdir(parents=True,exist_ok=False);inputs={}
    for name in ('qc.out','curriculum.log','curriculum_state.json'):
        src=args.logs/name;target=folder/name
        if src.exists():
            shutil.copyfile(src,target)
            inputs[name]=dict(bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    with (folder/'qc.out').open(errors='replace') as handle:result=parse(handle,cutoff)
    result['raw_row_audit']=audit_raw_rows(result['runs'],args.runs)
    result['cohorts']=aggregate(result['runs']);result['sources']=inputs
    result['notes']=['Only complete invocation segments with matching start/final progress are counted.',
                     'Cumulative controller games are retained only as audit data.',
                     'Open cycles are right-censored: their throughput is an upper bound.',
                     'Worker/memory/BuildNum missing from the source remain unknown.',
                     'No causal ratio is implied across different task mixtures or versions.']
    (folder/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.out/'latest.json').write_text(json.dumps(dict(folder=str(folder)))+'\n')
    print(json.dumps(dict(folder=str(folder),completed_runs=sum(r['complete'] for r in result['runs']),
                          excluded_runs=sum(not r['complete'] for r in result['runs']),
                          orphan_starts=len(result['orphan_starts']),
                          corrected_games=sum(r['new_valid_games'] for r in result['runs'] if r['complete']),
                          naive_cumulative_games=sum(r['reported_cumulative_games'] for r in result['runs'] if r['complete']),
                          rounds=result['rounds']),indent=2))


if __name__=='__main__':main()
