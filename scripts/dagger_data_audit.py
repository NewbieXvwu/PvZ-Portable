"""Read-only per-episode DAgger audit, aligned to the trainer's cached row identities."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
from pvz_frame_store import frame_paths

SIGNATURE = None


def init_worker():
    global SIGNATURE
    from pvz_frame_store import encoder_signature
    SIGNATURE = encoder_signature('v2')


def action_key(a):
    if not a:
        return None
    return ('plant', a['packet'], a['row'], a['col']) if a['type']=='plant' else (a['type'],)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan_one(job):
    from pvz_rollout_policy import LAWN_COLS
    from pvz_frame_store import candidate_cells
    path, out, source, cache = map(Path, job)
    ident = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    dest = out/'episodes'/ident
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.parent/'meta.json'
    stat = path.stat()
    if dest.with_suffix('.json').exists():
        old = json.loads(dest.with_suffix('.json').read_text())
        if (old['source_bytes']==stat.st_size and old['source_mtime_ns']==stat.st_mtime_ns
                and old['meta_sha256']==sha(meta_path) and dest.with_suffix('.npz').exists()):
            return old
    task = json.loads(meta_path.read_text())
    with np.load(path, allow_pickle=False) as z:
        raw = z['frames'].tobytes()
    cached = cache/'v2'/(ident+'.npz')
    assert cached.exists(), ('required existing v2 feature cache absent', str(path))
    with np.load(cached, allow_pickle=False) as z:
        rows=z['rows'];hints=z['hints'];plant=z['plant'];packet=z['packet'];cell=z['cell'];values=z['values']
        cache_digest=str(z['digest'])
    assert cache_digest==hashlib.sha256(SIGNATURE.encode()+raw).hexdigest(), ('cache digest',str(path))
    n=len(rows)
    assert len(plant)==len(packet)==len(cell)==len(values)==n
    known=np.isfinite(values);compared=known.sum(1)>=2
    spread=np.zeros(n,np.float32)
    spread[compared]=(np.where(known,values,-1e9).max(1)-np.where(known,values,1e9).min(1))[compared]
    override=np.zeros(n,bool);exact_override=np.zeros(n,bool);cell_soft=np.zeros(n,bool);annotated=np.zeros(n,bool)
    cursor=0
    # Parse one frame at a time: neither giant Python episode arrays nor boards are loaded.
    for index,line in enumerate(raw.splitlines()):
        if cursor==n or index!=rows[cursor]:
            continue
        frame=json.loads(line)
        label=frame.get('label') or frame.get('action');executed=frame.get('action')
        assert label and not frame.get('action_rejected')
        assert (label['type']=='plant')==bool(plant[cursor])
        if plant[cursor]:
            assert label['packet']==packet[cursor] and label['row']*LAWN_COLS+label['col']==cell[cursor]
        annotated[cursor]=frame.get('label') is not None
        override[cursor]=action_key(label)!=action_key(executed)
        exact_override[cursor]=label!=executed
        if plant[cursor]:
            cell_soft[cursor]=sum(int(p)==label['packet'] and np.isfinite(v)
                                  for p,c,v in candidate_cells(frame.get('targets',[])))>=2
        cursor+=1
    assert cursor==n and len(hints)==len(raw.splitlines())
    assert stat.st_size==path.stat().st_size and stat.st_mtime_ns==path.stat().st_mtime_ns
    # metadata arrays contain no boards and do not replace or alter the source archives.
    temp=dest.with_suffix('.tmp.npz')
    np.savez_compressed(temp, rows=rows, hints=hints, packet=packet.astype(np.int8),
                        spread=spread, compared=compared, override=override,
                        exact_override=exact_override, cell_soft=cell_soft, annotated=annotated)
    temp.replace(dest.with_suffix('.npz'))
    record={'source':str(path),'relative_source':str(path.relative_to(source)),
            'label':path.parent.name,'seed':int(path.parent.parent.name.split('_')[-1]),
            'multiplier':float(task['context']['multiplier']),
            'raw_frames':len(hints),'valid_rows':n,'plant_rows':int(plant.sum()),
            'override_rows':int(override.sum()),'raw_object_override_rows':int(exact_override.sum()),
            'compared_rows':int(compared.sum()),'cell_soft_rows':int(cell_soft.sum()),
            'spread_sum':float(spread[compared].astype(np.float64).sum()),
            'source_sha256':sha(path),'source_frames_sha256':hashlib.sha256(raw).hexdigest(),
            'source_bytes':stat.st_size,'source_mtime_ns':stat.st_mtime_ns,'meta_sha256':sha(meta_path),
            'cache_digest':cache_digest,'audit_arrays':str(dest.with_suffix('.npz'))}
    tmp=dest.with_suffix('.tmp.json');tmp.write_text(json.dumps(record)+'\n');tmp.replace(dest.with_suffix('.json'))
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path.home()/'PvZ-search/artifacts/student/frames_v2')
    p.add_argument('--cache',type=Path,default=Path.home()/'PvZ-search/artifacts/student/v2_cache')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--historical',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    models=json.loads(args.historical.read_text());labels=set(models['d35cs3w8s1']['labels'].split(','))
    for name in ('d35cs3w8s2','d35cs3w8s3','d35bals1','d35bals2'):
        assert set(models[name]['labels'].split(','))==labels,name
    paths=[v for v in frame_paths(args.source) if v.parent.name in labels]
    assert {v.parent.name for v in paths}==labels
    args.out.mkdir(parents=True,exist_ok=True)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(scan_one,(str(v),str(args.out),str(args.source),str(args.cache))) for v in paths]
        for f in as_completed(futures):
            results.append(f.result())
            if len(results)%500==0:print(json.dumps({'scanned':len(results),'total':len(paths)}),flush=True)
    results.sort(key=lambda v:v['source'])
    (args.out/'inventory.json').write_text(json.dumps(results,indent=1)+'\n')
    print(json.dumps({'complete':len(results),'labels':len(labels)}),flush=True)


if __name__=='__main__':main()
