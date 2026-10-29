"""Populate this run's missing side caches from the identical frozen parent encoder.

Parent files are read only. Complete private copies are published exclusively and
atomically; active workers may replace them through their existing atomic writer.
The unchanged training reader still checks source+encoder digests on every use.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from task27_common import ROOT, MAIN, OUT, read, write, sha
from run_dw5_study import verify


def main():
    plan=read(OUT/'study_plan.json');verify(plan)
    old=MAIN/'artifacts/archive/dw4/task23'
    parent=read(old/'study_plan.json')
    assert parent['sources']['python/pvz_frame_store.py']==sha(ROOT/'python/pvz_frame_store.py')
    allowed={hashlib.sha256(str((Path(row['source'])/'frames.npz').resolve()).encode()).hexdigest()+'.npz'
             for row in read(OUT/'dataset_inventory.json')}
    assert len(allowed)==plan['archives']
    counts={}
    inventory=OUT/'cache_reuse_inventory.jsonl'
    assert not inventory.exists(), 'Inspect an earlier run instead of overwriting its evidence'
    with inventory.open('x') as evidence:
        for kind,field in (('packets','packets'),('identity_scene','wave_scalar')):
            target_dir=OUT/'cache'/kind;target_dir.mkdir(parents=True,exist_ok=True)
            stats=dict(eligible=0,existing=0,published=0,lost_race=0,invalid_parent=0,bytes=0)
            for source in sorted((old/'cache'/kind).glob('*.npz')):
                if source.name not in allowed:continue
                stats['eligible']+=1;target=target_dir/source.name
                if target.exists():stats['existing']+=1;continue
                payload=source.read_bytes()
                try:
                    with np.load(io.BytesIO(payload),allow_pickle=False) as stored:
                        digest=str(stored['digest']);rows=stored['rows'];values=stored[field]
                        assert len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
                        assert rows.ndim==1 and len(rows)==len(values)
                except (ValueError,KeyError,AssertionError,EOFError,OSError):
                    stats['invalid_parent']+=1;continue
                digest_bytes=hashlib.sha256(payload).hexdigest()
                with tempfile.NamedTemporaryFile(dir=target_dir,prefix='.reuse-',suffix='.tmp',delete=False) as stream:
                    temporary=Path(stream.name);stream.write(payload)
                try:
                    assert sha(temporary)==digest_bytes
                    try:os.link(temporary,target)
                    except FileExistsError:
                        stats['lost_race']+=1;continue
                    assert (target.stat().st_dev,target.stat().st_ino)!=(source.stat().st_dev,source.stat().st_ino)
                    stats['published']+=1;stats['bytes']+=len(payload)
                    evidence.write(json.dumps(dict(kind=kind,key=source.name,source_sha256=digest_bytes,
                                                   encoder_source_digest=digest))+'\n')
                finally:temporary.unlink(missing_ok=True)
            counts[kind]=stats;print(kind,stats,flush=True)
    verify(plan)
    proof=dict(complete=True,counts=counts,inventory_sha256=sha(inventory),inventory_path=str(inventory),
               study_plan_sha256=sha(OUT/'study_plan.json'),
               frame_store_sha256=sha(ROOT/'python/pvz_frame_store.py'),
               private_byte_copies_verified=True,parent_files_read_only=True,
               validation='Unmodified pinned reader checks source+encoder digest on each cache hit; mismatches regenerate.',
               note='Only frozen archive keys. No command/source/recipe changes, restarts, or checkpoint writes.')
    write(OUT/'cache_reuse.json',proof);write(ROOT/'MEMORY/docs/task27-cache-reuse.json',proof)


if __name__=='__main__':main()
