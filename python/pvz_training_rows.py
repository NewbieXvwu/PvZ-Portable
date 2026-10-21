"""Optional experimental row multipliers, checked against exact cached row identities."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


class SourceRowCounts:
    """Count actual retained cached rows without changing sampling, weights or RNG state."""
    def __init__(self):
        self.sources = {}

    def observe(self, path, rows):
        path = Path(path)
        record = self.sources.setdefault(path.parent.name, dict(archives=0, seeds={}))
        record['archives'] += 1
        seed = int(path.parent.parent.name.split('_')[-1])
        record['seeds'][seed] = record['seeds'].get(seed, 0)+len(rows)

    def finish(self, total, validation_seeds, destination):
        result = {}
        for label, record in self.sources.items():
            rows = sum(record['seeds'].values())
            validation = sum(n for seed,n in record['seeds'].items() if seed in validation_seeds)
            result[label] = dict(archives=record['archives'], rows=rows, training_rows=rows-validation,
                                 validation_rows=validation)
        assert sum(r['rows'] for r in result.values()) == total
        destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(dict(rows=total, sources=result), indent=2)+'\n')


def update_identity(digest, relative_path, rows):
    digest.update(relative_path.encode()+b'\0')
    digest.update(len(rows).to_bytes(8,'little'))
    digest.update(np.asarray(rows,dtype='<i8').tobytes())


class RowWeights:
    def __init__(self,path,data):
        self.path=Path(path);self.data=Path(data)
        self.spec=json.loads(self.path.read_text())
        self.values=np.load(self.path.parent/self.spec['multipliers'],mmap_mode='r',allow_pickle=False)
        if self.values.dtype!=np.float32 or self.values.shape!=(self.spec['rows'],):
            raise ValueError('row multiplier shape or dtype mismatch')
        if not np.isfinite(self.values).all() or (self.values<0).any() or not (self.values>0).any():
            raise ValueError('row multipliers must be finite, nonnegative and not all zero')
        if hashlib.sha256(self.values.tobytes()).hexdigest()!=self.spec['multipliers_sha256']:
            raise ValueError('row multiplier checksum mismatch')
        self.digest=hashlib.sha256();self.observed=0

    def observe(self,path,rows):
        update_identity(self.digest,str(Path(path).relative_to(self.data)),rows)
        self.observed+=len(rows)

    def finish(self,n,is_val):
        if n!=self.spec['rows'] or self.observed!=n or self.digest.hexdigest()!=self.spec['row_identity_sha256']:
            raise ValueError('row weights do not match loaded dataset identities')
        split=hashlib.sha256(np.packbits(np.asarray(is_val)).tobytes()).hexdigest()
        if split!=self.spec['split_sha256']:
            raise ValueError('row weights do not match the seed validation split')
        if not np.equal(self.values[np.asarray(is_val)],1.).all():
            raise ValueError('experimental multipliers must leave validation rows unchanged')

    def apply(self,weights):
        import torch
        for start in range(0,len(weights),1_000_000):
            end=min(start+1_000_000,len(weights))
            weights[start:end]*=torch.from_numpy(np.array(self.values[start:end],copy=True))

    def metadata(self):
        return {'path':str(self.path),'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),
                'name':self.spec['name'],'row_identity_sha256':self.spec['row_identity_sha256'],
                'multipliers_sha256':self.spec['multipliers_sha256']}
