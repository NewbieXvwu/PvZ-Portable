"""Optional experimental row multipliers, checked against exact cached row identities."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

from pvz_frame_store import archive_seed


def balance_levels(levels, cap, power=1.0, max_multiplier=None):
    """Per-level training-row multipliers from retained rows and original weights.

    power=1 weights each level by the exact inverse of its retained training rows, which
    makes every level carry nearly the same effective share. power<1 tempers that: a level
    with a thousandth of the rows is only raised by 1/sqrt(1000) at power=.5, so the few
    hundred-row levels stop dominating the gradient with a handful of samples. power=0
    drops the inverse-rows term, so only the cap below can still move a level.
    max_multiplier caps the per-level multiplier; whatever the cap removes is simply not
    allocated, so the total training weight may end up below its original value.
    """
    total_rows = sum(record['rows'] for record in levels.values())
    total_weight = sum(record['original_weight'] for record in levels.values())
    for record in levels.values():
        record['inverse_rows'] = total_rows / (len(levels) * record['rows'])
    masses = {level: record['original_weight'] * (record['inverse_rows'] if power == 1.0
                                                  else record['inverse_rows'] ** power)
              for level, record in levels.items()}
    shares = {}
    while masses:
        remaining = 1 - sum(shares.values())
        total = sum(masses.values())
        capped = [level for level, mass in masses.items() if remaining * mass / total > cap]
        if not capped:
            shares.update({level: remaining * mass / total for level, mass in masses.items()})
            break
        for level in capped:
            shares[level] = cap
            del masses[level]
    for level, record in levels.items():
        record.update(effective_share=shares[level],
                      multiplier=total_weight * shares[level] / record['original_weight'])
        if max_multiplier is not None and record['multiplier'] > max_multiplier:
            record['multiplier_capped_from'] = record['multiplier']
            record['multiplier'] = max_multiplier
    allocated = (sum(record['multiplier'] * record['original_weight'] for record in levels.values())
                 if max_multiplier is not None else total_weight)
    return dict(total_weight=total_weight, allocated_weight=allocated,
                allocated_weight_ratio=allocated / total_weight)


class DemoRowWeights:
    """Optional final multipliers from the pack's existing archive provenance.

    No new pack columns/signatures. Human rows are label_source=human; teacher rows
    with demo_archive are continuations from a demonstration. Task names alone never
    classify a row. Default factors return before reading metadata or touching weights.
    Apply AFTER packet/spread/intervention/level balancing: these boosts are not capped
    or renormalized, so a final level share can exceed the earlier level-balance cap.
    """
    def __init__(self, demo_row_weight=1.0, demo_start_row_weight=1.0):
        self.demo_row_weight = demo_row_weight
        self.demo_start_row_weight = demo_start_row_weight
        self.enabled = demo_row_weight != 1.0 or demo_start_row_weight != 1.0
        self.observed = 0
        self.spans = []
        self.rows = dict(human=0, demo_start=0)

    def observe(self, path, rows, meta=None):
        if not self.enabled:
            return
        start, count = self.observed, len(rows)
        self.observed += count
        if not count:
            return
        if meta is None:
            meta_path = Path(path).parent / 'meta.json'
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        source = meta.get('label_source', 'teacher')
        if source == 'human':
            category, factor = 'human', self.demo_row_weight
        elif source == 'teacher' and meta.get('demo_archive'):
            category, factor = 'demo_start', self.demo_start_row_weight
        else:
            return
        self.rows[category] += count
        if factor != 1.0:
            self.spans.append((start, start + count, factor))

    def apply(self, weights):
        if not self.enabled:
            return
        if len(weights) != self.observed:
            raise ValueError('demo row provenance does not cover the loaded rows')
        for start, stop, factor in self.spans:
            weights[start:stop] *= factor

    def metadata(self):
        return dict(demo_row_weight=self.demo_row_weight,
                    demo_start_row_weight=self.demo_start_row_weight,
                    rows=self.rows, stage='after_level_balance', participates_in_cap=False)


class SourceRowCounts:
    """Count actual retained cached rows without changing sampling, weights or RNG state."""
    def __init__(self, balance=False):
        self.sources = {}
        self.balance = balance
        self.spans = []
        self.observed = 0

    def observe(self, path, rows, meta=None):
        path = Path(path)
        record = self.sources.setdefault(path.parent.name, dict(archives=0, seeds={}))
        record['archives'] += 1
        seed = archive_seed(path.parent.parent.name)
        record['seeds'][seed] = record['seeds'].get(seed, 0)+len(rows)
        if self.balance:
            if meta is None:
                meta = json.loads((path.parent / 'meta.json').read_text())
            level = int(meta['context']['level'])
            self.spans.append((level, self.observed, self.observed + len(rows)))
        self.observed += len(rows)

    def rebalance(self, weights, is_val, cap, power=1.0, max_multiplier=None):
        levels = {}
        for level, start, stop in self.spans:
            training = ~is_val[start:stop]
            record = levels.setdefault(level, dict(rows=0, original_weight=0.0))
            record['rows'] += int(training.sum())
            record['original_weight'] += float(weights[start:stop][training].sum())
        active = {level: record for level, record in levels.items()
                  if record['rows'] and record['original_weight'] > 0}
        if len(active) * cap < 1 - 1e-9:
            raise ValueError('too few training levels for source_weight.level_balance cap')
        totals = balance_levels(active, cap, power, max_multiplier)
        for level, start, stop in self.spans:
            if level in active:
                training = ~is_val[start:stop]
                part = weights[start:stop]
                part[training] *= active[level]['multiplier']
        return dict(cap=cap, power=power, max_multiplier=max_multiplier, levels=active, **totals)

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
