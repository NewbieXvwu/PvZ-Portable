"""Per-label level balancing and label-weight adjustment for row-weighted training."""
from __future__ import annotations
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


def source_kind(meta):
    """Classify the recorded prior and explicitly converted demonstrations once."""
    source = meta.get('label_source', 'teacher')
    category = source
    task = meta.get('task', {})
    prior = str(task.get('checkpoint', '')) + ' ' + str(task.get('policy', ''))
    if meta.get('llm_demo_sha256') and source.startswith('llm_'):
        category = 'llm_demo'
    elif source == 'teacher' and 'gptbc' in prior:
        category = 'teacher_gptbc'
    elif source == 'teacher' and 'round_000052' in prior:
        category = 'teacher_r52'
    return category


def source_factor(meta, rules):
    return rules.get(source_kind(meta), rules.get(meta.get('label_source', 'teacher'), rules.get('default', 1.)))


class DemoRowWeights:
    """Optional final multipliers from the pack's existing archive provenance.

    No new pack columns/signatures. Human rows and explicitly imported LLM demo rows
    share demo_row_weight; teacher rows with demo_archive are continuations. Task names never
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
        self.rows = dict(human=0, demo_start=0, llm_demo=0)

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
        elif source_kind(meta) == 'llm_demo':
            category, factor = 'llm_demo', self.demo_row_weight
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
            training = ~is_val[start:stop] & (weights[start:stop] > 0)
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


class LabelWeightAdjustment:
    """Rescale a trainer's private weights by archive spans, without rebuilding feature packs."""
    def __init__(self, desired, packed):
        self.factors = {}
        for label in desired.keys() | packed.keys():
            before, after = packed.get(label, 1.0), desired.get(label, 1.0)
            if before == 0 and after != 0:
                raise ValueError('cannot restore nonzero weights from a zero-weight pack')
            if before != after:
                self.factors[label] = after / before
        self.rows = 0
        self.spans = []

    def observe(self, path, rows):
        count = len(rows)
        factor = self.factors.get(path.parent.name, 1.0)
        if factor != 1 and count:
            self.spans.append((self.rows, self.rows + count, factor))
        self.rows += count

    def apply(self, weight):
        if self.rows != len(weight):
            raise ValueError('label weight spans do not cover the loaded rows')
        for start, stop, factor in self.spans:
            weight[start:stop] *= factor


class SharedPackAdjustment:
    """Archive selection and deck caps on private training weights; validation stays fixed."""
    def __init__(self, data, history_l7_keep=1.0, deck_weight_cap=1.0):
        self.data = Path(data)
        self.keep = history_l7_keep
        self.cap = deck_weight_cap
        self.spans = []
        self.rows = 0
        self.report = {}

    def observe(self, path, rows):
        from pvz_training_pack import keep_archive
        path = Path(path)
        count = len(rows)
        historical = path.parent.name.startswith(('z', 'd7'))
        if count and (self.cap < 1 or self.keep < 1 and historical):
            meta = json.loads((path.parent / 'meta.json').read_text())
            task = meta.get('task', {})
            level = int(task.get('level', meta.get('context', {}).get('level')))
            selected = (self.keep == 1 or not historical or level != 7
                        or keep_archive(self.data, path, {'.*': self.keep}))
            self.spans.append((self.rows, self.rows + count, level, task.get('deck'),
                               historical and level == 7, selected))
        self.rows += count

    def apply_keep(self, weights, is_val):
        if self.keep == 1:
            return
        eligible = kept = dropped_rows = 0
        before = after = 0.0
        for start, stop, _level, _deck, historical, selected in self.spans:
            if not historical:
                continue
            training = ~is_val[start:stop]
            if not training.any():
                continue
            eligible += 1
            kept += selected
            part = weights[start:stop]
            mass = float(part[training].sum())
            before += mass
            if selected:
                after += mass
            else:
                dropped_rows += int(training.sum())
                part[training] = 0
        self.report['history_l7'] = dict(keep=self.keep, eligible_archives=eligible,
                                       kept_archives=kept, dropped_training_rows=dropped_rows,
                                       before_weight=before, after_weight=after)

    def apply_deck_cap(self, weights, is_val, packets):
        if self.cap == 1:
            return
        groups, spans = {}, []
        for start, stop, level, deck, _historical, _selected in self.spans:
            if deck is None:
                # Fixed decks are recorded in the existing pack. Never infer an old deck
                # from today's task defaults. Explicit [] keeps original conveyors separate.
                deck = packets.columns[0][start]
                deck = [int(kind) for kind in deck if kind >= 0]
            signature = (level, tuple(sorted(deck)))
            training = ~is_val[start:stop]
            mass = float(weights[start:stop][training].sum())
            if mass > 0:
                groups[signature] = groups.get(signature, 0.0) + mass
                spans.append((start, stop, signature))
        if len(groups) * self.cap < 1 - 1e-12:
            raise ValueError('too few positive training deck signatures for deck_weight_cap')
        masses = sorted(groups.values(), reverse=True)
        total = sum(masses)
        threshold = max(masses)
        if len(groups) * self.cap <= 1 + 1e-12:
            threshold = min(masses)
        else:
            remaining = total
            for count, mass in enumerate(masses):
                bound = self.cap * remaining / (1 - self.cap * count)
                if mass <= bound:
                    threshold = bound
                    break
                remaining -= mass
        factors = {key: min(1.0, threshold / mass) for key, mass in groups.items()}
        for start, stop, signature in spans:
            factor = factors[signature]
            if factor < 1:
                part = weights[start:stop]
                training = ~is_val[start:stop]
                part[training] *= factor
        allocated = sum(groups[key] * factor for key, factor in factors.items())
        self.report['decks'] = dict(cap=self.cap, before_weight=total, after_weight=allocated,
            signatures=[dict(level=level, deck=list(deck), before_weight=mass,
                             multiplier=factors[(level, deck)],
                             effective_share=mass * factors[(level, deck)] / allocated)
                        for (level, deck), mass in sorted(groups.items())])


class ReplayAdjustment:
    """New immutable archives since the promoted corpus; no frame reads or new packs."""
    def __init__(self, data, old_archives, old_labels, new_weight):
        self.data, self.old_archives = Path(data), set(old_archives)
        self.old_labels, self.new_weight = set(old_labels), new_weight
        self.rows, self.spans = 0, []

    def observe(self, path, rows):
        path = Path(path)
        old = (str(path.parent.relative_to(self.data)) in self.old_archives
               or path.parent.name in self.old_labels)
        if not old and len(rows):
            self.spans.append((self.rows, self.rows + len(rows)))
        self.rows += len(rows)

    def apply(self, weight, is_val):
        for start, stop in self.spans:
            part = weight[start:stop]
            part[~is_val[start:stop]] *= self.new_weight
