"""Read-only description of old-row/split drift under the legacy cohort recipe."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import numpy as np


def load(folder):
    meta = json.loads((folder / 'cohort.json').read_text())
    arrays = {k: np.load(folder / (k + '.npy'), mmap_mode='r', allow_pickle=False)
              for k in ('episode', 'row', 'packet', 'is_val')}
    counts = np.bincount(arrays['episode'], minlength=len(meta['episodes']))
    bounds = np.concatenate(([0], counts.cumsum()))
    assert bounds[-1] == meta['rows']
    assert (np.diff(arrays['episode']) >= 0).all()
    return meta, arrays, bounds


def compare(old_folder, full_folder):
    old, a, ab = load(old_folder)
    full, b, bb = load(full_folder)
    old_index = {v['relative_source']: i for i, v in enumerate(old['episodes'])}
    full_index = {v['relative_source']: i for i, v in enumerate(full['episodes'])}
    assert set(old_index) <= set(full_index)
    n = dict.fromkeys(('old_rows', 'full_rows_from_old_archives', 'shared_rows',
                      'old_plant_rows', 'full_old_archive_plant_rows', 'shared_plant_rows',
                      'old_train_rows', 'full_train_rows_from_old_archives', 'shared_train_rows',
                      'old_train_rows_on_seeds_now_validation', 'full_train_rows_on_seeds_previously_validation'), 0)
    flags_a = {}
    flags_b = {}
    for name, ia in old_index.items():
        ib = full_index[name]
        ea, eb = old['episodes'][ia], full['episodes'][ib]
        assert ea['source_sha256'] == eb['source_sha256'] and ea['meta_sha256'] == eb['meta_sha256']
        x, y = slice(ab[ia], ab[ia + 1]), slice(bb[ib], bb[ib + 1])
        ar, br = a['row'][x], b['row'][y]
        av, bv = a['is_val'][x], b['is_val'][y]
        assert len(av) and len(bv) and (av == av[0]).all() and (bv == bv[0]).all()
        seed = ea['seed']
        assert seed == eb['seed']
        for flags, val in ((flags_a, bool(av[0])), (flags_b, bool(bv[0]))):
            assert seed not in flags or flags[seed] == val
            flags[seed] = val
        _, ax, bx = np.intersect1d(ar, br, assume_unique=True, return_indices=True)
        ap, bp = a['packet'][x] >= 0, b['packet'][y] >= 0
        assert np.array_equal(ap[ax], bp[bx])
        # All valid plant decisions survive wait thinning in both cohorts.
        assert ap.sum() == bp.sum() == ap[ax].sum()
        n['old_rows'] += len(ar)
        n['full_rows_from_old_archives'] += len(br)
        n['shared_rows'] += len(ax)
        n['old_plant_rows'] += int(ap.sum())
        n['full_old_archive_plant_rows'] += int(bp.sum())
        n['shared_plant_rows'] += int(ap[ax].sum())
        n['old_train_rows'] += int((~av).sum())
        n['full_train_rows_from_old_archives'] += int((~bv).sum())
        n['shared_train_rows'] += int((~av[ax] & ~bv[bx]).sum())
        if not av[0] and bv[0]:
            n['old_train_rows_on_seeds_now_validation'] += len(ar)
        if av[0] and not bv[0]:
            n['full_train_rows_on_seeds_previously_validation'] += len(br)
    assert n['old_rows'] == old['rows'] and n['old_train_rows'] == old['train_rows']
    n.update(old_wait_rows=n['old_rows'] - n['old_plant_rows'],
             full_old_archive_wait_rows=n['full_rows_from_old_archives'] - n['full_old_archive_plant_rows'],
             shared_wait_rows=n['shared_rows'] - n['shared_plant_rows'])
    changes = {'shared_environment_seeds': len(flags_a),
               'old_train_to_full_validation': sum(not flags_a[s] and flags_b[s] for s in flags_a),
               'old_validation_to_full_train': sum(flags_a[s] and not flags_b[s] for s in flags_a),
               'remain_train': sum(not flags_a[s] and not flags_b[s] for s in flags_a),
               'remain_validation': sum(flags_a[s] and flags_b[s] for s in flags_a)}
    for meta, flags in ((old, flags_a), (full, flags_b)):
        all_seeds = sorted({e['seed'] for e in meta['episodes']})
        random.Random(0).shuffle(all_seeds)
        val = set(all_seeds[:len(all_seeds)//10])
        assert all(flags[s] == (s in val) for s in flags)
    return {'audited_at': datetime.now(timezone.utc).isoformat(),
            'scope': 'exploratory training-data audit; no policy outcomes or new intervention',
            'shared_archives': len(old_index), 'rows': n, 'seed_split_transition': changes,
            'old_archives_prefix_in_full_order': [e['relative_source'] for e in old['episodes']]
                == [e['relative_source'] for e in full['episodes'][:len(old['episodes'])]],
            'old_selected_rows_unchanged': n['old_rows'] == n['shared_rows'] == n['full_rows_from_old_archives'],
            'wait_row_retention_overlap_fraction': n['shared_wait_rows'] / n['old_wait_rows'],
            'old_training_row_retention_fraction': n['shared_train_rows'] / n['old_train_rows'],
            'global_weight_normalizers': {'D30_mean_spread': old['mean_spread'], 'D35_mean_spread': full['mean_spread'],
                'D30_packet_scale': old['packet_scale'], 'D35_packet_scale': full['packet_scale']},
            'cohort_metadata_sha256': {c.name: hashlib.sha256((c / 'cohort.json').read_bytes()).hexdigest()
                                       for c in (old_folder, full_folder)},
            'interpretation': 'Check old_selected_rows_unchanged before attributing any drift to wait sampling. B-F measures the complete legacy D30-to-D35 recipe, including validation-seed repartition and global packet/spread normalization. These descriptive changes are not proven causes. M/T/I/R/S vs F keep the same D35 rows/split; no new arm is added.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohorts', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    report = compare(args.cohorts / 'D30', args.cohorts / 'D35')
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
