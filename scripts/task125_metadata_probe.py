"""Read small production metadata/NPY headers; never rebuild or replay a corpus."""
import collections
import datetime
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline')
LABEL = 'c18demoroof48doomm200s1100_fast_search'


def read(path):
    return json.loads(path.read_text())


def header(z, name):
    with z.zip.open(name + '.npy') as f:
        version = np.lib.format.read_magic(f)
        return (np.lib.format.read_array_header_1_0(f) if version == (1, 0)
                else np.lib.format.read_array_header_2_0(f))


def main():
    rounds = {r: read(BASE / f'round_{r:06d}/recipe.json') for r in (11, 20, 21, 22)}
    receipts = {}
    cut = read(BASE / 'round_000021/packed.json')['pack_started']
    segments = []
    for path in sorted((BASE / 'frames/packs/segments').glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            label = json.loads(str(z['meta']))['labels'][0]
            if label not in rounds[22]['labels']:
                continue
            n = int(z['rows'])
            descriptors = {}
            for prefix in ('board', 'scalar'):
                ids = header(z, prefix + '_plane_of')[0][0]
                plane_values = header(z, prefix + '_plane_values')
                offsets = header(z, prefix + '_plane_offsets')[0][0]
                dtype = np.dtype(str(z[prefix + '_dtype']))
                descriptors[prefix] = dict(ids=ids, planes=offsets - 1,
                                          values_dtype=str(plane_values[2]), dtype=str(dtype),
                                          valid_shape=ids == n or (ids == 0 and offsets == 1),
                                          valid_dtype=plane_values[2] == dtype)
            segments.append(dict(label=label, segment=path.name, rows=n,
                                 mtime=path.stat().st_mtime, descriptors=descriptors))
    added = set(rounds[21]['labels']) - set(rounds[20]['labels'])
    added_by_round = collections.defaultdict(lambda: dict(labels=0, rows=0))
    for s in segments:
        if s['label'] in added:
            key = re.match(r'c\d+', s['label']).group(0)
            added_by_round[key]['labels'] += 1
            added_by_round[key]['rows'] += s['rows']
    for r, recipe in rounds.items():
        directory = BASE / f'round_{r:06d}'
        chosen = [s for s in segments if s['label'] in recipe['labels']]
        paths = ([directory / 'student/source_rows.json'] if r == 11 else
                 sorted((directory / 'seeds').glob('*/student/source_rows.json')))
        receipts[r] = dict(labels=len(recipe['labels']), rows=sum(s['rows'] for s in chosen),
                           segments=len(chosen),
                           bad_headers=[s for s in chosen if not all(
                               d['valid_shape'] and d['valid_dtype'] for d in s['descriptors'].values())],
                           bad_label_present=LABEL in recipe['labels'],
                           bad_label_source_counts={str(p.relative_to(BASE)): read(p)['sources'].get(LABEL)
                                                    for p in paths},
                           pack_log=(directory / 'pack.log').read_text().splitlines(),
                           recipe_sha256=hashlib.sha256((directory / 'recipe.json').read_bytes()).hexdigest())
        if (directory / 'trained.json').exists():
            trained = read(directory / 'trained.json')
            receipts[r].update(model=trained.get('model'), model_sha256=trained.get('model_sha256'),
                               selection_best=trained.get('selection', {}).get('best'),
                               candidates=[{k: c.get(k) for k in ('seed', 'model_sha256', 'source_recipe')}
                                           for c in trained.get('candidates', [])])
    changed = [s for s in segments if s['label'] in rounds[20]['labels'] and s['mtime'] >= cut]
    evidence = dict(observed=datetime.datetime.now().astimezone().isoformat(),
                    method='NPY headers and small per-label metadata only; no CSR payload reads',
                    rounds=receipts, added_labels=len(added),
                    added_by_round=dict(added_by_round), old_segments_modified_since_r21_pack=changed,
                    old_common_unmodified=len(rounds[20]['labels'])-len(changed))
    target = ROOT / 'artifacts/task125/metadata_probe.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
