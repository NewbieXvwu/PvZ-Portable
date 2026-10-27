"""Small read-only production pack comparison for task125; output stays in this worktree."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import pack_training_rows as packer
import train_rollout_policy as trainer


def legacy_compact():
    old_source = subprocess.check_output(
        ['git', 'show', '88cb4310^:scripts/train_rollout_policy.py'], cwd=ROOT, text=True)
    node = next(n for n in ast.parse(old_source).body
                if isinstance(n, ast.ClassDef) and n.name == 'CompactBoards')
    namespace = dict(vars(trainer))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<pre-deploy14f CompactBoards>', 'exec'), namespace)
    return namespace['CompactBoards']


def common_sample():
    base = Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline')
    data = base / 'frames'
    out = ROOT / 'artifacts/task125'
    label = 'c10d7d7qm100_fast_search'
    for path in (data / 'packs/segments').glob('*.npz'):
        with np.load(path, allow_pickle=False) as z:
            options = json.loads(str(z['meta']))
            if options['labels'] != [label]:
                continue
            source = z['source'].tolist()[0]
            break
    else:
        raise RuntimeError('common day segment missing')
    kwargs = dict(labels={label}, wait_keep=options['wait_keep'],
                  label_weight=options['label_weight'], cache_dir=out / 'cache',
                  read_cache_dir=base / 'cache', with_context=options['with_context'],
                  obs=options['obs'], cells=options['cells'],
                  aux_horizon=options['aux_horizon'], packet_head=options['packet_head'],
                  archive_paths=[data / source])
    original = packer.CompactBoards
    results = []
    for cls in (legacy_compact(), original):
        packer.CompactBoards = cls
        results.append(packer.build_from_archives(data, **kwargs))
    packer.CompactBoards = original
    old, new = (r[0] for r in results)
    idx = np.arange(old[0].shape[0])
    equal = {name: np.array_equal(old[i].batch(idx, 'cpu').numpy(), new[i].batch(idx, 'cpu').numpy())
             for i, name in enumerate(('board', 'scalar'))}
    equal.update({name: np.array_equal(np.asarray(old[i]), np.asarray(new[i]), equal_nan=True)
                  for i, name in enumerate(('board', 'scalar', 'plant', 'packet', 'cell', 'weight', 'seed'))
                  if i >= 2})
    equal['packet_columns'] = all(np.array_equal(a, b) for a, b in zip(old[-1].columns, new[-1].columns))
    equal['source_rows'] = np.array_equal(results[0][3], results[1][3])
    receipt = dict(label=label, source=source, rows=len(idx), equal=equal,
                   old_board_plane_dtype=str(old[0].planes[0][1].dtype),
                   new_board_plane_dtype=str(new[0].planes[0][1].dtype),
                   first_difference=None if all(equal.values()) else 'see differing columns')
    (out / 'common_sample.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


def main():
    base = Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline')
    data = base / 'frames'
    out = ROOT / 'artifacts/task125'
    out.mkdir(parents=True, exist_ok=True)
    production = data / 'packs/segments/ce3ce4006691a0e1.npz'
    with np.load(production, allow_pickle=False) as z:
        options = json.loads(str(z['meta']))
        sources = z['source'].tolist()
    # Execute exactly the old CompactBoards class, keeping the featurizer and sampling fixed.
    old_class = legacy_compact()
    new_class = trainer.CompactBoards
    kwargs = dict(labels=set(options['labels']), wait_keep=options['wait_keep'],
                  label_weight=options['label_weight'], cache_dir=out / 'cache',
                  read_cache_dir=base / 'cache', with_context=options['with_context'],
                  obs=options['obs'], cells=options['cells'],
                  aux_horizon=options['aux_horizon'], packet_head=options['packet_head'],
                  archive_paths=[data / s for s in sources])
    results = []
    for cls in [old_class, new_class]:
        packer.CompactBoards = cls
        results.append(packer.build_from_archives(data, **kwargs))
    old, new = [r[0] for r in results]
    # Compare the real legacy trainer fallback, not a guessed shifted-plane decoder.
    old_ids = old[0].plane_of.copy()
    old[0].plane_of = None
    idx = np.arange(new[0].shape[0])
    before, after = old[0].batch(idx, 'cpu').numpy(), new[0].batch(idx, 'cpu').numpy()
    diffs = np.argwhere(before != after)
    changed = np.flatnonzero(np.any(before != after, axis=(1, 2, 3)))
    actual = trainer.read_pack(production)
    columns = ['board', 'scalar', 'plant', 'packet', 'cell', 'weight', 'seed', 'values', 'candidates', 'packets']
    same = {}
    for i in [1, 2, 3, 4, 5, 6]:
        a, b = old[i], new[i]
        if i == 1:
            a, b = a.batch(idx, 'cpu').numpy(), b.batch(idx, 'cpu').numpy()
        same[columns[i]] = np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    same['packet_columns'] = all(np.array_equal(a, b) for a, b in zip(old[-1].columns, new[-1].columns))
    same['source_rows'] = np.array_equal(results[0][3], results[1][3])
    same['production_repaired_board'] = np.array_equal(after, actual[0].batch(idx, 'cpu').numpy())
    same['csr_indices'] = np.array_equal(old[0].indices, new[0].indices)
    same['csr_values'] = np.array_equal(old[0].values, new[0].values)
    same['plane_values'] = all(np.array_equal(a[1], b[1]) for a, b in zip(old[0].planes, new[0].planes))
    archive, rows = results[1][2:]
    receipt = dict(label=options['labels'][0], archives=len(sources), rows=len(idx),
                   old_plane_ids=len(old_ids), new_plane_ids=len(new[0].plane_of),
                   changed_rows=changed.tolist(), changed_cells=len(diffs),
                   changed_channels=sorted(set(int(x) for x in diffs[:, 1])), equal=same,
                   changed_sources=[dict(source=sources[int(archive[i])], source_row=int(rows[i]), packed_row=int(i))
                                    for i in changed],
                   first_difference=None if not len(diffs) else dict(
                       index=diffs[0].tolist(), old=float(before[tuple(diffs[0])]),
                       new=float(after[tuple(diffs[0])])),
                   production_path=str(production),
                   production_sha256=hashlib.sha256(production.read_bytes()).hexdigest(),
                   old_class_revision='88cb4310^', new_class_revision='ad651176',
                   engine_processes_started=0, training_runs_started=0,
                   production_files_written=0)
    (out / 'pack_probe.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'changed_sources'}, indent=2))


if __name__ == '__main__':
    common_sample() if '--common-sample' in sys.argv else main()
