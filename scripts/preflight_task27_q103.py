"""Read-only rehearsal of task27's q103 freeze gate.

For every q103 label already reported complete in queue103.log, assert that frames_v2 holds
exactly the archives freeze() will demand: count == reported games and the exact manifest
seed slice.  Nothing is written outside artifacts/task27 and no game process is started.
"""
import argparse
import re
from pathlib import Path

from task27_common import MAIN, OUT, read, write, sha

LOG = Path('/home/newbiexvwu/pvz-logs/queue103.log')
PATTERN = re.compile(r'\s(\S+) \(dw4i_s1, ([^,]+), offset (\d+), (\d+) workers\): (\d+)/(\d+)')
MANIFESTS = (MAIN/'artifacts/stage4/night_panel.json', MAIN/'artifacts/stage4/day_panel.json',
             MAIN/'artifacts/stage4/pool_panel.json',
             MAIN/'.claude/worktrees/identity/artifacts/stage4/day_panel_identity.json')
FRAMES = MAIN/'artifacts/student/frames_v2'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(OUT/'q103_preflight.json'))
    parser.add_argument('--labels', default='')
    args = parser.parse_args()
    log = LOG.read_text()
    done = {m[1]:dict(task=m[2], offset=int(m[3]), games=int(m[6]), reported=int(m[5]),
                      workers=int(m[4])) for m in PATTERN.finditer(log)}
    taskmaps = {}
    for manifest in MANIFESTS:
        taskmaps.update({t['task_id']: t for t in read(manifest)['tasks']})
    wanted = [x for x in args.labels.split(',') if x] or sorted(done)
    rows = []
    for label in wanted:
        row = done[label]
        task = taskmaps[row['task']]
        seeds = task['seeds'][row['offset']:row['offset']+row['games']]
        folder = label if label.endswith('_fast_search') else label + '_fast_search'
        found = sorted(int(p.parent.parent.name.split('_', 1)[1])
                       for p in FRAMES.glob(f'*/{folder}/frames.npz'))
        missing = sorted(set(seeds)-set(found))
        extra = sorted(set(found)-set(seeds))
        rows.append(dict(label=label, task=row['task'], offset=row['offset'], games=row['games'],
                         reported=row['reported'], archives=len(found), deck=task['deck'],
                         seeds_exact=not missing and not extra and len(found) == len(set(found)),
                         missing_seeds=missing[:8], extra_seeds=extra[:8],
                         meta_complete=all((p.parent/'meta.json').is_file()
                                           for p in FRAMES.glob(f'*/{label}/frames.npz'))))
    failed = [r['label'] for r in rows
              if not r['seeds_exact'] or not r['meta_complete'] or r['archives'] != r['games']]
    data = dict(complete=not failed, scope='Read-only gate rehearsal; q103 round2 not frozen here.',
                log_sha256=sha(LOG), labels_checked=len(rows), failed=failed, rows=rows)
    write(Path(args.out), data)
    for r in rows:
        print(r['label'], r['task'], 'games', r['games'], 'archives', r['archives'],
              'exact', r['seeds_exact'], 'meta', r['meta_complete'], flush=True)
    print('failed:', failed, flush=True)
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
