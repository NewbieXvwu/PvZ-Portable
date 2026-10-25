"""Audit task78's reserved evaluation intervals against local panels and evidence.

No source writes. Covers all available worktree stage4 panels, tracked conclusion
JSONs, archive directory identities, and the baseline checkpoint's trained_on.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from task78_data import MAIN, ROOT, sha

RESERVED = set(range(78000000, 78002000)) | set(range(78100000, 78102000))


def collisions(value, location='$'):
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            here = location+'.'+key
            seed_field = 'seed' in key and not any(k in key for k in ('sha', 'path', 'source'))
            if seed_field:
                numbers = child if isinstance(child, list) else [child]
                result.extend(dict(location=here, seed=x) for x in numbers
                              if type(x) is int and x in RESERVED)
                if ('range' in key and isinstance(child, list) and len(child) == 2
                        and all(type(x) is int for x in child)):
                    result.extend(dict(location=here+' (inclusive range)', seed=x)
                                  for x in sorted(RESERVED) if child[0] <= x <= child[1])
            if key.endswith('seed_start') and type(child) is int:
                count_key = key[:-len('start')]+'count'
                count = value.get(count_key)
                if type(count) is int:
                    result.extend(dict(location=here+'..'+count_key, seed=x)
                                  for x in sorted(RESERVED) if child <= x < child+count)
            result.extend(collisions(child, here))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            if isinstance(child, (list, dict)):
                result.extend(collisions(child, f'{location}[{i}]'))
    return result


def main():
    bases = [MAIN] + sorted((MAIN/'.claude/worktrees').iterdir())
    files = set()
    for base in bases:
        if base == ROOT or not base.is_dir():
            continue
        files.update((base/'artifacts/stage4').glob('*.json'))
        files.update((base/'configs').glob('*.json'))
    # The checked-out conclusion layer contains the merged experiments' seed plans.
    files.update((ROOT/'MEMORY/docs').glob('*.json'))
    # Own task78 files declare our reservations rather than historical usage.
    files = {p for p in files if not p.name.startswith('task78')}
    records, errors, matches = [], [], []
    for path in sorted(files):
        try:
            before = sha(path)
            value = json.loads(path.read_text())
            found = collisions(value)
            if sha(path) != before:
                raise ValueError('changed during read')
            records.append(dict(path=str(path), sha256=before))
            matches.extend(dict(path=str(path), **row) for row in found)
        except (OSError, ValueError) as exc:
            errors.append(dict(path=str(path), reason=str(exc)))
    manifest = json.loads((ROOT/'artifacts/task78/sources.json').read_text())
    matches.extend(dict(path='baseline.trained_on', seed=x)
                   for x in manifest['dependencies']['trained_on'] if x in RESERVED)
    archive_seeds = set()
    for path in (MAIN/'artifacts/student/frames_v2').iterdir():
        if path.name.startswith('seed_'):
            for component in path.name[5:].split('_f'):
                archive_seeds.add(int(component))
    matches.extend(dict(path='frames_v2 directory identities', seed=x)
                   for x in sorted(archive_seeds & RESERVED))
    output = ROOT/'artifacts/task78/seed_audit.json'
    output.write_text(json.dumps(dict(files=records, errors=errors, collisions=matches), indent=1)+'\n')
    summary = dict(files=len(records), archive_seed_identities=len(archive_seeds),
                   file_groups=dict(Counter('panels' if 'stage4' in r['path'] else 'conclusions_or_configs'
                                            for r in records)),
                   error_count=len(errors), collision_count=len(matches),
                   passed=not errors and not matches,
                   scope='local stage4 panels, configs, merged JSON conclusions, archive seed/future identities, v5 trained_on',
                   unseen_remote_experiments_covered=False,
                   evidence=str(output), evidence_sha256=sha(output))
    (ROOT/'artifacts/task78/seed_audit_summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    print(json.dumps(summary, indent=1))
    if errors or matches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
