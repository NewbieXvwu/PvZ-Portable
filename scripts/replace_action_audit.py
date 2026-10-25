"""Repeatable integrity audit for the task69 paired replacement panel.

Reads only the frozen manifest and the raw games.jsonl batches; never launches games.
Reports per-arm completeness plus the record invariants the preregistration relies on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TYPES = {'plant', 'replace', 'wait'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_arm(manifest, raw, task, arm, problems):
    path = raw / task['task_id'] / arm / 'games.jsonl'
    label = f"{task['task_id']}/{arm}"
    if not path.exists():
        return dict(games=0, complete=False, pending='no batch started')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    seeds = [row['seed'] for row in rows]
    if len(set(seeds)) != len(seeds):
        problems.append(f'{label}: duplicate seeds')
    unexpected = sorted(set(seeds) - set(task['seeds']))
    if unexpected:
        problems.append(f'{label}: seeds outside frozen panel {unexpected[:5]}')
    for row in rows:
        tag = f"{label} seed{row['seed']}"
        if row.get('arm') != 'fast_search':
            problems.append(f'{tag}: arm={row.get("arm")!r}')
        study = row.get('study')
        if not isinstance(study, dict):
            problems.append(f'{tag}: missing study block')
            continue
        attempts = study.get('replace_attempts', 0)
        rejected = study.get('replace_rejected', 0)
        events = study.get('replace_events', [])
        if attempts < 0 or rejected < 0 or rejected > attempts:
            problems.append(f'{tag}: rejected {rejected} exceeds attempts {attempts}')
        if len(events) != attempts - rejected:
            problems.append(f'{tag}: {len(events)} events vs {attempts - rejected} executed')
        if arm == 'before' and (attempts or events):
            problems.append(f'{tag}: baseline arm performed replacements')
        for tick in [event['tick'] for event in events]:
            if tick > row.get('final_tick', 0):
                problems.append(f'{tag}: replace event after final tick')
        for kind, counts in study.get('verification_by_type', {}).items():
            if kind not in TYPES:
                problems.append(f'{tag}: unknown verification type {kind}')
            if counts.get('revoked', 0) > counts.get('attempts', 0):
                problems.append(f'{tag}: revoked exceeds attempts for {kind}')
        for key in ('sun_produced', 'sun_spent', 'sun_remaining'):
            if study.get(key, 0) < 0:
                problems.append(f'{tag}: negative {key}')
    missing = sorted(set(task['seeds']) - set(seeds))
    return dict(games=len(rows), complete=not missing, missing_seeds=len(missing))


def audit_provenance(manifest, manifest_path, raw, problems):
    checks = 0
    manifest_sha = sha(manifest_path)
    for path in sorted(raw.glob('*/*/command-offset-*.json')):
        record = json.loads(path.read_text())
        checks += 1
        arm = record['arm']
        if record.get('manifest_sha256') != manifest_sha:
            problems.append(f'{path}: manifest hash differs from the frozen manifest')
        if record['build']['engine_sha256'] != manifest[arm]['engine_sha256']:
            problems.append(f'{path}: engine hash differs from the frozen build')
        if record['build']['commit'] != manifest[arm]['commit']:
            problems.append(f'{path}: commit differs from the frozen build')
    for arm in ('before', 'after'):
        if sha(manifest[arm]['engine']) != manifest[arm]['engine_sha256']:
            problems.append(f'{arm}: engine binary no longer matches the frozen hash')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('raw', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    problems = []
    panels = {task['task_id']: {arm: audit_arm(manifest, args.raw, task, arm, problems)
                                for arm in ('before', 'after')}
              for task in manifest['tasks']}
    checks = audit_provenance(manifest, args.manifest, args.raw, problems)
    complete = all(panel[arm]['complete'] for panel in panels.values() for arm in ('before', 'after'))
    partial = sorted(f"{task}/{arm}" for task, panel in panels.items()
                     for arm in ('before', 'after')
                     if not panel[arm]['complete'] and panel[arm]['games'])
    result = dict(panels=panels, batches_checked=checks, complete=complete,
                  partial_batches=partial, problems=problems, passed=not problems)
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')
    return 0 if not problems else 1


if __name__ == '__main__':
    raise SystemExit(main())
