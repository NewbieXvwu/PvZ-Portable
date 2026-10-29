"""Prepared boundary-update helper. --plan reads only; --apply is called after review."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SNAPSHOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SNAPSHOT / 'scripts'), str(SNAPSHOT / 'python')]
from retire_task20 import retire_links, restore_links
from stabilize_production_paths import planned_changes

CORE = ('scripts/curriculum_queue.py', 'python/pvz_frame_store.py', 'python/pvz_recipe.py',
        'python/pvz_training_pack.py', 'python/pvz_corpus_policy.py')
PANELS = ('day_orig_panel.json', 'unified_v1.json', 'unified_v2.json', 'day_identity.json')


def current_engine(cq):
    needle = str(cq / 'scripts/curriculum_queue.py')
    matches = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            continue
        if needle in args:
            matches.append(Path(args[args.index('--executable') + 1]))
    if len(matches) != 1:
        raise RuntimeError(f'expected one live controller, found {len(matches)}')
    return matches[0]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def updates(root, logs, revision, engine, baseline='ad4fe41c'):
    cq = root / '.claude/worktrees/cq'
    mirrors = [cq, root / '.claude/worktrees/obsup']
    native_package = engine.parent.parent
    if (native_package / 'recipes').is_dir() and native_package not in mirrors:
        mirrors.append(native_package)  # discovered from the live executable, no fixed package worktree
    changes = {}
    for mirror in mirrors:
        for path, text in planned_changes(mirror):
            changes[path] = text.encode()
        changes[mirror / 'python/pvz_corpus_policy.py'] = (SNAPSHOT / 'python/pvz_corpus_policy.py').read_bytes()
    for name in CORE:
        path = cq / name
        replacement = (SNAPSHOT / name).read_bytes()
        if path.is_file() and path.read_bytes() != replacement:
            original = subprocess.run(['git', '-C', str(root), 'show', baseline + ':' + name],
                                      capture_output=True)
            if original.returncode or sha(path.read_bytes()) != sha(original.stdout):
                raise RuntimeError(f'production code changed outside reviewed baseline: {path}')
        changes[path] = replacement
    for mirror in [root, *mirrors]:
        for name in PANELS:
            changes[mirror / 'configs/panels' / name] = (SNAPSHOT / 'configs/panels' / name).read_bytes()
    launch = (SNAPSHOT / 'scripts/qc_deploy17.sh').read_text()
    launch = re.sub(r'(?m)^REV=\S+$', 'REV=' + revision, launch)
    changes[logs / 'qc_deploy17.sh'] = launch.encode()
    changes[logs / 'qc_deploy17.active'] = (revision + '\n').encode()
    return {p: b for p, b in changes.items() if not p.is_file() or p.read_bytes() != b}


def save_and_write(changes, backup):
    before = backup / 'before'
    before.mkdir(parents=True, exist_ok=True)
    records = []
    for path in changes:
        key = hashlib.sha256(str(path).encode()).hexdigest()
        existed = path.is_file()
        if existed:
            shutil.copy2(path, before / key)
        records.append(dict(path=str(path), key=key, existed=existed))
    (backup / 'files.json').write_text(json.dumps(records, indent=2) + '\n')
    for path, content in changes.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.paths-building')
        temporary.write_bytes(content)
        if path.exists():
            shutil.copymode(path, temporary)
        temporary.replace(path)


def restore(backup):
    records = json.loads((backup / 'files.json').read_text()) if (backup / 'files.json').exists() else []
    for record in records:
        path = Path(record['path'])
        if record['existed']:
            shutil.copy2(backup / 'before' / record['key'], path)
        else:
            path.unlink(missing_ok=True)
    if (backup / 'retired_links.json').exists():
        restore_links(json.loads((backup / 'retired_links.json').read_text()))


def stage_engine(source, target):
    # One immutable 73 MB binary, no rebuild or gameplay; never overwrite a different binary.
    source_sha = hashlib.file_digest(source.open('rb'), 'sha256').hexdigest()
    if target.exists():
        if hashlib.file_digest(target.open('rb'), 'sha256').hexdigest() != source_sha:
            raise RuntimeError(f'stable executable already has different bytes: {target}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.building')
        shutil.copy2(source, temporary)
        temporary.replace(target)
    return source_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'stage-engine', 'apply', 'restore'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--engine', type=Path, help='engine captured before stopping the old controller')
    args = parser.parse_args()
    if args.mode == 'restore':
        restore(args.backup)
        return
    cq = args.root / '.claude/worktrees/cq'
    engine = args.engine or current_engine(cq)
    target = args.root / 'artifacts/production/deploy16/pvz-portable'
    if args.mode == 'stage-engine':
        print(json.dumps(dict(executable=str(target), sha256=stage_engine(engine, target))))
        return
    changes = updates(args.root, args.logs, args.revision, engine)
    view = args.logs / 'curriculum_state.pipeline/frames'
    links = retire_links(view)
    if args.mode == 'plan':
        print(json.dumps(dict(revision=args.revision, files=[str(p) for p in sorted(changes)],
                             retired_links=len(links), executable_source=str(engine),
                             executable_target=str(target)), indent=2))
        return
    # Receipt is durable before deleting any links; immutable pack segments are untouched.
    args.backup.mkdir(parents=True, exist_ok=True)
    (args.backup / 'retired_links.json').write_text(json.dumps(links, indent=2) + '\n')
    save_and_write(changes, args.backup)
    retire_links(view, apply=True)
    print(json.dumps(dict(updated_files=len(changes), retired_links=len(links))))


if __name__ == '__main__':
    main()
