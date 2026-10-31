#!/usr/bin/env python3
"""Reviewed deploy26 switch; run by the controller, never by collection/watchdog."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from datetime import datetime

import qc_boundary as boundary

BASE = '97994745'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def code_paths(root, revision, kind):
    return [path for path in git(root, 'diff', '--no-renames', '--name-only', '--diff-filter=' + kind,
            BASE, revision, '--', 'python', 'scripts', 'recipes').decode().splitlines()
            if not path.startswith('python/test_') and path.endswith(('.py', '.sh', '.json'))]


def backup(path, saved):
    saved.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        saved.symlink_to(os.readlink(path))
    elif path.exists():
        shutil.copy2(path, saved)
    else:
        saved.with_name(saved.name + '.absent').touch()


def restore(path, saved):
    path.unlink(missing_ok=True)
    if saved.with_name(saved.name + '.absent').exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if saved.is_symlink():
        path.symlink_to(os.readlink(saved))
    else:
        shutil.copy2(saved, path)


def stop_watchdog(worktree, logs):
    scripts = {str(worktree / 'scripts/qc_watchdog.sh'), str(logs / 'qc_watchdog.sh')}
    for path in Path('/proc').glob('[0-9]*'):
        pid = int(path.name)
        command = boundary.command(pid)
        if len(command) > 1 and command[1] in scripts:
            record = boundary.freeze_if_idle(pid, worktree)
            if record:
                boundary.stop(record)


def start_watchdog(logs):
    with (logs / 'qc_setup.log').open('a') as output:
        subprocess.Popen(['bash', str(logs / 'qc_watchdog.sh')], stdin=subprocess.DEVNULL,
                         stdout=output, stderr=subprocess.STDOUT, start_new_session=True)


def launch(installation, args, worktree, rollback=False):
    command = [sys.executable, str(installation / 'scripts/qc_launch.py'),
        '--repo', str(args.repo), '--worktree', str(worktree), '--logs', str(args.logs),
        '--python', sys.executable]
    if rollback:
        command += ['--recipe', str(worktree / 'recipes/dw4i2_v2timer.json'),
                    '--alternate-recipe', str(worktree / 'recipes/dw4i2_v2timer_seed2cold.json')]
    subprocess.run(command, check=True)


def stop_collection(worktree, logs, record):
    running = subprocess.run(['tmux', 'has-session', '-t', '=qc'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if running:
        boundary.wait_boundary(worktree, logs / 'curriculum.log', record)
        boundary.stop(json.loads(record.read_text()))
    subprocess.run(['tmux', 'kill-session', '-t', '=qc'], check=False)


def interrupt(signum, _frame):
    raise SystemExit(128 + signum)


def execute(args):
    root, logs = args.repo, args.logs
    if (logs / 'stop_curr').exists():
        raise RuntimeError('existing stop request; deployment not started')
    lock = (logs / 'qc_deploy.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    worktree = root / '.claude/worktrees/cq'
    imports = root / '.claude/worktrees/llmimport'
    rollback = args.rollback is not None
    folder = args.rollback or logs / ('deploy26_' + datetime.now().strftime('%Y%m%dT%H%M%S') + '_' + str(os.getpid()))
    if rollback:
        revision = (folder / 'revision').read_text().strip()
    else:
        revision = git(root, 'rev-parse', args.revision).decode().strip()
        (folder / 'install').mkdir(parents=True)
        (folder / 'revision').write_text(revision + '\n')
    changed, removed = code_paths(root, revision, 'AMT'), code_paths(root, revision, 'D')
    targets = {'cq/' + path: worktree / path for path in changed + removed}
    targets.update({'links/watchdog': logs / 'qc_watchdog.sh', 'links/active': logs / 'qc.active'})
    for name in ('import_llm_run.py', 'llm_import_sha.py', 'pvz_live_import.py',
                 'deploy22_import.py', 'task105_import_batch.py'):
        targets['imports/' + name] = imports / 'scripts' / name
    installation, record = folder / 'install', folder / 'scheduler.json'
    if not rollback:
        for path in changed:
            target = installation / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(git(root, 'show', revision + ':' + path))
        for key, path in targets.items():
            backup(path, folder / 'before' / key)
        (folder / 'backed_up').touch()
    elif not (folder / 'backed_up').exists():
        raise RuntimeError('rollback backup was not completed')
    record.unlink(missing_ok=True)
    stopped = False
    try:
        stop_watchdog(worktree, logs)
        stop_collection(worktree, logs, record)
        stopped = True
        if rollback:
            for key, path in targets.items():
                restore(path, folder / 'before' / key)
        else:
            for path in changed:
                target = worktree / path
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.deploy26-building')
                shutil.copy2(installation / path, temporary)
                temporary.replace(target)
            for path in removed:
                (worktree / path).unlink(missing_ok=True)
            for key, path in targets.items():
                if key.startswith('imports/') or key == 'links/watchdog':
                    path.unlink(missing_ok=True)
            (imports / 'scripts/import_llm_run.py').symlink_to(worktree / 'scripts/import_llm_run.py')
            (logs / 'qc_watchdog.sh').symlink_to(worktree / 'scripts/qc_watchdog.sh')
            (logs / 'qc.active').write_text(revision + '\n')
        launch(installation, args, worktree, rollback=rollback)
        start_watchdog(logs)
        print(('rolled back' if rollback else 'activated') + f' {revision}; backup {folder}', flush=True)
    except BaseException:
        if stopped:
            stop_collection(worktree, logs, record)
            for key, path in targets.items():
                restore(path, folder / 'before' / key)
            launch(installation, args, worktree, rollback=True)
        elif record.exists():
            boundary.act(json.loads(record.read_text())['controller'], signal.SIGCONT)
        start_watchdog(logs)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.home() / 'PvZ-search')
    parser.add_argument('--logs', type=Path, default=Path.home() / 'pvz-logs')
    parser.add_argument('--revision', default='origin/deploy26')
    parser.add_argument('--rollback', type=Path, metavar='BACKUP_DIRECTORY')
    parser.add_argument('--execute', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.execute:
        subprocess.run(['systemd-run', '--user', '--wait', '--collect', '--property=KillMode=process',
            sys.executable, str(Path(__file__).resolve()), '--execute', *sys.argv[1:]], check=True)
        return
    signal.signal(signal.SIGTERM, interrupt)
    execute(args)


if __name__ == '__main__':
    main()
