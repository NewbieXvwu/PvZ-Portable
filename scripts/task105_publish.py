#!/usr/bin/env python3
"""Publish completed task105 scans without blocking the game pipeline.

Reads only the pipeline's small result JSON and generated text files. Never
runs a report, verifier, game, or deployment. Push failures retry in this
separate process while collection continues.
"""
import argparse
import datetime as dt
import fcntl
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/task105'
EVIDENCE = 'MEMORY/docs/task105-import-evidence.json'
FILES = ('DESIGN.md', EVIDENCE, 'MEMORY/docs/task105-gptpro-run1-import.md',
         'configs/human_demo_tasks.json', 'python/test_curriculum_queue.py')


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True,
                                   stderr=subprocess.STDOUT, timeout=60).strip()


def publish():
    if git('branch', '--show-current') != 'llmimport':
        raise ValueError('publisher requires branch llmimport')
    # Do not mix another actor's staged work into an automatic result commit.
    if git('diff', '--cached', '--name-only'):
        raise ValueError('index already contains staged changes; leaving them untouched')
    evidence = json.loads((ROOT / EVIDENCE).read_text())
    committed = json.loads(git('show', f'HEAD:{EVIDENCE}'))
    known = {s['task_id'] for s in committed['scans']}
    added = [s for s in evidence['scans'] if s['task_id'] not in known]
    status = json.loads((OUT / 'pipeline_status.json').read_text())
    stage = status.get('stage', '')
    if added and (stage.startswith('adaptive_') or stage == 'done'):
        # register/report have finished. Collection is already on the next scan.
        daily = ROOT / f'MEMORY/{dt.date.today():%Y-%m-%d}.md'
        previous = daily.read_text() if daily.exists() else ''
        with daily.open('a') as stream:
            for scan in added:
                marker = f'task105扫描完成：`{scan["task_id"]}`'
                if marker in previous:
                    continue
                if scan['registered']:
                    detail = (f'阶梯{scan["decisions"]}，未来块'
                              f'{scan["future_seed_start"]}–'
                              f'{scan["future_seed_start"] + scan["future_seed_count"] - 1}')
                else:
                    detail = f'未登记：{scan["reason"]}'
                stream.write(f'\n- {dt.datetime.now():%H:%M} {marker}，{detail}。'
                             f'累计{evidence["scanned"]}/{evidence["scan_expected"]}扫描、'
                             f'{evidence["registered"]}格、生产{evidence["deployed"]}格；未执行部署。\n')
        paths = [str(ROOT / name) for name in FILES] + [str(daily)]
        git('add', '--', *paths)
        # All allowed paths are text. An unexpected binary must never be committed.
        if any(line.startswith('-\t-\t') for line in git('diff', '--cached', '--numstat').splitlines()):
            git('reset', '--', *paths)
            raise ValueError('unexpected binary in result files')
        print(git('diff', '--cached', '--stat'), flush=True)
        print(git('commit', '-m', f'task105: record scan {evidence["scanned"]} '
                  f'({evidence["registered"]} start cells)'), flush=True)
    # A successful commit followed by a failed push must be retried on restart.
    if git('rev-list', '--count', 'origin/llmimport..HEAD') != '0':
        print(git('push', 'origin', 'llmimport'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        published = None
        while True:
            try:
                signature = tuple((path.stat().st_size, path.stat().st_mtime_ns) for path in
                                  (ROOT / EVIDENCE, OUT / 'pipeline_status.json'))
                if signature != published:
                    publish()
                    published = signature
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                if args.once:
                    raise
                print(f'publication deferred: {error}', flush=True)
            if args.once:
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
