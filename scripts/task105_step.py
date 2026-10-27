#!/usr/bin/env python3
"""Commit one completed task105 scan step: memory log line, stage, commit, push.

Reads only the small result JSONs the pipeline already writes. Refuses to commit
anything outside the six tracked files, and refuses to push when the remote
branch moved (someone else's commit must not be silently merged).
"""
import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/task105'
FILES = ['DESIGN.md', 'MEMORY/2026-10-28.md', 'MEMORY/docs/task105-gptpro-run1-import.md',
         'MEMORY/docs/task105-import-evidence.json', 'configs/human_demo_tasks.json',
         'python/test_curriculum_queue.py']


def git(*args, check=True):
    return subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True, check=check)


def scan_points(task_id):
    path = OUT / f'adaptive_{task_id}.out'
    teacher, student = {}, {}
    if not path.exists():
        return teacher, student
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        match = re.search(r'\bd(\d+) (teacher|student): (\d+)/(\d+)', line)
        if match:
            (teacher if match.group(2) == 'teacher' else student)[int(match.group(1))] = (
                int(match.group(3)), int(match.group(4)))
    return teacher, student


def describe(scan):
    teacher, student = scan_points(scan['task_id'])
    ladder = scan['decisions']
    top = scan['top']
    if top is None:
        head = '无过线档'
    elif scan['l50'] == top:
        head = f"L50/顶档d{top}"
    else:
        head = f"L50 d{scan['l50']}、顶档d{top}"
    def rate(table, decision):
        if decision not in table:
            return '—'
        wins, games = table[decision]
        return f'{wins}/{games}'
    if ladder:
        teacher_text = '、'.join(rate(teacher, d) for d in ladder)
        student_text = '、'.join(rate(student, d) for d in ladder)
        note = f'（老师{teacher_text}；学生{student_text}'
        missing = [d for d in ladder if d not in student]
        if missing:
            note += f"，无{'/'.join(map(str, missing))}学生点"
        note += '）'
    else:
        note = '（老师与学生在全部候选点均<50%）'
    start, count = scan['future_seed_start'], scan['future_seed_count']
    block = f'{start}–{start + count - 1}' if start is not None and count else '—'
    return f"{head}，阶梯[{', '.join(map(str, ladder)) or '—'}]{note}、未来块{block}"


def label(task_id):
    """demo_roof_48_doom_m400_s81017037_xxx -> L48 doom_m400 s81017037"""
    match = re.match(r'^(?:demo_roof_)?(\d+)_(.+)_s(\d+)(?:_[0-9a-f]+)?$', task_id)
    if not match:
        return task_id
    return f'L{match.group(1)} {match.group(2)} s{match.group(3)}'


def stage_label(stage):
    """adaptive_demo_roof_48_doom_m400_s81017037_a5f9721e6e57a -> 同格L48 doom_m400 s81017037、SHA前缀a5f9721e6e57"""
    if not stage.startswith('adaptive_'):
        return stage
    task = stage[len('adaptive_'):]
    return f"同格{label(task)}、SHA前缀{task.rsplit('_', 1)[-1][:12]}"


def log_line(scan, result, next_stage, stamp):
    task = scan['task_id']
    prefix = task.rsplit('_', 1)[-1][:12]
    entry = (f"- {stamp} task105完成{result['scanned']}/{result['scan_expected']}、"
             f"登记{result['registered']}格：{label(task)}"
             f"、SHA前缀{prefix}，共{scan['games']}局"
             f"（{len(scan_points(task)[0])}老师点、{len(scan_points(task)[1])}学生点），"
             f"{describe(scan)}。流水线已自动同步任务书、结论和部署计划，转入{next_stage}；"
             f"复用既有回执和对局断点，每任务两个游戏进程，生产仍{result['deployed']}格、未执行部署。")
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--next', default='', help='下一阶段描述，例如 L48 doom_m400 s81017037、SHA前缀a5f9721e6e57')
    parser.add_argument('--no-push', action='store_true')
    args = parser.parse_args()
    result = json.loads((OUT / 'delivery_summary.json').read_text())
    scan = result['scans'][-1]
    stamp = dt.datetime.now().strftime('%H:%M')
    stage = json.loads((OUT / 'pipeline_status.json').read_text()).get('stage', '')
    next_stage = args.next or stage_label(stage)
    line = log_line(scan, result, next_stage, stamp)
    journal = ROOT / 'MEMORY' / f'{dt.date.today().isoformat()}.md'
    text = journal.read_text(encoding='utf-8')
    if f"task105完成{result['scanned']}/" not in text:
        journal.write_text(text.rstrip('\n') + '\n\n' + line + '\n', encoding='utf-8')
    print(line)
    git('add', *FILES)
    changed = git('diff', '--cached', '--name-only').stdout.split()
    extra = sorted(set(changed) - set(FILES))
    if extra:
        raise SystemExit(f'暂存区出现未预期文件：{extra}')
    if not changed:
        print('没有变更，跳过提交')
        return
    print(git('diff', '--cached', '--numstat').stdout, end='')
    if not args.no_push:
        git('fetch', '-q', 'origin', 'llmimport')
        behind = git('rev-list', '--count', 'HEAD..origin/llmimport').stdout.strip()
        if behind != '0':
            git('reset', '-q')
            raise SystemExit(f'远端领先 {behind} 个提交，已撤销暂存，先人工处理再推送')
    git('commit', '-q', '-m', f"task105: register scan {result['scanned']} and refresh results")
    if not args.no_push:
        git('push', '-q', 'origin', 'llmimport')
    print(git('log', '--oneline', '-1').stdout.strip())


if __name__ == '__main__':
    main()
