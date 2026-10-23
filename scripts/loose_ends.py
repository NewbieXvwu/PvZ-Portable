#!/usr/bin/env python3
"""每次巡检跑一遍：列出“做了一半 / 做完没用上”的线头（只读）。

三类：
1. 近期有提交、还没合进 pvz-env 的 origin 分支（做完的分支不合，等于没做）；
2. 任务书“结果 / 问题”里交给主控、还没落地的话（待主控 / 待合并 / 默认关 …），只看主线上的任务书；
3. 计划文档组件状态表（## 00.）里不是“在役”的行。

用法：python3 scripts/loose_ends.py [--days 14]（先 git fetch origin）
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAINLINE = 'pvz-env'
PLAN = ROOT / 'MEMORY/docs/student-bottleneck-plan.md'
# 交接给主控、或“做了但没开”的说法。改这里就是改巡检的口径。
HANDOFF = re.compile(r'待主控|主控决定|主控批|留给主控|请主控|待合并|未合并|未上线|没上线|默认关|默认仍关|opt-in|待批准|尚未接入|待接入|未接线')


def git(*args: str) -> str:
    return subprocess.run(['git', '-C', str(ROOT), *args], check=True, capture_output=True, text=True).stdout


def unmerged_branches(days: int) -> list[str]:
    rows = []
    for ref in git('for-each-ref', '--format=%(refname:short)', 'refs/remotes/origin').split():
        if ref.endswith('/HEAD') or ref == f'origin/{MAINLINE}':
            continue
        if git('log', '-1', f'--since={days}.days', '--format=%h', ref).strip() == '':
            continue
        ahead = int(git('rev-list', '--count', f'{MAINLINE}..{ref}').strip())
        if ahead == 0:
            continue
        when, subject = git('log', '-1', '--date=format:%m-%d %H:%M', '--format=%ad\t%s', ref).strip().split('\t', 1)
        rows.append(f'{when}  +{ahead:<3} {ref:<24} {subject[:80]}')
    return sorted(rows, reverse=True)


def handoff_lines() -> list[str]:
    rows = []
    for path in sorted((ROOT / 'MEMORY/docs').glob('task*.md')):
        section = None
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.startswith('## '):
                section = line[3:].strip()
                continue
            if section in ('结果', '问题') and HANDOFF.search(line):
                rows.append(f'{path.relative_to(ROOT)}:{number}  {line.strip()[:150]}')
    return rows


def status_rows() -> list[str]:
    rows, inside = [], False
    for line in PLAN.read_text().splitlines():
        if line.startswith('## 00.'):
            inside = True
            continue
        if inside and line.startswith('## '):
            break
        if inside and line.startswith('| ') and not line.startswith('| 组件') and not line.startswith('|---'):
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            if len(cells) >= 2 and cells[1] != '在役':
                rows.append(f'{cells[0][:40]:<40} {cells[1][:60]}')
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--days', type=int, default=14)
    args = parser.parse_args()
    for title, rows in (('未合进主线的近期分支', unmerged_branches(args.days)),
                        ('任务书里交给主控、待落地的话', handoff_lines()),
                        ('组件状态表里不是“在役”的', status_rows())):
        print(f'== {title}（{len(rows)}）')
        print('\n'.join(rows) if rows else '（无）')
        print()


if __name__ == '__main__':
    main()
