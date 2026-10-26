#!/usr/bin/env python3
"""重复测量 task101 冒险模式记分板，并与上一份逐关比较。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from pvz_constants import ZOMBIE_NAME
from search_agreement_stats import wilson

PANEL = ROOT / 'configs/adventure_scoreboard.json'
BASELINE = ROOT / 'MEMORY/docs/adventure-scoreboard.json'
ARMS = {'student': 'fast', 'teacher': 'fast_search'}


def arm_stats(rows):
    wins = sum(row['won'] for row in rows)
    losses = [row for row in rows if not row['won']]
    counts = Counter(zombie['type'] for row in losses for zombie in row['final_zombies'])
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    return dict(wins=wins, games=len(rows), rate=wins / len(rows),
                wilson95=wilson(wins, len(rows)),
                loss_median_final_wave=statistics.median(row['final_wave'] for row in losses)
                                       if losses else None,
                loss_breach_zombies_top3=[dict(type_id=kind, name=ZOMBIE_NAME.get(kind, str(kind)),
                                               count=count) for kind, count in top])


def compare_scoreboards(current, previous):
    old_levels = {row['level']: row for row in previous['levels']}
    changes = []
    for row in current['levels']:
        old = old_levels.get(row['level'])
        if old is None:
            continue
        comparable = all(row.get(key) == old.get(key) for key in ('task_id', 'deck'))
        if old.get('task') is not None:
            comparable = comparable and row.get('task') == old['task']
        for arm in ARMS:
            before, after = old.get(arm), row.get(arm)
            if not before or not after:
                continue
            before_ci = wilson(before['wins'], before['games'])
            after_ci = wilson(after['wins'], after['games'])
            direction = ('上升' if after_ci[0] > before_ci[1] else
                         '下降' if before_ci[0] > after_ci[1] else '重叠')
            changes.append(dict(level=row['level'], arm=arm, comparable=comparable,
                previous_wins=before['wins'], previous_games=before['games'],
                wins=after['wins'], games=after['games'],
                delta_rate=after['wins'] / after['games'] - before['wins'] / before['games']
                           if comparable else None,
                previous_wilson95=before_ci, wilson95=after_ci,
                wilson_separated=direction != '重叠' if comparable else None,
                direction=direction if comparable else '任务不同'))
    return changes


def category(student, teacher):
    if student and student['rate'] >= .8:
        return '学生≥80%'
    if teacher:
        if teacher['rate'] >= .8:
            return '可教：老师≥80%，学生<80%' if student else '老师≥80%，学生未测'
        if teacher['rate'] < .5:
            return '需要示范、卡组或机制：老师<50%'
        return '继续观察：老师50%–80%'
    return '学生<80%，老师未测'


def scene(level):
    if level == 25:
        return '小僵尸原生传送带，原生已接入'
    if level in (5, 15, 35, 45):
        return '特殊小游戏/传送带，未接入'
    if level in (30, 40, 50):
        return {30: '泳池传送带', 40: '暴风雨传送带', 50: 'Boss屋顶传送带'}[level]
    return ('白天普通', '夜草普通', '泳池普通', '雾池普通', '屋顶普通')[(level - 1) // 10]


def markdown(report):
    lines = ['# 冒险模式实测记分板', '',
        f"参数：原版倍率 1.0，每关 {report['seeds_per_arm']} 个种子，"
        f"测量臂：{'、'.join('学生' if arm == 'student' else '老师' for arm in report['arms'])}。"
        '老师参数 beta=1、verify=9、student-candidates=2。', '',
        '任务定义冻结自 task101，L10/L20 保留当时的预装手牌变体，L30/L40/L50 为原生空手传送带。'
        '5、15、35、45 未接入。L25 按 task110 更正为原生已接入，本面板未测，保持45关测量范围。', '',
        '“破防僵尸前三”沿用 task101 的统计口径：输局终局仍在场的全部僵尸按种类计数，'
        '不能据此认定每一只都已到家。', '',
        '| 关 | 场景 | 卡组族 | 学生 | 老师 | 学生 Wilson95% | 老师 Wilson95% | '
        '输局末波（学生/老师） | 破防僵尸前三（学生/老师） | task55旧状态 | 分类 |',
        '|---:|---|---|---:|---:|---|---|---|---|---|---|']
    for row in report['levels']:
        values = []
        for arm in ARMS:
            stats = row.get(arm)
            values.append((f"{stats['wins']}/{stats['games']} ({stats['rate']:.1%})",
                f"{stats['wilson95'][0]:.1%}–{stats['wilson95'][1]:.1%}",
                str(stats['loss_median_final_wave']) if stats['loss_median_final_wave'] is not None else '—',
                '、'.join(f"{item['name']}({item['count']})" for item in stats['loss_breach_zombies_top3']) or '—')
                if stats else ('—', '—', '—', '—'))
        student, teacher = values
        deck = ','.join(map(str, row['deck'])) if row['deck'] is not None else '—'
        lines.append(f"| {row['level']} | {scene(row['level'])} | {deck} / {row['family']} | "
            f"{student[0]} | {teacher[0]} | {student[1]} | {teacher[1]} | {student[2]} / {teacher[2]} | "
            f"{student[3]} / {teacher[3]} | {row['old_status']} | {row['category']} |")
    if 'comparison' in report:
        lines += ['', '## 与上一份比较', '',
            f"上一份：`{report['previous']}`。胜率变化以百分点计。Wilson 区间分离只描述区间位置，"
            '不等同于配对显著性检验。缺测臂不参与比较，任务定义不同的行不计算变化。', '',
            '| 关 | 臂 | 上次 | 本次 | 变化 | Wilson95% 区间宽度（上次→本次） | Wilson95% 是否分离 |',
            '|---:|---|---:|---:|---:|---|---|']
        for change in report['comparison']:
            delta = f"{change['delta_rate'] * 100:+.2f}" if change['comparable'] else '—'
            separated = '是，' + change['direction'] if change['wilson_separated'] else change['direction']
            arm_name = '学生' if change['arm'] == 'student' else '老师'
            old_width = change['previous_wilson95'][1] - change['previous_wilson95'][0]
            width = change['wilson95'][1] - change['wilson95'][0]
            lines.append(f"| {change['level']} | {arm_name} | "
                f"{change['previous_wins']}/{change['previous_games']} | {change['wins']}/{change['games']} | "
                f"{delta} | {old_width:.2%} → {width:.2%} | {separated} |")
        for arm in report['arms']:
            changes = [item for item in report['comparison'] if item['arm'] == arm and item['comparable']]
            if changes:
                old_width = statistics.mean(item['previous_wilson95'][1] - item['previous_wilson95'][0]
                                            for item in changes)
                width = statistics.mean(item['wilson95'][1] - item['wilson95'][0] for item in changes)
                lines += ['', f"{'学生' if arm == 'student' else '老师'} {len(changes)} 关的 Wilson95% "
                    f"平均区间宽度：{old_width:.2%} → {width:.2%}。"
                    f"区间分离 {sum(item['wilson_separated'] for item in changes)} 关。"]
    return '\n'.join(lines) + '\n'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student', required=True, type=Path)
    parser.add_argument('--value', required=True, type=Path)
    parser.add_argument('--static-value', required=True, type=Path)
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--resource-dir', required=True, type=Path)
    parser.add_argument('--seeds', type=int, default=32, help='每关每臂种子数')
    parser.add_argument('--seed-offset', type=int, default=0, help='各关已登记种子块内的起点')
    parser.add_argument('--levels', default=','.join(map(str, range(1, 51))), help='逗号分隔的关号')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=2)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--student-only', action='store_true')
    selection.add_argument('--teacher-only', action='store_true')
    parser.add_argument('--panel', type=Path, default=PANEL,
                        help='任务面板（默认 configs/adventure_scoreboard.json）')
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--out', required=True, type=Path, help='运行目录，内含 scoreboard.json / scoreboard.md')
    args = parser.parse_args()
    panel = json.loads(args.panel.resolve().read_text())
    if args.seeds < 1 or args.seed_offset < 0 or args.seed_offset + args.seeds > panel['stride']:
        parser.error('种子数与起点必须落在每关已登记的 10000 个种子内')
    try:
        levels = sorted(set(int(level) for level in args.levels.split(',')))
    except ValueError:
        parser.error('--levels 必须是逗号分隔的整数关号')
    if not levels or not set(levels) <= set(range(1, 51)):
        parser.error('关号必须在 1–50 内')
    arms = ['student'] if args.student_only else ['teacher'] if args.teacher_only else list(ARMS)
    models = {name: getattr(args, name).resolve() for name in ('student', 'value', 'static_value')}
    executable = args.executable.resolve()
    args.out = args.out.resolve()
    baseline = json.loads(BASELINE.read_text())
    old_levels = {row['level']: row for row in baseline['levels']}
    tasks = []
    for definition in panel['tasks']:
        if definition['level'] not in levels:
            continue
        start = panel['seed_block'][0] + (definition['level'] - 1) * panel['stride'] + args.seed_offset
        tasks.append(dict(definition, seeds=list(range(start, start + args.seeds))))
    if not tasks:
        parser.error('所选关卡全部未纳入冻结测量面板')
    report = dict(task='adventure_scoreboard', generated=date.today().isoformat(),
        engine=str(executable), engine_sha256=sha256(executable), multiplier=1.0,
        seeds_per_arm=args.seeds, seed_offset=args.seed_offset, seed_block=panel['seed_block'],
        arms=arms, skipped_levels=[level for level in levels if level in panel['skipped_levels']],
        models={key: str(path) for key, path in models.items()},
        model_sha256={key: sha256(path) for key, path in models.items()}, levels=[])
    manifest = dict(schema_version=1, tasks=tasks)
    identity = dict(manifest=manifest, arms=arms, engine_sha256=report['engine_sha256'],
                    model_sha256=report['model_sha256'], resource_dir=str(args.resource_dir.resolve()))
    args.out.mkdir(parents=True, exist_ok=True)
    identity_path = args.out / 'run.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        parser.error('输出目录已用于不同任务、种子或模型，请使用另一个目录')
    identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + '\n')
    manifest_path = args.out / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    for task in tasks:
        run = args.out / 'runs' / task['task_id']
        command = [sys.executable, str(ROOT / 'scripts/search_play.py'),
            '--rollout-policy', str(models['student']), '--value', str(models['value']),
            '--static-value', str(models['static_value']), '--executable', str(executable),
            '--resource-dir', str(args.resource_dir.resolve()), '--manifest', str(manifest_path),
            '--task-id', task['task_id'], '--seeds', str(args.seeds), '--workers', str(args.workers),
            '--arms', ','.join(ARMS[arm] for arm in arms), '--dagger-beta', '1',
            '--verify-samples', '9', '--student-candidates', '2', '--out', str(run)]
        print(f"L{task['level']}：{args.seeds} 种子 × {len(arms)} 臂", flush=True)
        subprocess.run(command, check=True, env=dict(os.environ, PVZ_NO_POPUP='1'))
        rows = [json.loads(line) for line in (run / 'games.jsonl').read_text().splitlines()]
        expected = {(ARMS[arm], seed) for arm in arms for seed in task['seeds']}
        if (len(rows) != len(expected) or {(row['arm'], row['seed']) for row in rows} != expected
                or any(row.get('terminal') is not True for row in rows)):
            raise ValueError(f'{run}: 对局缺失、重复、非终局或臂/种子不符')
        old = old_levels[task['level']]
        result = dict(level=task['level'], task_id=task['task_id'], deck=task['deck'],
            task={key: value for key, value in task.items() if key != 'seeds'}, seeds=task['seeds'],
            family=old['family'], old_status=old['old_status'], student=None, teacher=None)
        for arm in arms:
            result[arm] = arm_stats([row for row in rows if row['arm'] == ARMS[arm]])
        result['category'] = category(result['student'], result['teacher'])
        report['levels'].append(result)
    for level in report['skipped_levels']:
        report['levels'].append(dict(old_levels[level]))
    report['levels'].sort(key=lambda row: row['level'])
    if args.previous:
        report['previous'] = str(args.previous.resolve())
        report['comparison'] = compare_scoreboards(report, json.loads(args.previous.read_text()))
    (args.out / 'scoreboard.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (args.out / 'scoreboard.md').write_text(markdown(report))
    print(f"已写入 {args.out / 'scoreboard.json'} 与 {args.out / 'scoreboard.md'}", flush=True)


if __name__ == '__main__':
    main()
