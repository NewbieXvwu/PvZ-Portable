#!/usr/bin/env python3
"""task31 第 1 步：按“被换掉的底卡”统计老师胜率，判据只有一个——被换掉后胜率 95% 上界 < 阈值。

数据源（全部是已采档案，本脚本不跑局）：
  * 夜间：night_l{12,14,17}_train_n10p 的老师局（--dagger-beta 1，dw4i_s1 推演）。
  * 白天：day_l{7..10}_train_d7p / 留出族的**学生**局（d7p 采集全部是 --dagger-beta 0，
    没有白天老师局档案；老师/学生按 run 的 config.json dagger_beta 区分）。
每局用 deck_for_seed（扰动族）或任务固定卡组（留出族）得到实际初始卡组，
removed = 底卡组 - 实际卡组。对每张底卡、按老师/学生分组统计：
  - 被换掉的局数 / 胜 / 胜率 / 95% 上界（Clopper–Pearson）；
  - 对照：同一批局里这张卡**保留**的局数 / 胜 / 胜率。

判据：**老师**表里 upper < --keep-threshold（默认 0.25）→ 列为保留卡。不按植物写任何特例。
学生表只作筛查：学生胜率已经很高（上界 ≥ 阈值）的底卡，老师只会更好，不必补老师局；
唯一贴着阈值的底卡才补（补样本用 scripts/pick_keep_seeds.py 做条件采样）。

用法：
  python3 scripts/analyze_keep_cards.py \
      --night-manifest /home/newbiexvwu/PvZ-search/artifacts/stage4/night_panel_n10p.json \
      --night-run night_l12_train_n10p=<dir> ... \
      --day-manifest /home/newbiexvwu/PvZ-search/.claude/worktrees/identity/artifacts/stage4/day_panel_identity.json \
      --day-run day_l7_train_d7p=<dir> ... \
      --out artifacts/task31/keep_cards.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_constants import PLANT_NAME  # noqa: E402
from pvz_tasks import deck_for_seed  # noqa: E402
from render_episode import load_training_task  # noqa: E402


def binom_cdf(n: int, x: int, p: float) -> float:
    return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(x + 1))


def clopper_pearson(n: int, x: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval (Clopper–Pearson)；n=0 返回 (0, 1)。"""
    if n == 0:
        return 0.0, 1.0
    if x == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0
        # P(X >= x; p) 随 p 递增；找它等于 alpha/2 的 p。
        for _ in range(80):
            mid = (lo + hi) / 2
            if 1 - binom_cdf(n, x - 1, mid) < alpha / 2:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    if x == n:
        upper = 1.0
    else:
        lo, hi = 0.0, 1.0
        # P(X <= x; p) 随 p 递减；找它等于 alpha/2 的 p。
        for _ in range(80):
            mid = (lo + hi) / 2
            if binom_cdf(n, x, mid) > alpha / 2:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return lower, upper


def run_dagger_beta(folder: Path) -> float | None:
    """run 的 dagger beta：1.0 = 老师推演执行（老师局），0.0 = 学生执行（老师只给标签）。

    games.jsonl 里没有 beta 字段，而且 beta=0 的 DAgger 采集同样会推演（searched > 0），
    所以“searched > 0”不能当老师局标记 —— 只有 config.json 的 dagger_beta 能区分。
    """
    path = folder / 'config.json'
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text()).get('dagger_beta'))
    except (TypeError, ValueError):
        return None


def load_games(folder: Path) -> list[dict]:
    path = folder / 'games.jsonl'
    if not path.exists():
        raise SystemExit(f'missing games.jsonl: {folder}')
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def collect(manifest: Path, runs: list[tuple[str, Path]], base_override=None) -> list[dict]:
    rows = []
    for task_id, folder in runs:
        task = load_training_task(manifest, task_id)
        # 扰动族：task['deck'] 就是底卡组；固定留出族：必须显式给族底卡组。
        if 'deck_perturb' in task:
            base = tuple(task['deck'])
        elif base_override:
            base = tuple(base_override)
        else:
            raise SystemExit(f'{task_id}: fixed deck task needs --*-base')
        beta = run_dagger_beta(folder)
        for game in load_games(folder):
            deck = tuple(deck_for_seed(task, game['seed']))
            rows.append(dict(task_id=task_id, level=int(task['level']), seed=game['seed'],
                             won=bool(game['won']), deck=deck, base=base, beta=beta,
                             teacher=bool(beta),
                             removed=sorted(set(base) - set(deck))))
    return rows


def summarize(rows: list[dict], group: str, kind: str, suspect=()) -> list[dict]:
    """suspect：同族里“可能也很关键”的底卡集合。测某张卡时把“同时换掉另一张 suspect”的局剔掉，
    否则那张卡造成的输局会算到这张卡头上（条件采样的 8 号卡批次里，9 号卡也常被一起换掉）。
    对照列同样只用剔完剩下的局。"""
    rows = [row for row in rows if row['teacher'] == (kind == 'teacher')]
    cards = sorted({card for row in rows for card in row['base']})
    table = []
    for card in cards:
        others = set(suspect) - {card}
        removed = [row for row in rows if card in row['removed'] and not (others & set(row['removed']))]
        kept = [row for row in rows if card in row['deck']]
        wins = sum(row['won'] for row in removed)
        kept_wins = sum(row['won'] for row in kept)
        lower, upper = clopper_pearson(len(removed), wins)
        table.append(dict(group=group, kind=kind, card=card,
                          removed_games=len(removed), removed_wins=wins,
                          removed_rate=(wins / len(removed) if removed else None),
                          removed_ci95=[lower, upper],
                          kept_games=len(kept), kept_wins=kept_wins,
                          kept_rate=(kept_wins / len(kept) if kept else None)))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--night-manifest', type=Path, required=True)
    parser.add_argument('--night-run', action='append', default=[], metavar='TASK=DIR')
    parser.add_argument('--day-manifest', type=Path, required=True)
    parser.add_argument('--day-run', action='append', default=[], metavar='TASK=DIR')
    parser.add_argument('--keep-threshold', type=float, default=0.25)
    parser.add_argument('--suspect', default='', help='逗号分隔：测一张卡时，把同时换掉这些卡的局剔掉')
    parser.add_argument('--night-base', default='0,1,2,3,4,5,7,8,9,11')
    parser.add_argument('--day-base', default='0,1,2,3,4,5,7')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--markdown', action='store_true', help='把表按 markdown 输出（直接贴进报告）')
    args = parser.parse_args()

    def parse(items):
        parsed = []
        for item in items:
            task_id, _, folder = item.partition('=')
            parsed.append((task_id, Path(folder)))
        return parsed

    result = {'keep_threshold': args.keep_threshold}
    tables = []
    for group, manifest, runs, base in (('night', args.night_manifest, args.night_run, args.night_base),
                                        ('day', args.day_manifest, args.day_run, args.day_base)):
        if not runs:
            continue
        rows = collect(manifest, parse(runs), [int(v) for v in base.split(',') if v != ''])
        suspect = [int(v) for v in args.suspect.split(',') if v.strip()]
        teacher = summarize(rows, group, 'teacher', suspect)
        student = summarize(rows, group, 'student', suspect)
        tables += teacher + student
        result[group] = dict(games=len(rows), wins=sum(row['won'] for row in rows),
                             teacher_games=sum(row['teacher'] for row in rows),
                             student_games=sum(not row['teacher'] for row in rows),
                             cards=teacher, student_cards=student)
    for table in tables:
        table['keep'] = table['kind'] == 'teacher' and table['removed_ci95'][1] < args.keep_threshold
    result['keep_cards'] = {group: [row['card'] for row in tables
                                    if row['group'] == group and row['keep']]
                            for group in ('day', 'night')}
    print(f"{'grp':<6}{'kind':<9}{'card':>5}{'n':>6}{'win':>5}{'rate':>7}{'upper':>7}"
          f"{'ctrl_n':>7}{'ctrl%':>7}  keep")
    for row in tables:
        rate = f"{row['removed_rate']:.2f}" if row['removed_rate'] is not None else '-'
        ctrl = f"{row['kept_rate']:.2f}" if row['kept_rate'] is not None else '-'
        print(f"{row['group']:<6}{row['kind']:<9}{row['card']:>5}{row['removed_games']:>6}"
              f"{row['removed_wins']:>5}{rate:>7}{row['removed_ci95'][1]:>7.2f}"
              f"{row['kept_games']:>7}{ctrl:>7}  {'YES' if row['keep'] else ''}")
    if args.markdown:
        print()
        print('| 族 | 底卡 | 被换掉的局数 | 胜 | 胜率 | 95% 上界 | 对照局数 | 对照胜率 | 保留 |')
        print('|---|---|---:|---:|---:|---:|---:|---:|:--:|')
        for row in tables:
            if row['kind'] != 'teacher':
                continue
            rate = f"{row['removed_rate']:.1%}" if row['removed_rate'] is not None else '—'
            ctrl = f"{row['kept_rate']:.1%}" if row['kept_rate'] is not None else '—'
            name = PLANT_NAME.get(row['card'], str(row['card']))
            print(f"| {row['group']} | {row['card']} {name} | {row['removed_games']} | {row['removed_wins']} | "
                  f"{rate} | {row['removed_ci95'][1]:.2f} | {row['kept_games']} | {ctrl} | "
                  f"{'**保留**' if row['keep'] else ''} |")
    print('keep cards:', result['keep_cards'])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
