"""Pool leaf dispersion: does v4 flatten candidate ranking where pool games are lost?

Runs the same teacher as the task30 paired A/B (dw4i_s1 rollout, beta=1, branch1 static
value, halving, verify=9, student-candidates=2) on pool L24/L26 with only the leaf value
net swapped, and records per search decision the spread of candidate leaf values
(the --leaf-probe hook in search_play.py). Offline A6 showed v4 predicting ~0.005 for the
all-loss pool cohort; this measures whether that saturation removes the ranking signal
the search needs to override the student.

Diagnostic only: no defaults change and no game result is published from here.
"""
from __future__ import annotations
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
TASKS = ('pool_l24_eval_p9k', 'pool_l26_eval_p9k')
# search_play.py's own override margin: a top1-top2 gap at or below this means the two
# best candidates are inside the threshold the search itself requires to switch choices.
MARGIN = 0.02


def run(args) -> None:
    for arm in ('v3', 'v4'):
        for task in TASKS:
            out = args.out / arm / task
            out.mkdir(parents=True, exist_ok=True)
            probe = out / 'leaves'
            for stale in out.glob('leaves.*.jsonl'):
                stale.unlink()
            cmd = [sys.executable, str(ROOT / 'scripts/search_play.py'),
                   '--rollout-policy', str(MAIN / 'artifacts/student/dw4i_s1/model.pt'),
                   '--static-value', str(MAIN / 'artifacts/hard/value_branch1/model.pt'),
                   '--value', str(args.v3 if arm == 'v3' else args.v4),
                   '--verify-samples', '9', '--student-candidates', '2',
                   '--manifest', str(MAIN / 'artifacts/stage4/pool_panel.json'), '--task-id', task,
                   '--seeds', str(args.games), '--seed-offset', str(args.seed_offset),
                   '--workers', str(args.workers),
                   '--executable', str(args.executable),
                   '--resource-dir', '/home/newbiexvwu/.cache/pvz-research-resources',
                   '--dagger-beta', '1', '--out', str(out),
                   '--leaf-probe', str(probe), '--leaf-probe-limit', str(args.limit)]
            print(f'{arm} {task}', flush=True)
            with (out / 'run.log').open('a') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)


def load(path: Path) -> list[dict]:
    rows = []
    for part in sorted(path.glob('leaves.*.jsonl')):
        rows.extend(json.loads(line) for line in part.read_text().splitlines() if line.strip())
    seen, unique = set(), []
    for row in rows:                      # identical (seed, decision) rows can repeat after a resume
        key = (row['task'], row['seed'], row['decision'])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return sorted(unique, key=lambda r: (r['task'], r['seed'], r['decision']))


def sample(rows: list[dict], n: int) -> list[dict]:
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[min(len(rows) - 1, int(i * step))] for i in range(n)]


def describe(rows: list[dict]) -> dict:
    if not rows:
        return dict(decisions=0)
    std = [r['std'] for r in rows]
    gap = [r['top1_top2'] for r in rows]
    return dict(decisions=len(rows), games=len({(r['task'], r['seed']) for r in rows}),
                mean_candidates=statistics.mean(r['candidates'] for r in rows),
                mean_scored=statistics.mean(r['scored'] for r in rows),
                mean_leaf=statistics.mean(r['mean'] for r in rows),
                mean_std=statistics.mean(std), median_std=statistics.median(std),
                mean_gap=statistics.mean(gap), median_gap=statistics.median(gap),
                share_gap_below_margin=sum(g <= MARGIN for g in gap) / len(gap),
                share_overridden=sum(r['overridden'] for r in rows) / len(rows))


def report(args) -> None:
    result = dict(protocol='teacher search decisions, pool L24/L26, only the leaf value net swapped',
                  margin=MARGIN, tasks=list(TASKS), games_per_task=args.games,
                  sampled_per_arm=args.sample, arms={}, per_level={})
    for arm in ('v3', 'v4'):
        rows = [r for task in TASKS for r in load(args.out / arm / task)]
        picked = sample(rows, args.sample)
        result['arms'][arm] = {**describe(picked), 'all_decisions': describe(rows)}
        result['per_level'][arm] = {task: describe([r for r in rows if r['task'] == task]) for task in TASKS}
    # Per-game means keep the clustering visible: decisions inside one game are not independent.
    for arm in ('v3', 'v4'):
        games = {}
        for task in TASKS:
            for row in load(args.out / arm / task):
                games.setdefault((task, row["seed"]), []).append(row)
        result['arms'][arm]['per_game_mean_gap'] = sorted(
            round(statistics.mean(r['top1_top2'] for r in rows), 6) for rows in games.values())
        result['arms'][arm]['per_game_override_rate'] = sorted(
            round(sum(r['overridden'] for r in rows) / len(rows), 4) for rows in games.values())
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=1) + chr(10))
    for arm, data in result['arms'].items():
        print(arm, "n=", data["decisions"], "mean_std=", round(data["mean_std"], 4),
              "mean_gap=", round(data["mean_gap"], 4), "flat=", round(data["share_gap_below_margin"], 4),
              "override=", round(data["share_overridden"], 4), "mean_leaf=", round(data["mean_leaf"], 4))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['run', 'report'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--json', type=Path, default=ROOT / 'MEMORY/docs/value-widen-dispersion.json')
    p.add_argument('--v3', type=Path, default=MAIN / 'artifacts/value_net/v3/model.pt')
    p.add_argument('--v4', type=Path, default=ROOT / 'artifacts/task30/v4/model.pt')
    p.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    p.add_argument('--games', type=int, default=4)
    p.add_argument('--seed-offset', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=80, help='per worker decision cap')
    p.add_argument('--sample', type=int, default=200)
    a = p.parse_args()
    if not 1 <= a.workers <= 4:
        p.error('--workers must be 1..4')
    run(a) if a.phase == 'run' else report(a)


if __name__ == '__main__':
    main()
