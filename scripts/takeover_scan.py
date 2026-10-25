#!/usr/bin/env python3
"""task91：接手扫描运行器 —— 把 LLM 通关局一条命令接到课程里。

三个子命令（对应 MEMORY/docs/task91-takeover-scan-runner.md）：

  run <scan_jobs.json>      按 task89 的作业清单跑老师/学生两臂，断点续跑，
                            结果按存档写 json（每局胜负、末波、破防行与僵尸类型）。
  adaptive <scan_jobs.json> 自适应取点：4 等距点老师臂 → 跨 50% 区间二分 →
                            50% 线到顶档之间补齐网格档，学生臂只打两端。
  register <result.json…>   判读：老师 ≥50% 的存档生成阶梯（档位全部 ≥50%，
                            顶档取最晚的稳定 8/8 点），写 demo_start；未来种子块
                            用独立加锁账本分配；同步 DESIGN 与测试期望集合。

生产老师参数（curriculum_queue 同款）：学生 r11、叶值 v5、静态网 branch1、
--verify-samples 9 --student-candidates 2；老师臂 dagger-beta 1，学生臂 0。
"""

import argparse
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from import_llm_run import allocate_future_block, occupied_seed_ranges  # noqa: E402

# 生产模型（task91 任务书：学生 r11、叶值 v5、静态网 branch1）。
DEFAULT_STUDENT = Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/round_000011/student/model.pt')
DEFAULT_VALUE = Path('/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt')
DEFAULT_STATIC = Path('/home/newbiexvwu/PvZ-search/artifacts/hard/value_branch1/model.pt')
DEFAULT_RESOURCES = Path.home() / '.cache/pvz-research-resources'
DEFAULT_MANIFEST = ROOT / 'configs/human_demo_tasks.json'
DEFAULT_OUT = ROOT / 'artifacts/llm_demo/scan_out'
DEFAULT_LEDGER = ROOT / 'artifacts/llm_demo/register_seed_ledger.json'
DEFAULT_REGISTER_BASE = 86_000_000   # import 的扫描账本从 84M 起每份 +1000，登记块挪到 86M 起
REGISTER_BLOCK = 256                 # task71b 先例：每个 demo_start 格 256 个未来种子
PASS_RATE = 0.5
# search_play 的臂名与 dagger-beta：teacher = 搜索老师（β=1），student = 学生独自走（β=0）。
ARM_SPEC = {'teacher': ('fast_search', 1.0), 'student': ('fast', 0.0)}
RESULT_SCHEMA = 'takeover_scan_results_v1'


def load_jobs(path: Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    if plan.get('schema') != 'llm_scan_jobs_v1':
        raise ValueError(f'{path}: 不是 llm_scan_jobs_v1 扫描清单')
    return plan


def resolve_archive(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def job_out(out: Path, job: dict) -> Path:
    return out / job['task_id'] / f"s{job['seed']}" / job['arm'] / f"d{int(job['decision']):05d}"


def read_rows(out: Path) -> list:
    games = out / 'games.jsonl'
    if not games.exists():
        return []
    return [json.loads(line) for line in games.read_text().splitlines() if line.strip()]


def trim_row(row: dict) -> dict:
    keep = ('future_seed', 'won', 'final_wave', 'final_tick', 'mowers_used', 'decisions')
    trimmed = {key: row[key] for key in keep if key in row}
    if row.get('final_zombies') is not None:
        trimmed['final_zombies'] = row['final_zombies']
    return trimmed


def search_play_command(job: dict, missing: list, out: Path, args) -> list:
    arm, beta = ARM_SPEC[job['arm']]
    return [str(args.python), str(ROOT / 'scripts/search_play.py'),
            '--rollout-policy', str(args.student),
            '--value', str(args.value), '--static-value', str(args.static_value),
            '--verify-samples', '9', '--student-candidates', '2',
            '--manifest', str(args.manifest), '--task-id', job['task_id'],
            '--seed-offset', '0', '--seeds', '1',
            '--demo-archive', str(resolve_archive(job['archive'])),
            '--demo-decision', str(job['decision']),
            '--future-seeds', ','.join(str(seed) for seed in missing),
            '--arms', arm, '--dagger-beta', f'{beta:g}',
            '--executable', str(args.executable),
            '--resource-dir', str(args.resource_dir),
            '--workers', str(args.workers), '--out', str(out)]


def run_job(job: dict, args) -> list:
    """跑一个 (存档, 点, 臂) 作业；断点续跑靠 search_play 自己的 games.jsonl。"""
    out = job_out(args.out, job)
    out.mkdir(parents=True, exist_ok=True)
    done = {row.get('future_seed') for row in read_rows(out)}
    missing = [seed for seed in job['future_seeds'] if seed not in done]
    if missing:
        env = dict(os.environ, PVZ_NO_POPUP='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        proc = subprocess.run(search_play_command(job, missing, out, args), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            tail = '\n'.join(proc.stdout.splitlines()[-15:])
            raise RuntimeError(f"search_play 失败 ({job['task_id']} d{job['decision']} {job['arm']}):\n{tail}")
    return read_rows(out)


def point_summary(rows: list) -> dict:
    return dict(wins=sum(1 for row in rows if row.get('won')), games=len(rows), rows=[trim_row(r) for r in rows])


def entry_filter(entry: dict, args) -> bool:
    if args.archive and args.archive not in entry['task_id'] and args.archive != str(entry['seed']):
        return False
    return True


def job_map(entry: dict, points: list, futures: list) -> dict:
    jobs = {}
    for job in entry['jobs']:
        if int(job['decision']) in points:
            copy = dict(job)
            copy['future_seeds'] = list(futures)
            jobs[(int(job['decision']), job['arm'])] = copy
    return jobs


def write_archive_result(entry: dict, points: dict, out: Path) -> Path:
    document = dict(schema=RESULT_SCHEMA, task_id=entry['task_id'], seed=entry['seed'],
                    archive=entry['archive'], grade=entry.get('grade'), layer=entry.get('layer'),
                    points=points)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{entry['task_id']}_s{entry['seed']}.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return path


def run_point(jobs: dict, decision: int, arm: str, arms: tuple, args) -> dict:
    if arm not in arms:
        return None
    job = jobs.get((decision, arm))
    if job is None:
        raise KeyError(f"清单里没有 {decision} 点的 {arm} 作业")
    rows = run_job(job, args)
    summary = point_summary(rows)
    print(f"  {job['task_id']} d{decision} {arm}: {summary['wins']}/{summary['games']}", flush=True)
    return summary


# ---------------------------------------------------------------- run

def cmd_run(args) -> None:
    plan = load_jobs(args.jobs)
    entries = [entry for entry in plan['archives'] if entry_filter(entry, args)]
    if args.max_archives:
        entries = entries[:args.max_archives]
    arms = ('teacher',) if args.teacher_only else ('teacher', 'student')
    for entry in entries:
        futures = list(entry['future_seeds'])[:args.futures]
        all_points = [int(d) for d in entry['decisions']]
        points = all_points if not args.decisions else [int(d) for d in args.decisions.split(',')]
        jobs = job_map(entry, points, futures)
        points_path = args.out / 'results'
        store = {}
        existing = points_path / f"{entry['task_id']}_s{entry['seed']}.json"
        if existing.exists():
            store = json.loads(existing.read_text(encoding='utf-8')).get('points', {})
        for decision in points:
            bucket = store.setdefault(str(decision), {})
            for arm in arms:
                if bucket.get(arm) and len(bucket[arm].get('rows', [])) >= len(futures):
                    continue    # 断点续跑：该点两臂已齐
                summary = run_point(jobs, decision, arm, arms, args)
                if summary is not None:
                    bucket[arm] = summary
                    write_archive_result(entry, store, points_path)
        write_archive_result(entry, store, points_path)
    print(f'完成 {len(entries)} 份存档，结果在 {args.out / "results"}')


# ---------------------------------------------------------------- 自适应取点

class AdaptiveScan:
    """自适应取点的纯决策逻辑（可用合成胜率单测）。

    阶段：4 等距点 → 跨 50% 区间二分 → [50% 线, 顶档] 之间补齐网格档。
    所有实测点都落在 task89 的 12 点网格上，与 12 点结果可直接比档。
    """

    def __init__(self, grid: list, equidistant: int = 4):
        self.grid = [int(d) for d in grid]
        self.wins: dict[int, dict] = {}       # decision -> {'teacher': (wins, games), 'student': …}
        self.equidistant = self._pick_equidistant(len(self.grid), equidistant)
        self.stage = 'equidistant'
        self.next_queue: list = []
        self.l50: int | None = None
        self.top: int | None = None
        self.student_points: list = []

    @staticmethod
    def _pick_equidistant(count: int, points: int) -> list:
        if count <= points:
            return list(range(count))
        return sorted({round(index * (count - 1) / (points - 1)) for index in range(points)})

    def rate(self, decision: int) -> float | None:
        data = self.wins.get(decision, {}).get('teacher')
        if not data or not data[1]:
            return None
        return data[0] / data[1]

    def next_teacher_point(self) -> int | None:
        """返回下一个要跑老师臂的决策号；None 表示老师臂结束。"""
        while True:
            if self.next_queue:
                decision = self.next_queue.pop(0)
                if decision not in self.wins:
                    return decision
                continue
            if self.stage == 'equidistant':
                pending = [self.grid[i] for i in self.equidistant if self.grid[i] not in self.wins]
                if pending:
                    return pending[0]
                self.stage = 'bisect'
                self._plan_bisect()
                continue
            if self.stage == 'bisect':
                pending = [d for d in self.next_queue if d not in self.wins]
                if pending:
                    return pending[0]
                self.stage = 'fill'
                self._plan_fill()
                continue
            if self.stage == 'fill':
                return None
            return None

    def _grid_index(self, decision: int) -> int:
        return self.grid.index(decision)

    def _passing(self, decision: int) -> bool:
        rate = self.rate(decision)
        return rate is not None and rate >= PASS_RATE

    def _plan_bisect(self) -> None:
        """找一个失败→过线的相邻已测区间，往队列放一个未测中点。

        中点结果回来后 observe 会再次调用这里，区间逐步收缩到一档。
        """
        tested = sorted((self._grid_index(d), d) for d in self.wins)
        for (ia, da), (ib, db) in zip(tested, tested[1:]):
            if ib - ia > 1 and not self._passing(da) and self._passing(db):
                mid = self.grid[(ia + ib) // 2]
                if mid not in self.wins and mid not in self.next_queue:
                    self.next_queue.append(mid)
                return

    def _l50_index(self) -> int | None:
        passing = [self._grid_index(d) for d in self.wins if self._passing(d)]
        return min(passing) if passing else None

    def _top_index(self) -> int | None:
        """顶档：最晚的 8/8 稳定点；没有 8/8 就取最晚的 ≥50% 点。"""
        full = [self._grid_index(d) for d, data in self.wins.items()
                if data.get('teacher') and data['teacher'][1]
                and data['teacher'][0] == data['teacher'][1]]
        passing = [self._grid_index(d) for d in self.wins if self._passing(d)]
        if full:
            return max(full)
        return max(passing) if passing else None

    def _plan_fill(self) -> None:
        l50, top = self._l50_index(), self._top_index()
        if l50 is None or top is None:
            return
        self.l50, self.top = self.grid[l50], self.grid[top]
        for index in range(l50 + 1, top):
            if self.grid[index] not in self.wins and self.grid[index] not in self.next_queue:
                self.next_queue.append(self.grid[index])

    def observe(self, decision: int, arm: str, wins: int, games: int) -> None:
        self.wins.setdefault(int(decision), {})[arm] = (int(wins), int(games))
        if arm == 'teacher' and self.stage == 'bisect' and not self.next_queue:
            # 二分队列跑完一批后重算：可能还有未探明的跨线区间。
            self._plan_bisect()

    def finish(self) -> dict:
        """学生臂补两端点后返回汇总（可重复调用，结果一致）。"""
        l50_index, top_index = self._l50_index(), self._top_index()
        self.l50 = self.grid[l50_index] if l50_index is not None else None
        self.top = self.grid[top_index] if top_index is not None else None
        self.student_points = [d for d in (self.l50, self.top) if d is not None]
        points = {str(d): dict(data) for d, data in sorted(self.wins.items())}
        return dict(l50=self.l50, top=self.top, student_points=self.student_points,
                    points=points, ladder=build_ladder({d: data['teacher'] for d, data in self.wins.items()
                                                        if data.get('teacher')}))


def build_ladder(points: dict) -> dict | None:
    """points: decision -> (wins, games)（老师臂）。返回阶梯定义；全程 <50% 返回 None。

    规则（task71b / 任务书第 3 步）：档位全部 ≥50%；顶档取最晚的稳定 8/8 点
    （没有 8/8 就取最晚 ≥50% 点，报告里注明）；顶档以上即使有 ≥50% 点也不入
    （不如 8/8 稳定）。
    """
    tested = {int(d): (int(w), int(g)) for d, (w, g) in points.items() if int(g) > 0}
    passing = sorted(d for d, (w, g) in tested.items() if w / g >= PASS_RATE)
    if not passing:
        return None
    full = [d for d, (w, g) in tested.items() if w == g]
    top = max(full) if full else max(passing)
    ladder = sorted((d for d in passing if d <= top), reverse=True)
    return dict(ladder=ladder, top=top, l50=min(passing),
                top_is_full=bool(full), full_line=max(full) if full else None)


def cmd_adaptive(args) -> None:
    plan = load_jobs(args.jobs)
    entries = [entry for entry in plan['archives'] if entry_filter(entry, args)]
    if args.max_archives:
        entries = entries[:args.max_archives]
    for entry in entries:
        futures = list(entry['future_seeds'])[:args.futures]
        scanner = AdaptiveScan(entry['decisions'], equidistant=args.equidistant)
        jobs = job_map(entry, set(entry['decisions']), futures)
        print(f"{entry['task_id']} s{entry['seed']}（{len(entry['decisions'])} 点网格）", flush=True)
        while True:
            decision = scanner.next_teacher_point()
            if decision is None:
                break
            summary = run_point(jobs, decision, 'teacher', ('teacher',), args)
            scanner.observe(decision, 'teacher', summary['wins'], summary['games'])
        state = scanner.finish()
        for decision in state['student_points']:
            summary = run_point(jobs, decision, 'student', ('student',), args)
            scanner.observe(decision, 'student', summary['wins'], summary['games'])
        state = scanner.finish()
        document = dict(schema=RESULT_SCHEMA, task_id=entry['task_id'], seed=entry['seed'],
                        archive=entry['archive'], grade=entry.get('grade'), layer=entry.get('layer'),
                        adaptive=True, **state)
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / 'results' / f"{entry['task_id']}_s{entry['seed']}.json"
        path.write_text(json.dumps(document, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        ladder = state['ladder']
        verdict = (f"L50={state['l50']} 顶档={state['top']} 阶梯 {len(ladder['ladder'])} 档"
                   if ladder else '老师全程 <50%，不登记')
        print(f"  => {verdict}", flush=True)


# ---------------------------------------------------------------- register

class RegisterLedger:
    """登记用未来种子块的加锁账本（schema 与 task89 扫描账本同格式，独立文件）。

    占用范围 = 清单/盘点既有段 + task89 扫描账本已分配块 + 本账本已登记块，
    保证与全部段不相交。
    """

    def __init__(self, path: Path, scan_ledger: Path | None):
        self.path = Path(path)
        self.scan_ledger = Path(scan_ledger) if scan_ledger else None
        self.lock = self.path.with_suffix('.lock')

    def _ranges(self) -> list:
        ranges = list(occupied_seed_ranges(ROOT))
        for ledger in (self.scan_ledger, self.path):
            if ledger and ledger.exists():
                document = json.loads(ledger.read_text(encoding='utf-8'))
                for entry in (document.get('reservations') or {}).values():
                    ranges.append((entry['start'], entry['start'] + entry['count'] - 1))
        return ranges

    def allocate(self, base: int, count: int = REGISTER_BLOCK) -> int:
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        with self.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            document = (json.loads(self.path.read_text(encoding='utf-8'))
                        if self.path.exists() else dict(schema='llm_scan_seed_ledger_v1', reservations={}))
            if document.get('schema') != 'llm_scan_seed_ledger_v1':
                raise ValueError('未知登记种子账本格式')
            start = allocate_future_block(self._ranges(), base, count)
            document['reservations'][f'register_{start}'] = dict(start=start, count=count)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,                                             delete=False) as temporary:
                temporary.write(json.dumps(document, ensure_ascii=False, indent=2) + '\n')
            Path(temporary.name).replace(self.path)
            return start


def load_results(paths: list) -> list:
    documents = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            paths = sorted(path.glob('*.json'))
            documents.extend(load_results([str(p) for p in paths]))
            continue
        document = json.loads(path.read_text(encoding='utf-8'))
        if document.get('schema') != RESULT_SCHEMA:
            raise ValueError(f'{path}: 不是 {RESULT_SCHEMA} 结果文件')
        documents.append(document)
    return documents


def update_manifest(manifest_path: Path, task_id: str, spec: dict) -> bool:
    document = json.loads(manifest_path.read_text(encoding='utf-8'))
    for task in document.get('tasks') or []:
        if task['task_id'] == task_id:
            existing = task.get('demo_start')
            if existing and existing.get('decisions') == spec['decisions']:
                return False    # 幂等
            task['demo_start'] = spec
            manifest_path.write_text(json.dumps(document, ensure_ascii=False, indent=1) + '\n',
                                     encoding='utf-8')
            return True
    raise KeyError(f'{task_id} 不在 {manifest_path}')


def update_test_expectations(test_path: Path, task_ids: list) -> bool:
    """把新起点格 task_id 加进 test_registered_manifest_start_cells 的期望集合。"""
    text = test_path.read_text(encoding='utf-8')
    missing = [task_id for task_id in task_ids if f"'{task_id}'" not in text]
    if not missing:
        return False
    marker = "self.assertEqual(set(cells), {"
    start = text.index(marker) + len(marker)
    end = text.index('})', start)
    lines = ''.join(f"                                      '{task_id}',\n" for task_id in missing)
    text = text[:start] + '\n' + lines + ' ' * 38 + text[start:end].lstrip('\n') + text[end:]
    test_path.write_text(text, encoding='utf-8')
    return True


def update_design(design_path: Path, lines: list) -> bool:
    text = design_path.read_text(encoding='utf-8')
    block = '\n'.join(lines)
    if block in text:
        return False
    stamp = dt.date.today().isoformat()
    text = text.rstrip('\n') + f"\n\n### 14.35 接手扫描运行器登记（task91，{stamp}）\n\n{block}\n"
    design_path.write_text(text, encoding='utf-8')
    return True


def cmd_register(args) -> None:
    results = load_results(args.results)
    ledger = RegisterLedger(args.ledger, args.scan_ledger)
    manifest_path = args.manifest
    registered, skipped, report = [], [], []
    for document in results:
        task_id = document['task_id']
        points = {int(d): data.get('teacher') for d, data in (document.get('points') or {}).items()
                  if data and data.get('teacher')}
        ladder = build_ladder(points)
        if not ladder:
            report.append(dict(task_id=task_id, seed=document['seed'],
                               registered=False, reason='老师全程 <50%'))
            skipped.append(task_id)
            print(f'{task_id} s{document["seed"]}: 老师 {len(points)} 点全程 <50%，不登记')
            continue
        spec = dict(archive=document['archive'],
                    decisions=ladder['ladder'], index=0,
                    future_seed_start=None, future_seed_count=REGISTER_BLOCK)
        if args.dry_run:
            start = DEFAULT_REGISTER_BASE
            print(f'[dry run] {task_id}: 阶梯 {ladder["ladder"]}（顶档 {ladder["top"]}'
                  f'{"，8/8" if ladder["top_is_full"] else "，非 8/8"}），种子块将从 {DEFAULT_REGISTER_BASE} 起')
        else:
            start = ledger.allocate(args.register_base)
            spec['future_seed_start'] = start
            changed = update_manifest(manifest_path, task_id, spec)
            print(f'{task_id}: 阶梯 {ladder["ladder"]}（顶档 {ladder["top"]}），种子块 {start}–{start + REGISTER_BLOCK - 1}'
                  f'{"，已写入清单" if changed else "，清单已是该阶梯"}')
            registered.append((task_id, spec, ladder))
        report.append(dict(task_id=task_id, seed=document['seed'], registered=bool(not args.dry_run),
                           ladder=ladder['ladder'], top=ladder['top'], top_is_full=ladder['top_is_full'],
                           l50=ladder['l50'], future_seed_start=start))
    if args.dry_run:
        return
    if registered:
        update_test_expectations(args.tests, [task_id for task_id, _, _ in registered])
        design_lines = [f"- 接手扫描运行器（task91）登记 {len(registered)} 个 `demo_start`："]
        design_lines += [f"  - `{task_id}` 阶梯 {ladder['ladder']}，未来种子块 {spec['future_seed_start']}–"
                         f"{spec['future_seed_start'] + REGISTER_BLOCK - 1}"
                         for task_id, spec, ladder in registered]
        update_design(args.design, design_lines)
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / 'register_report.json'
    report_path.write_text(json.dumps(dict(schema='takeover_scan_register_v1',
                                           registered=len(registered), skipped=skipped,
                                           entries=report), ensure_ascii=False, indent=1) + '\n',
                           encoding='utf-8')
    print(f'登记 {len(registered)}、跳过 {len(skipped)}；报告 {report_path}')


# ---------------------------------------------------------------- CLI

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    def common(sp):
        sp.add_argument('--out', type=Path, default=DEFAULT_OUT)
        sp.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
        sp.add_argument('--student', type=Path, default=DEFAULT_STUDENT)
        sp.add_argument('--value', type=Path, default=DEFAULT_VALUE)
        sp.add_argument('--static-value', type=Path, default=DEFAULT_STATIC)
        sp.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
        sp.add_argument('--resource-dir', type=Path, default=DEFAULT_RESOURCES)
        sp.add_argument('--python', type=Path, default=Path(sys.executable))
        sp.add_argument('--workers', type=int, default=2, help='同时游戏进程数（台式机内存紧，默认 2）')
        sp.add_argument('--futures', type=int, default=8, help='每个点用几个未来种子（测试可调小）')
        sp.add_argument('--archive', help='只跑 task_id / 种子包含该子串的存档')
        sp.add_argument('--max-archives', type=int)

    run = sub.add_parser('run', help='按清单跑老师/学生两臂')
    run.add_argument('jobs', type=Path)
    run.add_argument('--decisions', help='只跑这些决策号（逗号分隔；默认全部）')
    run.add_argument('--teacher-only', action='store_true')
    common(run)
    run.set_defaults(func=cmd_run)

    adaptive = sub.add_parser('adaptive', help='自适应取点（4 等距 → 二分 → 补档）')
    adaptive.add_argument('jobs', type=Path)
    adaptive.add_argument('--equidistant', type=int, default=4)
    common(adaptive)
    adaptive.set_defaults(func=cmd_adaptive)

    register = sub.add_parser('register', help='判读并登记 demo_start')
    register.add_argument('results', nargs='+', help='结果 json 或目录')
    register.add_argument('--ledger', type=Path, default=DEFAULT_LEDGER)
    register.add_argument('--scan-ledger', type=Path,
                          default=ROOT / 'artifacts/llm_demo/raw/scan_seed_ledger.json')
    register.add_argument('--register-base', type=int, default=DEFAULT_REGISTER_BASE)
    register.add_argument('--design', type=Path, default=ROOT / 'DESIGN.md')
    register.add_argument('--tests', type=Path, default=ROOT / 'python/test_curriculum_queue.py')
    register.add_argument('--dry-run', action='store_true')
    register.set_defaults(func=cmd_register)
    return p


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
