#!/usr/bin/env python3
"""task91：接手扫描运行器 —— 把 LLM 通关局一条命令接到课程里。

三个子命令（对应 MEMORY/docs/task91-takeover-scan-runner.md）：

  run <scan_jobs.json>      按 task89 的作业清单跑老师/学生两臂，断点续跑，
                            结果按存档写 json（每局胜负、末波、破防行与僵尸类型）。
  adaptive <scan_jobs.json> 自适应取点：4 等距老师点 → 跨50%二分、失败区间补探 →
                            50%线附近补阶梯；最多9个老师点，学生臂只打两端。
  register <result.json…>   判读：老师 ≥50% 的存档生成阶梯（档位全部 ≥50%，
                            顶档取最晚的稳定 8/8 点），写 demo_start；未来种子块
                            与 task89 共用加锁账本分配；同步 DESIGN 与测试期望集合。

生产老师参数（curriculum_queue 同款）：学生 r11、叶值 v5、静态网 branch1、
--verify-samples 9 --student-candidates 2；老师臂 dagger-beta 1，学生臂 0。
"""

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from import_llm_run import allocate_future_block, occupied_seed_ranges, DEFAULT_SEED_LEDGER  # noqa: E402

# 生产模型（task91 任务书：学生 r11、叶值 v5、静态网 branch1）。
DEFAULT_STUDENT = Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/round_000011/student/model.pt')
DEFAULT_VALUE = Path('/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt')
DEFAULT_STATIC = Path('/home/newbiexvwu/PvZ-search/artifacts/hard/value_branch1/model.pt')
DEFAULT_RESOURCES = Path.home() / '.cache/pvz-research-resources'
DEFAULT_MANIFEST = ROOT / 'configs/human_demo_tasks.json'
DEFAULT_OUT = ROOT / 'artifacts/llm_demo/scan_out'
DEFAULT_LEDGER = DEFAULT_SEED_LEDGER
LEGACY_REGISTER_LEDGER = ROOT / 'artifacts/llm_demo/register_seed_ledger.json'
DEFAULT_REGISTER_BASE = 86_000_000   # import 的扫描账本从 84M 起每份 +1000，登记块挪到 86M 起
REGISTER_BLOCK = 256                 # task71b 先例：每个 demo_start 格 256 个未来种子
PASS_RATE = 0.5
# search_play 的臂名与 dagger-beta：teacher = 搜索老师（β=1），student = 学生独自走（β=0）。
ARM_SPEC = {'teacher': ('fast_search', 1.0), 'student': ('fast', 0.0)}
RESULT_SCHEMA = 'takeover_scan_results_v1'


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as tmp:
        tmp.write(json.dumps(document, ensure_ascii=False, indent=1) + '\n')
    Path(tmp.name).replace(path)


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
    rows = read_rows(out)
    validate_job_context(job, args, out, bool(rows))
    done = {row.get('future_seed') for row in rows if row.get('arm') == ARM_SPEC[job['arm']][0]}
    missing = [seed for seed in job['future_seeds'] if seed not in done]
    if missing:
        env = dict(os.environ, PVZ_NO_POPUP='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        proc = subprocess.run(search_play_command(job, missing, out, args), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            tail = '\n'.join(proc.stdout.splitlines()[-15:])
            raise RuntimeError(f"search_play 失败 ({job['task_id']} d{job['decision']} {job['arm']}):\n{tail}")
    wanted = set(job['future_seeds'])
    selected = [r for r in read_rows(out) if r.get('future_seed') in wanted
                and r.get('arm') == ARM_SPEC[job['arm']][0] and r.get('seed') == job['seed']]
    if len(selected) != len(wanted) or {r['future_seed'] for r in selected} != wanted:
        raise ValueError('扫描结果缺少未来种子或有重复局')
    return sorted(selected, key=lambda row: job['future_seeds'].index(row['future_seed']))


_DIGESTS = {}


def file_digest(path: Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _DIGESTS:
        _DIGESTS[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _DIGESTS[key]


def validate_job_context(job: dict, args, out: Path, has_rows: bool) -> None:
    """固定存档与搜索输入，未来种子可扩展；不把别的存档/模型结果当作断点。"""
    archive = resolve_archive(job['archive']).resolve()
    task = next(t for t in json.loads(args.manifest.read_text())['tasks'] if t['task_id'] == job['task_id'])
    context = dict(archive=str(archive), task_id=job['task_id'], seed=job['seed'],
                   decision=job['decision'], arm=job['arm'],
                   task={k: v for k, v in task.items() if k != 'demo_start'},
                   resource_dir=str(args.resource_dir.resolve()),
                   inputs={str(Path(p).resolve()): file_digest(p) for p in
                           (args.student, args.value, args.static_value, args.executable,
                            archive / 'meta.json', archive / 'ticks.jsonl',
                            *[archive / n for n in ('frames.jsonl', 'frames.npz') if (archive / n).exists()])})
    path = out / 'takeover_context.json'
    if path.exists():
        if json.loads(path.read_text()) != context:
            raise ValueError(f'{out}: 存档或搜索输入已变化，请另用输出目录')
        return
    if has_rows:
        # Existing CodeBuddy runs predate fingerprints; validate their recorded config
        # before freezing the current inputs. Their historical weight hashes are unavailable.
        config = json.loads((out / 'config.json').read_text())
        paths = dict(rollout_policy=args.student, value=args.value, static_value=args.static_value,
                     executable=args.executable, demo_archive=archive, resource_dir=args.resource_dir)
        if any(Path(config.get(k, '')).resolve() != Path(v).resolve() for k, v in paths.items()):
            raise ValueError(f'{out}: 旧扫描配置与当前存档/模型不符')
        expected = dict(task_id=job['task_id'], demo_decision=str(job['decision']),
                        arms=ARM_SPEC[job['arm']][0], verify_samples='9', student_candidates='2',
                        samples='3', horizon='2500', per_card='1', halving='True')
        if any(config.get(k) != v for k, v in expected.items()) or float(config['dagger_beta']) != ARM_SPEC[job['arm']][1]:
            raise ValueError(f'{out}: 旧扫描搜索参数不符')
    write_json(path, context)


def point_summary(rows: list) -> dict:
    return dict(wins=sum(1 for row in rows if row.get('won')), games=len(rows), rows=[trim_row(r) for r in rows])


def cached_points_for_futures(points: dict, futures: list) -> dict:
    """汇总只保留本次请求的完整未来集合；行数相同不代表同一批未来。"""
    result = {}
    expected = set(futures)
    for decision, bucket in points.items():
        for arm, data in bucket.items():
            if not isinstance(data, dict):
                continue
            rows = [row for row in data.get('rows', []) if row.get('future_seed') in expected]
            if len(rows) == len(expected) and {r['future_seed'] for r in rows} == expected:
                result.setdefault(decision, {})[arm] = point_summary(rows)
    return result


def entry_filter(entry: dict, args) -> bool:
    if getattr(args, 'task_id', None) and entry['task_id'] != args.task_id:
        return False
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


def write_archive_result(entry: dict, points: dict, out: Path, extra: dict | None = None) -> Path:
    document = dict(schema=RESULT_SCHEMA, task_id=entry['task_id'], seed=entry['seed'],
                    archive=entry['archive'], grade=entry.get('grade'), layer=entry.get('layer'),
                    **(extra or {}), points=points)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{entry['task_id']}_s{entry['seed']}.json"
    write_json(path, document)
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
        wanted = [int(d) for d in args.decisions.split(',')] if args.decisions else all_points
        points = [d for d in wanted if d in all_points]   # 各存档网格不同，取交集
        if not points:
            continue
        jobs = job_map(entry, points, futures)
        points_path = args.out / 'results'
        store = {}
        extra = {}
        existing = points_path / f"{entry['task_id']}_s{entry['seed']}.json"
        if existing.exists():
            previous = json.loads(existing.read_text(encoding='utf-8'))
            store = cached_points_for_futures(previous.get('points', {}), futures)
            # 保留 adaptive 等来源写入的汇总字段（l50/顶档/阶梯），只更新 points 与基础字段。
            extra = {k: v for k, v in previous.items()
                     if k not in ('schema', 'points', 'task_id', 'seed', 'archive',
                                  'grade', 'layer')}
            prior_futures = previous.get('future_seeds')
            if prior_futures is None:
                sets = {tuple(sorted(row['future_seed'] for row in arm.get('rows', [])))
                        for bucket in previous.get('points', {}).values() for arm in bucket.values()
                        if isinstance(arm, dict) and arm.get('rows')}
                prior_futures = list(next(iter(sets))) if len(sets) == 1 else None
            if prior_futures != futures or previous.get('complete') is False:
                extra = {}  # 派生的 adaptive L50/阶梯属于另一组未来，不可保留。
        extra['future_seeds'] = futures
        extra['complete'] = False
        write_archive_result(entry, store, points_path, extra)
        for decision in points:
            bucket = store.setdefault(str(decision), {})
            for arm in arms:
                summary = run_point(jobs, decision, arm, arms, args)
                if summary is not None:
                    bucket[arm] = summary
                    write_archive_result(entry, store, points_path, extra)
        extra['complete'] = True
        write_archive_result(entry, store, points_path, extra)
    print(f'完成 {len(entries)} 份存档，结果在 {args.out / "results"}')


# ---------------------------------------------------------------- 自适应取点

class AdaptiveScan:
    """自适应取点的纯决策逻辑（可用合成胜率单测）。

    阶段：4 等距点 → 跨50%二分 → 失败区间补探 → 50%线附近补阶梯。
    所有实测点都落在 task89 的 12 点网格上，与 12 点结果可直接比档。
    失败区间补探用于发现非单调隆起；不假设所有未测点失败。
    最多9个老师点+2个学生点，8未来时至多88局，低于全量192局的一半。
    """

    def __init__(self, grid: list, equidistant: int = 4):
        self.grid = [int(d) for d in grid]
        self.wins: dict[int, dict] = {}       # decision -> {'teacher': {'wins','games','rows'}, …}
        self.equidistant = self._pick_equidistant(len(self.grid), equidistant)
        self.stage = 'equidistant'
        self.next_queue: list = []
        self.l50: int | None = None
        self.top: int | None = None
        self.student_points: list = []
        self.teacher_budget = min(9, len(self.grid))

    @staticmethod
    def _pick_equidistant(count: int, points: int) -> list:
        if count <= points:
            return list(range(count))
        return sorted({round(index * (count - 1) / (points - 1)) for index in range(points)})

    def rate(self, decision: int) -> float | None:
        data = self.wins.get(decision, {}).get('teacher')
        if not data or not data['games']:
            return None
        return data['wins'] / data['games']

    def next_teacher_point(self) -> int | None:
        """返回下一个要跑老师臂的决策号；None 表示老师臂结束。"""
        while True:
            if sum('teacher' in data for data in self.wins.values()) >= self.teacher_budget:
                self.stage = 'done'
                return None
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
                self.stage = 'guard'
                self._plan_guard()
                continue
            if self.stage == 'guard':
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

    def _plan_guard(self) -> None:
        """在已知50%线之前的失败区间补一个中点，防止漏掉非单调隆起。"""
        tested = sorted((self._grid_index(d), d) for d in self.wins if 'teacher' in self.wins[d])
        limit = self._l50_index()
        gaps = [(ia, ib) for (ia, da), (ib, db) in zip(tested, tested[1:])
                if ib - ia > 2 and not self._passing(da) and not self._passing(db)
                and (limit is None or ib < limit)]
        if gaps:
            ia, ib = max(gaps, key=lambda gap: gap[1])
            self.next_queue.append(self.grid[round((ia + ib) / 2)])

    def _top_index(self) -> int | None:
        """顶档：最晚的 8/8 稳定点；没有 8/8 就取最晚的 ≥50% 点。"""
        full = [self._grid_index(d) for d, data in self.wins.items()
                if data.get('teacher') and data['teacher']['games']
                and data['teacher']['wins'] == data['teacher']['games']]
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

    def observe(self, decision: int, arm: str, summary: dict) -> None:
        self.wins.setdefault(int(decision), {})[arm] = {
            'wins': int(summary['wins']), 'games': int(summary['games']),
            'rows': summary.get('rows', [])}
        if arm == 'teacher' and self.stage in ('bisect', 'guard') and not self.next_queue:
            # 二分队列跑完一批后重算：可能还有未探明的跨线区间。
            self.stage = 'bisect'
            self._plan_bisect()

    def finish(self) -> dict:
        """学生臂补两端点后返回汇总（可重复调用，结果一致）。"""
        l50_index, top_index = self._l50_index(), self._top_index()
        self.l50 = self.grid[l50_index] if l50_index is not None else None
        self.top = self.grid[top_index] if top_index is not None else None
        self.student_points = list(dict.fromkeys(d for d in (self.l50, self.top) if d is not None))
        points = {str(d): dict(data) for d, data in sorted(self.wins.items())}
        return dict(l50=self.l50, top=self.top, student_points=self.student_points,
                    teacher_budget=self.teacher_budget,
                    teacher_points=sum('teacher' in data for data in self.wins.values()),
                    scan_games=sum(arm['games'] for data in self.wins.values() for arm in data.values()),
                    points=points,
                    ladder=build_ladder({d: (data['teacher']['wins'], data['teacher']['games'])
                                         for d, data in self.wins.items() if data.get('teacher')}))


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
        path = args.out / 'results' / f"{entry['task_id']}_s{entry['seed']}.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        cached = cached_points_for_futures(previous.get('points', {}), futures)
        def checkpoint(complete):
            state = scanner.finish()
            merged = {d: dict(bucket) for d, bucket in cached.items()}
            for decision, bucket in state['points'].items():
                merged.setdefault(decision, {}).update(bucket)
            trace = dict(algorithm=f'coarse{args.equidistant}_guard_budget9_v1', complete=complete,
                         grid=entry['decisions'], future_seeds=futures,
                         teacher_decisions=[d for d, bucket in scanner.wins.items() if 'teacher' in bucket],
                         l50=state['l50'], top=state['top'], points=state['points'],
                         teacher_points=state['teacher_points'], scan_games=state['scan_games'])
            document = dict(schema=RESULT_SCHEMA, task_id=entry['task_id'], seed=entry['seed'],
                            archive=entry['archive'], grade=entry.get('grade'), layer=entry.get('layer'),
                            adaptive=True, complete=complete, future_seeds=futures,
                            adaptive_trace=trace, **{**state, 'points': merged})
            write_json(path, document)
            return state
        print(f"{entry['task_id']} s{entry['seed']}（{len(entry['decisions'])} 点网格）", flush=True)
        while True:
            decision = scanner.next_teacher_point()
            if decision is None:
                break
            summary = run_point(jobs, decision, 'teacher', ('teacher',), args)
            scanner.observe(decision, 'teacher', summary)
            checkpoint(False)
        state = scanner.finish()
        for decision in state['student_points']:
            summary = run_point(jobs, decision, 'student', ('student',), args)
            scanner.observe(decision, 'student', summary)
            checkpoint(False)
        state = checkpoint(True)
        ladder = state['ladder']
        verdict = (f"L50={state['l50']} 顶档={state['top']} 阶梯 {len(ladder['ladder'])} 档"
                   if ladder else '已测老师点均 <50%，暂不登记')
        print(f"  => {verdict}", flush=True)


# ---------------------------------------------------------------- register

class RegisterLedger:
    """登记与 task89 扫描共用未来种子加锁账本，兼容读取旧独立账本。

    占用范围 = 清单/盘点既有段 + task89 扫描账本已分配块 + 本账本已登记块，
    保证与全部段不相交。
    """

    def __init__(self, path: Path, scan_ledger: Path | None):
        self.path = Path(path)
        self.scan_ledger = Path(scan_ledger) if scan_ledger else None
        self.lock = self.path.with_suffix('.lock')

    def _ranges(self) -> list:
        ranges = list(occupied_seed_ranges(ROOT))
        for ledger in {self.scan_ledger, self.path, LEGACY_REGISTER_LEDGER}:
            if ledger and ledger.exists():
                document = json.loads(ledger.read_text(encoding='utf-8'))
                for entry in (document.get('reservations') or {}).values():
                    ranges.append((entry['start'], entry['start'] + entry['count'] - 1))
        return ranges

    def allocate(self, base: int, count: int = REGISTER_BLOCK, key: str | None = None) -> int:
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        with self.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            document = (json.loads(self.path.read_text(encoding='utf-8'))
                        if self.path.exists() else dict(schema='llm_scan_seed_ledger_v1', reservations={}))
            if document.get('schema') != 'llm_scan_seed_ledger_v1':
                raise ValueError('未知登记种子账本格式')
            existing = document['reservations'].get(key) if key else None
            if existing:
                if existing['count'] != count:
                    raise ValueError('既有登记分配的块大小不符')
                return existing['start']
            start = allocate_future_block(self._ranges(), base, count)
            document['reservations'][key or f'register_{start}'] = dict(start=start, count=count)
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
        validate_result(document)
        documents.append(document)
    return documents


def validate_summary(data: dict, futures: list) -> None:
    rows = data.get('rows') or []
    if (data.get('games') != len(rows) or len(rows) != len(futures)
            or {row.get('future_seed') for row in rows} != set(futures)
            or any(type(row.get('won')) is not bool for row in rows)
            or data.get('wins') != sum(row['won'] for row in rows)):
        raise ValueError('扫描汇总与逐局未来/胜负证据不一致')


def validate_result(document: dict) -> list:
    if document.get('complete') is False:
        raise ValueError('自适应扫描尚未完成，不可登记')
    futures = document.get('future_seeds')
    for bucket in document.get('points', {}).values():
        for data in bucket.values():
            if not isinstance(data, dict):
                raise ValueError('旧扫描汇总缺逐局证据，请从原始对局缓存重建')
            if futures is None:
                futures = [row.get('future_seed') for row in data.get('rows', [])]
            if not futures or len(futures) != len(set(futures)):
                raise ValueError('扫描未来集合为空或重复')
            validate_summary(data, futures)
    return futures or []


def update_manifest(manifest_path: Path, task_id: str, spec: dict) -> bool:
    document = json.loads(manifest_path.read_text(encoding='utf-8'))
    for task in document.get('tasks') or []:
        if task['task_id'] == task_id:
            existing = task.get('demo_start')
            if existing:
                if existing == spec:
                    return False
                raise ValueError(f'{task_id}: 已有起点格，不得覆盖')
            task['demo_start'] = spec
            manifest_path.write_text(json.dumps(document, ensure_ascii=False, indent=1) + '\n',
                                     encoding='utf-8')
            return True
    raise KeyError(f'{task_id} 不在 {manifest_path}')


def update_test_expectations(test_path: Path, task_ids: list) -> bool:
    """把新起点格 task_id 加进 test_registered_manifest_start_cells 的期望集合。"""
    text = test_path.read_text(encoding='utf-8')
    marker = "self.assertEqual(set(cells), {"
    start = text.index(marker) + len(marker)
    end = text.index('})', start)
    missing = [task_id for task_id in task_ids if f"'{task_id}'" not in text[start:end]]
    if not missing:
        return False
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
    numbers = [int(match.group(1)) for match in re.finditer(r'^### 14\.(\d+)', text, re.M)]
    section = max(numbers, default=34) + 1
    text = text.rstrip('\n') + f"\n\n### 14.{section} 接手扫描运行器登记（task91，{stamp}）\n\n{block}\n"
    design_path.write_text(text, encoding='utf-8')
    return True


def resolve_target_task(document: dict, manifest_path: Path) -> tuple[str, dict | None]:
    """结果该登记到哪个 task_id；返回 (族名, 既有 demo_start 或 None)。

    task87 先例：本机种子的 LLM 局与人类示范共用 base task_id，登记为 ``_llm`` 后缀族。
    按 archive 匹配已有 demo_start 最可靠；base 名已被人类示范占用时用 ``_llm``。
    """
    document = dict(document)
    tasks = json.loads(manifest_path.read_text(encoding='utf-8')).get('tasks') or []
    for task in tasks:
        if (task.get('demo_start') or {}).get('archive') == document['archive']:
            return task['task_id'], task['demo_start']
    base = document['task_id']
    occupied = None
    for candidate in (base, f'{base}_llm'):
        for task in tasks:
            if task['task_id'] != candidate:
                continue
            if not task.get('demo_start'):
                return candidate, None
            occupied = (candidate, task['demo_start']['archive'])
            break
    if occupied:
        raise ValueError(
            f'{base}: 候选族 {occupied[0]} 已被存档 {occupied[1]} 占用，需人工指定登记族')
    raise KeyError(f'{base}: manifest 里没有可登记的族（试过 {base} / {base}_llm）')


def cmd_register(args) -> None:
    results = load_results(args.results)
    ledger = RegisterLedger(args.ledger, args.scan_ledger)
    manifest_path = args.manifest
    registered, skipped, report = [], [], []
    dry_ranges = ledger._ranges() if args.dry_run else []
    for document in results:
        base_id = document['task_id']
        try:
            task_id, existing = resolve_target_task(document, manifest_path)
        except (KeyError, ValueError) as exc:
            report.append(dict(task_id=base_id, seed=document['seed'], registered=False,
                               reason=str(exc)))
            skipped.append(base_id)
            print(f'{base_id} s{document["seed"]}: 跳过 —— {exc}')
            continue
        points = {int(d): (data['teacher']['wins'], data['teacher']['games'])
                  for d, data in (document.get('points') or {}).items()
                  if data and data.get('teacher')}
        ladder = build_ladder(points)
        if not ladder:
            report.append(dict(task_id=task_id, seed=document['seed'],
                               registered=False, reason='已测老师点均 <50%'))
            skipped.append(task_id)
            print(f'{task_id} s{document["seed"]}: 已测老师 {len(points)} 点均 <50%，不登记')
            continue
        if existing:
            # 既有格不覆盖：已登记的阶梯来自全量扫描（task87 等），这里只核对与报告。
            same = existing.get('decisions') == ladder['ladder']
            print(f'{task_id}: 已有起点格（{len(existing.get("decisions") or [])} 档），'
                  f'自适应阶梯 {ladder["ladder"]} —— {"一致" if same else "不一致（保留既有，见报告）"}')
            report.append(dict(task_id=task_id, seed=document['seed'], registered=False,
                               reason='已有起点格，保留既有', existing_decisions=existing.get('decisions'),
                               adaptive_ladder=ladder['ladder'], l50=ladder['l50']))
            continue
        spec = dict(archive=document['archive'],
                    decisions=ladder['ladder'], index=0,
                    future_seed_start=None, future_seed_count=REGISTER_BLOCK)
        if args.dry_run:
            start = allocate_future_block(dry_ranges, args.register_base, REGISTER_BLOCK)
            dry_ranges.append((start, start + REGISTER_BLOCK - 1))
            print(f'[dry run] {task_id}: 阶梯 {ladder["ladder"]}（顶档 {ladder["top"]}'
                  f'{"，全胜" if ladder["top_is_full"] else "，非全胜"}），种子块 {start} 起')
        else:
            key = json.dumps(['register', task_id, document['archive']], ensure_ascii=False)
            start = ledger.allocate(args.register_base, key=key)
            spec['future_seed_start'] = start
            update_manifest(manifest_path, task_id, spec)
            print(f'{task_id}: 阶梯 {ladder["ladder"]}（顶档 {ladder["top"]}），种子块 {start}–{start + REGISTER_BLOCK - 1}，已写入清单')
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
        sp.add_argument('--workers', type=int, choices=(1, 2), default=2, help='同时游戏进程数（台式机内存紧，默认 2）')
        sp.add_argument('--futures', type=int, choices=range(1, 9), default=8, help='每个点用几个未来种子（测试可调小）')
        sp.add_argument('--archive', help='只跑 task_id / 种子包含该子串的存档')
        sp.add_argument('--task-id', help='只跑这个 task_id（精确匹配）')
        sp.add_argument('--max-archives', type=int)

    run = sub.add_parser('run', help='按清单跑老师/学生两臂')
    run.add_argument('jobs', type=Path)
    run.add_argument('--decisions', help='只跑这些决策号（逗号分隔；默认全部）')
    run.add_argument('--teacher-only', action='store_true')
    common(run)
    run.set_defaults(func=cmd_run)

    adaptive = sub.add_parser('adaptive', help='自适应取点（4 等距 → 二分 → 补档）')
    adaptive.add_argument('jobs', type=Path)
    adaptive.add_argument('--equidistant', type=int, choices=(4, 5), default=4)
    common(adaptive)
    adaptive.set_defaults(func=cmd_adaptive)

    register = sub.add_parser('register', help='判读并登记 demo_start')
    register.add_argument('results', nargs='+', help='结果 json 或目录')
    common(register)
    register.add_argument('--ledger', type=Path, default=DEFAULT_LEDGER)
    register.add_argument('--scan-ledger', type=Path,
                          default=DEFAULT_SEED_LEDGER)
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
