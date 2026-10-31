#!/usr/bin/env python3
"""task29 第 2 步：自动课程采集控制器（取代手写的 queue101–103）。

分组 = 关卡 × 卡组族 × 倍率（1.0 档用原任务 id，其余用 `<base>_m125` … `<base>_m300`）。
每轮按 DESIGN §3.2 的元规则分配局数，只硬编码“优先学边缘任务”这一条：

1. **升级 / 降级**：某格当前倍率的滑动窗口胜率 > 0.85 且样本够 → 倍率升一档；
   < 0.5 → 降一档（不低于 1.0）。
   示范起点格（task71）的档位不是倍率，而是"起点决策号"：开局由（示范存档、第 N 个决策）
   恢复，未来随机流走 RESEED（--future-seeds，每组换一批种子）。阶梯写在清单任务的
   demo_start 字段里（降序，0 号最晚 = 老师已稳定会赢）；同样用 0.85 / 0.5 升降，
   升档 = 起点往前挪一档（更早、更难）。
2. **分配**（task57，2026-10-24 起）：
   a. **保底防饿死**：超过 FLOOR_ROUNDS 轮没被采过的格（含 last_round=0 从未采过的）
      按最久优先轮转，每格 per_group // FLOOR_SHARE 局；预算 = 每轮
      ceil(len(cells) / FLOOR_ROUNDS) 组。旧版 12% guard 预算（max(48, 92)=48 局）每轮
      只能救 1 个格，并列按字母序，13 个排在后面的格三轮 2304 局一次都没轮到。
   b. **其余预算按"可教差距"降序**：老师胜率（有证据时）− 学生胜率（当前倍率窗口，
      无数据按 0）。老师没有证据时按可教处理——beta0 采集本身就是老师探针；
      学生全败而老师会赢的格（如 L20：老师 29/32、学生 0/128）排最前，这正是
      该采的数据；老师也赢不了的格差距 ≈ 0 自然沉底（雾区格本就不在训练集合）。
      同分按最久没采的先。老师胜率证据读 `--teacher-rates`
      （默认 ~/pvz-logs/teacher_rates.json，缺文件按全部未知处理）。
3. **落盘**：状态 json（默认 `~/pvz-logs/curriculum_state.json`），每轮一行日志；
   偏移在**跑之前**写进状态，崩溃/中断不会重采同一批种子。
4. **停止标志** `~/pvz-logs/stop_curr`：只在两组之间检查，绝不 SIGTERM 子进程。
5. **后台打包 / 训练 / 晋升评估**：每轮保存“本队列写过的全部标签 + `--recipe` 来源标签”的快照，
   复用已有标签段、只打新标签，然后训练并做配对门禁。前台立即采下一轮，组间只切换通过者。
   打包并发按待建段的原始档案体积估算（build_plan），不按已建段——冷启动也能吃满内存预算。

采集参数照 `~/pvz-logs/queue103.sh` 的 `run()`（beta=0，学生自走、搜索打标签）；
task59 起学生胜率 < 阈值（默认 0.1）的格改用 β 混合（--starved-beta，默认 0.5）：
部分步改由搜索执行、每个决策仍由搜索打标签，把标签覆盖的状态范围从"开局即崩"扩到中后盘
（Ross et al. 2011 的 DAgger β-mixing；与 beta0 的行同格式，可一起训练）；
β 混合组的胜负不是学生胜率，不进学生窗口，另存该倍率桶的 mixed 计数；每
--probe-interval（默认 3）组保留一个 β=0 探针组测量真实学生胜率（修正 2b），
帧默认直接写 `~/PvZ-search/artifacts/student/frames_v2`（只增不改）。并发不再手动定：
每开一组前读 MemAvailable，`--workers` 只是上限，实际 worker 数按
`(MemAvailable − 6 GiB) / 一个 worker 的实测 RSS` 取整，至少 1。每 worker 的 RSS 在
控制器启动时用一个单 worker 的真实对局实测（Python + 引擎峰值；对局写临时目录，不占偏移）。
非 Linux（没有 /proc）时不做缩放，直接用 `--workers`。

用法：
  python3 scripts/curriculum_queue.py --dry-run                 # 第一轮分配表（用已有结果初始化）
  python3 scripts/curriculum_queue.py --round-games 4 --games-per-group 2 \
      --state artifacts/task29/smoke_state.json --out artifacts/task29/smoke_runs \
      --frames artifacts/task29/smoke_frames                     # 冒烟（每组 2 局）
  python3 scripts/curriculum_queue.py --rounds 8                # 正式跑（主控决定何时替换 q103）
"""
from __future__ import annotations

import argparse
import copy
import functools
import glob
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from make_multiplier_tasks import (CURRICULUM_LEVELS, FAMILIES, MULTIPLIERS, SUPERSEDED,
                                   load_tasks, source_paths)  # noqa: E402
from pvz_recipe import load_recipe, training_row_options, write_recipe
from pvz_promotion import (GATE_VERSION, SEED_LEDGER_DEFAULT, START_SEED_BLOCK,
                           START_SEED_LEDGER_DEFAULT, START_SEED_POOL,
                           policy as promotion_policy)
from promotion_gate import (SeedPoolExhausted, allocate_seed_block, contract as promotion_contract,
                            contract_key, valid_certificate)
from pvz_training_pack import (pack_flags, pack_options, fingerprint_paths, segment_options,
                               segment_path)
from pvz_metrics import Metrics  # noqa: E402
from pvz_common import OOM_SCORE_CONTROLLER, set_oom_score_adj  # noqa: E402

PROMOTE_ABOVE = 0.85
DEMOTE_BELOW = 0.5
MIN_SAMPLES = 8            # 少于这个局数不做升降级，只继续采
WINDOW = 256               # 滑动窗口：每组只记最近 256 局的结果（0/1）
FLOOR_ROUNDS = 4           # 保底：任何格至多隔这么久必须被采一次（防饿死）
FLOOR_SHARE = 2            # 保底组局数 = per_group // FLOOR_SHARE（小局数换轮转覆盖）
DEFAULT_STATE = Path.home() / 'pvz-logs/curriculum_state.json'
DEFAULT_TEACHER_RATES = Path.home() / 'pvz-logs/teacher_rates.json'
DEFAULT_LOG = Path.home() / 'pvz-logs/curriculum.log'
CURRICULUM_LADDERS = {
    'native_l25': [0.5, 1.0] + [m for m, _ in MULTIPLIERS],
}
# task137: v5 随机卡组格的阶梯写在清单里（每个任务带 curriculum_ladder），
# 起点 0.5 让老师先能赢，再按 PROMOTE_ABOVE / DEMOTE_BELOW 往上走。
RANDOM_DECK_SHARE = 0.15   # 随机卡组格每轮最多占 round-games 的 15%
REVIEW_SHARE = 0.25       # 非 1 倍任务在原配额之外追加的 1 倍复习
CLEAN_B_LINEAGE = 'task205-clean-b'

STOP_FLAG = Path.home() / 'pvz-logs/stop_curr'
DEFAULT_FRAMES = ROOT / 'artifacts/student/frames_v2'
DEFAULT_MODEL = ROOT / 'artifacts/student/dw4i_s1/model.pt'
DEFAULT_B_INIT = Path.home() / 'PvZ-search/artifacts/student/gptbc_s1/model.pt'
DEFAULT_STATIC_VALUE = ROOT / 'artifacts/hard/value_branch1/model.pt'
DEFAULT_VALUE = ROOT / 'artifacts/value_net/v5/model.pt'
DEFAULT_RESOURCES = Path.home() / '.cache/pvz-research-resources'
DEFAULT_PYTHON = Path(sys.executable)

GIB = 1024 ** 3
MAX_BACKGROUND_ATTEMPTS = 2   # task98: 同一快照最多尝试两次，失败保留证据，采集继续
MEMORY_RESERVE_BYTES = 3 * GIB          # 留给 WSL 接入通道、控制器与训练，不交给采集
PACK_BUILDER_BYTES = 2 * GIB           # 一个段建造者的 Python/torch 与解压缓冲（pack() 同一假设）
# 训练阶段的不可回收峰值 = 匿名内存（read_segments 的目标数组是文件后备 memmap，task61）。
# 实测（2026-10-24，round 2 全量 7.73 GiB 段）：读入 + 建批 RssAnon 峰值 1.59 GiB = 0.205×
# 紧凑语料；系数取 0.25 留余量，常数 2 GiB 覆盖 torch 模型/优化器/批次。
TRAIN_ANON_FACTOR = 0.25
TRAIN_ANON_EXTRA = 2 * GIB
TRAIN_GPU_BYTES = int(1.5 * GIB)  # measured reserved 1.2 GiB/seed, rounded up
GPU_RESERVE_BYTES = GIB
DEFAULT_WORKER_BYTES = GIB              # 实测失败时的保守值（一个 worker 的 Python + 引擎）
MIN_PLAUSIBLE_WORKER_BYTES = GIB // 4  # a real python + engine worker is ~0.4-0.9 GiB
PROBE_TIMEOUT_SECONDS = 900.0
RSS_POLL_SECONDS = 0.2
# task140: 轮边界重试必须有界。原来失败就 sleep(120) 后重来，永不放弃、也永不退出，
# 2026-10-28 一次 pytest 用打了补丁的 sleep 跑这条路径，几十秒写出 20 GB 的 curriculum.log。
ROUND_RETRY_LIMIT = 5                  # 连续失败这么多次就退出，交给看门狗重启
ROUND_RETRY_SECONDS = 120              # 首次退避，之后翻倍
ROUND_RETRY_MAX_SECONDS = 1800
# 这些错误重试多少次都不会自己好：配置/接口写错、字段缺失。等下去只会刷日志。
ROUND_FATAL_ERRORS = (AttributeError, TypeError, NameError, KeyError, AssertionError)
ZERO_GAME_GROUP_LIMIT = 3
GAME_RECOVERY_SECONDS = 180


class StateWriteError(OSError):
    """No new games may start until the seed reservation is durable."""


class GameRecoveryStopped(RuntimeError):
    """A system-failed snapshot stays resumable when the controller stops."""


def systemic_game_failure(error):
    return isinstance(error, GameRecoveryStopped) or any(marker in str(error) for marker in (
        'Inconsistency detected by ld.so', 'error while loading shared libraries',
        'PvZ-Portable exited unexpectedly with code'))


class GameHealth:
    def __init__(self, args):
        self.args = args
        self.paused = threading.Event()
        self.closed = threading.Event()
        self.probe_lock = threading.Lock()
        args.game_health = self

    def pause(self, reason):
        if not self.paused.is_set():
            self.paused.set()
            reason = ' '.join(str(reason).splitlines())
            log_line(self.args.log, f'systemic game failure: collection/background paused at group boundary; '
                                    f'one-game probe every {GAME_RECOVERY_SECONDS}s: {reason}')

    def probe(self):
        args = self.args
        with tempfile.TemporaryDirectory(prefix='pvz-recovery-', dir=args.state.parent) as temporary:
            folder = Path(temporary)
            task = read_json(args.promotion_panel)['tasks'][0]
            manifest = folder / 'manifest.json'
            manifest.write_text(json.dumps(dict(tasks=[dict(task, seeds=(task.get('seeds') or [0])[:1])])))
            command = command_for(args, manifest, task['task_id'], 0, 1,
                                  'recovery', workers=1, frames=folder / 'frames', out=folder)
            result = subprocess.run(command, cwd=ROOT, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, timeout=PROBE_TIMEOUT_SECONDS)
            if result.returncode:
                return False
            rows = (folder / 'recovery_1/games.jsonl').read_text().splitlines()
            return len(rows) == 1 and type(json.loads(rows[0]).get('won')) is bool

    def wait(self):
        with self.probe_lock:
            while self.paused.is_set():
                if self.closed.is_set() or getattr(self.args, 'stop_flag', STOP_FLAG).exists():
                    return False
                if self.closed.wait(GAME_RECOVERY_SECONDS):
                    return False
                if getattr(self.args, 'stop_flag', STOP_FLAG).exists():
                    return False
                try:
                    recovered = self.probe()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    log_line(self.args.log, f'one-game recovery probe failed: {error}')
                    continue
                if recovered:
                    self.paused.clear()
                    log_line(self.args.log, 'one-game recovery probe succeeded; collection/background resumed')
                else:
                    log_line(self.args.log, 'one-game recovery probe failed; system remains paused')
        return True


def retry_io(operation):
    """Retry an IO operation once; never rerun a completed game on a nonzero exit."""
    @functools.wraps(operation)
    def call(*args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except OSError as error:
            if isinstance(error, StateWriteError):
                raise
            print(f'WARN {operation.__name__}: IO retry once: {error}', file=sys.stderr, flush=True)
            return operation(*args, **kwargs)
    return call


def read_json(path):
    return json.loads(retry_io(path.read_text)())


def retire_cell(state, base, error):
    state.setdefault('retired_cells', {})[base] = state['cells'].pop(base)
    state.setdefault('cell_errors', {})[base] = str(error)
    return f'cell retired: {base}: {type(error).__name__}: {error}'


def isolate_cells(state, values):
    """Validate individual saved cells before shared sorting/adjustment/reconciliation."""
    changes = []
    for base, cell in list(state['cells'].items()):
        try:
            if not isinstance(cell, dict):
                raise ValueError('cell must be an object')
            ladder = cell.get('ladder', values)
            if not isinstance(ladder, list) or not ladder or any(
                    type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                    for value in ladder):
                raise ValueError('invalid ladder')
            index = cell['mult_index']
            if type(index) is not int or not 0 <= index < len(ladder):
                raise ValueError('invalid mult_index')
            if type(cell['last_round']) is not int or type(cell['level']) is not int:
                raise ValueError('invalid last_round/level')
            if not isinstance(cell['windows'], dict):
                raise ValueError('windows must be an object')
            for bucket in cell['windows'].values():
                recent = bucket.get('recent', [])
                if not isinstance(recent, list) or any(value not in (0, 1) for value in recent):
                    raise ValueError('invalid recent outcomes')
                for counter in ('games', 'wins'):
                    if type(bucket.get(counter, 0)) is not int or bucket.get(counter, 0) < 0:
                        raise ValueError(f'invalid {counter}')
            cell_level(cell, values)
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as error:
            changes.append(retire_cell(state, base, error))
    return changes


def mem_available_bytes() -> int | None:
    """Linux 的 MemAvailable（字节）；没有 /proc 的平台返回 None。"""
    try:
        text = Path('/proc/meminfo').read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return None


def mem_total_bytes() -> int | None:
    """Linux 的 MemTotal（字节）；没有 /proc 的平台返回 None。"""
    try:
        text = Path('/proc/meminfo').read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith('MemTotal:'):
            return int(line.split()[1]) * 1024
    return None


def gpu_free_bytes() -> int | None:
    """Read the single production GPU without importing torch or allocating CUDA memory."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True, timeout=5)
        return int(result.stdout.splitlines()[0].strip()) * (1 << 20)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def worker_count(available: int | None, per_worker: int, upper: int, background: int = 0) -> int:
    """clamp(⌊(MemAvailable − 6 GiB) / per_worker⌋, 1, upper)；未知内存或未实测时用上限。"""
    if available is None or per_worker <= 0:
        return upper
    return max(1, min(upper, (available - MEMORY_RESERVE_BYTES - background) // per_worker))


def build_plan(pending_bytes: list[int], recipe_workers: int,
               available: int | None, measured_per_builder: int = 0) -> tuple[int, int]:
    """返回 (打包并发, 单建造者峰值字节)。

    打包并行是"一个建造者建一个标签段"：单建造者峰值 = 打包基线 + 待建标签里
    最大的原始档案体积；基线优先用最近一次实测（pack_peak_anon_bytes / 建造者数，
    task61 追加：round 2 实测 0.37 GiB，比 PACK_BUILDER_BYTES 的 2 GiB 假设小得多），
    没有实测回退 PACK_BUILDER_BYTES。并发 = min(配方值, (可用内存 − 预留) / 单建造者峰值, 待建段数)。
    旧算法按已建段体积估（冷启动 = 0 → 2 GiB → 并发 1），20 核机器单核打包近 9 小时
    （2026-10-24 task59）。待建段为空时并发 1（pack 无活可干，数值无意义）。
    """
    per_builder = (measured_per_builder or PACK_BUILDER_BYTES) + max(pending_bytes, default=0)
    workers = len(pending_bytes) or 1
    if available is not None:
        workers = min(workers, max(1, (available - MEMORY_RESERVE_BYTES) // per_builder))
    return max(1, min(recipe_workers, workers)), per_builder


def _proc_table() -> dict[int, tuple[int, int]]:
    """pid -> (ppid, 常驻字节)，一次扫完 /proc；没有 /proc 时为空。"""
    try:
        page = os.sysconf('SC_PAGE_SIZE')
        entries = list(os.scandir('/proc'))
    except (OSError, ValueError):
        return {}
    table = {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            fields = Path(entry.path, 'stat').read_text().rsplit(')', 1)[1].split()
            table[int(entry.name)] = (int(fields[1]), int(fields[21]) * page)
        except (OSError, ValueError, IndexError):
            continue
    return table


def _descendant_rss(root: int, table: dict[int, tuple[int, int]]) -> int:
    """root 的所有后代（不含 root）的常驻字节之和。"""
    children: dict[int, list[int]] = {}
    for pid, (ppid, _rss) in table.items():
        children.setdefault(ppid, []).append(pid)
    total, stack = 0, list(children.get(root, ()))
    while stack:
        pid = stack.pop()
        total += table[pid][1]
        stack.extend(children.get(pid, ()))
    return total


def _anon_bytes(pid: int) -> int:
    """一个进程的 RssAnon（匿名常驻字节）；读不到为 0。"""
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('RssAnon:'):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _tree_anon_bytes(root: int) -> int:
    """root 及其所有后代的匿名常驻字节之和，不含文件映射页。"""
    table = _proc_table()
    children: dict[int, list[int]] = {}
    for pid, (ppid, _rss) in table.items():
        children.setdefault(ppid, []).append(pid)
    total, stack = _anon_bytes(root), list(children.get(root, ()))
    while stack:
        pid = stack.pop()
        total += _anon_bytes(pid)
        stack.extend(children.get(pid, ()))
    return total


def training_data_bytes(model: Path) -> int:
    metrics = model.with_name('metrics.json')
    if not metrics.is_file():
        return 0
    memory = json.loads(metrics.read_text()).get('data_memory', {})
    return sum(value for name, value in memory.items() if name.endswith('_bytes'))


def measure_worker_bytes(args, item: dict) -> int:
    """用一个单 worker 的真实对局测一个 worker 的 Python + 引擎峰值 RSS。

    对局写在临时目录，不动状态偏移，也不进帧档案；正式第一组会用同一个种子重跑。
    """
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pvz-oom-probe-', dir=args.state.parent) as tmp:
        tmp_path = Path(tmp)
        command = command_for(args, Path(item['manifest']), item['task_id'], item['offset'], 1,
                              'oomprobe', workers=1, frames=tmp_path / 'frames', out=tmp_path,
                              save_candidates=item.get('save_candidates', 0),
                              demo_archive=item.get('demo_archive'),
                              demo_decision=item.get('demo_decision'),
                              future_seeds=(item.get('future_seeds') or [])[:1],
                              task_definition=item.get('task_definition'),
                              seed_offset=item.get('seed_offset'))
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   text=True)
        peak = 0
        deadline = time.monotonic() + PROBE_TIMEOUT_SECONDS
        while process.poll() is None:
            peak = max(peak, _descendant_rss(process.pid, _proc_table()))
            if time.monotonic() > deadline:
                process.kill()
                process.communicate()
                raise RuntimeError(f'probe timed out after {PROBE_TIMEOUT_SECONDS:.0f}s')
            time.sleep(RSS_POLL_SECONDS)
        _stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f'probe exited {process.returncode}: {stderr.strip()[-500:]}')
        return peak


def startup_worker_estimate(args, plan: list[dict]) -> int:
    """控制器启动时实测一次；没有 /proc 返回 0（worker_count 会退回 --workers）。"""
    if mem_available_bytes() is None:
        return 0
    if not plan:
        return DEFAULT_WORKER_BYTES
    started = time.time()
    try:
        measured = measure_worker_bytes(args, plan[0])
    except (OSError, RuntimeError) as error:
        log_line(args.log, f'worker probe failed ({error}); assuming {DEFAULT_WORKER_BYTES / GIB:.1f} GiB/worker')
        return DEFAULT_WORKER_BYTES
    if measured < MIN_PLAUSIBLE_WORKER_BYTES:
        # 10-26: a demo start-cell probe ended in 2 s and measured 0.02 GiB; admission then ran 16 workers
        # (real ~0.43 GiB each) and the background trainer swapped at a tenth of its normal speed.
        log_line(args.log, f'worker probe measured {measured / GIB:.2f} GiB in {time.time() - started:.0f}s, '
                           f'implausibly small; assuming {DEFAULT_WORKER_BYTES / GIB:.1f} GiB/worker')
        return DEFAULT_WORKER_BYTES
    log_line(args.log, f'worker estimate: {max(measured, 1) / GIB:.2f} GiB peak RSS (python+engine), '
                       f'{time.time() - started:.0f}s probe')
    return max(measured, 1)


def _panel_names(tasks) -> dict[str, float]:
    names: dict[str, float] = {}
    for task in tasks:
        if '_m' in task['task_id']:
            suffix = 'm' + task['task_id'].rsplit('_m', 1)[1]
            names[suffix] = float(task['zombie_count_multiplier'])
    return names


@retry_io
def load_panel(path: Path, extras=()) -> tuple[dict[str, dict], list[float], dict[str, float], dict[str, Path]]:
    """返回 (任务表, 倍率档, 后缀->倍率, 任务->它所在的清单)。

    1.0 档不是新任务：它们就是各源清单里的原任务，所以每个任务的清单要单独记，
    `search_play --manifest` 必须拿到含这个 task_id 的那一份。

    task137: `extras` 是额外登记的清单（随机卡组族），只并进任务表 / 后缀 / 清单映射。
    全局倍率表 values 不动 —— 0.5 / 0.75 这类 <1 档只对自带 ladder 的格子有意义，
    混进 values 会改变所有老格子的 mult_index 语义。
    """
    from unified_panel import read_panel
    document = read_panel(path)
    tasks = {task['task_id']: task for task in document['tasks']}
    manifests = {task['task_id']: path for task in document['tasks']}
    names = _panel_names(document['tasks'])
    for extra in extras:
        extra_document = read_panel(extra)
        for task in extra_document['tasks']:
            if task['task_id'] in tasks:
                raise ValueError(f'extra manifest duplicates task {task["task_id"]}: {extra}')
            tasks[task['task_id']] = task
            manifests[task['task_id']] = Path(extra)
        names.update(_panel_names(extra_document['tasks']))
    for path_ in source_paths({}).values():
        for task_id, task in load_tasks(path_).items():
            if task_id not in tasks:
                tasks[task_id] = task
                manifests[task_id] = path_
    values = sorted({float(task['zombie_count_multiplier']) for task in document['tasks']
                     if task['zombie_count_multiplier'] > 1})
    values = [1.0] + values
    return tasks, values, names, manifests


def parse_task_id(task_id: str, names: dict[str, float]) -> tuple[str, float]:
    """`day_l7_train_d7_m150` -> (`day_l7_train_d7`, 1.5)；原任务 id -> 1.0。"""
    for suffix, value in names.items():
        if task_id.endswith('_' + suffix):
            return task_id[: -len(suffix) - 1], value
    return task_id, 1.0


def training_base_ids() -> set[str]:
    ids = set()
    for family, _key, pattern, levels, _terrain in FAMILIES:
        if family in SUPERSEDED:
            continue
        for level in levels:
            if family in CURRICULUM_LEVELS and level not in CURRICULUM_LEVELS[family]:
                continue
            ids.add(pattern.format(level=level))
    return ids


def cell_of(task_id: str, names: dict[str, float]) -> str:
    return parse_task_id(task_id, names)[0]


def init_from_runs(patterns: list[str], names: dict[str, float],
                   extra_bases=()) -> tuple[dict, dict]:
    """从已有 run 的 config.json / games.jsonl 初始化窗口胜率与已用偏移。"""
    windows: dict[str, dict[str, dict]] = {}
    offsets: dict[str, int] = {}
    base_ids = training_base_ids() | set(extra_bases)
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            config_path = Path(path)
            if config_path.name != 'config.json':
                continue
            try:
                config = read_json(config_path)
                games_path = config_path.parent / 'games.jsonl'
                if not games_path.exists():
                    continue
                rows = [json.loads(line) for line in retry_io(games_path.read_text)().splitlines() if line.strip()]
                if not isinstance(config, dict) or any(not isinstance(row, dict) for row in rows):
                    raise ValueError('run config/rows must be objects')
                multiplier = float(config.get('zombie_count_multiplier') or 1.0)
                if not math.isfinite(multiplier):
                    raise ValueError('nonfinite multiplier')
                task_id = config.get('task_id')
                if not isinstance(task_id, str):
                    continue
            except (ValueError, TypeError, OSError) as error:
                print(f'WARN history run skipped {config_path}: {error}', file=sys.stderr)
                continue
            base = cell_of(task_id, names)
            if base not in base_ids:
                continue
            bucket = windows.setdefault(base, {}).setdefault(
                str(multiplier), {'recent': [], 'wins': 0, 'games': 0})
            outcomes = [1 if row.get('won') else 0 for row in rows]
            bucket['recent'] = (bucket['recent'] + outcomes)[-WINDOW:]
            bucket['wins'] += sum(outcomes)
            bucket['games'] += len(outcomes)
            try:
                end = int(config.get('seed_offset', 0)) + int(config.get('seeds', 0))
            except (TypeError, ValueError):
                end = 0
            offsets[task_id] = max(offsets.get(task_id, 0), end)
    return windows, offsets


def family_of(base: str) -> str | None:
    """训练 base id -> FAMILIES 里的族短名（`night_l12_train_n10p` -> `night_n10p`）。"""
    for family, _key, pattern, levels, _terrain in FAMILIES:
        if any(base == pattern.format(level=level) for level in levels):
            return family
    return None


def reconcile_cells(state: dict, tasks: dict[str, dict], names: dict[str, float],
                    windows_from_runs: dict) -> list[str]:
    """把状态里的格子对齐到当前训练集合；返回日志行，由调用方逐行写 curriculum.log。"""
    lines = []
    # task137: 随机卡组族的 base id 不在 FAMILIES 里，由清单自己的 random_deck_cell 标记认领。
    training = (training_base_ids()
                | {task_id for task_id, task in tasks.items()
                   if task.get('opening_specialist') or task.get('random_deck_cell')})
    cells = state.setdefault('cells', {})
    retired = state.setdefault('retired_cells', {})
    wanted = {}
    for task_id, task in tasks.items():
        base, multiplier = parse_task_id(task_id, names)
        if multiplier != 1.0 or base in wanted or base not in training:
            continue
        wanted[base] = task
    for base in sorted(cells):
        if base in training or cells[base].get('archive') or cells[base].get('live193'):
            continue   # 示范起点格（task71，带 archive）由 reconcile_start_cells 管，不看训练集合
        retired[base] = cells.pop(base)
        family = family_of(base)
        reason = (f'superseded by {SUPERSEDED[family]}' if family in SUPERSEDED
                  else 'not in the training set')
        lines.append(f'cell retired: {base} ({reason})')
    for base in sorted(retired):
        if not isinstance(retired[base], dict):
            state.setdefault('cell_errors', {})[base] = 'invalid retired cell object'
            lines.append(f'cell remains retired: {base}: invalid retired cell object')
            continue
        if base not in training or base in state.get('cell_errors', {}) or retired[base].get('archive'):
            continue
        cells[base] = retired.pop(base)
        lines.append(f'cell restored: {base}')
    for base, task in wanted.items():
        if base in cells or base in state.get('cell_errors', {}):
            continue
        family = family_of(base)
        cell = dict(family=base.rsplit('_train', 1)[0], level=int(task['level']),
                    mult_index=0, last_round=0, windows=dict(windows_from_runs.get(base, {})))
        if family in CURRICULUM_LADDERS:
            cell['ladder'] = CURRICULUM_LADDERS[family]
        if task.get('curriculum_ladder'):
            # task137: 随机卡组格自带 0.5..3.0 的阶梯（0.5 起）。
            cell['ladder'] = list(task['curriculum_ladder'])
        if task.get('random_deck_cell'):
            cell['random_deck_cell'] = True
        if task.get('opening_specialist'):
            cell.update(opening_specialist=True, ladder=[1.0])
        cells[base] = cell
        lines.append(f'cell added: {base}')
    return lines


@retry_io
def load_demo_start_cells(path: Path) -> tuple[dict[str, dict], dict[str, Path]]:
    """示范起点格（task71）：任务清单里带 demo_start 的任务 -> (格定义, 任务 -> 清单)。

    demo_start 字段：
      archive       —— 已验证的示范存档（相对仓库根或绝对路径）
      decisions     —— 起点决策号阶梯，**降序**：0 号最晚（老师已稳定会赢），末位最早
      index         —— 起始档位（缺省 0 = 最晚那档）
      future_seeds  —— RESEED 未来种子块（字面列表）；或写 future_seed_start +
                       future_seed_count（缺省 256）。每组按偏移往后取，跨档共用同一块

    起点之前的示范动作不在这里导出：它们走 human_demo.py export 的
    label_source=human 那条既有路径，一次导出、不重复。
    """
    document = json.loads(path.read_text(encoding='utf-8'))
    cells: dict[str, dict] = {}
    manifests: dict[str, Path] = {}
    for task in document.get('tasks') or []:
        start = task.get('demo_start')
        if not start:
            continue
        ladder = [int(value) for value in start['decisions']]
        if not ladder or ladder != sorted(ladder, reverse=True):
            raise ValueError(f'{task["task_id"]}: demo_start.decisions must be descending')
        if len(set(ladder)) != len(ladder):
            raise ValueError(f'{task["task_id"]}: demo_start.decisions has duplicates')
        seeds = [int(seed) for seed in start.get('future_seeds') or []]
        if not seeds:
            # 清单里写块起点而不是 256 个字面值：future_seed_start / future_seed_count。
            first = start.get('future_seed_start')
            if first is None:
                raise ValueError(f'{task["task_id"]}: demo_start needs future_seeds or future_seed_start')
            seeds = list(range(int(first), int(first) + int(start.get('future_seed_count', 256))))
        if any(START_SEED_POOL[0] <= seed < START_SEED_POOL[1] for seed in seeds):
            # 清单块必须在续领池之外：续领块由账本保证唯一，清单块不在账本里，一旦落进池里
            # 就可能和将来发出的块撞上，而且撞了也查不出来。
            raise ValueError(f'{task["task_id"]}: demo_start future seeds {seeds[0]}..{seeds[-1]} '
                             f'overlap the renewal pool {list(START_SEED_POOL)}')
        archive = Path(start['archive'])
        archive = archive if archive.is_absolute() else ROOT / archive
        # Archive checks belong to planning: a bad cell must not abort qc startup.
        cells[task['task_id']] = dict(
            family='demo_start', level=int(task['level']), mult_index=int(start.get('index', 0)),
            last_round=0, windows={}, ladder=ladder, task_id=task['task_id'],
            archive=str(archive), seeds=seeds,
            seed_blocks=start_seed_segments(seeds))
        manifests[task['task_id']] = path
    return cells, manifests


def start_seed_segments(seeds: list[int]) -> list[list[int]]:
    """Encode contiguous runs without changing the manifest's seed order."""
    blocks = []
    for seed in seeds:
        if blocks and blocks[-1][1] == seed:
            blocks[-1][1] = seed + 1
        else:
            blocks.append([seed, seed + 1])
    return blocks


def renew_start_seeds(cell: dict, ledger_path: Path, size: int = START_SEED_BLOCK) -> list[int]:
    """给起点格续领一段未来种子，接到种子表尾部（task99）。

    偏移语义不变：offset 仍是从种子表头数的绝对下标，续领只是把表变长。领到的块记进
    格子的 seed_blocks（第一段永远是清单里原来的块），账本在 git 外、带文件锁，块只发一次。
    """
    if not cell.get('seed_blocks') and cell.get('seeds'):
        cell['seed_blocks'] = start_seed_segments(cell['seeds'])
    block = allocate_seed_block(ledger_path, size, pool=START_SEED_POOL)
    cell.setdefault('seed_blocks', []).append([int(block[0]), int(block[1])])
    cell['seeds'] = list(cell['seeds']) + list(range(int(block[0]), int(block[1])))
    return [int(block[0]), int(block[1])]


def reconcile_start_cells(state: dict, specs: dict[str, dict], windows: dict | None = None,
                          restore: tuple[str, ...] = ()) -> list[str]:
    """把状态里的示范起点格对齐到清单：新增、更新阶梯/种子块、撤掉已删的格。

    普通格的 reconcile_cells 会把不在训练集合里的格退休；起点格的 base 是示范任务 id
    （如 demo_fog_33_cactus_snow_m200），永远不在 training_base_ids 里，所以它由这里管，
    并且 reconcile_cells 会跳过带 ladder 的格。

    task99：种子表可以有多段（续领）。清单里的块只是第一段，重启时**不能**用它覆盖
    已续领的表——只有当状态里的表不再以清单块开头（清单换了块）才重置。restore 里的格
    从 retired_cells 原样取回（保留偏移/续领段/档位）；清单里已经没有的格标 legacy_restored，
    否则下一次重启会被当成“已删的格”再退休一次。
    """
    lines = []
    cells = state.setdefault('cells', {})
    retired = state.setdefault('retired_cells', {})
    for base in sorted(cells):
        # 起点格以 archive 识别：原生倍率阶梯格（task75b）也带 ladder，不能被当成起点格退休。
        if (cells[base].get('archive') and base not in specs
                and not cells[base].get('legacy_restored') and not cells[base].get('live193')):
            retired[base] = cells.pop(base)
            lines.append(f'start cell retired: {base}')
    for base in sorted(set(restore)):
        cell = retired.pop(base, None)
        if cell is None or not cell.get('archive'):
            continue
        if base not in specs:
            cell['legacy_restored'] = True
        cells[base] = cell
        offset = int(state.get('offsets', {}).get(base, 0))
        seeds = list(cell.get('seeds') or [])
        remaining = set(seeds[offset:])
        clashes = sorted(other for other, live in cells.items()
                         if other != base and remaining & set(live.get('seeds') or []))
        lines.append(f'start cell restored: {base} (retired {cell.get("retired_reason", "unknown")}, '
                     f'offset {offset}/{len(seeds)})')
        if clashes:
            # 恢复的格若与在跑的格共用未来种子，采出来的证据不独立（L32 的旧格与 _hv2 就是同一块）。
            lines.append(f'start cell restored: {base} WARNING future seeds overlap '
                         f'{", ".join(clashes)}; renew or drop it before collecting')
    for base, spec in sorted(specs.items()):
        if base in state.get('cell_errors', {}):
            continue
        cell = cells.get(base)
        if cell is None:
            if base in retired:
                # Re-adding a retired cell from the manifest would discard its
                # renewed table, windows and rung. Explicit restore above keeps them.
                continue
            cells[base] = dict(spec, windows=dict((windows or {}).get(base, {})))
            lines.append(f'start cell added: {base} ({len(spec["ladder"])} rungs)')
            continue
        seeds = list(cell.get('seeds') or [])
        if seeds[:len(spec['seeds'])] != list(spec['seeds']):
            seeds = list(spec['seeds'])                      # 清单换了块：回到第一段
            cell['seed_blocks'] = list(spec.get('seed_blocks') or [])
        changed = [key for key in ('ladder', 'archive') if cell.get(key) != spec[key]]
        if seeds != cell.get('seeds'):
            changed.append('seeds')
        cell.update({key: spec[key] for key in ('ladder', 'archive', 'family', 'level', 'task_id')})
        cell['seeds'] = seeds
        cell['mult_index'] = min(int(cell.get('mult_index', 0)), len(spec['ladder']) - 1)
        if changed:
            lines.append(f'start cell updated: {base} ({", ".join(changed)})')
    return lines


def make_state(tasks: dict[str, dict], names: dict[str, float], windows: dict, offsets: dict,
               panel: str, start_cells: dict[str, dict] | None = None) -> dict:
    state = dict(schema_version=1, panel=panel, round=0, cells={}, offsets=offsets, log=[])
    reconcile_cells(state, tasks, names, windows)
    if start_cells:
        reconcile_start_cells(state, start_cells)
    return state


def cell_level(cell: dict, values: list[float]) -> float:
    """格子当前档位的值：普通格是倍率表里的倍率，示范起点格（task71）是它自己的起点决策号表。

    起点表按**决策号降序**排：0 号是最晚、最容易接手的起点（老师已稳定会赢），
    升档 = 下标 +1 = 起点往前挪一档（更早、更难）。
    """
    ladder = cell.get('ladder')
    if ladder:
        return ladder[cell['mult_index']]
    return values[cell['mult_index']]


def _ceiling(cell: dict, values: list[float]) -> int:
    return len(cell['ladder']) - 1 if cell.get('ladder') else len(values) - 1


def rate(cell: dict, multiplier: float) -> float | None:
    bucket = cell['windows'].get(str(multiplier))
    if not bucket or not bucket.get('recent'):
        return None
    return sum(bucket['recent']) / len(bucket['recent'])


def student_rate(cell: dict, multiplier: float) -> float | None:
    """学生在该格的最近胜率：先看当前倍率窗口，没有就退回最近一个非空窗口
    （刚换档还没数据时，这就是执行现实）。全无数据返回 None（第一个组保持 β=0 的探针）。"""
    current = rate(cell, multiplier)
    if current is not None:
        return current
    latest = None
    for bucket in cell['windows'].values():
        if bucket.get('recent'):
            latest = sum(bucket['recent']) / len(bucket['recent'])
    return latest


def adjust(cells: dict, values: list[float]) -> list[str]:
    """按窗口胜率升降档；返回变更描述（用于日志）。"""
    changes = []
    for base, cell in sorted(cells.items()):
        current = cell_level(cell, values)
        bucket = cell['windows'].get(str(current)) or {}
        if len(bucket.get('recent') or []) < MIN_SAMPLES:
            continue
        win_rate = sum(bucket['recent']) / len(bucket['recent'])
        if win_rate > PROMOTE_ABOVE and cell['mult_index'] < _ceiling(cell, values):
            cell['mult_index'] += 1
            changes.append(f'{base}: {current:g} -> {cell_level(cell, values):g} (win {win_rate:.2f})')
        elif win_rate < DEMOTE_BELOW and cell['mult_index'] > 0:
            cell['mult_index'] -= 1
            changes.append(f'{base}: {current:g} -> {cell_level(cell, values):g} (win {win_rate:.2f})')
    return changes


def allocate(cells: dict, values: list[float], budget: int, per_group: int,
             round_index: int, teacher: dict | None = None,
             random_deck_cap: int | None = None) -> list[tuple[str, float, int]]:
    """返回 [(base, multiplier, games)]，总局数不超过 budget。

    1) 保底（防饿死）：超过 FLOOR_ROUNDS 轮没被采过（含 last_round=0 从未采过）的格，
       按 (last_round, base) 最久优先轮转，每格 per_group // FLOOR_SHARE 局，走当前倍率；
       预算 = 每轮 ceil(len(cells) / FLOOR_ROUNDS) 个保底组。
    2) 其余预算按"可教差距"降序：老师胜率（有证据时）− 学生胜率（当前倍率窗口，无数据
       按 0）。老师没有证据时按可教处理——beta0 采集本身就是老师探针；学生全败而老师
       会赢的格（如 L20：老师 29/32、学生 0/128）排最前；老师也赢不了的格差距趋 0，
       自然沉底（雾区格本就不在训练集合）。同分按最久没采的先。
    allocate 不改 last_round（task59）：它由 run_round 在每组完成时写并落盘。plan 只在
    内存里标 picked；控制器组间崩溃重启后重新 plan，没跑的组照常排出、本轮已完成的组
    不重采（last_round == round_index 跳过）。
    """
    if not cells or budget < 1 or per_group < 1:
        return []
    teacher = teacher or {}
    picks: list[tuple[str, float, int]] = []
    picked: set[str] = set()
    spent = 0
    # task137: 随机卡组格单独限流，别挤掉现有课程（默认每轮 <= 15% 局数）。
    random_spent = 0

    def fits(base: str, games: int) -> bool:
        if not cells[base].get('random_deck_cell'):
            return True
        return random_deck_cap is None or random_spent + games <= random_deck_cap

    def take(base: str, multiplier: float, games: int) -> None:
        nonlocal random_spent
        picks.append((base, multiplier, games))
        picked.add(base)
        if cells[base].get('random_deck_cell'):
            random_spent += games
    floor_games = max(1, per_group // FLOOR_SHARE)
    floor_budget = min(budget, -(-len(cells) // FLOOR_ROUNDS) * floor_games)
    for base in sorted(cells, key=lambda name: (cells[name]['last_round'], name)):
        if spent + floor_games > floor_budget or cells[base]['last_round'] > round_index - FLOOR_ROUNDS:
            break
        if not fits(base, floor_games):
            continue
        take(base, cell_level(cells[base], values), floor_games)
        spent += floor_games
    # 随机卡组格在保底排序里永远排在最后（名字以 rd 开头，同 age 时 alphabetical 落后），
    # 单靠上面那一轮会被现有课程饿死。这里给它一份独立预算：按最久没采轮转，
    # 每轮最多 random_deck_cap 局（默认 = round-games 的 15%）。
    if random_deck_cap:
        for base in sorted((name for name, value in cells.items()
                            if value.get('random_deck_cell') and name not in picked
                            and (value['last_round'] == 0
                                 or value['last_round'] <= round_index - FLOOR_ROUNDS)),
                           key=lambda name: (cells[name]['last_round'], name)):
            if random_spent + floor_games > random_deck_cap or spent + floor_games > budget:
                break
            take(base, cell_level(cells[base], values), floor_games)
            spent += floor_games
    scored = []
    for base, cell in cells.items():
        if base in picked or cells[base]['last_round'] == round_index:
            continue   # 本轮已完成的组（组间崩溃重启后的重排）不重采
        current = cell_level(cell, values)
        student_win = rate(cell, current)
        student = 0.0 if student_win is None else student_win
        teacher_rate = teacher.get(base, teacher.get(f'level:{cell["level"]}', 1.0))
        gap = min(1.0, max(0.0, teacher_rate - student))
        scored.append((-gap, cells[base]['last_round'], base, current))
    remaining = budget - spent
    for _gap, _age, base, multiplier in sorted(scored):
        if remaining < per_group:
            break
        if not fits(base, per_group):
            continue
        take(base, multiplier, per_group)
        remaining -= per_group
    return picks


def load_teacher_rates(path: Path) -> dict:
    """读老师胜率证据（默认 ~/pvz-logs/teacher_rates.json，缺文件返回空 = 全部按未知处理）。

    格式：{"levels": {"<level>": <rate 或 {"wins": w, "games": g}>},
           "bases": {"<base>": 同上}}；bases 优先于 levels。采样过小的条目应由
           生成方把关（见 MEMORY/docs/task57 的生成脚本），这里照单全收。
    """
    try:
        document = read_json(path)
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict):
        return {}
    rates: dict[str, float] = {}
    for kind in ('bases', 'levels'):
        entries = document.get(kind) or {}
        if not isinstance(entries, dict):
            continue
        for name, value in entries.items():
            try:
                if isinstance(value, dict):
                    games = value.get('games', 0)
                    value = value.get('wins', 0) / games if games else 1.0
                key = name if kind == 'bases' else f'level:{name}'
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError('nonfinite rate')
                rates[key] = min(1.0, max(0.0, value))
            except (TypeError, ValueError, ZeroDivisionError) as error:
                print(f'WARN teacher-rate entry skipped {kind}.{name}: {error}', file=sys.stderr)
    return rates


def task_id_for(base: str, multiplier: float, tasks: dict[str, dict], names: dict[str, float]) -> str:
    for task_id in tasks:
        candidate_base, candidate_mult = parse_task_id(task_id, names)
        if candidate_base == base and candidate_mult == multiplier:
            return task_id
    raise KeyError(f'no task for {base} at {multiplier:g}')


def label_for(round_index: int, base: str, multiplier: float, start: int | None = None) -> str:
    short = base.replace('_train', '').replace('day_l', 'd').replace('night_l', 'n').replace('pool_l', 'p')
    short = short.replace('_', '')
    if start is not None:
        # 示范起点格：档位是起点决策号，不是倍率，标签里记 s<决策号>。
        return f'c{round_index}{short}s{int(start)}'
    return f'c{round_index}{short}m%03d' % round(multiplier * 100)


def parse_levels(text: str) -> frozenset[int]:
    """`31-40`, `31,32,37` or `` -> the levels whose cells get --save-candidates."""
    levels: set[int] = set()
    for part in (piece.strip() for piece in text.split(',')):
        if not part:
            continue
        low, _, high = part.partition('-')
        start, stop = int(low), int(high or low)
        if stop < start:
            raise ValueError(f'invalid level range: {part}')
        levels.update(range(start, stop + 1))
    return frozenset(levels)


def save_candidates_for(cell: dict, task: dict | None, levels) -> int:
    """Per-cell switch for search_play --save-candidates (task120).

    A manifest task may declare `save_candidates` itself; otherwise the zone list decides.
    Default is the fog night curriculum 31-40: task120 measured the teacher's blind spot there
    (first 150+ cards ever planted in L32/L37) and the wall-clock cost lands on those cells.
    """
    declared = (task or {}).get('save_candidates')
    if declared is not None:
        return int(declared)
    return 1 if cell.get('level') in levels else 0


def growth_args(args, workers, upper_workers, per_worker_bytes, headroom_bytes) -> list[str]:
    """task138: --workers 是上限，开工宽度走 --initial-workers。

    只有实测过单 worker RSS 才让 search_play 自己扩容；没实测时它保持开工宽度
    （按未知成本加进程就是 OOM 的来源）。上限取 pipeline.foreground_limit()，
    它已经扣掉了可能开始的评估进程。
    """
    initial = args.workers if workers is None else workers
    # 没有声明上限 = 调用方要固定宽度（探针、一次性诊断），保持老语义。
    if upper_workers is None:
        return ['--workers', str(initial)]
    upper = upper_workers
    if initial >= upper and per_worker_bytes <= 0:
        return ['--workers', str(initial)]
    return ['--workers', str(upper), '--initial-workers', str(initial),
            '--worker-memory-bytes', str(max(0, per_worker_bytes)),
            '--memory-headroom-bytes', str(max(0, headroom_bytes))]


def command_for(args, manifest: Path, task_id: str, offset: int, games: int, label: str,
                workers: int | None = None, frames: Path | None = None,
                out: Path | None = None, beta: float = 0.0,
                demo_archive: str | None = None, demo_decision: int | None = None,
                future_seeds: list[int] | None = None,
                save_candidates: int = 0, upper_workers: int | None = None,
                per_worker_bytes: int = 0, headroom_bytes: int = 0,
                task_definition: dict | None = None,
                seed_offset: int | None = None) -> list[str]:
    """生产采集命令。

    示范起点格（task71）：局面由（存档、起点决策号）恢复，未来随机流走 RESEED
    （--future-seeds），一个未来种子一局。--seed-offset/--seeds 仍给 0/局数——
    search_play 在 --demo-archive 下只接受"示范种子在清单选中段里"，给偏移会把它挤出段外。
    """
    python = str(args.python)
    if demo_archive:
        seed_args = ['--seed-offset', '0', '--seeds', str(games),
                     '--demo-archive', str(demo_archive), '--demo-decision', str(demo_decision),
                     '--future-seeds', ','.join(str(seed) for seed in future_seeds or [])]
    else:
        seed_args = ['--seed-offset', str(offset if seed_offset is None else seed_offset),
                     '--seeds', str(games)]
    return [python, str(ROOT / 'scripts/search_play.py'),
            '--rollout-policy', str(args.rollout_policy),
            *(['--rollout-policy-gpt', str(args.rollout_policy_gpt)]
              if getattr(args, 'rollout_policy_gpt', None) else []),
            '--native-games' if getattr(args, 'native_games', True) else '--no-native-games',
            '--verify-samples', '9', '--student-candidates', '2',
            *(['--save-candidates', str(save_candidates)] if save_candidates else []),
            '--static-value', str(args.static_value), '--value', str(args.value),
            *(['--value-roof', str(args.value_roof)] if getattr(args, 'value_roof', None) else []),
            '--manifest', str(manifest), '--task-id', task_id,
            *(['--task-json', json.dumps(task_definition)] if task_definition is not None else []),
            *seed_args,
            '--executable', str(args.executable), '--dagger-beta', f'{beta:g}',
            *growth_args(args, workers, upper_workers, per_worker_bytes, headroom_bytes),
            '--resource-dir', str(args.resource_dir),
            '--frames', str(args.frames if frames is None else frames), '--frames-label', label,
            *(['--obs-v3'] if getattr(args, 'obs_v3', False) else []),
            '--out', str((args.out if out is None else out) / f'{label}_{games}')]


def pack_labels(state: dict, extra: str = '') -> list[str]:
    """Every label this queue has written so far, plus ``extra`` (the base corpus training also uses)."""
    labels = {group.get('frame_label', group['label'] + '_fast_search')
              for entry in state['log'] for group in entry['groups']
              if group.get('status', 'done') == 'done'}
    labels.update(name for name in extra.split(',') if name)
    return sorted(labels)


def pack_command(args, labels: list[str]) -> list[str]:
    """Pack the same row options the trainer resolves from this recipe."""
    recipe = load_recipe(args.recipe, 'training')
    row_options = training_row_options(recipe, labels)
    row_options.pop('source_weight', None)
    options = pack_options(**row_options)
    python = str(args.python)
    return [python, str(ROOT / 'scripts/pack_training_rows.py'),
            '--data', str(args.frames), '--labels', ','.join(sorted(labels)),
            *pack_flags(options), '--precompute-workers', str(recipe['precompute_workers']),
            '--cache-dir', str(recipe['cache_dir'] or (args.frames / '.feature_cache')), '--prune']


class TrainingPipeline:
    """One durable, serial pack/train queue, overlapping the foreground game groups.

    Only the foreground thread updates curriculum state. The background publishes a completion
    marker after a quality decision; interrupted snapshots can be resumed independently
    of seed offsets. Models are immutable per round and activated only by poll at group boundaries.
    """

    def __init__(self, args, state):
        self.args, self.state = args, state
        self.game_health = getattr(args, 'game_health', None) or GameHealth(args)
        self.directory = args.state.parent / (args.state.stem + '.pipeline')
        retry_io(self.directory.mkdir)(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='curriculum-training')
        self.lock = threading.Lock()
        self.process = None
        self.processes = {}
        self.train_parallelism = 1
        self.training_admission = False
        self.lock_fd = None
        self.failed = False
        self.futures = []
        self.build_workers, self.build_peak = 1, 0
        self.last_train_peak = 0
        self.last_pack_peak = 0     # 单建造者实测打包峰值（pack_peak_anon_bytes / 建造者数）
        self.eval_units = self.minimum_eval_units()  # task166: reserve before training finishes
        self.per_worker_bytes = 0   # 单 worker 实测 RSS（run_round 启动探针后填）
        self.foreground_workers = 0 # 前台当前组正在跑的游戏 worker 数
        log_line(args.log, f'evaluation reservation: requested={getattr(args, "min_eval_units", 4)} '
                           f'effective={self.minimum_eval_units()} workers={args.workers}')
        for index, job in enumerate(state.setdefault('training_jobs', [])):
            if (not isinstance(job, dict) or type(job.get('round')) is not int
                    or not isinstance(job.get('recipe'), str) or not isinstance(job.get('model'), str)
                    or not isinstance(job.get('status'), str)):
                state['training_jobs'][index] = dict(round=-1, status='failed', isolated=True,
                    invalid_job=job, error='invalid saved training job')
            elif job.get('status') not in ('done', 'superseded') and job.get('b_lineage') != CLEAN_B_LINEAGE:
                job.update(status='superseded', superseded_reason='clean B experiment replaced old recipes')
        parent, parent_round = self._b_parent(float('inf'))
        state.update(b_student=str(parent), b_round=parent_round, b_lineage=CLEAN_B_LINEAGE)
        for job in reversed(state.get('training_jobs', [])):
            timing = job.get('timing') or {}
            if not isinstance(timing, dict):
                continue
            value = timing.get('train_peak_anon_bytes')
            if value:
                try:
                    self.last_train_peak = int(value)   # 重启后沿用最近一次后台训练的实测峰值
                except (ValueError, TypeError):
                    pass
            if not self.last_pack_peak and timing.get('pack_peak_anon_bytes'):
                try:      # 建造者数记在当轮配方的 precompute_workers 里
                    builders = int(load_recipe(Path(job['recipe']), 'training')['precompute_workers']) or 1
                    self.last_pack_peak = max(1, int(timing['pack_peak_anon_bytes']) // builders)
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if self.last_train_peak and self.last_pack_peak:
                break
        from pvz_promotion import initial_panel
        state.setdefault('gate_panel', initial_panel(read_json(args.promotion_panel)))
        approved = [j for j in state['training_jobs'] if j.get('promotion_approved')
                    and j['round'] > state.get('student_round', 0)]
        if approved:
            latest = max(approved, key=lambda j: j['round'])
            state.update(student=latest['model'], student_round=latest['round'])
            recovered = latest.get('timing') or {}
            if recovered.get('corpus_archives') is not None:
                state['student_corpus_archives'] = recovered['corpus_archives']
            if recovered.get('promotion', {}).get('next_panel') is not None:
                state['gate_panel'] = recovered['promotion']['next_panel']
        save_state(args.state, state)
        queued = [job for job in state.setdefault('training_jobs', [])
                  if job.get('gate_version') == GATE_VERSION and job['status'] not in ('done', 'superseded') and not job.get('isolated')
                  and job.get('attempts', 0) < MAX_BACKGROUND_ATTEMPTS]
        if queued:
            try:       # 重启后恢复待建估算，reservation 才不至于沿用旧段体积
                self._plan_build(load_recipe(Path(queued[-1]['recipe']), 'training'))
            except (OSError, ValueError, KeyError, RuntimeError):
                pass   # 退回 estimate_peak 的已建段估算；_admit 仍按 MemAvailable 把关
        if state.get('student'):
            model = Path(state['student'])
            if not model.is_file():
                raise RuntimeError(f'saved collection student is missing: {model}')
            args.rollout_policy = model
        for job in state.setdefault('training_jobs', []):
            if job in queued:
                if job['status'] == 'paused':
                    self.game_health.pause(job.get('error', 'saved background system failure'))
                job['status'] = 'queued'
                try:
                    self._seeded(job)
                except (Exception, SystemExit) as error:
                    job.update(status='failed', isolated=True, error=str(error))
                    log_line(args.log, f'background round {job["round"]} rejected: {error}')
                    continue
                save_state(self.args.state, self.state)
                self.futures.append((job, self.executor.submit(self._run, copy.deepcopy(job))))

    def pending(self):
        return any(not future.done() for _, future in self.futures)

    def _superseded_by(self, job):
        """Newest round that makes this waiting job obsolete, or None (task141).

        Every round retrains from the whole corpus, so training an older backlog entry
        only delays the newest data. A job is skipped when any newer round is already
        queued, running or trained (or is waiting to retry a background failure).
        Isolated and already-superseded jobs never make an older job obsolete.
        """
        round_index = job.get('round')
        if type(round_index) is not int:
            return None
        newer = [saved['round'] for saved in list(getattr(self, 'state', {}).get('training_jobs', []))
                 if type(saved.get('round')) is int and saved['round'] > round_index
                 and (saved.get('status') in ('queued', 'running', 'done', 'paused')
                      or (saved.get('status') == 'failed' and not saved.get('isolated')
                          and saved.get('attempts', 0) < MAX_BACKGROUND_ATTEMPTS))]
        return max(newer, default=None)

    def foreground_limit(self):
        # Reserve from enqueue onward: training can finish inside a collection group.
        return max(1, self.args.workers - int(self.pending()) * self.minimum_eval_units())

    def minimum_eval_units(self):
        # Small installations retain one collector (workers=1 keeps the old 1+1 case).
        return min(getattr(self.args, 'min_eval_units', 4), max(1, self.args.workers - 1))

    def _evaluation_units(self):
        """Admit the reserved width for both selection and promotion, never silently 1.

        A group already running at enqueue may still occupy every slot. Its successor
        reserves our slots and memory; no live collector is killed or resized here.
        """
        units = self.minimum_eval_units()
        total = mem_total_bytes()
        last_log = None
        while True:
            required = MEMORY_RESERVE_BYTES + units * self.per_worker_bytes
            if total is not None and required > total:
                raise RuntimeError(f'evaluation cannot fit {units} units: required '
                                   f'{required / GIB:.2f} GiB > MemTotal {total / GIB:.2f} GiB')
            available = mem_available_bytes()
            slots = max(1, self.args.workers - self.foreground_workers)
            if slots >= units and (available is None or available >= required):
                self.eval_units = units
                return units
            now = time.monotonic()
            if last_log is None or now - last_log >= 30:
                shown = 'unknown' if available is None else f'{available / GIB:.2f} GiB'
                log_line(self.args.log, f'evaluation waiting: min_units={units} free_slots={slots} '
                                       f'available={shown} required={required / GIB:.2f} GiB')
                last_log = now
            time.sleep(1)

    def incumbent(self):
        with self.lock:
            return Path(self.args.rollout_policy)

    @property
    def gate_version(self):
        return GATE_VERSION

    def evaluation_contract(self, job):
        source = getattr(self.args, 'promotion_panel', None)
        if source is None:
            source = ROOT / 'configs/gate_v5_panel.json'
        extra = {}
        incumbent = self.incumbent()
        trained = read_json(Path(job['recipe']).parent / 'trained.json')
        with self.lock:
            incumbent = Path(self.args.rollout_policy)
            extra = dict(candidates={c['recipe_family']: c['model'] for c in trained['candidates']
                                     if c['status'] == 'trained'},
                         calibration=copy.deepcopy(self.state['gate_panel']))
        return promotion_contract(Path(job['model']), incumbent, source, self.args.executable,
                                  self.args.value, self.args.resource_dir,
                                  seed_block=job.get('seed_block'), **extra)

    @retry_io
    def _seeded(self, job):
        """task73b: one fresh-seed block per evaluation job, allocated on the
        persisted job dict (never on the _run copy), so resumed and stale re-evaluations
        reuse the same block and the ledger is never double-spent. Pool exhaustion
        raises and closes the gate; seeds are never wrapped around and reused."""
        if 'seed_block' not in job:
            job['seed_block'] = list(allocate_seed_block(self.args.seed_ledger,
                                                         promotion_policy()['pair_count']))
        return job

    def _evaluate(self, job):
        """Evaluate one already trained immutable candidate against the current collector."""
        frozen = self.evaluation_contract(job)
        folder = Path(job['recipe']).parent
        out = folder / 'evaluations' / contract_key(frozen)
        python = str(self.args.python)
        units = self._evaluation_units()
        command = [python, str(ROOT / 'scripts/promotion_gate.py'), '--candidate', job['model'],
                   '--incumbent', frozen['incumbent'], '--source-panel', frozen['source_panel'],
                   '--executable', str(self.args.executable), '--value', str(self.args.value),
                   '--resource-dir', str(self.args.resource_dir), '--out', str(out), '--workers', '1',
                   '--eval-units', str(units),
                   '--native-games' if getattr(self.args, 'native_games', True) else '--no-native-games',
                   '--seed-block', f'{frozen["seed_block"][0]},{frozen["seed_block"][1]}',
                   '--seed-ledger', str(self.args.seed_ledger)]
        command += ['--candidates', json.dumps(frozen['candidates']),
                    '--calibration', json.dumps(frozen['calibration'])]
        started = time.time()
        log_line(self.args.log, f'background round {job["round"]} evaluate started against '
                                f'{frozen["incumbent"]} eval_units={units}')
        try:
            self._command(command, folder / 'evaluation.log')
            decision = json.loads((out / 'decision.json').read_text())
        except Exception as error:
            if systemic_game_failure(error):
                raise
            # Evaluation failure is a closed gate, not permission to stop accumulating data.
            decision = dict(version=self.gate_version, approved=False, policy=promotion_policy(),
                            contract=frozen, reasons=['evaluation_error: ' + str(error)])
            out.mkdir(parents=True, exist_ok=True)
            (out / 'error.json').write_text(json.dumps(decision, indent=2) + '\n')
        finished = time.time()
        log_line(self.args.log, f'background round {job["round"]} evaluate done: {finished-started:.3f}s '
                               f'approved={decision["approved"]} reasons={decision["reasons"]}')
        if decision.get('monitor'):
            # task128: heldout cells are monitored, never vetoed. Their wins go into the round report.
            log_line(self.args.log, f'background round {job["round"]} heldout monitor (no veto): ' + ' '.join(
                f'{name}={cell["candidate_wins"]}/{cell["pairs"]}' for name, cell in decision['monitor'].items()))
        self._marker(folder / 'ab_result.json', dict(round=job['round'], experiment=job.get('b_lineage'),
            seed_block=frozen['seed_block'], incumbent=frozen['incumbent'],
            candidates=frozen['candidates'], look=decision.get('sequential', {}).get('look'),
            calibration=frozen['calibration'], **(decision.get('ab') or
                dict(available=False, reason='evaluation or training candidate unavailable'))))
        if decision.get('approved'):
            import shutil
            published = Path(job['model'])
            published.parent.mkdir(parents=True, exist_ok=True)
            temporary = published.with_suffix('.building')
            shutil.copyfile(decision['winner'], temporary)
            temporary.replace(published)
        for side, result in decision.get('candidates', {}).items():
            for name, cell in result['monitor'].items():
                if cell['delta_rate'] <= -.2:
                    log_line(self.args.log, f'WARNING heldout regression round {job["round"]} '
                                           f'{side} {name}: delta={cell["delta_rate"]:.3f}')
        return dict(model=job['model'],
                    evaluation_started=started, evaluation_finished=finished, eval_units=units,
                    promotion=decision, evaluation_directory=str(out))

    def estimate_peak(self):
        # 后台流水线的峰值 = max(打包阶段, 训练阶段)。打包按 _plan_build 的待建段估算
        # （并发建造者 × 单建造者实测优先峰值），且只在打包进行时有值——_run 的 pack
        # 一结束（成败都算）就清零，训练/评估阶段只按训练估算（task61 追加）；
        compact_bytes = sum(p.stat().st_size for p in (self.directory / 'frames/packs/segments').glob('*.npz'))
        if self.last_train_peak:
            train_peak = self.last_train_peak       # 最近一次后台训练的实测匿名峰值优先
        else:
            train_peak = TRAIN_ANON_EXTRA + int(compact_bytes * TRAIN_ANON_FACTOR)
        return max(self.build_peak, train_peak * getattr(self, 'train_parallelism', 1))

    def pending_builds(self, labels: list[str], recipe: dict) -> list[tuple[str, int]]:
        """待建段 [(label, 原始档案字节)]：与 _prepare_corpus 同一合并视图（采集优先），
        判定与 pack() 完全同一套函数——fingerprint_paths / segment_options / segment_path。

        task59 的独立实现把指纹行写成"episode 目录:大小"，丢了 frames 文件名，与 pack 的
        "相对路径:大小"（含文件名）对不上，生产上 202 段全判成待建（2026-10-24 12:33）。
        enqueue 时本轮新采的帧还没有链进视图——缺链接的标签直接待建（pack 必然要建）；
        链接齐全的标签在视图路径上调用真 fingerprint_paths 后查段文件是否存在。
        """
        from pvz_frame_store import frame_paths
        base = Path(load_recipe(self.args.recipe, 'training')['data'])
        sources = [base] if base == self.args.frames else [base, self.args.frames]
        from mac_collect_import import batch_sources
        sources += batch_sources(self.args.frames)
        wanted = set(labels)
        catalog: dict[str, Path] = {}
        for source in sources:
            for path in frame_paths(source):
                if path.parent.name in wanted:
                    catalog[str(path.parent.relative_to(source))] = path
        options = pack_options(**training_row_options(recipe, labels))
        view = self.directory / 'frames'
        by_label = {}
        for rel, path in catalog.items():
            by_label.setdefault(Path(rel).name, []).append((rel, path))
        pending = []
        for label in sorted(wanted):
            archives = sorted(by_label.get(label, []))
            if not archives:
                continue   # 没有档案的标签 pack 时会自己报错，不进估算
            total = sum(path.stat().st_size for _, path in archives)
            view_paths = [view / rel / path.name for rel, path in archives]
            if not all(view_path.exists() for view_path in view_paths):
                pending.append((label, total))
                continue
            fingerprint = fingerprint_paths(view, view_paths)
            target = segment_path(view, segment_options(options, label, fingerprint))
            if not target.is_file():
                pending.append((label, total))
        return pending

    def _plan_build(self, recipe: dict) -> None:
        """按待建段估算打包并发与峰值；enqueue 写进配方，estimate_peak/reservation 引用。"""
        pending = self.pending_builds(recipe['labels'], recipe)
        self.build_workers, per_builder = build_plan([size for _, size in pending],
                                                     recipe['precompute_workers'], mem_available_bytes(),
                                                     measured_per_builder=self.last_pack_peak)
        self.build_peak = self.build_workers * per_builder
        log_line(self.args.log, f'pack plan: {len(pending)} pending segments, '
                                f'workers={self.build_workers}, peak~{self.build_peak / GIB:.1f} GiB')

    @retry_io
    def reservation(self):
        if not self.pending():
            return 0
        with self.lock:
            processes = list(getattr(self, 'processes', {}).values())
            if not processes and self.process is not None:
                processes = [self.process]
            anonymous = sum(_tree_anon_bytes(process.pid) for process in processes)
        peak = max(self.estimate_peak(), self.minimum_eval_units() * self.per_worker_bytes)
        return max(0, peak - anonymous)

    def _b_parent(self, before):
        parent = Path(getattr(self.args, 'b_init', DEFAULT_B_INIT))
        round_index = 0
        if self.state.get('b_lineage') == CLEAN_B_LINEAGE and self.state.get('b_round', 0) < before:
            parent, round_index = Path(self.state['b_student']), self.state['b_round']
        for old in self.state.get('training_jobs', []):
            if old.get('b_lineage') != CLEAN_B_LINEAGE or not round_index < old['round'] < before:
                continue
            marker = Path(old['recipe']).parent / 'seeds/seed_000002/trained.json'
            if marker.is_file():
                candidate = read_json(marker)
                if candidate['status'] == 'trained':
                    parent, round_index = Path(candidate['model']), old['round']
        return parent, round_index

    def _b_recipe(self, job):
        path = Path(job['recipe']).with_name('recipe_b.json')
        recipe = load_recipe(path, 'training')
        parent, parent_round = self._b_parent(job['round'])
        recipe.update(init=str(parent), epochs=job['b_epochs'] if parent_round else job['b_initial_epochs'])
        write_recipe(path, recipe)
        return recipe

    @retry_io
    def enqueue(self):
        round_index = self.state['round']
        if any(job['round'] == round_index for job in self.state['training_jobs']):
            return
        recipe = load_recipe(self.args.recipe, 'training')
        from mac_collect_import import batch_sources
        recipe['labels'] = pack_labels(self.state, ','.join(
            recipe['labels'] + [path.name for path in batch_sources(self.args.frames)]))
        recipe['data'] = str(self.directory / 'frames')
        folder = self.directory / f'round_{round_index:06d}'
        recipe['out'] = str(folder / 'student')
        recipe['cache_dir'] = str(self.directory / 'cache')
        self._plan_build(recipe)
        recipe['precompute_workers'] = self.build_workers
        path = folder / 'recipe.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        write_recipe(path, recipe)
        job = dict(round=round_index, recipe=str(path), model=str(Path(recipe['out']) / 'model.pt'),
                   status='queued')
        alternate = getattr(self.args, 'alternate_recipe', None)
        alternate = alternate or ROOT / 'recipes/dw4i2_v2timer_clean_b.json'
        job['gate_version'] = 7
        job.update(b_lineage=CLEAN_B_LINEAGE,
                   b_initial_epochs=getattr(self.args, 'b_initial_epochs', 8),
                   b_epochs=getattr(self.args, 'b_epochs', 4))
        if alternate:
            variant = load_recipe(alternate, 'training')
            variant.update({key: recipe[key] for key in
                            ('labels', 'data', 'out', 'cache_dir', 'precompute_workers')})
            base_options, variant_options = training_row_options(recipe), training_row_options(variant)
            base_options.pop('source_weight', None)
            variant_options.pop('source_weight', None)
            base_options.pop('label_weight', None)
            variant_options.pop('label_weight', None)
            if base_options != variant_options:
                raise ValueError('A/B recipes must share pack row options')
            parent, parent_round = self._b_parent(round_index)
            variant.update(init=str(parent),
                           epochs=job['b_epochs'] if parent_round else job['b_initial_epochs'])
            # Promoted training corpus, not the in-flight collector snapshot. Labels are
            # an initial fallback for the pre-v7 incumbent; after promotion use archives.
            if 'student_corpus_archives' in self.state:
                variant['replay_old_archives'] = self.state['student_corpus_archives']
                variant['replay_old_labels'] = []
            else:
                old = dict(self.state, log=[entry for entry in self.state['log']
                           if entry['round'] <= self.state.get('student_round', 0)])
                variant['replay_old_labels'] = pack_labels(old, ','.join(
                    load_recipe(self.args.recipe, 'training')['labels']))
            variant_path = folder / 'recipe_b.json'
            write_recipe(variant_path, variant)
            repeats = 1
            job['candidate_recipes'] = [
                dict(recipe=str(candidate_path), source_recipe=str(source), recipe_family=family)
                for candidate_path, source, family in ((path, self.args.recipe, 'A'),
                                                       (variant_path, alternate, 'B'))
                for _ in range(repeats)]
            job['training_seeds'] = list(range(1, len(job['candidate_recipes']) + 1))
        # Freeze the plan on the durable job. Historical jobs without this field
        # retain their original single-seed training/resume semantics.
        job['training_parallelism'] = getattr(self.args, 'training_parallelism', 1)
        self.state['training_jobs'].append(job)
        try:
            self._seeded(job)
        except (Exception, SystemExit) as error:
            job.update(status='failed', isolated=True, error=str(error))
            save_state(self.args.state, self.state)
            raise
        save_state(self.args.state, self.state)  # durable before the worker is submitted
        log_line(self.args.log, f'background round {round_index} queued: {path}')
        self.futures.append((job, self.executor.submit(self._run, copy.deepcopy(job))))



    def _prepare_corpus(self, labels):
        """An append-only local view of recipe archives + collected archives, without copying bytes.

        Production frames_v2 historically lacks two recipe labels. Never silently train a
        smaller corpus: merge both roots per archive (collection wins duplicates), and reject
        labels absent from both. Each link is an episode directory; all packs/cache stay local.
        """
        from pvz_frame_store import frame_paths
        base = Path(load_recipe(self.args.recipe, 'training')['data'])
        sources = [base] if base == self.args.frames else [base, self.args.frames]
        from mac_collect_import import batch_sources
        sources += batch_sources(self.args.frames)
        wanted = set(labels)
        catalog = {}
        for source in sources:
            for path in frame_paths(source):
                if path.parent.name in wanted:
                    catalog[path.parent.relative_to(source)] = path.parent.resolve()
        missing = wanted - {relative.name for relative in catalog}
        if missing:
            raise RuntimeError(f'no archives for labels in recipe or collection roots: {sorted(missing)}')
        view = self.directory / 'frames'
        for relative, source in sorted(catalog.items()):
            target = view / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                if target.resolve() != source:
                    raise RuntimeError(f'archive source changed for pinned corpus: {target}')
            elif target.exists():
                raise RuntimeError(f'corpus view expects an archive link: {target}')
            else:
                target.symlink_to(source, target_is_directory=True)

    def _command(self, command, log):
        health = getattr(self.args, 'game_health', None)
        if health is not None and not health.wait():
            raise GameRecoveryStopped('background stopped while waiting for game recovery')
        with log.open('a') as output:
            offset = output.tell()
            process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                       pass_fds=() if self.lock_fd is None else (self.lock_fd,))
            with self.lock:
                if not hasattr(self, 'processes'):
                    self.processes = {}
                self.processes[process.pid] = process
                self.process = process
            peak = 0
            try:
                while True:
                    code = process.poll()
                    if code is not None:
                        break
                    peak = max(peak, _tree_anon_bytes(process.pid))
                    time.sleep(2)
            finally:
                with self.lock:
                    self.processes.pop(process.pid, None)
                    self.process = next(iter(self.processes.values()), None)
        if code:
            with log.open('rb') as output:
                output.seek(max(offset, log.stat().st_size - 8192))
                diagnostic = output.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'exit {code}; see {log}\n{diagnostic}')
        return peak

    def _admit(self):
        total = mem_total_bytes()
        # Group admission happens before either CUDA process starts. A failed
        # two-worker admission falls back immediately rather than waiting for two.
        available, gpu = mem_available_bytes(), None
        if getattr(self, 'training_admission', False):
            gpu = gpu_free_bytes()
            required = MEMORY_RESERVE_BYTES + self.estimate_peak()
            reasons = []
            if self.train_parallelism > 1:
                if available is None or available < required or total is not None and total < required:
                    reasons.append('host_memory')
                if gpu is None or gpu < GPU_RESERVE_BYTES + self.train_parallelism * TRAIN_GPU_BYTES:
                    reasons.append('gpu_memory_unknown_or_insufficient')
                if reasons:
                    self.train_parallelism = 1
            self.train_admission_record = dict(workers=self.train_parallelism,
                mode='parallel' if self.train_parallelism > 1 else 'serial', fallback_reasons=reasons,
                available_bytes=available, gpu_free_bytes=gpu, required_bytes=MEMORY_RESERVE_BYTES + self.estimate_peak())
        required = MEMORY_RESERVE_BYTES + self.estimate_peak()
        if total is not None and required > total:
            # 不可能满足的需求不能静默死等（task61）：让该训练任务失败退出、写明原因，
            # poll() 会把它变成控制器可见的错误，停止标志与控制器退出不再被卡住。
            raise RuntimeError(
                f'background training cannot fit: required {required / GIB:.1f} GiB '
                f'(reserve {MEMORY_RESERVE_BYTES / GIB:.1f} + estimate {self.estimate_peak() / GIB:.1f}) '
                f'> MemTotal {total / GIB:.1f} GiB')
        last_log = 0
        while True:
            available = mem_available_bytes()
            # Current game workers are already excluded from MemAvailable. Future groups
            # separately subtract reservation(), so adding their RSS here would count twice.
            required = MEMORY_RESERVE_BYTES + self.estimate_peak()
            gpu_ok = True
            if getattr(self, 'training_admission', False):
                gpu = gpu_free_bytes()
                if self.train_parallelism > 1 and ((available is not None and available < required)
                        or gpu is None or gpu < GPU_RESERVE_BYTES + self.train_parallelism * TRAIN_GPU_BYTES):
                    self.train_parallelism = 1
                    required = MEMORY_RESERVE_BYTES + self.estimate_peak()
                    self.train_admission_record.update(workers=1, mode='serial', required_bytes=required,
                        available_bytes=available, gpu_free_bytes=gpu)
                    self.train_admission_record['fallback_reasons'].append('resources_changed_before_launch')
                gpu_ok = gpu is None or gpu >= GPU_RESERVE_BYTES + self.train_parallelism * TRAIN_GPU_BYTES
            if (available is None or available >= required) and gpu_ok:
                return
            if time.monotonic() - last_log > 30:
                shown = 'unknown' if available is None else f'{available / GIB:.2f} GiB'
                gpu_shown = ''
                if getattr(self, 'training_admission', False):
                    free = 'unknown' if gpu is None else f'{gpu / GIB:.2f} GiB'
                    gpu_shown = f' gpu_free={free} gpu_required={(GPU_RESERVE_BYTES + self.train_parallelism * TRAIN_GPU_BYTES) / GIB:.2f} GiB'
                log_line(self.args.log, f'background waiting for memory: available={shown} '
                                       f'required={required / GIB:.2f} GiB{gpu_shown}')
                last_log = time.monotonic()
            time.sleep(1)

    def _run(self, job):
        while self.game_health.wait():
            try:
                return self._run_snapshot(job)
            except (Exception, SystemExit) as error:
                if not systemic_game_failure(error):
                    raise
                self.game_health.pause(str(error))
        raise GameRecoveryStopped('background stopped while waiting for game recovery')

    def _run_snapshot(self, job):
        recipe_path = Path(job['recipe'])
        folder = recipe_path.parent
        done = folder / 'complete.json'
        trained = folder / 'trained.json'
        try:
            # A second controller cannot train this snapshot concurrently, even after its
            # parent's unexpected death. OS file locks follow the live process, not stale files.
            import fcntl
            with (folder / 'training.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self.lock_fd = lock.fileno()  # child retains lock if the controller unexpectedly dies
                old = json.loads(done.read_text()) if done.is_file() else None
                if old:
                    if valid_certificate(old.get('promotion', {}), self.evaluation_contract(job)):
                        return old
                # task141: a finished round keeps its report; everything still waiting behind
                # a newer round is skipped instead of trained and gated on stale data.
                newer = self._superseded_by(job)
                if newer is not None:
                    return dict(superseded_by=newer)
                if trained.is_file():
                    report = json.loads(trained.read_text())
                else:
                    report = self._train_candidates(job)
                temporary = trained.with_suffix('.building')
                temporary.write_text(json.dumps(report, indent=2) + '\n')
                temporary.replace(trained)
                report.update(self._evaluate(job))
                temporary = done.with_suffix('.building')
                temporary.write_text(json.dumps(report, indent=2) + '\n')
                temporary.replace(done)
                return report
        except BaseException:
            self.failed = True
            raise

    @staticmethod
    def _marker(path, document):
        temporary = path.with_suffix('.building')
        temporary.write_text(json.dumps(document, indent=2) + '\n')
        temporary.replace(path)

    def _train_candidates(self, job):
        """One corpus, one resource-admitted candidate per recipe, then the shared-incumbent gate.

        A failed seed is local to its candidate. Only an entirely failed round
        reaches the existing B7 retry/stop mechanism in _run/poll.
        """
        recipe_path = Path(job['recipe'])
        folder = recipe_path.parent
        recipe = load_recipe(recipe_path, 'training')
        report = dict(round=job['round'], model=job['model'])
        pack_marker = folder / 'packed.json'
        if pack_marker.is_file():
            packed = json.loads(pack_marker.read_text())
            self.build_peak = 0
        else:
            self._admit()
            local_args = copy.copy(self.args)
            local_args.recipe, local_args.frames = recipe_path, Path(recipe['data'])
            packed = dict(pack_started=time.time())
            try:
                self._prepare_corpus(recipe['labels'])
                packed['pack_peak_anon_bytes'] = self._command(
                    pack_command(local_args, recipe['labels']), folder / 'pack.log')
            finally:
                self.build_peak = 0
            packed['pack_finished'] = time.time()
            self._marker(pack_marker, packed)
        report.update({key: value for key, value in packed.items() if key.startswith('pack_')})
        from pvz_training_pack import packed_file_from_log
        shared_pack = packed_file_from_log(folder / 'pack.log')
        python = str(self.args.python)
        def train_seed(seed, admitted=False):
            directory = folder / 'seeds' / f'seed_{seed:06d}'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / 'recipe.json'
            variant = job['candidate_recipes'][job['training_seeds'].index(seed)]
            child_recipe = dict(load_recipe(Path(variant['recipe']), 'training'),
                                seed=seed, out=str(directory / 'student'))
            if variant['recipe_family'] == 'B' and not path.exists():
                child_recipe.update(self._b_recipe(job), seed=seed, out=str(directory / 'student'))
            if not path.exists():
                write_recipe(path, child_recipe)
            model, marker = directory / 'student/model.pt', directory / 'trained.json'
            if marker.is_file():
                candidate = json.loads(marker.read_text())
            else:
                candidate = dict(seed=seed, model=str(model), recipe=str(path),
                                 source_recipe=variant.get('source_recipe', str(recipe_path)),
                                 recipe_family=variant.get('recipe_family', 'A'),
                                 init=child_recipe['init'], epochs=child_recipe['epochs'],
                                 train_started=time.time())
                try:
                    if not admitted:
                        self._admit()
                    command = [python, str(ROOT / 'scripts/train_rollout_policy.py'), '--recipe', str(path)]
                    command += ['--pack-recipe', str(recipe_path)]
                    if shared_pack is not None:
                        command += ['--pack-file', str(shared_pack)]
                    candidate['train_peak_anon_bytes'] = self._command(command, directory / 'train.log')
                    candidate['train_data_bytes'] = training_data_bytes(model)
                    if not model.is_file():
                        raise RuntimeError(f'trainer did not publish {model}')
                    candidate.update(status='trained', train_finished=time.time())
                    self._marker(marker, candidate)
                except Exception as error:
                    if systemic_game_failure(error):
                        raise
                    candidate.update(status='failed', train_finished=time.time(), error=str(error))
                    log_line(self.args.log, f'background round {job["round"]} seed {seed} failed; '
                                            f'continuing other seeds: {error}')
            return candidate
        pending = [seed for seed in job['training_seeds']
                   if not (folder / 'seeds' / f'seed_{seed:06d}' / 'trained.json').is_file()]
        self.train_parallelism = min(job.get('training_parallelism', 2), max(1, len(pending)))
        self.training_admission = bool(pending)
        try:
            if pending:
                self._admit()
                mode = dict(self.train_admission_record)
            else:
                mode = dict(workers=0, mode='reused', fallback_reasons=[])
            mode_path = folder / 'training_mode.json'
            previous = json.loads(mode_path.read_text()) if mode_path.exists() else None
            history = [] if previous is None else previous.get('attempt_history', [previous])
            mode['attempt_history'] = history + [dict(mode, recorded=time.time())]
            self._marker(mode_path, mode)
            if mode['workers'] > 1:
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix='student-seed') as trainers:
                    futures = [trainers.submit(train_seed, seed, True) for seed in job['training_seeds']]
                    candidates = [future.result() for future in futures]
            else:
                candidates = [train_seed(seed) for seed in job['training_seeds']]
        finally:
            self.train_parallelism, self.training_admission = 1, False
        self._marker(folder / 'candidates.json', dict(candidates=candidates))
        trained = [candidate for candidate in candidates if candidate['status'] == 'trained']
        if not trained:
            raise RuntimeError('all training seeds failed; see candidates.json')
        report.update(candidates=candidates, training_seeds=job['training_seeds'], training_mode=mode,
                      train_started=min(candidate['train_started'] for candidate in candidates),
                      train_finished=max(candidate['train_finished'] for candidate in candidates),
                      train_peak_anon_bytes=max(candidate.get('train_peak_anon_bytes', 0) or 0
                                                for candidate in trained),
                      train_data_bytes=max(candidate.get('train_data_bytes', 0) or training_data_bytes(Path(candidate['model']))
                                           for candidate in trained))
        self.last_train_peak = report['train_peak_anon_bytes']
        from pvz_frame_store import frame_paths
        report['corpus_archives'] = sorted({str(path.parent.relative_to(Path(recipe['data'])))
            for path in frame_paths(Path(recipe['data'])) if path.parent.name in set(recipe['labels'])})
        return report
    def poll(self, retry: bool = True):
        # A failed state write after enqueue may have left a durable job without
        # a submitted future. Recover it before polling, without duplicating workers.
        for job in self.state.get('training_jobs', []):
            if job.get('gate_version') == GATE_VERSION and job['status'] == 'queued' and not any(saved is job for saved, _ in self.futures):
                try:
                    self._seeded(job)
                    save_state(self.args.state, self.state)
                    self.futures.append((job, self.executor.submit(self._run, copy.deepcopy(job))))
                except StateWriteError:
                    raise
                except (Exception, SystemExit) as error:
                    job.update(status='failed', isolated=True, error=str(error))
                    save_state(self.args.state, self.state)
                    log_line(self.args.log, f'background round {job["round"]} recovery isolated: {error}')
        for index, (job, future) in enumerate(list(self.futures)):
            if job['status'] in ('done', 'failed', 'superseded') or not future.done():
                continue
            try:
                retry_io(self._poll_job)(index, job, future, retry)
            except StateWriteError:
                raise
            except (Exception, SystemExit) as error:
                job.update(status='failed', isolated=True, error=f'{type(error).__name__}: {error}')
                self.failed = False
                save_state(self.args.state, self.state)
                log_line(self.args.log, f'background round {job["round"]} isolated: {job["error"]}')

    def _poll_job(self, index, job, future, retry):
        try:
            report = future.result()
        except BaseException as error:
            if systemic_game_failure(error):
                job.update(status='paused', error=str(error))
                self.failed = False
                save_state(self.args.state, self.state)
                log_line(self.args.log, f'background round {job["round"]} paused; retry budget retained: {error}')
                return
            # task70 B7: a background failure must not kill collection. Record it,
            # retry once; task98 isolates the snapshot after its second failure.
            job['status'] = 'failed'
            job['error'] = str(error)
            job['attempts'] = job.get('attempts', 0) + 1
            save_state(self.args.state, self.state)
            if retry and job['attempts'] < MAX_BACKGROUND_ATTEMPTS:
                log_line(self.args.log, f'background round {job["round"]} failed '
                                        f'(attempt {job["attempts"]}/{MAX_BACKGROUND_ATTEMPTS}); '
                                        f'retrying once: {error}')
                self.failed = False
                job['status'] = 'queued'
                self.futures[index] = (job, self.executor.submit(self._run, copy.deepcopy(job)))
                return
            self.failed = False
            job['isolated'] = job['attempts'] >= MAX_BACKGROUND_ATTEMPTS
            log_line(self.args.log, f'background round {job["round"]} failed '
                                    f'{job["attempts"]} times; snapshot isolated, collection continues: {error}')
            return
        if isinstance(report, dict) and report.get('superseded_by') is not None:
            job.update(status='superseded', superseded_by=report['superseded_by'])
            save_state(self.args.state, self.state)
            log_line(self.args.log, f'background round {job["round"]} superseded by round '
                                    f'{report["superseded_by"]}; training and gate skipped')
            return
        promotion = report.get('promotion', {})
        for candidate in report.get('candidates', []):
            if (job.get('b_lineage') == CLEAN_B_LINEAGE and candidate.get('recipe_family') == 'B'
                    and candidate['status'] == 'trained' and job['round'] > self.state.get('b_round', 0)):
                self.state.update(b_student=candidate['model'], b_round=job['round'], b_lineage=CLEAN_B_LINEAGE)
        current_contract = self.evaluation_contract(job)
        if promotion.get('contract') != current_contract:
            job['status'] = 'queued'
            log_line(self.args.log, f'background round {job["round"]} comparison stale; re-evaluate current student')
            self._seeded(job)
            self.futures[index] = (job, self.executor.submit(self._run, copy.deepcopy(job)))
            save_state(self.args.state, self.state)
            return
        if not valid_certificate(promotion, current_contract):
            promotion = dict(version=self.gate_version, approved=False, policy=promotion_policy(),
                             contract=current_contract, reasons=['evaluation_error: invalid paired evaluation artifacts'])
            report['promotion'] = promotion
            completed = Path(job['recipe']).parent / 'complete.json'
            temporary = completed.with_suffix('.building')
            temporary.write_text(json.dumps(report, indent=2) + '\n')
            temporary.replace(completed)
        job.update(status='done', timing=report, promotion_approved=promotion['approved'],
                   promotion_reasons=promotion['reasons'])
        if promotion.get('next_panel') is not None:
            self.state['gate_panel'] = promotion['next_panel']
        if promotion['approved'] and job['round'] > self.state.get('student_round', 0):
            with self.lock:
                self.state.update(student=job['model'], student_round=job['round'])
                if report.get('corpus_archives') is not None:
                    self.state['student_corpus_archives'] = report['corpus_archives']
                save_state(self.args.state, self.state)
                self.args.rollout_policy = Path(job['model'])
            log_line(self.args.log, f'student activated at group boundary: round {job["round"]} {job["model"]}')
        else:
            save_state(self.args.state, self.state)
            if not promotion['approved']:
                log_line(self.args.log, f'student retained: round {job["round"]} rejected {promotion["reasons"]}')

    def close(self):
        health = getattr(self.args, 'game_health', None)
        if health is not None:
            health.closed.set()
        # poll may schedule a stale comparison again; drain it before closing the executor.
        # B7: failed jobs stay failed here (no retry), so only wait on queued ones.
        try:
            while any(job['status'] == 'queued' for job, _ in self.futures):
                for _, future in list(self.futures):
                    if not future.done():
                        try:
                            future.result()
                        except BaseException:
                            pass  # poll persists and reports the failure
                try:
                    self.poll(retry=False)
                except RuntimeError as error:
                    # B7: a single background failure at final drain is recorded, not fatal.
                    log_line(self.args.log, f'background job failed during final drain: {error}')
        finally:
            self.executor.shutdown(wait=True)


def log_line(path: Path, message: str) -> None:
    stamp = datetime.now().isoformat(timespec='seconds')
    @retry_io
    def append():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as handle:
            handle.write(f'{stamp} {message}\n')
    try:
        append()
    except OSError as error:
        print(f'ERROR curriculum log unavailable: {error}', file=sys.stderr, flush=True)
    print(f'{stamp} {message}', flush=True)


def save_state(path: Path, state: dict) -> None:
    try:
        _write_state(path, state)
    except OSError as error:
        raise StateWriteError(str(error)) from error


@retry_io
def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state['updated'] = datetime.now().isoformat(timespec='seconds')
    temporary = path.with_suffix(path.suffix + '.building')
    temporary.write_text(json.dumps(state, indent=1) + '\n')
    temporary.replace(path)


def review_groups(state, groups, tasks, share, start_seed_ledger, renew_size, changes):
    """Extra 1x data, consuming the original task's next seeds without adapting its ladder."""
    finished = {group.get('review_of') for group in state.get('completed_groups', [])}
    reviews = []
    for group in groups:
        task = group.get('task_definition') or tasks.get(group['task_id'], {})
        games = math.ceil(group['games'] * share)
        if not games or task.get('zombie_count_multiplier', 1.) == 1. or group['label'] in finished:
            continue
        cell = state['cells'].get(group['base'])
        if cell is None:
            continue
        key = group.get('offset_key', group['task_id'])
        renewable = cell.get('live193') or cell.get('archive') is not None
        seeds = cell['seeds'] if renewable else task['seeds']
        offset = max(group['offset'] + group['games'], state['offsets'].get(key, 0))
        while offset + games > len(seeds) and renewable and start_seed_ledger is not None:
            try:
                renew_start_seeds(cell, start_seed_ledger, renew_size)
            except (SeedPoolExhausted, ValueError, OSError) as error:
                changes.append(f'{key}: 1x review seed renewal failed ({error})')
                break
            seeds = cell['seeds']
        available = max(0, len(seeds) - offset)
        if available < games:
            changes.append(f'{key}: 1x review seeds exhausted; {min(available, games)}/{games} games')
            games = available
        if games:
            definition = dict(task, zombie_count_multiplier=1., seeds=list(seeds[offset:offset + games]))
            definition.pop('demo_start', None)  # A recorded non-1x board cannot be replayed at 1x.
            reviews.append(dict(base=group['base'], task_id=group['task_id'], multiplier=1.,
                task_definition=definition, manifest=group['manifest'], games=games,
                offset=offset, offset_key=key, seed_offset=0, review_of=group['label'],
                label=group['label'] + f'_review1_o{offset}', save_candidates=group.get('save_candidates', 0)))
    return reviews


def plan_round(state: dict, tasks: dict[str, dict], names: dict[str, float], values: list[float],
               manifests: dict[str, Path], budget: int, per_group: int,
               teacher: dict | None = None, start_seed_ledger: Path | None = None,
               renew_size: int = START_SEED_BLOCK,
               save_candidates_levels=frozenset(),
               random_deck_share: float = RANDOM_DECK_SHARE,
               demo_start_share: float | None = None,
               live_catalog: dict | None = None,
               review_share: float = REVIEW_SHARE) -> tuple[list[dict], list[str]]:
    changes = isolate_cells(state, values)
    changes += adjust(state['cells'], values)
    specialists = {base: cell for base, cell in state['cells'].items() if cell.get('opening_specialist')}
    ordinary = {base: cell for base, cell in state['cells'].items()
                if base not in specialists and (not specialists or cell['level'] != 7 or cell.get('live193'))
                and (not cell.get('live193') or base in (live_catalog or {}))
                and (live_catalog is None or cell.get('live193') or
                     not (cell.get('random_deck_cell') or cell.get('live193_grouped')))}
    reserved = 24 * len(specialists)
    if reserved > budget:
        raise ValueError('round budget cannot cover opening specialists')
    completed = [group for group in state.get('completed_groups', [])
                 if group.get('status', 'done') == 'done' and not group.get('review_of')]
    completed_betas = {base: {group.get('beta') for group in completed if group['base'] == base}
                       for base in specialists}
    special_picks = [(base, 1.0, 12 * (2 - len(completed_betas[base])))
                     for base in sorted(specialists) if len(completed_betas[base]) < 2]
    remaining = budget - sum(group['games'] for group in completed)
    pending_special = sum(games for _, _, games in special_picks)
    if specialists and pending_special > remaining:
        return [], changes + ['opening specialists start next round: partial-round budget already spent']
    ordinary_budget = remaining - pending_special
    if demo_start_share is None:
        # Default remains the deploy16 planner, including its ordering.
        picks = special_picks + allocate(ordinary, values, ordinary_budget,
                                        per_group, state['round'] + 1, teacher,
                                        random_deck_cap=int(budget * random_deck_share))
    else:
        def category(cell):
            return 'random' if cell.get('random_deck_cell') else 'demo' if (cell.get('archive') is not None or cell.get('demo_archive') is not None) else 'multiplier'
        completed_games = {kind: sum(group['games'] for group in completed
                           if category(state['cells'].get(group['base'], state.get('retired_cells', {}).get(group['base'], group))) == kind)
                           for kind in ('random', 'demo')}
        caps = dict(random=max(0, int(budget * random_deck_share) - completed_games['random']),
                    demo=max(0, int(budget * demo_start_share) - completed_games['demo']))
        picks = list(special_picks)
        # Reserve random decks before the large demo floor can consume their budget.
        # Remaining non-demo budget goes to multiplier cells, including any unused quota.
        for kind in ('random', 'demo', 'multiplier'):
            subset = {base: cell for base, cell in ordinary.items() if category(cell) == kind}
            quota = min(ordinary_budget, caps[kind]) if kind in caps else ordinary_budget
            selected = allocate(subset, values, quota, per_group, state['round'] + 1, teacher)
            picks.extend(selected)
            ordinary_budget -= sum(games for _, _, games in selected)
    plan = []
    for base, multiplier, games in picks:
        cell = state['cells'][base]
        if cell.get('live193'):
            from pvz_live_curriculum import materialize
            task, extra = materialize(base, cell, live_catalog, state['round'] + 1, state)
            offset = state['offsets'].get(base, 0)
            while offset + games > len(cell['seeds']):
                try:
                    block = renew_start_seeds(cell, start_seed_ledger, renew_size)
                except (SeedPoolExhausted, ValueError, OSError) as error:
                    # Pool exhausted, ledger corrupt, or ledger directory unwritable: only this
                    # cell misses out; the rest of the round still runs (task98's lesson).
                    changes.append(f'{base}: seed renewal failed ({error})')
                    break
                else:
                    changes.append(f'{base}: seeds renewed +{block[1] - block[0]} from '
                                   f'[{block[0]},{block[1]}) (table {len(cell["seeds"])}, '
                                   f'segments {len(cell.get("seed_blocks") or [])})')
            if offset + games > len(cell['seeds']):
                changes.append(f'{base}: seeds exhausted (offset {offset} + {games} > '
                               f'{len(cell["seeds"])}); group skipped')
                continue
            seed_offset = None
            if not extra.get('demo_archive'):
                # Only this group's window goes into --task-json: the cell keeps accumulating
                # blocks, and the whole table would eventually exceed Linux's 128 KiB argument cap.
                task['seeds'] = list(cell['seeds'][offset:offset + games])
                seed_offset = 0
            label = label_for(state['round'] + 1, base, multiplier,
                              start=extra.get('demo_decision')) + f'_o{offset}'
            item = dict(base=base, multiplier=multiplier, task_id=task['task_id'], offset=offset,
                        seed_offset=seed_offset, offset_key=base, games=games,
                        manifest=str(next(iter(manifests.values()))),
                        label=label, task_definition=task,
                        save_candidates=save_candidates_for(cell, task, save_candidates_levels), **extra)
            if extra.get('demo_archive'):
                item['future_seeds'] = cell['seeds'][offset:offset + games]
            plan.append(item)
            continue
        start = cell.get('archive') is not None
        try:
            task_id = cell['task_id'] if start else task_id_for(base, multiplier, tasks, names)
            seeds = cell['seeds'] if start else tasks[task_id]['seeds']
            offset = state['offsets'].get(task_id, 0)
            if not isinstance(seeds, list) or type(offset) is not int or offset < 0:
                raise ValueError('invalid seeds/offset')
            manifest = str(manifests[task_id])
            if start and (offset + games <= len(seeds) or start_seed_ledger is not None):
                archive = cell['archive']
                if not isinstance(archive, str):
                    raise ValueError('invalid archive')
                try:
                    meta = read_json(Path(archive) / 'meta.json')
                except FileNotFoundError:
                    raise ValueError(f'configuration invalid: demo_start archive missing: {archive}') from None
                if not meta.get('verified'):
                    raise ValueError(f'configuration invalid: demo_start archive is not verified: {archive}')
                demo_seed = meta['task']['seed']
                if demo_seed not in tasks[task_id]['seeds']:
                    raise ValueError(f'configuration invalid: demo_start seed {demo_seed} is outside task seeds')

        except (OSError, KeyError, TypeError, ValueError) as error:
            changes.append(retire_cell(state, base, error))
            continue
        while start and offset + games > len(seeds) and start_seed_ledger is not None:
            # task99: a start cell's future seeds are renewable. Renew before giving the cell up —
            # the first block in the manifest is only segment one; the offset keeps counting from
            # the head of the table, so old groups and their labels stay comparable.
            try:
                block = renew_start_seeds(cell, start_seed_ledger, renew_size)
            except (SeedPoolExhausted, ValueError, OSError) as error:
                # 池耗尽、账本损坏/池不符、账本目录不可写：都只是这一格领不到，不能让 qc 停在
                # plan 阶段（task98 的教训：单点异常不该终止整个采集）。错误原文进日志。
                changes.append(f'{task_id}: seed renewal failed ({error})')
                break
            else:
                seeds = cell['seeds']
                changes.append(f'{task_id}: seeds renewed +{block[1] - block[0]} from '
                               f'[{block[0]},{block[1]}) (table {len(seeds)}, '
                               f'segments {len(cell.get("seed_blocks") or [])})')
        if offset + games > len(seeds):
            if start:
                cell['retired_reason'] = 'seeds exhausted'
                state.setdefault('retired_cells', {})[base] = state['cells'].pop(base)
            changes.append(f'{task_id}: seeds exhausted (offset {offset} + {games} > {len(seeds)}); '
                           + ('start cell retired' if start else 'group skipped'))
            continue
        item = dict(base=base, multiplier=multiplier, task_id=task_id, offset=offset,
                    games=games, manifest=manifest,
                    label=label_for(state['round'] + 1, base, multiplier,
                                    start=int(multiplier) if start else None) + f'_o{offset}',
                    save_candidates=save_candidates_for(cell, tasks.get(task_id),
                                                        save_candidates_levels))
        if start:
            item.update(demo_archive=cell['archive'], demo_decision=int(multiplier),
                        future_seeds=seeds[offset:offset + games])
        if cell.get('opening_specialist'):
            for index, beta in enumerate((0.0, 0.5)):
                if beta not in completed_betas[base]:
                    plan.append(dict(item, games=12, offset=offset,
                                     beta=beta, label=item['label'] + f'_b{index}'))
                    offset += 12
        else:
            plan.append(item)
    plan += review_groups(state, completed + plan, tasks, review_share,
                          start_seed_ledger, renew_size, changes)
    return plan, changes


def _run_group(args, state, item, round_index, per_worker):
    pipeline = getattr(args, 'pipeline', None)
    state['offsets'][item.get('offset_key', item['task_id'])] = item['offset'] + item['games']
    group_cell = state['cells'][item['base']]
    if group_cell.get('live193') and item.get('demo_archive'):
        group_cell['archive_cursor'] = group_cell.get('archive_cursor', 0) + 1
    save_state(args.state, state)          # 先占偏移，崩溃也不会重采
    available = mem_available_bytes()
    if pipeline is not None:
        # reservation() needs the probe result even for the first group after restart.
        pipeline.per_worker_bytes = per_worker
    background = pipeline.reservation() if pipeline is not None else 0
    upper = pipeline.foreground_limit() if pipeline is not None else args.workers
    workers = worker_count(available, per_worker, upper, background)
    if pipeline is not None:
        # search_play can grow from `workers` to `upper` inside this group (task138).
        # Claim that ceiling so evaluation cannot spend its future growth slots.
        pipeline.foreground_workers = upper
    per_worker_shown = f'{per_worker / GIB:.2f} GiB' if per_worker > 0 else 'unmeasured'
    log_line(args.log, f'round {round_index} {item["label"]}: workers={workers}/{args.workers} '
                       f'(MemAvailable {"n/a" if available is None else f"{available / GIB:.1f} GiB"}, '
                       f'per-worker {per_worker_shown}, background headroom {background / GIB:.2f} GiB) '
                       f'save={item.get("save_candidates", 0)}')
    cell = state['cells'][item.get('adaptation_base', item['base'])]
    adaptation_value = item.get('adaptation_value', item['multiplier'])
    student = student_rate(cell, adaptation_value)
    starved = student is not None and student < args.beta_threshold
    if starved and cell.get('mixed_since_probe', 0) >= max(1, args.probe_interval) - 1:
        beta = 0.0    # 探针：每 probe_interval 组保留一个 β=0 组，测真实学生胜率
    elif starved:
        beta = args.starved_beta
    else:
        beta = 0.0
    beta = item.get('beta', beta)
    item['beta'] = beta
    command = command_for(args, Path(item['manifest']), item['task_id'], item['offset'],
                          item['games'], item['label'], workers=workers, beta=beta,
                          demo_archive=item.get('demo_archive'),
                          demo_decision=item.get('demo_decision'),
                          future_seeds=item.get('future_seeds'),
                          save_candidates=item.get('save_candidates', 0),
                          upper_workers=upper, per_worker_bytes=per_worker,
                          headroom_bytes=background, task_definition=item.get('task_definition'),
                          seed_offset=item.get('seed_offset'))
    started = time.time()
    item.update(collection_started=started, valid_games=0, student=str(args.rollout_policy),
                student_round=state.get('student_round', 0), workers=workers)
    try:
        result = retry_io(subprocess.run)(command, cwd=ROOT)
    finally:
        item['collection_finished'] = time.time()
        if pipeline is not None:
            pipeline.foreground_workers = 0
    seconds = time.time() - started
    folder = args.out / f'{item["label"]}_{item["games"]}'
    wins = games = 0
    rows = []
    if (folder / 'games.jsonl').exists():
        rows = [json.loads(line) for line in retry_io((folder / 'games.jsonl').read_text)().splitlines() if line.strip()]
        games = len(rows)
        wins = sum(1 for row in rows if row.get('won'))
    item['valid_games'] = sum(type(row.get('won')) is bool for row in rows)
    item['wins'] = wins
    if games != item['games'] or any(type(row.get('won')) is not bool for row in rows):
        raise ValueError(f'incomplete/invalid game results: {games}/{item["games"]}')
    metrics = getattr(args, 'metrics', None)
    if metrics is not None:
        args.metrics_games += games
        args.metrics_decisions += sum(row.get('decisions', 0) for row in rows)
        try:
            metrics.update(games=args.metrics_games, decisions=args.metrics_decisions, round=round_index)
        except Exception as error:
            log_line(args.log, f'metrics update skipped: {error}')
    log_line(args.log, f'round {round_index} {item["label"]} {item["task_id"]} '
                       f'mult={item["multiplier"]:g}'
                       + (f' start={item["demo_decision"]}' if 'demo_decision' in item else '') +
                       f' offset={item["offset"]} games={games} wins={wins} '
                       f'rate={(wins / games if games else 0):.2f} beta={beta:g} '
                       f'rc={result.returncode} {seconds:.0f}s')
    if result.returncode != 0:
        raise RuntimeError(f'{item["label"]}: search_play exited {result.returncode}')
    if item.get('review_of'):
        return  # Reviews feed training only; do not update ladder windows or probe counters.
    bucket = cell['windows'].setdefault(
        str(adaptation_value), {'recent': [], 'wins': 0, 'games': 0})
    if beta > 0:
        # β 混合组的胜负不是学生的胜率（一部分步是搜索在走）：不进学生窗口，另存 mixed；
        # 否则 adjust/allocate/student_rate 会把混合胜率当学生胜率——全败格被误升档、
        # 可教差距被算小、student_rate 在阈值附近来回振荡（task59 修正 2b）。
        target = bucket.setdefault('mixed', {'recent': [], 'wins': 0, 'games': 0})
        cell['mixed_since_probe'] = cell.get('mixed_since_probe', 0) + 1
    else:
        target = bucket
        cell['mixed_since_probe'] = 0
    target['recent'] = (target['recent'] + [1 if row.get('won') else 0
                                            for row in rows])[-WINDOW:]
    target['wins'] += wins
    target['games'] += games
    state['cells'][item['base']]['last_round'] = round_index
    cell['last_round'] = round_index


def pipeline_action(args, action):
    pipeline = getattr(args, 'pipeline', None)
    if pipeline is None:
        return
    try:
        getattr(pipeline, action)()
    except StateWriteError:
        raise
    except (Exception, SystemExit) as error:
        log_line(args.log, f'background {action} isolated; collector retained: {type(error).__name__}: {error}')
        save_state(args.state, pipeline.state)


def run_round(args, state: dict, plan: list[dict], changes: list[str], per_worker: int = 0) -> bool | None:
    """Group failures do not stop other cells; offsets are durable before collection."""
    round_index = state['round'] + 1
    health = getattr(args, 'game_health', None) or GameHealth(args)
    if state.get('zero_game_groups', 0) >= ZERO_GAME_GROUP_LIMIT:
        health.pause('consecutive groups produced zero valid games before restart')
    for item in plan:
        if not health.wait():
            return False
        pipeline_action(args, 'poll')
        if not health.wait():
            return False
        if getattr(args, 'stop_flag', STOP_FLAG).exists():
            log_line(args.log, f'round {round_index}: stop flag seen before {item["label"]}; stopping at group boundary')
            return False
        try:
            _run_group(args, state, item, round_index, per_worker)
            item['status'] = 'done'
            state['zero_game_groups'] = 0
        except StateWriteError:
            raise
        except (Exception, SystemExit) as error:
            item.update(status='failed', error=f'{type(error).__name__}: {error}',
                        collection_finished=time.time())
            log_line(args.log, f'round {round_index} {item["label"]}: group skipped: {item["error"]}')
            if item.get('valid_games', 0) and item['base'] in state['cells']:
                state['cells'][item['base']]['last_round'] = round_index
            state['zero_game_groups'] = (state.get('zero_game_groups', 0) + 1
                                         if item.get('valid_games') == 0 else 0)
        state.setdefault('completed_groups', []).append(dict(item))
        save_state(args.state, state)
        if state['zero_game_groups'] >= ZERO_GAME_GROUP_LIMIT:
            health.pause(f'{state["zero_game_groups"]} consecutive groups produced zero valid games')
            if not health.wait():
                return False
            state['zero_game_groups'] = 0
            save_state(args.state, state)
            return None
    state['round'] = round_index
    groups = list({group['label']: group for group in state.pop('completed_groups', [])}.values())
    state['log'].append(dict(round=round_index, groups=groups, changes=changes))
    save_state(args.state, state)
    completed = [group for group in groups if group.get('status', 'done') == 'done']
    log_line(args.log, f'round {round_index} done: {len(completed)} groups, '
                       f'{sum(group["games"] for group in completed)} games, changes={len(changes)}'
                       + ((' [' + '; '.join(changes) + ']') if changes else ''))
    if completed and not getattr(args, 'stop_flag', STOP_FLAG).exists():
        pipeline_action(args, 'enqueue')
    return True


def print_plan(state: dict, plan: list[dict], changes: list[str]) -> None:
    print(f"{'cell':<34}{'level':>6}{'win%':>7}{'n':>6}{'games':>7}  label")
    for item in plan:
        cell = state['cells'][item['base']]
        win_rate = rate(cell, item['multiplier'])
        shown = f'{win_rate * 100:.1f}' if win_rate is not None else '-'
        games = len(cell['windows'].get(str(item['multiplier']), {}).get('recent') or [])
        level = f'{item["multiplier"]:g}' + ('s' if 'demo_decision' in item else '')
        print(f'{item["base"]:<34}{level:>6}{shown:>7}{games:>6}{item["games"]:>7}  {item["label"]}')
    print(f'total groups {len(plan)}, games {sum(item["games"] for item in plan)}')
    if changes:
        print('multiplier changes:')
        for change in changes:
            print('  ' + change)


def refresh_live_curriculum(args, state, tasks):
    directory = getattr(args, 'gpt_import_dir', None)
    if directory is None:
        return None
    from pvz_live_curriculum import refresh, cold_teacher_rates
    evidence = getattr(args, 'gpt_cold_rates', None) or args.teacher_rates
    cold = {int(key.split(':')[1]): value for key, value in
            load_teacher_rates(evidence).items() if key.startswith('level:')}
    cold.update(cold_teacher_rates(state, tasks))
    root = getattr(args, 'gpt_archive_root', None) or directory.parents[2]
    catalog = refresh(state, tasks, directory, root,
                      [ROOT / 'configs/deploy17_deck_holdouts.json'], cold)
    log_line(args.log, f'task193 live pool: {sum(v["kind"] == "random" for v in catalog.values())} levels, '
                      f'{sum(v["kind"] == "demo" for v in catalog.values())} demo deck groups, '
                      f'{len(cold)} levels with cold teacher evidence (bootstrap {evidence})')
    return catalog


def collect_rounds(args, state, tasks, names, values, manifests, teacher):
    per_worker = -1
    completed = 0
    failures = 0
    while completed < args.rounds:
        if args.stop_flag.exists():
            log_line(args.log, 'stop flag seen at round boundary')
            break
        try:
            # Also the recovery barrier after an IO failure: no collection may run
            # with an uncommitted reservation left from a previous attempt.
            save_state(args.state, state)
            live_catalog = refresh_live_curriculum(args, state, tasks)
            plan, changes = plan_round(state, tasks, names, values, manifests, args.round_games,
                                       args.games_per_group, teacher,
                                       start_seed_ledger=getattr(args, 'start_seed_ledger', None),
                                       save_candidates_levels=args.save_candidates_levels,
                                       random_deck_share=args.random_deck_share,
                                       demo_start_share=getattr(args, 'demo_start_share', None), live_catalog=live_catalog,
                                       review_share=getattr(args, 'review_share', REVIEW_SHARE))
            print_plan(state, plan, changes)
            if per_worker < 0:
                if args.stop_flag.exists():
                    break
                per_worker = startup_worker_estimate(args, plan)
            finished = run_round(args, state, plan, changes, per_worker)
            if finished is False:
                break
            if finished is None:
                continue
            completed += 1
            failures = 0        # progress resets the retry budget
        except ROUND_FATAL_ERRORS as error:
            log_line(args.log, f'FATAL round boundary: {type(error).__name__}: {error}')
            raise
        except (Exception, SystemExit) as error:
            # KeyboardInterrupt and process termination are deliberately not caught.
            failures += 1
            if failures > ROUND_RETRY_LIMIT:
                log_line(args.log, f'FATAL round boundary failed {failures} times in a row, giving up: '
                                   f'{type(error).__name__}: {error}')
                raise
            delay = min(ROUND_RETRY_SECONDS * 2 ** (failures - 1), ROUND_RETRY_MAX_SECONDS)
            log_line(args.log, f'ERROR round boundary paused; retry {failures}/{ROUND_RETRY_LIMIT} '
                               f'in {delay}s: {type(error).__name__}: {error}')
            time.sleep(delay)




def main() -> None:
    # task34: die after the workers and training, but before the WSL access path.
    set_oom_score_adj(OOM_SCORE_CONTROLLER)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/stage4/multiplier_panel.json')
    parser.add_argument('--extra-manifest', type=Path, action='append', default=[],
                        help='additional registered training panel (task137 random-deck cells); repeatable')
    parser.add_argument('--random-deck-share', type=float, default=RANDOM_DECK_SHARE,
                        help='max share of the round budget for random-deck cells (default 0.15)')
    parser.add_argument('--gpt-import-dir', type=Path,
                        help='task193: reread the existing task105 import indexes at each round')
    parser.add_argument('--gpt-archive-root', type=Path, help='root for relative task105 archive paths')
    parser.add_argument('--gpt-cold-rates', type=Path,
                        help='existing cold all-teacher level rates; defaults to --teacher-rates; '
                             'actual cold beta=1 state outcomes take precedence')
    parser.add_argument('--demo-start-manifest', type=Path, default=ROOT / 'configs/human_demo_tasks.json',
                        help='task manifest whose demo_start entries become start cells (task71); '
                             'missing file = no start cells')
    parser.add_argument('--state', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--log', type=Path, default=DEFAULT_LOG)
    parser.add_argument('--stop-flag', type=Path, default=STOP_FLAG,
                        help='checked only between groups; independent runs should use an independent flag')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/hard')
    parser.add_argument('--frames', type=Path, default=DEFAULT_FRAMES)
    parser.add_argument('--obs-v3', action=argparse.BooleanOptionalAction, default=False,
                        help='write rich token attachment with each collected episode (deploy18 enables this)')
    parser.add_argument('--native-games', action=argparse.BooleanOptionalAction, default=True,
                        help='use native fog games in collection/selection/promotion (deploy19)')
    parser.add_argument('--rollout-policy', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--rollout-policy-gpt', type=Path,
                        help='GPT behavior-clone prior for fog/roof/boss levels; default --rollout-policy')
    parser.add_argument('--static-value', type=Path, default=DEFAULT_STATIC_VALUE)
    parser.add_argument('--value', type=Path, default=DEFAULT_VALUE)
    parser.add_argument('--value-roof', type=Path,
                        help='leaf value net for roof levels 41-50 (including the boss); default --value')
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--resource-dir', type=Path, default=DEFAULT_RESOURCES)
    parser.add_argument('--workers', type=int, default=16,
                        help='总游戏 worker 上限；后台队列非空时扣除 --min-eval-units，内存再限流')
    parser.add_argument('--min-eval-units', type=int, default=4,
                        help='门禁/挑选预留单元数；默认4，最多 workers-1（workers=1仍为1+1）；'
                             '提前预留内存，资源不足时等待而不缩成1单元')
    parser.add_argument('--python', type=Path, default=DEFAULT_PYTHON,
                        help='跑 search_play 的解释器（默认 ~/.venvs/ml/bin/python3）')
    parser.add_argument('--round-games', type=int, default=240)
    parser.add_argument('--games-per-group', type=int, default=48)
    parser.add_argument('--review-share', type=float, default=REVIEW_SHARE,
                        help='非1倍任务在原配额之外追加的1倍复习份额，默认0.25，0关闭')
    parser.add_argument('--demo-start-share', type=float, default=None,
                        help='optional demo-start game cap per round; reserves random-deck quota before demo floors, remainder to multiplier cells')
    parser.add_argument('--teacher-rates', type=Path, default=DEFAULT_TEACHER_RATES,
                        help='teacher win-rate evidence json (level/base -> rate); missing file = all unknown')
    parser.add_argument('--beta-threshold', type=float, default=0.1,
                        help='学生最近胜率低于该值的格改用 β 混合（DAgger beta-mixing）')
    parser.add_argument('--starved-beta', type=float, default=0.5,
                        help='β 混合格的 dagger-beta：该比例的步由搜索执行，标签仍全部来自搜索')
    parser.add_argument('--probe-interval', type=int, default=3,
                        help='全败格每这么多个组保留一个 β=0 探针组（测真实学生胜率），其余 β 混合')
    parser.add_argument('--save-candidates-levels', default='31-40', metavar='LEVELS',
                        help='这些关的采集格给 search_play 传 --save-candidates 1（task120：让搜索'
                             '评估"攒够再种"的卡）；清单任务里写 save_candidates 可逐格覆盖。'
                             '默认雾夜区 31-40，空串 = 全关关闭')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--init-glob', action='append', default=None,
                        help='用于初始化胜率/已用偏移的 config.json glob（可重复）')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--recipe', type=Path, default=ROOT / 'recipes/dw4i.json',
                        help='training recipe for the collection student; supplies every pack row option')
    parser.add_argument('--alternate-recipe', type=Path,
                        help='recipe B candidates, sharing recipe A pack row options')
    parser.add_argument('--b-init', type=Path, default=DEFAULT_B_INIT)
    parser.add_argument('--b-initial-epochs', type=int, default=8)
    parser.add_argument('--b-epochs', type=int, default=4)
    parser.add_argument('--training-parallelism', type=int, choices=(1, 2), default=2,
                        help='maximum simultaneous trainers, frozen per job; 1 forces serial admission')
    parser.add_argument('--opening-manifest', type=Path,
                        help='fixed 1.0x cold-start specialists, 12 beta=0 + 12 beta=0.5 per cell')
    parser.add_argument('--promotion-panel', type=Path,
                        default=ROOT / 'configs/gate_v5_panel.json',
                        help='registered unified v2 source manifest for the mandatory paired promotion gate')
    parser.add_argument('--seed-ledger', type=Path, default=Path(SEED_LEDGER_DEFAULT),
                        help='out-of-git gate seed ledger (task73b; v2 gates allocate fresh-seed blocks here)')
    parser.add_argument('--start-seed-ledger', type=Path, default=Path(START_SEED_LEDGER_DEFAULT),
                        help='out-of-git ledger for renewed demo-start future seed blocks (task99)')
    parser.add_argument('--restore-start-cell', action='append', default=[], metavar='TASK_ID',
                        help='reactivate a start cell retired for seed exhaustion, keeping its offset '
                             'and renewed segments (task99); repeatable, one-off')
    args = parser.parse_args()
    try:
        load_recipe(args.recipe, 'training')
        if args.alternate_recipe:
            load_recipe(args.alternate_recipe, 'training')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.min_eval_units < 1:
        parser.error('--min-eval-units must be positive')
    if args.b_initial_epochs < 1 or args.b_epochs < 1:
        parser.error('B epoch counts must be positive')
    if args.round_games < 1 or args.games_per_group < 1 or args.rounds < 1:
        parser.error('rounds and game budgets must be positive')
    if not 0.0 <= args.review_share <= 1.0:
        parser.error('--review-share must be within 0..1')
    try:
        args.save_candidates_levels = parse_levels(args.save_candidates_levels)
    except ValueError as error:
        parser.error(str(error))
    for name in ('manifest', 'state', 'log', 'out', 'frames', 'rollout_policy', 'static_value',
                 'value', 'executable', 'resource_dir', 'recipe', 'python', 'stop_flag',
                 'promotion_panel', 'teacher_rates', 'demo_start_manifest',
                 'gpt_import_dir', 'gpt_archive_root', 'gpt_cold_rates'):
        value = getattr(args, name)
        if value is not None and not value.is_absolute():
            parser.error(f'all paths must be absolute: {name}')
    if not args.seed_ledger.is_absolute():
        parser.error('seed ledger paths must be absolute')
    if not args.start_seed_ledger.is_absolute():
        parser.error('--start-seed-ledger must be absolute')
    for extra in args.extra_manifest:
        if not extra.is_absolute():
            parser.error('all paths must be absolute: extra_manifest')
        if not extra.is_file():
            parser.error(f'--extra-manifest does not exist: {extra}')
    if not 0.0 <= args.random_deck_share <= 1.0:
        parser.error('--random-deck-share must be within 0..1')
    if args.demo_start_share is not None and not 0.0 <= args.demo_start_share <= 1.0:
        parser.error('--demo-start-share must be within 0..1')
    tasks, values, names, manifests = load_panel(args.manifest, args.extra_manifest)
    extra_bases = {cell_of(task_id, names) for task_id, task in tasks.items()
                   if task.get('random_deck_cell')}
    if args.opening_manifest:
        opening = load_tasks(args.opening_manifest)
        tasks.update(opening)
        manifests.update({task_id: args.opening_manifest for task_id in opening})
    start_cells, start_manifests = ({}, {})
    if args.demo_start_manifest.exists():
        start_cells, start_manifests = load_demo_start_cells(args.demo_start_manifest)
        tasks.update({task_id: task for task_id, task in load_tasks(args.demo_start_manifest).items()
                      if task_id in start_cells})
        manifests.update(start_manifests)
    teacher = load_teacher_rates(args.teacher_rates)
    if not teacher:
        log_line(args.log, f'no teacher-rate evidence at {args.teacher_rates}; ranking by student rate only')
    patterns = args.init_glob or [str(ROOT / 'artifacts/hard/*/config.json'),
                                  str(ROOT / 'artifacts/task28/*/config.json')]
    if args.state.exists():
        state = read_json(args.state)
        if (not isinstance(state, dict) or type(state.get('round')) is not int or state['round'] < 0
                or not isinstance(state.get('cells'), dict) or not isinstance(state.get('offsets'), dict)
                or not isinstance(state.get('log'), list) or not isinstance(state.get('training_jobs', []), list)):
            parser.error('invalid global curriculum state structure')
        for line in isolate_cells(state, values):
            log_line(args.log, line)
        # 已有状态按当前训练集合增删格子；n10q 没有旧局，除非显式给 --init-glob 否则从空窗口起步。
        windows = (init_from_runs(args.init_glob, names, extra_bases)[0]
                   if args.init_glob else {})
        for line in reconcile_cells(state, tasks, names, windows):
            log_line(args.log, line)
        for line in reconcile_start_cells(state, start_cells, windows,
                                          restore=tuple(args.restore_start_cell)):
            log_line(args.log, line)
        for base in args.restore_start_cell:
            if base not in state['cells']:
                parser.error(f'--restore-start-cell {base}: not a retired start cell in {args.state}')
            manifests.setdefault(base, args.demo_start_manifest)
    else:
        windows, offsets = init_from_runs(patterns, names, extra_bases)
        state = make_state(tasks, names, windows, offsets, str(args.manifest), start_cells)
    state['gate_version'] = GATE_VERSION
    if args.dry_run:
        # 预览也要走续领逻辑，但不能真花种子：在账本副本上领，块与生产下次会拿到的一致。
        with tempfile.TemporaryDirectory() as preview_dir:
            ledger = Path(preview_dir) / args.start_seed_ledger.name
            if args.start_seed_ledger.exists():
                ledger.write_bytes(args.start_seed_ledger.read_bytes())
            live_catalog = refresh_live_curriculum(args, state, tasks)
            plan, changes = plan_round(state, tasks, names, values, manifests, args.round_games,
                                       args.games_per_group, teacher, start_seed_ledger=ledger,
                                       save_candidates_levels=args.save_candidates_levels,
                                       random_deck_share=args.random_deck_share,
                                       demo_start_share=getattr(args, 'demo_start_share', None), live_catalog=live_catalog,
                                       review_share=args.review_share)
        print_plan(state, plan, changes)
        for item in plan:
            print('collect: ' + shlex.join(command_for(args, Path(item['manifest']), item['task_id'],
                                                       item['offset'], item['games'], item['label'],
                                                       beta=item.get('beta', 0.0),
                                                       demo_archive=item.get('demo_archive'),
                                                       demo_decision=item.get('demo_decision'),
                                                       future_seeds=item.get('future_seeds'),
                                                       save_candidates=item.get('save_candidates', 0),
                                                       task_definition=item.get('task_definition'),
                                                       seed_offset=item.get('seed_offset'))))
        # The pack command this controller would run at the end of the round: reviewable before
        # a round is spent, and the only place the recipe's row options become a real command.
        labels = pack_labels(state, ','.join(load_recipe(args.recipe, 'training')['labels']))
        if labels:
            print('pack: ' + ' '.join(pack_command(args, labels)))
        print('gate7: two candidates A/B, one seed each, shared incumbent; no selection panel')
        return
    args.metrics = None
    try:
        args.metrics = Metrics("curriculum_queue").start()
        args.metrics.update(games=0, decisions=0, round=state["round"])
    except Exception as error:
        log_line(args.log, f'metrics unavailable: {error}')
    args.metrics_games = args.metrics_decisions = 0
    args.pipeline = TrainingPipeline(args, state)
    try:
        # Recover the saved last round if a previous controller died before enqueuing it.
        if state['round'] and state['log']:
            pipeline_action(args, 'enqueue')
        collect_rounds(args, state, tasks, names, values, manifests, teacher)
    finally:
        # Finite invocations and stop flags drain already saved snapshots; games are never
        # killed to switch students, and a final training failure remains visible in state.
        try:
            pipeline_action(args, 'close')
        finally:
            if args.metrics is not None:
                try:
                    args.metrics.close()
                except Exception as error:
                    log_line(args.log, f'metrics close skipped: {error}')


if __name__ == '__main__':
    main()
