"""Low-cost process-tree telemetry; no torch/psutil dependency or CLI switches.

CPU percent uses one core = 100%. GPU samples are device-wide (not process
attribution); unavailable probes remain null. Progress counters are cumulative.
"""
from __future__ import annotations

import atexit
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

DEFAULT_DIRECTORY = Path.home() / 'pvz-logs/metrics'
INTERVAL = 30.0
WINDOW = 600.0
RETENTION_DAYS = 7


def _read_numbers(path):
    try:
        return {k.rstrip(':'): int(v.split()[0]) for k, v in
                (line.split(':', 1) for line in path.read_text().splitlines())}
    except (OSError, ValueError):
        return {}


def pressure(kind):
    try:
        return {parts[0]: {k: float(v) for k, v in
                          (part.split('=') for part in parts[1:])}
                for parts in (line.split() for line in
                              Path('/proc/pressure', kind).read_text().splitlines())}
    except (OSError, ValueError):
        return None


def process_tree(pid):
    """Return root/descendant RSS and identity-keyed CPU clocks from /proc."""
    table = {}
    try:
        page = os.sysconf('SC_PAGE_SIZE')
        ticks = os.sysconf('SC_CLK_TCK')
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                table[int(entry.name)] = (int(fields[1]), int(fields[21]) * page,
                                         (int(fields[11]) + int(fields[12])) / ticks,
                                         fields[19])
            except (OSError, ValueError, IndexError):
                continue
    except (OSError, ValueError):
        return None, None, {}
    children = {}
    for child, record in table.items():
        children.setdefault(record[0], []).append(child)
    stack, clocks, rss = [pid], {}, 0
    while stack:
        child = stack.pop()
        if child not in table:
            continue
        record = table[child]
        clocks[(child, record[3])] = record[2]
        if child != pid:
            rss += record[1]
        stack.extend(children.get(child, ()))
    return table[pid][1] if pid in table else None, rss, clocks


def gpu_sample():
    executable = shutil.which('nvidia-smi')
    if executable is None:
        return None
    try:
        result = subprocess.run([executable,
            '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'], capture_output=True, text=True,
            timeout=3, check=True)
        devices = []
        for line in result.stdout.splitlines():
            index, uuid, util, used, total = [part.strip() for part in line.split(',')]
            devices.append(dict(index=int(index), uuid=uuid, utilization_percent=float(util),
                                memory_used_bytes=int(used) * 1024 ** 2,
                                memory_total_bytes=int(total) * 1024 ** 2))
        return devices or None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def gpu_utilization(row):
    devices = row.get('gpu')
    return max(d['utilization_percent'] for d in devices) if devices else None


class ActiveWindow:
    """Time-weighted, fully observed trailing 10 minutes; unknown GPU breaks it."""
    def __init__(self):
        self.intervals = deque()
        self.last = None
        self.last_warn = float('-inf')

    def observe(self, now, utilization):
        if self.last is not None:
            start, previous = self.last
            # An interrupted sampler is not evidence about the unobserved gap.
            if now - start <= INTERVAL * 2 and previous is not None and utilization is not None:
                self.intervals.append((start, now, utilization > 50))
            else:
                self.intervals.clear()
        self.last = (now, utilization)
        cutoff = now - WINDOW
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()
        covered = sum(end - max(start, cutoff) for start, end, _ in self.intervals)
        active = sum(end - max(start, cutoff) for start, end, busy in self.intervals if busy)
        ratio = active / covered if covered else None
        warn = covered >= WINDOW - 1e-6 and ratio < .5 and now - self.last_warn >= WINDOW
        if warn:
            self.last_warn = now
        return ratio, covered, warn


class Metrics:
    def __init__(self, name, *, training=False, directory=DEFAULT_DIRECTORY, interval=INTERVAL, progress_source=None):
        self.name = re.sub(r'[^\w.-]', '_', name)
        self.pid = os.getpid()
        self.training = training
        self.directory = Path(directory)
        self.path = self.directory / f'{self.name}-{self.pid}.jsonl'
        self.interval = interval
        self._lock = threading.Lock()
        self._sample_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._progress = {}
        self._progress_source = progress_source
        self._previous = None
        self._day = None
        self._window = ActiveWindow()
        self._closed = False

    def update(self, **counters):
        with self._lock:
            self._progress.update(counters)

    def advance(self, **increments):
        with self._lock:
            for name, value in increments.items():
                self._progress[name] = self._progress.get(name, 0) + value

    def _rotate(self, stamp):
        day = datetime.fromtimestamp(stamp, timezone.utc).date().isoformat()
        self.directory.mkdir(parents=True, exist_ok=True)
        if self._day and day != self._day and self.path.exists():
            self.path.rename(self.path.with_name(f'{self.name}-{self.pid}.{self._day}.jsonl'))
        if day != self._day:
            cutoff = stamp - RETENTION_DAYS * 86400
            for path in self.directory.glob('*.jsonl'):
                try:
                    if path != self.path and path.stat().st_mtime < cutoff:
                        path.unlink()
                except OSError:
                    pass
        self._day = day

    def sample(self):
        with self._sample_lock:
            stamp, monotonic = time.time(), time.monotonic()
            rss, children_rss, clocks = process_tree(self.pid)
            io = _read_numbers(Path('/proc/self/io'))
            available = _read_numbers(Path('/proc/meminfo')).get('MemAvailable')
            with self._lock:
                progress = dict(self._progress)
            if self._progress_source is not None:
                progress.update(self._progress_source())
            row = dict(schema_version=1, event='sample', time=datetime.fromtimestamp(
                stamp, timezone.utc).isoformat(), timestamp=stamp, monotonic_s=monotonic,
                name=self.name, pid=self.pid, training=self.training,
                rss_bytes=rss, children_rss_bytes=children_rss,
                mem_available_bytes=None if available is None else available * 1024,
                pressure_memory=pressure('memory'), pressure_io=pressure('io'),
                io=io or None, cpu_percent=None, tree_cpu_percent=None,
                gpu=gpu_sample(), progress=progress, rates={})
            if self._previous is not None:
                previous_time, previous_clocks, previous_progress = self._previous
                seconds = monotonic - previous_time
                delta = {key: max(0, value - previous_clocks.get(key, value))
                         for key, value in clocks.items()}
                if clocks and previous_clocks:
                    row['cpu_percent'] = sum(v for (pid, _), v in delta.items() if pid == self.pid) / seconds * 100
                    row['tree_cpu_percent'] = sum(delta.values()) / seconds * 100
                row['interval_s'] = seconds
                row['rates'] = {key + '_per_second': (value - previous_progress[key]) / seconds
                    for key, value in progress.items() if isinstance(value, (int, float))
                    and isinstance(previous_progress.get(key), (int, float))
                    and value >= previous_progress[key]}
            self._previous = monotonic, clocks, progress
            if self.training:
                ratio, observed, warn = self._window.observe(monotonic, gpu_utilization(row))
                row.update(gpu_active_ratio_10m=ratio, gpu_observed_seconds_10m=observed)
                if warn:
                    row.update(level='WARN', message='training GPU utilization >50% for less than 50% of the last 10 minutes')
            self._rotate(stamp)
            with self.path.open('a') as handle:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
            return row

    def _safe_sample(self):
        try:
            self.sample()
        except Exception as error:
            # Observability must never terminate training/collection.
            print(f'WARN metrics {self.name}: {type(error).__name__}: {error}', file=sys.stderr)

    def _run(self):
        self._safe_sample()
        while not self._stop.wait(self.interval):
            self._safe_sample()

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name='pvz-metrics', daemon=True)
            self._thread.start()
            atexit.register(self.close)
        return self

    def close(self):
        if os.getpid() != self.pid or self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._safe_sample()


def summarize(rows):
    """Summarize one PID across daily segments; right endpoint labels each interval."""
    rows = sorted((row for row in rows if row.get('event') == 'sample'),
                  key=lambda row: row['monotonic_s'])
    active = observed = 0.0
    read_bytes = write_bytes = 0
    for previous, row in zip(rows, rows[1:]):
        seconds = row['monotonic_s'] - previous['monotonic_s']
        utilization = gpu_utilization(row)
        if 0 < seconds <= INTERVAL * 2 and utilization is not None and gpu_utilization(previous) is not None:
            observed += seconds
            active += seconds * (utilization > 50)
        for key in ('read_bytes', 'write_bytes'):
            delta = max(0, (row.get('io') or {}).get(key, 0) - (previous.get('io') or {}).get(key, 0))
            if key == 'read_bytes':
                read_bytes += delta
            else:
                write_bytes += delta
    available = [r['mem_available_bytes'] for r in rows if r.get('mem_available_bytes') is not None]
    resident = [r['rss_bytes'] + (r.get('children_rss_bytes') or 0)
                for r in rows if r.get('rss_bytes') is not None]
    return dict(samples=len(rows), elapsed_seconds=rows[-1]['monotonic_s'] - rows[0]['monotonic_s'] if rows else 0,
                gpu_observed_seconds=observed, training_effective_ratio=active / observed if observed else None,
                disk_read_bytes=read_bytes, disk_write_bytes=write_bytes,
                minimum_mem_available_bytes=min(available) if available else None,
                peak_tree_rss_bytes=max(resident) if resident else None,
                warnings=sum(row.get('level') == 'WARN' for row in rows),
                progress=rows[-1].get('progress', {}) if rows else {})
