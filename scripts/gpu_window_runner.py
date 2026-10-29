"""Run queued research only in observable production pack/selection/gate windows.

No torch import, production writes, production locks, or production signals. Host/GPU
reserves leave room for production admission even before a phase change is observed.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import shutil
import time
from contextlib import suppress

GIB = 1 << 30
YIELDED = 75


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.building')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def processes(proc=Path('/proc')):
    result = {}
    for folder in proc.glob('[0-9]*'):
        try:
            argv = folder.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            # Read only actual script argv, never shell strings containing a command.
            script = next((Path(arg).name for arg in argv[1:3] if arg.endswith('.py')), '')
            status = folder.joinpath('status').read_text()
            parent = int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
            result[int(folder.name)] = dict(script=script, argv=argv, parent=parent)
        except (OSError, ValueError, StopIteration, UnicodeError):
            continue
    return result


def descendants(table, roots):
    found = set(roots)
    while True:
        added = {pid for pid, info in table.items() if info['parent'] in found} - found
        if not added:
            return found
        found.update(added)


def production_phase(table, own_pid):
    own = descendants(table, {own_pid})
    external = {pid: info for pid, info in table.items() if pid not in own}
    trainers = [pid for pid, info in external.items() if info['script'] in
                ('train_rollout_policy.py', 'train_value_net.py')]
    if trainers:
        return dict(open=False, phase='training', pids=trainers)
    controllers = {pid for pid, info in external.items() if info['script'] == 'curriculum_queue.py'}
    if not controllers:
        return dict(open=False, phase='controller_absent', pids=[])
    children = descendants(external, controllers) - controllers
    # No inference from persisted state: it is only flushed at foreground boundaries.
    # training.lock is held across pack/train/select/gate, so it is not a GPU lock.
    for script, phase in (('pack_training_rows.py', 'pack'), ('select_student.py', 'selection'),
                          ('promotion_gate.py', 'gate')):
        pids = [pid for pid in children if external[pid]['script'] == script]
        if pids:
            return dict(open=True, phase=phase, pids=pids)
    return dict(open=False, phase='transition_or_collection', pids=[])


def resources(query_gpu=True):
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    if not query_gpu:
        return available, None
    try:
        binary = '/usr/lib/wsl/lib/nvidia-smi' if Path('/usr/lib/wsl/lib/nvidia-smi').exists() else 'nvidia-smi'
        query = subprocess.run([binary, '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, check=True, timeout=1)
        gpu = int(query.stdout.splitlines()[0]) * (1 << 20)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        gpu = None
    return available, gpu


def may_run(phase, available, gpu, job, args, running=False):
    if not phase['open']:
        return False, phase['phase']
    host = args.host_reserve_gib * GIB + (0 if running else job['host_gib'] * GIB)
    device = args.gpu_reserve_gib * GIB + (0 if running else job['gpu_gib'] * GIB)
    if available is None or available < host:
        return False, 'host_reserve'
    if gpu is None or gpu < device:
        return False, 'gpu_reserve'
    return True, phase['phase']


class Runner:
    def __init__(self, args):
        self.args = args
        self.stopping = False
        self.child = None
        self.active = None
        self.log_position = args.production_log.stat().st_size if args.production_log.exists() else 0
        self.waiting_since = None
        self.wait_seconds = 0.0
        self.phase_started = time.time()
        self.last_phase = None
        old_status = args.queue / 'status.json'
        previous = json.loads(old_status.read_text()) if old_status.exists() else {}
        self.recovered_at = previous.get('updated', time.time())
        self.window_seconds = previous.get('phase_seconds', {})
        self.wait_seconds = previous.get('production_wait_observed_seconds', 0.0)

    def event(self, kind, **fields):
        with (self.args.queue / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(time=time.time(), kind=kind, **fields)) + '\n')

    def production_wait(self):
        """Observe new log messages only; never reread or replay historical jobs."""
        try:
            with self.args.production_log.open() as handle:
                if os.fstat(handle.fileno()).st_size < self.log_position:
                    self.log_position = 0
                handle.seek(self.log_position)
                # A long service outage cannot turn the next poll into an unbounded log scan.
                lines = handle.read(65536)
                self.log_position = handle.tell()
        except OSError:
            return False
        waiting = False
        for line in lines.splitlines():
            if 'background waiting for memory:' in line:
                self.event('production_wait_observed', line=line)
                waiting = True
                if self.waiting_since is None:
                    self.waiting_since = time.time()
        return waiting

    def phase(self):
        phase = production_phase(processes(), os.getpid())
        now = time.time()
        if phase['phase'] != self.last_phase:
            if self.last_phase:
                self.window_seconds[self.last_phase] = self.window_seconds.get(self.last_phase, 0) + now - self.phase_started
            self.phase_started, self.last_phase = now, phase['phase']
            self.event('production_phase', **phase)
        if self.waiting_since is not None and phase['phase'] == 'training':
            self.wait_seconds += now - self.waiting_since
            self.waiting_since = None
        return phase

    def candidates(self):
        pending = []
        completed = {}
        def dependency_done(job_id):
            if job_id not in completed:
                source = self.args.queue / (job_id+'.json')
                dependency = json.loads(source.read_text())
                marker = Path(dependency['out'])/'window.json'
                completed[job_id] = marker.is_file() and json.loads(marker.read_text()).get('status') == 'complete'
            return completed[job_id]
        for path in sorted(self.args.queue.glob('*.json')):
            if path.name == 'status.json':
                continue
            try:
                job = json.loads(path.read_text())
                out = Path(job['out'])
                if not out.is_absolute() or not Path(job['recipe']).is_absolute():
                    raise ValueError('out and recipe must be absolute')
                if not job.get('enabled', True):
                    continue
                if not (0 < job['host_gib'] <= self.args.memory_max_gib) or not (job['gpu_gib'] >= 0):
                    raise ValueError('invalid resource declaration')
                if not job['command'] or not all(isinstance(arg, str) for arg in job['command']):
                    raise ValueError('command must be an argv list')
                state_path = out / 'window.json'
                state = json.loads(state_path.read_text()) if state_path.exists() else dict(status='queued', seconds=0, attempts=[])
                if state['status'] == 'running':
                    previous = state['attempts'][-1]
                    previous.update(finished=self.recovered_at, interrupted='service_restart')
                    previous['seconds'] = max(0, self.recovered_at - previous['started'])
                    state['seconds'] += previous['seconds']
                    state['status'] = 'paused'
                    write_json(state_path, state)
                if state['status'] in ('complete', 'failed'):
                    continue
                if any(not Path(dependency).is_file() for dependency in job.get('depends_on', [])):
                    continue
                if any(not dependency_done(job_id) for job_id in job.get('after_jobs', [])):
                    continue
                if state.get('not_before', 0) <= time.time():
                    pending.append((job, state))
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.event('invalid_job', path=str(path), error=str(error))
        yield from sorted(pending, key=lambda pair: (pair[0].get('priority', 100), pair[1]['seconds'], pair[0]['id']))

    def start(self, job, state, phase):
        out = Path(job['out'])
        out.mkdir(parents=True, exist_ok=True)
        preparing = job.get('kind') == 'prepare'
        evaluating = not preparing and (out / 'model.pt').is_file()
        command = list(job['evaluate'] if evaluating else job['command'])
        resume = out / 'resume.pt'
        if not evaluating and resume.exists():
            command += ['--resume', str(resume)]
        with (out / ('prepare.log' if preparing else 'evaluate.log' if evaluating else 'train.log')).open('a') as log:
            self.child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                          cwd=job.get('code_root'),
                                          env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        attempt = dict(started=time.time(), phase=phase['phase'], stage='prepare' if preparing else 'evaluate' if evaluating else 'train',
                       pid=self.child.pid, revision=job.get('revision'))
        state.update(status='running', command=command)
        state['attempts'].append(attempt)
        self.active = job, state, attempt
        write_json(out / 'window.json', state)
        self.event('started', job=job['id'], **attempt)

    def yield_child(self, reason):
        job, state, attempt = self.active
        if 'yield_requested' not in attempt:
            attempt.update(yield_requested=time.time(), yield_reason=reason)
            # Only our new process group; never search for or signal a production PID.
            with suppress(ProcessLookupError):
                os.killpg(self.child.pid, signal.SIGUSR1)
            self.event('yield_requested', job=job['id'], reason=reason)
        elif time.time() - attempt['yield_requested'] > self.args.yield_timeout:
            with suppress(ProcessLookupError):
                os.killpg(self.child.pid, signal.SIGKILL)
            attempt['forced'] = True  # Periodic checkpoint bounds lost optimizer work.

    def finish(self):
        code = self.child.poll()
        if code is None:
            return
        job, state, attempt = self.active
        now = time.time()
        attempt.update(finished=now, seconds=now - attempt['started'], returncode=code)
        if 'yield_requested' in attempt:
            attempt['release_seconds'] = now - attempt['yield_requested']
        state['seconds'] += attempt['seconds']
        out = Path(job['out'])
        if code == 0 and attempt['stage'] == 'train':
            handoff = self.args.handoff / job['id']
            handoff.mkdir(parents=True, exist_ok=True)
            model = handoff / 'model.pt'
            if not model.exists():
                os.link(out / 'model.pt', model)
            shutil.copyfile(job['recipe'], handoff / 'recipe.json')
            if job.get('deck_split'):
                shutil.copyfile(job['deck_split'], handoff / 'deck_split.json')
            write_json(handoff / 'ready.json', dict(job=job['id'], stage='trained', model=str(model),
                source=str(out / 'model.pt'), cpu_metrics=str(out / 'heldout.json'), revision=job.get('revision'),
                code_root=job.get('code_root'), arch=job.get('arch'), seed=job.get('seed')))
            state['model_bytes'] = model.stat().st_size
            resume = out/'resume.pt'
            if resume.exists():
                size = resume.stat().st_size
                resume.unlink()
                self.event('intermediate_removed', job=job['id'], path=str(resume), bytes=size)
        if code == 0 and attempt['stage'] in ('evaluate', 'prepare'):
            state['status'] = 'complete'
            if attempt['stage'] == 'evaluate':
                write_json(self.args.handoff / (job['id'] + '.json'), dict(job=job['id'], model=str(out / 'model.pt'),
                    recipe=job['recipe'], metrics=str(out / 'heldout.json'), bench='Mac bench_play.py', finished=now,
                    revision=job.get('revision'), arch=job.get('arch'), seed=job.get('seed')))
        elif code == 0 or code == YIELDED or 'yield_requested' in attempt:
            state['status'] = 'paused'
            state['not_before'] = now + 5
        else:
            state.update(status='failed', error=f'child exited {code}; see {attempt["stage"]}.log')
        progress = out / ('eval_progress.json' if attempt['stage'] == 'evaluate' else 'progress.json')
        if progress.is_file():
            state['progress'] = json.loads(progress.read_text())
        write_json(out / 'window.json', state)
        self.event('finished', job=job['id'], status=state['status'], **attempt)
        self.child, self.active = None, None

    def run(self):
        while not self.stopping or self.child:
            phase = self.phase()
            waiting = self.production_wait()
            available, gpu = resources(query_gpu=phase['open'] or self.child is not None)
            if self.child:
                self.finish()
                if self.child:
                    allowed, reason = may_run(phase, available, gpu, self.active[0], self.args, running=True)
                    if 'yield_requested' in self.active[2]:
                        self.yield_child(self.active[2]['yield_reason'])
                    elif not allowed or self.stopping or waiting:
                        self.yield_child('stop' if self.stopping else 'production_wait' if waiting else reason)
                    elif time.time() - self.active[2]['started'] > self.active[0].get('slice_seconds', 900):
                        self.yield_child('time_slice')
            elif not self.stopping and not waiting:
                for job, state in self.candidates():
                    allowed, _ = may_run(phase, available, gpu, job, self.args)
                    if allowed:
                        # Reobserve immediately before spawning, narrowing the transition race.
                        fresh = self.phase()
                        if fresh['open']:
                            try:
                                self.start(job, state, fresh)
                            except OSError as error:
                                state.update(status='failed', error=str(error))
                                write_json(Path(job['out']) / 'window.json', state)
                                self.event('launch_failed', job=job['id'], error=str(error))
                        break
            now = time.time()
            elapsed = dict(self.window_seconds)
            if self.last_phase:
                elapsed[self.last_phase] = elapsed.get(self.last_phase, 0) + now - self.phase_started
            write_json(self.args.queue / 'status.json', dict(updated=now, phase=phase,
                host_available=available, gpu_free=gpu, phase_seconds=elapsed,
                production_wait_observed_seconds=self.wait_seconds + (now - self.waiting_since if self.waiting_since else 0),
                active=self.active[0]['id'] if self.active else None))
            time.sleep(self.args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--production-log', type=Path, required=True)
    parser.add_argument('--host-reserve-gib', type=float, default=12)
    parser.add_argument('--gpu-reserve-gib', type=float, default=8)
    parser.add_argument('--memory-max-gib', type=float, default=5)
    parser.add_argument('--poll-seconds', type=float, default=.5)
    parser.add_argument('--yield-timeout', type=float, default=2)
    args = parser.parse_args()
    args.queue.mkdir(parents=True, exist_ok=True)
    with (args.queue / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Runner(args)
        def stop(_signal, _frame):
            runner.stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            runner.run()
        finally:
            if runner.child and runner.child.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(runner.child.pid, signal.SIGKILL)


if __name__ == '__main__':
    main()
