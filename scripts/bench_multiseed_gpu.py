#!/usr/bin/env python3
"""Measure one/two real trainer workers, using immutable packs and private spill directories."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
GIB = 1 << 30


def qc_training():
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            words = path.read_bytes().split(b'\0')
        except OSError:
            continue
        if any(word.endswith(b'/train_rollout_policy.py') for word in words) and any(
                b'/pvz-logs/curriculum_state.pipeline/' in word for word in words):
            return True
    return False


def available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) * 1024


def worker(args):
    import torch
    import train_rollout_policy as trainer
    # Only replace the corpus lookup. The real trainer still performs its ordinary
    # split, reweighting, batching, model forward/backward, AdamW and LR schedule.
    original_load = trainer.load
    trainer.load = lambda data, **kw: trainer.read_pack(args.pack, data, None, kw.get('source_weight'))
    start = time.monotonic()
    report = dict(status='loading', pid=os.getpid(), worker=args.worker)
    target = args.out / f'worker_{args.worker}.json'
    target.write_text(json.dumps(report))

    class Finished(BaseException):
        pass

    training_start = None

    def observe(step, loss):
        nonlocal training_start
        if step == 1:
            ready = args.out / f'ready_{args.worker}'
            ready.write_text(str(os.getpid()))
            deadline = time.monotonic() + 300
            while len(list(args.out.glob('ready_*'))) < args.parallel:
                if time.monotonic() > deadline:
                    raise RuntimeError('other benchmark worker did not reach the barrier')
                time.sleep(.1)
            torch.cuda.synchronize()
            training_start = time.monotonic()
            report.update(status='training', load_seconds=training_start - start)
            target.write_text(json.dumps(report))
        if step == args.steps + 1:
            torch.cuda.synchronize()
            elapsed = time.monotonic() - training_start
            status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines()
                          if ':' in line)
            report.update(status='complete', steps=args.steps, training_seconds=elapsed,
                          seconds_per_step=elapsed / args.steps,
                          cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                          cuda_peak_reserved=torch.cuda.max_memory_reserved(),
                          rss_anon_bytes=int(status['RssAnon'].split()[0]) * 1024,
                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            target.write_text(json.dumps(report, indent=2) + '\n')
            raise Finished()

    sys.argv = [str(ROOT / 'scripts/train_rollout_policy.py'), '--recipe', str(args.recipe)]
    try:
        trainer.main(step_observer=observe)
    except Finished:
        pass
    finally:
        trainer.load = original_load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--parallel', type=int, default=1, choices=(1, 2))
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args)
        return
    from pvz_recipe import load_recipe, write_recipe
    args.out.mkdir(parents=True, exist_ok=True)
    if list(args.out.glob('ready_*')):
        raise RuntimeError('use a new benchmark directory; stale barriers are not valid')
    required = (6 + 4 * args.parallel) * GIB
    while qc_training() or available() < required:
        print(json.dumps(dict(waiting_for_qc_or_memory=True, available=available(), required=required)), flush=True)
        time.sleep(10)
    base = load_recipe(args.recipe, 'training')
    processes, logs = [], []
    try:
        for index in range(args.parallel):
            directory = args.out / f'seed_{index + 1}'
            recipe = dict(base, seed=index + 1, out=str(directory / 'student'),
                          data=str(directory / 'data'), cache_dir=str(directory / 'cache'),
                          source_row_report=False, max_steps=args.steps + 1)
            path = directory / 'recipe.json'
            directory.mkdir(parents=True)
            write_recipe(path, recipe)
            log = (directory / 'train.out').open('w')
            logs.append(log)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(index),
                       '--recipe', str(path), '--pack', str(args.pack), '--out', str(args.out),
                       '--parallel', str(args.parallel), '--steps', str(args.steps)]
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        while any(process.poll() is None for process in processes):
            if qc_training() or available() < 6 * GIB:
                raise RuntimeError('yielding GPU/memory to qc; benchmark interrupted, no result accepted')
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError('benchmark worker failed; inspect its private train.out')
            time.sleep(1)
        if any(process.returncode for process in processes):
            raise RuntimeError('benchmark worker failed')
        reports = [json.loads((args.out / f'worker_{index}.json').read_text()) for index in range(args.parallel)]
        if not all(report['status'] == 'complete' for report in reports):
            raise RuntimeError('benchmark did not complete the requested optimizer steps')
        result = dict(parallel=args.parallel, steps=args.steps, workers=reports,
                      pack=str(args.pack), pack_sha256=hashlib.sha256(args.pack.read_bytes()).hexdigest(),
                      source_recipe=str(args.recipe),
                      source_recipe_sha256=hashlib.sha256(args.recipe.read_bytes()).hexdigest())
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
