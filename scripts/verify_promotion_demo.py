#!/usr/bin/env python3
"""task49A: real paired gates + foreground publication for existing trained checkpoints.

Use two independent states: weak candidate versus dw4i (reject), then dw4i
candidate versus the weak incumbent (approve). This evaluates real pre-trained
checkpoints; it does not claim a new training run or fabricate training receipts.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from curriculum_queue import (TrainingPipeline, save_state, command_for, worker_count,
                              mem_available_bytes, _proc_table, MAIN, DEFAULT_STATIC_VALUE, GIB)
from promotion_gate import sha
from pvz_recipe import load_recipe, write_recipe


def demonstrate(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'demonstration.json').exists():
        raise ValueError('use a new independent output directory for each demonstration')
    candidate = args.out / 'checkpoint/student/model.pt'
    incumbent = args.out / 'incumbent/model.pt'
    for source, dest in [(args.candidate, candidate), (args.incumbent, incumbent)]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha(dest) != sha(source):
            raise ValueError(f'checkpoint copy changed: {dest}')
        if not dest.exists():
            shutil.copyfile(source, dest)
    recipe = load_recipe(ROOT / 'artifacts/task49/acceptance/recipe.json', 'training')
    recipe.update(data=str(ROOT / 'artifacts/task49/acceptance/frames'),
                  out=str(candidate.parent), cache_dir=str(args.out / 'cache'))
    recipe_path = args.out / 'checkpoint/recipe.json'
    write_recipe(recipe_path, recipe)
    config = SimpleNamespace(state=args.out / 'state.json', log=args.out / 'curriculum.log',
        frames=args.out / 'collection_frames', out=args.out / 'collection_runs',
        static_value=DEFAULT_STATIC_VALUE, recipe=recipe_path,
        python=Path(sys.executable), rollout_policy=incumbent, workers=2,
        executable=args.executable, resource_dir=args.resource_dir, value=args.value,
        promotion_panel=args.source_panel)
    state = dict(round=1, log=[], cells={}, offsets={}, training_jobs=[], student_round=0)
    job = dict(round=1, recipe=str(recipe_path), model=str(candidate), status='queued',
               demonstration='evaluation of an existing trained checkpoint; no new training claimed')
    pipeline = TrainingPipeline(config, state)
    state['training_jobs'].append(job)
    save_state(config.state, state)
    started = time.time()
    future = pipeline.executor.submit(pipeline._evaluate, copy.deepcopy(job))
    pipeline.futures.append((job, future))
    groups, samples = [], []
    max_games = 0
    def game_ticks(root_pid, table):
        if root_pid is None:
            return {}
        descendants, pending = {root_pid}, [root_pid]
        while pending:
            parent = pending.pop()
            for pid, (ppid, _) in table.items():
                if ppid == parent and pid not in descendants:
                    descendants.add(pid); pending.append(pid)
        result = {}
        for pid in descendants:
            try:
                argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
                if argv and argv[0] == str(args.executable).encode():
                    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                    result[str(pid)] = int(fields[11]) + int(fields[12])
            except (OSError, ValueError, IndexError):
                pass
        return result
    try:
        assert pipeline.foreground_limit() == 1
        assert config.rollout_policy == incumbent and state['student_round'] == 0
        # Run real beta=0 teacher-labelled groups while the quality gate runs. Keep collecting
        # until it finishes; never wait for the gate to start the next foreground group.
        while (not future.done() or not groups) and len(groups) < 3:
            label = f'Agroup{len(groups):03d}'
            offset = 128 + 4 * len(groups)
            before = config.rollout_policy
            workers = worker_count(mem_available_bytes(), GIB, min(1, pipeline.foreground_limit()), pipeline.reservation())
            assert workers == 1
            command = command_for(config, MAIN / 'artifacts/stage4/day_panel.json', 'day_l1_train_d7',
                                  offset, 4, label, workers=workers)
            start = time.time()
            with (args.out / 'collection.log').open('a') as log:
                collector = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                while collector.poll() is None:
                    table = _proc_table()
                    with pipeline.lock:
                        backend = pipeline.process
                        backend_pid = backend.pid if backend is not None else None
                    collection_ticks = game_ticks(collector.pid, table)
                    evaluation_ticks = game_ticks(backend_pid, table)
                    total = len(set(collection_ticks) | set(evaluation_ticks))
                    max_games = max(max_games, total)
                    assert total <= 2, f'own game limit exceeded: {total}'
                    samples.append(dict(time=time.time(), collection=collection_ticks, evaluation=evaluation_ticks))
                    time.sleep(.25)
            if collector.returncode:
                raise RuntimeError(f'collection exited {collector.returncode}; see {args.out / "collection.log"}')
            finish = time.time()
            assert config.rollout_policy == before, 'a background worker switched the model inside a group'
            folder = config.out / f'{label}_4'
            rows = [json.loads(line) for line in (folder / 'games.jsonl').read_text().splitlines()]
            collected_config = json.loads((folder / 'config.json').read_text())
            assert len(rows) == 4 and collected_config['rollout_policy'] == str(before)
            assert float(collected_config['dagger_beta']) == 0
            groups.append(dict(label=label, start=start, end=finish, student=str(before),
                               games=len(rows), seeds=[row['seed'] for row in rows], workers=workers))
            state['offsets']['day_l1_train_d7'] = offset + 4
            pipeline.poll()  # the only place publication is permitted
            save_state(config.state, state)
        report = future.result()
        boundary = time.time()
        pipeline.poll()
        approved = report['promotion']['approved']
        assert approved == (args.expect == 'approved'), report['promotion']['reasons']
        assert config.rollout_policy == (candidate if approved else incumbent)
        assert state['student_round'] == (1 if approved else 0)
        assert job['status'] == 'done' and job['promotion_approved'] == approved
        concurrent = [sample for sample in samples if sample['collection'] and sample['evaluation']]
        ticks = sum(max(0, row['collection'][pid] - previous['collection'][pid])
                    for previous, row in zip(samples, samples[1:])
                    if previous['evaluation'] and row['evaluation']
                    for pid in set(row['collection']) & set(previous['collection']))
        assert concurrent and ticks > 0, 'no evidence of active real collection during evaluation'
        (args.out / 'overlap_samples.jsonl').write_text(''.join(json.dumps(sample) + '\n' for sample in samples))
        summary = dict(passed=True, expected=args.expect, approved=approved,
            candidate_source=str(args.candidate), incumbent_source=str(args.incumbent),
            candidate_sha256=sha(candidate), incumbent_sha256=sha(incumbent),
            source_receipt='existing real trained checkpoints; evaluation/publication only',
            started=started, group_boundary=boundary, finished=time.time(),
            student_before=str(incumbent), student_after=str(config.rollout_policy),
            student_round=state['student_round'], promotion=report['promotion'],
            evaluation_directory=report['evaluation_directory'], collection_groups=groups,
            max_game_processes=max_games, concurrent_game_samples=len(concurrent),
            collection_cpu_seconds_during_evaluation=ticks / os.sysconf('SC_CLK_TCK'))
        (args.out / 'demonstration.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({key: summary[key] for key in ('passed', 'expected', 'approved', 'student_round',
                                                       'student_before', 'student_after')}), flush=True)
        return summary
    finally:
        pipeline.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'incumbent', 'source-panel', 'executable', 'value', 'resource-dir', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expect', choices=['approved', 'rejected'], required=True)
    args = parser.parse_args()
    if any(not getattr(args, name).is_absolute() for name in
           ('candidate', 'incumbent', 'source_panel', 'executable', 'value', 'resource_dir', 'out')):
        parser.error('use absolute paths')
    demonstrate(args)


if __name__ == '__main__':
    main()
