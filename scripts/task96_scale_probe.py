"""Task96 mid-scale control: does the same recipe reproduce bitwise, and does
free GPU memory change the numbers?

The full 348-segment control (task96_full_pack.py) is not runnable on this host
while other tasks hold memory: the 9.6 GB spill cannot stay in page cache, so a
full epoch degrades to ~10 s/step (measured 2026-10-26 17:04-17:18, 542 GB read
in 700 s). This probe therefore uses a fixed subset of the same 348 segments
(first N by path, real rows, unchanged trainer, batch 2048, seed 1, one epoch)
and asks two questions the one-segment probe could not:

  * does the same pinned configuration reproduce bitwise at ~1.5M rows;
  * does reducing free GPU memory (the one environment axis that plausibly
    differs between the four historical full runs) change the numbers, with and
    without a pinned cuBLAS workspace.

No production process is controlled. qc keeps priority; this task runs at most
one GPU trainer. All writes stay under artifacts/task96/scale_probe.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/task96/scale_probe'
SOURCE = Path('/tmp/exp96')
SEGMENTS = 30
# case, deterministic algorithms, pinned cuBLAS workspace, hold GPU memory
# The last four are an interleaved default/strict/default/strict pair so that cache
# warm-up and host load drift cancel instead of being mistaken for the determinism cost.
CASES = [('strict_a', True, True, False),
         ('strict_b', True, True, False),
         ('default_c', False, False, False),
         ('default_squeeze_d', False, False, True),
         ('strict_squeeze_e', True, True, True),
         ('default_p1', False, False, False),
         ('strict_q1', True, True, False),
         ('default_p2', False, False, False),
         ('strict_q2', True, True, False)]
SQUEEZE_BYTES = 4*1024**3


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    assert path.is_absolute() and path.is_relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1)+'\n')


def training_peers():
    found = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            args = directory.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            is_python = bool(args) and Path(args[0]).name.startswith('python')
            is_trainer = is_python and any(a.endswith('/scripts/train_rollout_policy.py') for a in args)
            is_own = is_python and str(Path(__file__).resolve()) in args and 'child' in args
            if is_trainer or is_own:
                found.append(dict(pid=int(directory.name), argv=args))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def resources():
    memory = {line.split(':')[0]: int(line.split()[1])*1024
              for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    total, used, utilisation = [int(x.strip()) for x in result.stdout.splitlines()[0].split(',')]
    return dict(mem_available_bytes=memory['MemAvailable'], gpu_free_mib=total-used,
                gpu_used_mib=used, gpu_utilization_percent=utilisation)


def admission():
    peers = training_peers()
    usage = resources()
    priority = [peer for peer in peers
                if any('/.claude/worktrees/cq/scripts/train_rollout_policy.py' in arg
                       or 'curriculum_state.pipeline/round_' in arg for arg in peer['argv'])
                or (str(Path(__file__).resolve()) in peer['argv'] and 'child' in peer['argv'])]
    return dict(peers=peers, priority_peers=priority, resources=usage,
                admitted=not priority and usage['mem_available_bytes'] >= 6*1024**3
                and usage['gpu_free_mib'] >= 3072)


def prepare():
    import numpy as np
    plan = read(ROOT/'artifacts/task96/full_pack/plan.json')
    segments = sorted(plan['segments'], key=lambda entry: entry['path'])[:SEGMENTS]
    rows = []
    for entry in segments:
        with np.load(entry['path'], allow_pickle=False) as stored:
            rows.append(int(np.asarray(stored['rows'])))
    subset = dict(schema=1, task=96, source_plan='artifacts/task96/full_pack/plan.json',
                  source_plan_sha256=sha(ROOT/'artifacts/task96/full_pack/plan.json'),
                  source_index=plan['index'], source_index_sha256=plan['index_sha256'],
                  selection='first %d segments by path, identical for every case' % SEGMENTS,
                  segments=segments, segment_rows=rows, rows=sum(rows),
                  recipe=plan['recipe'], cases=[list(case) for case in CASES],
                  squeeze_bytes=SQUEEZE_BYTES,
                  purpose='Mid-scale determinism and free-memory sensitivity control; not a rerun of the full-index experiment and not a production cost measurement.')
    path = OUT/'plan.json'
    if path.exists():
        prior = read(path)
        assert prior['segments'] == segments, 'prior plan differs; preserve evidence'
        if prior['cases'] != subset['cases']:
            prior['cases'] = subset['cases']
            prior['cases_note'] = 'extended with an interleaved cost pair after the first five cases completed'
            write(path, prior)
    else:
        write(path, subset)
    print(json.dumps(dict(rows=subset['rows'], segments=len(segments), admission=admission())), flush=True)


def verify_inputs(plan):
    assert sha(Path(plan['source_index'])) == plan['source_index_sha256']
    for entry in plan['segments']:
        stat = Path(entry['path']).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (entry['size'], entry['mtime_ns']), entry['path']


def child(case):
    mode, pinned, squeeze = dict((name, (strict, pin, hold)) for name, strict, pin, hold in CASES)[case]
    if pinned:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '1'
    os.environ['PVZ_NO_POPUP'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    plan = read(OUT/'plan.json')
    verify_inputs(plan)
    folder = OUT/case
    assert not (folder/'trace.json').exists(), 'never overwrite a previous receipt'
    gate = admission()
    assert gate['admitted'], json.dumps(gate)
    sys.path[:0] = [str(SOURCE/'scripts'), str(SOURCE/'python'), str(ROOT/'scripts')]
    import torch
    import train_rollout_policy as trainer
    from task96_existing_runs import tensor_digest
    assert Path(trainer.__file__).resolve() == SOURCE/'scripts/train_rollout_policy.py'
    for entry in read(ROOT/'MEMORY/docs/task96-existing-runs.json')['source_snapshot']:
        assert sha(SOURCE/entry['path']) == entry['sha256']
    torch.use_deterministic_algorithms(mode)
    torch.backends.cudnn.deterministic = mode
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(int(plan['recipe']['seed']))
    held = None
    free_before = resources()['gpu_free_mib']
    if squeeze:
        held = torch.empty(SQUEEZE_BYTES, dtype=torch.uint8, device='cuda')
    free_after = resources()['gpu_free_mib']
    recipe = dict(plan['recipe'])
    recipe.update(data=str(folder/'input'), out=str(folder), source_row_report=True)
    write(folder/'recipe.json', recipe)
    trace = dict(task=96, case=case, strict=mode, pinned_workspace=pinned, squeeze=squeeze,
                 workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                 plan_sha256=sha(OUT/'plan.json'), admission=gate, started=time.time(),
                 environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                  cudnn=torch.backends.cudnn.version(),
                                  gpu=torch.cuda.get_device_name(0),
                                  matmul_precision=torch.get_float32_matmul_precision(),
                                  allow_tf32=torch.backends.cuda.matmul.allow_tf32),
                 gpu_free_mib_before_squeeze=free_before, gpu_free_mib_at_start=free_after,
                 permutations=[], losses=[], resource_samples=[])
    paths = [Path(entry['path']) for entry in plan['segments']]
    original_net = trainer.RolloutNet

    def network(*args, **kwargs):
        model = original_net(*args, **kwargs)
        trace['initial_tensor_sha256'] = tensor_digest(model.state_dict())
        return model

    trainer.RolloutNet = network

    def load(*args, **kwargs):
        value = trainer.read_segments(paths, data=folder/'input', row_observer=kwargs.get('row_observer'))
        assert len(value[2]) == plan['rows'], (len(value[2]), plan['rows'])
        return value

    trainer.load = load
    original_randperm = torch.randperm

    def randperm(*args, **kwargs):
        result = original_randperm(*args, **kwargs)
        trace['permutations'].append(dict(rows=result.numel(),
                                          sha256=hashlib.sha256(result.numpy().tobytes()).hexdigest()))
        return result

    torch.randperm = randperm
    original_metrics = trainer.Metrics
    trainer.Metrics = lambda *args, **kwargs: original_metrics(*args, directory=folder/'telemetry', **kwargs)
    checked = 0.0

    def observe(step, loss):
        nonlocal checked
        trace['losses'].append(loss)
        now = time.monotonic()
        if now-checked < 5:
            return
        checked = now
        gate = admission()
        if gate['priority_peers']:
            trace['priority_yield_peers'] = gate['priority_peers']
            raise RuntimeError('task96 yields: qc or another task96 trainer became active')
        trace['resource_samples'].append(dict(step=step, peers=gate['peers'], **gate['resources']))

    sys.argv = [str(SOURCE/'scripts/train_rollout_policy.py'), '--recipe', str(folder/'recipe.json')]
    began = time.monotonic()
    try:
        trainer.main(step_observer=observe)
        verify_inputs(plan)
        checkpoint = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
        trace.update(status='complete', model_file_sha256=sha(folder/'model.pt'),
                     tensor_sha256=tensor_digest(checkpoint['state_dict']),
                     metrics=read(folder/'metrics.json'),
                     source_rows_sha256=sha(folder/'source_rows.json'))
    except Exception:
        trace.update(status='error', error=traceback.format_exc())
    trace.update(wall_seconds=time.monotonic()-began, finished=time.time(), final_resources=resources())
    if held is not None:
        del held
        torch.cuda.empty_cache()
    write(folder/'trace.json', trace)
    print(json.dumps({k: trace.get(k) for k in ['case', 'status', 'tensor_sha256', 'error', 'wall_seconds']}), flush=True)


def run():
    plan = read(OUT/'plan.json')
    for case, _, _, _ in CASES:
        path = OUT/case/'trace.json'
        if path.exists():
            assert read(path)['status'] == 'complete', 'prior case failed: inspect it rather than blindly retry'
            continue
        while True:
            gate = admission()
            if gate['admitted']:
                break
            print(json.dumps(dict(waiting=gate, case=case)), flush=True)
            time.sleep(30)
        log = OUT/(case+'.log')
        assert not log.exists(), 'prior launched case has no receipt: inspect before relaunch'
        with log.open('w') as output:
            subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'child', '--case', case],
                           cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=1800)
        assert read(path)['status'] == 'complete', f'{case} incomplete; preserve evidence and stop'
    traces = [read(OUT/case/'trace.json') for case, _, _, _ in CASES]
    for trace in traces:
        trace['losses_sha256'] = hashlib.sha256(json.dumps(trace['losses']).encode()).hexdigest()
        trace['steps'] = len(trace['losses'])
        trace['first_losses'] = trace['losses'][:5]
        trace['last_losses'] = trace['losses'][-5:]
        del trace['losses']
    by = {trace['case']: trace for trace in traces}
    epoch = {name: trace['metrics']['history'][0] for name, trace in by.items()}
    summary = dict(task=96, scope='%d segments / %d real rows, seed1, epoch1, batch2048; same frozen trainer as the historical runs. Not the full index and not a production cost measurement.' % (len(plan['segments']), plan['rows']),
                   plan={k: v for k, v in plan.items() if k not in ['segments', 'recipe', 'segment_rows']},
                   cases=traces,
                   same_config_repeats_bitwise=by['strict_a']['tensor_sha256'] == by['strict_b']['tensor_sha256'],
                   initial_tensors_equal=len({t['initial_tensor_sha256'] for t in traces}) == 1,
                   permutations_equal=all(t['permutations'] == traces[0]['permutations'] for t in traces),
                   pinned_survives_memory_pressure=by['strict_a']['tensor_sha256'] == by['strict_squeeze_e']['tensor_sha256'],
                   unpinned_changes_with_memory_pressure=by['default_c']['tensor_sha256'] != by['default_squeeze_d']['tensor_sha256'],
                   val_loss={name: epoch[name]['val_loss'] for name in epoch},
                   train_loss={name: epoch[name]['train_loss'] for name in epoch},
                   epoch_seconds={name: epoch[name]['seconds'] for name in epoch},
                   gpu_free_mib={name: by[name]['gpu_free_mib_at_start'] for name in by})
    if 'default_p1' in by:
        default_seconds = [epoch[name]['seconds'] for name in ['default_p1', 'default_p2']]
        strict_seconds = [epoch[name]['seconds'] for name in ['strict_q1', 'strict_q2']]
        summary['cost_pairs'] = dict(
            order=['default_p1', 'strict_q1', 'default_p2', 'strict_q2'],
            default_seconds=default_seconds, strict_seconds=strict_seconds,
            default_mean=sum(default_seconds)/2, strict_mean=sum(strict_seconds)/2,
            ratio=(sum(strict_seconds)/2)/(sum(default_seconds)/2),
            note='Interleaved to cancel cache warm-up and load drift; single host, other tasks running, one epoch each.')
        summary['cost_pairs']['strict_repeats_tensor_equal'] = by['strict_q1']['tensor_sha256'] == by['strict_q2']['tensor_sha256']
        summary['cost_pairs']['default_repeats_tensor_equal'] = by['default_p1']['tensor_sha256'] == by['default_p2']['tensor_sha256']
    write(ROOT/'MEMORY/docs/task96-scale-probe.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['cases', 'plan']}, indent=1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'child'])
    parser.add_argument('--case', choices=[case[0] for case in CASES])
    args = parser.parse_args()
    globals()[args.mode](args.case) if args.mode == 'child' else globals()[args.mode]()
