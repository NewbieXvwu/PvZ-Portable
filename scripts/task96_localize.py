"""Task96 divergence localisation: which step and which module first differ
between two runs of the same recipe when the cuBLAS workspace is not pinned?

The nine-case mid-scale control showed pinned runs are bitwise identical and
unpinned runs never repeat, but it could not say where the difference enters.
This runs the same 30-segment subset twice pinned and twice unpinned, hashing the
model inputs, every module output and the parameters after each of the first few
steps. All writes stay under artifacts/task96/localize; production is untouched.
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
OUT = ROOT/'artifacts/task96/localize'
SOURCE = Path('/tmp/exp96')
PLAN = ROOT/'artifacts/task96/scale_probe/plan.json'
CASES = [('unpinned_a', False), ('unpinned_b', False), ('pinned_a', True), ('pinned_b', True)]
TRACE_STEPS = 6


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    assert path.is_absolute() and path.is_relative_to(ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1)+'\n')


def tensor_sha(tensor):
    import torch
    if isinstance(tensor, (tuple, list)):
        return [tensor_sha(item) for item in tensor]
    if isinstance(tensor, dict):
        return {key: tensor_sha(value) for key, value in tensor.items()}
    if not isinstance(tensor, torch.Tensor):
        return None
    value = tensor.detach().to('cpu').contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def child(case):
    pinned = dict(CASES)[case]
    if pinned:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = '1'
    os.environ['PVZ_NO_POPUP'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    plan = read(PLAN)
    folder = OUT/case
    assert not (folder/'trace.json').exists(), 'never overwrite a previous receipt'
    sys.path[:0] = [str(SOURCE/'scripts'), str(SOURCE/'python'), str(ROOT/'scripts')]
    import torch
    import train_rollout_policy as trainer
    from task96_existing_runs import tensor_digest
    assert Path(trainer.__file__).resolve() == SOURCE/'scripts/train_rollout_policy.py'
    torch.use_deterministic_algorithms(pinned)
    torch.backends.cudnn.deterministic = pinned
    torch.backends.cudnn.benchmark = False
    recipe = dict(plan['recipe'])
    recipe.update(data=str(folder/'input'), out=str(folder), source_row_report=False)
    write(folder/'recipe.json', recipe)
    paths = [Path(entry['path']) for entry in plan['segments']]
    trace = dict(task=96, case=case, pinned=pinned, workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                 plan_sha256=sha(PLAN), steps=[], started=time.time())
    state = dict(step=0, record=True, inputs=None, modules={})
    original_net = trainer.RolloutNet

    def network(*args, **kwargs):
        model = original_net(*args, **kwargs)
        trace['initial_tensor_sha256'] = tensor_digest(model.state_dict())
        names = {id(module): name for name, module in model.named_modules()}

        def pre(module, inputs):
            if state['record']:
                state['inputs'] = tensor_sha(inputs)

        def post(module, inputs, output):
            if state['record']:
                state['modules'][names[id(module)]] = tensor_sha(output)

        model.register_forward_pre_hook(pre)
        for module in model.modules():
            module.register_forward_hook(post)
        return model

    trainer.RolloutNet = network
    trainer.load = lambda *a, **k: trainer.read_segments(paths, data=folder/'input',
                                                         row_observer=k.get('row_observer'))
    original_metrics = trainer.Metrics
    trainer.Metrics = lambda *a, **k: original_metrics(*a, directory=folder/'telemetry', **k)

    def observe(step, loss):
        state['step'] = step
        if step <= TRACE_STEPS:
            trace['steps'].append(dict(step=step, loss=repr(loss), inputs=state.get('inputs'),
                                       modules=dict(state.get('modules') or {})))
        if step >= TRACE_STEPS:
            state['record'] = False
        state['inputs'] = None
        state['modules'] = {}

    trainer.RolloutNet = network
    sys.argv = [str(SOURCE/'scripts/train_rollout_policy.py'), '--recipe', str(folder/'recipe.json')]
    began = time.monotonic()
    try:
        trainer.main(step_observer=observe)
        checkpoint = torch.load(folder/'model.pt', map_location='cpu', weights_only=False)
        trace.update(status='complete', tensor_sha256=tensor_digest(checkpoint['state_dict']),
                     metrics=read(folder/'metrics.json'))
    except Exception:
        trace.update(status='error', error=traceback.format_exc())
    trace['wall_seconds'] = time.monotonic()-began
    write(folder/'trace.json', trace)
    print(json.dumps({k: trace.get(k) for k in ['case', 'status', 'tensor_sha256', 'error', 'wall_seconds']}), flush=True)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    for case, _ in CASES:
        path = OUT/case/'trace.json'
        if path.exists():
            assert read(path)['status'] == 'complete', 'prior case failed: inspect it'
            continue
        log = OUT/(case+'.log')
        assert not log.exists(), 'prior launch without receipt: inspect before relaunch'
        with log.open('w') as output:
            subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'child', '--case', case],
                           cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=1800)
    traces = {case: read(OUT/case/'trace.json') for case, _ in CASES}
    summary = dict(task=96, scope='First %d steps of the 30-segment subset; module and parameter hashes per step.' % TRACE_STEPS,
                   cases={case: dict(pinned=trace['pinned'], workspace=trace['workspace'],
                                     initial=trace['initial_tensor_sha256'], final=trace['tensor_sha256'],
                                     steps=trace['steps']) for case, trace in traces.items()})
    for left, right in (('unpinned_a', 'unpinned_b'), ('pinned_a', 'pinned_b')):
        a, b = traces[left], traces[right]
        summary[left+'__vs__'+right] = dict(
            final_equal=a['tensor_sha256'] == b['tensor_sha256'],
            first_differing_step=next((x['step'] for x, y in zip(a['steps'], b['steps'])
                                       if x['modules'] != y['modules'] or x['inputs'] != y['inputs']), None),
            first_differing_module=next((name for x, y in zip(a['steps'], b['steps'])
                                         for name in x['modules']
                                         if x['modules'].get(name) != y['modules'].get(name)), None))
    write(ROOT/'MEMORY/docs/task96-localize.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}, indent=1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'child'])
    parser.add_argument('--case', choices=[case[0] for case in CASES])
    args = parser.parse_args()
    globals()[args.mode](args.case) if args.mode == 'child' else globals()[args.mode]()
