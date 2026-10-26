"""Task96 gradient localisation: in the first training step the forward pass and the
loss are bitwise identical between two unpinned runs, so the difference must enter
in the backward pass or the optimizer. This hashes every parameter gradient during
step 1 of two unpinned runs (and two pinned controls) and reports which gradient
differs first in backward order. All writes stay under artifacts/task96/grad_probe.
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
OUT = ROOT/'artifacts/task96/grad_probe'
SOURCE = Path('/tmp/exp96')
PLAN = ROOT/'artifacts/task96/scale_probe/plan.json'
CASES = [('unpinned_g1', False), ('unpinned_g2', False), ('pinned_g1', True), ('pinned_g2', True)]


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
    if tensor is None:
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
                 plan_sha256=sha(PLAN), grads=[], started=time.time())
    state = dict(step=0)
    trainer_model = [None]
    original_net = trainer.RolloutNet

    def network(*args, **kwargs):
        model = original_net(*args, **kwargs)
        trace['initial_tensor_sha256'] = tensor_digest(model.state_dict())
        trainer_model[0] = model
        return model

    trainer.RolloutNet = network
    trainer.load = lambda *a, **k: trainer.read_segments(paths, data=folder/'input',
                                                         row_observer=k.get('row_observer'))
    original_metrics = trainer.Metrics
    trainer.Metrics = lambda *a, **k: original_metrics(*a, directory=folder/'telemetry', **k)

    # The trainer calls step_observer after the forward/loss and before backward, so at
    # observer(2) the gradients still held are step 1's (zero_grad runs after the observer)
    # and the parameters are those the step-1 update produced.
    def observe(step, loss):
        state['step'] = step
        model = trainer_model[0]
        if step == 1:
            trace['step1_loss'] = repr(loss)
            trace['step1_params_before_backward'] = {name: tensor_sha(param.detach())
                                                     for name, param in model.named_parameters()}
        elif step == 2:
            trace['step1_grads'] = {name: tensor_sha(param.grad)
                                    for name, param in model.named_parameters()}
            trace['step1_grads_present'] = sum(param.grad is not None
                                               for _, param in model.named_parameters())
            trace['step2_params'] = {name: tensor_sha(param.detach())
                                     for name, param in model.named_parameters()}

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
    summary = dict(task=96, scope='Parameter gradients of training step 1, hashed in backward order; 30-segment subset.',
                   cases={case: dict(pinned=trace['pinned'], initial=trace['initial_tensor_sha256'],
                                     step1_loss=trace.get('step1_loss'), final=trace['tensor_sha256'],
                                     step1_grads_present=trace.get('step1_grads_present'),
                                     step1_grads=trace.get('step1_grads') or {},
                                     step1_params_before_backward=trace.get('step1_params_before_backward') or {},
                                     step2_params=trace.get('step2_params') or {}) for case, trace in traces.items()})
    for left, right in (('unpinned_g1', 'unpinned_g2'), ('pinned_g1', 'pinned_g2')):
        a, b = traces[left], traces[right]
        order_a = list(a['step1_grads'])
        differing = [name for name in order_a
                     if a['step1_grads'].get(name) != b['step1_grads'].get(name)]
        params_differ = [name for name in order_a
                         if a['step2_params'].get(name) != b['step2_params'].get(name)]
        step2_differ = [name for name in order_a
                        if a['step2_params'].get(name) != b['step2_params'].get(name)]
        summary[left+'__vs__'+right] = dict(
            step1_loss_equal=a.get('step1_loss') == b.get('step1_loss'),
            final_equal=a['tensor_sha256'] == b['tensor_sha256'],
            gradient_count_equal=len(a['step1_grads']) == len(b['step1_grads']),
            differing_gradients=differing,
            first_differing_gradient=differing[0] if differing else None,
            params_differ_after_step1_update=params_differ,
            params_differ_after_step2=step2_differ,
            parameter_order=order_a)
    write(ROOT/'MEMORY/docs/task96-grad-probe.json', summary)
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'parameter_order'}
                            if isinstance(v, dict) else v)
                      for k, v in summary.items() if k != 'cases'}, indent=1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'child'])
    parser.add_argument('--case', choices=[case[0] for case in CASES])
    args = parser.parse_args()
    globals()[args.mode](args.case) if args.mode == 'child' else globals()[args.mode]()
