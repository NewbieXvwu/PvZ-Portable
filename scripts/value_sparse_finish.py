"""Finish the approved task43 sparse run after its live data pipeline exits.

One absolute plan argument. Uses frozen training/evaluation recipes, sequential
four-worker games, and the existing strict raw-evidence audit. Never changes defaults.
"""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from value_teacher_feasibility import ROOT, digest, write_json, process_sample
from value_teacher_sparse import artifact_bytes, load_plan
from value_sparse_pipeline import PipelineStop, process_identity, publish


def resource_gate(plan):
    size = artifact_bytes(plan['artifact_root'])
    if size > plan['artifact_budget_bytes']:
        raise PipelineStop(f'task43 artifact budget exceeded: {size} bytes')
    return size


def data_gate(plan):
    out = Path(plan['out'])
    state = json.loads((out / 'pipeline_status.json').read_text())
    if state['stage'] != 'data_complete_training_pending':
        raise PipelineStop(f'data pipeline did not finish successfully: {state["stage"]}')
    exported = json.loads((out / 'export.json').read_text())
    rows = exported['seeds']
    if exported['status'] != 'complete' or len(rows) != 64 or {r['seed'] for r in rows} != set(plan['seeds']):
        raise PipelineStop('export does not cover the frozen 64 seeds exactly')
    if exported['held_out_seeds'] != plan['held_out_seeds']:
        raise PipelineStop('export validation split differs from the frozen split')
    if {p.name for p in (out / 'data').glob('*.pt')} != {f'seed_{s}.pt' for s in plan['seeds']}:
        raise PipelineStop('unexpected or missing exported shard')
    for row in rows:
        if row['rows'] <= 0 or digest(out / 'data' / f"seed_{row['seed']}.pt") != row['sha256']:
            raise PipelineStop('empty or changed exported shard')
    return exported


def wait_data(plan, status):
    path = Path(plan['out']) / 'pipeline_status.json'
    state = json.loads(path.read_text()); pid = state['pid']; identity = process_identity(pid)
    if identity is not None:
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if str(ROOT / 'scripts/value_sparse_pipeline.py').encode() not in argv:
            raise PipelineStop('data pipeline PID belongs to a different process')
    status('waiting_data', dict(pid=pid, start_ticks=identity))
    while identity is not None and process_identity(pid) == identity:
        resource_gate(plan)
        time.sleep(30)
    resource_gate(plan)
    return data_gate(plan)


def training_ready():
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    trainer = subprocess.run(['pgrep', '-f', 'train_rollout_policy.py'], capture_output=True)
    if trainer.returncode not in (0, 1):
        raise PipelineStop('could not check competing student trainer')
    return available >= 10 * 1024**3 and trainer.returncode == 1, available


def checked_call(plan, command, log_path):
    resource_gate(plan)
    # Only the process group created here can be stopped on a budget violation.
    with log_path.open('a') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                resource_gate(plan)
                time.sleep(5)
            if child.returncode:
                raise PipelineStop(f'{Path(command[1]).name} failed ({child.returncode}); see {log_path}')
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
            raise
    resource_gate(plan)


def verify_models(plan, model_sha):
    load_plan(Path(plan['out']) / 'plan.json')
    if digest(Path(plan['training']['out']) / 'model.pt') != model_sha:
        raise PipelineStop('trained checkpoint changed during acceptance')


def training_command(plan, device):
    cfg = plan['training']
    command = [sys.executable, str(ROOT / 'scripts/train_value_widen.py'), 'train',
               '--data', '/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2',
               '--audit-manifest', cfg['audit_manifest'], '--out', cfg['out'], '--init', cfg['init'],
               '--decision-data', str(Path(plan['out']) / 'data'), '--device', device]
    for key in ('decision_fraction', 'branch_fraction', 'epochs', 'steps', 'batch_size', 'lr', 'seed'):
        command.extend(['--' + key.replace('_', '-'), str(cfg[key])])
    return command


def evaluate(plan, model_sha, status, record):
    out = Path(plan['out']); evaluation = out / 'evaluation'; model = Path(plan['training']['out']) / 'model.pt'
    verify_models(plan, model_sha)
    status('calibration', {})
    checked_call(plan, [sys.executable, str(ROOT / 'scripts/value_widen_audit.py'),
        '--data', '/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2',
        '--out', str(evaluation / 'calibration'), '--manifest', plan['evaluation']['calibration_manifest'],
        '--model', f"v4={plan['teacher']['value']}", '--model', f'v5={model}'], evaluation / 'calibration.log')
    metrics = json.loads((evaluation / 'calibration' / 'metrics.json').read_text())
    calibration = {k: {name: v['calibration'] for name, v in row['models'].items()}
                   for k, row in metrics['cohorts'].items()}
    record('teacher_v5_calibration_complete', calibration)
    status('dispersion', {})
    # The tool's internal manifest parameter reuses the already-frozen extended seed list.
    source = ('import sys; from pathlib import Path; from argparse import Namespace; '
              f'sys.path.insert(0,{str(ROOT / "scripts")!r}); '
              'from value_leaf_dispersion import run,report; '
              f'a=Namespace(out=Path({str(evaluation / "dispersion")!r}), '
              f'json=Path({str(evaluation / "dispersion.json")!r}), '
              f'v3=Path({plan["teacher"]["value"]!r}), v4=Path({str(model)!r}), '
              f'executable=Path({plan["teacher"]["executable"]!r}), '
              f'manifest=Path({plan["evaluation"]["manifest"]!r}), '
              'games=4,seed_offset=0,workers=4,limit=80,sample=200); run(a); report(a)')
    verify_models(plan, model_sha)
    checked_call(plan, [sys.executable, '-c', source], evaluation / 'dispersion.log')
    dispersion = json.loads((evaluation / 'dispersion.json').read_text())
    record('teacher_v5_dispersion_complete', dispersion['per_level'])
    # Complete every predeclared paired task even when dispersion fails.
    for label in ('l21', 'day', 'night'):
        directory = evaluation / f'ab_{label}'
        for arm in ('v3', 'v4', 'report'):
            status(f'ab_{label}_{arm}', {})
            verify_models(plan, model_sha)
            checked_call(plan, [sys.executable, str(ROOT / 'scripts/value_widen_ab.py'), arm,
                '--out', str(directory), '--v4', str(model), '--baseline', plan['teacher']['value'],
                '--executable', plan['teacher']['executable'], '--workers', '4', '--games', '32'],
                evaluation / f'ab_{label}.log')
        paired = json.loads((directory / 'paired.json').read_text())
        if paired['complete'] is not True:
            raise PipelineStop(f'incomplete paired evidence for {label}')
        record(f'teacher_v5_paired_{label}_complete', paired['results'])


def upload_if_accepted(plan, acceptance):
    if acceptance['all_three_pass'] is not True:
        return None
    # Fixed v5 destination; v4 and production defaults remain unchanged.
    from huggingface_hub import HfApi, hf_hub_download
    repo = os.environ.get('PVZ_HF_REPO'); token = os.environ.get('HF_TOKEN')
    if not repo:
        raise PipelineStop('all gates passed; $PVZ_HF_REPO is required for the authorized HF upload')
    out = Path(plan['out']); model = Path(plan['training']['out']) / 'model.pt'
    if digest(model) != acceptance['model_sha256']['v5']:
        raise PipelineStop('candidate changed before upload')
    commit = HfApi(token=token).upload_file(repo_id=repo, repo_type='model', path_or_fileobj=str(model),
        path_in_repo='transfer/value_v5.pt', commit_message='task43: teacher-continuation value v5, acceptance complete')
    local = Path(hf_hub_download(repo_id=repo, repo_type='model', filename='transfer/value_v5.pt',
                 revision=commit.oid, local_dir=str(out / 'hf_verify'), token=token))
    if digest(local) != digest(model):
        raise PipelineStop('HF v5 roundtrip checksum mismatch')
    receipt = dict(repo=repo, path='transfer/value_v5.pt', revision=commit.oid,
                   sha256=digest(model), verified_download=True, default_changed=False)
    write_json(out / 'evaluation' / 'hf_receipt.json', receipt)
    return receipt


def main():
    plan = load_plan(sys.argv[1]); out = Path(plan['out']); evaluation = out / 'evaluation'
    def status(stage, details):
        write_json(out / 'finish_status.json', dict(pid=os.getpid(), stage=stage, details=details, updated=time.time()))
    trained = False
    def record(stage, details, problem=None, complete=False):
        publish(stage, details, problem, progress_state=dict(training_complete=trained,
            acceptance_complete=complete, task_complete=complete))
    with (out / 'finish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            wait_data(plan, status)
            # The collector has exited; hold its lock for the remaining sequential work.
            with (out / 'controller.lock').open('a') as controller:
                fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if process_sample()['own_engines']:
                    raise PipelineStop('collector still has live engines')
                while True:
                    ready, available = training_ready()
                    status('waiting_training_resources', dict(available_bytes=available, ready=ready))
                    if ready: break
                    resource_gate(plan); time.sleep(30)
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                status('training', dict(device=device, available_bytes=available))
                checked_call(plan, training_command(plan, device), out / 'training.log')
                model = Path(plan['training']['out']) / 'model.pt'; model_sha = digest(model)
                ck = torch.load(model, map_location='cpu', weights_only=False)
                if not all(torch.isfinite(v).all().item() for v in ck['state_dict'].values()):
                    raise PipelineStop('nonfinite trained checkpoint')
                accuracy = json.loads((Path(plan['training']['out']) / 'pair_accuracy.json').read_text())
                if accuracy['v5']['checkpoint_sha256'] != model_sha:
                    raise PipelineStop('validation did not score the saved checkpoint')
                trained = True
                record('teacher_v5_training_complete', dict(sha256=model_sha, best_epoch=ck['epoch'],
                    pair_accuracy=accuracy, artifact_bytes=resource_gate(plan)))
                del ck
                evaluate(plan, model_sha, status, record)
                status('acceptance_audit', {})
                checked_call(plan, [sys.executable, str(ROOT / 'scripts/value_sparse_acceptance_check.py'),
                             str(out / 'plan.json')], evaluation / 'acceptance.log')
                acceptance = json.loads((evaluation / 'acceptance.json').read_text())
                receipt = upload_if_accepted(plan, acceptance)
                acceptance.update(hf_uploaded=receipt is not None, hf_receipt=receipt)
                write_json(evaluation / 'acceptance.json', acceptance)
                summary = {k: v for k, v in acceptance.items() if k != 'artifact_sha256'}
                summary.update(acceptance_sha256=digest(evaluation / 'acceptance.json'),
                    artifact_bytes=resource_gate(plan), recommendation='adopt v5' if receipt else 'retain v4')
                record('teacher_v5_task_complete', summary, complete=True)
                status('complete', summary)
        except BaseException as exc:
            status('stopped', dict(reason=str(exc)))
            record('teacher_v5_finish_stopped', dict(reason=str(exc)), str(exc))
            raise


if __name__ == '__main__':
    main()
