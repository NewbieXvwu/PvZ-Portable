"""Resume the frozen width screen after preserving its closed smoke timeout."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import signal
import shutil
import subprocess
import time
import traceback
from huggingface_hub import HfApi

ROOT = Path('/home/newbiexvwu/PvZ-Portable')
OUT = ROOT/'artifacts/research/mainline_training_queue_v11'
OUT.mkdir(parents=True, exist_ok=True)
if (OUT/'progress.json').exists():
    raise RuntimeError('Existing width controller state: inspect its owner; do not restart or duplicate a live queue')
PY = '/home/newbiexvwu/.venvs/ml/bin/python'
plan = json.loads((ROOT/'experiments/t6/informative_width_v1/queue_v2.json').read_text())
if json.loads((OUT/'plan.json').read_text()) != plan:
    raise RuntimeError('Saved width execution plan differs from frozen tracked plan')
SOURCE = Path(plan['source_worktree'])
api = HfApi(token=os.environ.get('HF_TOKEN'))
repo = os.environ['PVZ_HF_REPO']
uploads = []
owned_child = None
report = dict(schema_version=1, status='checking_closed_timeout',
              controller_pid=os.getpid(), controller_script=str(Path(__file__).relative_to(ROOT)),
              controller_revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
              plan=plan, stages=[], hf_deliveries=[],
              wall_points={}, acceptance='T5-E sustained3/5, T6 actual comparisons and original full acceptance remain unachieved.')


def now():
    return datetime.now(timezone.utc).isoformat()


def save():
    report['utc'] = now()
    temporary = OUT/'progress.json.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(OUT/'progress.json')


def state(run):
    path = ROOT/'artifacts/research'/run/'training_state.json'
    return json.loads(path.read_text()) if path.exists() else {}


def compact(run):
    s = state(run)
    result = dict(run=run, **{k:s.get(k) for k in
                 ('updates','counters','wall_seconds','phase','status','resources')})
    if s.get('update_history'):
        result['latest_loss'] = s['update_history'][-1]['losses']
    return result


def collect_uploads():
    for child, stream, entry in uploads:
        if entry['status'] != 'uploading' or child.poll() is None:
            continue
        stream.close()
        entry.update(exit_code=child.returncode, end_utc=now())
        if child.returncode:
            entry['status'] = 'failed_preserved'
        else:
            try:
                entry.update(status='complete', repo=repo,
                             revision=api.repo_info(repo_id=repo, revision=entry['branch']).sha)
            except Exception:
                entry.update(status='uploaded_revision_resolution_pending', repo=repo,
                             revision_error=traceback.format_exc())
        save()


def start_upload(run, suffix, extra_logs=()):
    entry = dict(run=run, branch='evidence-'+run.replace('_','-')+'-'+suffix,
                 status='preparing', start_utc=now())
    report['hf_deliveries'].append(entry)
    save()
    try:
        api.create_branch(repo_id=repo, branch=entry['branch'],
                          revision='e860a9c97634b2db1f47ac0dae7563138b2c1e91', exist_ok=True)
        stream = (ROOT/'logs/t5_research'/f'{run}_queue_v11_hf.log').open('a')
        command = [PY, 'scripts/hf_sync.py', '--revision', entry['branch'], 'push', run,
                   '--include-rollouts', '--include-trained-window']
        logs = [ROOT/'logs/t5_research'/f'{run}.log', Path(__file__).resolve(), OUT/'plan.json',
                OUT/'progress.json', ROOT/'logs/t5_research/mainline_training_queue_v11.log',
                *extra_logs]
        used_names = set()
        for path in logs:
            if path.name in used_names:
                directory = OUT/'delivery_metadata'/run
                directory.mkdir(parents=True, exist_ok=True)
                renamed = directory/f'{path.parent.name}_{path.name}'
                shutil.copyfile(path, renamed)
                path = renamed
            if path.name in used_names:
                raise RuntimeError('Duplicate external delivery name after explicit parent prefix')
            used_names.add(path.name)
            command.extend(['--log', str(path)])
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        entry.update(status='uploading', pid=child.pid)
        uploads.append((child, stream, entry))
        save()
    except Exception:
        entry.update(status='delivery_failed_preserved', error=traceback.format_exc())
        save()


def retain_wall_points(run):
    """Preserve actual policies before normal trained-window pruning can remove them."""
    s = state(run)
    points = report['wall_points'].setdefault(run, [])
    due = [cap for cap in plan['wall_budgets_seconds']
           if s.get('wall_seconds', 0) >= cap
           and not any(p['wall_cap_seconds'] == cap for p in points)]
    if not due:
        return
    started = time.monotonic()
    import torch
    directory = ROOT/'artifacts/research'/run/'runs/run_1'
    candidates = []
    for path in directory.glob('*.pt'):
        if path.name.startswith('wallcap_'):
            continue
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        except FileNotFoundError:
            continue  # The owning trainer can prune an old trained file meanwhile.
        cs = checkpoint['training_state']
        candidates.append((float(cs['wall_seconds']), cs['updates'],
                           dict(cs['counters']), path))
        del checkpoint
    for cap in due:
        eligible = [p for p in candidates if p[0] <= cap]
        if not eligible:
            points.append(dict(wall_cap_seconds=cap, status='no_retained_policy_before_cap'))
            continue
        seconds, update, counters, source = max(eligible, key=lambda p:(p[0],p[1]))
        target = directory/f'wallcap_{cap:.6f}_update_{update:06d}_{time.time_ns()}.pt'
        os.link(source, target)
        points.append(dict(wall_cap_seconds=cap, status='retained_scoring_pending',
                           active_wall_seconds=seconds, unused_budget_seconds=cap-seconds,
                           updates=update, counters=counters,
                           original_checkpoint=str(source.relative_to(directory.parent.parent)),
                           retained_checkpoint=str(target.relative_to(directory.parent.parent)),
                           selection_seconds=time.monotonic()-started,
                           scope='Actual unchanged complete-update policy, hardlinked without modifying resume state. No interpolation or evaluation/adoption claim.'))
    save()


def train(entry, updates, resume, engineering_probe=False):
    global owned_child
    config = json.loads((SOURCE/entry['config']).read_text())
    run = config['experiment_id']
    output = ROOT/entry['output_dir']
    if not resume and (output/'training_state.json').exists():
        raise RuntimeError('Fresh width run already exists; refuse accidental reinitialization')
    stage = dict(run=run, seed=entry['seed'], width=entry['width'],
                 requested_updates=updates, resume=resume, engineering_probe=engineering_probe, start_utc=now(),
                 start_counters=state(run).get('counters'), status='running')
    command = [PY, '-u', 'python/train_pvz_ppo_task_family.py', '--resource-dir',
               '/home/newbiexvwu/.cache/pvz-research-resources', '--experiment-config',
               entry['config'], '--output-dir', str(output), '--stop-after-updates', str(updates)]
    if resume:
        command.append('--resume')
    stage['command'] = command
    started = time.monotonic()
    sent = False
    with (ROOT/entry['log']).open('a') as log:
        child = subprocess.Popen(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        owned_child = child
        stage['pid'] = child.pid
        report['stages'].append(stage)
        report['status'] = 'training_width_candidate'
        save()
        while child.poll() is None:
            stage['latest'] = compact(run)
            s = state(run)
            if s:
                retain_wall_points(run)
            elapsed = time.monotonic()-started
            if not sent and resume and s.get('counters',{}).get('decisions',0) >= plan['target_decisions']:
                child.send_signal(signal.SIGTERM)
                stage['requested_complete_target_stop'] = now()
                sent = True
            if not sent and (not resume or engineering_probe) and elapsed > plan['smoke_timeout_seconds']:
                child.send_signal(signal.SIGTERM)
                stage['smoke_timeout'] = True
                sent = True
            collect_uploads()
            save()
            time.sleep(30)
    stage.update(exit_code=child.wait(), seconds=time.monotonic()-started,
                 final=compact(run))
    owned_child = None
    if stage['exit_code'] or stage.get('smoke_timeout'):
        stage['status'] = 'failed_preserved'
        save()
        raise RuntimeError('Required width smoke/resume failed; preserve all state and stop expansion')
    s = state(run)
    if (s.get('phase') != 'ready' or s.get('status') not in ('update_boundary_stop','budget_complete')
            or not s.get('learning_curve') or not (output/s['checkpoint']).is_file()
            or not (output/'evaluations/idle_control.json.gz').is_file()):
        raise RuntimeError('Required checkpoint, completed evaluation/idle control or complete state missing')
    if resume and s['counters']['decisions'] <= stage['start_counters']['decisions']:
        raise RuntimeError('Completed width resume made no actual interaction progress')
    retain_wall_points(run)
    stage['status'] = 'complete'
    save()


save()
try:
    prior=json.loads((ROOT/plan['prior_queue']).read_text())
    if prior['status']!='failed_preserved' or not prior['stages'][-1].get('smoke_timeout'):
        raise RuntimeError('Require the preserved original timeout, not a live or relabeled queue')
    for stage in prior['stages']:
        if Path(f"/proc/{stage['pid']}").exists():
            raise RuntimeError('Prior trainer still present; no overlapping GPU trainer')
    if Path(f"/proc/{prior['controller_pid']}").exists():
        raise RuntimeError('Prior controller still present')
    report['prior_final']=prior
    failed=plan['prior_run']
    if state(failed)['phase']!='ready' or not any(n['counters']['decisions']==plan['prior_required_decisions'] for n in state(failed)['learning_curve']):
        raise RuntimeError('Require the original complete2k evaluation and closed restore boundary')
    start_upload(failed,'smoke-timeout',
        [ROOT/plan['prior_queue'],ROOT/'artifacts/research/mainline_training_queue_v10/plan.json',
         ROOT/'scripts/run_informative_width_queue.py',ROOT/'logs/t5_research/mainline_training_queue_v10.log'])
    for entry in plan['engineering_stages']:
        if entry['resume']:
            while any(child.poll() is None for child,_,delivery in uploads if delivery['run']==failed):
                collect_uploads();save();time.sleep(30)
            collect_uploads()
            if not any(d['run']==failed and d['status']=='complete' for d in report['hf_deliveries']):
                raise RuntimeError('Original failed full-run delivery incomplete; do not overwrite its working state')
        train(entry,plan['smoke_updates'],entry['resume'],engineering_probe=True)
    for entry in plan['order']:
        run=json.loads((SOURCE/entry['config']).read_text())['experiment_id'];s=state(run)
        output=ROOT/entry['output_dir']
        if (s.get('phase')!='ready' or s['updates']<1 or not s.get('learning_curve')
                or not (output/s['checkpoint']).is_file()
                or not (output/'evaluations/idle_control.json.gz').is_file()):
            raise RuntimeError('All six actual engineering closures required before expansion')
        if s.get('initialization_provenance') or s['invocations'][0]['kind']!='random_initialization':
            raise RuntimeError('Every width retains its own genuine random origin')
    report['all_six_engineering_closures_complete_utc']=now();save()
    round_number=0
    while True:
        remaining=[e for e in plan['order'] if state(json.loads((SOURCE/e['config']).read_text())['experiment_id'])['counters']['decisions']<plan['target_decisions']]
        if not remaining:break
        round_number+=1;report['round']=round_number
        for entry in remaining:
            train(entry,plan['round_updates'][0 if round_number==1 else 1],True)
            run=json.loads((SOURCE/entry['config']).read_text())['experiment_id'];s=state(run)
            if s['counters']['decisions']>=plan['target_decisions']:
                if not any(n['counters']['decisions']==plan['target_decisions'] for n in s['learning_curve']):
                    raise RuntimeError('Require original completed125k assessment')
                start_upload(run,'first125k-width')
    report['status']='six_width125k_complete_uploads_pending_assessment';save()
    while any(p.poll() is None for p,_,_ in uploads):
        collect_uploads();save();time.sleep(30)
    collect_uploads();report['status']='six_width125k_complete_pending_actual_comparison';save()
except BaseException:
    report.update(status='failed_preserved',error=traceback.format_exc());save()
    if owned_child is not None and owned_child.poll() is None:
        owned_child.send_signal(signal.SIGTERM);report['owned_child_complete_update_stop_requested']=now();save()
        while owned_child.poll() is None:time.sleep(30);save()
        report['owned_child_final_exit_code']=owned_child.wait();save()
    raise
