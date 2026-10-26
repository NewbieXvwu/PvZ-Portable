"""Run task93's frozen two-draw experiment, serially, after its live collector exits.

Does not collect games. Evaluations use two workers, and run only after the attached
collector has exited. QC training takes priority over this experiment's GPU jobs.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'artifacts/task93'
CONTROL = Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/gatepower/artifacts/task84')
PANEL = Path('/home/newbiexvwu/PvZ-search/artifacts/stage4/random_deck_panel.json')
ENGINE = Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build4/pvz-portable')
VALUE = Path('/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt')
INCUMBENT = Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/round_000013/evaluations/a41fdfa96075ae608a40/incumbent')
LOCK_FD = None
QC_STATE = Path.home()/'pvz-logs/curriculum_state.json'
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
from pvz_tasks import deck_for_seed
from task93_analyze import evaluation_config, load_games


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def student_training():
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().split(b'\0')
        except (OSError, ProcessLookupError):
            continue
        # Match individual argv entries; never match a shell's quoted script text.
        if any(arg.endswith(b'/train_rollout_policy.py') and b' ' not in arg and b'\n' not in arg for arg in argv):
            text = b' '.join(argv).decode(errors='replace')
            found.append((int(path.parent.name), text))
    return found


def qc_training():
    return [pid for pid, text in student_training()
            if '/cq/' in text or 'curriculum_state.pipeline' in text]


class QCPreempted(RuntimeError):
    """Only an actual priority preemption may retry a training attempt."""


def qc_controllers():
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().split(b'\0')
        except OSError:
            continue
        if any(arg.endswith(b'/curriculum_queue.py') and b'/cq/' in arg
               and b' ' not in arg and b'\n' not in arg for arg in argv):
            found.append(int(path.parent.name))
    return found


def qc_pending():
    # A stale state file alone is not evidence of a queued production job.
    if not qc_controllers():
        return False
    try:
        state = json.loads(QC_STATE.read_text())
        return any(job.get('status') == 'queued' for job in state.get('training_jobs', []))
    except (OSError, ValueError, TypeError, AttributeError):
        return None  # retry the observation; do not terminate a worker on a read failure


def training_memory_requirement():
    proof = DATA/'seed_1/trained.json'
    peak = json.loads(proof.read_text()).get('observed_peak_rss_bytes') if proof.exists() else None
    # Same corpus, model, batch size and loader for seed2. Use measured resident
    # memory plus 15%, with a 12 GiB floor; retain the initial 16 GiB fallback.
    return max(12*1024**3, (peak*115+99)//100) if type(peak) is int and peak > 0 else 16*1024**3


def complete_evaluation(games, task, model):
    if not games.exists():
        return False
    fallback, _ = evaluation_config(games, task)
    config = json.loads(games.with_name('config.json').read_text())
    if config['rollout_policy'] != str(model):
        raise ValueError(f'evaluation model differs: {games}')
    rows = load_games(games, task['seeds'], 'fast',
                      {s: deck_for_seed(task, s) for s in task['seeds']},
                      legacy_decks=fallback, require_complete=False)
    return set(rows) == set(task['seeds'])


def complete_pack(recipe):
    import numpy as np
    from pvz_recipe import training_row_options
    from pvz_training_pack import archive_fingerprint, pack_options, pack_path
    data = Path(recipe['data'])
    options = training_row_options(recipe)
    options.pop('source_weight', None)
    options = pack_options(**options, sources=archive_fingerprint(data, set(recipe['labels'])))
    path = pack_path(data, options)
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as stored:
        if json.loads(str(stored['meta'])) != options:
            raise ValueError('pack row contract differs')
        return all((path.parent/str(segment)).is_file() for segment in stored['segments'])


def status(phase, **extra):
    document = dict(pid=os.getpid(), phase=phase, updated=time.time(), **extra)
    temporary = DATA/'validation_status.tmp'
    temporary.write_text(json.dumps(document, indent=2)+'\n')
    temporary.replace(DATA/'validation_status.json')


def run(command, name, *, gpu=False):
    while gpu:
        active = student_training()
        available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        required = training_memory_requirement()
        pending = qc_pending()
        if not active and pending is False and available >= required:
            break
        status('waiting_for_qc_or_memory', training_pids=[pid for pid, _ in active],
               qc_pids=qc_training(), qc_pending=pending,
               available_bytes=available, required_bytes=required, next=name)
        time.sleep(10)
    print(name, flush=True)
    environment = {**os.environ, 'PVZ_NO_POPUP': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    if not gpu:
        environment['CUDA_VISIBLE_DEVICES'] = ''
    with (DATA/f'{name}.log').open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                 pass_fds=() if LOCK_FD is None else (LOCK_FD,))
        status(name, child_pid=child.pid, command=command)
        while child.poll() is None:
            if gpu and (qc_training() or qc_pending() is True):
                # Yield only our process. Do not signal or modify the production queue.
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                with (DATA/'preemptions.jsonl').open('a') as record:
                    record.write(json.dumps(dict(phase=name, training_pid=child.pid,
                        reason='yielded to QC', time=time.time(), complete_model=False))+'\n')
                raise QCPreempted(f'{name}: yielded to QC; no completed model claimed')
            time.sleep(5)
        if child.returncode:
            raise RuntimeError(f'{name}: exit {child.returncode}; see {DATA/name}.log')
        return child.pid


def complete_model(seed):
    folder = DATA/f'seed_{seed}'
    model = folder/'student/model.pt'
    metrics = folder/'student/metrics.json'
    proof = folder/'trained.json'
    if not proof.exists():
        return False
    record = json.loads(proof.read_text())
    if (not model.exists() or not metrics.exists() or record['model_sha256'] != sha(model)
            or record['recipe_sha256'] != sha(folder/'recipe.json')
            or len(json.loads(metrics.read_text())['history']) != 12):
        raise ValueError(f'invalid saved training proof: {folder}')
    return True


def train_seed(seed):
    folder = DATA/f'seed_{seed}'
    while not complete_model(seed):
        if any((folder/'student'/name).exists() for name in ('model.pt', 'metrics.json')):
            raise ValueError('uncertified training output requires verification before retraining')
        try:
            training_pid = run(['nice', '-n', '10', sys.executable,
                str(ROOT/'scripts/train_rollout_policy.py'),
                '--recipe', str(folder/'recipe.json')], f'train_seed_{seed}', gpu=True)
        except QCPreempted as error:
            status('waiting_for_qc_or_memory', next=f'train_seed_{seed}',
                   reason=str(error), retry_same_frozen_recipe=True)
            continue
        metrics = json.loads((folder/'student/metrics.json').read_text())
        if len(metrics['history']) != 12:
            raise ValueError('incomplete training history')
        proof = dict(model_sha256=sha(folder/'student/model.pt'),
                     recipe_sha256=sha(folder/'recipe.json'), epochs=12, training_pid=training_pid)
        telemetry = Path.home()/f'pvz-logs/metrics/train_rollout_policy-{training_pid}.jsonl'
        if telemetry.exists():
            samples = [json.loads(line) for line in telemetry.read_text().splitlines()]
            proof.update(observed_peak_rss_bytes=max(row.get('rss_bytes') or 0 for row in samples),
                         rss_telemetry_sha256=sha(telemetry))
        (folder/'trained.json').write_text(json.dumps(proof, indent=2)+'\n')


def experiment():
    python = sys.executable
    plan = DATA/'training_plan.json'
    if not plan.exists() or not json.loads(plan.read_text()).get('prepared'):
        run([python, str(ROOT/'scripts/task93_prepare_training.py'), '--baseline-recipe',
             str(CONTROL/'variant_a/recipe.json'), '--manifest', str(PANEL)], 'prepare')
    recipes = [json.loads((DATA/f'seed_{seed}/recipe.json').read_text()) for seed in (1, 2)]
    if (recipes[1] != {**recipes[0], 'seed': 2, 'out': str(DATA/'seed_2/student')}
            or recipes[0]['seed'] != 1):
        raise ValueError('frozen two-seed recipes differ beyond seed/output')
    # Use the recipe row contract and the cached baseline segments without rebuilding them.
    pack_code = """import json,sys
from pathlib import Path
sys.path[:0]=[sys.argv[1]+'/python',sys.argv[1]+'/scripts']
from pack_training_rows import pack
from pvz_recipe import training_row_options
from pvz_training_pack import archive_fingerprint,pack_options
r=json.loads(Path(sys.argv[2]).read_text());data=Path(r['data']);o=training_row_options(r)
o.pop('source_weight',None)
options=pack_options(**o,sources=archive_fingerprint(data,set(r['labels'])))
pack(data,options,cache_dir=Path(r['cache_dir']),read_cache_dir=Path(r['read_cache_dir']),precompute_workers=r['precompute_workers'])
"""
    if not complete_pack(recipes[0]):
        run([python, '-c', pack_code, str(ROOT), str(DATA/'seed_1/recipe.json')], 'pack')
    baseline_rows = json.loads((CONTROL/'variant_a/student/source_rows.json').read_text())
    for seed in (1, 2):
        folder = DATA/f'seed_{seed}'
        train_seed(seed)
        rows = json.loads((folder/'student/source_rows.json').read_text())
        for label, old in baseline_rows['sources'].items():
            # New seeds can change the validation split; retained row counts must be identical.
            if any(rows['sources'][label][key] != old[key] for key in ('rows', 'archives')):
                raise ValueError(f'baseline corpus changed: {label}')
        for filename in ('heldout_eval_manifest.json', 'control_eval_manifest.json'):
            manifest = DATA/filename
            source = (CONTROL/filename).read_bytes()
            if manifest.exists() and manifest.read_bytes() != source:
                raise ValueError(f'evaluation manifest changed: {manifest}')
            manifest.write_bytes(source)
            task = json.loads(source)['tasks'][0]
            out = folder/task['task_id']
            games = out/'games.jsonl'
            if complete_evaluation(games, task, folder/'student/model.pt'):
                continue
            run([python, str(ROOT/'scripts/search_play.py'), '--rollout-policy', str(folder/'student/model.pt'),
                 '--value', str(VALUE), '--manifest', str(manifest), '--task-id', task['task_id'],
                 '--executable', str(ENGINE), '--resource-dir', '/home/newbiexvwu/.cache/pvz-research-resources',
                 '--seeds', '192', '--workers', '2', '--arms', 'fast', '--dagger-beta', '0', '--out', str(out)],
                f'eval_seed_{seed}_{task["task_id"]}')
            fallback, _ = evaluation_config(games, task)
            load_games(games, task['seeds'], 'fast', {s: deck_for_seed(task,s) for s in task['seeds']},
                       legacy_decks=fallback)
    for filename in ('heldout_eval_manifest.json', 'control_eval_manifest.json'):
        task = json.loads((DATA/filename).read_text())['tasks'][0]
        command = [python, str(ROOT/'scripts/task93_analyze.py'), '--manifest', str(DATA/filename),
                   '--out', str(DATA/f'comparison_{task["task_id"]}.json'), 'compare', '--task-id', task['task_id']]
        for seed in (1, 2):
            command += ['--candidate', f'seed_{seed}={DATA}/seed_{seed}/{task["task_id"]}/games.jsonl']
        for name, variant in (('A', 'a'), ('D2', 'd2')):
            command += ['--baseline', f'{name}={CONTROL}/variant_{variant}/eval_{task["task_id"]}/games.jsonl']
        command += ['--incumbent', str(INCUMBENT/task['task_id']/'games.jsonl')]
        run(command, f'compare_{task["task_id"]}')
    status('complete')


def main():
    global LOCK_FD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collector-pid', type=int, help='attach to the existing serial collection shell')
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA/'validation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        LOCK_FD = lock.fileno()
        try:
            status('verifying_completed_stages')
            print('verifying completed stages', flush=True)
            if args.collector_pid:
                handle = os.pidfd_open(args.collector_pid)
                try:
                    status('waiting_for_collector', collector_pid=args.collector_pid)
                    print(f'attached to collector {args.collector_pid}', flush=True)
                    while not select.select([handle], [], [], 10)[0]:
                        pass
                finally:
                    os.close(handle)
            experiment()
        except Exception as error:
            status('failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
