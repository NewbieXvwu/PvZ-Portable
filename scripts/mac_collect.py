"""Mac DAgger batches: production plan -> HF -> separate desktop frame source."""
import argparse
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FIRST_SEED, LAST_SEED = 1_970_000_000, 1_980_000_000


def snapshot(payload):
    """Executed on the desktop; all planning mutations remain in this process."""
    logs = Path.home() / 'pvz-logs'
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes()
        except (FileNotFoundError, PermissionError):
            continue
        if b'/scripts/curriculum_queue.py\0' in command:
            argv = command.decode().strip('\0').split('\0')
            break
    else:
        raise RuntimeError('production curriculum controller is not running')
    script = next(value for value in argv if value.endswith('/scripts/curriculum_queue.py'))
    root = Path(script).parents[1]
    sys.path[:0] = [str(root / 'scripts'), str(root / 'python')]
    import curriculum_queue as q
    from types import SimpleNamespace
    def option(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default
    state = json.loads((logs / 'curriculum_state.json').read_text())
    revision = next((logs / f'qc_deploy{version}.active').read_text().strip()
                    for version in (22, 21) if (logs / f'qc_deploy{version}.active').exists())
    head = dict(revision=revision, student=state['student'], round=state['round'],
        value=option('--value', str(root.parents[2] / 'artifacts/value_net/v5/model.pt')),
        static=option('--static-value', str(root.parents[2] / 'artifacts/hard/value_branch1/model.pt')),
        frames=option('--frames', str(root.parents[2] / 'artifacts/student/frames_v2')),
        obs_v3='--obs-v3' in argv)
    if payload.get('head_only'):
        return head
    print(f'planning production round {state["round"]}, {revision}', file=sys.stderr, flush=True)
    extras = [Path(argv[i + 1]) for i, value in enumerate(argv) if value == '--extra-manifest']
    tasks, values, names, manifests = q.load_panel(Path(option('--manifest')), extras)
    for flag, default in (('--opening-manifest', None),
                          ('--demo-start-manifest', str(root / 'configs/human_demo_tasks.json'))):
        path = option(flag, default)
        if path and Path(path).exists():
            entries = q.load_tasks(Path(path))
            tasks.update(entries)
            manifests.update({name: Path(path) for name in entries})
    state['completed_groups'] = []
    budget = int(option('--round-games', '768'))
    args = SimpleNamespace(log=Path('/dev/null'),
        teacher_rates=Path(option('--teacher-rates', str(logs / 'teacher_rates.json'))),
        gpt_import_dir=Path(option('--gpt-import-dir')) if option('--gpt-import-dir') else None,
        gpt_archive_root=Path(option('--gpt-archive-root')) if option('--gpt-archive-root') else None,
        gpt_cold_rates=Path(option('--gpt-cold-rates')) if option('--gpt-cold-rates') else None)
    with contextlib.redirect_stdout(io.StringIO()):
        catalog = q.refresh_live_curriculum(args, state, tasks) if hasattr(q, 'refresh_live_curriculum') else None
        for base, cell in state['cells'].items():
            if cell.get('archive') or cell.get('live193'):
                state['offsets'][cell.get('task_id', base)] = 0
                cell['seeds'] = list(range(FIRST_SEED, FIRST_SEED + budget))
        plan, _ = q.plan_round(state, tasks, names, values, manifests, budget,
            int(option('--games-per-group', '48')), q.load_teacher_rates(args.teacher_rates),
            save_candidates_levels=q.parse_levels(option('--save-candidates-levels', '31-40')),
            random_deck_share=float(option('--random-deck-share', '.15')),
            demo_start_share=float(option('--demo-start-share')) if option('--demo-start-share') else None,
            **(dict(live_catalog=catalog) if catalog is not None else {}))
    result = []
    for group in plan:
        cell = state['cells'][group.get('adaptation_base', group['base'])]
        rate = q.student_rate(cell, group.get('adaptation_value', group['multiplier']))
        starved = rate is not None and rate < float(option('--beta-threshold', '.1'))
        probe = cell.get('mixed_since_probe', 0) >= max(1, int(option('--probe-interval', '3'))) - 1
        group['beta'] = group.get('beta', float(option('--starved-beta', '.5')) if starved and not probe else 0.)
        task = dict(group.get('task_definition') or tasks[group['task_id']])
        if group.get('demo_archive'):
            meta = json.loads((Path(group['demo_archive']) / 'meta.json').read_text())
            task['seeds'] = [meta['task']['seed']]
            group['demo_files'] = [name for name in ('meta.json', 'ticks.jsonl', 'frames.npz', 'frames.jsonl')
                                   if (Path(group['demo_archive']) / name).is_file()]
        else:
            seeds = task['seeds'][group['offset']:group['offset'] + group['games']]
            task['seeds'] = seeds
        group['task'] = task
        group.pop('task_definition', None)
        result.append(group)
    print(f'planned {len(result)} production groups', file=sys.stderr, flush=True)
    return dict(head, plan=result)


def export(payload):
    from huggingface_hub import HfApi, CommitOperationAdd
    operations = [CommitOperationAdd(path_in_repo=payload['prefix'] + '/' + name, path_or_fileobj=path)
                  for name, path in payload['files'].items()]
    HfApi().create_commit(repo_id=payload['repo'], operations=operations, commit_message='Mac production collection inputs')
    return list(payload['files'])


def remote(args, action, payload):
    # Send code/input through stdin, including credentials only in the SSH process memory.
    source = Path(__file__).read_text()
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    program = 'import os\nos.environ["HF_TOKEN"]=' + repr(token or '') + '\n'
    program += 'namespace={"__name__":"mac197_remote","__file__":"/tmp/mac_collect.py"}\n'
    program += 'exec(' + repr(source) + ',namespace)\n'
    program += 'print(json.dumps(namespace[' + repr(action) + '](' + repr(payload) + ')))\n'
    # json is imported by the transmitted module in its namespace, not this wrapper.
    program = 'import json\n' + program
    response = subprocess.run(['ssh', '-o', 'ConnectTimeout=40', '-o', 'ServerAliveInterval=10',
        '-o', 'ServerAliveCountMax=2', args.host,
        'wsl -d Ubuntu -e /home/newbiexvwu/.venvs/ml/bin/python -'],
        input=program, text=True, stdout=subprocess.PIPE, check=True)
    return json.loads(response.stdout)


def reserve(cache, count):
    path = cache / 'cursor'
    with path.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        previous = handle.read().split()
        seed, group = map(int, previous or [FIRST_SEED, 0])
        if seed + count > LAST_SEED:
            raise RuntimeError('Mac collection seed segment exhausted')
        handle.seek(0); handle.truncate()
        handle.write(f'{seed + count} {group + 1}\n'); handle.flush(); os.fsync(handle.fileno())
    return seed, group


def code_at(args, revision):
    directory = args.cache / 'code'
    marker = args.cache / 'revision'
    if not marker.exists() or marker.read_text() != revision:
        subprocess.run(['git', '-C', str(ROOT), 'fetch', 'origin', 'deploy21', 'deploy22'], check=True)
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir()
        archive = subprocess.Popen(['git', '-C', str(ROOT), 'archive', revision,
                                    'python', 'scripts', 'configs', 'recipes', 'src'], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=archive.stdout, mode='r|') as bundle:
            bundle.extractall(directory, filter='data')
        if archive.wait():
            raise RuntimeError('git archive failed')
        marker.write_text(revision)
    return directory


def inputs(args, state, group, label):
    from huggingface_hub import HfApi, CommitOperationDelete, hf_hub_download
    directory = args.cache / 'models'
    directory.mkdir(exist_ok=True)
    student_name = 'student_' + Path(state['student']).parts[-3] + '.pt'
    files = {}
    for name, source in ((student_name, state['student']), ('value.pt', state['value']), ('static.pt', state['static'])):
        if not (directory / name).exists():
            files[name] = source
    shutil.rmtree(directory / 'demo', ignore_errors=True)
    if group.get('demo_archive'):
        # Only numerical replay files; game assets never enter HF.
        for name in group['demo_files']:
            files['demo/' + name] = group['demo_archive'] + '/' + name
    if files:
        prefix = 'transfer/mac197/inputs/' + label
        remote(args, 'export', dict(repo=args.repo, prefix=prefix, files=files))
        with contextlib.redirect_stdout(sys.stderr):
            for name in files:
                downloaded = Path(hf_hub_download(repo_id=args.repo, filename=prefix + '/' + name, local_dir=args.cache / 'download'))
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(downloaded, target)
        HfApi().create_commit(repo_id=args.repo, operations=[CommitOperationDelete(path_in_repo=prefix + '/' + name)
            for name in files], commit_message='Mac consumed collection inputs')
        shutil.rmtree(args.cache / 'download')
    for previous in directory.glob('student_*.pt'):
        if previous.name != student_name:
            previous.unlink()
    return directory, directory / student_name


def play(payload):
    """The matching production collector, with an in-memory task instead of a new manifest."""
    root = Path(payload['code'])
    sys.path[:0] = [str(root / 'scripts'), str(root / 'python')]
    import render_episode
    import curriculum_queue as q
    import search_play
    from types import SimpleNamespace
    task = payload['task']
    render_episode.load_training_task = lambda path, task_id: task
    original_kwargs = render_episode.training_task_kwargs
    def task_kwargs(definition, seed=None):
        original_seed = payload.get('deck_seeds', {}).get(str(seed), seed)
        kwargs = original_kwargs(definition, original_seed)
        kwargs['seed'] = seed
        return kwargs
    render_episode.training_task_kwargs = task_kwargs
    args = SimpleNamespace(**payload['args'])
    args.python = Path(sys.executable)
    args.out, args.frames = Path(args.out), Path(args.frames)
    command = q.command_for(args, Path('builtin:mac197'), task['task_id'], 0, payload['games'], payload['label'],
        workers=args.workers, beta=payload['beta'], demo_archive=payload.get('demo_archive'),
        demo_decision=payload.get('demo_decision'), future_seeds=payload.get('future_seeds'),
        save_candidates=payload.get('save_candidates', 0))
    if payload.get('max_actions'):
        search_play.MAX_ACTIONS = payload['max_actions']
    sys.argv = command[1:]
    search_play.main()


def deliver(args, folder, label, frames):
    archive = args.cache / (label + '.tar.gz')
    temporary = archive.with_suffix('.building')
    with tarfile.open(temporary, 'w:gz', compresslevel=1) as bundle:
        for path in sorted((folder / 'frames').iterdir()):
            bundle.add(path, arcname=path.name)
    temporary.rename(archive)
    send(args, archive, frames)


def send(args, archive, frames):
    from huggingface_hub import HfApi
    label = archive.name.removesuffix('.tar.gz')
    package = 'transfer/mac197/frames/' + archive.name
    HfApi().upload_file(repo_id=args.repo, path_in_repo=package, path_or_fileobj=archive)
    result = remote(args, 'import_batch', dict(repo=args.repo, package=package, frames=frames,
        import_source=(ROOT / 'scripts/mac_collect_import.py').read_text()))
    shutil.rmtree(args.cache / label)
    archive.unlink()
    print('delivered', result, flush=True)


def import_batch(payload):
    # Deployment installs the same tiny importer; the smoke run can use an isolated target.
    namespace = {'__name__': 'mac197_import'}
    exec(payload['import_source'], namespace)
    with contextlib.redirect_stdout(sys.stderr):
        return str(namespace['receive'](payload['repo'], payload['package'], Path(payload['frames'])))


def run(args):
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    args.cache.mkdir(parents=True, exist_ok=True)
    (args.cache / 'tmp').mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(args.cache / 'tmp')
    state = None
    for number in range(args.batches):
        full = state is None or number % len(state['plan']) == 0
        latest = remote(args, 'snapshot', dict(head_only=not full))
        if not full and (latest['revision'], latest['round']) != (state['revision'], state['round']):
            latest = remote(args, 'snapshot', {})
        state = dict(state or {}, **latest)
        seed, index = reserve(args.cache, args.games or max(group['games'] for group in state['plan']))
        group = state['plan'][index % len(state['plan'])]
        games = min(args.games, group['games']) if args.games else max(1, group['games'] // 2)
        label = f'mac197_{seed}_fast_search'
        folder = args.cache / label
        folder.mkdir()
        model, student = inputs(args, state, group, label)
        task = group['task']
        deck_seeds = {}
        if not group.get('demo_archive'):
            deck_seeds = {str(seed + i): old for i, old in enumerate(task['seeds'][:games])}
            task['seeds'] = list(range(seed, seed + games))
        payload = dict(code=str(code_at(args, state['revision'])), task=task, games=games,
            label=label.removesuffix('_fast_search'), beta=group['beta'], deck_seeds=deck_seeds,
            save_candidates=group.get('save_candidates', 0), max_actions=args.max_actions if args.profile else None,
            args=dict(rollout_policy=str(student), value=str(model / 'value.pt'), static_value=str(model / 'static.pt'),
                executable=str(args.executable), resource_dir=str(args.resource_dir), frames=str(folder / 'frames'),
                out=str(folder / 'games'), workers=args.workers, native_games=True, obs_v3=state['obs_v3']))
        if group.get('demo_archive'):
            payload.update(demo_archive=str(model / 'demo'), demo_decision=group['demo_decision'], future_seeds=list(range(seed, seed + games)))
        print(f'collect revision={state["revision"]} student={state["student"]} task={task["task_id"]} games={games} seed={seed} beta={group["beta"]}', flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), 'play'], input=json.dumps(payload), text=True, check=True)
        if args.profile:
            shutil.rmtree(folder)
        else:
            for path in (folder / 'frames').glob('*/*/meta.json'):
                meta = json.loads(path.read_text())
                meta['task']['policy'] = state['student']
                if group.get('demo_archive'):
                    meta['demo_archive'] = group['demo_archive']
                path.write_text(json.dumps(meta, indent=2) + '\n')
        shutil.rmtree(model / 'demo', ignore_errors=True)
        shutil.rmtree(args.cache / 'tmp')
        (args.cache / 'tmp').mkdir()
        if not args.profile:
            deliver(args, folder, label, args.destination or state['frames'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'play', 'send'))
    parser.add_argument('--host', default='newbiexvwu@100.120.155.85')
    parser.add_argument('--repo', default=os.environ.get('PVZ_HF_REPO'))
    parser.add_argument('--cache', type=Path, default=Path.home() / 'pvz-mac-collect')
    parser.add_argument('--executable', type=Path, default=Path.home() / 'pvz-mac-collect/build/pvz-portable')
    parser.add_argument('--resource-dir', type=Path, default=Path.home() / 'Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--batches', type=int, default=1)
    parser.add_argument('--games', type=int)
    parser.add_argument('--destination', help='isolated desktop frame root for the smoke batch')
    parser.add_argument('--batch-path', type=Path, help='completed .tar.gz retained after a failed delivery')
    parser.add_argument('--profile', action='store_true', help='PGO workload only; discard these frames')
    parser.add_argument('--max-actions', type=int, help='PGO workload action budget')
    args = parser.parse_args()
    if args.action != 'play':
        for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            os.environ.pop(name, None)
    if args.action == 'play':
        play(json.load(sys.stdin))
    elif args.action == 'send':
        if args.batch_path is None or args.destination is None:
            parser.error('send requires --batch-path and --destination')
        send(args, args.batch_path, args.destination)
    else:
        run(args)


if __name__ == '__main__':
    main()
