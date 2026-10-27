"""Small file checks and reversible code overlay bookkeeping for task131.

No model replay, archive scan, training, or writes to production state/ledgers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    document = json.loads(args.baselines.read_text())
    baseline = document['baselines'][args.service]
    if args.action in ('preflight', 'save', 'restored'):
        for name, expected in baseline.items():
            if sha(args.root / name) != expected:
                raise ValueError(f'deploy14{args.service} baseline changed: {name}; refresh deploy15 package')
        if args.action != 'save':
            for path, expected in ((args.root / 'build6/pvz-portable', document['old_engine_sha256']),
                                   (args.package / 'build7/pvz-portable', document['new_engine_sha256'])):
                if sha(path) != expected:
                    raise ValueError(f'engine changed: {path}')
    if args.action == 'save':
        for name, expected in baseline.items():
            if expected is not None:
                target = args.output / 'code.before' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.root / name, target)
    elif args.action == 'restore':
        for name, expected in baseline.items():
            target = args.root / name
            if expected is None:
                target.unlink(missing_ok=True)
            else:
                backup = args.output / 'code.before' / name
                if sha(backup) != expected:
                    raise ValueError(f'code backup changed: {name}')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, target)
    elif args.action == 'installed':
        for name in baseline:
            source = subprocess.check_output(['git', '-C', str(args.package), 'show',
                                              f'{args.revision}:{name}'])
            if sha(args.root / name) != hashlib.sha256(source).hexdigest():
                raise ValueError(f'candidate overlay differs: {name}')
    elif args.action == 'boundary':
        state = json.loads(args.state.read_text())
        if any(job.get('status') != 'done' and not job.get('isolated')
               for job in state.get('training_jobs', [])):
            raise ValueError('queued/running training or gate jobs remain; wait for a boundary')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'save', 'restore', 'restored', 'installed', 'boundary'))
    for name in ('root', 'output', 'package', 'baselines'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--service', choices=('g', 'h', 'i'), required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--state', type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
