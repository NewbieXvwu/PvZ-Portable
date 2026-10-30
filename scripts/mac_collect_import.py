"""Receive a Mac frame batch outside the production pack path."""
import argparse
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile


def batch_sources(frames):
    return sorted(path for path in (Path(frames) / 'mac197').glob('mac197_*') if path.is_dir())


def receive(repo, package, frames):
    from huggingface_hub import HfApi, hf_hub_download
    root = Path(frames) / 'mac197'
    root.mkdir(parents=True, exist_ok=True)
    label = PurePosixPath(package).name.removesuffix('.tar.gz')
    target = root / label
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix='.incoming-', dir=root) as temporary:
            temporary = Path(temporary)
            archive = hf_hub_download(repo_id=repo, filename=package, local_dir=temporary)
            with tarfile.open(archive) as bundle:
                bundle.extractall(temporary / 'frames', filter='data')
            (temporary / 'frames').rename(target)
    HfApi().delete_file(repo_id=repo, path_in_repo=package)
    print(target, flush=True)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package')
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--repo', default=os.environ.get('PVZ_HF_REPO'))
    args = parser.parse_args()
    receive(args.repo, args.package, args.frames)
