"""Small locks for concurrent task105 scans and their shared text metadata."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def metadata_lock(root: Path):
    return file_lock(root / 'artifacts/task105/delivery.lock')
