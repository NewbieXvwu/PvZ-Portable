"""task268: sequential, resumable level-1 NPZ migration; run after deploying the new reader."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from pvz_training_pack import atomic_savez, npz_members, shared_lock


def prepare_index(data: Path):
    """Upgrade legacy records from ZIP directories once, outside the training lock."""
    marker = data / '.merge/shared-v2/index.json'
    if not marker.is_file():
        return
    snapshot = json.loads(marker.read_text())
    identities = {}
    for signature, record in snapshot['segments'].items():
        if 'members' in record:
            continue
        name, size, mtime = json.loads(signature)
        path = Path(name)
        if not path.is_file():
            continue
        stat = path.stat()
        if [stat.st_size, stat.st_mtime_ns] == record.get('stamp', [size, mtime]):
            members = npz_members(path)
            after = path.stat()
            if (stat.st_ino, stat.st_size, stat.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns):
                identities[signature] = members
    if not identities:
        return
    with shared_lock(data):
        if not marker.is_file():
            return
        current = json.loads(marker.read_text())
        for signature, members in identities.items():
            if signature in current['segments']:
                current['segments'][signature].setdefault('members', members)
        building = marker.with_suffix('.building')
        building.write_text(json.dumps(current))
        building.replace(marker)


def migrate_segment(path: Path, data: Path) -> int:
    """Compress outside shared.lock; publish the identity before replacing the source."""
    before = path.stat()
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if all(member.compress_type == zipfile.ZIP_DEFLATED for member in members):
            return 0
        # Compressed output plus ZIP overhead fits below the 2 GB temporary-file budget.
        if sum(member.file_size for member in members) > 1_990_000_000:
            raise ValueError(f'{path}: exceeds the 2 GB migration budget')
    temporary = path.with_suffix('.task268')
    try:
        atomic_savez(temporary, source=path)
        with shared_lock(data):
            current = path.stat()
            if (current.st_ino, current.st_size, current.st_mtime_ns) != (
                    before.st_ino, before.st_size, before.st_mtime_ns):
                return 0  # A packer published a new generation while this copy was compressing.
            os.replace(temporary, path)
        return before.st_size - path.stat().st_size
    finally:
        temporary.unlink(missing_ok=True)


def disk_free(disk):
    result = subprocess.check_output(['df', '-B1', str(disk)], text=True)
    available = int(result.splitlines()[-1].split()[3])
    print(result.strip(), flush=True)
    return available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='pipeline frames root')
    parser.add_argument('--disk', type=Path, default=Path('/mnt/c'))
    args = parser.parse_args()
    directory = args.data / 'packs/segments'
    # One migration owns the one-file temporary budget. A second invocation exits immediately.
    with (directory / '.task268.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for orphan in [*directory.glob('.*.task268.*.building'), *directory.glob('*.task268')]:
            orphan.unlink()
        since_check = saved = converted = 0
        if disk_free(args.disk) < 20_000_000_000:
            return
        prepare_index(args.data)
        started = time.monotonic()
        for path in sorted(directory.glob('*.npz')):
            size = path.stat().st_size
            gain = migrate_segment(path, args.data)
            if gain:
                converted += 1
                saved += gain
                since_check += size
                print(json.dumps(dict(file=path.name, saved=gain, total_saved=saved,
                                      converted=converted, seconds=round(time.monotonic()-started, 2))), flush=True)
            if since_check >= 1_000_000_000:
                since_check = 0
                if disk_free(args.disk) < 20_000_000_000:
                    break


if __name__ == '__main__':
    main()
