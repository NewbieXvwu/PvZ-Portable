"""Incrementally import Mac HF transfer labels into the shared npz corpus.

Only HF_TOKEN and PVZ_HF_REPO are read for HF configuration. Reports and the
per-package crash journal live outside git; old task23/task27 ledgers seed them.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile

import numpy as np

MAIN = Path('/home/newbiexvwu/PvZ-search')
DESTINATION = MAIN / 'artifacts/student/frames_v2'
STATE = MAIN / 'artifacts/mac_import'
LEGACY_REPORTS = (
    MAIN / 'artifacts/archive/dw4/task23/mac_import.json',
    MAIN / 'artifacts/archive/dw5/task27/mac_import.json',
    MAIN / 'artifacts/archive/dw5/task27/mac_package_journal.json',
)
LABEL_PATTERNS = {
    'high_difficulty': re.compile(r'z\d+dag\d+[bc]'),
    'day': re.compile(r'd\d+(?:dag\d+[bm]|p\d+m)'),
    'night': re.compile(r'n\d+dag\d+m'),
    'pool': re.compile(r'p\d+w\d+m'),
}
MIN_FREE = 15 * 1024**3


def hf_environment():
    """Require an exported repository; never execute or inspect shell profiles."""
    repo = os.environ.get('PVZ_HF_REPO')
    if not repo:
        raise ValueError('export PVZ_HF_REPO (and HF_TOKEN for a private repository)')
    return repo


def utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def default_delete_file(path_in_repo, *, repo_id, token):
    """huggingface_hub.delete_file, imported lazily like the rest of the HF surface."""
    from huggingface_hub import delete_file
    delete_file(path_in_repo, repo_id=repo_id, token=token)


def delete_remote(record, *, repo, token, deleter=None):
    """Delete one verified package from HF and journal the deletion time.

    Failures are recorded on the record (never silent) and the package simply stays
    on the Hub; the next refresh/cleanup retries it.
    """
    try:
        (deleter or default_delete_file)(record['remote_path'], repo_id=repo, token=token)
    except Exception as error:  # noqa: BLE001 - every failure mode must land in the journal
        record['remote_delete_error'] = str(error)
        record.pop('remote_deleted_at', None)
        return False
    record.pop('remote_delete_error', None)
    record['remote_deleted_at'] = utc_now()
    return True


def verified_import(record):
    """True when the journal receipt proves this exact package was imported and verified."""
    return bool(record.get('archive_sha256')) and record.get('status') != 'frozen_selection'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    """Atomically replace a report so an interruption cannot truncate the journal."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, indent=2)
        stream.write('\n')
    os.replace(temporary, path)


def absolute(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('absolute path required: ' + str(path))
    return path


def check_disk(required_bytes=0, disk=Path('/mnt/c')):
    free = shutil.disk_usage(disk).free
    if free - required_bytes < MIN_FREE:
        raise RuntimeError(f'问题: {disk} free={free} bytes; need 15 GiB reserve '
                           f'plus {required_bytes} bytes; import stopped')
    return free


def label_from_remote(filename):
    if not filename.startswith('transfer/') or not filename.endswith('.tgz'):
        return None
    label = filename[len('transfer/'):-len('.tgz')]
    return label if any(p.fullmatch(label) for p in LABEL_PATTERNS.values()) else None


def ledger_digest(entries):
    return hashlib.sha256(json.dumps(sorted(entries), separators=(',', ':')).encode()).hexdigest()


def training_rows(raw):
    """Full rows before wait sampling/splitting, as pack_training_rows builds them.

    Count frame i-1 -> labelled action at i, excluding rejected/missing actions.
    Read the lossless frames rather than trust a possibly obsolete encoder cache.
    """
    import io
    with np.load(io.BytesIO(raw), allow_pickle=False) as stored:
        lines = stored['frames'].tobytes().splitlines()
    return sum(bool(frame.get('label') or frame.get('action')) and not frame.get('action_rejected')
               for frame in (json.loads(line) for line in lines[1:] if line.strip()))


def local_files(destination, label):
    return sorted(destination.glob(f'seed_*/{label}_fast_search/frames.npz'))


def local_matches(destination, label, record):
    frames = local_files(destination, label)
    if len(frames) != record.get('archives'):
        return False
    files = [p for frame in frames for p in (frame, frame.with_name('meta.json'))]
    if not all(p.is_file() and p.resolve().is_relative_to(destination.resolve()) for p in files):
        return False
    expected = record.get('local_ledger_sha256')
    return not expected or ledger_digest([(str(p.relative_to(destination)), sha(p)) for p in files]) == expected


def import_archive(path, destination, label, *, disk_check=check_disk):
    """Preflight and stage a whole package; exclusive atomic links never overwrite data."""
    destination = absolute(destination)
    if label_from_remote(f'transfer/{label}.tgz') != label:
        raise ValueError('unsupported Mac label: ' + label)
    destination.mkdir(parents=True, exist_ok=True)
    added = existing = rows = 0
    ledger = []
    with tarfile.open(path, 'r:gz') as archive:
        members = {}
        pairs = {}
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if '..' in parts or Path(member.name).is_absolute():
                raise ValueError('unsafe archive path: ' + member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError('nonregular archive entry: ' + member.name)
            indices = [i for i, part in enumerate(parts) if re.fullmatch(r'seed_\d+', part)]
            if len(indices) != 1:
                raise ValueError('missing/ambiguous seed directory: ' + member.name)
            relative = Path(*parts[indices[0]:])
            if (len(relative.parts) != 3 or relative.parts[1] != label + '_fast_search'
                    or relative.name not in ('frames.npz', 'meta.json')):
                raise ValueError('unexpected archive layout: ' + member.name)
            if relative in members:
                raise ValueError('duplicate archive member: ' + str(relative))
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('escaping existing symlink: ' + str(relative))
            members[relative] = member
            pairs.setdefault(relative.parts[0], set()).add(relative.name)
        if not pairs or any(names != {'frames.npz', 'meta.json'} for names in pairs.values()):
            raise ValueError('empty or incomplete frame/meta pairs')
        disk_check(sum(m.size for m in members.values()))
        # Same filesystem makes link() atomic, with no half-written file on interruption.
        with tempfile.TemporaryDirectory(prefix='.mac-stage-', dir=destination) as directory:
            stage = Path(directory)
            for relative, member in members.items():
                raw = archive.extractfile(member).read()
                digest = hashlib.sha256(raw).hexdigest()
                ledger.append((str(relative), digest))
                if relative.name == 'frames.npz':
                    rows += training_rows(raw)
                else:
                    metadata = json.loads(raw)
                    if metadata['task']['seed'] != int(relative.parts[0].split('_')[1]):
                        raise ValueError('metadata seed mismatch: ' + str(relative))
                target = destination / relative
                if target.exists() and sha(target) != digest:
                    raise ValueError('existing content differs (never replaced): ' + str(relative))
                staged = stage / relative
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(raw)
            for relative, digest in ledger:
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError('escaping existing symlink: ' + str(relative))
                try:
                    os.link(stage / relative, target)
                    added += 1
                except FileExistsError:
                    if sha(target) != digest:
                        raise ValueError('concurrent content differs: ' + str(relative))
                    existing += 1
    return dict(episodes=len(pairs), archives=len(pairs), rows=rows, added_files=added,
                existing_identical_files=existing, archive_sha256=sha(path),
                local_ledger_sha256=ledger_digest(ledger))


@contextmanager
def import_lock(state):
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'import.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another Mac importer is running for this state directory') from None
        yield


def refresh(*, destination=DESTINATION, state=STATE, labels=None, legacy_reports=LEGACY_REPORTS,
            api=None, download=None, repo=None, disk_check=check_disk, deleter=None,
            delete_remote_packages=True):
    """Snapshot HF once, reconcile legacy records, checkpoint each completed label."""
    from huggingface_hub import HfApi, hf_hub_download
    destination, state = absolute(destination), absolute(state)
    repo = repo or hf_environment()
    api = api if api is not None else HfApi(token=os.environ.get('HF_TOKEN') or False)
    download = download if download is not None else hf_hub_download
    with import_lock(state):
        disk_check()
        info = api.model_info(repo, files_metadata=True)
        entries = {label: entry for entry in info.siblings
                   if (label := label_from_remote(entry.rfilename))}
        selected = set(entries) if labels is None else set(labels)
        if selected - entries.keys():
            raise ValueError('requested labels missing from HF: ' + ','.join(sorted(selected - entries.keys())))
        known, introduced, excluded = {}, set(), set()
        for report in (*legacy_reports, state / 'mac_import.json', state / 'mac_package_journal.json'):
            if not Path(report).exists():
                continue
            data = read(report)
            # A receipt from another corpus/repository cannot authorize a skip here.
            if (data.get('destination', str(DESTINATION)) != str(destination)
                    or data.get('repo', repo) != repo):
                continue
            known.update({row['label']: row for row in data.get('rounds', [])})
            introduced.update(data.get('introduced_labels', []))
            excluded.update(data.get('excluded_archival_variants', []))
        eligible = {label + '_fast_search' for label in entries}
        excluded.update(introduced - eligible)
        introduced.intersection_update(eligible)

        def checkpoint(path, records):
            data = dict(repo=repo, revision=info.sha, destination=str(destination), rounds=records,
                        introduced_labels=sorted(introduced), excluded_archival_variants=sorted(excluded))
            write(path, data)
            return data

        records = []
        for label in sorted(selected):
            entry, old = entries[label], known.get(label)
            remote_sha = getattr(entry.lfs, 'sha256', None)
            if not remote_sha or not re.fullmatch(r'[a-f0-9]{64}', remote_sha):
                raise ValueError('missing HF LFS SHA256: ' + entry.rfilename)
            frozen = (label.startswith('z') and label + '_fast_search' not in introduced
                      and bool(local_files(destination, label)))
            if frozen:
                frames = local_files(destination, label)
                if not all(p.with_name('meta.json').is_file() for p in frames):
                    raise ValueError('incomplete frozen selection: ' + label)
                result = dict(label=label, remote_path=entry.rfilename, archives=len(frames),
                              already_present_old_round=True, remote_sha256=remote_sha,
                              refresh_added_files=0, status='frozen_selection',
                              note='Historical z selection preserved; remote package not downloaded or claimed verified.')
            elif (old and old.get('archive_sha256') == remote_sha
                  and local_matches(destination, label, old)):
                result = {**old, 'refresh_added_files': 0, 'reused_verified_package': True}
            else:
                # Persist intent first so a crash during a new z label cannot freeze a partial import.
                introduced.add(label + '_fast_search')
                checkpoint(state / 'mac_package_journal.json', list(known.values()))
                disk_check(entry.size or 0)
                with tempfile.TemporaryDirectory(prefix='download-', dir=state) as directory:
                    path = Path(download(repo, entry.rfilename, revision=info.sha, local_dir=directory,
                                         token=os.environ.get('HF_TOKEN') or False))
                    if sha(path) != remote_sha:
                        raise ValueError('HF LFS SHA256 mismatch: ' + entry.rfilename)
                    result = dict(label=label, remote_path=entry.rfilename,
                                  **import_archive(path, destination, label, disk_check=disk_check))
                result['refresh_added_files'] = result['added_files']
                result['reused_verified_package'] = False
            known[label] = result
            checkpoint(state / 'mac_package_journal.json', list(known.values()))
            # The package is now provably imported (or re-verified against the local corpus);
            # the Hub copy is redundant.  Frozen selections were never verified, so they stay.
            if delete_remote_packages and verified_import(result) and label in entries:
                if delete_remote(result, repo=repo, token=os.environ.get('HF_TOKEN') or False,
                                 deleter=deleter):
                    entries.pop(label, None)
                checkpoint(state / 'mac_package_journal.json', list(known.values()))
            records.append(result)
            summary = {key: result[key] for key in ('label', 'archives', 'refresh_added_files')}
            if 'rows' in result:
                summary['rows'] = result['rows']
            if result.get('remote_deleted_at'):
                summary['remote_deleted_at'] = result['remote_deleted_at']
            if result.get('remote_delete_error'):
                summary['remote_delete_error'] = result['remote_delete_error']
            print(json.dumps(summary), flush=True)
        report = checkpoint(state / 'mac_import.json', records)
        report['added_files_this_refresh'] = sum(row['refresh_added_files'] for row in records)
        write(state / 'mac_import.json', report)
        return report


def cleanup_remote(*, destination=DESTINATION, state=STATE, legacy_reports=LEGACY_REPORTS,
                   api=None, repo=None, deleter=None, apply=False):
    """One-time sweep: delete HF transfer/ packages whose receipts prove a verified import.

    A package is deleted only when the journal receipt carries the archive sha of the
    current remote upload AND the local corpus still matches that receipt
    (local_matches).  Everything else is listed as kept, never silently dropped.
    Dry-run by default; apply=True deletes and journals the deletion times.
    """
    from huggingface_hub import HfApi
    destination, state = absolute(destination), absolute(state)
    repo = repo or hf_environment()
    api = api if api is not None else HfApi(token=os.environ.get('HF_TOKEN') or False)
    token = os.environ.get('HF_TOKEN') or False
    with import_lock(state):
        info = api.model_info(repo, files_metadata=True)
        entries = {label: entry for entry in info.siblings
                   if (label := label_from_remote(entry.rfilename))}
        known = {}
        for report in (*legacy_reports, state / 'mac_package_journal.json', state / 'mac_import.json'):
            if not Path(report).exists():
                continue
            data = read(report)
            # A receipt from another corpus/repository cannot authorize a deletion here.
            if (data.get('destination', str(DESTINATION)) != str(destination)
                    or data.get('repo', repo) != repo):
                continue
            known.update({row['label']: row for row in data.get('rounds', [])})
        deletable, kept, no_receipt = [], [], []
        for label in sorted(entries):
            record = known.get(label)
            remote_sha = getattr(entries[label].lfs, 'sha256', None)
            if not record or not verified_import(record) or record.get('archive_sha256') != remote_sha:
                no_receipt.append(label)
                continue
            if not local_matches(destination, label, record):
                kept.append(dict(label=label, reason='local corpus does not match the receipt'))
                continue
            deletable.append(label)
        report = dict(repo=repo, revision=info.sha, apply=apply, deletable=deletable,
                      kept=kept, no_verified_receipt=no_receipt, deleted=[], errors=[])
        if apply:
            updated = {}
            for label in deletable:
                record = dict(known[label], remote_path=f'transfer/{label}.tgz')
                if delete_remote(record, repo=repo, token=token, deleter=deleter):
                    report['deleted'].append(dict(label=label, remote_deleted_at=record['remote_deleted_at']))
                    updated[label] = record
                else:
                    report['errors'].append(dict(label=label,
                                                 remote_delete_error=record['remote_delete_error']))
            if updated:
                journal_path = state / 'mac_package_journal.json'
                data = read(journal_path) if journal_path.exists() else dict(
                    rounds=[], introduced_labels=[], excluded_archival_variants=[])
                rounds = {row['label']: row for row in data.get('rounds', [])}
                rounds.update(updated)
                data['rounds'] = [rounds[key] for key in sorted(rounds)]
                write(journal_path, data)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=DESTINATION)
    parser.add_argument('--state', type=Path, default=STATE)
    parser.add_argument('--report', type=Path, help='also write the completed report to this absolute path')
    parser.add_argument('--labels', help='optional comma-separated bare labels; default: every eligible Mac upload')
    parser.add_argument('--cleanup-remote-dry-run', action='store_true',
                        help='list HF transfer/ packages that a verified-import cleanup would delete')
    parser.add_argument('--cleanup-remote', action='store_true',
                        help='apply the cleanup: delete verified-import packages from HF')
    args = parser.parse_args()
    if args.report:
        absolute(args.report)
    if args.cleanup_remote or args.cleanup_remote_dry_run:
        print(json.dumps(cleanup_remote(destination=args.destination, state=args.state,
                                        apply=args.cleanup_remote), ensure_ascii=False, indent=1))
        return
    result = refresh(destination=args.destination, state=args.state,
                     labels=args.labels.split(',') if args.labels else None)
    if args.report:
        write(args.report, result)


if __name__ == '__main__':
    main()
