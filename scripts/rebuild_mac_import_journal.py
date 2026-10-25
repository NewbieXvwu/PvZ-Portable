"""Rebuild lost Mac import receipts from surviving ledgers and the local corpus.

No network access, downloads, extraction or frame writes. Unknown original package
hashes stay unknown; import_mac_labels refuses automatic re-import of those receipts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import import_mac_labels as mac


def rebuild(*, remote_paths, labels_file, destination=mac.DESTINATION, state=mac.STATE,
            legacy_reports=mac.LEGACY_REPORTS):
    destination, state = mac.absolute(destination), mac.absolute(state)
    paths = {line.strip() for line in Path(remote_paths).read_text().splitlines() if line.strip()}
    if Path(labels_file).exists():
        paths.update('transfer/' + line.strip().removesuffix('_fast_search') + '.tgz'
                     for line in Path(labels_file).read_text().splitlines() if line.strip())
    labels = {label for path in paths if (label := mac.label_from_remote(path))}
    known, introduced, excluded, repos = {}, set(), set(), set()
    for report in legacy_reports:
        if not Path(report).is_file():
            continue
        data = mac.read(report)
        if data.get('destination', str(destination)) != str(destination):
            raise ValueError('legacy receipt belongs to another corpus: ' + str(report))
        if data.get('repo'):
            repos.add(data['repo'])
        known.update({row['label']: row for row in data.get('rounds', [])})
        introduced.update(data.get('introduced_labels', []))
        excluded.update(data.get('excluded_archival_variants', []))
    if len(repos) != 1:
        raise ValueError('need one unambiguous repository in the surviving ledgers')
    # Single traversal rather than a glob over every seed directory for every label.
    catalog = defaultdict(list)
    for seed in destination.iterdir():
        if not seed.name.startswith('seed_') or not seed.is_dir():
            continue
        for leaf in seed.iterdir():
            label = leaf.name.removesuffix('_fast_search')
            if label in labels:
                catalog[label].append(leaf)
    missing = labels - catalog.keys()
    if missing:
        raise ValueError('import marker without local frames: ' + ','.join(sorted(missing)))
    records = []
    with mac.import_lock(state):
        if any((state / name).exists()
               for name in ('mac_import.json', 'mac_package_journal.json')):
            raise ValueError('journal already exists; refuse to overwrite live import receipts')
        for label in sorted(labels):
            entries = []
            for leaf in sorted(catalog[label]):
                for path in (leaf / 'frames.npz', leaf / 'meta.json'):
                    if not path.is_file() or not path.resolve().is_relative_to(destination.resolve()):
                        raise ValueError('missing or escaping corpus member: ' + str(path))
                    entries.append((str(path.relative_to(destination)), mac.sha(path)))
                if mac.read(leaf / 'meta.json')['task']['seed'] != int(leaf.parent.name[5:]):
                    raise ValueError('metadata seed mismatch: ' + str(leaf))
            old = known.get(label, {})
            row = {**old, 'label': label, 'remote_path': f'transfer/{label}.tgz',
                   'archives': old.get('archives', len(catalog[label])),
                   'local_archives_observed': len(catalog[label]),
                   'local_ledger_sha256': mac.ledger_digest(entries),
                   'recovered_local_receipt': True, 'refresh_added_files': 0,
                   'recovery_note': 'Local bytes verified; original package SHA256 only retained '
                                    'where a surviving legacy receipt supplies it.'}
            records.append(row)
            if not label.startswith('z') or label + '_fast_search' in introduced:
                introduced.add(label + '_fast_search')
            print(json.dumps({'label': label, 'archives': len(catalog[label]),
                              'original_package_hash_known': bool(old.get('archive_sha256'))}),
                  flush=True)
        report = dict(repo=next(iter(repos)), revision=None, destination=str(destination),
                      rounds=records, introduced_labels=sorted(introduced),
                      excluded_archival_variants=sorted(excluded),
                      recovery=dict(remote_snapshot_verified=False, remote_paths=str(remote_paths),
                                    labels_file=str(labels_file),
                                    unsupported_remote_paths=sorted(
                                        p for p in paths if mac.label_from_remote(p) is None)))
        mac.write(state / 'mac_package_journal.json', report)
        mac.write(state / 'mac_import.json', {**report, 'added_files_this_refresh': 0})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-paths', type=Path, required=True)
    parser.add_argument('--labels-file', type=Path, required=True)
    parser.add_argument('--destination', type=Path, default=mac.DESTINATION)
    parser.add_argument('--state', type=Path, default=mac.STATE)
    args = parser.parse_args()
    rebuild(remote_paths=args.remote_paths, labels_file=args.labels_file,
            destination=args.destination, state=args.state)


if __name__ == '__main__':
    main()
