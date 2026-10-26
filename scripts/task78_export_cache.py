"""Reuse a successful panel export while its files and inputs are unchanged."""
import hashlib
import json
from pathlib import Path

W = Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/v6')


def fingerprint(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt_path(export):
    return Path(str(export) + '.reuse.json')


def request(manifest, out_dir, export, offset):
    return dict(manifest=str(Path(manifest).resolve()),
                out_dir=str(Path(out_dir).resolve()),
                export=str(Path(export).resolve()), offset=int(offset))


def ready(manifest, out_dir, export, offset):
    try:
        saved = json.loads(receipt_path(export).read_text())
        return (saved['request'] == request(manifest, out_dir, export, offset)
                and saved['normal_export_complete'] is True
                and saved['export_sha256'] == digest(export)
                and all(fingerprint(p) == value for p, value in saved['files'].items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def seal(manifest, out_dir, export, offset):
    source = json.loads(Path(manifest).read_text())
    document = json.loads(Path(export).read_text())
    expected = {(i + int(offset), h) for i in range(len(source['archives']))
                for h in (2500, 4000, 6000)}
    actual = {(s['archive_index'], s['horizon']) for s in document['shards']}
    if (document.get('complete') is not True or document.get('training_eligible') is not False
            or document['source_manifest_sha256'] != digest(manifest)
            or document['index_offset'] != int(offset)
            or actual != expected or len(document['shards']) != len(expected)):
        raise ValueError('only a successful complete panel export may be sealed')
    replay = Path(manifest).with_name('replay.json')
    files = {str(Path(manifest)), str(Path(export)), str(replay)}
    files.add(json.loads(replay.read_text())['executable'])
    for row in source['archives']:
        files.update((row['frames_path'], str(Path(row['archive']) / 'meta.json')))
    for s in document['shards']:
        files.update(s['provenance'])
        files.update(str(Path(p).parent) for p in s['provenance'])
        if s.get('path'):
            files.add(s['path'])
        for p in s['provenance']:
            path = Path(p)
            if path.name == 'complete.json':
                files.update(v['path'] for v in json.loads(path.read_text())['identity']['models'].values())
            elif path.name == 'root.json':
                files.add(str(path.with_name('summary.json')))
    files.update(str(W / 'scripts' / name) for name in (
        'task78_export.py', 'task78_collect.py', 'task78_data.py', 'task78_state.py',
        'task78_student_context.py', 'search_play.py', 'value_teacher_feasibility.py',
        'task78_native_history.py', 'task78_native_labels.py',
        'value_teacher_feasibility_report.py', 'value_widen_audit.py'))
    files.add(str(W / 'python/pvz_value_net.py'))
    files.add(str(W / 'python/pvz_student_observation.py'))
    files.add(str(W / 'python/pvz_rollout_policy.py'))
    files.add(str(Path(__file__).resolve()))
    saved = dict(request=request(manifest, out_dir, export, offset),
                 normal_export_complete=True, export_sha256=digest(export),
                 files={p: fingerprint(p) for p in sorted(files)},
                 validation='file metadata reuse; final raw-label scoring remains unchanged')
    target = receipt_path(export)
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(saved, indent=1) + '\n')
    temporary.replace(target)
    return saved
