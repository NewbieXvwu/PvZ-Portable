"""Shared episode metadata and the three numerical archive files."""
import json
from pathlib import Path
import shutil

from pvz_frame_store import frame_path, read_frames, write_frames

SCHEMA = 'episode_archive_v2'
SCHEMA_VERSION = 2
READ_SCHEMAS = ('episode_archive_v1', SCHEMA)
FILES = ('meta.json', 'ticks.jsonl', 'frames.npz')


def metadata(document):
    meta = dict(document)
    task = dict(meta.get('task') or {})
    task.setdefault('id', task.get('task_id'))
    task.setdefault('deck', list(meta.get('deck') or []))
    outcome = dict(meta.get('outcome') or {})
    outcome.setdefault('result', 1 if outcome.get('won') else 0)
    outcome.setdefault('won', outcome['result'] == 1)
    for key in ('reason', 'final_tick', 'final_wave', 'wave_count'):
        outcome.setdefault(key, None)
    meta.update(schema=SCHEMA, schema_version=SCHEMA_VERSION, task=task,
                deck=list(task['deck']), outcome=outcome)
    for key, default in (('label_source', 'unknown'), ('environment_task', {}),
                         ('context', {}), ('totals', {}), ('verified', False),
                         ('frame_count', 0)):
        meta.setdefault(key, default)
    meta.setdefault('operations', max(0, meta['frame_count'] - 1))
    return meta


def read_meta(archive):
    return metadata(json.loads((Path(archive) / 'meta.json').read_text()))


def write_meta(archive, document):
    meta = metadata(document)
    (Path(archive) / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    return meta


def copy_archive(source, target, *, consume=False):
    """Copy only numerical archive files; old JSONL is converted once on admission."""
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    transfer = shutil.move if consume else shutil.copyfile
    for path in (source / 'meta.json', source / 'ticks.jsonl', frame_path(source)):
        transfer(path, target / path.name)
    convert_archive(target)


def convert_archive(archive):
    """Upgrade an existing admitted archive without replaying it or changing its receipt."""
    archive = Path(archive)
    if not (archive / 'frames.npz').is_file():
        write_frames(archive / 'frames.npz', read_frames(frame_path(archive)), cache=False)
    (archive / 'frames.jsonl').unlink(missing_ok=True)
    meta = json.loads((archive / 'meta.json').read_text())
    if (meta.get('schema'), meta.get('schema_version')) != (SCHEMA, SCHEMA_VERSION):
        write_meta(archive, meta)
