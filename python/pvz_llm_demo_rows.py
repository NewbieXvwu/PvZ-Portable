"""task152 GPT trajectory rows, converted after unified admission and cached without replay."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pvz_frame_store import frame_path, read_frames, write_frames

DEMO_LABEL = 'llm_demo_decisions'


def demo_batch(data: Path, path: Path) -> str:
    """Import batch (archive sha) that owns a converted archive; a new import packs alone."""
    try:
        parts = Path(path).relative_to(data).parts
    except ValueError:
        return 'external'
    if '.llm_demo_rows' in parts:
        index = parts.index('.llm_demo_rows')
        return parts[index + 1] if index + 1 < len(parts) else 'unknown'
    return 'external'


def demo_paths(data: Path, imported: Path, results=(1,)):
    """Read the task194/task152 converted pool; admission belongs to import_llm_run."""
    cache = imported / '.llm_demo_rows'
    if not cache.is_dir():
        raise FileNotFoundError(cache)
    target = data / '.llm_demo_rows'
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(cache.resolve(), target_is_directory=True)
    return [path for path in sorted(target.glob('*/seed_*/' + DEMO_LABEL + '/frames.npz'))
            if json.loads((path.parent / 'meta.json').read_text())['outcome']['result'] in results]


def import_demo(data, source, sha, results=(1,), *, min_operation=0, skip_operations=(),
                force=False):
    """task152 conversion, called after the unified importer has verified an archive.

    ``min_operation`` / ``skip_operations`` are task238's drill/scenario filter: the
    prefix and the injection operations are not GPT decisions, so they never become
    demonstration rows.  ``force`` admits a short-horizon success whose engine result
    is still 0 (the drill stopped at its horizon).
    """
    meta = json.loads((source / 'meta.json').read_text())
    if not str(meta.get('label_source', '')).startswith('llm_'):
        return None
    if not force and meta['outcome']['result'] not in results:
        return None
    level = meta['task'].get('level', meta.get('context', {}).get('level'))
    if level in (31, 32, 33, 34, 36, 37, 38, 39, 40) and meta.get('environment_task', {}).get('fog_observation') != 'human_v2':
        return None
    relative = Path(f'seed_{meta["task"]["seed"]}') / DEMO_LABEL / 'frames.npz'
    destination = Path(data) / '.llm_demo_rows'
    target = destination / sha
    if not target.exists():
        frames = read_frames(frame_path(source))
        # task224：中盘接回存档的 frames 从截断点开始；截断点本身和学生前缀都不能变成示范行。
        resume = meta.get('resume') or {}
        cutoff = resume.get('cutoff_tick')
        prefix = resume.get('prefix_operations')
        min_operation = max(int(min_operation), int(prefix or 0))
        blocked = {int(value) for value in skip_operations or ()}
        rows = []
        for index, frame in enumerate(frames):
            action = frame.get('action') or {}
            operation = int(frame.get('operation') or index)
            if not index or frame.get('action_rejected') or action.get('type') not in ('plant', 'wait'):
                continue
            if operation <= int(min_operation) or operation in blocked:
                continue
            if prefix is None and cutoff is not None and float(frame.get('tick') or 0) <= float(cutoff):
                continue
            if action['type'] == 'plant' and not any(
                    card[0] == action['packet'] and card[1] >= 0 for card in frames[index - 1]['packets']):
                continue
            rows.append(index)
        meta.update(training_rows=rows, llm_demo_sha256=sha)
        meta.setdefault('context', {}).setdefault('level', level)
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination) as temporary:
            staged = Path(temporary) / 'archive'
            frame = staged / relative
            frame.parent.mkdir(parents=True)
            write_frames(frame, frames, cache=False)
            from pvz_episode_archive import write_meta
            write_meta(frame.parent, meta)
            staged.rename(target)
    return target / relative
