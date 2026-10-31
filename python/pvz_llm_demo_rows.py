"""task152 GPT trajectory rows, converted after unified admission and cached without replay."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pvz_frame_store import frame_path, read_frames, write_frames

DEMO_LABEL = 'llm_demo_decisions'


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


def import_demo(data, source, sha, results=(1,)):
    """task152 conversion, called after the unified importer has verified an archive."""
    meta = json.loads((source / 'meta.json').read_text())
    if not str(meta.get('label_source', '')).startswith('llm_') or meta['outcome']['result'] not in results:
        return None
    level = meta['task'].get('level', meta.get('context', {}).get('level'))
    if level in (31, 32, 33, 34, 36, 37, 38, 39, 40) and meta.get('environment_task', {}).get('fog_observation') != 'human_v2':
        return None
    relative = Path(f'seed_{meta["task"]["seed"]}') / DEMO_LABEL / 'frames.npz'
    destination = Path(data) / '.llm_demo_rows'
    target = destination / sha
    if not target.exists():
        frames = read_frames(frame_path(source))
        rows = []
        for index, frame in enumerate(frames):
            action = frame.get('action') or {}
            if not index or frame.get('action_rejected') or action.get('type') not in ('plant', 'wait'):
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
