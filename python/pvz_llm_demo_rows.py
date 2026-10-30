"""task105 receipts select optional full-trajectory supervision, without replay."""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from pvz_frame_store import frame_path, read_frames, write_frames

DEPLOY14_ENGINE = '76ff36dc026f67641aefccd5f30f6cdae65eee14ee197dc24497413123a7874a'
DEMO_LABEL = 'llm_demo_decisions'


def demo_paths(data: Path, imported: Path, results=(1,), runs=None):
    ledger = json.loads((imported / 'merged_progress.json').read_text())
    selected = {}
    for row in ledger['archives']:
        if row.get('verify') != '通过' or row.get('result') not in results:
            continue
        sha = row['archive_sha256']
        if not isinstance(sha, str) or re.fullmatch('[0-9a-f]{64}', sha) is None:
            raise ValueError('invalid task105 archive SHA')
        if type(row.get('seed')) is not int or row['seed'] < 0:
            raise ValueError('invalid task105 seed')
        receipt = json.loads((imported / 'receipts' / f'{sha}.json').read_text())
        if (receipt.get('archive_sha256') != sha or receipt.get('verified') is not True
                or receipt.get('context', {}).get('engine') != DEPLOY14_ENGINE):
            continue
        selected[sha] = row
    paths = []
    destination = data / '.llm_demo_rows'
    for sha, row in sorted(selected.items()):
        relative = Path(f'seed_{row["seed"]}') / DEMO_LABEL / 'frames.npz'
        target = destination / sha
        source = imported / 'sha256' / sha
        meta = json.loads((source / 'meta.json').read_text())
        if runs and not any(str(meta.get('run_id', '')).startswith(prefix) for prefix in runs):
            continue
        level = meta['task'].get('level', (meta.get('environment_task') or {}).get('level'))
        if level is not None and 31 <= level <= 40 and level != 35 and (
                (meta.get('environment_task') or {}).get('fog_observation') != 'human_v2'):
            # Without the recorded human-view renderer the model could see through fog.
            continue
        if not target.exists():
            if (meta['task']['seed'] != row['seed'] or meta['outcome']['result'] != row['result']
                    or not str(meta.get('label_source', '')).startswith('llm_')):
                raise ValueError(f'{source}: task105 provenance differs')
            frames = read_frames(frame_path(source))
            meta['training_rows'] = [index for index, frame in enumerate(frames) if index > 0
                                     and (frame.get('action') or {}).get('type') in ('plant', 'wait')
                                     and not frame.get('action_rejected')]
            meta['llm_demo_sha256'] = sha
            destination.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination) as temporary:
                staged = Path(temporary) / 'archive'
                frame = staged / relative
                frame.parent.mkdir(parents=True)
                write_frames(frame, frames)
                (frame.parent / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False) + '\n')
                staged.rename(target)
        paths.append(target / relative)
    return paths
