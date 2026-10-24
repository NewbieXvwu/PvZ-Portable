"""Archive actual production-search games through episode_query, with optional diagnostics."""
from __future__ import annotations
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def collect_search(resource_dir, seed, checkpoint, manifest, task_id, value, static_value,
                   executable=None, timeline=None, arm='fast_search'):
    import search_play as sp
    from curriculum_queue import command_for
    from pvz_frame_store import frame_path, read_frames
    from fog_timeline import Timeline
    from render_episode import _mower_fired
    from pvz_fog import task45_engine_pids
    if task45_engine_pids():
        raise RuntimeError('capture requires no other task45 game process')
    with tempfile.TemporaryDirectory(prefix='pvz-search-archive-') as temporary:
        temp = Path(temporary)
        cmd_args = SimpleNamespace(python=Path(sys.executable), rollout_policy=Path(checkpoint),
            value=Path(value), static_value=Path(static_value or value),
            resource_dir=Path(resource_dir), executable=Path(executable or ROOT/'build/pvz-portable'),
            workers=1, frames=temp/'frames', out=temp/'result')
        cmd = command_for(cmd_args, Path(manifest), task_id, 0, 1, 'capture')
        args = sp.parser().parse_args(cmd[2:])
        args.dagger_beta = 1
        observer = Timeline(timeline or temp/'timeline.jsonl')
        keep, paths = sp._export_values(str(value), str(static_value or value))
        try:
            sp._load_nets(paths[0], str(checkpoint), paths[1])
            sp._init_worker(paths[0], str(checkpoint), args.rollout_sim_skip, paths[1])
            result = sp._play(dict(seed=seed, arm=arm, args=args), observer=observer)
        finally:
            observer.close()
            sp._close_worker_env()
            if sp._W.get('rollout_tmp'):
                sp._W['rollout_tmp'].cleanup()
            sp._W.clear()
            keep.cleanup()
        archive = temp/'frames'/f'seed_{seed}'/f'capture_{arm}'
        meta = json.loads((archive/'meta.json').read_text())
        frames = read_frames(frame_path(archive))
        final = frames[-1]
        meta['task'].update(policy='teacher' if arm=='fast_search' else 'student',
            search_recipe={'verify_samples':9,'student_candidates':2,'horizon':2500,
                           'samples':3,'rollout_sim_skip':61,'beta':1},
            manifest=str(manifest), task_id=task_id,
            value=str(value), static_value=str(static_value or value))
        outcome = dict(meta['outcome'], result=final['result'],
            reason='通关' if result['won'] else '僵尸进屋' if final['terminal'] else '动作上限截断',
            actions=result['decisions'], rejected_actions=sum(bool(f.get('action_rejected')) for f in frames),
            mowers_used=result['mowers_used'], plants_eaten=observer.events['plants_eaten'])
        return dict(task=meta['task'], outcome=outcome, totals=dict(observer.events),
                    frames=frames, log=[], search_result=result)
