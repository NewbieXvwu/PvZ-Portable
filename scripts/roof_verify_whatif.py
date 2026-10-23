"""Independently replay Gemini's roof interventions and every pre-branch frame."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('PVZ_NO_POPUP', '1')
from episode_query import load, _replay_task_extra, _parse_action, _hand_at
from render_episode import collect


def proposals(level):
    folder = ROOT / f'artifacts/task46/phase2/gemini-{level}'
    points = {}
    for response in sorted(folder.glob('response-*.json')):
        result = json.loads(response.read_text())
        for part in result['candidates'][0]['content'].get('parts', []):
            call = part.get('functionCall')
            if not call or call.get('args', {}).get('command') != 'whatif':
                continue
            args = call['args']['arguments']
            decision = int(args[args.index('--decision') + 1])
            actions = [args[i+1] for i, x in enumerate(args) if x == '--try']
            points.setdefault(decision, set()).update(actions)
    return points


def main():
    results = []
    for level in range(42, 46):
        archive = ROOT / f'artifacts/task46/phase2/l{level}-45000000-teacher'
        meta, frames = load(archive)
        task = meta['task']
        points = proposals(level)
        assert len(points) >= 2, (level, points)
        for decision, actions in sorted(points.items()):
            baseline = frames[decision]['action']
            parsed = [(None, baseline)]
            for raw in sorted(actions):
                try:
                    action = _parse_action(raw, _hand_at(frames, decision, task['deck']))
                except ValueError:
                    continue  # Invalid teacher spelling is not an intervention.
                if action != baseline:
                    parsed.append((raw, action))
            for raw, action in parsed:
                rec = collect('/home/newbiexvwu/.cache/pvz-research-resources', task['seed'], level,
                    'student', 4000, task['deck'], override={decision: action},
                    prefix_actions=[f['action'] for f in frames[1:decision]],
                    checkpoint=task['checkpoint'], task_extra=_replay_task_extra(task))
                actual = json.loads(json.dumps(rec['frames']))
                excluded = {'action', 'eaten', 'targets', 'label', 'verification', 'action_rejected'}
                keys = set(frames[0]) & set(actual[0]) - excluded
                for i in range(decision):
                    assert {k: frames[i][k] for k in keys} == {k: actual[i][k] for k in keys}, (
                        level, decision, raw, i)
                results.append(dict(level=level, seed=task['seed'], decision=decision,
                    root_tick=frames[decision-1]['tick'], raw_action=raw, action=action,
                    baseline=raw is None, prefix_frames_equal=decision, compared_fields=sorted(keys),
                    archived_finite_targets=frames[decision].get('targets') or [],
                    accepted=not actual[decision].get('action_rejected'), outcome=rec['outcome']))
                print(level, decision, raw or 'original', rec['outcome'], flush=True)
    path = ROOT / 'MEMORY/docs/task46-roof-whatif-verified.json'
    path.write_text(json.dumps(dict(
        semantics='original teacher action prefix; original action and all variants continue with frozen student',
        results=results), ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
