"""task46 engine gate: replay learned-student actions and compare every public/hidden frame."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from engine_replay_compare import student_action
from pvz_event_env import EventWaitEnv
from pvz_env import TaskSpec
from pvz_tasks import deck_for_level, profile_for_deck
from pvz_rollout_policy import load_student
from pvz_student_observation import StudentHistory
from pvz_value_net import observation_context


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--student', type=Path,
                        default=Path('/home/newbiexvwu/PvZ-search/artifacts/student/dw4i_s1/model.pt'))
    parser.add_argument('--resource-dir', type=Path, default=Path('/home/newbiexvwu/.cache/pvz-research-resources'))
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/task46/engine_gate.json')
    parser.add_argument('--levels', type=int, nargs='+', default=list(range(1, 50)))
    parser.add_argument('--max-decisions', type=int, default=2000)
    args = parser.parse_args()
    if any(not 1 <= level <= 49 for level in args.levels):
        parser.error('gate covers L1..L49 only')
    os.environ['PVZ_NO_POPUP'] = '1'
    import torch
    torch.set_num_threads(1)
    net = load_student(args.student)
    report = dict(complete=False, frame_differences=0, levels_requested=args.levels,
                  baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  student_sha256=hashlib.sha256(args.student.read_bytes()).hexdigest(),
                  max_decisions=args.max_decisions, cases=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with EventWaitEnv(args.resource_dir, executable=args.baseline) as baseline, \
            EventWaitEnv(args.resource_dir, executable=args.candidate) as candidate:
        for level in args.levels:
            deck = deck_for_level(level, family='roof_r7' if level >= 41 else None)
            task = TaskSpec(level=level, seed=45000000, profile=profile_for_deck(deck))
            before, _ = baseline.reset(deck=deck, task=task)
            after, _ = candidate.reset(deck=deck, task=task)
            context = observation_context(before) if net.context else None
            history = StudentHistory(net.obs) if net.obs != 'v1' else None
            state, frames, digest = {}, 0, hashlib.sha256()
            while True:
                public = encoded(before)
                hidden = encoded(baseline.privileged_state())
                if public != encoded(after) or hidden != encoded(candidate.privileged_state()):
                    report['frame_differences'] += 1
                    args.out.write_text(json.dumps(report, indent=1) + '\n')
                    raise RuntimeError(f'L{level}: state differs at tick {before["tick"]}')
                digest.update(public)
                digest.update(hidden)
                frames += 1
                if before['terminal'] or frames > args.max_decisions:
                    break
                action = student_action(net, before, state, history, context)
                before, *_ = baseline.step(action)
                after, *_ = candidate.step(action)
            report['cases'].append(dict(level=level, seed=45000000, deck=list(deck), frames=frames,
                final_tick=before['tick'], terminal=before['terminal'],
                public_and_hidden_sha256=digest.hexdigest(), frame_differences=0))
            args.out.write_text(json.dumps(report, indent=1) + '\n')
            print(f'L{level}: {frames} frames, terminal={before["terminal"]}, differences=0', flush=True)
    report['complete'] = True
    args.out.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
