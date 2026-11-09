#!/usr/bin/env python3
"""The production collector launch command, shared by deployment and watchdog."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def command(args):
    root, worktree = args.repo, args.worktree or args.repo / '.claude/worktrees/cq'
    python = args.python or Path.home() / '.venvs/ml/bin/python3'
    imports = root / '.claude/worktrees/llmimport'
    focus = json.loads((Path(__file__).resolve().parents[1] / 'configs/curriculum_focus.json').read_text())
    focus_share = getattr(args, 'focus_share', None)
    l29_start_share = getattr(args, 'l29_start_share', None)
    focus_share = focus['focus_share'] if focus_share is None else focus_share
    l29_start_share = focus['l29_start_share'] if l29_start_share is None else l29_start_share
    routes = getattr(args, 'value_routes', None)
    if routes is None:
        # task252: the checkpoint already contains the T=3 output-head weights.
        # task255: balanced pool branch labels; fitted T=1, absolute 0.02 override (margin 0).
        routes = {'BACKGROUND_5_ROOF': dict(
                      value=str(root / 'artifacts/value_net/task252_finetune_t3/model.pt'), margin_relative=.02),
                  'BACKGROUND_3_POOL': dict(
                      value=str(root / 'artifacts/value_net/task255_balanced_pool/model.pt'), margin_relative=0)}
    return [str(python), str(worktree / 'scripts/curriculum_queue.py'),
        *(['--review-share', str(args.review_share)] if args.review_share is not None else []),
        '--state', str(args.logs / 'curriculum_state.json'), '--log', str(args.logs / 'curriculum.log'),
        '--seed-ledger', str(args.logs / 'gate_seed_ledger.json'),
        '--start-seed-ledger', str(args.logs / 'start_cell_seed_ledger.json'),
        '--rounds', str(args.rounds), '--workers', str(args.workers), '--min-eval-units', '4',
        '--python', str(python), '--round-games', '768', '--games-per-group', '48',
        '--manifest', str(worktree / 'artifacts/stage4/multiplier_panel.json'),
        # 11-05: A is a second seed of the distilled incumbent fine-tune. The gptbc restart lost
        # nine rounds in a row (r111-r119, -1.4 to -12 pp wins) while B passed four times.
        '--recipe', str(args.recipe or worktree / 'recipes/dw4i2_v2timer_clean_b.json'),
        '--alternate-recipe', str(args.alternate_recipe or worktree / 'recipes/dw4i2_v2timer_clean_b.json'),
        *(['--b-init', str(args.b_init)] if args.b_init is not None else []),
        *(['--b-initial-epochs', str(args.b_initial_epochs)] if args.b_initial_epochs is not None else []),
        *(['--b-epochs', str(args.b_epochs)] if args.b_epochs is not None else []),
        '--a-restart' if args.a_restart else '--no-a-restart',
        '--b-from-incumbent' if getattr(args, 'b_from_incumbent', True) else '--no-b-from-incumbent',
        '--training-parallelism', '1', '--opening-manifest', str(worktree / 'artifacts/task119/opening_panel.json'),
        '--executable', str(args.executable or root / 'artifacts/production/imitater1/pvz-portable'),
        '--promotion-panel', str(worktree / 'configs/gate_v8_panel.json'),
        '--extra-manifest', str(worktree / 'artifacts/task165/training_panel.json'),
        '--demo-start-share', '0.25', '--random-deck-share', '0.25',
        *(['--focus-share', str(focus_share), '--l29-start-share', str(l29_start_share),
           '--focus-manifest', str(worktree / 'configs/curriculum_focus.json')]
          if focus_share or l29_start_share else []),
        '--gpt-import-dir', str(imports / 'artifacts/task105/import'), '--gpt-archive-root', str(imports),
        '--gpt-cold-rates', str(args.logs / 'teacher_rates.json'), '--obs-v3', '--native-games',
        '--resource-dir', str(Path.home() / '.cache/pvz-research-resources'),
        '--frames', str(root / 'artifacts/student/frames_v2'), '--out', str(root / 'artifacts/hard'),
        '--rollout-policy', str(root / 'artifacts/student/dw4i_s1/model.pt'),
        '--rollout-policy-gpt', str(root / 'artifacts/student/gptbc_s1/model.pt'),
        '--value', str(root / 'artifacts/value_net/v5/model.pt'),
        '--value-roof', str(root / 'artifacts/value_net/v5roof/model.pt'),
        '--value-routes', json.dumps(routes),
        '--static-value', str(root / 'artifacts/hard/value_branch1/model.pt')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.home() / 'PvZ-search')
    parser.add_argument('--worktree', type=Path)
    parser.add_argument('--logs', type=Path, default=Path.home() / 'pvz-logs')
    parser.add_argument('--python', type=Path)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--alternate-recipe', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--review-share', type=float)
    parser.add_argument('--focus-share', type=float, help='覆盖curriculum_focus.json中的份额，0关闭')
    parser.add_argument('--l29-start-share', type=float, help='覆盖curriculum_focus.json中的份额，0关闭')
    # Standard library only: deploy26 runs this file from a partial install (changed files only),
    # so importing search_play failed there on 11-07. curriculum_queue validates the routes.
    parser.add_argument('--value-routes', type=json.loads,
                        help='override the task252 roof route; {} restores production leaf values')
    parser.add_argument('--b-init', type=Path)
    parser.add_argument('--b-initial-epochs', type=int)
    parser.add_argument('--b-epochs', type=int)
    parser.add_argument('--a-restart', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--b-from-incumbent', action=argparse.BooleanOptionalAction, default=True,
                        help='B fine-tunes the incumbent at queue time (production default)')
    args = parser.parse_args()
    shell = 'env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ' + shlex.join(command(args))
    shell += ' 2>&1 | tee -a ' + shlex.quote(str(args.logs / 'qc.out'))
    subprocess.run(['tmux', 'new-session', '-d', '-s', 'qc', '-x', '200', '-y', '50', shell], check=True)


if __name__ == '__main__':
    main()
