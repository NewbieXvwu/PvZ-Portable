#!/usr/bin/env python3
"""The production collector launch command, shared by deployment and watchdog."""
import argparse
from pathlib import Path
import shlex
import subprocess


def command(args):
    root, worktree = args.repo, args.worktree or args.repo / '.claude/worktrees/cq'
    python = args.python or Path.home() / '.venvs/ml/bin/python3'
    imports = root / '.claude/worktrees/llmimport'
    return [str(python), str(worktree / 'scripts/curriculum_queue.py'),
        *(['--review-share', str(args.review_share)] if args.review_share is not None else []),
        '--state', str(args.logs / 'curriculum_state.json'), '--log', str(args.logs / 'curriculum.log'),
        '--seed-ledger', str(args.logs / 'gate_seed_ledger.json'),
        '--start-seed-ledger', str(args.logs / 'start_cell_seed_ledger.json'),
        '--rounds', str(args.rounds), '--workers', str(args.workers), '--min-eval-units', '4',
        '--python', str(python), '--round-games', '768', '--games-per-group', '48',
        '--manifest', str(worktree / 'artifacts/stage4/multiplier_panel.json'),
        '--recipe', str(args.recipe or worktree / 'recipes/dw4i2_v2timer_rebal_seed2cold.json'),
        '--alternate-recipe', str(args.alternate_recipe or worktree / 'recipes/dw4i2_v2timer_clean_b.json'),
        *(['--b-init', str(args.b_init)] if args.b_init is not None else []),
        *(['--b-initial-epochs', str(args.b_initial_epochs)] if args.b_initial_epochs is not None else []),
        *(['--b-epochs', str(args.b_epochs)] if args.b_epochs is not None else []),
        '--training-parallelism', '1', '--opening-manifest', str(worktree / 'artifacts/task119/opening_panel.json'),
        '--executable', str(args.executable or root / 'artifacts/production/deploy20/pvz-portable'),
        '--promotion-panel', str(worktree / 'configs/gate_v5_panel.json'),
        '--extra-manifest', str(worktree / 'artifacts/task165/training_panel.json'),
        '--demo-start-share', '0.25', '--random-deck-share', '0.25',
        '--gpt-import-dir', str(imports / 'artifacts/task105/import'), '--gpt-archive-root', str(imports),
        '--gpt-cold-rates', str(args.logs / 'teacher_rates.json'), '--obs-v3', '--native-games',
        '--resource-dir', str(Path.home() / '.cache/pvz-research-resources'),
        '--frames', str(root / 'artifacts/student/frames_v2'), '--out', str(root / 'artifacts/hard'),
        '--rollout-policy', str(root / 'artifacts/student/dw4i_s1/model.pt'),
        '--rollout-policy-gpt', str(root / 'artifacts/student/gptbc_s1/model.pt'),
        '--value', str(root / 'artifacts/value_net/v5/model.pt'),
        '--value-roof', str(root / 'artifacts/value_net/v5roof/model.pt'),
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
    parser.add_argument('--b-init', type=Path)
    parser.add_argument('--b-initial-epochs', type=int)
    parser.add_argument('--b-epochs', type=int)
    args = parser.parse_args()
    shell = 'env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ' + shlex.join(command(args))
    shell += ' 2>&1 | tee -a ' + shlex.quote(str(args.logs / 'qc.out'))
    subprocess.run(['tmux', 'new-session', '-d', '-s', 'qc', '-x', '200', '-y', '50', shell], check=True)


if __name__ == '__main__':
    main()
