"""白天扰动卡组族 d7q（task28 第 6 步第 3 条）：与 d7p 同做法，但替换池与整组检查都走引擎。

旧族 `day_lN_train_d7p` 已经采过大量数据，不动它；d7q 是新族名（AGENTS §8 第 4 条）。
替换池 = `CrazyDavePickSeeds` 候选；整组检查 = `OnStartButton` 的 `CheckDeckWarnings`，
会触发警告的卡组（例如没有向日葵）在生成时穷举排除。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_tasks import deck_for_seed, profile_for_deck, terrain_covered
from perturb_family import run_family

LEVELS = (7, 8, 9, 10)
BASE_SEEDS = {7: 2500000, 8: 2510000, 9: 2520000, 10: 2530000}
TRAIN_COUNT = 8192
HOLDOUT_SEEDS = list(range(30000, 30256))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/stage4/day_panel_d7q.json')
    parser.add_argument('--resource-dir', type=Path, default=Path.home() / '.cache/pvz-research-resources')
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--train-count', type=int, default=TRAIN_COUNT)
    parser.add_argument('--scan', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    if not 1 <= args.train_count <= 8192:
        parser.error('--train-count must be in 1..8192')
    run_family(
        source_path=args.manifest, out_path=args.out, levels=LEVELS, base_seeds=BASE_SEEDS,
        train_count=args.train_count, source_format='day_l{level}_train_d7',
        train_format='day_l{level}_train_d7q',
        holdout_format='day_l{level}_eval_d7q_holdout{index}',
        holdouts_each=3, holdout_seed_start=2700000, holdout_eval_seeds=HOLDOUT_SEEDS,
        required_plants=(), scan_paths=args.scan, resource_dir=args.resource_dir,
        executable=args.executable,
        purpose='task28 day d7q: perturbation pool and whole-deck checks from the engine',
        task_spec=TaskSpec, profile_for_deck=profile_for_deck, deck_for_seed=deck_for_seed,
        terrain_covered=terrain_covered, env_factory=EventWaitEnv)


if __name__ == '__main__':
    main()
