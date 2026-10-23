"""夜间扰动卡组族 n10p（task28）：以 n10 为底，每种子确定性替换 1–2 张。

替换池 = 引擎 `SeedChooserScreen::CrazyDavePickSeeds` 的候选（`PvZEnv.deck_pool`，按真实 Board 的
僵尸名单/地形/雾判断，叶子保护伞与三叶草只在引擎认为需要时进池）；
整组检查 = `SeedChooserScreen::OnStartButton` 的 `CheckDeckWarnings`（`PvZEnv.deck_warnings`），
会触发任何警告的卡组在生成时穷举出来写进 `engine_warned_decks`，运行时排除。
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

LEVELS = (11, 12, 13, 14, 16, 17, 18, 19)
BASE_SEEDS = {11: 2400000, 12: 2410000, 13: 2420000, 14: 2430000,
              16: 2450000, 17: 2460000, 18: 2470000, 19: 2480000}
TRAIN_COUNT = 8192
HOLDOUT_SEEDS = list(range(30000, 30256))
REQUIRED_POOL = (10, 12, 13, 14, 15)   # task26：2-10 传送带发得到、但从未进过任何任务卡组


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/stage4/night_panel_n10p.json')
    parser.add_argument('--resource-dir', type=Path, default=Path.home() / '.cache/pvz-research-resources')
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--train-count', type=int, default=TRAIN_COUNT)
    parser.add_argument('--scan', type=Path, nargs='*', default=[], help='extra manifests for the seed-overlap audit')
    args = parser.parse_args()
    if not 1 <= args.train_count <= 8192:
        parser.error('--train-count must be in 1..8192')
    run_family(
        source_path=args.manifest, out_path=args.out, levels=LEVELS, base_seeds=BASE_SEEDS,
        train_count=args.train_count, source_format='night_l{level}_train_n10',
        train_format='night_l{level}_train_n10p',
        holdout_format='night_l{level}_eval_n10p_holdout{index}',
        holdouts_each=2, holdout_seed_start=2600000, holdout_eval_seeds=HOLDOUT_SEEDS,
        required_plants=REQUIRED_POOL, scan_paths=args.scan, resource_dir=args.resource_dir,
        executable=args.executable,
        purpose='task28 night n10p coverage collection; new task families only',
        task_spec=TaskSpec, profile_for_deck=profile_for_deck, deck_for_seed=deck_for_seed,
        terrain_covered=terrain_covered, env_factory=EventWaitEnv)


if __name__ == '__main__':
    main()
