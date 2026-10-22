"""夜间扰动卡组族 n10p（task28）：以 n10 为底，每种子确定性替换 1–2 张，补没见过的植物。

与 task19 的白天 d7p 同一套机制（`deck_perturb` + `pvz_tasks.deck_for_seed`），差别只有：
- 底卡组是夜间 n10 `[0,1,2,3,4,5,7,8,9,11]`（10 张 = SEEDBANK_MAX，替换不改变长度）；
- 关卡是夜间草地 11–14、16–19（15 关不存在于夜间面板）；
- 池子从引擎 `DECK_OPTIONS` 现查（playthrough 2 已解锁、非升级、无引擎警告），
  生成时**断言** task26 缺的 5 种卡（大喷菇 10 / 魅惑菇 12 / 胆小菇 13 / 寒冰菇 14 / 毁灭菇 15）都在池子里。

留出：每关 2 套卡组（`night_lN_eval_n10p_holdout{1,2}`，评估段 30000–30255），
所有训练族的 `excluded_decks` 都显式排除这些集合（按无序集合比较，与槽位顺序无关）。
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_tasks import deck_for_seed, profile_for_deck, terrain_covered

LEVELS = (11, 12, 13, 14, 16, 17, 18, 19)
BASE_SEEDS = {11: 2400000, 12: 2410000, 13: 2420000, 14: 2430000,
              16: 2450000, 17: 2460000, 18: 2470000, 19: 2480000}
TRAIN_COUNT = 8192
HOLDOUT_SEEDS = list(range(30000, 30256))
REQUIRED_POOL = (10, 12, 13, 14, 15)   # task26：2-10 传送带发得到、但从未进过任何任务卡组


def make(source, catalog, train_count=TRAIN_COUNT, holdouts_each=2):
    tasks = []
    heldout = []
    signatures = set()
    for level in LEVELS:
        original = next(t for t in source['tasks'] if t['task_id'] == f'night_l{level}_train_n10')
        options = catalog[level]
        warnings = dict(options['not_recommended'])
        pool = sorted(p for p in options['available_plants'] if 0 <= p < 40 and not warnings.get(p, 1))
        missing = [p for p in REQUIRED_POOL if p not in pool]
        if missing:
            raise ValueError(f'level {level}: perturbation pool misses required plants {missing}')
        task = copy.deepcopy(original)
        start = BASE_SEEDS[level]
        task.update(task_id=f'night_l{level}_train_n10p', seeds=list(range(start, start + train_count)))
        task['deck_perturb'] = dict(swaps=[1, 2], pool=pool, engine_options=options, excluded_decks=[])
        tasks.append(task)
        attempt = 0
        for i in range(holdouts_each):
            while True:
                deck = deck_for_seed(task, 2600000 + 10000 * level + attempt)
                attempt += 1
                signature = tuple(sorted(deck))
                if signature not in signatures:
                    break
                if attempt > 4096:
                    raise ValueError('not enough distinct held-out decks')
            signatures.add(signature)
            held = copy.deepcopy(original)
            held.update(task_id=f'night_l{level}_eval_n10p_holdout{i + 1}', deck=list(deck),
                        seeds=list(HOLDOUT_SEEDS), split='heldout_deck',
                        holdout_semantics='unordered plant set excluded from every perturbation training family')
            heldout.append(held)
    for task in tasks:
        task['deck_perturb']['excluded_decks'] = [list(v) for v in sorted(signatures)]
    return dict(schema_version=1, purpose='task28 night n10p coverage collection; new task families only',
                warning_scope='raw engine per-card recommendation flags; interactive whole-deck OnStartButton checks not called',
                tasks=tasks + heldout)


def existing_seeds(manifests):
    """穷举所有现有清单里的种子（不是抽样），用于自查新种子段不与任何任务重叠。"""
    seen = {}
    for path in manifests:
        data = json.loads(Path(path).read_text())
        entries = data['tasks'] if isinstance(data, dict) and 'tasks' in data else data
        if isinstance(entries, dict):
            entries = list(entries.values())
        for task in entries:
            if not isinstance(task, dict):
                continue
            for seed in task.get('seeds') or []:
                seen.setdefault(int(seed), task.get('task_id') or task.get('id'))
    return seen


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
    source = json.loads(args.manifest.read_text())
    catalog = {}
    with EventWaitEnv(args.resource_dir, executable=args.executable) as env:
        for level in LEVELS:
            task = next(t for t in source['tasks'] if t['task_id'] == f'night_l{level}_train_n10')
            deck = tuple(task['deck'])
            env.reset(deck=deck, task=TaskSpec(level=level, seed=BASE_SEEDS[level],
                                              profile=profile_for_deck(deck), playthrough=task.get('playthrough', 2)))
            catalog[level] = env.deck_options()
    result = make(source, catalog, args.train_count)
    excluded = {tuple(sorted(t['deck'])) for t in result['tasks'] if t.get('split') == 'heldout_deck'}
    decks = set()
    coverage = {p: 0 for p in REQUIRED_POOL}
    for task in result['tasks']:
        if 'deck_perturb' not in task:
            continue
        for seed in task['seeds']:
            deck = deck_for_seed(task, seed)
            assert len(set(deck) - set(task['deck'])) in (1, 2)
            assert len(deck) == len(task['deck']) == 10 and len(set(deck)) == 10
            assert not (set(deck) & set(range(40, 48))), 'upgrade plant in deck'
            assert terrain_covered(task['level'], deck) and tuple(sorted(deck)) not in excluded
            for p in REQUIRED_POOL:
                if p in deck:
                    coverage[p] += 1
            decks.add(tuple(sorted(deck)))
    ours = [s for t in result['tasks'] if 'deck_perturb' in t for s in t['seeds']]
    seen = existing_seeds(args.scan)
    clashes = sorted({(s, seen[s]) for s in ours if s in seen})
    if clashes:
        raise SystemExit(f'seed overlap with existing manifests: {clashes[:5]} ({len(clashes)} seeds)')
    if not all(coverage[p] for p in REQUIRED_POOL):
        raise SystemExit(f'required plants missing from training decks: {coverage}')
    result['generation_audit'] = dict(
        training_seeds=len(ours), distinct_training_card_sets=len(decks),
        heldout_card_sets=len(excluded), all_heldout_sets_excluded=True,
        required_pool=list(REQUIRED_POOL), required_pool_deck_counts=coverage,
        scanned_manifests=[str(p) for p in args.scan], scanned_existing_seeds=len(seen),
        seed_overlap=0,
        source_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result['generation_audit'], indent=1), flush=True)


if __name__ == '__main__':
    main()
