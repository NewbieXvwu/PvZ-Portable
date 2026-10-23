#!/usr/bin/env python3
"""task31：夜间扰动族 n10q = n10p 的做法 + 实测“保留卡”不参与替换。

与 make_night_n10p.py 的唯一区别是 deck_perturb 多一个 keep_plants：
被换掉后老师 95% 上界 < 0.25 的底卡，其槽位不再作为替换目标
（判据与表在 DESIGN 新节；测量脚本 scripts/analyze_keep_cards.py）。

**不重查引擎**：n10q 能生成的卡组是 n10p 的子集（只是少了换掉保留卡的那些），
所以整组检查表（engine_warned_decks）、替换池（pool）、逐卡选项（engine_options）
直接沿用 n10p —— 对 n10q 是**保守超集**，只会多排除、不会漏排除。
留出卡组重新选（每个 level 两个），并且把 n10p 原有的留出卡组一起写进 excluded_decks：
n10q 仍可能生成 n10p 的留出卡组，不排除就会污染留出评估。

种子段：训练 3,300,000 + 10,000×(level−11)，每关 8192 个（与既有清单审计不重叠）；
评估种子沿用 30000–30255（与 n10p 同一批种子 → 评估可比对，评估种子跨族共用是设计）。
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_tasks import deck_for_seed, terrain_covered  # noqa: E402

LEVELS = (11, 12, 13, 14, 16, 17, 18, 19)
BASE_SEED = 3_300_000
LEVEL_STRIDE = 10_000
HOLDOUT_SEED_BASE = 3_400_000
HOLDOUT_EVAL_SEEDS = list(range(30000, 30256))
REQUIRED_POOL = (10, 12, 13, 14, 15)
SOURCE_FORMAT = 'night_l{level}_train_n10p'
TRAIN_FORMAT = 'night_l{level}_train_n10q'
HOLDOUT_FORMAT = 'night_l{level}_eval_n10q_holdout{index}'


def build(source, levels, keep, train_count, holdouts_each):
    """两遍：先选留出卡组，再把“全部留出卡组”写进每个 level 的 excluded_decks。

    一遍做不到：留出卡组是从同一族里抽的，而夜间各关替换池相同，第 16 关的留出卡组
    可能与第 11 关的某个种子生成的卡组是同一套牌（实测撞过一次）。第一遍生成的第 11 关
    清单里还没有它，校验就会抓到“训练卡组 = 留出卡组”。第二遍用完整排除集重新生成，
    deck_for_seed 会跳过被排除的卡组继续取，种子不变、卡组重算。
    """
    holdout_decks = {}
    signatures = set()
    for level in levels:
        original = next(t for t in source['tasks'] if t['task_id'] == SOURCE_FORMAT.format(level=level))
        spec = original['deck_perturb']
        picker = copy.deepcopy(original)
        picker['deck_perturb'] = dict(swaps=spec['swaps'], pool=spec['pool'],
                                      engine_options=spec['engine_options'],
                                      excluded_decks=list(spec.get('excluded_decks', [])),
                                      require_no_warnings=True,
                                      engine_warned_decks=spec['engine_warned_decks'],
                                      keep_plants=list(keep))
        for index in range(holdouts_each):
            attempt = 0
            while True:
                deck = deck_for_seed(picker, HOLDOUT_SEED_BASE + LEVEL_STRIDE * (level - LEVELS[0]) + attempt)
                attempt += 1
                signature = tuple(sorted(deck))
                if signature not in signatures:
                    break
                if attempt > 4096:
                    raise ValueError('not enough distinct held-out decks')
            signatures.add(signature)
            holdout_decks.setdefault(level, []).append(deck)
    tasks, heldout = [], []
    for level in levels:
        original = next(t for t in source['tasks'] if t['task_id'] == SOURCE_FORMAT.format(level=level))
        spec = original['deck_perturb']
        task = copy.deepcopy(original)
        start = BASE_SEED + LEVEL_STRIDE * (level - LEVELS[0])
        task.update(task_id=TRAIN_FORMAT.format(level=level), seeds=list(range(start, start + train_count)))
        # n10p 的留出卡组对 n10q 仍可达，必须一起排除；跨关留出卡组同理。
        previous = {tuple(sorted(d)) for d in spec.get('excluded_decks', [])}
        task['deck_perturb'] = dict(swaps=spec['swaps'], pool=spec['pool'],
                                    engine_options=spec['engine_options'],
                                    excluded_decks=[list(v) for v in sorted(previous | signatures)],
                                    require_no_warnings=True,
                                    engine_warned_decks=spec['engine_warned_decks'],
                                    keep_plants=list(keep))
        tasks.append(task)
        for index, deck in enumerate(holdout_decks[level]):
            held = copy.deepcopy(original)
            held.update(task_id=HOLDOUT_FORMAT.format(level=level, index=index + 1), deck=list(deck),
                        seeds=list(HOLDOUT_EVAL_SEEDS), split='heldout_deck',
                        holdout_semantics='unordered plant set excluded from every perturbation training family')
            held.pop('deck_perturb', None)
            heldout.append(held)
    return tasks, heldout


def audit(result, levels, keep, scan_paths):
    training = [t for t in result['tasks'] if 'deck_perturb' in t]
    excluded = {tuple(sorted(t['deck'])) for t in result['tasks'] if t.get('split') == 'heldout_deck'}
    seeds, decks = [], set()
    coverage = {p: 0 for p in REQUIRED_POOL}
    for task in training:
        spec = task['deck_perturb']
        if list(spec['keep_plants']) != list(keep):
            raise SystemExit(f'{task["task_id"]}: keep_plants mismatch')
        warned = {tuple(sorted(d)) for d in spec['engine_warned_decks']}
        for seed in task['seeds']:
            deck = deck_for_seed(task, seed)
            assert all(card in deck for card in keep), 'keep card was replaced'
            assert len(set(deck) - set(task['deck'])) in (1, 2)
            assert len(deck) == len(task['deck']) and len(set(deck)) == len(deck)
            assert not (set(deck) & set(range(40, 48))), 'upgrade plant in deck'
            assert terrain_covered(task['level'], deck)
            assert tuple(sorted(deck)) not in excluded, 'training deck is a held-out deck'
            assert tuple(sorted(deck)) not in warned, 'training deck would trigger a whole-deck warning'
            for plant in REQUIRED_POOL:
                if plant in deck:
                    coverage[plant] += 1
            decks.add(tuple(sorted(deck)))
            seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise SystemExit('duplicate training seeds inside the new family')
    seen = {}
    for path in scan_paths:
        data = json.loads(Path(path).read_text())
        entries = data['tasks'] if isinstance(data, dict) and 'tasks' in data else data
        for task in entries:
            if not isinstance(task, dict):
                continue
            for seed in task.get('seeds') or []:
                if isinstance(seed, int):
                    seen.setdefault(seed, task.get('task_id'))
    clashes = sorted({(s, seen[s]) for s in seeds if s in seen})
    if clashes:
        raise SystemExit(f'training seed overlap with existing manifests: {clashes[:5]} ({len(clashes)})')
    if not all(coverage[p] for p in REQUIRED_POOL):
        raise SystemExit(f'required plants missing from training decks: {coverage}')
    return dict(training_seeds=len(seeds), distinct_training_card_sets=len(decks),
                heldout_card_sets=len(excluded), keep_plants=list(keep),
                required_pool_deck_counts={str(p): coverage[p] for p in REQUIRED_POOL},
                levels=list(levels),
                scanned_existing_seeds=len(seen), seed_overlap=0,
                eval_seeds='30000-30255 (shared with n10p for paired evaluation)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path,
                        default=Path.home() / 'PvZ-search/artifacts/stage4/night_panel_n10p.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/stage4/night_panel_n10q.json')
    parser.add_argument('--keep', default='8,9', help='保留卡（植物 id，逗号分隔）')
    parser.add_argument('--train-count', type=int, default=8192)
    parser.add_argument('--holdouts-each', type=int, default=2)
    parser.add_argument('--scan', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    keep = [int(v) for v in args.keep.split(',') if v.strip()]
    source = json.loads(args.source.read_text())
    tasks, heldout = build(source, LEVELS, keep, args.train_count, args.holdouts_each)
    result = dict(schema_version=1,
                  purpose='task31 night n10q: n10p perturbation with measured keep cards never replaced',
                  keep_rule='base card whose teacher win rate after replacement has a 95% upper bound below 0.25',
                  warning_scope='inherited from night_panel_n10p.json (engine tables, conservative superset)',
                  tasks=tasks + heldout)
    result['generation_audit'] = audit(result, LEVELS, keep, args.scan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result['generation_audit'], indent=1))


if __name__ == '__main__':
    main()
