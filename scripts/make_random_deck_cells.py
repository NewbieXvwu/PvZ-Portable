#!/usr/bin/env python3
"""task137：v5 随机卡组族（训练 / 评估两批，互不重叠）。

* 训练批：每区 2 关 × 2 套 = 16 个格子，每格一条 0.5 → 3.0 的倍率阶梯。
* 评估批：每区 2 关 × 1 套 = 8 套，只进挑选面板与门禁监控，永不进训练。

卡组按 MEMORY/docs/deck-rules-v4.md 的 v5 规则从固定 draw_seed 抽出，
登记在 configs/random_deck_cells_v5.json（进 git）。面板是派生文件：

  python3 scripts/make_random_deck_cells.py --check          # 校验登记表可复现
  python3 scripts/make_random_deck_cells.py --out <目录>      # 写 training_panel.json / eval_panel.json

种子段（与仓库现有清单不重叠）：

  倍率档  210_000_000 + 格子序号 * 100_000 + 档位 * 4_096   （每档 2048 个种子）
  1.0 档  213_000_000 + 格子序号 * 4_096
  评估批不写种子：挑选面板每轮从挑选账本里现分。

倍率档 0.5 / 0.75 的语义（src/Lawn/Board.cpp Board::SpawnZombieWave）：

  aSpawnCount = lround(原版该波僵尸数 * multiplier)

所以 <1 只在 ≥2 只的波生效（单只波 lround(0.5)=1 仍是 1 只），
坟头 / 泳池 / 屋顶的收尾额外刷怪不读倍率。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts')]

from deck_rules import RULE_VERSION, deck_violations, generate_deck, region_for_level  # noqa: E402
from pvz_tasks import profile_for_deck, terrain_covered  # noqa: E402

REGISTRY = ROOT / 'configs/random_deck_cells_v5.json'
LEVELS = (7, 8, 12, 14, 24, 26, 33, 37)
LADDER = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
DRAW_SEED = 137
SEEDS_PER_TASK = 2048
MULT_SEED_BASE = 210_000_000
MULT_CELL_STRIDE = 100_000
MULT_BAND_STRIDE = 4096
BASE_SEED = 213_000_000
TRAINING_SEED_BLOCK = [MULT_SEED_BASE, 214_000_000]
DECKS_PER_LEVEL = (('train', 2), ('eval', 1))
REGISTERED_CONFIGS = ('student_selection_v2.json', 'gate_v5_panel.json', 'human_demo_tasks.json',
                      'student_selection_holdout.json')
POOL_SEED_BASE = 1_590_000_000


def heldout_signatures(registry, panels):
    """只比卡组多重集，不比较种子，留出卡组在所有关卡都不能进训练。"""
    gate_v6 = ROOT / 'configs/gate_v6_panel.json'
    if not panels and not gate_v6.exists():
        raise ValueError('external deck pools require --heldout-panel with the task148 gate v6 draft')
    paths = [ROOT / 'configs' / name for name in REGISTERED_CONFIGS if name != 'human_demo_tasks.json']
    paths += list(panels)
    if gate_v6.exists():
        paths.append(gate_v6)
    tasks = [task for path in paths for task in json.loads(path.read_text())['tasks']]
    tasks += [task for task in registry['tasks'] if task['split'] == 'eval']
    return {tuple(sorted(task['deck'])) for task in tasks if task.get('deck')}


def load_deck_pool(registry, *, gpt_ledger=None, archive_root=ROOT, roof_candidates=None, heldout_panels=()):
    """读取已验证胜局的 meta 与 task148 候选，不读取帧或重新重放。"""
    if gpt_ledger is None and roof_candidates is None:
        return []
    heldout = heldout_signatures(registry, heldout_panels)
    seen = heldout | {tuple(sorted(task['deck'])) for task in registry['tasks']}
    candidates = []
    if gpt_ledger is not None:
        for row in json.loads(gpt_ledger.read_text())['archives']:
            if (row['verify'] != '通过' or row['result'] != 1 or row['grade'] == 'loss'
                    or not row.get('archive')):
                continue
            archive = Path(row['archive'])
            meta = json.loads((archive_root / archive / 'meta.json').read_text())
            if meta['outcome']['result'] == 1:
                candidates.append((meta['task'], 'task105'))
    if roof_candidates is not None:
        candidates += [(entry['task'], 'task148')
                       for entry in json.loads(roof_candidates.read_text())['candidates']]
    tasks = []
    for task, source in sorted(candidates, key=lambda item: (item[0]['level'], sorted(item[0]['deck']), item[1])):
        level, deck = task['level'], tuple(sorted(task['deck']))
        try:
            problems = deck_violations(level, deck, fixed_bank=True)
        except ValueError:  # 原生传送带与特殊关没有可供本课程复用的固定卡组。
            continue
        if problems or not terrain_covered(level, deck) or deck in seen:
            continue
        seen.add(deck)
        identity = hashlib.sha256(json.dumps([level, deck]).encode()).hexdigest()[:12]
        profile = profile_for_deck(deck)
        tasks.append(dict(task_id=f'rd159_l{level}_train_{identity}', level=level,
                          terrain=region_for_level(level, fixed_bank=True), deck=list(deck), source=source,
                          profile=dict(seed_slot_count=profile.seed_slot_count,
                                       owned_upgrade_plants=list(profile.owned_upgrade_plants))))
    if not tasks:
        raise ValueError('no eligible external decks after terrain/rule/heldout filtering')
    if {tuple(sorted(task['deck'])) for task in tasks} & heldout:
        raise ValueError('external training decks overlap selection/gate holdouts')
    return tasks


def registered_signatures() -> set[tuple[int, ...]]:
    """仓库里已登记的卡组集合：新抽的卡组不能和它们重复。"""
    seen = set()
    for name in REGISTERED_CONFIGS:
        document = json.loads((ROOT / 'configs' / name).read_text())
        for task in document.get('tasks') or []:
            deck = task.get('deck')
            if isinstance(deck, list) and deck:
                seen.add(tuple(sorted(deck)))
    return seen


def load_frozen_pool(path, registry, heldout_panels):
    """Small reviewed task165 snapshot; never reread/replay the source archives at deployment."""
    tasks = json.loads(path.read_text())['tasks']
    heldout = heldout_signatures(registry, heldout_panels)
    seen = {tuple(sorted(task['deck'])) for task in registry['tasks']}
    for task in tasks:
        deck = tuple(sorted(task['deck']))
        if deck in heldout or deck in seen:
            raise ValueError('frozen training deck duplicates a training/evaluation/heldout deck')
        if deck_violations(task['level'], deck, fixed_bank=True) or not terrain_covered(task['level'], deck):
            raise ValueError('frozen training deck violates terrain/v5 rules')
        seen.add(deck)
    base_train = {tuple(sorted(task['deck'])) for task in registry['tasks'] if task['split']=='train'}
    if base_train & heldout:
        raise ValueError('base random training decks overlap heldouts')
    return tasks


def generate_registry() -> dict:
    rng, seen, tasks = random.Random(DRAW_SEED), registered_signatures(), []
    for level in LEVELS:
        for split, count in DECKS_PER_LEVEL:
            for index in range(count):
                while True:
                    length, draw_seed = rng.randint(7, 10), rng.getrandbits(32)
                    deck = generate_deck(level, length, draw_seed)
                    signature = tuple(sorted(deck))
                    if signature in seen:
                        continue
                    break
                seen.add(signature)
                tasks.append(dict(task_id=f'rd137_l{level}_{split}_d{index + 1:02d}',
                                  level=level, terrain=region_for_level(level), split=split,
                                  deck=list(deck), draw_seed=draw_seed))
    return dict(schema_version=1, rule_version=RULE_VERSION, draw_seed=DRAW_SEED,
                training_seed_block=TRAINING_SEED_BLOCK, levels=list(LEVELS),
                ladder=list(LADDER), tasks=tasks)


def seed_start(cell_index: int, multiplier: float) -> int:
    if multiplier == 1.0:
        return BASE_SEED + cell_index * MULT_BAND_STRIDE
    band = [value for value in LADDER if value != 1.0].index(multiplier)
    return MULT_SEED_BASE + cell_index * MULT_CELL_STRIDE + band * MULT_BAND_STRIDE


def task_document(task: dict, multiplier: float, seeds) -> dict:
    entry = dict(task_id=task['task_id'], level=task['level'], terrain=task['terrain'],
                 deck=list(task['deck']), playthrough=2, zombie_count_multiplier=multiplier,
                 seeds=list(seeds), random_deck_cell=True, rule_version=RULE_VERSION,
                 curriculum_ladder=list(LADDER))
    if task['terrain'] == 'fog':
        entry['fog_observation'] = 'human_v2'
    if 'profile' in task:
        entry['profile'] = task['profile']
    return entry


def build_panels(registry: dict, pool_tasks=()) -> dict:
    training = [task for task in registry['tasks'] if task['split'] == 'train']
    tasks = []
    for cell_index, task in enumerate(training):
        for multiplier in LADDER:
            suffix = '' if multiplier == 1.0 else '_m%03d' % round(multiplier * 100)
            start = seed_start(cell_index, multiplier)
            entry = task_document(task, multiplier, range(start, start + SEEDS_PER_TASK))
            entry['task_id'] = task['task_id'] + suffix
            tasks.append(entry)
    for cell_index, task in enumerate(pool_tasks):
        for band, multiplier in enumerate(LADDER):
            suffix = '' if multiplier == 1.0 else '_m%03d' % round(multiplier * 100)
            start = POOL_SEED_BASE + cell_index * MULT_CELL_STRIDE + band * MULT_BAND_STRIDE
            entry = task_document(task, multiplier, range(start, start + SEEDS_PER_TASK))
            entry.update(task_id=task['task_id'] + suffix, deck_source=task['source'])
            tasks.append(entry)
    evaluation = [task_document(task, 1.0, []) for task in registry['tasks'] if task['split'] == 'eval']
    panels = dict(training_panel=dict(schema_version=1, purpose='task137 v5 random-deck curriculum cells',
                                    training_seed_block=TRAINING_SEED_BLOCK, tasks=tasks),
                eval_panel=dict(schema_version=1, purpose='task137 v5 unseen decks for selection and monitoring',
                                tasks=evaluation))
    if pool_tasks:
        panels['training_panel']['external_training_seed_block'] = [
            POOL_SEED_BASE, POOL_SEED_BASE + len(pool_tasks) * MULT_CELL_STRIDE]
    return panels


def check(registry: dict) -> None:
    """登记表自检：可复现、卡组合法、训练 / 评估互不重叠。"""
    fresh = generate_registry()
    assert json.dumps(fresh, sort_keys=True) == json.dumps(registry, sort_keys=True), \
        'registry is not reproducible from draw_seed 137'
    train = [tuple(sorted(task['deck'])) for task in registry['tasks'] if task['split'] == 'train']
    evaluation = [tuple(sorted(task['deck'])) for task in registry['tasks'] if task['split'] == 'eval']
    assert len(set(train)) == len(train) and len(set(evaluation)) == len(evaluation)
    assert not set(train) & set(evaluation), 'eval decks overlap training decks'
    for task in registry['tasks']:
        assert generate_deck(task['level'], len(task['deck']), task['draw_seed']) == tuple(task['deck'])
        problems = deck_violations(task['level'], task['deck'])
        assert not problems, (task['task_id'], problems)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/task137')
    parser.add_argument('--check', action='store_true', help='only verify the committed registry')
    parser.add_argument('--freeze-registry', action='store_true', help='print a fresh registry')
    parser.add_argument('--gpt-ledger', type=Path, help='task105 import/merged_progress.json (or import_progress.json)')
    parser.add_argument('--archive-root', type=Path, default=ROOT, help='root for ledger archive paths')
    parser.add_argument('--roof-candidates', type=Path, help='task148-roof-candidates.json')
    parser.add_argument('--pool-registry', type=Path, help='reviewed frozen external pool (task165); exclusive with live source arguments')
    parser.add_argument('--heldout-panel', type=Path, action='append', default=[],
                        help='additional selection/gate panel; supply gate_v6_panel.json before using external pools')
    args = parser.parse_args()
    if args.freeze_registry:
        print(json.dumps(generate_registry(), indent=1) + '\n')
        return
    registry = json.loads(REGISTRY.read_text())
    check(registry)
    if args.check:
        print(f'registry ok: {len(registry["tasks"])} tasks, rule v{registry["rule_version"]}')
        return
    if args.pool_registry:
        if args.gpt_ledger or args.roof_candidates:
            parser.error('--pool-registry cannot be combined with live deck sources')
        pool = load_frozen_pool(args.pool_registry, registry, args.heldout_panel)
    else:
        pool = load_deck_pool(registry, gpt_ledger=args.gpt_ledger, archive_root=args.archive_root,
                              roof_candidates=args.roof_candidates, heldout_panels=args.heldout_panel)
    panels = build_panels(registry, pool)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, document in panels.items():
        (args.out / f'{name}.json').write_text(json.dumps(document, indent=1) + '\n')
    print(f'wrote {args.out}/training_panel.json and eval_panel.json: '
          f'{len(panels["training_panel"]["tasks"])} training / '
          f'{len(panels["eval_panel"]["tasks"])} eval tasks')


if __name__ == '__main__':
    main()
