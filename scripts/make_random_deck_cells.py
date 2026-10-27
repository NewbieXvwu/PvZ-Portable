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
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts')]

from deck_rules import RULE_VERSION, deck_violations, generate_deck, region_for_level  # noqa: E402

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
    return entry


def build_panels(registry: dict) -> dict:
    training = [task for task in registry['tasks'] if task['split'] == 'train']
    tasks = []
    for cell_index, task in enumerate(training):
        for multiplier in LADDER:
            suffix = '' if multiplier == 1.0 else '_m%03d' % round(multiplier * 100)
            start = seed_start(cell_index, multiplier)
            entry = task_document(task, multiplier, range(start, start + SEEDS_PER_TASK))
            entry['task_id'] = task['task_id'] + suffix
            tasks.append(entry)
    evaluation = [task_document(task, 1.0, []) for task in registry['tasks'] if task['split'] == 'eval']
    return dict(training_panel=dict(schema_version=1, purpose='task137 v5 random-deck curriculum cells',
                                    training_seed_block=TRAINING_SEED_BLOCK, tasks=tasks),
                eval_panel=dict(schema_version=1, purpose='task137 v5 unseen decks for selection and monitoring',
                                tasks=evaluation))


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
    args = parser.parse_args()
    if args.freeze_registry:
        print(json.dumps(generate_registry(), indent=1) + '\n')
        return
    registry = json.loads(REGISTRY.read_text())
    check(registry)
    if args.check:
        print(f'registry ok: {len(registry["tasks"])} tasks, rule v{registry["rule_version"]}')
        return
    panels = build_panels(registry)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, document in panels.items():
        (args.out / f'{name}.json').write_text(json.dumps(document, indent=1) + '\n')
    print(f'wrote {args.out}/training_panel.json and eval_panel.json: '
          f'{len(panels["training_panel"]["tasks"])} training / '
          f'{len(panels["eval_panel"]["tasks"])} eval tasks')


if __name__ == '__main__':
    main()
