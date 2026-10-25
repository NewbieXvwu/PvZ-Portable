#!/usr/bin/env python3
"""task29 第 1 步：给现有训练族生成**僵尸数量倍率**变体（新族名 `<base>_m<XXX>`）。

倍率语义（引擎源码，`src/Lawn/Board.cpp:4913` `Board::SpawnZombieWave`）：

    aBaseCount = NumberZombiesInWave(wave)          # 这一波原版僵尸数
    aSpawnCount = round(aBaseCount * multiplier)     # 本环境按倍率放大
    type = mZombiesInWave[wave][i % aBaseCount]      # 原版名单循环取用

所以它放大的是**每一波常规出怪的数量**，种类与比例沿用原版名单（`round` 与取模会让
名单前几张多出现 0–1 次）。它**不**放大原版在最后一波之后额外刷的那批：
坟头 `Board::SpawnZombiesFromGraves`（Board.cpp:4850，每个坟头一只）、
泳池 `Board::SpawnZombiesFromPool`（Board.cpp:4728，固定 2–3 只）、
屋顶 `SpawnZombiesFromSky`。这三处不读 multiplier，是原版行为。

本文件只做“复制任务定义 + 换 id / 换种子段 / 改 multiplier”，不改卡组、不改
`deck_for_level`，也不碰 `deck_perturb`（扰动族的 `deck_perturb` 原样带过来，
`deck_for_seed` 仍按新种子的确定性结果选卡组）。

用法：
  python3 scripts/make_multiplier_tasks.py --dry-run     # 只打印分配表
  python3 scripts/make_multiplier_tasks.py               # 写清单 + 拷贝
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = Path('/home/newbiexvwu/PvZ-search')
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_tasks import deck_for_seed, terrain_covered  # noqa: E402

# 倍率档 -> 族名后缀。任务书要求 1.25 / 1.5 / 1.75 / 2.0 / 2.5 / 3.0。
MULTIPLIERS = ((1.25, 'm125'), (1.5, 'm150'), (1.75, 'm175'),
               (2.0, 'm200'), (2.5, 'm250'), (3.0, 'm300'))

# 种子段：5_000_000 起，每档 700_000。档内按 (族序号, 关卡) 排槽：
#   slot = family_index * 32 + (level - 1)，每槽 SEEDS_PER_TASK 个种子。
# 8 个族 × 32 槽 × 2048 = 524_288 < 700_000，档与档之间留 175k 空隙（够再加一个族：
# 9 个族是 589_824）。task31 加第 8 个族 n10q 时 524_288 已经超过原来的 500_000 档距，
# 会串到下一档，所以整档加宽。加宽后旧倍率种子整体平移：课程控制器 10-22 起已经用
# 500k 档距的清单跑过 _m125 / _m150 的局（`curriculum_state.json` 的 offsets 里记着，
# 例如 day_l7_train_d7_m125: 96）。控制器按 task_id 记偏移、按偏移往后取新种子，
# 不会重复；只是同一 task_id 的前后两段种子来自不同档距的段，游戏记录里存的是种子值本身。
# 与仓库里所有现有清单（最高约 3.4M 的 n10q 段）以及 DESIGN 登记的 DAgger 段都不相交。
SEED_BASE = 5_000_000
MULT_STRIDE = 700_000
SLOTS_PER_FAMILY = 32
SEEDS_PER_TASK = 2048

# task46: roof levels exceed the legacy 32-level slot width. Give them an independent
# range rather than shifting any existing family's seeds (also independent of task45 fog).
ROOF_MULT_SEED_BASE = 48_000_000
ROOF_MULT_STRIDE = 100_000

# task68: the ten legacy families already fill 655_360 of the 700_000-wide band, so a new
# family cannot take another slot without moving every preregistered seed. It gets its own
# block instead (same rule as roof). The base task keeps 43_000_000..43_008_191; the six
# multiplier bands start at 43_100_000.
FOG_USER_MULT_SEED_BASE = 43_100_000
FOG_USER_MULT_STRIDE = 100_000

# task70: fog_user_v2 (same deck/level as fog_user, fog_observation human_v2) is another
# independent block. 48.52M-50M is the first free window above roof (verified against all
# 1.67M seeds already present in the repository manifests): base task keeps
# 49_000_000..49_008_191, eval sibling 49_100_000..49_100_031, multiplier bands 49_200_000+.
FOG_USER_V2_MULT_SEED_BASE = 49_200_000
FOG_USER_V2_MULT_STRIDE = 100_000
INDEPENDENT_BLOCKS = {
    # family -> (multiplier seed base, band width, first level of the block)
    'roof': (ROOF_MULT_SEED_BASE, ROOF_MULT_STRIDE, 41),
    'fog_user': (FOG_USER_MULT_SEED_BASE, FOG_USER_MULT_STRIDE, 31),
    'fog_user_v2': (FOG_USER_V2_MULT_SEED_BASE, FOG_USER_V2_MULT_STRIDE, 31),
}

DAY_D7_LEVELS = (1, 3, 4, 6, 7, 8, 9)
NIGHT_N10_LEVELS = (11, 12, 13, 14, 16, 17, 18, 19)
POOL_P9K_LEVELS = (21, 22, 23, 24, 26, 27, 28, 29)
DAY_D7Q_LEVELS = (7, 8, 9, 10)
NIGHT_N10P_LEVELS = (11, 12, 13, 14, 16, 17, 18, 19)
NIGHT_N10Q_LEVELS = (11, 12, 13, 14, 16, 17, 18, 19)
ROOF_LEVELS = (41, 42, 43, 44, 46, 47, 48, 49)

# (族短名, 源清单 key, 源 task_id 模板, levels, 地形)
FAMILIES = (
    ('day_d7', 'day', 'day_l{level}_train_d7', DAY_D7_LEVELS, 'day'),
    ('day_orig', 'day_orig', 'day_l10_orig_train', (10,), 'day'),
    ('night_n10', 'night', 'night_l{level}_train_n10', NIGHT_N10_LEVELS, 'night'),
    ('night_orig', 'night', 'night_l20_train_n10_orig', (20,), 'night'),
    ('pool_p9k', 'pool', 'pool_l{level}_train_p9k', POOL_P9K_LEVELS, 'pool'),
    ('day_d7q', 'day_d7q', 'day_l{level}_train_d7q', DAY_D7Q_LEVELS, 'day'),
    ('night_n10p', 'night_n10p', 'night_l{level}_train_n10p', NIGHT_N10P_LEVELS, 'night'),
    ('night_n10q', 'night_n10q', 'night_l{level}_train_n10q', NIGHT_N10Q_LEVELS, 'night'),
    ('fog_f8', 'fog', 'fog_l{level}_train_f8', (31, 32, 33, 34, 36, 37, 38, 39), 'fog'),
    ('fog_f8p', 'fog', 'fog_l{level}_train_f8p', (31, 32, 33, 34, 36, 37, 38, 39), 'fog'),
    ('roof', 'roof', 'roof_l{level}_train_r7', ROOF_LEVELS, 'roof'),
    # task68: the only fog cold-start family above 50% (teacher 25/32, task67).
    ('fog_user', 'fog_user', 'fog_user_l{level}_snow', (31,), 'fog'),
    # task70: same deck/level/multipliers as fog_user, but the fog observation is human_v2
    # (task66 rendered-opacity rule). Observation is a family property (AGENTS.md section 8),
    # so this is a new family with a fresh seed block; fog_user is superseded below.
    ('fog_user_v2', 'fog_user_v2', 'fog_user_v2_l{level}_snow', (31,), 'fog'),
)

# Registry coverage and curriculum admission are separate: unmeasured roof levels have
# reproducible tasks/multipliers, but cannot enter collection before the cold-start gate.
# Filled from the complete, preregistered teacher evaluation (>=50% at original difficulty).
# task46-roof-admission.json: r7 teacher L41 = 15/16.
# task68: fog_user L31 is admitted at multiplier 1.0 (task67 teacher 25/32 = 78.1%, paired
# p<1e-6 against 0/32 for the fog_f8 deck on the same seeds). The task45 fog_f8/f8p families
# stay outside the curriculum: they were never measured above 50%.
# task70: fog_user_v2 replaces fog_user in the curriculum now that its own teacher cold start
# is on file (B5 revision, 31/32 on deploy4 with production v5+branch1; paired human control
# 29/32, exact McNemar p=0.625 -- the observation is not what separates them). The old family
# keeps its manifest, seeds and data.
CURRICULUM_LEVELS = {'roof': (41,), 'fog_f8': (), 'fog_f8p': (), 'fog_user_v2': (31,)}

# 课程里被新族取代的族（旧族的清单、种子段、数据、评估全部保留，只是课程不再给它排新局）。
# 依据：task31 / DESIGN §14.14 —— n10p 换掉 8/9 号卡的局是结构性输局。
SUPERSEDED = {'night_n10p': 'night_n10q', 'fog_user': 'fog_user_v2'}


def variant_id(base_id: str, suffix: str) -> str:
    return f'{base_id}_{suffix}'


def seed_start(mult_index: int, family_index: int, level: int) -> int:
    family = FAMILIES[family_index][0]
    if family in INDEPENDENT_BLOCKS:
        base, stride, first = INDEPENDENT_BLOCKS[family]
        if not first <= level <= first + SLOTS_PER_FAMILY - 1:
            raise ValueError(f'{family} multiplier level must be {first}..{first + SLOTS_PER_FAMILY - 1}')
        return base + mult_index * stride + (level - first) * SEEDS_PER_TASK
    slot = family_index * SLOTS_PER_FAMILY + (level - 1)
    return SEED_BASE + mult_index * MULT_STRIDE + slot * SEEDS_PER_TASK


def source_paths(overrides: dict[str, Path]) -> dict[str, Path]:
    return {
        'day': overrides.get('day') or MAIN / 'artifacts/stage4/day_panel.json',
        # task20 的原版传送带族只在 conveyor 工作树的清单里（主清单是 task23 的 26 项版）。
        'day_orig': overrides.get('day_orig') or MAIN / '.claude/worktrees/conveyor/artifacts/stage4/day_panel.json',
        'night': overrides.get('night') or MAIN / 'artifacts/stage4/night_panel.json',
        'pool': overrides.get('pool') or MAIN / 'artifacts/stage4/pool_panel.json',
        'day_d7q': overrides.get('day_d7q') or MAIN / 'artifacts/stage4/day_panel_d7q.json',
        'night_n10p': overrides.get('night_n10p') or MAIN / 'artifacts/stage4/night_panel_n10p.json',
        'night_n10q': overrides.get('night_n10q') or MAIN / 'artifacts/stage4/night_panel_n10q.json',
        'roof': overrides.get('roof') or ROOT / 'artifacts/task46/roof_panel.json',
        'fog': overrides.get('fog') or ROOT / 'artifacts/task45/fog_panel.json',
        # task68: local manifest from scripts/make_fog_user_panel.py (not in git).
        'fog_user': overrides.get('fog_user') or ROOT / 'artifacts/task68/fog_user_panel.json',
        # task70: local manifest from scripts/make_fog_user_v2_panel.py (not in git).
        'fog_user_v2': overrides.get('fog_user_v2') or ROOT / 'artifacts/task70/fog_user_v2_panel.json',
    }


def load_tasks(path: Path) -> dict[str, dict]:
    document = json.loads(path.read_text())
    entries = document['tasks'] if isinstance(document, dict) and 'tasks' in document else document
    return {task['task_id']: task for task in entries}


def build_variants(sources: dict[str, dict[str, dict]], seeds: int = SEEDS_PER_TASK) -> list[dict]:
    """按 FAMILIES × MULTIPLIERS 生成变体；源任务除 id / seeds / multiplier 外逐字段复制。"""
    tasks: list[dict] = []
    for family_index, (family, key, pattern, levels, _terrain) in enumerate(FAMILIES):
        table = sources[key]
        for level in levels:
            base = table[pattern.format(level=level)]
            for index, (multiplier, suffix) in enumerate(MULTIPLIERS):
                task = copy.deepcopy(base)
                start = seed_start(index, family_index, level)
                task.update(task_id=variant_id(base['task_id'], suffix),
                            seeds=list(range(start, start + seeds)),
                            zombie_count_multiplier=multiplier)
                tasks.append(task)
    return tasks


def existing_seeds(paths: list[Path], exclude: set[Path]) -> dict[int, str]:
    seen: dict[int, str] = {}
    for path in paths:
        if path.resolve() in exclude or path.name == 'multiplier_panel.json':
            # 本脚本生成的那份（工作树 / 主目录 / 别的工作树的副本）是派生文件，整份重写；
            # 把它自己算进“现有清单”会让新种子和自己的旧种子撞车（task31 加族时撞过）。
            continue
        try:
            document = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        entries = document['tasks'] if isinstance(document, dict) and 'tasks' in document else document
        if not isinstance(entries, list):
            continue
        for task in entries:
            if not isinstance(task, dict):
                continue
            seeds = task.get('seeds')
            if not isinstance(seeds, list):
                continue
            for seed in seeds:
                if isinstance(seed, int):
                    seen.setdefault(seed, str(task.get('task_id')))
    return seen


def scan_manifests() -> list[Path]:
    patterns = [str(MAIN / 'artifacts/**/*.json'), str(MAIN / '.claude/worktrees/*/artifacts/**/*.json')]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(pattern, recursive=True) if Path(p).is_file())  # skip dangling links
    return sorted(set(paths))


def audit(tasks: list[dict], sources: dict[str, dict[str, dict]], manifest_paths: list[Path],
          exclude: set[Path]) -> dict:
    assert len(tasks) == sum(len(levels) for _, _, _, levels, _ in FAMILIES) * len(MULTIPLIERS)
    # 1) 新种子全局唯一。
    owner: dict[int, str] = {}
    for task in tasks:
        assert len(task['seeds']) == SEEDS_PER_TASK
        for seed in task['seeds']:
            assert seed not in owner, ('duplicate new seed', seed, owner[seed], task['task_id'])
            owner[seed] = task['task_id']
    # 2) 与所有现有清单穷举比对。
    seen = existing_seeds(manifest_paths, exclude)
    clashes = sorted(seed for seed in owner if seed in seen)
    assert not clashes, ('seed overlap', clashes[:5], len(clashes))
    # 3) 除 id / seeds / multiplier 外逐字段等于源任务；扰动族的 deck_perturb 原样保留。
    by_id = {task['task_id']: task for task in tasks}
    checked_decks = 0
    for family, key, pattern, levels, terrain in FAMILIES:
        table = sources[key]
        for level in levels:
            base = table[pattern.format(level=level)]
            for multiplier, suffix in MULTIPLIERS:
                task = by_id[variant_id(base['task_id'], suffix)]
                assert task['zombie_count_multiplier'] == multiplier
                assert task['level'] == level and task['terrain'] == terrain
                for field, value in base.items():
                    if field in ('task_id', 'seeds', 'zombie_count_multiplier'):
                        continue
                    assert task.get(field) == value, (task['task_id'], field)
                # 4) 每个种子都能取到合法卡组（扰动族按 deck_for_seed，其余等于底卡组）。
                for seed in task['seeds']:
                    deck = deck_for_seed(task, seed)
                    assert len(deck) == len(base['deck']) and len(set(deck)) == len(deck)
                    assert terrain_covered(level, deck), (task['task_id'], seed, deck)
                    if 'deck_perturb' not in task:
                        assert tuple(deck) == tuple(base['deck'])
                    checked_decks += 1
    return dict(tasks=len(tasks), seeds=len(owner), decks_checked=checked_decks,
                scanned_manifests=len(manifest_paths), scanned_seeds=len(seen), seed_overlap=len(clashes),
                seed_base=SEED_BASE, mult_stride=MULT_STRIDE,
                independent_blocks={name: base for name, (base, _stride, _first)
                                    in sorted(INDEPENDENT_BLOCKS.items())},
                slots_per_family=SLOTS_PER_FAMILY, seeds_per_task=SEEDS_PER_TASK,
                multipliers=[m for m, _ in MULTIPLIERS])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/stage4/multiplier_panel.json')
    parser.add_argument('--copy-to', type=Path,
                        default=MAIN / 'artifacts/stage4/multiplier_panel.json')
    parser.add_argument('--day-manifest', type=Path)
    parser.add_argument('--day-orig-manifest', type=Path)
    parser.add_argument('--night-manifest', type=Path)
    parser.add_argument('--pool-manifest', type=Path)
    parser.add_argument('--day-d7q-manifest', type=Path)
    parser.add_argument('--night-n10p-manifest', type=Path)
    parser.add_argument('--night-n10q-manifest', type=Path)
    parser.add_argument('--roof-manifest', type=Path)
    parser.add_argument('--fog-manifest', type=Path)
    parser.add_argument('--fog-user-manifest', type=Path)
    parser.add_argument('--seeds', type=int, default=SEEDS_PER_TASK)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.seeds <= SEEDS_PER_TASK
    paths = source_paths({'day': args.day_manifest, 'day_orig': args.day_orig_manifest,
                          'night': args.night_manifest, 'pool': args.pool_manifest,
                          'day_d7q': args.day_d7q_manifest, 'night_n10p': args.night_n10p_manifest,
                          'night_n10q': args.night_n10q_manifest, 'roof': args.roof_manifest,
                          'fog': args.fog_manifest, 'fog_user': args.fog_user_manifest})
    sources = {key: load_tasks(path) for key, path in paths.items()}
    tasks = build_variants(sources, args.seeds)
    manifests = scan_manifests()
    summary = audit(tasks, sources, manifests, {args.out.resolve(), args.copy_to.resolve()})
    if args.dry_run:
        print(f"{'family':<12}{'levels':>7}{'tasks':>7}{'seeds/task':>12}{'seed range':>28}")
        for family_index, (family, key, pattern, levels, _terrain) in enumerate(FAMILIES):
            starts = [seed_start(i, family_index, level) for i, _ in enumerate(MULTIPLIERS) for level in levels]
            ends = [seed_start(i, family_index, level) + args.seeds - 1
                    for i, _ in enumerate(MULTIPLIERS) for level in levels]
            print(f'{family:<12}{len(levels):>7}{len(levels) * len(MULTIPLIERS):>7}{args.seeds:>12}'
                  f'{min(starts):>16}..{max(ends):<11}')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    document = dict(schema_version=1, purpose='task29 adaptive curriculum: zombie_count_multiplier variants',
                    multiplier_semantics='Board::SpawnZombieWave scales round(base * mult) per wave; '
                                         'graveyard/pool/sky final-wave bonus spawns are not scaled',
                    generation_audit=summary, tasks=tasks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + '\n')
    args.copy_to.parent.mkdir(parents=True, exist_ok=True)
    args.copy_to.write_text(json.dumps(document, indent=1) + '\n')
    print(f'wrote {args.out} and {args.copy_to}: {summary["tasks"]} tasks, {summary["seeds"]} seeds')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
