#!/usr/bin/env python3
"""按区生成“原版难度评估面板”的公共核心（task21 夜间 / task22 泳池共用）。

一个区 = 10 关，每关一个评估任务 `<prefix>N_eval`：
  * 原版难度：`zombie_count_multiplier = 1.0`、`wave_cap = null`（不限波）；
  * 卡组默认用 `scripts/pvz_tasks.py` 的 `deck_for_level`（本文件不另写卡组规则），
    候选卡组通过 `Family` 声明，任务族名互不覆盖；
  * 其余字段照白天面板（`scripts/make_day_panel.py`，task17）：
    `sun_start = 50`、`preplanted = []`、`playthrough = 2`；
  * 种子默认用评估段 30000..31023（与白天/夜间面板同一段，1024 个/关）。

特殊关由各区的 `Zone.excluded` 声明（默认不收）与 `Zone.special_notes`（收了就在
清单和报告里标注语义）。卡组核对（AGENTS §8）在 `check_decks` 里做：地形覆盖、
升级植物所有权、长度与去重；不合法直接报错，不会写进清单。

用法见薄包装 `scripts/make_night_panel.py` / `scripts/make_pool_panel.py`。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

import pvz_constants  # noqa: E402
from pvz_tasks import conveyor_belt_level, deck_for_level, terrain_covered  # noqa: E402
from make_day_panel import SUN_START, unlocked_plants  # noqa: E402

SEED_START = 30000
SEEDS_PER_LEVEL = 1024


@dataclass(frozen=True)
class Family:
    """一个任务族 = 一副卡组（或 deck_for_level）+ 任务名后缀。"""
    suffix: str                 # "" = 对照族；"_n8" -> night_l11_eval_n8
    deck: tuple | None = None   # None = deck_for_level(level)
    kind: str = "eval"          # "eval" | "train"
    note: str = ""
    exclude_levels: tuple = ()  # 这个族不收的关（如训练族不收传送带 / 小游戏关）
    include_levels: tuple = ()
    phase: int = 0             # later additions follow all original tasks
    train_seed_base: int | None = None
    conveyor_start: str | None = None

    @property
    def label(self) -> str:
        return self.suffix.lstrip("_") or "control"


@dataclass(frozen=True)
class Zone:
    key: str                    # "night" / "pool"
    task_prefix: str            # "night_l" -> night_l11_eval
    first_level: int
    last_level: int
    terrain: str                # 任务清单里的 terrain 标签（元数据）
    default_out: Path
    purpose: str
    excluded: dict = field(default_factory=dict)        # level -> reason
    special_notes: dict = field(default_factory=dict)   # level -> note
    families: tuple = ()                                # (Family, ...) 第一个是对照
    train_seed_base: int | None = None
    train_seed_stride: int = 10000
    train_seeds: int = 8192
    reserved_seed_ranges: tuple = ()
    deck_check_note: str = ""


def train_seed_start(zone: Zone, level: int, family: Family | None = None) -> int:
    base = family.train_seed_base if family is not None and family.train_seed_base is not None else zone.train_seed_base
    if base is None:
        raise RuntimeError(f"{zone.key}: 没有训练段配置")
    return base + (level - zone.first_level) * zone.train_seed_stride


def check_seed_overlap(zone: Zone, tasks: list[dict]) -> None:
    """自查：任务内部种子不重复；训练种子不撞任何已用段。

    不同关卡、不同卡组共用同一段评估种子是**设计如此**（配对比较要同种子），
    所以重复只在“同一个任务内部”才算错。
    """
    owners = {}
    for task in tasks:
        if len(set(task["seeds"])) != len(task["seeds"]):
            raise RuntimeError(f"{task['task_id']}: 任务内部有重复种子")
        if "_train_" not in task["task_id"]:
            continue
        family = next(f for f in zone.families if f.kind == 'train' and task['task_id'] ==
                      f"{zone.task_prefix}{task['level']}_train{f.suffix}")
        start = train_seed_start(zone, task['level'], family)
        if min(task['seeds']) < start or max(task['seeds']) >= start + zone.train_seed_stride:
            raise RuntimeError(f"{task['task_id']}: 训练种子越出本关段")
        base = family.train_seed_base if family.train_seed_base is not None else zone.train_seed_base
        own_high = base + (zone.last_level-zone.first_level)*zone.train_seed_stride + zone.train_seeds-1
        for seed in task["seeds"]:
            for low, high in zone.reserved_seed_ranges:
                if low <= seed <= high and not (base <= low and high <= own_high):
                    raise RuntimeError(f"{task['task_id']}: seed {seed} 落在已用段 {low}..{high}")
            if seed in owners:
                raise RuntimeError(f"{task['task_id']}: seed {seed} 与 {owners[seed]} 重叠")
            owners[seed] = task['task_id']


def check_decks(zone: Zone, level: int, decks: dict[str, tuple]) -> list[str]:
    """AGENTS §8：地形覆盖 / 升级植物所有权 / 长度。返回问题列表（空 = 全合法）。"""
    problems = []
    upgrades = pvz_constants.upgrade_plants()
    for label, deck in decks.items():
        if not 6 <= len(deck) <= 10:
            problems.append(f"{label}: 卡组长度 {len(deck)} 不在 6..10")
        if len(set(deck)) != len(deck):
            problems.append(f"{label}: 卡组内有重复")
        if not terrain_covered(level, deck):
            problems.append(f"{label}: 地形没覆盖（{pvz_constants.background_for_level(level)} 需要睡莲 16 或香蒲 43）")
        owned = set(deck) & set(upgrades)
        if 43 in deck and 43 not in owned:  # 防御性：upgrade_plants() 应包含香蒲
            problems.append(f"{label}: 香蒲 43 未声明所有权")
        if 43 in deck and 16 not in deck:
            problems.append(f"{label}: 香蒲 43 没有睡莲底座")
    return problems


def task_document(zone: Zone, level: int, deck, task_id: str, seeds: list[int]) -> dict:
    return {
        "task_id": task_id,
        "terrain": zone.terrain,
        "level": level,
        "deck": list(deck),
        "zombie_count_multiplier": 1.0,
        "wave_cap": None,
        "sun_start": SUN_START,
        "preplanted": [],
        "playthrough": 2,
        "seeds": seeds,
    }


def family_note(zone: Zone, family: Family) -> str:
    if family.kind == "train":
        base=family.train_seed_base if family.train_seed_base is not None else zone.train_seed_base
        return (f"{zone.task_prefix}*_train{family.suffix} ({base} + (N-{zone.first_level})*"
                f"{zone.train_seed_stride}, {zone.train_seeds} each)")
    if family.suffix == "":
        return f"{zone.task_prefix}*_eval (control)"
    return f"{zone.task_prefix}*_eval{family.suffix}"


def build_document(zone: Zone, args) -> tuple[dict, list[dict], dict]:
    fixed_deck = tuple(int(seed) for seed in args.deck.split(",")) if args.deck else None

    skipped = dict(zone.excluded)
    for level in args.include:
        skipped.pop(level, None)
    for level in args.skip:
        skipped.setdefault(level, "skipped on the command line")

    levels = [level for level in range(zone.first_level, zone.last_level + 1) if level not in skipped]
    families = list(zone.families)
    if not args.combined:
        families = families[:1]
    if fixed_deck is not None:
        families = [Family(args.suffix, fixed_deck, "eval")]

    tasks, checks, problems = [], [], []

    def emit(family: Family, level: int) -> None:
        deck = family.deck if family.deck is not None else deck_for_level(level)
        if family.kind == "train":
            start = train_seed_start(zone, level, family)
            seeds = list(range(start, start + args.train_seeds))
            task_id = f"{zone.task_prefix}{level}_train{family.suffix}"
        else:
            seeds = list(range(args.seed_start, args.seed_start + args.seeds))
            task_id = f"{zone.task_prefix}{level}_eval{family.suffix}"
        task = task_document(zone, level, deck, task_id, seeds)
        if family.conveyor_start:
            if family.conveyor_start != 'original' or deck or not conveyor_belt_level(level):
                raise RuntimeError('original conveyor task needs a conveyor level and empty deck')
            task['conveyor_start'] = family.conveyor_start
        tasks.append(task)

    def family_levels(family: Family) -> list[int]:
        return [level for level in levels if level not in family.exclude_levels
                and (not family.include_levels or level in family.include_levels)]

    # 顺序与 task21 的清单逐字节一致：对照族先跑完所有关，再按关排其余族。
    for level in family_levels(families[0]):
        emit(families[0], level)
        deck = families[0].deck if families[0].deck is not None else deck_for_level(level)
        unlocked = unlocked_plants(level)
        checks.append({"level": level, "background": pvz_constants.background_for_level(level),
                       "deck": list(deck), "unlocked": list(unlocked),
                       "not_yet_unlocked": [seed for seed in deck if seed not in unlocked],
                       "special": zone.special_notes.get(level, "")})
    for phase in sorted({family.phase for family in families[1:]}):
        for level in levels:
            for family in families[1:]:
                if family.phase == phase and level in family_levels(family):
                    emit(family, level)

    check_seed_overlap(zone, tasks)

    # AGENTS §8 卡组核对：每关 × 每副候选卡组。
    for level in levels:
        decks = {}
        for family in families:
            if level not in family_levels(family) or family.conveyor_start:
                continue
            deck = family.deck if family.deck is not None else deck_for_level(level)
            decks[family.label] = deck
        for message in check_decks(zone, level, decks):
            problems.append(f"L{level} {message}")
    if problems:
        raise RuntimeError("卡组不合法（AGENTS §8）：\n" + "\n".join(problems))

    note = (f"local manifest, not in git; decks come from scripts/pvz_tasks.py deck_for_level, "
            f"seeds {args.seed_start}.. are the same evaluation block as the day panel")
    if args.combined and len(zone.families) > 1:
        note += "; families: " + ", ".join(family_note(zone, family) for family in zone.families)

    document = {
        "schema_version": 1,
        "split": "stage4",
        "purpose": zone.purpose,
        "note": note,
        "skipped_levels": {str(level): reason for level, reason in sorted(skipped.items())},
        "tasks": tasks,
    }
    return document, checks, {"levels": levels, "families": families, "skipped": skipped}


def main_for(zone: Zone, argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=zone.purpose, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=zone.default_out)
    p.add_argument("--seeds", type=int, default=SEEDS_PER_LEVEL)
    p.add_argument("--seed-start", type=int, default=SEED_START)
    p.add_argument("--train-seeds", type=int, default=zone.train_seeds)
    p.add_argument("--skip", type=int, action="append", default=[],
                   help="不收的关卡（默认见各区的 excluded）")
    p.add_argument("--include", type=int, action="append", default=[],
                   help="强制收进默认排除的关卡")
    p.add_argument("--deck", default="",
                   help="逗号分隔的种子编号，覆盖 deck_for_level（全关卡同一副卡）")
    p.add_argument("--suffix", default="",
                   help="配合 --deck 用的任务名后缀（如 _n8 -> night_l11_eval_n8）")
    p.add_argument("--combined", action="store_true",
                   help="一份清单里写本区的全部任务族（主控的采集队列读这份）")
    args = p.parse_args(argv)
    if args.seeds <= 0 or not 0 < args.train_seeds <= zone.train_seed_stride:
        p.error('seed counts must be positive; training count cannot exceed the per-level stride')

    document, checks, meta = build_document(zone, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(document['tasks'])} tasks, {args.seeds} seeds each "
          f"({args.seed_start}..{args.seed_start + args.seeds - 1})")
    print(f"skipped: {document['skipped_levels']}")
    print()
    print("level  background             deck                  not unlocked yet  special")
    for row in checks:
        print(f"{row['level']:5d}  {row['background']:<22s} {str(row['deck']):<21s} "
              f"{str(row['not_yet_unlocked'] or '-'):<17s} {row['special']}")
    expected = pvz_constants.background_for_level(zone.first_level)
    bad_terrain = [row["level"] for row in checks if row["background"] != expected]
    if bad_terrain:
        print(f"WARNING: background != {expected}: {bad_terrain}")
    if args.combined:
        for family in zone.families[1:]:
            if family.kind != "eval" or family.deck is None:
                continue
            bad = {row["level"]: [seed for seed in family.deck if seed not in row["unlocked"]] for row in checks}
            bad = {level: missing for level, missing in bad.items() if missing}
            print(f"{family.label} deck {list(family.deck)}: "
                  + ("all unlocked" if not bad else f"NOT unlocked: {bad}"))
        for family in zone.families:
            if family.kind != "train":
                continue
            for level in [lv for lv in meta["levels"] if lv not in family.exclude_levels
                          and (not family.include_levels or lv in family.include_levels)]:
                start = train_seed_start(zone, level, family)
                print(f"train{family.suffix} L{level}: {start}..{start + args.train_seeds - 1}")
