"""任务卡组：按关卡地形配卡组（AGENTS.md §8「任务与卡组」），以及对应的玩家存档设定。

卡组规则只改这里。从 scripts/scripted_baseline.py 搬来（规则脚本策略已于 2026-10-19 删除）。
"""
from __future__ import annotations

import sys
import random
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pvz_env import PlayerProfileContext, SeedCard  # noqa: E402
import pvz_constants  # noqa: E402

DECK = (0, 1, 2, 3, 4, 5)
SUN_SHROOM = 9      # 阳光菇 25☀，夜间经济（夜间天不掉阳光，实测见 pvz_constants terrain 节）
LILY_PAD = 16       # 睡莲 25☀，水路底座
CATTAIL = 43        # 香蒲 225☀，睡莲的升级植物（只能种在睡莲上）
FLOWER_POT = 33     # 花盆 25☀，屋顶底座（本环境屋顶关开局已预置 c0–c4）
# task46 pilot c02: actual teacher 2/2; separate r7 family, legacy defaults stay frozen.
ROOF_R7_DECK = (0, 1, 2, 3, 32, 35, 33)


@lru_cache(maxsize=50)
def conveyor_belt_level(level: int) -> bool:
    """Read Adventure conveyor eligibility from Board/LawnApp, including special levels."""
    return level in pvz_constants.adventure_conveyor_levels()


def player_deck_levels(*, include_boss: bool = False) -> tuple[int, ...]:
    """普通冒险选卡关；沿用传送带判定，另排除砸罐子与打僵尸。

    L50 仅供已经支持任务卡组覆盖 Boss 卡槽的接口显式启用。
    """
    return tuple(level for level in range(1, 51)
                 if (level == 50 and include_boss)
                 or (level not in (15, 35) and not conveyor_belt_level(level)))


def deck_pool_for_level(level: int, blacklist) -> list[int]:
    """二周目可表达的选卡范围减任务黑名单；不使用随机戴夫推荐。"""
    scene = pvz_constants.background_for_level(level)
    region = scene.split("_")[-1].lower()
    banned = set(blacklist.get("all", [])) | set(blacklist.get(region, []))
    return sorted(d["type"] for d in pvz_constants.plant_defs()
                  if 0 <= d["type"] < 48 and d["type"] not in banned)


def deck_for_level(level: int, conveyor_start: str | None = None, *,
                   family: str | None = None) -> tuple[int, ...]:
    """按关卡地形排卡组。规则：**卡组必须覆盖这一关的每一格地形**。

    这是 2026-10-03 定下的任务约束（AGENTS.md「任务与卡组」节）：
    之前全关卡共用 0..5 六张牌，泳池关（21–30）两条水路没有任何植物能种，
    属于结构性输局 —— 诊断它会得出"卡组×地形不匹配"这种一次性的结论，
    而不是可教的政策缺陷。场景判定用 `pvz_constants.background_for_level`
    （从 Board::PickBackground 解析，含第 35 关 ScaryPotter 特例）。
    """
    if family is not None:
        if family != 'roof_r7' or not 41 <= level <= 49 or conveyor_start is not None:
            raise ValueError('roof_r7 requires level 41..49 and the fixed initial-bank start')
        return ROOF_R7_DECK
    # 原版传送带开局没有任务卡组：卡由 Challenge::UpdateConveyorBelt 发。
    if conveyor_start == "original":
        if not conveyor_belt_level(level):
            raise ValueError('original conveyor start requires an engine conveyor level')
        return ()
    bg = pvz_constants.background_for_level(level)
    if bg == "BACKGROUND_3_POOL":          # 21–30 白天泳池：水路两张
        return DECK + (LILY_PAD, CATTAIL)
    if bg == "BACKGROUND_4_FOG":           # 31–40 夜间泳池有雾：经济换阳光菇
        return (0, 9, 2, 3, 4, 5, LILY_PAD, CATTAIL)
    if bg == "BACKGROUND_2_NIGHT":         # 11–20（含 35）夜间草地：加阳光菇
        return DECK[:2] + (SUN_SHROOM,) + DECK[2:]
    if bg in ("BACKGROUND_5_ROOF", "BACKGROUND_6_BOSS"):   # 41–50：加花盆备用
        return DECK + (FLOWER_POT,)
    return DECK                            # 1–10 白天草地：原六卡


def profile_for_deck(deck: tuple[int, ...], first_aid_owned: bool = False) -> PlayerProfileContext:
    """卡组 -> PlayerProfileContext。seed 槽位数 = 卡组长度；
    卡组里出现的升级植物必须声明所有权，否则模拟器直接拒绝 reset
    （LawnApp.cpp:1329 校验；升级植物集合从 Plant::IsUpgrade 解析）。

    first_aid_owned（坚果包扎术）是**任务属性**，不是卡组属性：打开后
    Board::CanPlantAt 多允许一种落点（坚果 / 高坚果血量 < 2/3 时可以在它身上
    再种一个同种坚果），合法动作由模拟器自己给出，这里只负责把所有权传进
    profile。缺省 False，旧任务的 reset 报文逐字节不变（AGENTS.md §8：
    改任务 = 换任务族，任务清单里用 "first_aid_owned": true 显式打开）。"""
    seed_types, imitater_owned = [], False
    for card in deck:
        if type(card) is int:
            seed_type = card
        elif isinstance(card, SeedCard):
            seed_type = card.seed_type
            imitater_owned |= seed_type == pvz_constants._seed_index('IMITATER')
        elif isinstance(card, tuple) and len(card) == 2 and all(type(value) is int for value in card):
            seed_type = card[0]
            imitater_owned |= seed_type == pvz_constants._seed_index('IMITATER')
        else:
            raise ValueError('deck entries must be seed IDs, SeedCard values, or (imitater, target) pairs')
        seed_types.append(seed_type)
    upgrades = tuple(sorted(set(seed_types) & set(pvz_constants.upgrade_plants())))
    # 空卡组 = 原版传送带开局（任务 deck: []）：没有任务卡组，profile 仍用正常 6 槽，
    # 棋盘用引擎自己的 10 格传送带。
    return PlayerProfileContext(seed_slot_count=len(deck) if deck else 6,
                                owned_upgrade_plants=upgrades, imitater_owned=imitater_owned,
                                first_aid_owned=bool(first_aid_owned))


@lru_cache(maxsize=50)
def _perturbation_background(level: int) -> str:
    return pvz_constants.background_for_level(level)


def terrain_covered(level: int, deck) -> bool:
    """AGENTS §8 coverage only; no handwritten playing or card-strength policy."""
    background = _perturbation_background(level)
    if background in ('BACKGROUND_3_POOL', 'BACKGROUND_4_FOG'):
        return LILY_PAD in deck or CATTAIL in deck
    if background in ('BACKGROUND_5_ROOF', 'BACKGROUND_6_BOSS'):
        return FLOWER_POT in deck
    return True


def random_deck_draws(task: dict, seed: int):
    """Uniform length and ordered samples from the engine candidate pool recorded in the task.

    Whole-deck eligibility is queried by the manifest generator, never reimplemented here.
    Keep the chosen length fixed during rejection sampling so all five lengths remain equiprobable.
    """
    spec = task['deck_random']
    options = spec['engine_options']
    if options['level'] != task['level'] or options['playthrough'] != task.get('playthrough', 2):
        raise ValueError('random deck engine options do not match task level/profile')
    pool = sorted(set(int(card) for card in options['available_plants']))
    if len(pool) < 10:
        raise ValueError('random deck pool must support every length in 6..10')
    rng = random.Random(int(seed))
    length = rng.choice(range(6, 11))
    for attempt in range(1, 4097):
        yield attempt, tuple(rng.sample(pool, length))


def random_deck_covered(task: dict, deck) -> bool:
    """Only terrain coverage and held-out card-set exclusion; no strength judgement."""
    spec = task['deck_random']
    excluded = {tuple(sorted(cards)) for cards in spec['excluded_decks']}
    background = _perturbation_background(task['level'])
    if background in ('BACKGROUND_3_POOL', 'BACKGROUND_4_FOG') and LILY_PAD not in deck:
        return False
    return terrain_covered(task['level'], deck) and tuple(sorted(deck)) not in excluded


def random_deck_for_seed(task: dict, seed: int) -> tuple[int, ...]:
    """Regenerate the exact engine-validated draw for a registered seed.

    A receipt includes the actual deck as well as its draw index: changing the sampler/runtime must
    fail visibly instead of silently changing a task. Archives replay their stored actual deck.
    """
    spec = task['deck_random']
    key = str(int(seed))
    receipt = spec.get('validated_decks', {}).get(key)
    if receipt is None:
        raise ValueError('random deck seed has no engine-validation receipt; regenerate the task manifest')
    accepted = receipt['attempt']
    if type(accepted) is not int or not 1 <= accepted <= 4096:
        raise ValueError('invalid random deck draw receipt')
    for attempt, deck in random_deck_draws(task, seed):
        if attempt != accepted:
            continue
        if tuple(receipt['deck']) != deck or not random_deck_covered(task, deck):
            raise ValueError('random deck differs from its engine-validated receipt')
        profile = profile_for_deck(deck)
        if profile.seed_slot_count != len(deck):
            raise ValueError('random deck profile slot count differs from deck length')
        return deck
    raise ValueError('random deck draw receipt is out of range')


def deck_for_seed(task: dict, seed: int) -> tuple[int, ...]:
    """Deterministic exact-set replacements then slot shuffle; held-out sets are never returned.

    Eligibility is supplied by a frozen DECK_OPTIONS engine query, not a manually ranked card list.
    New choices exclude upgrades, imitater and special seeds; pre-existing upgrades retain their profile.
    """
    if 'deck_random' in task:
        if 'deck_perturb' in task:
            raise ValueError('random and perturbation deck families are distinct')
        return random_deck_for_seed(task, seed)
    base = tuple(task['deck'])
    if 'deck_perturb' not in task:
        return base
    spec = task['deck_perturb']
    options = spec['engine_options']
    if options['level'] != task['level'] or options['playthrough'] != task.get('playthrough', 2):
        raise ValueError('deck perturbation engine options do not match task level/profile')
    if not 6 <= len(base) <= 10 or len(set(base)) != len(base):
        raise ValueError('deck perturbation requires 6..10 distinct cards')
    available = set(options['available_plants'])
    warnings = dict(options['not_recommended'])
    pool = sorted(set(spec['pool']))
    if not pool or any(p not in available or not 0 <= p < 40 or warnings.get(p, 1) for p in pool):
        raise ValueError('perturbation pool includes locked, upgraded/special, or engine-warned plants')
    if not set(base) <= available:
        raise ValueError('base deck includes locked plants')
    swaps = spec.get('swaps', [1, 2])
    if not swaps or any(n not in (1, 2) for n in swaps):
        raise ValueError('deck perturbation supports exactly one or two replacements')
    additions = sorted(set(pool) - set(base))
    if len(additions) < max(swaps):
        raise ValueError('not enough new eligible plants for the requested swaps')
    # task31: cards measured to be structurally necessary (engine-agnostic, data-driven table).
    # A kept card's slot is never a replacement target; everything else is unchanged.
    keep = tuple(sorted(set(spec.get('keep_plants', []))))
    if any(card not in base for card in keep):
        raise ValueError('keep_plants must be cards of the base deck')
    replaceable = [slot for slot, card in enumerate(base) if card not in keep]
    if len(replaceable) < max(swaps):
        raise ValueError('not enough replaceable slots after keep_plants')
    excluded = {tuple(sorted(deck)) for deck in spec.get('excluded_decks', [])}
    # task28: whole-deck OnStartButton warnings (no sun producer, pool without an aquatic plant,
    # roof without a flower pot, ...).  The list is produced by the engine query
    # PvZEnv.deck_warnings over every deck this perturbation can generate; nothing is re-derived here.
    warned = {tuple(sorted(deck)) for deck in spec.get('engine_warned_decks', [])}
    if spec.get('require_no_warnings') and 'engine_warned_decks' not in spec:
        raise ValueError('require_no_warnings needs the engine_warned_decks table')
    rng = random.Random(int(seed))
    for _ in range(4096):
        n = rng.choice(swaps)
        slots = rng.sample(replaceable, n)
        new = rng.sample(additions, n)
        deck = list(base)
        for slot, plant in zip(slots, new):
            deck[slot] = plant
        if not terrain_covered(task['level'], deck) or tuple(sorted(deck)) in excluded:
            continue
        if tuple(sorted(deck)) in warned:
            continue
        if any(warnings.get(plant, 1) for plant in deck):
            continue
        rng.shuffle(deck)
        return tuple(deck)
    raise ValueError('no admissible non-held-out deck under the requested perturbation')

# ---------------------------------------------------------------------------
# task208：每关「好卡组」。门禁 / 北极星 / 训练评测的卡组以这张表为准。
# deck_for_level 仍然保留，但它的职责缩回 AGENTS.md §8 的地形覆盖检查；
# 评测选卡不再走它（旧卡组大多不是好解法，基于它们的评测没有公信力）。
# 表由 MEMORY/docs/task208-good-decks.md 记录的候选 + 老师证据生成，
# 刷新流程见该文档「复现」节；同一关的卡组彼此至少差 3 张卡，思路不同。
# ---------------------------------------------------------------------------
# BEGIN GOOD_DECKS (task208)
GOOD_DECKS: dict[int, tuple[tuple[int, ...], ...]] = {
    1: ((1,2,3,4,17,21,23,32,39,41), (1,2,4,17,18,20,21,23,26,41), (0,1,3,4,5,20)),
    2: ((0,1,3,10,14,15,35,42), (1,3,4,17,39,44), (1,4,7,17,21,22,46)),
    3: ((0,1,3,10,14,15,35,42), (1,3,10,17,35,41,42), (1,2,4,18,21,23,41)),
    4: ((0,1,3,7,22,40), (1,3,5,20,21,34,46), (1,17,21,23,34,39,44)),
    6: ((1,2,4,18,23,26), (1,3,4,6,17,23,29), (1,2,4,7,22,23,40)),
    7: ((0,1,7,17,23,30,34,36,39), (1,5,17,21,23,46), (1,3,4,5,6,17,20,21,23,29)),
    8: ((1,4,17,20,23,32,39,44), (1,2,3,4,7,17,20,21,22,46), (1,2,4,17,18,20,21,23,26,41)),
    9: ((0,1,7,8,13,35,39), (1,2,4,10,12,32,34,35,36,41), (0,1,6,7,8,15,30,35,41)),
    11: ((0,2,3,8,9,11,14,21,26,28), (0,2,8,9,11,14,21,46), (1,8,11,17,18,41)),
    12: ((3,4,7,8,9,12,17,22,29,40), (3,4,8,9,12,17,23,32,39,44), (3,8,9,10,11,12,15,21,31,46)),
    13: ((8,9,10,11,15,30,42), (3,8,9,10,11,12,15,21), (3,8,9,10,11,14,17)),
    14: ((8,9,10,11,14,23,42), (8,9,10,11,21,23,46), (3,8,9,10,11,12,14,15,31,42)),
    16: ((3,4,7,8,9,12,17,22,29,40), (3,4,8,9,12,17,23,32,39,44), (3,8,9,11,12,28)),
    17: ((3,8,9,10,11,15,21,26,31,36), (1,8,10,11,31,42), (3,8,9,10,31,42)),
    18: ((3,4,7,8,9,12,17,22,29,40), (3,4,8,9,12,17,23,32,39,44), (3,4,8,9,11,17,39)),
    19: ((8,9,10,11,15,31,42), (3,8,9,10,11,15,21,31,42,46), (8,9,10,11,36,42)),
    21: ((1,3,4,10,16,21,32,35,42,46), (0,1,3,7,16,17,22,40), (1,8,16,17,29,41,43)),
    22: ((0,1,3,4,10,16,21,35,42,46), (0,1,3,4,10,14,15,16,35,42), (1,3,10,16,21,32,35,42,46)),
    23: ((1,2,5,16,19,28), (1,3,10,16,32,35,42), (1,3,16,17,19,20,21,23,29,41)),
    24: ((0,1,10,16,19,23,35,41,42), (0,1,7,16,17,22,23,40), (1,16,19,23,32,39,44)),
    26: ((1,10,16,17,28,30,34,35,43), (0,1,5,8,9,10,16,20,34,35), (1,5,16,17,21,46)),
    27: ((1,3,4,7,8,16,17,19,21,22), (0,1,16,17,19,23,43), (0,1,3,4,10,16,21,35,42,46)),
    28: ((0,1,2,5,7,16,17,22,30,40), (0,1,15,16,19,35,43), (0,1,10,16,19,23,35,42)),
    29: ((1,2,16,17,19,21,30,41,43,46), (1,2,4,7,16,19,22,23,41,43), (1,16,17,19,21,23,34,46)),
    31: ((8,9,10,12,15,16,19,23,25,42), (8,9,10,12,14,15,16,19,25,26), (6,8,9,16,19,25,29)),
    32: ((8,9,10,15,16,19,23,25,26,36), (8,9,10,15,16,19,25,27,36,42)),
    33: ((4,8,9,14,16,18,19,23,25,27), (8,9,12,13,16,19,25,27,29), (8,9,16,19,25,43)),
    34: ((5,8,9,16,19,23,25,43), (3,8,9,16,19,23,25,27,32,43), (8,9,16,17,20,23,25,43)),
    36: ((4,8,9,14,16,17,18,19,23,25), (1,3,5,8,16,17,20,25,41,43), (8,9,12,13,16,19,25,29)),
    37: ((8,9,10,15,16,19,23,25,36), (8,9,14,16,17,25,31,43), (3,8,9,15,16,19,25,29,31,43)),
    38: ((0,3,6,8,9,16,17,19,23,25), (0,3,8,9,16,19,21,23,25,46), (8,9,10,15,16,19,23,25)),
    39: ((8,9,10,15,16,19,23,25,27,31), (3,7,8,9,16,22,23,25,27,43), (8,9,16,17,23,25,26,31)),
    41: ((1,14,15,23,32,33,35,37,39,44), (1,2,10,17,30,32,33,35,37,42), (1,3,7,22,32,33,34,37,40,41)),
    42: ((1,2,17,30,32,33,34,37,39,44), (1,2,3,4,17,23,33,37,39,44), (1,2,10,17,30,32,33,35,37,42)),
    43: ((1,2,3,5,17,20,30,32,33,34), (1,2,3,4,6,20,29,30,32,33), (1,10,14,20,23,32,33,35,36,42)),
    44: ((1,2,3,5,17,20,30,32,33,34), (1,2,3,4,6,20,29,30,32,33), (1,2,20,30,32,33,34,37,39,44)),
    46: ((1,2,17,30,32,33,34,37,39,44), (1,2,10,17,30,32,33,35,37,42), (1,4,17,20,23,32,33,37,39,44)),
    47: ((1,2,17,30,32,33,34,37,39,44), (1,2,10,17,30,32,33,35,37,42), (1,4,17,20,23,32,33,37,39,44)),
    48: ((1,2,3,5,17,20,30,32,33,34), (1,2,20,30,32,33,34,37,39,44), (1,3,14,15,17,23,32,33,34,35)),
    49: ((1,2,17,30,32,33,34,37,39,44), (1,2,10,17,30,32,33,35,37,42), (1,4,17,20,23,32,33,37,39,44)),
}
# END GOOD_DECKS (task208)

USER_DECKS: dict[int, tuple[int, ...]] = {}
"""用户在游戏里亲自选的卡组（关卡 -> 卡组）。用户卡组一旦登记就是该关第一套，
优先级高于表里的候选；门禁 / 北极星 / 训练都必须以它为准。"""


def good_decks_for_level(level: int) -> tuple[tuple[int, ...], ...]:
    """这一关选中的好卡组（第一套 = 门禁 / 北极星用的卡组）；没有记录时退回 deck_for_level。"""
    user = USER_DECKS.get(level)
    if user:
        return (tuple(user),) + tuple(d for d in GOOD_DECKS.get(level, ()) if tuple(d) != tuple(user))
    return GOOD_DECKS.get(level) or (deck_for_level(level),)


def good_deck_for_level(level: int) -> tuple[int, ...]:
    """门禁 / 北极星用的第一套：用户卡组 > 表内首选 > deck_for_level 地形卡组。"""
    return good_decks_for_level(level)[0]
