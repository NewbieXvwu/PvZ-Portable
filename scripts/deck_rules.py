"""卡组生成规则 v5（2026-10-27）：只禁“零作用”，其余交给模型。

规则正文见 MEMORY/docs/deck-rules-v4.md 末尾的 v5 一节。要点：

* 全局禁 38 金盏花 / 45 吸金磁 / 47 玉米加农炮（动作空间没有发射）。
* 按区黑名单只放**物理上种不了或永远不会触发**的卡（例如无水放睡莲、屋顶放地刺）。
* 白天蘑菇解禁，但带蘑菇必带 35 咖啡豆；不带蘑菇则禁 35。白天 42 忧郁菇 → 10 + 35。
* 取消弱卡池（大蒜、仙人掌、裂荚、杨桃、大嘴花按普通卡抽）。
* 屋顶 26 仙人掌 / 22 火炬树桩解禁；火炬仍须配一张 {0,7,18,40}。
* 最少必备：地形（睡莲 / 花盆）、经济、≤100 阳光起手、屋顶投手、对空、冰车硬解、巨人重击。

卡组长度 6–10，普通选卡关 1–49（不含 5 的倍数：传送带关与特殊关）。
"""
from __future__ import annotations

import argparse
import json
import random

RULE_VERSION = 5
REGIONS = ('day', 'night', 'pool', 'fog', 'roof')

# 只与钱有关或动作空间里无法触发。
GLOBAL_BANNED = {38, 45, 47}

# 按区“零作用”黑名单（v5 表；白天 / 泳池 / 屋顶的 35 由蘑菇规则另行处理）。
REGION_BANNED = {
    'day': {11, 16, 19, 24, 25, 27, 33, 37, 43},
    'night': {16, 19, 24, 25, 27, 33, 35, 37, 43},
    'pool': {11, 25, 27, 33, 37},
    'fog': {11, 33, 35, 37},
    'roof': {11, 16, 19, 21, 24, 25, 27, 43, 46},
}

# 按关条件：31 磁力菇只在有金属僵尸的关；37 叶子保护伞只在有蹦极 / 投石车的关。
CONDITIONAL_BANS = {
    31: {1, 2, 3, 4, 6, 7, 11, 21, 23, 26, 28, 33, 34, 41, 46, 48},
    37: set(range(1, 50)) - {41, 42, 46, 47, 49},
}

# 白天 / 白天泳池 / 屋顶会睡着的蘑菇（咖啡豆规则用）。
DAY_MUSHROOMS = {8, 9, 10, 12, 13, 14, 15, 24, 31, 42}

# 升级植物 -> 底座（CheckDeckWarnings 的 upgrade 项；缺底座会被引擎整组告警）。
DEPENDENCIES = {40: {7}, 41: {1}, 42: {10}, 43: {16}, 44: {39},
                46: {21}, 22: {0, 7, 18, 40}}

# ≤100 阳光起手卡。
OPENING = {0, 4, 17, 32, 34}

REQUIRED_BY_REGION = {
    'day': {'economy': {1}, 'opening': OPENING},
    'night': {'economy': {1, 9}, 'opening': OPENING | {8}},
    'pool': {'terrain': {16}, 'economy': {1}, 'opening': OPENING | {19}},
    'fog': {'terrain': {16}, 'economy': {1, 9}, 'opening': OPENING | {8, 19, 24}},
    'roof': {'terrain': {33}, 'economy': {1}, 'opening': OPENING,
             'roof_output': {32, 34, 39, 44}},
}

# 按关硬解：对空（气球只在 33/34/39）、冰车（26/27/29）、巨人（48/49）。
REQUIRED_BY_LEVEL = (
    ('air', {33, 34, 39}, {27, 43, 26}),
    ('zomboni', {26, 27, 29}, {21, 46, 17, 20, 2}),
    ('gargantuar', {48, 49}, {2, 17, 20, 39, 44}),
)


def region_for_level(level: int) -> str:
    """1–49 的普通选卡关 -> 区；传送带关与特殊关（5 的倍数）不接受。"""
    if type(level) is not int or not 1 <= level <= 49 or level % 5 == 0:
        raise ValueError('关号必须是1至49的普通选卡关，不含5的倍数')
    return REGIONS[(level - 1) // 10]


def banned_for_level(level: int) -> set[int]:
    region = region_for_level(level)
    return GLOBAL_BANNED | REGION_BANNED[region] | {
        card for card, levels in CONDITIONAL_BANS.items() if level in levels}


def requirements(level: int) -> dict[str, set[int]]:
    region = region_for_level(level)
    required = dict(REQUIRED_BY_REGION[region])
    for name, levels, choices in REQUIRED_BY_LEVEL:
        if level in levels:
            required[name] = choices
    return required


def deck_violations(level: int, deck) -> list[str]:
    """返回违反 v5 的条目（空列表 = 合法）。给生成器与测试共用。"""
    deck = tuple(deck)
    problems = []
    region = region_for_level(level)
    cards = set(deck)
    if not 6 <= len(deck) <= 10:
        problems.append(f'长度 {len(deck)} 不在 6..10')
    if len(cards) != len(deck):
        problems.append('有重复卡')
    if not all(type(card) is int and 0 <= card < 48 for card in deck):
        problems.append('卡号超出 0..47')
        return problems
    banned = banned_for_level(level)
    for card in sorted(cards & banned):
        problems.append(f'禁卡 {card}（{region}）')
    for card, bases in DEPENDENCIES.items():
        if card in cards and not cards & bases:
            problems.append(f'{card} 缺少底座 {sorted(bases)}')
    if region in ('day', 'pool', 'roof') and ((35 in cards) != bool(cards & DAY_MUSHROOMS)):
        problems.append('咖啡豆 35 必须与白天蘑菇同进同出')
    for name, choices in requirements(level).items():
        if not cards & choices:
            problems.append(f'缺少必备组件 {name} {sorted(choices)}')
    return problems


def generate_deck(level: int, length: int, seed: int = 0) -> tuple[int, ...]:
    """从 v5 合法池里抽一套卡组；同一 (level, length, seed) 永远同一套。"""
    if type(length) is not int or not 6 <= length <= 10:
        raise ValueError('卡组长度必须是6至10的整数')
    region = region_for_level(level)
    pool = sorted(set(range(48)) - banned_for_level(level))
    required = requirements(level).values()
    rng = random.Random(seed)
    for _ in range(100_000):
        deck = tuple(rng.sample(pool, length))
        cards = set(deck)
        if not all(cards & choices for choices in required):
            continue
        if any(card in cards and not cards & choices for card, choices in DEPENDENCIES.items()):
            continue
        if region in ('day', 'pool', 'roof') and ((35 in cards) != bool(cards & DAY_MUSHROOMS)):
            continue
        return deck
    raise RuntimeError(f'未生成满足规则的卡组：level={level}, length={length}, seed={seed}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--level', type=int, required=True)
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(dict(rule_version=RULE_VERSION, level=args.level,
                          deck=list(generate_deck(args.level, args.length, args.seed)))))


if __name__ == '__main__':
    main()
