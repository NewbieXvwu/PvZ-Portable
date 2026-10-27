"""按卡组规则 v5 生成冒险模式普通选卡关的卡组。"""
from __future__ import annotations

import argparse
import json
import random

RULE_VERSION = 5
REGIONS = ('day', 'night', 'pool', 'fog', 'roof')
GLOBAL_BANNED = {38, 45, 47}
REGION_BANNED = {
    'day': {11, 16, 19, 24, 25, 27, 33, 37, 43},
    'night': {16, 19, 24, 25, 27, 33, 35, 37, 43},
    'pool': {11, 25, 27, 33, 37},
    'fog': {11, 33, 35, 37},
    'roof': {11, 16, 19, 21, 24, 25, 27, 43, 46},
}
CONDITIONAL_BANS = {
    31: {1, 2, 3, 4, 6, 7, 11, 21, 23, 26, 28, 33, 34, 41, 46, 48},
    37: set(range(1, 50)) - {41, 42, 46, 47, 49},
}
DAY_MUSHROOMS = {8, 9, 10, 12, 13, 14, 15, 24, 31, 42}
DEPENDENCIES = {40: {7}, 41: {1}, 42: {10}, 43: {16}, 44: {39},
                46: {21}, 22: {0, 7, 18, 40}}
OPENING = {0, 4, 17, 32, 34}
REQUIRED_BY_REGION = {
    'day': {'economy': {1}, 'opening': OPENING},
    'night': {'economy': {1, 9}, 'opening': OPENING | {8}},
    'pool': {'terrain': {16}, 'economy': {1}, 'opening': OPENING | {19}},
    'fog': {'terrain': {16}, 'economy': {1, 9}, 'opening': OPENING | {8, 19, 24}},
    'roof': {'terrain': {33}, 'economy': {1}, 'opening': OPENING,
             'roof_output': {32, 34, 39, 44}},
}
REQUIRED_BY_LEVEL = (
    ('air', {33, 34, 39}, {27, 43, 26}),
    ('zomboni', {26, 27, 29}, {21, 46, 17, 20, 2}),
    ('gargantuar', {48, 49}, {2, 17, 20, 39, 44}),
)


def region_for_level(level: int) -> str:
    if type(level) is not int or not 1 <= level <= 49 or level % 5 == 0:
        raise ValueError('关号必须是1至49的普通选卡关，不含5的倍数')
    return REGIONS[(level - 1) // 10]


def banned_for_level(level: int) -> set[int]:
    return GLOBAL_BANNED | REGION_BANNED[region_for_level(level)] | {
        card for card, levels in CONDITIONAL_BANS.items() if level in levels}


def requirements(level: int) -> dict[str, set[int]]:
    return {**REQUIRED_BY_REGION[region_for_level(level)],
            **{name: choices for name, levels, choices in REQUIRED_BY_LEVEL if level in levels}}


def generate_deck(level: int, length: int, seed: int = 0) -> tuple[int, ...]:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level', type=int, required=True)
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(dict(rule_version=RULE_VERSION, level=args.level,
                          deck=generate_deck(args.level, args.length, args.seed))))


if __name__ == '__main__':
    main()
