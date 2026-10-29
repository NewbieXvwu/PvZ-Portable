"""Seed-span arithmetic shared by the gate and selection panels."""
from __future__ import annotations

TRAINING_SEEDS = (1, 2)


def seed_spans(document):
    """Conservatively reserve each declared list's full span, including demo futures."""
    if isinstance(document, dict):
        for key in ('seeds', 'future_seeds'):
            values = document.get(key)
            if isinstance(values, list) and values and all(type(seed) is int for seed in values):
                yield min(values), max(values) + 1
        start, count = document.get('future_seed_start'), document.get('future_seed_count')
        if type(start) is int and type(count) is int and count > 0:
            yield start, start + count
        for value in document.values():
            yield from seed_spans(value)
    elif isinstance(document, list):
        for value in document:
            yield from seed_spans(value)


def assert_disjoint(panel, reserved):
    seeds = [seed for task in panel['tasks'] for seed in task['seeds']]
    if len(seeds) != len(set(seeds)):
        raise ValueError('selection panel repeats a seed')
    for low, high in reserved:
        if any(low <= seed < high for seed in seeds):
            raise ValueError(f'selection seeds overlap registered interval [{low}, {high})')

