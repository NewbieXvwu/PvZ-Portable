"""Shared temporal value semantics for search, imitation learning, and PPO."""

DISCOUNT_REFERENCE_TICKS = 300
VALUE_GAMMA = 0.99
VALUE_SEMANTICS = "discounted_terminal_v1"
SEARCH_LABEL_VERSION = 1


def discounted_terminal_value(won: bool, remaining_ticks: int) -> float:
    sign = 1.0 if won else -1.0
    return sign * VALUE_GAMMA ** (max(0, int(remaining_ticks)) / DISCOUNT_REFERENCE_TICKS)
