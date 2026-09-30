"""Shared temporal value semantics for the PPO path.

``VALUE_SEMANTICS`` is the version tag stamped into every checkpoint's provenance
and checked on load, so its spelling is frozen even though the ``search`` and
imitation callers it was named after are gone.
"""

DISCOUNT_REFERENCE_TICKS = 300
VALUE_GAMMA = 0.99
VALUE_SEMANTICS = "discounted_terminal_v1"
SEARCH_LABEL_VERSION = 3
