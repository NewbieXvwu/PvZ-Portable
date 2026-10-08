"""Shared temporal value semantics for the PPO path.

``VALUE_SEMANTICS`` is the version tag stamped into every checkpoint's provenance
and checked on load. Change it whenever the value target semantics change.
"""

DISCOUNT_REFERENCE_TICKS = 300
VALUE_GAMMA = 0.99
VALUE_SEMANTICS = "undiscounted_terminal_v1"
SEARCH_LABEL_VERSION = 3
