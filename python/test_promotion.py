"""Quality promotion gate (v5): paired inference, complete coverage, zone vetoes."""
import copy
import json
from pathlib import Path

import pytest

from pvz_promotion import SEED_POOLS_V2, paired_interval

ROOT = Path(__file__).resolve().parents[1]
ADVENTURE_ZONES = ('day', 'fog', 'night', 'pool', 'roof')


def test_newcombe_published_reference_and_edge_uncertainty():
    # Official statsmodels method-10 example [[10,5],[2,20]], independently published limits.
    low, high = paired_interval(10, 5, 2, 20)
    assert low == pytest.approx(-.0643926050711, abs=1e-12)
    assert high == pytest.approx(.2211160178958, abs=1e-12)
    low, high = paired_interval(64, 0, 0, 0)
    assert low < -.03 and high > 0  # no fictitious zero-width CI for all-win A/A
    with pytest.raises(ValueError):
        paired_interval(0, 0, 0, 0)


def test_ordered_gate_pools_never_overlap_and_fit_many_rounds():
    assert len(SEED_POOLS_V2) >= 2
    ordered = sorted(SEED_POOLS_V2)
    assert ordered == list(SEED_POOLS_V2)  # consumption order is the registered order
    for (low, high), (next_low, _) in zip(ordered, ordered[1:]):
        assert low < high <= next_low  # disjoint, never touching
    assert SEED_POOLS_V2[-1][1] - SEED_POOLS_V2[-1][0] >= 200 * 1920
